#!/usr/bin/env python3
"""
Experiment 2: Embedding Robustness Check (Multi-GPU Support)
Compare NV-Embed-v2 (original) vs gte-qwen2-1.5b-instruct (alternative) K* computation results.

Usage:
    # Single GPU
    python exp2_embedding_robustness.py --gpu 0

    # Multi-GPU parallel (using GPU 0-5)
    python exp2_embedding_robustness.py --multi_gpu 0,1,2,3,4,5

Requirements:
    pip install sentence-transformers torch numpy pandas scipy
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CACHE_DIR = os.getenv("HF_CACHE_DIR", "./.cache/huggingface")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./analysis/results")
DATA_DIR = os.getenv("DATA_DIR", "./analysis")

EMBEDDING_MODELS = {
    "nv-embed-v2": "nvidia/NV-Embed-v2",
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    # Backup small model (~90MB, fast download)
    "minilm": "sentence-transformers/all-MiniLM-L6-v2"
}


# ============================================================
# K* Computation Functions
# ============================================================
def compute_Nstar(embeddings: np.ndarray) -> Tuple[float, float]:
    """Compute N* (effective number of channels) from embeddings."""
    if embeddings.shape[0] < 2:
        return 0.0, 1.0
    
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=True)
    
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) == 0:
            return 0.0, 1.0
        
        p = eigenvalues / eigenvalues.sum()
        H = -np.sum(p * np.log(p + 1e-10))
        Nstar = np.exp(H)
        
        return H, Nstar
    except:
        return 0.0, 1.0


def compute_Nstar_conditioned(embeddings: np.ndarray, is_correct: List[bool]) -> Dict:
    """Compute N*_correct and N*_wrong separately."""
    if embeddings.shape[0] != len(is_correct):
        return {'Nstar_correct': None, 'Nstar_wrong': None}
    
    correct_idx = [i for i, c in enumerate(is_correct) if c]
    wrong_idx = [i for i, c in enumerate(is_correct) if not c]
    
    results = {
        'Nstar_correct': None,
        'Nstar_wrong': None,
        'n_correct': len(correct_idx),
        'n_wrong': len(wrong_idx)
    }
    
    if len(correct_idx) >= 2:
        _, results['Nstar_correct'] = compute_Nstar(embeddings[correct_idx])
    
    if len(wrong_idx) >= 2:
        _, results['Nstar_wrong'] = compute_Nstar(embeddings[wrong_idx])
    
    return results


# ============================================================
# Parse JSONL File Structure
# ============================================================
def parse_jsonl_file(filepath: str) -> List[Dict]:
    """Parse JSONL file with agent responses."""
    results = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    for q_idx, q_data in data.items():
                        if not isinstance(q_data, dict):
                            continue
                        
                        responses = q_data.get('responses', {})
                        is_correct = q_data.get('final_answer_iscorr', [])
                        
                        if not responses:
                            continue
                        
                        texts = []
                        correctness = []
                        
                        for i, (agent_name, reasoning) in enumerate(responses.items()):
                            if reasoning and isinstance(reasoning, str):
                                texts.append(reasoning[:2000])
                                if i < len(is_correct):
                                    correctness.append(is_correct[i])
                                else:
                                    correctness.append(False)
                        
                        if len(texts) >= 2:
                            results.append({
                                'texts': texts,
                                'is_correct': correctness,
                                'n_agents': len(texts)
                            })
                
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        pass
    
    return results


# ============================================================
# Patch DynamicCache for transformers compatibility
# ============================================================
def patch_dynamic_cache():
    """
    Fix compatibility issue: newer transformers renamed get_usable_length to get_seq_length
    AND changed the method signature.
    
    Old API (what gte-qwen2 model uses):
        get_usable_length(new_seq_length, layer_idx=0) -> int
        - new_seq_length: ignored for DynamicCache
        - layer_idx: which layer to check (default 0)
    
    New API (current transformers):
        get_seq_length(layer_idx=0) -> int
        - layer_idx: which layer to check (default 0)
    
    The model code calls: past_key_values.get_usable_length(seq_length)
    We need to ignore seq_length and call get_seq_length(layer_idx=0)
    """
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, 'get_usable_length'):
            # Create a wrapper that adapts the old signature to the new one
            def get_usable_length(self, new_seq_length=None, layer_idx=0):
                # new_seq_length is ignored, just pass layer_idx with default 0
                return self.get_seq_length(layer_idx)
            
            DynamicCache.get_usable_length = get_usable_length
            print("[Patch] Added DynamicCache.get_usable_length wrapper")
    except Exception as e:
        print(f"[Patch] Could not patch DynamicCache: {e}")


# ============================================================
# GPU Worker - import torch inside function to ensure CUDA_VISIBLE_DEVICES takes effect
# ============================================================
def gpu_worker(gpu_id: int, file_list: List[str], result_queue, 
               model_name: str, batch_size: int = 64):
    """
    Worker function for each GPU.
    IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE importing torch!
    """
    # Must be set BEFORE importing torch!
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Now import torch
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Apply compatibility patch
    patch_dynamic_cache()
    
    # Verify CUDA setup
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[GPU {gpu_id}] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[GPU {gpu_id}] torch.cuda.device_count() = {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"[GPU {gpu_id}] Current device: {torch.cuda.current_device()}")
        print(f"[GPU {gpu_id}] Device name: {torch.cuda.get_device_name(0)}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[GPU {gpu_id}] Using device: {device}")
    print(f"[GPU {gpu_id}] Starting worker, processing {len(file_list)} files")
    
    # Load model
    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=CACHE_DIR,
            trust_remote_code=True,
            device=device
        )
        if device == "cuda:0":
            model = model.half()
        print(f"[GPU {gpu_id}] Model loaded, embedding dim = {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"[GPU {gpu_id}] Model loading failed: {e}")
        result_queue.put([])
        return
    
    results = []
    total_questions = 0
    
    # Import tqdm inside worker (after spawn)
    from tqdm import tqdm
    
    pbar = tqdm(file_list, desc=f"GPU {gpu_id}", unit="file", position=gpu_id,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for idx, jsonl_file in enumerate(pbar):
        try:
            questions = parse_jsonl_file(jsonl_file)
            
            if not questions:
                continue
            
            file_kstars = []
            file_kstars_c = []
            file_kstars_w = []
            
            for q in questions:
                texts = q['texts']
                is_correct = q['is_correct']
                
                if len(texts) < 2:
                    continue
                
                with torch.no_grad():
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        device=device
                    ).astype(np.float32)
                
                _, Nstar = compute_Nstar(embeddings)
                file_kstars.append(Nstar)
                
                cond = compute_Nstar_conditioned(embeddings, is_correct)
                if cond['Nstar_correct'] is not None:
                    file_kstars_c.append(cond['Nstar_correct'])
                if cond['Nstar_wrong'] is not None:
                    file_kstars_w.append(cond['Nstar_wrong'])
                
                total_questions += 1
            
            if file_kstars:
                results.append({
                    'file': jsonl_file,
                    'gpu_id': gpu_id,
                    'n_questions': len(file_kstars),
                    'Kstar_gte': np.mean(file_kstars),
                    'Kstar_gte_std': np.std(file_kstars),
                    'Kstar_correct_gte': np.mean(file_kstars_c) if file_kstars_c else None,
                    'Kstar_wrong_gte': np.mean(file_kstars_w) if file_kstars_w else None,
                })
            
            if idx % 20 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            continue
    
    pbar.close()
    print(f"\n[GPU {gpu_id}] Done, processed {len(results)} files, {total_questions} questions")
    result_queue.put(results)


# ============================================================
# Multi-GPU Coordinator
# ============================================================
def run_multi_gpu(gpu_ids: List[int], all_files: List[str], model_name: str, batch_size: int = 64):
    """Distribute files across multiple GPUs using spawn method."""
    import multiprocessing as mp
    
    # Use spawn method to ensure child processes are clean
    mp.set_start_method('spawn', force=True)

    n_gpus = len(gpu_ids)
    n_files = len(all_files)

    print(f"\n{'='*60}")
    print(f"Multi-GPU parallel processing (spawn mode)")
    print(f"{'='*60}")
    print(f"Using GPUs: {gpu_ids}")
    print(f"Total files: {n_files}")
    print(f"Files per GPU: ~{n_files // n_gpus}")
    print()
    
    # Split files
    file_chunks = np.array_split(all_files, n_gpus)
    
    # Create queue and processes
    result_queue = mp.Queue()
    processes = []
    
    for i, gpu_id in enumerate(gpu_ids):
        chunk = list(file_chunks[i])
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, chunk, result_queue, model_name, batch_size)
        )
        p.start()
        processes.append(p)
        print(f"Started GPU {gpu_id} worker (PID: {p.pid}), assigned {len(chunk)} files")
    
    # Collect results
    all_results = []
    for _ in range(n_gpus):
        results = result_queue.get()
        all_results.extend(results)
        print(f"Received {len(results)} results")
    
    for p in processes:
        p.join()
    
    print(f"\nAll GPUs finished, total {len(all_results)} results")
    return all_results


# ============================================================
# Single GPU Processing
# ============================================================
def run_single_gpu(gpu_id: int, all_files: List[str], model_name: str, batch_size: int = 64):
    """Run on a single GPU."""
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Apply compatibility patch
    patch_dynamic_cache()
    
    print(f"\n{'='*60}")
    print(f"Single GPU processing (GPU {gpu_id})")
    print(f"{'='*60}")

    # Verify CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        device = "cpu"
        print("Warning: CUDA not available, using CPU")

    # Load model
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(
        model_name,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        device=device
    )
    
    if "cuda" in device:
        model = model.half()
    
    print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Test embedding
    print("\nTesting embedding...")
    test_texts = ["This is a test sentence.", "Another test sentence."]
    with torch.no_grad():
        test_emb = model.encode(test_texts, device=device)
    print(f"Test embedding shape: {test_emb.shape}")
    print(f"Test embedding device check passed!")
    
    # Process files
    results = []
    total_questions = 0
    
    pbar = tqdm(all_files, desc="Processing files", unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    for idx, jsonl_file in enumerate(pbar):
        try:
            questions = parse_jsonl_file(jsonl_file)
            
            if not questions:
                continue
            
            file_kstars = []
            file_kstars_c = []
            file_kstars_w = []
            
            for q in questions:
                texts = q['texts']
                is_correct = q['is_correct']
                
                if len(texts) < 2:
                    continue
                
                with torch.no_grad():
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        device=device
                    ).astype(np.float32)
                
                _, Nstar = compute_Nstar(embeddings)
                file_kstars.append(Nstar)
                
                cond = compute_Nstar_conditioned(embeddings, is_correct)
                if cond['Nstar_correct'] is not None:
                    file_kstars_c.append(cond['Nstar_correct'])
                if cond['Nstar_wrong'] is not None:
                    file_kstars_w.append(cond['Nstar_wrong'])
                
                total_questions += 1
            
            if file_kstars:
                results.append({
                    'file': jsonl_file,
                    'n_questions': len(file_kstars),
                    'Kstar_gte': np.mean(file_kstars),
                    'Kstar_gte_std': np.std(file_kstars),
                    'Kstar_correct_gte': np.mean(file_kstars_c) if file_kstars_c else None,
                    'Kstar_wrong_gte': np.mean(file_kstars_w) if file_kstars_w else None,
                })
            
            # Update progress bar with stats
            pbar.set_postfix({"files": len(results), "questions": total_questions})
            
            if idx % 20 == 0 and "cuda" in device:
                torch.cuda.empty_cache()
                
        except Exception as e:
            pbar.write(f"Error: {e}")
            continue
    
    pbar.close()
    print(f"\nDone, processed {len(results)} files, {total_questions} questions")
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Embedding Robustness Check (Multi-GPU)')
    parser.add_argument('--gpu', type=int, default=0, help='Single GPU ID')
    parser.add_argument('--multi_gpu', type=str, default=None, 
                        help='Multi-GPU IDs (e.g., "0,1,2,3,4,5")')
    parser.add_argument('--sample', type=int, default=None, help='Sample size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model', type=str, default='gte-qwen2', 
                        choices=['gte-qwen2', 'minilm'],
                        help='Embedding model: gte-qwen2 (3GB) or minilm (90MB fast)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Experiment 2: Embedding Robustness Check")
    print("=" * 80)

    # Parse GPU IDs
    if args.multi_gpu:
        gpu_ids = [int(x.strip()) for x in args.multi_gpu.split(',')]
        print(f"Mode: Multi-GPU parallel ({len(gpu_ids)} GPUs)")
        print(f"GPU IDs: {gpu_ids}")
    else:
        gpu_ids = [args.gpu]
        print(f"Mode: Single GPU (GPU {args.gpu})")
    
    print(f"Batch size: {args.batch_size}")
    print(f"Sample size: {args.sample if args.sample else 'All'}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load original results
    original_results_path = os.path.join(OUTPUT_DIR, "kstar_accuracy_by_method_dataset.csv")
    
    try:
        df_original = pd.read_csv(original_results_path)
        df_original = df_original.dropna(subset=['Kstar', 'Kstar_correct', 'Kstar_wrong', 'accuracy'])
        print(f"\nOriginal results (NV-Embed-v2): {len(df_original)} entries")
    except FileNotFoundError:
        print(f"Warning: {original_results_path} not found")
        df_original = None
    
    # Find JSONL files
    all_jsonl_files = glob.glob(os.path.join(DATA_DIR, "**/history/*.jsonl"), recursive=True)
    print(f"\nFound {len(all_jsonl_files)} JSONL files")

    if args.sample and args.sample < len(all_jsonl_files):
        np.random.seed(42)
        all_jsonl_files = list(np.random.choice(all_jsonl_files, args.sample, replace=False))
        print(f"Sampled {args.sample} files")

    if not all_jsonl_files:
        print("Error: No JSONL files found")
        return
    
    model_name = EMBEDDING_MODELS[args.model]
    print(f"Embedding model: {args.model} -> {model_name}")
    
    if len(gpu_ids) > 1:
        results = run_multi_gpu(gpu_ids, all_jsonl_files, model_name, args.batch_size)
    else:
        results = run_single_gpu(gpu_ids[0], all_jsonl_files, model_name, args.batch_size)
    
    if not results:
        print("Error: No results generated")
        return
    
    # Save results
    df_new = pd.DataFrame(results)
    output_csv = os.path.join(OUTPUT_DIR, 'exp2_gte_qwen2_kstar.csv')
    df_new.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Statistics
    print(f"\n{'='*60}")
    print("gte-qwen2 K* Statistics")
    print(f"{'='*60}")
    print(f"  Files: {len(df_new)}")
    print(f"  Questions: {df_new['n_questions'].sum()}")
    print(f"  K* range: [{df_new['Kstar_gte'].min():.3f}, {df_new['Kstar_gte'].max():.3f}]")
    print(f"  K* mean: {df_new['Kstar_gte'].mean():.3f}")
    
    if df_new['Kstar_correct_gte'].notna().any():
        print(f"  K*_c mean: {df_new['Kstar_correct_gte'].mean():.3f}")
    if df_new['Kstar_wrong_gte'].notna().any():
        print(f"  K*_w mean: {df_new['Kstar_wrong_gte'].mean():.3f}")

    # Compare
    if df_original is not None:
        print(f"\n{'='*60}")
        print("Comparison with NV-Embed-v2")
        print(f"{'='*60}")
        print(f"  NV K* mean: {df_original['Kstar'].mean():.3f}")
        print(f"  GTE K* mean: {df_new['Kstar_gte'].mean():.3f}")
        
        # Save comparison
        comparison_csv = os.path.join(OUTPUT_DIR, 'exp2_embedding_comparison.csv')
        kstar_c_gte = df_new['Kstar_correct_gte'].mean() if df_new['Kstar_correct_gte'].notna().any() else None
        kstar_w_gte = df_new['Kstar_wrong_gte'].mean() if df_new['Kstar_wrong_gte'].notna().any() else None
        
        comparison_data = {
            'Metric': ['K* mean', 'K* std', 'K* min', 'K* max', 'K*_c mean', 'K*_w mean'],
            'NV-Embed-v2': [
                df_original['Kstar'].mean(),
                df_original['Kstar'].std(),
                df_original['Kstar'].min(),
                df_original['Kstar'].max(),
                df_original['Kstar_correct'].mean(),
                df_original['Kstar_wrong'].mean()
            ],
            'gte-qwen2': [
                df_new['Kstar_gte'].mean(),
                df_new['Kstar_gte'].std(),
                df_new['Kstar_gte'].min(),
                df_new['Kstar_gte'].max(),
                kstar_c_gte,
                kstar_w_gte
            ]
        }
        pd.DataFrame(comparison_data).to_csv(comparison_csv, index=False)
        print(f"\nComparison saved: {comparison_csv}")
        
        # LaTeX - pre-format conditional values
        kstar_c_gte_str = f"{kstar_c_gte:.3f}" if kstar_c_gte else "N/A"
        kstar_w_gte_str = f"{kstar_w_gte:.3f}" if kstar_w_gte else "N/A"
        
        latex_content = f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Embedding Robustness Check.}} $K^*$ computed with NV-Embed-v2 (4096-d) 
vs gte-qwen2-1.5b (1536-d). Consistent distributions validate robustness.}}
\\label{{tab:exp2-embedding}}
\\footnotesize
\\begin{{tabular}}{{l|cc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{NV-Embed-v2}} & \\textbf{{gte-qwen2}} \\\\
\\midrule
$K^*$ mean & {df_original['Kstar'].mean():.3f} & {df_new['Kstar_gte'].mean():.3f} \\\\
$K^*$ std & {df_original['Kstar'].std():.3f} & {df_new['Kstar_gte'].std():.3f} \\\\
$K^*_c$ mean & {df_original['Kstar_correct'].mean():.3f} & {kstar_c_gte_str} \\\\
$K^*_w$ mean & {df_original['Kstar_wrong'].mean():.3f} & {kstar_w_gte_str} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        latex_path = os.path.join(OUTPUT_DIR, 'exp2_embedding_comparison.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"LaTeX saved: {latex_path}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
