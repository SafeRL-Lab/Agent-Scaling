#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis_improved.py
Extended N* analysis with three new metrics:
  1) Answer-Conditioned N*: N* for correct/wrong answer groups separately
  2) Weighted N*: N* with answer-correctness weighting
  3) Delta-N*: Marginal N* contribution per agent

This script reuses cached embeddings from analysis.py (round_agent_avg mode).

Outputs (CSV):
  - improved_file_summary.csv   : one row per jsonl file (new metrics)
  - improved_question_table.csv : one row per file x question (detailed N* variants)
  - improved_agent_delta.csv    : one row per file x agent (delta-N* per agent)
"""

import os
import sys
import re
import json
import time
import math
import hashlib
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

# -------------------------
# Constants / Regex
# -------------------------
CHOICES = ("A", "B", "C", "D")

_ANS_PATTS = [
    re.compile(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", re.IGNORECASE),
    re.compile(r"\(([ABCD])\)", re.IGNORECASE),
    re.compile(r"final\s*answer\s*[:=]\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE),
]


# -------------------------
# Data structures
# -------------------------
@dataclass
class TextItem:
    file_id: str
    q_idx: int
    round_idx: int
    agent_id: str
    text: str


# -------------------------
# Utilities
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def digit_keys(d: Dict[str, Any]) -> List[str]:
    ks = [k for k in d.keys() if isinstance(k, str) and k.isdigit()]
    ks.sort(key=lambda x: int(x))
    return ks


def get_text(v: Any) -> Optional[str]:
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(v, dict):
        for k in ["output_text", "text", "content", "output", "response"]:
            t = v.get(k)
            if isinstance(t, str) and t.strip():
                return t.strip()
    return None


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_choice_token(s: Any) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().upper()
    if t in CHOICES:
        return t
    t2 = t.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    t2 = t2.replace("\\", " ").replace("**", " ").replace("_", " ")
    for pat in _ANS_PATTS:
        m = pat.search(t2)
        if m:
            c = m.group(1).upper()
            if c in CHOICES:
                return c
    return None


def safe_nanmean(arr):
    arr = np.array(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def safe_nanstd(arr):
    arr = np.array(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.std(valid)) if len(valid) > 1 else float("nan")


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


# -------------------------
# N* computation (optimized for n << d)
# -------------------------
def compute_Nstar(M: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Effective channel number N* = 2^H
    Returns (H, N*)
    
    Optimized: when n_samples << n_features, compute eigenvalues of 
    the smaller (n x n) Gram matrix instead of (d x d) covariance matrix.
    """
    if M.size == 0 or M.shape[0] == 0:
        return 0.0, 1.0
    
    n, d = M.shape
    Mf = M.astype(np.float64, copy=False)
    
    # Use smaller matrix: M @ M.T is (n, n) instead of M.T @ M which is (d, d)
    # The non-zero eigenvalues are the same
    if n <= d:
        G = Mf @ Mf.T  # (n, n) Gram matrix
    else:
        G = Mf.T @ Mf  # (d, d) covariance matrix
    
    mu = np.linalg.eigvalsh(G)
    mu = np.clip(mu, 0.0, None)
    s = mu.sum()
    if s <= eps:
        return 0.0, 1.0
    lam = mu / s
    lam_nz = lam[lam > eps]
    H = float(-(lam_nz * np.log2(lam_nz)).sum())
    Nstar = float(2.0 ** H)
    return H, Nstar


# -------------------------
# Answer-Conditioned N*
# -------------------------
def compute_Nstar_conditioned(
    embeddings: np.ndarray,
    agent_answers: List[Optional[str]],
    ground_truth: Optional[str]
) -> Dict[str, float]:
    """Compute N* separately for correct and wrong answer groups."""
    if ground_truth is None or embeddings.shape[0] == 0:
        return {
            "Nstar_correct": float("nan"),
            "Nstar_wrong": float("nan"),
            "Nstar_effective": float("nan"),
            "n_correct": 0,
            "n_wrong": 0,
        }
    
    correct_idx = []
    wrong_idx = []
    
    for i, ans in enumerate(agent_answers):
        if ans is None:
            continue
        if ans == ground_truth:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)
    
    n_correct = len(correct_idx)
    n_wrong = len(wrong_idx)
    
    if n_correct >= 2:
        _, Nstar_correct = compute_Nstar(embeddings[correct_idx])
    elif n_correct == 1:
        Nstar_correct = 1.0
    else:
        Nstar_correct = float("nan")
    
    if n_wrong >= 2:
        _, Nstar_wrong = compute_Nstar(embeddings[wrong_idx])
    elif n_wrong == 1:
        Nstar_wrong = 1.0
    else:
        Nstar_wrong = float("nan")
    
    alpha = 0.5
    if np.isfinite(Nstar_correct) and np.isfinite(Nstar_wrong):
        Nstar_effective = Nstar_correct - alpha * Nstar_wrong
    elif np.isfinite(Nstar_correct):
        Nstar_effective = Nstar_correct
    else:
        Nstar_effective = float("nan")
    
    return {
        "Nstar_correct": Nstar_correct,
        "Nstar_wrong": Nstar_wrong,
        "Nstar_effective": Nstar_effective,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
    }


# -------------------------
# Weighted N*
# -------------------------
def compute_Nstar_weighted(
    embeddings: np.ndarray,
    agent_answers: List[Optional[str]],
    ground_truth: Optional[str],
    correct_weight: float = 1.0,
    wrong_weight: float = 0.3
) -> Dict[str, float]:
    """Compute N* with weighted embeddings based on answer correctness."""
    if embeddings.shape[0] == 0:
        return {"Nstar_weighted": float("nan"), "H_weighted": float("nan")}
    
    n = embeddings.shape[0]
    weights = np.ones(n, dtype=np.float64)
    
    if ground_truth is not None:
        for i, ans in enumerate(agent_answers):
            if ans is None:
                weights[i] = 0.5
            elif ans == ground_truth:
                weights[i] = correct_weight
            else:
                weights[i] = wrong_weight
    
    weights = weights / weights.sum() * n
    weighted_emb = embeddings * weights[:, np.newaxis]
    H, Nstar = compute_Nstar(weighted_emb)
    
    return {"Nstar_weighted": Nstar, "H_weighted": H}


# -------------------------
# Delta-N* (Marginal Information Gain)
# -------------------------
def compute_delta_Nstar(
    embeddings: np.ndarray,
    agent_ids: List[str]
) -> Dict[str, Any]:
    """Compute marginal N* contribution for each agent."""
    if embeddings.shape[0] == 0:
        return {
            "delta_per_agent": {},
            "Nstar_full": float("nan"),
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "delta_max": float("nan"),
            "delta_min": float("nan"),
        }
    
    n = embeddings.shape[0]
    _, Nstar_full = compute_Nstar(embeddings)
    
    delta_per_agent = {}
    deltas = []
    
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        emb_without_i = embeddings[mask]
        
        if emb_without_i.shape[0] >= 1:
            _, Nstar_without_i = compute_Nstar(emb_without_i)
        else:
            Nstar_without_i = 1.0
        
        delta = Nstar_full - Nstar_without_i
        delta_per_agent[agent_ids[i]] = delta
        deltas.append(delta)
    
    return {
        "delta_per_agent": delta_per_agent,
        "Nstar_full": Nstar_full,
        "delta_mean": safe_nanmean(deltas),
        "delta_std": safe_nanstd(deltas),
        "delta_max": max(deltas) if deltas else float("nan"),
        "delta_min": min(deltas) if deltas else float("nan"),
    }


# -------------------------
# Cache loading
# -------------------------
def load_existing_cache(jsonl_path: str, cache_dir_name: str = "cache") -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """Load cached embeddings from analysis.py runs."""
    base_dir = os.path.dirname(os.path.abspath(jsonl_path))
    cache_dir = os.path.join(base_dir, cache_dir_name)
    stem = os.path.splitext(os.path.basename(jsonl_path))[0]
    npy_path = os.path.join(cache_dir, f"{stem}.npy")
    meta_path = os.path.join(cache_dir, f"{stem}.json")
    
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        emb = np.load(npy_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return emb, {k: int(v) for k, v in meta.items()}
    return None, {}


def get_signature_for_round_agent_avg() -> str:
    """Get the signature used by analysis.py for round_agent_avg mode."""
    CODE_VERSION = "analysis_py_v1.0"
    base = {
        "code_version": CODE_VERSION,
        "model_path": "nvidia/NV-Embed-v2",
        "device": "cuda",
        "max_seq_length": 32768,
        "add_eos": True,
        "normalize_embeddings": True,
        "trust_remote_code": True,
        "analysis_mode": "round_agent_avg",
        "require_intersection": True,
    }
    raw = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return sha1(raw)


# -------------------------
# File discovery
# -------------------------
def discover_jsonl_paths(path_in: str) -> List[str]:
    paths: List[str] = []
    if os.path.isfile(path_in) and path_in.lower().endswith(".jsonl"):
        paths.append(os.path.abspath(path_in))
    elif os.path.isdir(path_in):
        for root, _, files in os.walk(path_in):
            for fn in files:
                if fn.lower().endswith(".jsonl") and not fn.startswith("."):
                    paths.append(os.path.abspath(os.path.join(root, fn)))
    return sorted(paths)


# -------------------------
# CSV writing
# -------------------------
def write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            out = []
            for c in columns:
                v = r.get(c, "")
                if isinstance(v, float):
                    if math.isnan(v):
                        out.append("")
                    else:
                        out.append(f"{v:.6g}")
                else:
                    out.append(str(v))
            f.write(",".join(out) + "\n")


# -------------------------
# Extract items from jsonl (same as analysis.py)
# -------------------------
def extract_items_from_question(one_q: Dict[str, Any], file_id: str, q_idx: int) -> List[TextItem]:
    """Extract all text items from one question."""
    rks = digit_keys(one_q)
    if not rks:
        return []
    
    items = []
    for rk in rks:
        round_obj = one_q.get(rk, {})
        r_idx = int(rk) + 1
        responses = round_obj.get("responses", {})
        if isinstance(responses, dict):
            for agent_id, payload in responses.items():
                t = get_text(payload)
                if t:
                    items.append(TextItem(
                        file_id=file_id,
                        q_idx=q_idx,
                        round_idx=r_idx,
                        agent_id=str(agent_id),
                        text=t
                    ))
    return items


# -------------------------
# Main processing for one file
# -------------------------
def process_one_jsonl(
    jsonl_path: str,
    cache_emb: np.ndarray,
    cache_meta: Dict[str, int],
    sig: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process one jsonl file using cached embeddings.
    Uses round_agent_avg cached embeddings, averages per agent per question.
    """
    file_id = os.path.basename(jsonl_path)
    rows = load_jsonl(jsonl_path)
    
    question_results = []
    agent_delta_results = []
    
    n_questions = len(rows)
    
    for q_idx, one_q in enumerate(rows):
        if (q_idx + 1) % 50 == 0:
            log(f"      Q {q_idx + 1}/{n_questions}")
        
        rks = digit_keys(one_q)
        if not rks:
            continue
        
        # Get ground truth
        gt = ""
        for rk in reversed(rks):
            round_obj = one_q.get(rk, {})
            ans = round_obj.get("answer", None)
            if isinstance(ans, str) and ans.strip():
                gt = ans.strip()
                break
        gt_norm = normalize_choice_token(gt)
        
        # Get final answers
        final_answers_raw = []
        for rk in reversed(rks):
            round_obj = one_q.get(rk, {})
            fa = round_obj.get("final_answers", None)
            if isinstance(fa, list) and fa:
                final_answers_raw = fa
                break
        
        # Extract all text items for this question
        items = extract_items_from_question(one_q, file_id, q_idx)
        if not items:
            continue
        
        # Group items by agent and collect embeddings
        agent_items: Dict[str, List[Tuple[TextItem, int]]] = defaultdict(list)
        
        for item in items:
            key = f"{sig}:{sha1(item.text)}"
            if key in cache_meta:
                emb_idx = cache_meta[key]
                agent_items[item.agent_id].append((item, emb_idx))
        
        if not agent_items:
            continue
        
        # For each agent, average their embeddings across rounds
        agent_ids = sorted(agent_items.keys())
        agent_embeddings = []
        agent_answers = []
        
        for i, aid in enumerate(agent_ids):
            items_with_idx = agent_items[aid]
            if not items_with_idx:
                continue
            
            # Average embeddings for this agent
            emb_list = [cache_emb[idx] for _, idx in items_with_idx]
            avg_emb = np.mean(np.vstack(emb_list), axis=0)
            agent_embeddings.append(avg_emb)
            
            # Get answer for this agent
            if i < len(final_answers_raw):
                agent_answers.append(normalize_choice_token(final_answers_raw[i]))
            else:
                agent_answers.append(None)
        
        if len(agent_embeddings) < 2:
            continue
        
        embeddings = np.vstack(agent_embeddings).astype(np.float32)
        
        # Compute standard N*
        H_std, Nstar_std = compute_Nstar(embeddings)
        
        # Compute Answer-Conditioned N*
        cond_results = compute_Nstar_conditioned(embeddings, agent_answers, gt_norm)
        
        # Compute Weighted N*
        weighted_results = compute_Nstar_weighted(embeddings, agent_answers, gt_norm)
        
        # Compute Delta-N*
        delta_results = compute_delta_Nstar(embeddings, agent_ids)
        
        # Store results
        q_result = {
            "file": file_id,
            "q_idx": q_idx,
            "n_agents": len(agent_ids),
            "ground_truth": gt_norm if gt_norm else "",
            "H": H_std,
            "Nstar": Nstar_std,
            "Nstar_correct": cond_results["Nstar_correct"],
            "Nstar_wrong": cond_results["Nstar_wrong"],
            "Nstar_effective": cond_results["Nstar_effective"],
            "n_correct": cond_results["n_correct"],
            "n_wrong": cond_results["n_wrong"],
            "Nstar_weighted": weighted_results["Nstar_weighted"],
            "H_weighted": weighted_results["H_weighted"],
            "delta_mean": delta_results["delta_mean"],
            "delta_std": delta_results["delta_std"],
            "delta_max": delta_results["delta_max"],
            "delta_min": delta_results["delta_min"],
        }
        question_results.append(q_result)
        
        for aid, delta in delta_results["delta_per_agent"].items():
            agent_delta_results.append({
                "file": file_id,
                "q_idx": q_idx,
                "agent_id": aid,
                "delta_Nstar": delta,
                "Nstar_full": delta_results["Nstar_full"],
            })
    
    return question_results, agent_delta_results


def process_file(
    jsonl_path: str,
    cache_dir_name: str = "cache",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process one jsonl file using cached embeddings."""
    t0 = time.time()
    file_id = os.path.basename(jsonl_path)
    
    out: Dict[str, Any] = {
        "file": file_id,
        "path": jsonl_path,
        "error": "",
    }
    question_rows = []
    agent_delta_rows = []
    
    try:
        cache_emb, cache_meta = load_existing_cache(jsonl_path, cache_dir_name)
        
        if cache_emb is None:
            out["error"] = "No cached embeddings found"
            out["secs"] = float(round(time.time() - t0, 3))
            return out, question_rows, agent_delta_rows
        
        sig = get_signature_for_round_agent_avg()
        
        question_rows, agent_delta_rows = process_one_jsonl(
            jsonl_path=jsonl_path,
            cache_emb=cache_emb,
            cache_meta=cache_meta,
            sig=sig,
        )
        
        rows = load_jsonl(jsonl_path)
        out["questions"] = len(rows)
        
        if question_rows:
            out["n_questions_processed"] = len(question_rows)
            out["Nstar_mean"] = safe_nanmean([r["Nstar"] for r in question_rows])
            out["Nstar_std"] = safe_nanstd([r["Nstar"] for r in question_rows])
            out["Nstar_correct_mean"] = safe_nanmean([r["Nstar_correct"] for r in question_rows])
            out["Nstar_wrong_mean"] = safe_nanmean([r["Nstar_wrong"] for r in question_rows])
            out["Nstar_effective_mean"] = safe_nanmean([r["Nstar_effective"] for r in question_rows])
            out["Nstar_weighted_mean"] = safe_nanmean([r["Nstar_weighted"] for r in question_rows])
            out["delta_mean_avg"] = safe_nanmean([r["delta_mean"] for r in question_rows])
            out["delta_max_avg"] = safe_nanmean([r["delta_max"] for r in question_rows])
            
            n_correct_total = sum(r["n_correct"] for r in question_rows)
            n_wrong_total = sum(r["n_wrong"] for r in question_rows)
            if n_correct_total + n_wrong_total > 0:
                out["correct_ratio"] = n_correct_total / (n_correct_total + n_wrong_total)
            else:
                out["correct_ratio"] = float("nan")
        
    except Exception as e:
        import traceback
        out["error"] = f"{type(e).__name__}: {e}"
        log(f"    ERROR: {out['error']}")
    
    out["secs"] = float(round(time.time() - t0, 3))
    return out, question_rows, agent_delta_rows


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Improved N* analysis with conditioned, weighted, and delta metrics."
    )
    parser.add_argument("jsonl_path", type=str, help="Path to .jsonl file or directory.")
    parser.add_argument("--cache-dir-name", type=str, default="cache", help="Cache subdirectory name.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    
    args = parser.parse_args()
    
    input_path = args.jsonl_path
    jsonl_files = discover_jsonl_paths(input_path)
    
    if not jsonl_files:
        log(f"No .jsonl files found at: {input_path}")
        return
    
    if args.out_dir is None:
        if os.path.isdir(input_path):
            out_dir = os.path.abspath(input_path)
        else:
            out_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    else:
        out_dir = os.path.abspath(args.out_dir)
    
    log(f"Found {len(jsonl_files)} jsonl files.")
    log(f"Output: {out_dir}")
    log("")
    
    file_summary_rows = []
    question_rows_all = []
    agent_delta_rows_all = []
    
    for idx, p in enumerate(jsonl_files, 1):
        log(f"[{idx}/{len(jsonl_files)}] {os.path.basename(p)}")
        
        summary_row, question_rows, agent_delta_rows = process_file(
            jsonl_path=p,
            cache_dir_name=args.cache_dir_name,
        )
        
        if summary_row.get("error"):
            log(f"  ERROR: {summary_row['error'][:80]}")
        else:
            log(f"  Done: q={summary_row.get('n_questions_processed', 0)}, "
                f"N*_eff={summary_row.get('Nstar_effective_mean', float('nan')):.3f}, "
                f"t={summary_row.get('secs', 0):.1f}s")
        
        file_summary_rows.append(summary_row)
        question_rows_all.extend(question_rows)
        agent_delta_rows_all.extend(agent_delta_rows)
    
    # Write outputs
    file_summary_path = os.path.join(out_dir, "improved_file_summary.csv")
    question_table_path = os.path.join(out_dir, "improved_question_table.csv")
    agent_delta_path = os.path.join(out_dir, "improved_agent_delta.csv")
    
    file_cols = [
        "file", "path", "error", "secs",
        "questions", "n_questions_processed",
        "Nstar_mean", "Nstar_std",
        "Nstar_correct_mean", "Nstar_wrong_mean", "Nstar_effective_mean",
        "Nstar_weighted_mean",
        "delta_mean_avg", "delta_max_avg",
        "correct_ratio",
    ]
    write_csv(file_summary_path, file_summary_rows, file_cols)
    
    q_cols = [
        "file", "q_idx", "n_agents", "ground_truth",
        "H", "Nstar",
        "Nstar_correct", "Nstar_wrong", "Nstar_effective",
        "n_correct", "n_wrong",
        "Nstar_weighted", "H_weighted",
        "delta_mean", "delta_std", "delta_max", "delta_min",
    ]
    write_csv(question_table_path, question_rows_all, q_cols)
    
    agent_cols = ["file", "q_idx", "agent_id", "delta_Nstar", "Nstar_full"]
    write_csv(agent_delta_path, agent_delta_rows_all, agent_cols)
    
    log("")
    log(f"Saved: {file_summary_path}")
    log(f"Saved: {question_table_path}")
    log(f"Saved: {agent_delta_path}")


if __name__ == "__main__":
    main()
