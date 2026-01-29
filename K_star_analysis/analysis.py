#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis.py
Unified MAS collaboration analysis with NV-Embed-v2 (SentenceTransformer) and N* metrics.

Modes:
  1) round_cum_text     : cumulative-text N* by round (diagnostic; sample size grows)
  2) per_question_agent : per-question agent-aggregated N*(q) + correlations
  3) round_agent_avg    : per-round agent-average N* (agent-centric, round-comparable)

Outputs (CSV):
  - file_summary.csv    : one row per jsonl file (key metrics)
  - round_table.csv     : one row per file x round (N*, H, delta, counts)
  - question_table.csv  : one row per file x question (N*(q) + PartB; mainly for per_question_agent)

This script assumes "Structure B" jsonl:
Each line is a JSON object for one question:
{
  "0": {"responses": {agent_id: "..."/{...}}, "final_answers": [...], "answer": "..."},
  "1": {...},
  ...
}
"""

import os
# Set offline mode to force using locally cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import json
import time
import math
import hashlib
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x


# -------------------------
# Constants / Regex
# -------------------------
CHOICES = ("A", "B", "C", "D")
_FINAL_RE = re.compile(r"(?i)(?:final\s*answer\s*[:：=]\s*)?\(?\s*([ABCD])\s*\)?")

# Robust letter token extraction (avoid matching inside words)
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
    round_idx: int      # 1-based; may use 0 for aggregated
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


def normalize_choice(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    if s in CHOICES:
        return s
    m = _FINAL_RE.search(s)
    if m:
        c = m.group(1).upper()
        if c in CHOICES:
            return c
    return None


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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    A = a[mask]
    B = b[mask]
    if float(A.std()) == 0.0 or float(B.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(A, B)[0, 1])


# -------------------------
# Extraction (Structure B)
# -------------------------
def extract_one_question_B(one_q: Dict[str, Any], file_id: str, q_idx: int) -> Tuple[List[TextItem], List[str], str, int]:
    rks = digit_keys(one_q)
    if not rks:
        # fallback: treat as single-round (rare)
        gt = one_q.get("answer", "")
        return [], [], str(gt) if gt is not None else "", 1

    R_max = int(rks[-1]) + 1

    items: List[TextItem] = []
    final_answers_last: List[str] = []
    gt_last: str = ""

    # collect responses
    for rk in rks:
        round_obj = one_q.get(rk, {})
        r_idx = int(rk) + 1
        responses = round_obj.get("responses", {})
        if isinstance(responses, dict):
            for agent_id, payload in responses.items():
                t = get_text(payload)
                if t:
                    items.append(TextItem(file_id=file_id, q_idx=q_idx, round_idx=r_idx,
                                          agent_id=str(agent_id), text=t))

    # find last available final_answers and answer
    for rk in reversed(rks):
        round_obj = one_q.get(rk, {})
        if not final_answers_last:
            fa = round_obj.get("final_answers", None)
            if isinstance(fa, list):
                fa_clean = [str(x).strip() for x in fa if str(x).strip()]
                if fa_clean:
                    final_answers_last = fa_clean
        if not gt_last:
            ans = round_obj.get("answer", None)
            if isinstance(ans, str) and ans.strip():
                gt_last = ans.strip()
        if final_answers_last and gt_last:
            break

    return items, final_answers_last, gt_last, R_max


def extract_dataset_B(rows: List[Dict[str, Any]], file_id: str) -> Tuple[List[TextItem], List[List[str]], List[str], int]:
    all_items: List[TextItem] = []
    all_final_answers: List[List[str]] = []
    all_gt: List[str] = []
    R_global = 1

    for q_idx, one_q in enumerate(rows):
        items, fa, gt, R_max = extract_one_question_B(one_q, file_id=file_id, q_idx=q_idx)
        all_items.extend(items)
        all_final_answers.append(fa)
        all_gt.append(gt)
        R_global = max(R_global, R_max)

    return all_items, all_final_answers, all_gt, R_global


# -------------------------
# Part B (disagreement/consensus/accuracy)
# -------------------------
def gini_proxy(ans_list: List[str]) -> float:
    xs = [normalize_choice_token(a) for a in (ans_list or [])]
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    cnt = {c: 0 for c in CHOICES}
    for x in xs:
        cnt[x] += 1
    n = len(xs)
    p2 = sum((cnt[c] / n) ** 2 for c in CHOICES)
    return 1.0 - p2


def consensus(ans_list: List[str]) -> float:
    xs = [normalize_choice_token(a) for a in (ans_list or [])]
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    cnt = {c: 0 for c in CHOICES}
    for x in xs:
        cnt[x] += 1
    n = len(xs)
    return sum((cnt[c] / n) ** 2 for c in CHOICES)


def majority_vote(ans_list: List[str]) -> Optional[str]:
    xs = [normalize_choice_token(a) for a in (ans_list or [])]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    cnt = {c: 0 for c in CHOICES}
    for x in xs:
        cnt[x] += 1
    best = max(cnt.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else None


# -------------------------
# N* computation
# -------------------------
def compute_Nstar(M: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Effective channel number N* = 2^H, where H is Shannon entropy of normalized eigenvalues of C = M^T M.
    M: (n, d)
    """
    if M.size == 0:
        return 0.0, 1.0
    Mf = M.astype(np.float64, copy=False)
    C = Mf.T @ Mf  # (d,d)
    mu = np.linalg.eigvalsh(C)
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
# Embedding model + caching (signature-fixed)
# -------------------------
class Embedder:
    """
    SentenceTransformer wrapper with explicit signature-aware caching.
    """
    CODE_VERSION = "analysis_py_v1.0"  # bump if you change encoding semantics

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_seq_length: int = 32768,
        add_eos: bool = True,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
        show_progress_bar: bool = True,
        cache_folder: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path
        self.max_seq_length = int(max_seq_length)
        self.add_eos_flag = bool(add_eos)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.trust_remote_code = bool(trust_remote_code)
        self.show_progress_bar = bool(show_progress_bar)
        self.cache_folder = cache_folder

        print(f"Loading model: {model_path} on {self.device}")
        if cache_folder:
            print(f"Cache folder: {cache_folder}")
        
        self.model = SentenceTransformer(
            model_path, 
            trust_remote_code=trust_remote_code, 
            device=self.device,
            cache_folder=cache_folder,
        )
        # This is the actual knob SentenceTransformer uses
        self.model.max_seq_length = self.max_seq_length
        self.model.tokenizer.padding_side = "right"

        # EOS token handling
        self.eos_token = getattr(self.model.tokenizer, "eos_token", None)

        print("Model loaded.")

    def encoding_signature(self, extra: Optional[Dict[str, Any]] = None) -> str:
        """
        A stable signature that must change when embeddings may change.
        """
        base = {
            "code_version": self.CODE_VERSION,
            "model_path": self.model_path,
            "device": self.device,  # device shouldn't change results but keep it anyway for debugging
            "max_seq_length": self.max_seq_length,
            "add_eos": self.add_eos_flag,
            "normalize_embeddings": self.normalize_embeddings,
            "trust_remote_code": self.trust_remote_code,
        }
        if extra:
            base.update(extra)
        raw = json.dumps(base, sort_keys=True, ensure_ascii=False)
        return sha1(raw)  # short fixed signature

    def _maybe_add_eos(self, texts: List[str]) -> List[str]:
        if not self.add_eos_flag:
            return texts
        if not self.eos_token:
            return texts
        return [t + self.eos_token for t in texts]

    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        texts2 = self._maybe_add_eos(texts)
        emb = self.model.encode(
            texts2,
            batch_size=int(batch_size),
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    @staticmethod
    def derive_cache_paths(jsonl_path: str, cache_dir_name: str = "cache") -> Tuple[str, str]:
        base_dir = os.path.dirname(os.path.abspath(jsonl_path))
        cache_dir = os.path.join(base_dir, cache_dir_name)
        os.makedirs(cache_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(jsonl_path))[0]
        npy_path = os.path.join(cache_dir, f"{stem}.npy")
        meta_path = os.path.join(cache_dir, f"{stem}.json")
        return npy_path, meta_path

    @staticmethod
    def load_cache(npy_path: str, meta_path: str) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
        if os.path.exists(npy_path) and os.path.exists(meta_path):
            emb = np.load(npy_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return emb, {k: int(v) for k, v in meta.items()}
        return None, {}

    @staticmethod
    def save_cache(npy_path: str, meta_path: str, emb: np.ndarray, meta: Dict[str, int]) -> None:
        np.save(npy_path, emb)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def get_embeddings_with_cache(
        self,
        texts: List[str],
        npy_path: str,
        meta_path: str,
        batch_size: int,
        signature_extra: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Cache is append-only. Keys include encoding signature, preventing semantic cache collisions.
        """
        cache_emb, cache_meta = self.load_cache(npy_path, meta_path)

        sig = self.encoding_signature(signature_extra)
        keys = [f"{sig}:{sha1(t)}" for t in texts]

        need_idx = [i for i, k in enumerate(keys) if k not in cache_meta]
        need_texts = [texts[i] for i in need_idx]

        new_vecs = self.encode(need_texts, batch_size=batch_size) if need_texts else None

        if cache_emb is None:
            if new_vecs is None:
                raise RuntimeError("No cache and nothing to embed.")
            cache_emb = new_vecs
            for j, i in enumerate(need_idx):
                cache_meta[keys[i]] = j
        else:
            start = int(cache_emb.shape[0])
            if new_vecs is not None and int(new_vecs.shape[0]) > 0:
                cache_emb = np.vstack([cache_emb, new_vecs]).astype(np.float32)
                for j, i in enumerate(need_idx):
                    cache_meta[keys[i]] = start + j

        self.save_cache(npy_path, meta_path, cache_emb, cache_meta)

        # materialize in original order
        out = np.vstack([cache_emb[cache_meta[k]] for k in keys]).astype(np.float32)
        return out


# -------------------------
# Mode 1: round_cum_text
# -------------------------
def compute_round_cum_text(items: List[TextItem], emb: np.ndarray, R_max: int) -> Dict[int, Dict[str, float]]:
    idx_by_round: Dict[int, List[int]] = {r: [] for r in range(1, R_max + 1)}
    for i, it in enumerate(items):
        if 1 <= it.round_idx <= R_max:
            idx_by_round[it.round_idx].append(i)

    res: Dict[int, Dict[str, float]] = {}
    cum: List[int] = []
    prev = None
    for r in range(1, R_max + 1):
        cum.extend(idx_by_round.get(r, []))
        if not cum:
            res[r] = {"n_texts": 0, "H": 0.0, "Nstar": 1.0, "delta": float("nan")}
            continue
        M = emb[cum, :]
        H, Nstar = compute_Nstar(M)
        delta = (Nstar - prev) if prev is not None else float("nan")
        prev = Nstar
        res[r] = {"n_texts": int(M.shape[0]), "H": H, "Nstar": Nstar, "delta": delta}
    return res


# -------------------------
# Mode 2: per_question_agent
# -------------------------
def aggregate_per_question_agent_texts(
    file_id: str,
    rows: List[Dict[str, Any]],
    mode: str = "concat",
    join_with: str = "\n\n[R-SEP]\n\n"
) -> Tuple[List[TextItem], Dict[int, List[int]]]:
    """
    Build one aggregated text per question per agent.
    Returns:
      agg_items: list[TextItem] (round_idx=0)
      q2idx: question -> indices into agg_items
    """
    assert mode in ("concat", "last")
    agg_items: List[TextItem] = []
    q2idx: Dict[int, List[int]] = {}

    for q_idx, one_q in enumerate(rows):
        rks = digit_keys(one_q)
        per_agent_round_texts: Dict[str, List[str]] = defaultdict(list)

        for rk in rks:
            round_obj = one_q.get(rk, {})
            responses = round_obj.get("responses", {})
            if not isinstance(responses, dict):
                continue
            for agent_id, payload in responses.items():
                t = get_text(payload)
                if t:
                    per_agent_round_texts[str(agent_id)].append(t)

        for aid, lst in per_agent_round_texts.items():
            if not lst:
                continue
            if mode == "last":
                txt = lst[-1].strip()
            else:
                txt = join_with.join([x.strip() for x in lst if x and x.strip()])
            if not txt:
                continue
            idx = len(agg_items)
            agg_items.append(TextItem(file_id=file_id, q_idx=q_idx, round_idx=0, agent_id=aid, text=txt))
            q2idx.setdefault(q_idx, []).append(idx)

    return agg_items, q2idx


def compute_Nstar_per_question(emb: np.ndarray, q2idx: Dict[int, List[int]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for q, idxs in q2idx.items():
        if not idxs:
            out[q] = {"n_agents": 0, "H": 0.0, "Nstar": 1.0}
            continue
        M = emb[idxs, :]
        H, Nstar = compute_Nstar(M)
        out[q] = {"n_agents": int(M.shape[0]), "H": H, "Nstar": Nstar}
    return out


# -------------------------
# Mode 3: round_agent_avg (fixed)
# -------------------------
def build_embeddings_by_q_r_a(
    items: List[TextItem],
    emb: np.ndarray
) -> Dict[Tuple[int, int, str], List[np.ndarray]]:
    """
    Fix for D(2): allow multiple entries per (q,r,a) by storing a list and averaging later.
    """
    mp: Dict[Tuple[int, int, str], List[np.ndarray]] = defaultdict(list)
    for i, it in enumerate(items):
        key = (it.q_idx, it.round_idx, it.agent_id)
        mp[key].append(emb[i, :])
    return mp


def compute_round_agent_avg(
    items: List[TextItem],
    emb: np.ndarray,
    R_max: int,
    require_question_intersection: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Per round r:
      - For each agent, average embeddings across questions (and within (q,r,a) if multiple texts)
      - To fix D(1): optionally restrict to intersection set of questions where all agents have data at that round.
    Output N* across agents for each round.

    n_agents is fixed per round (= number of agents with any data in round r),
    but can shrink if intersection is required and some agents become empty after intersection filtering.
    """
    mp = build_embeddings_by_q_r_a(items, emb)

    res: Dict[int, Dict[str, float]] = {}
    prev = None

    for r in range(1, R_max + 1):
        # agents present at round r
        agents_r = sorted({a for (q, rr, a) in mp.keys() if rr == r})
        if not agents_r:
            res[r] = {"n_agents": 0, "n_questions": 0, "H": 0.0, "Nstar": 1.0, "delta": float("nan")}
            continue

        # for each agent: which questions exist?
        agent_qs: Dict[str, set] = {}
        for a in agents_r:
            qs = {q for (q, rr, aa) in mp.keys() if rr == r and aa == a}
            agent_qs[a] = qs

        if require_question_intersection:
            # intersection of questions across all agents at this round
            q_inter = set.intersection(*[agent_qs[a] for a in agents_r]) if agents_r else set()
        else:
            # union (less strict, but less comparable)
            q_inter = set.union(*[agent_qs[a] for a in agents_r]) if agents_r else set()

        if not q_inter:
            res[r] = {"n_agents": len(agents_r), "n_questions": 0, "H": 0.0, "Nstar": 1.0, "delta": float("nan")}
            continue

        # build agent vectors
        mats = []
        kept_agents = []
        for a in agents_r:
            vecs = []
            for q in sorted(q_inter):
                lst = mp.get((q, r, a), [])
                if not lst:
                    # if intersection, this shouldn't happen; if union, it can
                    continue
                if len(lst) == 1:
                    vecs.append(lst[0])
                else:
                    vecs.append(np.mean(np.vstack(lst), axis=0))
            if vecs:
                v = np.mean(np.vstack(vecs), axis=0)
                mats.append(v)
                kept_agents.append(a)

        if not mats:
            res[r] = {"n_agents": 0, "n_questions": int(len(q_inter)), "H": 0.0, "Nstar": 1.0, "delta": float("nan")}
            continue

        M = np.vstack(mats).astype(np.float32)
        H, Nstar = compute_Nstar(M)
        delta = (Nstar - prev) if prev is not None else float("nan")
        prev = Nstar

        res[r] = {
            "n_agents": int(len(kept_agents)),
            "n_questions": int(len(q_inter)),
            "H": H,
            "Nstar": Nstar,
            "delta": delta,
        }

    return res


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
# CSV writing (3 tables)
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
# Main per-file processing
# -------------------------
def process_file(
    jsonl_path: str,
    embedder: Embedder,
    mode: str,
    batch_size: int,
    agg_mode: str,
    agg_sep: str,
    require_intersection: bool,
    cache_dir_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      file_summary_row,
      round_rows (for round_table),
      question_rows (for question_table)
    """
    t0 = time.time()
    file_id = os.path.basename(jsonl_path)
    out: Dict[str, Any] = {
        "file": file_id,
        "path": jsonl_path,
        "mode": mode,
        "error": "",
    }
    round_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []

    try:
        rows = load_jsonl(jsonl_path)
        items, final_answers_list, gt_list, R_max = extract_dataset_B(rows, file_id=file_id)
        out["questions"] = int(len(rows))
        out["R_max"] = int(R_max)
        out["texts"] = int(len(items))

        # Part B per-question vectors
        ginis = np.full(len(rows), np.nan, dtype=np.float64)
        concs = np.full(len(rows), np.nan, dtype=np.float64)
        acc01 = np.full(len(rows), np.nan, dtype=np.float64)
        valid_counts = np.zeros(len(rows), dtype=np.int32)

        for q in range(len(rows)):
            fa_raw = final_answers_list[q]
            fa = [normalize_choice_token(x) for x in (fa_raw or [])]
            fa = [x for x in fa if x is not None]
            valid_counts[q] = int(len(fa))

            gt = normalize_choice_token(gt_list[q])
            ginis[q] = gini_proxy(fa)
            concs[q] = consensus(fa)
            mv = majority_vote(fa)
            if mv is None or gt is None:
                acc01[q] = np.nan
            else:
                acc01[q] = 1.0 if mv == gt else 0.0

        out["valid_answers_mean"] = float(np.mean(valid_counts)) if len(valid_counts) else float("nan")
        out["gini_mean"] = float(np.nanmean(ginis))
        out["consensus_mean"] = float(np.nanmean(concs))
        out["acc_mean"] = float(np.nanmean(acc01))

        # Mode-specific embeddings and N*
        npy_path, meta_path = Embedder.derive_cache_paths(jsonl_path, cache_dir_name=cache_dir_name)

        if mode == "per_question_agent":
            agg_items, q2idx = aggregate_per_question_agent_texts(file_id=file_id, rows=rows, mode=agg_mode, join_with=agg_sep)
            texts = [it.text for it in agg_items]

            sig_extra = {"analysis_mode": mode, "agg_mode": agg_mode, "agg_sep_sha1": sha1(agg_sep)}
            emb = embedder.get_embeddings_with_cache(texts, npy_path, meta_path, batch_size=batch_size, signature_extra=sig_extra)
            out["dim"] = int(emb.shape[1])

            nstar_q = compute_Nstar_per_question(emb, q2idx)
            ns_by_q = np.full(len(rows), np.nan, dtype=np.float64)

            for q_idx, d in nstar_q.items():
                ns_by_q[q_idx] = float(d["Nstar"])
                question_rows.append({
                    "file": file_id,
                    "q_idx": int(q_idx),
                    "n_agents": int(d["n_agents"]),
                    "H": float(d["H"]),
                    "Nstar": float(d["Nstar"]),
                    "gini": float(ginis[q_idx]),
                    "consensus": float(concs[q_idx]),
                    "acc01": float(acc01[q_idx]),
                    "valid_answers": int(valid_counts[q_idx]),
                })

            out["Nstar_q_mean"] = float(np.nanmean(ns_by_q))
            out["Nstar_q_std"] = float(np.nanstd(ns_by_q))
            out["corr_Nstar_gini"] = float(safe_corr(ns_by_q, ginis))
            out["corr_Nstar_consensus"] = float(safe_corr(ns_by_q, concs))
            out["corr_Nstar_acc01"] = float(safe_corr(ns_by_q, acc01))

            # For round_table in this mode, we can optionally provide "Nstar_by_round" based on original items
            # (still useful, but keep it explicit: this is cumulative text N* diagnostic)
            # We will compute BOTH cumulative-text and agent-avg per round only if possible? To keep table stable,
            # we compute cumulative-text from original items.
            # Need embeddings for original items too, which could be expensive; skip by default.
            # Instead, emit round_table as empty in this mode; file_summary captures main metrics.
            # Users can run round modes separately when needed.

        elif mode == "round_cum_text":
            texts = [it.text for it in items]
            sig_extra = {"analysis_mode": mode}
            emb = embedder.get_embeddings_with_cache(texts, npy_path, meta_path, batch_size=batch_size, signature_extra=sig_extra)
            out["dim"] = int(emb.shape[1])

            nstar_r = compute_round_cum_text(items, emb, R_max=R_max)
            out["Nstar_last"] = float(nstar_r.get(R_max, {}).get("Nstar", np.nan))
            out["H_last"] = float(nstar_r.get(R_max, {}).get("H", np.nan))

            for r in range(1, R_max + 1):
                d = nstar_r.get(r, {})
                round_rows.append({
                    "file": file_id,
                    "round": int(r),
                    "kind": "cum_text",
                    "n_texts": int(d.get("n_texts", 0)),
                    "n_agents": "",
                    "n_questions": "",
                    "H": float(d.get("H", np.nan)),
                    "Nstar": float(d.get("Nstar", np.nan)),
                    "delta": float(d.get("delta", np.nan)),
                })

        elif mode == "round_agent_avg":
            texts = [it.text for it in items]
            sig_extra = {"analysis_mode": mode, "require_intersection": bool(require_intersection)}
            emb = embedder.get_embeddings_with_cache(texts, npy_path, meta_path, batch_size=batch_size, signature_extra=sig_extra)
            out["dim"] = int(emb.shape[1])

            nstar_r = compute_round_agent_avg(items, emb, R_max=R_max, require_question_intersection=require_intersection)
            out["Nstar_last"] = float(nstar_r.get(R_max, {}).get("Nstar", np.nan))
            out["H_last"] = float(nstar_r.get(R_max, {}).get("H", np.nan))
            out["round_require_intersection"] = int(1 if require_intersection else 0)

            for r in range(1, R_max + 1):
                d = nstar_r.get(r, {})
                round_rows.append({
                    "file": file_id,
                    "round": int(r),
                    "kind": "agent_avg",
                    "n_texts": "",
                    "n_agents": int(d.get("n_agents", 0)),
                    "n_questions": int(d.get("n_questions", 0)),
                    "H": float(d.get("H", np.nan)),
                    "Nstar": float(d.get("Nstar", np.nan)),
                    "delta": float(d.get("delta", np.nan)),
                })

        else:
            raise ValueError(f"Unknown mode: {mode}")

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    out["secs"] = float(round(time.time() - t0, 3))
    return out, round_rows, question_rows


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified N* analysis for MAS results (NV-Embed-v2).")
    parser.add_argument("jsonl_path", type=str, help="Path to a .jsonl file or a directory containing .jsonl files.")
    parser.add_argument("--mode", type=str, default="round_agent_avg",
                        choices=["round_cum_text", "per_question_agent", "round_agent_avg"],
                        help="Analysis mode.")
    parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2", help="SentenceTransformer model id or path.")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: auto.")
    parser.add_argument("--max-seq-length", type=int, default=32768, help="SentenceTransformer max_seq_length.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size. Default: auto (cuda=16, cpu=4).")
    parser.add_argument("--no-eos", action="store_true", help="Do NOT append eos_token to each text.")
    parser.add_argument("--no-normalize", action="store_true", help="Do NOT normalize embeddings.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar in encoding.")
    parser.add_argument("--cache-dir-name", type=str, default="cache", help="Cache subdir name under each jsonl directory.")

    # per_question_agent params
    parser.add_argument("--agg-mode", type=str, default="concat", choices=["concat", "last"],
                        help="Aggregation for per_question_agent: concat or last.")
    parser.add_argument("--agg-sep", type=str, default="\n\n[R-SEP]\n\n", help="Separator used in concat aggregation.")

    # round_agent_avg params
    parser.add_argument("--no-intersection", action="store_true",
                        help="For round_agent_avg: use union of questions instead of intersection across agents (less comparable).")

    # outputs
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Default: base dir of input (file's dir or the directory itself).")
    parser.add_argument("--cache-folder", type=str, default=None,
                            help="Hugging Face model cache directory. Default: ~/.cache/huggingface/")
    
    args = parser.parse_args()

    input_path = args.jsonl_path
    jsonl_files = discover_jsonl_paths(input_path)
    if not jsonl_files:
        print(f"No .jsonl files found at or under: {input_path}")
        return

    if args.out_dir is None:
        if os.path.isdir(input_path):
            out_dir = os.path.abspath(input_path)
        else:
            out_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    else:
        out_dir = os.path.abspath(args.out_dir)

    # batch size default
    if args.batch_size is None:
        # conservative defaults; user can override
        if args.device is None:
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = (args.device.lower() == "cuda")
        batch_size = 16 if use_cuda else 4
    else:
        batch_size = int(args.batch_size)

    embedder = Embedder(
            model_path=args.model,
            device=args.device,
            max_seq_length=args.max_seq_length,
            add_eos=(not args.no_eos),
            normalize_embeddings=(not args.no_normalize),
            trust_remote_code=True,
            show_progress_bar=(not args.no_progress),
            cache_folder=args.cache_folder,  
        )

    file_summary_rows: List[Dict[str, Any]] = []
    round_rows_all: List[Dict[str, Any]] = []
    question_rows_all: List[Dict[str, Any]] = []

    print(f"Found {len(jsonl_files)} jsonl files.")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {out_dir}")
    print("")

    for idx, p in enumerate(jsonl_files, 1):
        print(f"[{idx}/{len(jsonl_files)}] Processing: {p}")
        summary_row, round_rows, question_rows = process_file(
            jsonl_path=p,
            embedder=embedder,
            mode=args.mode,
            batch_size=batch_size,
            agg_mode=args.agg_mode,
            agg_sep=args.agg_sep,
            require_intersection=(not args.no_intersection),
            cache_dir_name=args.cache_dir_name,
        )
        if summary_row.get("error"):
            print(f"  ERROR: {summary_row['error']}")
        else:
            # concise status
            if args.mode == "per_question_agent":
                print(f"  questions={summary_row.get('questions')} | N*(q)_mean={summary_row.get('Nstar_q_mean', np.nan):.3f} | acc_mean={summary_row.get('acc_mean', np.nan):.3f}")
            else:
                print(f"  R_max={summary_row.get('R_max')} | N*_last={summary_row.get('Nstar_last', np.nan):.3f} | acc_mean={summary_row.get('acc_mean', np.nan):.3f}")
        file_summary_rows.append(summary_row)
        round_rows_all.extend(round_rows)
        question_rows_all.extend(question_rows)

    # -------------------------
    # Write outputs (3 tables)
    # -------------------------
    file_summary_path = os.path.join(out_dir, "file_summary.csv")
    round_table_path = os.path.join(out_dir, "round_table.csv")
    question_table_path = os.path.join(out_dir, "question_table.csv")

    # file_summary columns: stable superset
    file_cols = [
        "file", "path", "mode", "error", "secs",
        "questions", "R_max", "texts", "dim",
        "valid_answers_mean", "gini_mean", "consensus_mean", "acc_mean",
        # round modes:
        "Nstar_last", "H_last", "round_require_intersection",
        # per_question mode:
        "Nstar_q_mean", "Nstar_q_std",
        "corr_Nstar_gini", "corr_Nstar_consensus", "corr_Nstar_acc01",
    ]
    write_csv(file_summary_path, file_summary_rows, file_cols)

    # round_table columns
    round_cols = ["file", "round", "kind", "n_texts", "n_agents", "n_questions", "H", "Nstar", "delta"]
    write_csv(round_table_path, round_rows_all, round_cols)

    # question_table columns
    q_cols = ["file", "q_idx", "n_agents", "H", "Nstar", "gini", "consensus", "acc01", "valid_answers"]
    write_csv(question_table_path, question_rows_all, q_cols)

    print("")
    print(f"Saved: {file_summary_path}")
    print(f"Saved: {round_table_path}")
    print(f"Saved: {question_table_path}")


if __name__ == "__main__":
    main()