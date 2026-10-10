"""Survival metrics. G = positive-qrel pages, C = candidates. Macro (per query) and micro (pooled) reported."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

NAN = float("nan")


def gain(score: int) -> float:
    return 2.0 ** score - 1.0            # 1 -> 1, 2 -> 3; unjudged/unlabelled pages -> 0


def ndcg_at_k(ranked, grades: dict, k: int) -> float:
    """Ranked list must already be tie-broken deterministically. Unjudged pages gain 0 (not true negatives)."""
    dcg = sum(gain(grades.get(c, 0)) / math.log2(i + 2) for i, c in enumerate(ranked[:k]) if c in grades)
    ideal = sorted((gain(g) for g in grades.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else NAN


def survival(grades: dict, retrieved) -> dict:
    ret = set(retrieved)
    G = set(grades)
    s1 = {c for c, g in grades.items() if g == 1}
    s2 = {c for c, g in grades.items() if g == 2}
    hit = len(G & ret)
    return {
        "n_pos": len(G), "n_hit": hit, "recall": hit / len(G), "coverage": float(G <= ret),
        "n_s1": len(s1), "hit_s1": len(s1 & ret),
        "recall_s1": len(s1 & ret) / len(s1) if s1 else NAN, "coverage_s1": float(s1 <= ret) if s1 else NAN,
        "n_s2": len(s2), "hit_s2": len(s2 & ret),
        "recall_s2": len(s2 & ret) / len(s2) if s2 else NAN,
        "any_s2": float(bool(s2 & ret)) if s2 else NAN,   # availability of >=1 complete-answer-capable page
    }


def per_query_table(results: dict, grades_by_q: dict, budgets, ks, full_late: dict | None = None) -> pd.DataFrame:
    """results[qid] = {'dense_ids': [...100], 'per_B': {B: {'cand_ids': [...], 'rerank_ids': [...]}}}"""
    rows = []
    for qid, r in results.items():
        g = grades_by_q[qid]
        for B in budgets:
            pb = r["per_B"][B]
            rows.append({"query_id": qid, "stage": "candidate", "B": B, "k": NAN, **survival(g, pb["cand_ids"]),
                         "ndcg": NAN})
            for k in ks:
                rows.append({"query_id": qid, "stage": "rerank", "B": B, "k": k, **survival(g, pb["rerank_ids"][:k]),
                             "ndcg": ndcg_at_k(pb["rerank_ids"], g, k)})
        for k in ks:
            rows.append({"query_id": qid, "stage": "dense", "B": NAN, "k": k, **survival(g, r["dense_ids"][:k]),
                         "ndcg": ndcg_at_k(r["dense_ids"], g, k)})
            if full_late is not None and qid in full_late:
                rows.append({"query_id": qid, "stage": "full_late", "B": NAN, "k": k,
                             **survival(g, full_late[qid][:k]), "ndcg": ndcg_at_k(full_late[qid], g, k)})
    return pd.DataFrame(rows)


METRICS = ["recall", "coverage", "recall_s1", "coverage_s1", "recall_s2", "any_s2", "ndcg"]


def bootstrap_means(X: np.ndarray, n_boot=2000, seed=42, idx=None):
    """X (n_queries, m) with NaN = not applicable. Resamples whole query rows (same draws for every column)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n)) if idx is None else idx
    out = np.full((n_boot, X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        col = X[:, j]
        v = col[idx]
        with np.errstate(invalid="ignore"):
            cnt = (~np.isnan(v)).sum(1)
            out[:, j] = np.where(cnt > 0, np.nansum(v, 1) / np.maximum(cnt, 1), np.nan)
    return out


def ci(samples: np.ndarray):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns = metric not applicable
        return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


def aggregate(pq: pd.DataFrame, n_boot=2000, seed=42) -> pd.DataFrame:
    """One row per (stage,B,k): macro means + 95% bootstrap CI for each metric, micro totals, denominators."""
    n = pq.query_id.nunique()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))          # identical draws everywhere => paired
    rows = []
    for (stage, B, k), d in pq.groupby(["stage", "B", "k"], dropna=False, sort=False):
        d = d.sort_values("query_id")
        X = d[METRICS].to_numpy(float)
        bs = bootstrap_means(X, n_boot, seed, idx if len(d) == n else None)
        lo, hi = ci(bs)
        row = {"stage": stage, "B": B, "k": k, "n_queries": len(d), "denominator_pages": int(d.n_pos.sum()),
               "micro_recall": d.n_hit.sum() / d.n_pos.sum(),
               "micro_recall_s1": d.hit_s1.sum() / d.n_s1.sum() if d.n_s1.sum() else NAN,
               "n_queries_with_s1": int((d.n_s1 > 0).sum()), "n_queries_with_s2": int((d.n_s2 > 0).sum()),
               "n_pages_s1": int(d.n_s1.sum()), "n_pages_s2": int(d.n_s2.sum()), "n_boot": n_boot, "seed": seed}
        for j, mname in enumerate(METRICS):
            row[f"macro_{mname}"] = np.nanmean(X[:, j]) if np.isfinite(X[:, j]).any() else NAN
            row[f"{mname}_ci_lo"], row[f"{mname}_ci_hi"] = lo[j], hi[j]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["B"], out["k"] = out["B"].astype("Int64"), out["k"].astype("Int64")
    return out


def paired_diff(pq: pd.DataFrame, a: dict, b: dict, metric: str, n_boot=2000, seed=42) -> dict:
    """mean(a-b) over whole-query resamples; a/b are filters like {'stage':'rerank','B':25,'k':5}."""
    def sel(f):
        d = pq
        for kx, v in f.items():
            d = d[d[kx] == v]
        return d.set_index("query_id")[metric].sort_index()
    x, y = sel(a), sel(b)
    assert (x.index == y.index).all()
    diff = (x - y).to_numpy(float)
    bs = bootstrap_means(diff[:, None], n_boot, seed)[:, 0]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"metric": metric, "a": str(a), "b": str(b), "n_queries": int(np.isfinite(diff).sum()),
            "mean_diff": float(np.nanmean(diff)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_a_better": int((diff > 0).sum()), "n_b_better": int((diff < 0).sum()), "n_equal": int((diff == 0).sum())}


def loss_label(in_cand: bool, rank_after_rerank, k: int) -> str:
    if not in_cand:
        return "candidate_loss"
    return "survived_top_k" if rank_after_rerank <= k else "reranking_loss"
