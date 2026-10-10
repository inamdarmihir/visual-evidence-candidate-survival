"""Parity checks, the dense-candidate -> late-rerank experiment, and trace/table construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import qdrant_store as qs
from .config import RunConfig, corpus_point_id
from .data import qrels_by_query
from .metrics import loss_label
from .scoring import (exact_dense_reference, rank_order, ranks_equivalent, reference_mean_maxsim)
from .util import atomic_write_json, read_jsonl, write_jsonl

SCORE_ATOL = 1e-4      # float32 server vs torch; scores live in ~[-1, 1]
TIE_ATOL = 1e-5


def pid_map(cids):
    return {corpus_point_id(c): int(c) for c in cids}


def load_dense(pcache, qcache, bundle):
    P = np.stack([pcache.read(c)["dense"] for c in bundle.corpus_ids]).astype(np.float32)
    qids = [int(q) for q in bundle.queries.query_id]
    Q = np.stack([qcache.read(q)["dense"] for q in qids]).astype(np.float32)
    return P, Q, qids


def run_parity(cfg, c, name, bundle, pcache, qcache, pmap, device="cpu", n_queries=6, n_pages=8):
    """Real queries vs real pages: torch reference (masked, padded as in the model card) vs cached-unpadded vs
    Qdrant MAX_SIM / n_q. Raises if parity fails."""
    qlens = {int(q): qcache.meta(q)["n_tokens"] for q in bundle.queries.query_id}
    plens = {int(p): pcache.meta(p)["n_tokens"] for p in bundle.corpus_ids}
    qs_sorted = sorted(qlens, key=lambda q: (qlens[q], q))
    pick_q = sorted({qs_sorted[0], qs_sorted[-1], *[qs_sorted[i] for i in np.linspace(0, len(qs_sorted) - 1, n_queries).astype(int)]})
    ps_sorted = sorted(plens, key=lambda p: (plens[p], p))
    pick_p = sorted({ps_sorted[0], ps_sorted[-1], *[ps_sorted[i] for i in np.linspace(0, len(ps_sorted) - 1, n_pages).astype(int)]})
    q_arr = [qcache.read(q)["tokens"] for q in pick_q]
    d_arr = [pcache.read(p)["tokens"] for p in pick_p]
    ref = reference_mean_maxsim(q_arr, d_arr, device=device)                       # (nq, nd) padded + masks
    cpu_ref = reference_mean_maxsim(q_arr, d_arr, device="cpu") if device != "cpu" else ref
    unmasked = reference_mean_maxsim(q_arr, d_arr, device=device, masked=False, extra_pad=64)
    masked_extra = reference_mean_maxsim(q_arr, d_arr, device=device, masked=True, extra_pad=64)
    # per-pair unpadded exact recomputation straight from cached vectors
    direct = np.array([[(q @ d.T).max(1).mean() for d in d_arr] for q in q_arr], dtype=np.float32)
    rows, worst, tie_fail = [], 0.0, 0
    min_best_sim = np.inf
    for i, q in enumerate(pick_q):
        ids, sc = qs.rerank_id_filter(c, name, q_arr[i], pick_p, pmap)
        got = dict(zip(ids, sc))
        assert set(got) == set(pick_p), "Qdrant returned a different page set"
        d = np.array([got[p] for p in pick_p])
        worst = max(worst, float(np.abs(d - ref[i]).max()))
        r_ids, r_sc = rank_order(ref[i], pick_p)
        ok, tied, bad = ranks_equivalent(r_ids, r_sc, ids, atol=TIE_ATOL)
        tie_fail += bad
        min_best_sim = min(min_best_sim, min(float((q_arr[i] @ dd.T).max(1).min()) for dd in d_arr))
        rows.append({"query_id": pick_q[i], "q_tokens": int(q_arr[i].shape[0]), "max_abs_diff_qdrant_vs_ref": float(np.abs(d - ref[i]).max()),
                     "rank_ties": tied, "rank_mismatches": bad})
    rep = {
        "queries": pick_q, "pages": pick_p, "query_token_counts": [int(a.shape[0]) for a in q_arr],
        "page_token_counts": [int(a.shape[0]) for a in d_arr], "score_atol": SCORE_ATOL, "tie_atol": TIE_ATOL,
        "max_abs_qdrant_vs_ref": worst,
        "max_abs_ref_vs_unpadded_direct": float(np.abs(ref - direct).max()),
        "max_abs_cpu_vs_device_ref": float(np.abs(ref - cpu_ref).max()), "reference_device": str(device),
        "extra_zero_padding_rows_exercised": 64,
        "max_abs_masked_extra_padding_vs_ref": float(np.abs(ref - masked_extra).max()),
        "max_abs_masked_vs_unmasked_padded_docs": float(np.abs(ref - unmasked).max()),
        "real_page_lengths_all_equal": bool(len(set(int(a.shape[0]) for a in d_arr)) == 1),
        "min_over_query_tokens_of_best_page_sim": float(min_best_sim),
        "negative_best_similarity_present": bool(min_best_sim < 0),
        "rank_mismatches_beyond_ties": int(tie_fail), "rows": rows,
        "divisor": "Qdrant MAX_SIM sum divided by valid query-token count (constant per query; never doc tokens)"}
    rep["passed"] = bool(worst <= SCORE_ATOL and rep["max_abs_ref_vs_unpadded_direct"] <= SCORE_ATOL
                         and rep["max_abs_masked_extra_padding_vs_ref"] <= SCORE_ATOL and tie_fail == 0)
    if not rep["passed"]:
        raise AssertionError("Qdrant/torch MeanMaxSim parity FAILED: " + json.dumps({k: v for k, v in rep.items() if k != "rows"}))
    return rep


def run_queries(cfg: RunConfig, c, name, bundle, pcache, qcache, results_dir, progress=None):
    """Dense exact top-100 once, nested prefixes per B, prefetch-rerank with independent candidate verification."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pm = pid_map(bundle.corpus_ids)
    P, Q, qids = load_dense(pcache, qcache, bundle)
    ref = exact_dense_reference(Q, P, bundle.corpus_ids, limit=max(cfg.budgets))   # blockwise cosine reference
    bar = progress(total=len(qids), desc="queries") if progress else None
    for qi, qid in enumerate(qids):
        f = results_dir / f"query_{qid}.json"
        if f.exists():
            bar and bar.update(1)
            continue
        z = qcache.read(qid)
        dids, dsc = qs.dense_topk(c, name, z["dense"], pm, limit=max(cfg.budgets), exact=cfg.dense_search_exact)
        ok, tied, bad = ranks_equivalent(ref[qi][0], ref[qi][1], dids, atol=TIE_ATOL)
        if not ok:
            raise AssertionError(f"query {qid}: Qdrant exact dense ranking != blockwise cosine reference ({bad} mismatches)")
        # canonical saved ranking: Qdrant scores, explicit tie rule (score desc, corpus_id asc)
        dids, dsc = rank_order(dsc, dids)
        per_B = {}
        for B in cfg.budgets:
            prefix = dids[:B]
            method = "prefetch(dense exact, limit=B) -> late MAX_SIM / n_q"
            rids, rsc = qs.rerank_prefetch(c, name, z["dense"], z["tokens"], B, pm, exact=cfg.dense_search_exact)
            if set(rids) != set(prefix) or len(rids) != B:       # independent check against the SAVED prefix
                method = "id_filter_fallback(saved candidates) -> late MAX_SIM / n_q"
                rids, rsc = qs.rerank_id_filter(c, name, z["tokens"], prefix, pm)
                assert set(rids) == set(prefix), "reranker left the candidate set"
            rids, rsc = rank_order(rsc, rids)
            per_B[str(B)] = {"method": method, "cand_ids": prefix, "cand_dense_scores": dsc[:B],
                             "rerank_ids": rids, "rerank_scores": rsc, "n_q_tokens": int(z["tokens"].shape[0])}
        atomic_write_json(f, {"query_id": qid, "dense_ids": dids, "dense_scores": dsc,
                              "dense_ref_tied_positions": tied, "per_B": per_B})
        bar and bar.update(1)
    bar and bar.close()
    return load_results(results_dir, cfg)


def load_results(results_dir, cfg):
    out = {}
    for f in sorted(Path(results_dir).glob("query_*.json")):
        r = json.loads(f.read_text())
        r["per_B"] = {int(k): v for k, v in r["per_B"].items()}
        out[int(r["query_id"])] = r
    return out


def rank_in(lst, cid):
    return lst.index(cid) + 1 if cid in lst else None


def build_traces(cfg, bundle, results, full_late=None, geometry=None):
    """Per-query trace dicts + page-level rows. Page numbers: stored 0-based `page_number_in_doc`; 1-based = +1."""
    grades = qrels_by_query(bundle.qrels)
    qrel_rows = {(int(r.query_id), int(r.corpus_id)): r for r in bundle.qrels.itertuples(index=False)}
    qmeta = bundle.queries.set_index("query_id")
    traces, page_rows = [], []
    for qid in sorted(results):
        r, g = results[qid], grades[qid]
        qm = qmeta.loc[qid]
        pos = []
        for cid, gr in sorted(g.items()):
            qr = qrel_rows[(qid, cid)]
            pos.append({"corpus_id": cid, "grade": gr, "doc_id": bundle.doc_of[cid],
                        "page_number_in_doc_0based": bundle.page_of[cid], "page_number_1based": bundle.page_of[cid] + 1,
                        "row_offset": bundle.row_of[cid], "content_type": [str(x) for x in qr.content_type],
                        "bounding_boxes": [{k: (int(v) if k != "annotator" else int(v)) for k, v in b.items()} for b in qr.bounding_boxes],
                        "bbox_coordinate_space": "original page image pixels (x right, y down)",
                        **({"geometry": geometry[cid]} if geometry and cid in geometry else {})})
        per_B = {}
        for B in cfg.budgets:
            pb = r["per_B"][B]
            entry = {"method": pb["method"], "candidates": pb["cand_ids"], "rerank_ids": pb["rerank_ids"],
                     "rerank_scores_mean_maxsim": pb["rerank_scores"], "lost_at_candidate_stage": [c for c in g if c not in pb["cand_ids"]]}
            for k in cfg.ks:
                top = pb["rerank_ids"][:k]
                entry[f"top{k}"] = top
                entry[f"lost_by_reranking_top{k}"] = [c for c in g if c in pb["cand_ids"] and c not in top]
            per_B[str(B)] = entry
        traces.append({"query_id": qid, "query": qm["query"], "query_types": [str(x) for x in qm.query_types],
                       "content_type": [str(x) for x in qm.content_type], "n_positive": len(g), "positives": pos,
                       "dense_top100": r["dense_ids"], "dense_top100_scores": r["dense_scores"], "per_B": per_B,
                       "full_late_top_ids": (full_late or {}).get(qid, None)})
        for p in pos:
            cid = p["corpus_id"]
            d_rank = rank_in(r["dense_ids"], cid)
            for B in cfg.budgets:
                pb = r["per_B"][B]
                rr = rank_in(pb["rerank_ids"], cid)
                rs = pb["rerank_scores"][rr - 1] if rr else None
                for k in cfg.ks:
                    in_c = cid in pb["cand_ids"]
                    in_dense_k = d_rank is not None and d_rank <= k
                    in_rr_k = rr is not None and rr <= k
                    page_rows.append({
                        "query_id": qid, "corpus_id": cid, "grade": p["grade"], "B": B, "k": k,
                        "dense_rank_top100": d_rank, "in_candidates": in_c, "rerank_rank_within_B": rr,
                        "rerank_score_mean_maxsim": rs, "in_dense_topk": in_dense_k, "in_rerank_topk": in_rr_k,
                        "label": loss_label(in_c, rr, k),
                        "vs_dense_topk": ("rescued_by_rerank" if in_rr_k and not in_dense_k else
                                          "displaced_by_rerank" if in_dense_k and not in_rr_k else
                                          "kept_in_topk" if in_rr_k else "not_in_topk_either"),
                        "doc_id": p["doc_id"], "page_number_in_doc_0based": p["page_number_in_doc_0based"],
                        "page_number_1based": p["page_number_1based"], "row_offset": p["row_offset"],
                        "content_type": "|".join(p["content_type"]), "bounding_boxes_json": json.dumps(p["bounding_boxes"]),
                        "orig_w": (p.get("geometry") or {}).get("orig_w"), "orig_h": (p.get("geometry") or {}).get("orig_h")})
    return traces, pd.DataFrame(page_rows)


def full_late_reference(cfg, bundle, pcache, qcache, out_dir, device="cpu", progress=None, chunk=64):
    """Optional unrestricted late-interaction reference (bounded memory, resumable in page chunks)."""
    import torch
    from .scoring import full_late_scores
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qids = [int(q) for q in bundle.queries.query_id]
    q_arrs = [qcache.read(q)["tokens"] for q in qids]
    cids = bundle.corpus_ids
    parts = []
    for s in range(0, len(cids), chunk):
        sub = cids[s:s + chunk]
        f = out_dir / f"chunk_{s:06d}.npy"
        if not f.exists():
            gen = full_late_scores(q_arrs, ((c, pcache.read(c)["tokens"]) for c in sub), len(sub), device=device)
            m = np.stack([sc for _, sc in gen], axis=1)                  # (nq, len(sub))
            tmp = f.with_suffix(".tmp.npy")
            np.save(tmp, m.astype(np.float32))
            tmp.replace(f)
        parts.append(np.load(f))
        if progress:
            progress(f"full late {min(s + chunk, len(cids))}/{len(cids)}")
    S = np.concatenate(parts, axis=1)
    top = {}
    for i, q in enumerate(qids):
        ids, _ = rank_order(S[i], cids)
        top[q] = ids[:max(cfg.ks + cfg.budgets)]
    return S, top
