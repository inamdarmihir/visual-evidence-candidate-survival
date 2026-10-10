"""Reference scoring: model-card MeanMaxSim, blockwise exact dense cosine, bounded full-corpus late scoring."""
from __future__ import annotations

import numpy as np
import torch


def pad_stack(arrs, device="cpu", extra=0):
    """list of unpadded (n_i,128) -> padded tensor + mask (padding rebuilt ONLY to call the reference).
    `extra` appends additional zero/masked rows so masking is exercised even when real lengths are equal."""
    L = max(a.shape[0] for a in arrs) + extra
    x = torch.zeros(len(arrs), L, arrs[0].shape[1])
    m = torch.zeros(len(arrs), L, dtype=torch.long)
    for i, a in enumerate(arrs):
        x[i, :a.shape[0]] = torch.from_numpy(a)
        m[i, :a.shape[0]] = 1
    return x.to(device), m.to(device)


def reference_mean_maxsim(q_arrs, d_arrs, device="cpu", masked=True, extra_pad=0):
    """sentence_transformers.util.mean_maxsim with a_mask/b_mask (the model-card scoring)."""
    from sentence_transformers.util import mean_maxsim
    qa, qm = pad_stack(q_arrs, device, extra_pad)
    da, dm = pad_stack(d_arrs, device, extra_pad)
    if masked:
        return mean_maxsim(qa, da, a_mask=qm, b_mask=dm).cpu().numpy()
    # hazard diagnostic: padding left in place and unmasked (zero doc vectors can beat negative sims)
    from sentence_transformers.util import maxsim
    s = maxsim(qa, da, a_mask=qm, b_mask=None)  # doc side unmasked
    return (s / qm.sum(1, keepdim=True).clamp_min(1)).cpu().numpy()


def rank_order(scores, ids):
    """Deterministic ranking: score desc, then id asc (explicit tie rule)."""
    o = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), int(ids[i])))
    return [ids[i] for i in o], [float(scores[i]) for i in o]


def exact_dense_reference(Q, P, cids, limit=100, block=256, device="cpu"):
    """Blockwise cosine reference. Q (nq,1024), P (N,1024) float32 -> per-query (ids, scores) top-`limit`."""
    Qt = torch.nn.functional.normalize(torch.from_numpy(Q), dim=-1).to(device)
    out_s = torch.full((Q.shape[0], 0), -2.0)
    out_i = torch.zeros((Q.shape[0], 0), dtype=torch.long)
    for s in range(0, P.shape[0], block):
        Pb = torch.nn.functional.normalize(torch.from_numpy(P[s:s + block]), dim=-1).to(device)
        sc = (Qt @ Pb.T).cpu()
        idx = torch.arange(s, s + sc.shape[1]).expand_as(sc)
        out_s = torch.cat([out_s, sc], 1)
        out_i = torch.cat([out_i, idx], 1)
        if out_s.shape[1] > limit * 2:
            top = out_s.topk(limit, dim=1)
            out_s, out_i = top.values, out_i.gather(1, top.indices)
    res = []
    for r in range(Q.shape[0]):
        ids = [int(cids[j]) for j in out_i[r].tolist()]
        res.append(rank_order(out_s[r].tolist(), ids))
    return [(i[:limit], s[:limit]) for i, s in res]


def ranks_equivalent(ref_ids, ref_scores, got_ids, atol=1e-5):
    """True if `got` is a valid ordering of the reference up to explicit score ties (|diff|<=atol).
    Returns (ok, n_positions_differing_but_tied, n_real_mismatches)."""
    ref_score_of = dict(zip(ref_ids, ref_scores))
    tied, bad = 0, 0
    for pos, g in enumerate(got_ids):
        if g == ref_ids[pos]:
            continue
        if g in ref_score_of and abs(ref_score_of[g] - ref_scores[pos]) <= atol:
            tied += 1
        else:
            bad += 1
    return bad == 0, tied, bad


@torch.inference_mode()
def full_late_scores(q_arrs, page_iter, n_pages, device="cpu", doc_block=2048, progress=None):
    """Exact MeanMaxSim of every query vs every page, streaming pages (one at a time) and blocking over
    document tokens; never allocates a query x page x token tensor. Yields (page_key, scores[nq])."""
    lens = torch.tensor([a.shape[0] for a in q_arrs])
    Q = torch.from_numpy(np.concatenate(q_arrs)).to(device)              # (sum_q, 128)
    seg = torch.repeat_interleave(torch.arange(len(q_arrs)), lens).to(device)
    bar = progress(total=n_pages, desc="full late") if progress else None
    for key, D in page_iter:
        best = torch.full((Q.shape[0],), -float("inf"), device=device)
        Dt = torch.from_numpy(D)
        for s in range(0, Dt.shape[0], doc_block):
            blk = Dt[s:s + doc_block].to(device)
            best = torch.maximum(best, (Q @ blk.T).max(dim=1).values)    # exact max over doc-token blocks
        tot = torch.zeros(len(q_arrs), device=device).index_add_(0, seg, best)
        yield key, (tot / lens.to(device)).cpu().numpy()                  # masked query-token mean
        if bar:
            bar.update(1)
    if bar:
        bar.close()
