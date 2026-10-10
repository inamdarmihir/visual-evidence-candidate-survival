"""Run only against a finished run directory (real cached embeddings). Skipped otherwise, never faked."""
import numpy as np

from vesk.cache import ShardCache
from vesk.scoring import reference_mean_maxsim


def _caches(run_dir):
    import json
    ident = json.loads((run_dir / "emb" / "pages" / "CACHE_IDENTITY.json").read_text())
    return ShardCache(run_dir / "emb", "pages", ident), ShardCache(run_dir / "emb", "queries", ident)


def test_checksums_and_masks(run_dir):
    pc, qc = _caches(run_dir)
    assert pc.verify_all() > 0 and qc.verify_all() > 0
    k = pc.done_keys()[0]
    z = pc.read(k)
    assert z["tokens"].shape[0] == len(z["valid_index"]) == pc.meta(k)["n_tokens"]
    assert np.isfinite(z["tokens"]).all() and abs(np.linalg.norm(z["dense"]) - 1) < 1e-3


def test_padded_masked_reference_equals_unpadded_direct(run_dir):
    pc, qc = _caches(run_dir)
    ds = sorted(pc.done_keys(), key=lambda c: pc.meta(c)["n_tokens"])
    qs = sorted(qc.done_keys(), key=lambda c: qc.meta(c)["n_tokens"])
    docs = [pc.read(c)["tokens"] for c in (ds[0], ds[len(ds) // 2], ds[-1])]
    qrs = [qc.read(c)["tokens"] for c in (qs[0], qs[-1])]
    ref = reference_mean_maxsim(qrs, docs)
    direct = np.array([[(q @ d.T).max(1).mean() for d in docs] for q in qrs])
    assert np.abs(ref - direct).max() < 1e-5
