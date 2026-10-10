"""Cross-domain aggregation of finished runs: python -m vesk.aggregate OUT_DIR RUN_DIR_OR_ZIP [...]

Reads only the per-query tables the runs already wrote (per_query_metrics.csv, run_manifest.json).
No new model or retrieval computation. Queries are resampled with replacement WITHIN each domain
(stratified bootstrap), so intervals respect the fixed domain sizes.
"""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .util import atomic_write_json

N_BOOT, SEED = 2000, 42
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def _resolve(p) -> Path:
    p = Path(p)
    if p.suffix == ".zip":
        d = Path(tempfile.mkdtemp(prefix="vesk_agg_"))
        zipfile.ZipFile(p).extractall(d)
        hits = list(d.rglob("per_query_metrics.csv"))
        if len(hits) != 1:
            raise FileNotFoundError(f"{p}: expected one per_query_metrics.csv, found {len(hits)}")
        return hits[0].parent
    return p


def load_run(p) -> pd.DataFrame:
    d = _resolve(p)
    man = json.loads((d / "run_manifest.json").read_text())
    ident = man["identity"]
    if ident["mode"] != "full":
        raise ValueError(f"{d}: only full runs can be aggregated (got {ident['mode']})")
    q = pd.read_csv(d / "per_query_metrics.csv")
    q["domain"] = ident["dataset"].split("vidore_v3_")[-1]
    q["preset"] = ident["preset"]
    q["dataset_revision"] = ident["dataset_revision"]
    q["run_id"] = man["run_id"]
    return q


def _arms(q: pd.DataFrame, k: int, metric: str) -> pd.DataFrame:
    """One row per (domain, query_id); columns = arms at top-k. Ceiling uses n_pos."""
    base = q[(q.stage == "dense") & (q.k == k)][["domain", "query_id", "n_pos", metric]].rename(columns={metric: "dense"})
    out = base.copy()
    fl = q[(q.stage == "full_late") & (q.k == k)][["domain", "query_id", metric]].rename(columns={metric: "full_late"})
    out = out.merge(fl, on=["domain", "query_id"], how="left")
    for B in sorted(q[q.stage == "rerank"].B.dropna().unique()):
        r = q[(q.stage == "rerank") & (q.B == B) & (q.k == k)][["domain", "query_id", metric]]
        out = out.merge(r.rename(columns={metric: f"rerank_B{int(B)}"}), on=["domain", "query_id"], how="left")
        c = q[(q.stage == "candidate") & (q.B == B)][["domain", "query_id", metric]]
        out = out.merge(c.rename(columns={metric: f"cand_B{int(B)}"}), on=["domain", "query_id"], how="left")
    out["ceiling"] = np.minimum(out.n_pos, k) / out.n_pos if metric == "recall" else np.nan
    if out.drop(columns="full_late").isna().any().any():
        raise ValueError("per-query arms are incomplete; refusing to aggregate")
    return out


def _boot(wide: pd.DataFrame, cols, fn, n_boot=N_BOOT, seed=SEED):
    """Stratified bootstrap: resample query rows within each domain; fn maps resampled frame -> value."""
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in wide.groupby("domain")]
    vals = []
    for _ in range(n_boot):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        vals.append(fn(pd.concat(parts)))
    v = np.asarray(vals, dtype=float)
    return np.percentile(v, [2.5, 97.5], axis=0)


def summarize(runs: list[pd.DataFrame], k: int = 10) -> dict:
    q = pd.concat(runs, ignore_index=True)
    w = _arms(q, k, "recall")
    cols = [c for c in w.columns if c not in ("domain", "query_id", "n_pos")]
    rows, paired = [], []
    for dom, g in list(w.groupby("domain")) + [("POOLED (all queries)", w)]:
        for c in cols:
            if g[c].isna().all():
                continue
            lo, hi = _boot(g, [c], lambda f, c=c: f[c].mean())
            rows.append({"domain": dom, "arm": c, "k": k, "n_queries": len(g), "macro_recall": g[c].mean(),
                         "ci_lo": lo, "ci_hi": hi})
        for B in sorted(int(c.split("B")[1]) for c in cols if c.startswith("rerank_B")):
            d = g[f"rerank_B{B}"] - g["dense"]
            lo, hi = _boot(g, None, lambda f, B=B: (f[f"rerank_B{B}"] - f["dense"]).mean())
            paired.append({"domain": dom, "comparison": f"rerank(B={B}) - dense", "k": k, "n_queries": len(g),
                           "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                           "n_rerank_better": int((d > 1e-12).sum()), "n_dense_better": int((d < -1e-12).sum()),
                           "n_equal": int((d.abs() <= 1e-12).sum())})
    # domain-equal macro (each domain weighted equally) as a robustness view
    dom_means = w.groupby("domain")[cols].mean()
    equal = dom_means.mean().rename("domain_equal_macro_recall").reset_index().rename(columns={"index": "arm"})
    ceil = w.groupby("domain").ceiling.mean().rename("recall_ceiling_at_k").reset_index()
    return {"per_query": w, "table": pd.DataFrame(rows), "paired": pd.DataFrame(paired),
            "domain_equal": equal, "ceiling": ceil}


def plot(res: dict, out: Path, k: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = res["table"]
    doms = [d for d in t.domain.unique() if not d.startswith("POOLED")] + ["POOLED (all queries)"]
    Bs = sorted(int(a.split("B")[1]) for a in t.arm.unique() if a.startswith("rerank_B"))
    fig, axs = plt.subplots(1, len(doms), figsize=(3.1 * len(doms), 3.4), sharey=True)
    for ax, d in zip(np.atleast_1d(axs), doms):
        s = t[t.domain == d].set_index("arm")
        m = [s.loc[f"rerank_B{b}"] for b in Bs]
        ax.plot(Bs, [r.macro_recall for r in m], "-o", color=C1, lw=2, ms=6, label="dense top-B → late rerank")
        ax.fill_between(Bs, [r.ci_lo for r in m], [r.ci_hi for r in m], color=C1, alpha=.15, lw=0)
        ax.axhline(s.loc["dense", "macro_recall"], color=C2, ls="--", lw=2, label="dense order (no rerank)")
        if "full_late" in s.index:
            ax.axhline(s.loc["full_late", "macro_recall"], color=C3, ls=":", lw=2.4, label="full-corpus late reference")
        ax.set_xscale("log"); ax.set_xticks(Bs); ax.set_xticklabels(Bs)
        ax.set_title(f"{d.split(' ')[0]}\n(n={int(s.n_queries.iloc[0])})", fontsize=10)
        ax.set_xlabel("candidate budget B"); ax.set_ylim(0, 1); ax.grid(alpha=.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel(f"recall@{k} of annotated pages")
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle(f"Final top-{k} evidence recall by domain (95% bootstrap CI over queries)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "cross_domain_recall.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) < 2:
        raise SystemExit(__doc__)
    out = Path(a[0]); out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in a[1:]]
    ids = {r.domain.iloc[0]: (r.run_id.iloc[0], r.preset.iloc[0], r.dataset_revision.iloc[0]) for r in runs}
    if len({v[1] for v in ids.values()}) != 1:
        raise SystemExit(f"runs mix presets: {ids}")
    if len(ids) != len(runs):
        raise SystemExit("duplicate domains among the supplied runs")
    for k in (5, 10):
        res = summarize(runs, k)
        res["table"].to_csv(out / f"cross_domain_recall_at{k}.csv", index=False)
        res["paired"].to_csv(out / f"cross_domain_paired_at{k}.csv", index=False)
        res["domain_equal"].to_csv(out / f"domain_equal_macro_at{k}.csv", index=False)
        res["ceiling"].to_csv(out / f"recall_ceiling_at{k}.csv", index=False)
        if k == 10:
            plot(res, out, k)
    atomic_write_json(out / "aggregated_runs.json", {"runs": ids, "n_boot": N_BOOT, "seed": SEED,
                                                      "bootstrap": "queries resampled with replacement within each domain"})
    print(f"wrote cross-domain tables/plot for {sorted(ids)} -> {out}")


if __name__ == "__main__":
    main()
