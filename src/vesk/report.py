"""Tables, plots (rendered only from saved measurements), trace viewer, export zip."""
from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS, paired_diff

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (CVD/normal-vision pass, all pairs)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "regular", "legend.frameon": False})
    return plt


# ------------------------------------------------------------------ strata -----------------------------
def strata_table(pq: pd.DataFrame, queries: pd.DataFrame, grades_by_q: dict, focus_B=25, ks=(5, 10)) -> pd.DataFrame:
    """Multi-label strata: a query appears in every stratum it carries. n shown for each; n<10 is descriptive only."""
    qinfo = {}
    for r in queries.itertuples(index=False):
        qid = int(r.query_id)
        qinfo[qid] = {"annotated_pages": "single (1 page)" if len(grades_by_q[qid]) == 1 else "multiple (>1 pages)",
                      "query_type": [str(x) for x in r.query_types], "content_type": [str(x) for x in r.content_type]}
    members = {}
    for qid, i in qinfo.items():
        members.setdefault(("annotated_pages", i["annotated_pages"]), set()).add(qid)
        for t in i["query_type"]:
            members.setdefault(("query_type", t), set()).add(qid)
        for t in i["content_type"]:
            members.setdefault(("content_type", t), set()).add(qid)
    members[("all", "all")] = set(qinfo)
    rows = []
    for (kind, name), qids in sorted(members.items()):
        sub = pq[pq.query_id.isin(qids)]
        for stage, B, k in [("candidate", B, np.nan) for B in sorted(pq[pq.stage == "candidate"].B.unique())] + \
                [(s, focus_B if s == "rerank" else np.nan, k) for s in ("dense", "rerank", "full_late") for k in ks]:
            d = sub[(sub.stage == stage) & ((sub.B == B) | (pd.isna(B) & sub.B.isna())) & ((sub.k == k) | (pd.isna(k) & sub.k.isna()))]
            if d.empty:
                continue
            rows.append({"stratum_kind": kind, "stratum": name, "n_queries": len(qids),
                         "descriptive_only_n_lt_10": len(qids) < 10, "stage": stage, "B": B, "k": k,
                         "n_pages": int(d.n_pos.sum()), "macro_recall": d.recall.mean(), "macro_coverage": d.coverage.mean(),
                         "macro_recall_s1": d.recall_s1.mean(), "n_queries_s1": int((d.n_s1 > 0).sum()),
                         "macro_ndcg": d.ndcg.mean()})
    return pd.DataFrame(rows)


def paired_table(pq, budgets, ks, n_boot=2000, seed=42, has_full=False):
    rows = []
    for B in budgets:
        for k in ks:
            for m in ("recall", "coverage", "ndcg"):
                rows.append({"comparison": "rerank(B) - dense", "B": B, "k": k,
                             **paired_diff(pq, {"stage": "rerank", "B": B, "k": k}, {"stage": "dense", "k": k}, m, n_boot, seed)})
                if has_full:
                    rows.append({"comparison": "rerank(B) - full_late", "B": B, "k": k,
                                 **paired_diff(pq, {"stage": "rerank", "B": B, "k": k}, {"stage": "full_late", "k": k}, m, n_boot, seed)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots ------------------------------
def _band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)


def plot_candidate(agg, out):
    plt = _mpl()
    c = agg[agg.stage == "candidate"].sort_values("B")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (m, title) in zip(axs, [("recall", "Annotated-page candidate recall"), ("coverage", "All-annotated-page coverage")]):
        for key, col, mk, lab in [(m, C1, "o", "all annotated pages"), (f"{m}_s1", C2, "s", "critical (score-1) pages")]:
            if f"macro_{key}" not in c:
                continue
            ax.plot(c.B, c[f"macro_{key}"], color=col, marker=mk, ms=6, lw=2, label=lab)
            _band(ax, c.B, c[f"{key}_ci_lo"], c[f"{key}_ci_hi"], col)
        ax.set_xscale("log"); ax.set_xticks(c.B); ax.set_xticklabels(c.B.astype(int)); ax.minorticks_off()
        ax.set_xlabel("dense candidate budget B"); ax.set_title(title); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("macro mean over queries (95% bootstrap CI)")
    axs[1].legend(loc="lower right")
    n = int(c.n_queries.max())
    fig.suptitle(f"Candidate stage: dense exact top-B  (n={n} human English queries)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_final(agg, out, ks, has_full):
    plt = _mpl()
    fig, axs = plt.subplots(2, len(ks), figsize=(4.6 * len(ks), 6.2), sharex=True)
    axs = np.atleast_2d(axs)
    for j, k in enumerate(ks):
        for i, (m, ylab) in enumerate([("recall", "recall of annotated pages"), ("coverage", "all-annotated-page coverage")]):
            ax = axs[i, j]
            rr = agg[(agg.stage == "rerank") & (agg.k == k)].sort_values("B")
            ax.plot(rr.B, rr[f"macro_{m}"], color=C1, marker="o", ms=6, lw=2, label="after late rerank of top-B")
            _band(ax, rr.B, rr[f"{m}_ci_lo"], rr[f"{m}_ci_hi"], C1)
            dn = agg[(agg.stage == "dense") & (agg.k == k)]
            if len(dn):
                v = float(dn[f"macro_{m}"].iloc[0])
                ax.axhline(v, color=C2, lw=2, ls="--", label="dense order (no rerank)")
            if has_full:
                fl = agg[(agg.stage == "full_late") & (agg.k == k)]
                if len(fl):
                    ax.axhline(float(fl[f"macro_{m}"].iloc[0]), color=C3, lw=2, ls=":", label="full-corpus late reference")
            ax.set_xscale("log"); ax.set_xticks(rr.B); ax.set_xticklabels(rr.B.astype(int)); ax.minorticks_off()
            ax.set_title(f"top-{k}"); ax.set_ylim(0, 1.02)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("candidate budget B")
    axs[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Final top-k evidence survival: dense vs. reranked", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_strata(strata, out, focus_B, k=10):
    plt = _mpl()
    s = strata[(strata.stratum_kind != "all")]
    cand = s[(s.stage == "candidate") & (s.B == focus_B)].set_index(["stratum_kind", "stratum"])
    dn = s[(s.stage == "dense") & (s.k == k)].set_index(["stratum_kind", "stratum"])
    rr = s[(s.stage == "rerank") & (s.k == k)].set_index(["stratum_kind", "stratum"])
    idx = [i for i in cand.index if i in dn.index and i in rr.index]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(idx) + 1.6))
    y = np.arange(len(idx))[::-1]
    for vals, col, mk, sz, hollow, lab in [(cand, C1, "o", 120, True, f"candidate recall (dense top-{focus_B})"),
                                           (dn, C2, "s", 40, False, f"dense top-{k}"),
                                           (rr, C3, "D", 36, False, f"rerank of top-{focus_B} -> top-{k}")]:
        xs = [vals.loc[i, "macro_recall"] for i in idx]
        if hollow:   # ring drawn behind so coincident markers stay visible
            ax.scatter(xs, y, facecolors="none", edgecolors=col, marker=mk, s=sz, linewidths=2, label=lab, zorder=2)
        else:
            ax.scatter(xs, y, color=col, marker=mk, s=sz, label=lab, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i[0].replace('_', ' ')}: {i[1]}  (n={int(cand.loc[i, 'n_queries'])}{'*' if cand.loc[i, 'descriptive_only_n_lt_10'] else ''})" for i in idx], fontsize=8)
    ax.set_xlim(0, 1.02); ax.set_xlabel("macro recall of annotated pages")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    ax.text(0, -0.9 / len(idx) - 0.07, "* n<10: descriptive only. Query types / content types are multi-label (overlapping strata).",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_loss(page_rows, out, ks):
    plt = _mpl()
    fig, axs = plt.subplots(1, len(ks), figsize=(5 * len(ks), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    order = [("survived_top_k", C1, "survived to top-k"), ("reranking_loss", C2, "in candidates, below top-k (reranking loss)"),
             ("candidate_loss", C3, "never a candidate (candidate loss)")]
    for ax, k in zip(axs, ks):
        d = page_rows[page_rows.k == k]
        Bs = sorted(d.B.unique())
        left = np.zeros(len(Bs))
        for lab, col, text in order:
            frac = np.array([(d[d.B == B].label == lab).mean() for B in Bs])
            ax.barh(range(len(Bs)), frac, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62, label=text)
            for i, (f, l) in enumerate(zip(frac, left)):
                if f > 0.07:
                    ax.text(l + f / 2, i, f"{f:.0%}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            left += frac
        ax.set_yticks(range(len(Bs))); ax.set_yticklabels([f"B={int(b)}" for b in Bs]); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_title(f"top-{k}  (n={d.groupby('B').size().iloc[0]} annotated query-page judgments)"); ax.grid(axis="y", visible=False)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.14), ncol=1, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_storage_timing(pages: pd.DataFrame, out, dev_label):
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    axs[0].hist(pages.n_tokens, bins=30, color=C1, edgecolor=SURFACE)
    axs[0].set_xlabel("valid vectors per page (all kept; padding trimmed only)"); axs[0].set_ylabel("pages")
    axs[0].set_title(f"Token counts, total {pages.n_tokens.sum():,} ({pages.n_tokens.sum() * 128 * 4 / 1e9:.2f} GB fp32 multivector)")
    if pages.n_tokens.nunique() == 1:
        axs[0].set_title(f"All {len(pages)} pages have {int(pages.n_tokens.iloc[0])} vectors (uniform page size); "
                         f"total {pages.n_tokens.sum():,} = {pages.n_tokens.sum() * 128 * 4 / 1e9:.2f} GB fp32", fontsize=8)
        axs[1].hist(pages.seconds, bins=30, color=C2, edgecolor=SURFACE)
        axs[1].set_xlabel("seconds per page"); axs[1].set_ylabel("pages")
    else:
        axs[1].scatter(pages.n_tokens, pages.seconds, s=10, color=C2)
        axs[1].set_xlabel("valid vectors per page"); axs[1].set_ylabel("seconds per page")
    axs[1].set_title(f"Measured embed time on {dev_label}")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ viewer -----------------------------
def select_examples(traces, focus_B=25, k=10, n_each=4):
    """Deterministic, stated rules; no manual picking. Ties broken by query_id ascending."""
    def loss(t):  return len(t["per_B"][str(focus_B)]["lost_at_candidate_stage"])
    def rer(t):   return len(t["per_B"][str(focus_B)][f"lost_by_reranking_top{k}"])
    sel, why = {}, {}
    rules = [
        (f"Rule A: most annotated pages lost at the candidate stage (B={focus_B})", sorted(traces, key=lambda t: (-loss(t), t["query_id"])), lambda t: loss(t) > 0),
        (f"Rule B: most annotated pages that survived to the candidates but fell below top-{k} after reranking (B={focus_B})", sorted(traces, key=lambda t: (-rer(t), t["query_id"])), lambda t: rer(t) > 0),
        (f"Rule C: highest coverage -- all annotated pages are candidates at the smallest B, most annotated pages first", sorted(traces, key=lambda t: (-t["n_positive"], t["query_id"])), lambda t: len(t["per_B"][str(min(int(b) for b in t["per_B"]))]["lost_at_candidate_stage"]) == 0),
    ]
    for text, ordered, pred in rules:
        n = 0
        for t in ordered:
            if pred(t) and n < n_each:
                sel.setdefault(t["query_id"], []).append(text)
                n += 1
    by = {t["query_id"]: t for t in traces}
    return [(by[q], why_) for q, why_ in sorted(sel.items())]


def save_thumbnails(bundle, cids, out_dir, width=420):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for c in sorted(set(cids)):
        f = out_dir / f"{c}.jpg"
        if not f.exists():
            im = bundle.image(c).convert("RGB")
            im.resize((width, round(im.height * width / im.width))).save(f, quality=80)


VIEWER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence survival trace viewer</title><style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e1;--ok:#2a78d6;--rl:#eb6834;--cl:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--mut:#c3c2b7;--line:#33322f;--ok:#3987e5;--rl:#d95926;--cl:#e66767}}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:20px}h2{font-size:15px;margin:0}.card{border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}
.mut{color:var(--mut)}.pages{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px}.pg{width:210px}
.wrap{position:relative;width:210px}.wrap img,.wrap canvas{width:210px;display:block}.wrap canvas{position:absolute;left:0;top:0;pointer-events:none}.wrap img:not([src]){display:none}.wrap button{position:relative;z-index:2}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:12px;color:#fff}
.survived_top_k{background:var(--ok)}.reranking_loss{background:var(--rl)}.candidate_loss{background:var(--cl)}
button{font:inherit}code{font-size:12px}select{font:inherit}</style></head><body>
<h1>Evidence survival trace viewer</h1>
<p class="mut" id="hdr"></p>
<p class="mut"><b>Page numbers:</b> <code>page_number_in_doc</code> is stored 0-based; the label shows <i>book p.N</i> = stored + 1.
<b>Boxes:</b> human annotation boxes are in <i>original page-image pixels</i> (W&times;H shown per page); they are drawn scaled by
thumbnail width / W. The retriever saw the page resized (longest side capped, aspect kept, no crop) and padded to a 32 px patch grid; boxes are never transformed for the model.
Unjudged pages are not known negatives. Examples are chosen by the stated deterministic rules, not by hand.</p>
<label>Budget B <select id="B"></select></label> <label>k <select id="K"></select></label>
<div id="root"></div>
<script>
const DATA = __DATA__;
const hdr = document.getElementById('hdr');
hdr.textContent = DATA.title;
const Bsel = document.getElementById('B'), Ksel = document.getElementById('K');
DATA.budgets.forEach(b=>Bsel.add(new Option(b,b))); DATA.ks.forEach(k=>Ksel.add(new Option(k,k)));
Bsel.value = DATA.focus_B; Ksel.value = DATA.focus_k;
function label(p,B,k,q){const pb=q.per_B[B]; if(!pb.candidates.includes(p.corpus_id)) return 'candidate_loss';
 const r=pb.rerank_ids.indexOf(p.corpus_id)+1; return r<=k?'survived_top_k':'reranking_loss';}
function draw(cv,img,p){const s=img.clientWidth/p.orig_w; cv.width=img.clientWidth; cv.height=img.clientHeight;
 const g=cv.getContext('2d'); g.lineWidth=2; g.strokeStyle='#ff2d55';
 (p.bounding_boxes||[]).forEach(b=>g.strokeRect(b.x1*s,b.y1*s,(b.x2-b.x1)*s,(b.y2-b.y1)*s));}
async function fetchOnDemand(p,img,cv,btn){btn.disabled=true; btn.textContent='fetching...';
 try{const u=`https://datasets-server.huggingface.co/rows?dataset=${DATA.dataset}&config=corpus&split=test&offset=${p.row_offset}&length=1`;
 const j=await (await fetch(u)).json(); const row=j.rows[0].row;
 if(row.corpus_id!==p.corpus_id) throw new Error('corpus_id mismatch (server serves a different revision)');
 img.src=row.image.src; img.onload=()=>draw(cv,img,p); btn.remove();}catch(e){btn.textContent='fetch failed: '+e.message;}}
function render(){const B=Bsel.value,k=+Ksel.value,root=document.getElementById('root'); root.innerHTML='';
 DATA.examples.forEach(q=>{const c=document.createElement('div'); c.className='card';
  const pb=q.per_B[B]; const nloss=q.positives.filter(p=>label(p,B,k,q)==='candidate_loss').length;
  c.innerHTML=`<h2>query ${q.query_id}: ${q.query.replace(/</g,'&lt;')}</h2><div class="mut">types: ${q.query_types.join(', ')} &middot; annotated pages: ${q.n_positive} &middot; lost at candidate stage (B=${B}): ${nloss}</div><div class="mut">selected by: ${q.why.join(' | ')}</div>`;
  const row=document.createElement('div'); row.className='pages';
  q.positives.forEach(p=>{const L=label(p,B,k,q); const rr=pb.rerank_ids.indexOf(p.corpus_id)+1; const dr=q.dense_top100.indexOf(p.corpus_id)+1;
   const d=document.createElement('div'); d.className='pg';
   d.innerHTML=`<span class="tag ${L}">${L.replace(/_/g,' ')}</span> grade ${p.grade}<div class="mut">${p.doc_id.replace(/_/g,' ')}, book p.${p.page_number_1based} (stored ${p.page_number_in_doc_0based})<br>corpus_id ${p.corpus_id} &middot; ${p.orig_w}&times;${p.orig_h}px<br>dense rank ${dr||'>100'} &middot; rerank rank ${rr||'not a candidate'}</div>`;
   const w=document.createElement('div'); w.className='wrap'; const img=new Image(); const cv=document.createElement('canvas');
   img.src=DATA.thumbs+p.corpus_id+'.jpg'; img.onload=()=>draw(cv,img,p);
   img.onerror=()=>{img.onerror=null; img.removeAttribute('src'); const b=document.createElement('button'); b.textContent='fetch page image on demand'; b.onclick=()=>fetchOnDemand(p,img,cv,b); w.appendChild(b);};
   w.appendChild(img); w.appendChild(cv); d.appendChild(w); row.appendChild(d);});
  c.appendChild(row); root.appendChild(c);});}
Bsel.onchange=Ksel.onchange=render; render();
</script></body></html>"""


def build_viewer(out_html, traces, bundle, geometry, cfg, thumbs_dir, focus_B=25, focus_k=10, dataset=None):
    from .config import DATASET_ID
    dataset = dataset or DATASET_ID
    ex = select_examples(traces, focus_B, focus_k)
    cids = []
    examples = []
    for t, why in ex:
        t = json.loads(json.dumps(t))
        t["why"] = why
        for p in t["positives"]:
            g = geometry[p["corpus_id"]]
            p["orig_w"], p["orig_h"] = g["orig_w"], g["orig_h"]
            cids.append(p["corpus_id"])
        examples.append({k: t[k] for k in ("query_id", "query", "query_types", "n_positive", "positives", "dense_top100", "per_B", "why")})
    if bundle is not None:
        save_thumbnails(bundle, cids, thumbs_dir)
    data = {"title": f"run {cfg.run_id()} - {cfg.mode} mode - {len(examples)} deterministically selected queries",
            "budgets": list(cfg.budgets), "ks": list(cfg.ks), "focus_B": focus_B, "focus_k": focus_k,
            "thumbs": "thumbs/", "dataset": dataset, "examples": examples}
    Path(out_html).write_text(VIEWER.replace("__DATA__", json.dumps(data)))
    return len(examples)


# ------------------------------------------------------------------ export -----------------------------
def export_zip(root: Path, run_dir: Path, out_zip: Path):
    """Code, manifest, tables, plots, trace metadata, attribution. NOT corpus images, embeddings or weights."""
    root, run_dir = Path(root), Path(run_dir)
    include = ["src", "tools", "tests", "docs", "README.md", "PROTOCOL.md", "ATTRIBUTION.md", "SOURCES.md", "LIMITATIONS.md",
               "LICENSE", "pyproject.toml", "requirements.txt", "constraints.txt", "visual_evidence_candidate_survival_t4.ipynb"]
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in include:
            p = root / name
            if p.is_file():
                z.write(p, name)
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc",):
                        z.write(f, f.relative_to(root))
        for f in sorted(run_dir.iterdir()):
            if f.is_file() and f.suffix in (".csv", ".json", ".jsonl", ".md", ".txt", ".html", ".png"):
                z.write(f, f"run/{f.name}")
        for sub in ("plots", "traces"):
            if (run_dir / sub).is_dir():
                for f in sorted((run_dir / sub).rglob("*")):
                    if f.is_file():
                        z.write(f, f"run/{sub}/{f.relative_to(run_dir / sub)}")
    return out_zip


# ------------------------------------------------------------------ findings ---------------------------
def findings_markdown(agg, pages, paired, cfg, has_full, mode):
    """Plain-language readout generated ONLY from the saved measurements (no hand-written conclusions)."""
    k = max(cfg.ks)
    L = []
    if mode == "smoke":
        L.append("> SMOKE fixture (qrel-aware, 8 queries): engineering check only; do not interpret or compare.\n")
    n = int(agg[agg.stage == "candidate"].n_queries.max())
    L.append(f"## Readout (n = {n} human-origin English queries; macro means; 95% bootstrap CI over whole queries)\n")
    L.append(f"Annotated pages = pages with positive human qrels. Final cutoff shown: top-{k}. Unjudged pages are not negatives; coverage is not answer completeness.\n")
    L.append("| B | candidate recall | candidate coverage | pages lost before rerank | pages lost by rerank (in candidates, below top-%d) | pages surviving to top-%d |" % (k, k))
    L.append("|---|---|---|---|---|---|")
    for B in cfg.budgets:
        c = agg[(agg.stage == "candidate") & (agg.B == B)].iloc[0]
        d = pages[(pages.B == B) & (pages.k == k)]
        f = lambda lab: (d.label == lab).mean()
        L.append(f"| {B} | {c.macro_recall:.3f} [{c.recall_ci_lo:.3f}, {c.recall_ci_hi:.3f}] | {c.macro_coverage:.3f} [{c.coverage_ci_lo:.3f}, {c.coverage_ci_hi:.3f}] | "
                 f"{f('candidate_loss'):.1%} | {f('reranking_loss'):.1%} | {f('survived_top_k'):.1%} |")
    L.append("")
    dn = agg[(agg.stage == "dense") & (agg.k == k)].iloc[0]
    L.append(f"Dense order alone, top-{k}: recall {dn.macro_recall:.3f} [{dn.recall_ci_lo:.3f}, {dn.recall_ci_hi:.3f}], coverage {dn.macro_coverage:.3f}.\n")
    L.append(f"### Does reranking change top-{k} recall vs. the dense order? (paired, same queries)\n")
    L.append("| B | mean diff in recall | 95% CI | queries better / worse / equal | reading |")
    L.append("|---|---|---|---|---|")
    p = paired[(paired.comparison == "rerank(B) - dense") & (paired.k == k) & (paired.metric == "recall")]
    for x in p.itertuples():
        reading = ("identical in every query (reranking only reorders the same B pages)" if x.n_equal == x.n_queries else
                   "no clear difference (CI includes 0)" if x.ci_lo <= 0 <= x.ci_hi else
                   "reranking higher" if x.mean_diff > 0 else "reranking lower")
        L.append(f"| {x.B} | {x.mean_diff:+.3f} | [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}] | {x.n_a_better} / {x.n_b_better} / {x.n_equal} | {reading} |")
    if has_full:
        fl = agg[(agg.stage == "full_late") & (agg.k == k)].iloc[0]
        L.append(f"\nUnrestricted full-corpus late-interaction reference, top-{k}: recall {fl.macro_recall:.3f} [{fl.recall_ci_lo:.3f}, {fl.recall_ci_hi:.3f}], coverage {fl.macro_coverage:.3f}. "
                 "This is a reference ranking, not a gold oracle.")
    else:
        L.append("\nFull-corpus late-interaction reference: not computed (metrics depending on it are not computed, not zero).")
    L.append("\nScope: one domain (two OpenStax books), one model, 1024 px preset unless noted, 110 independent queries. A flat result is a valid result; no general claim about reranking is supported.")
    return "\n".join(L) + "\n"
