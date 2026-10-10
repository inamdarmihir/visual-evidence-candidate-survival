"""Builds visual_evidence_candidate_survival_t4.ipynb (self-extracting: embeds src/vesk, docs, tests)."""
import json
import sys
import uuid
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
EMBED = {}
for p in sorted((ROOT / "src" / "vesk").glob("*.py")):
    EMBED[f"src/vesk/{p.name}"] = p.read_text()
for p in sorted((ROOT / "tests").glob("*.py")):
    EMBED[f"tests/{p.name}"] = p.read_text()
for p in sorted((ROOT / "docs").glob("*.md")):
    name = p.name
    EMBED[name if name in ("PROTOCOL.md", "ATTRIBUTION.md", "SOURCES.md", "LIMITATIONS.md") else f"docs/{name}"] = p.read_text()
for n in ("README.md", "LICENSE", "requirements.txt", "pyproject.toml"):
    EMBED[n] = (ROOT / n).read_text()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Visual Evidence Candidate-Survival Kit (Colab T4)

**What this measures.** When a fast dense retriever keeps only **B = 10 / 25 / 50 / 100** candidates for a stronger late-interaction reranker, how many *human-annotated relevant pages* survive, and where are they lost: at the candidate stage or inside the reranker? Data: ViDoRe v3 *computer_science* (pinned revision), **110 human-origin English queries**. Model: `Hcompany/NeoMME-260M-Retriever` (pinned), **one forward pass** gives dense and multivector embeddings. Store/retrieve: a pinned local **Qdrant v1.19.2** server.

**What it is not.** A diagnostic, not a leaderboard reproduction (pages are resized to a **1024 px** longest side, the model default is 2048) and not a claim that reranking helps. It measures *label survival and ranking*, not answer correctness. Score 1 = critically relevant (needs additional info), score 2 = page can contain the complete answer; "all annotated pages covered" is **not** answer completeness. Unjudged pages are not established negatives.

**How to run.** Runtime → *Change runtime type* → **T4 GPU**, then *Run all*. The defaults run `MODE="full"` (after a quick isolated smoke check, `RUN_SMOKE_FIRST`). Smoke = 8 queries / 256 real pages, a qrel-aware *engineering* fixture, never comparable to full results; set `MODE="smoke"` to run only that. Set `FULL_LATE_REFERENCE=False` to skip the optional unrestricted late-interaction reference.

**If a stage fails.** Fix the cause and re-run the same cell; completed pages/queries are skipped. Out of VRAM: lower `DOC_BATCH`/`QUERY_BATCH` (do not change dtype or resolution mid-run; a different `PRESET` is a new run). Pins/torch problem: restart the session and run from the top. "Refusing stale cache": you changed the preset/config; set a new `RUN_TAG`.

**Restart / resume.** The install cell may require **Runtime → Restart session** (it says so explicitly); after a restart run again from the top, the install cell becomes a no-op. Embeddings, Qdrant upserts and per-query results are checkpointed per item with checksums; after a disconnect just re-run the cells (set `USE_DRIVE=True` to keep checkpoints across sessions). Nothing is published; no paid service or secret is needed.
""")

code('''
# ---- 0. Configuration (edit here) -------------------------------------------------------------
MODE = "full"               # "full" = scientific target (1,360 pages, 110 queries);  "smoke" = engineering fixture (8 queries / 256 pages)
DATASET = "computer_science"  # pinned ViDoRe v3 domain for the main run: computer_science | hr | physics | energy | pharmaceuticals
PRESET = "primary_1024"     # "primary_1024" (longest side 1024 px, T4-friendly) | "optional_2048" (model default cap; separate cache)
USE_DRIVE = False           # True: mount Drive for DURABLE cache/results only (the live Qdrant DB stays on /content)
FULL_LATE_REFERENCE = True  # optional (set False to skip) unrestricted full-corpus late-interaction reference (time reported separately)
RUN_SMOKE_FIRST = True      # when MODE=="full": first run the smoke fixture end-to-end (cheap, isolated directory)
RUN_TAG = ""                # change to start a NEW run instead of reusing/refusing an existing cache
DOC_BATCH, QUERY_BATCH = 1, 4   # start at 1 / 4; if VRAM is short lower these (never switch dtype/resolution silently)
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # quieter logs; our own tqdm bars stay
os.environ["VESK_DATASET"] = DATASET               # read by vesk.config at import time (pinned revision per domain)
''')

md("## 1. Install (inspects the existing GPU torch first; never replaces it)")
code('''
import importlib.metadata as md, subprocess, sys, os

PINS = {"transformers": "5.19.0", "sentence-transformers": "6.1.0", "qdrant-client": "1.19.1", "datasets": "5.1.0"}
ver = lambda n: (md.version(n) if n in {d.metadata["Name"] for d in md.distributions()} else None)

torch_v, tv_v = ver("torch"), ver("torchvision")
print("existing torch:", torch_v, "| torchvision:", tv_v)
assert torch_v and tv_v, "A GPU-enabled torch + torchvision must already be installed (Colab ships them). NeoMME's image processor needs torchvision."
assert "+cpu" not in torch_v, "CPU-only torch detected; use a GPU runtime (Runtime > Change runtime type > T4 GPU)."
import torch
print("CUDA runtime:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("GPU:", torch.cuda.get_device_name(0), "| compute capability", cc, "| bf16 supported:", torch.cuda.is_bf16_supported())
    # T4 = sm_75: the Colab torch wheel supports it; no wheel change is made or needed.
    assert f"sm_{cc[0]}{cc[1]}" in torch.cuda.get_arch_list() or any(a.startswith("compute_") for a in torch.cuda.get_arch_list()), "torch wheel lacks this GPU architecture"

need = [f"{k}=={v}" for k, v in PINS.items() if ver(k) != v]
already_imported = [m for m in ("transformers", "sentence_transformers", "qdrant_client", "datasets") if m in sys.modules]
if need:
    # Constraints freeze the existing torch/torchvision so resolving the pins cannot swap the CUDA wheel.
    open("/content/constraints_gpu_torch.txt", "w").write(f"torch=={torch_v}\\ntorchvision=={tv_v}\\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need, "psutil", "matplotlib", "tqdm",
                           "-c", "/content/constraints_gpu_torch.txt"])
    print("installed:", need)
else:
    print("pins already satisfied")
assert ver("torch") == torch_v and ver("torchvision") == tv_v, "torch/torchvision changed during install: restart the runtime and inspect"
bad = {k: ver(k) for k, v in PINS.items() if ver(k) != v}
assert not bad, f"pins not satisfied: {bad}"
if need and already_imported:
    print("\\n>>> RESTART REQUIRED: packages were imported before being re-pinned. Runtime > Restart session, then run from the top. <<<")
else:
    print("environment OK (no restart needed)")
''')

md("## 2. Write the `vesk` package (shared code, embedded in this notebook)")
code("EMBEDDED = " + json.dumps(EMBED) + '''

from pathlib import Path
import sys
PKG = Path("/content/vesk_pkg") if Path("/content").exists() else Path("./vesk_pkg")
for rel, text in EMBEDDED.items():
    f = PKG / rel; f.parent.mkdir(parents=True, exist_ok=True); f.write_text(text)
sys.path.insert(0, str(PKG / "src"))
import vesk; print("vesk", vesk.__version__, "->", PKG)
''')

md("## 3. Hardware / disk / RAM preflight")
code('''
import torch, shutil, psutil, subprocess
from vesk.env import capture_environment, free_disk_gb, host_ram_gb, check_pins
IN_COLAB = Path("/content").exists()
WORK_ROOT = Path("/content/vesk_work") if IN_COLAB else Path("./vesk_work"); WORK_ROOT.mkdir(parents=True, exist_ok=True)
env = capture_environment()
g = env["gpu"]
print({k: g.get(k) for k in ("gpu_name", "gpu_total_gb", "compute_capability", "bf16_supported", "driver", "torch", "torch_cuda", "torchvision")})
print("host RAM (GB):", host_ram_gb(), "| free disk (GB):", round(free_disk_gb(WORK_ROOT), 1))
if not g["cuda_available"]:
    print("WARNING: no GPU. Smoke may run on CPU; FULL on CPU is slow. Use a T4 runtime.")
elif "T4" not in str(g.get("gpu_name")):
    print("NOTE: GPU is not a T4; timings below are for this GPU only, not a T4 estimate.")
need_gb = 10.0 if MODE == "full" else 3.0
assert free_disk_gb(WORK_ROOT) >= need_gb, f"need >= {need_gb} GB free disk for MODE={MODE}"
print("precision: float32, no autocast (BF16 is not native on T4; float16 is never enabled silently).")
''')

md("## 4. Paths, optional Drive (durable checkpoints only), run object")
code('''
from vesk.config import RunConfig
from vesk.pipeline import Run
DURABLE = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DURABLE = "/content/drive/MyDrive/vesk_durable"   # cache + results only; the live Qdrant DB stays on /content
common = dict(preset=PRESET, doc_batch_size=DOC_BATCH, query_batch_size=QUERY_BATCH, full_late_reference=FULL_LATE_REFERENCE, run_tag=RUN_TAG)
cfg = RunConfig(mode=MODE, **common)
run = Run(cfg, WORK_ROOT, DURABLE)
print("run id:", run.run_id, "\\nrun dir:", run.dir)
''')

md("## 5. Data audit (pinned revision; stops with an explanation if invariants fail)")
code('''
import pandas as pd
from IPython.display import display
run.preflight()
b = run.load_data()
print("language x generator (all 1,290 query rows):"); display(b.audit)
a = run.manifest["data_audit"]
print({k: a[k] for k in ("n_eval_queries", "n_positive_qrels", "grade_counts", "n_multi_page_queries", "n_queries_with_score1", "n_corpus_pages_used")})
print("query types (multi-label):", b.queries.query_types.explode().value_counts().to_dict())
print("MODE:", MODE, "| selection saved to", run.dir / "selection.json")
''')

md("""
## 6. Real-subset smoke run (isolated directory)
Runs every stage on 8 deterministically chosen human English queries (chosen without looking at retrieval scores), all of their positive pages, plus real filler pages up to 256. **Engineering fixture only**: it keeps positives, so its scores are easier and must not be compared with the full corpus.
""")
code('''
if MODE == "full" and RUN_SMOKE_FIRST:
    smoke = Run(RunConfig(mode="smoke", **common), WORK_ROOT, DURABLE)
    smoke.run_all()
    print((smoke.dir / "RUN_SUMMARY.md").read_text())
    print("parity passed:", smoke.manifest["parity"]["passed"], "| max |Qdrant - torch| =", smoke.manifest["parity"]["max_abs_qdrant_vs_ref"])
elif MODE == "smoke":
    print("MODE='smoke': the staged cells below ARE the smoke run.")
else:
    print("skipped (RUN_SMOKE_FIRST=False)")
''')

md("## 7. Load model + measured calibration (8-16 real pages on THIS GPU)")
code('''
cal = run.calibrate(n=12)
print({k: cal[k] for k in ("device", "seconds_per_page_mean", "seconds_per_page_std", "mean_tokens", "peak_vram_allocated_gb", "peak_vram_reserved_gb", "peak_rss_gb", "projected_full_corpus_pages", "projected_full_minutes")})
print("assumptions:", cal["assumptions"])
''')

md("## 8. Embedding cache (resumable; one forward pass -> dense + multivector; padding trimmed only)")
code('''
pc, qc = run.embed()
print("pages cached:", len(pc.done_keys()), "| queries cached:", len(qc.done_keys()))
print("peaks:", run.guard.summary())
''')
code('''
# Reload from cache: verify every checksum, then re-embed a few real pages and compare.
print(run.reverify_cache())
''')

md("## 9. Qdrant: pinned local server + collection (named vectors `dense`, `late` MAX_SIM, m=0, cold)")
code('''
info = run.start_qdrant()
print(info); print(run.manifest["qdrant"]["binary"])
n = run.index()
print("points:", n); print(run.manifest["qdrant_collection"]["status"])
''')

md("## 10. Parity tests (real queries vs real pages): torch `mean_maxsim` with masks vs cached vs Qdrant (sum / n_query_tokens)")
code('''
rep = run.parity()
print({k: v for k, v in rep.items() if k != "rows"})
display(pd.DataFrame(rep["rows"]))
''')

md("## 11. Evaluation: exact dense top-100 once -> nested prefixes B -> server prefetch rerank (candidate set verified against the saved dense prefix)")
code('''
results = run.evaluate()
print(len(results), "queries evaluated; methods used:", {v["method"] for r in results.values() for v in r["per_B"].values()})
''')

md("## 12. Analysis (metrics, bootstrap, strata, traces, plots, viewer)")
code('''
agg = run.analyze()
print((run.dir / "RUN_SUMMARY.md").read_text())
for f in ("candidate_metrics.csv", "final_metrics.csv", "paired_differences.csv", "query_strata.csv", "storage_and_runtime.csv"):
    print("\\n==", f); display(pd.read_csv(run.dir / f).head(40))
''')
code('''
from IPython.display import Markdown
print("Generated only from the saved measurements of THIS run (also saved as FINDINGS.md in the export).")
display(Markdown((run.dir / "FINDINGS.md").read_text()))
''')
code('''
from IPython.display import Image
for p in sorted((run.dir / "plots").glob("*.png")):
    print(p.name); display(Image(str(p)))
''')
code('''
# Trace viewer (real page thumbnails + human boxes for deterministically selected queries)
from IPython.display import IFrame, HTML
print("selection rules are printed inside the viewer; file:", run.dir / "trace_viewer.html")
display(HTML("<p>Open <code>trace_viewer.html</code> from the exported zip's run/ folder (thumbs are not exported; the viewer fetches page images on demand by row offset and verifies corpus_id).</p>"))
''')

md("## 13. Export (code, manifest, tables, plots, trace metadata, attribution; **no** images, embeddings or weights)")
code('''
zip_path = run.export()
print(zip_path, round(zip_path.stat().st_size / 1e6, 2), "MB")
if IN_COLAB:
    from google.colab import files
    files.download(str(zip_path))
''')

md("## 14. Cleanup (stops the Qdrant subprocess; checkpoints are kept)")
code('''
run.stop_qdrant(); run.save_manifest(); run.sync()
import gc, torch; run.enc = None; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("qdrant stopped; manifest saved:", run.dir / "run_manifest.json")
''')

md("""
## 15. Optional: cross-domain replication (same protocol, same pins, other ViDoRe v3 domains)
One domain is an anecdote; this section repeats the identical pipeline on further pinned domains, each in its own process and cache, then aggregates **only the per-query tables** the runs wrote. Runtime on a T4 is roughly 10-25 min per domain (measured for computer_science: embed ~5.5 min, Qdrant indexing ~5 min for 1,360 pages). Skip it by leaving `REPLICATE_DOMAINS = []`.
""")
code('''
REPLICATE_DOMAINS = ["hr", "physics", "energy"]      # any of: hr | physics | energy | pharmaceuticals ; [] to skip
import subprocess, sys, os
from pathlib import Path
run.stop_qdrant()                                     # free port 6333 and VRAM before the child runs
zips = {DATASET: str(zip_path)}
for dom in REPLICATE_DOMAINS:
    if dom == DATASET:
        continue
    cmd = [sys.executable, "-m", "vesk.cli", "full", "--root", str(WORK_ROOT), "--preset", PRESET,
           "--doc-batch", str(DOC_BATCH), "--query-batch", str(QUERY_BATCH), "--run-tag", RUN_TAG] + (["--full-late"] if FULL_LATE_REFERENCE else []) \\
          + (["--durable-root", DURABLE] if DURABLE else [])
    env = dict(os.environ, VESK_DATASET=dom, PYTHONPATH=str(PKG / "src"), HF_HUB_DISABLE_PROGRESS_BARS="1")
    print(f"\\n######## {dom} ########", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in proc.stdout:
        print(line, end="")
        if line.strip().endswith(".zip"):
            last = line.strip()
    if proc.wait() != 0 or not last:
        raise RuntimeError(f"{dom}: run failed (see log above); re-run this cell to resume from checkpoints")
    zips[dom] = last
print(zips)
''')
code('''
if len(zips) > 1:
    import shutil, zipfile
    from vesk import aggregate
    AGG = Path(WORK_ROOT) / "cross_domain"
    aggregate.main([str(AGG)] + list(zips.values()))
    from IPython.display import Image, display, Markdown
    display(Image(str(AGG / "cross_domain_recall.png")))
    import pandas as pd
    pd.set_option("display.width", 200)
    display(pd.read_csv(AGG / "cross_domain_paired_at10.csv").round(3))
    display(pd.read_csv(AGG / "recall_ceiling_at10.csv").round(3))
    agg_zip = Path(WORK_ROOT) / "cross_domain_export.zip"
    with zipfile.ZipFile(agg_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in AGG.iterdir():
            z.write(f, f"cross_domain/{f.name}")
        for dom, zp in zips.items():
            z.write(zp, f"runs/{Path(zp).name}")
    print(agg_zip, round(agg_zip.stat().st_size / 1e6, 2), "MB")
    if IN_COLAB:
        from google.colab import files
        files.download(str(agg_zip))
''')

md("""
## Handoff: what is and isn't established
- **Validated:** pinned API inspection; dataset invariants (110 queries / 545 judgments / 451+94); a complete smoke run on a Colab Tesla T4 (user-run) and on CPU with identical metrics; Qdrant-vs-torch parity ~1e-7; the optional full-corpus late reference verified against `mean_maxsim` (6e-8) on cached real vectors.
- **Also validated (user-run on a Colab Tesla T4):** the full computer_science run (parity 1.4e-7, all checksums, ~5.5 min embedding, ~5 min indexing) and the trace viewer in a browser.
- **Not yet verified:** the 2048 px preset, `USE_DRIVE=True` resume, and the cross-domain section (section 15) on a T4. The sibling-domain invariants were checked against the pinned revisions on a CPU, not on a T4.
- **Estimates only (before you run):** ~0.52 MB/page of float32 patch vectors at 1024 px square (planning envelope); actual sizes come from `storage_and_runtime.csv`.
- Expect an install-time **restart prompt** if Colab's preinstalled `transformers`/`datasets` differ from the pins.
- Licenses: annotations CC BY 4.0; page images inherit the OpenStax licenses (see `ATTRIBUTION.md` in the export); model Apache 2.0; code MIT.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
for c in nb.cells:
    c["id"] = uuid.uuid4().hex[:8]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "language_info": {"name": "python"}}
nbf.validate(nb)
out = ROOT / "notebooks" / "visual_evidence_candidate_survival_t4.ipynb"
nbf.write(nb, out)
print("wrote", out, f"{out.stat().st_size/1e3:.0f} KB", len(cells), "cells")
