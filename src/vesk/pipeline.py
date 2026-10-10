"""Stage orchestration shared by the CLI and the notebook. Every stage is resumable."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import qdrant_store as qs
from . import report
from .cache import ShardCache
from .config import (DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION, QDRANT_CLIENT_VERSION,
                     QDRANT_SERVER_VERSION, RunConfig)
from .data import Bundle, load_bundle, qrels_by_query
from .env import Guard, capture_environment, check_pins, free_disk_gb, host_ram_gb
from .util import atomic_write_json, dir_size, read_jsonl, write_jsonl


def _tqdm(**kw):
    from tqdm.auto import tqdm
    return tqdm(**kw)


class Run:
    def __init__(self, cfg: RunConfig, work_root, durable_root=None, progress=_tqdm):
        self.cfg, self.progress = cfg, progress
        self.work_root = Path(work_root)
        self.durable_root = Path(durable_root) if durable_root else None
        self.run_id = cfg.run_id()
        self.dir = self.work_root / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.guard = Guard()
        self.bundle: Bundle | None = None
        self.enc = None
        self.server = None
        self.client = None
        self.manifest = self._load_manifest()
        if self.durable_root:
            self.restore()

    # ---------------------------------------------------------------- manifest
    def _load_manifest(self):
        f = self.dir / "run_manifest.json"
        if f.exists():
            m = json.loads(f.read_text())
            if m.get("identity") != json.loads(json.dumps(self.cfg.identity(), sort_keys=True)):
                raise RuntimeError("Existing run directory has a different identity; use a new run_tag.")
            return m
        return {"run_id": self.run_id, "identity": json.loads(json.dumps(self.cfg.identity(), sort_keys=True)),
                "config": self.cfg.to_dict(), "stages": {}, "timings_s": {}}

    def save_manifest(self):
        atomic_write_json(self.dir / "run_manifest.json", self.manifest)

    def note(self, key, value):
        self.manifest[key] = value
        self.save_manifest()

    @contextmanager
    def timed(self, name):
        t0 = time.perf_counter()
        yield
        self.manifest["timings_s"][name] = self.manifest["timings_s"].get(name, 0.0) + time.perf_counter() - t0
        self.save_manifest()

    @property
    def manifest_hash(self):
        sel = self.bundle.selection
        blob = json.dumps({"identity": self.manifest["identity"], "corpus_ids": sel["corpus_ids"],
                           "query_ids": sel["eval_query_ids"]}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    # ---------------------------------------------------------------- durable sync
    def sync(self):
        """Copy checkpoints/results to the durable root (e.g. Drive). Active DB files are never synced."""
        if not self.durable_root:
            return
        dst = self.durable_root / "runs" / self.run_id
        for p in self.dir.rglob("*"):
            if p.is_file() and "thumbs" not in p.parts:
                t = dst / p.relative_to(self.dir)
                if not t.exists() or t.stat().st_size != p.stat().st_size:
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, t)

    def restore(self):
        src = self.durable_root / "runs" / self.run_id
        if src.exists():
            for p in src.rglob("*"):
                if p.is_file():
                    t = self.dir / p.relative_to(src)
                    if not t.exists():
                        t.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, t)
            self.manifest = self._load_manifest()

    # ---------------------------------------------------------------- stages
    def preflight(self):
        pins = check_pins(strict=True)
        env = capture_environment()
        need = 10.0 if self.cfg.mode == "full" else 3.0
        free = free_disk_gb(self.work_root)
        if free < need:
            raise RuntimeError(f"only {free:.1f} GB free on {self.work_root}; need >= {need} GB for {self.cfg.mode}")
        self.note("environment", env | {"pins_installed": pins, "free_disk_gb_at_start": free,
                                        "transformers_extra": {"torchvision_required_by_image_processor": True}})
        (self.dir / "pip_freeze.txt").write_text("\n".join(env["pip_freeze"]) + "\n")
        return env

    def load_data(self):
        with self.timed("load_data"):
            self.bundle = load_bundle(self.cfg)
        b = self.bundle
        atomic_write_json(self.dir / "selection.json", b.selection)
        b.audit.to_csv(self.dir / "query_audit_counts.csv", index=False)
        grades = qrels_by_query(b.qrels)
        qa = b.queries_all[["query_id", "language", "query_generator", "query_format"]].copy()
        used = set(int(x) for x in b.queries.query_id)
        qa["used_in_this_run"] = [(int(i) in used) and l == "english" and g == "human" for i, l, g in
                                  zip(qa.query_id, qa.language, qa.query_generator)]
        qa["exclusion_reason"] = [
            "" if u else ("translation" if l != "english" else ("sdg (synthetic) query" if g == "sdg" else "not in smoke subset"))
            for u, l, g in zip(qa.used_in_this_run, qa.language, qa.query_generator)]
        qa["n_positive_qrels"] = qa.query_id.map({q: len(v) for q, v in grades.items()}).fillna(0).astype(int)
        qa["note"] = "query_id is shared across translations of the same base query; only english+human rows are evaluated"
        qa.to_csv(self.dir / "query_audit.csv", index=False)
        self.note("data_audit", {
            "dataset": DATASET_ID, "revision": DATASET_REVISION, "n_query_rows": int(len(b.queries_all)),
            "counts_language_generator": b.audit.to_dict("records"),
            "n_eval_queries": len(b.queries), "n_positive_qrels": int(len(b.qrels)),
            "grade_counts": {int(k): int(v) for k, v in b.qrels.score.value_counts().items()},
            "n_multi_page_queries": int(sum(len(v) > 1 for v in grades.values())),
            "n_queries_with_score1": int(sum(any(g == 1 for g in v.values()) for v in grades.values())),
            "n_corpus_pages_used": len(b.corpus_ids), "pinned_assertions": "passed",
            "answer_column": "not used", "markdown_column": "not used",
            "metadata_license_statements": b.meta[["doc_id", "url", "license"]].to_dict("records"),
            "qrel_content_type_values": {str(k): int(v) for k, v in b.qrels.content_type.explode().value_counts().items()}})
        return b

    def load_encoder(self):
        from .embed import Encoder
        if self.enc is None:
            with self.timed("load_model"):
                self.enc = Encoder(self.cfg)
            self.note("model", {"id": MODEL_ID, "revision": MODEL_REVISION, "n_params": self.enc.n_params,
                                "device": str(self.enc.device), "dtype": self.cfg.dtype, "autocast": False,
                                "image_max_side": self.cfg.max_side, "image_patch_size": self.enc.patch,
                                "model_default_max_side": 2048,
                                "deviation": "max_side=1024 is a resource-bounded preset, not a leaderboard reproduction"
                                if self.cfg.max_side != 2048 else "model default"})
        return self.enc

    def caches(self):
        ident = dict(self.manifest["identity"])
        return (ShardCache(self.dir / "emb", "pages", ident), ShardCache(self.dir / "emb", "queries", ident))

    def calibrate(self, n=12):
        self._need()
        from .embed import calibrate
        enc = self.load_encoder()
        cal = calibrate(enc, self.bundle, n=n, seed=self.cfg.seed, guard=self.guard)
        atomic_write_json(self.dir / "calibration.json", cal)
        self.note("calibration", cal)
        return cal

    def embed(self):
        self._need()
        from .embed import embed_pages, embed_queries
        enc = self.load_encoder()
        pc, qc = self.caches()
        with self.timed("embed_queries"):
            embed_queries(enc, self.bundle, qc, self.guard, self.progress)
        with self.timed("embed_pages"):
            embed_pages(enc, self.bundle, pc, self.guard, self.progress)
        self.note("embed_peaks", self.guard.summary())
        self.sync()
        return pc, qc

    def reverify_cache(self, n_sample=8):
        self._need()
        """After loading caches: checksum everything, then re-embed a few real pages and compare."""
        from .embed import check_embeddings
        pc, qc = self.caches()
        npg, nq = pc.verify_all(), qc.verify_all()
        enc = self.load_encoder()
        rng = np.random.default_rng(self.cfg.seed)
        sample = sorted(int(x) for x in rng.choice(self.bundle.corpus_ids, size=min(n_sample, len(self.bundle.corpus_ids)), replace=False))
        worst = 0.0
        for c in sample:
            r = enc.encode_pages([self.bundle.image(c)])[0]
            check_embeddings(r, f"reverify {c}")
            z = pc.read(c)
            worst = max(worst, float(np.abs(r["tokens"] - z["tokens"]).max()), float(np.abs(r["dense"] - z["dense"]).max()))
        rep = {"pages_checksum_ok": npg, "queries_checksum_ok": nq, "sample_pages": sample, "max_abs_reembed_diff": worst}
        if worst > 1e-3:
            raise AssertionError(f"cache vs fresh re-embedding differ by {worst:g}")
        self.note("cache_reverification", rep)
        return rep

    # -- qdrant
    def start_qdrant(self):
        prov = qs.ensure_binary(self.work_root / "qdrant_bin")
        self.server = qs.QdrantServer(Path(prov["binary"]).resolve(), self.work_root / f"qdrant_{self.run_id}", self.cfg.qdrant_port)
        info = self.server.start()
        self.client = qs.client(self.cfg.qdrant_port)
        self.note("qdrant", {"server_version": info.get("version"), "client_version": QDRANT_CLIENT_VERSION, "binary": prov,
                             "config_path": str(self.server.config_path), "log_path": str(self.server.log_path),
                             "storage_on": "local working disk (never Drive)"})
        return info

    def stop_qdrant(self):
        if self.server:
            self.server.stop()
            self.server = None
        shutil.copy2(self.work_root / f"qdrant_{self.run_id}" / "qdrant.log", self.dir / "qdrant.log") \
            if (self.work_root / f"qdrant_{self.run_id}" / "qdrant.log").exists() else None

    def _need(self, qdrant=False):
        if self.bundle is None:
            self.load_data()
        if qdrant and self.client is None:
            self.start_qdrant()

    def index(self):
        self._need(qdrant=True)
        pc, _ = self.caches()
        name = qs.collection_name(self.cfg)
        with self.timed("qdrant_index"):
            schema = qs.ensure_collection(self.client, self.cfg, self.manifest_hash)
            have = {p.id for p in self._scroll_ids(name)}
            from .config import corpus_point_id
            todo = [c for c in self.bundle.corpus_ids if corpus_point_id(c) not in have]
            # re-upserting is idempotent; only missing points are sent
            qs.upsert_pages(self.client, name, pc, self.bundle, todo, batch=4, progress=self.progress) if todo else None
            n = self.client.count(name, exact=True).count
            if n != len(self.bundle.corpus_ids):
                raise RuntimeError(f"collection has {n} points, expected {len(self.bundle.corpus_ids)}")
            qs.wait_green(self.client, name)
        info = self.client.get_collection(name)
        self.note("qdrant_collection", {"name": name, "schema": schema, "points": n, "status": qs.schema_report(info),
                                        "dense_search": "exact=True (not an ANN benchmark)",
                                        "late_vectors": "multivector MAX_SIM, hnsw m=0, memory=cold, float32",
                                        "vector_completeness": "every batch read back and compared with the cache after upsert",
                                        "manifest_hash": self.manifest_hash})
        return n

    def _scroll_ids(self, name):
        out, off = [], None
        while True:
            pts, off = self.client.scroll(name, limit=512, with_payload=False, with_vectors=False, offset=off)
            out += pts
            if off is None:
                return out

    def parity(self):
        self._need(qdrant=True)
        import torch
        pc, qc = self.caches()
        from .experiment import pid_map, run_parity
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        rep = run_parity(self.cfg, self.client, qs.collection_name(self.cfg), self.bundle, pc, qc,
                         pid_map(self.bundle.corpus_ids), device=dev)
        atomic_write_json(self.dir / "parity.json", rep)
        self.note("parity", {k: v for k, v in rep.items() if k != "rows"})
        return rep

    def evaluate(self):
        self._need(qdrant=True)
        from .experiment import run_queries
        pc, qc = self.caches()
        with self.timed("rerank_experiment"):
            res = run_queries(self.cfg, self.client, qs.collection_name(self.cfg), self.bundle, pc, qc,
                              self.dir / "results", self.progress)
        self.sync()
        return res

    # ---------------------------------------------------------------- analysis & export
    def analyze(self):
        self._need()
        import torch
        from .experiment import build_traces, full_late_reference, load_results
        from .metrics import aggregate, per_query_table
        cfg, b = self.cfg, self.bundle
        pc, qc = self.caches()
        results = load_results(self.dir / "results", cfg)
        if len(results) != len(b.queries):
            raise RuntimeError(f"results for {len(results)} of {len(b.queries)} queries; run evaluate() first")
        grades = qrels_by_query(b.qrels)
        full_top = None
        if cfg.full_late_reference:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            with self.timed("full_late_reference_SEPARATE"):
                S, full_top = full_late_reference(cfg, b, pc, qc, self.dir / "full_late", device=dev)
            np.save(self.dir / "full_late_scores.npy", S)
        pq = per_query_table(results, grades, cfg.budgets, cfg.ks, full_top)
        pq.to_csv(self.dir / "per_query_metrics.csv", index=False)
        agg = aggregate(pq, cfg.n_boot, cfg.seed)
        agg["status"] = "computed"
        agg[agg.stage == "candidate"].dropna(axis=1, how="all").to_csv(self.dir / "candidate_metrics.csv", index=False)
        fin = agg[agg.stage != "candidate"].copy()
        if full_top is None:
            fin = pd.concat([fin, pd.DataFrame([{"stage": "full_late", "status": "not computed (optional reference skipped)", "k": k}
                                                for k in cfg.ks])], ignore_index=True)
        fin.to_csv(self.dir / "final_metrics.csv", index=False)
        paired = report.paired_table(pq, cfg.budgets, cfg.ks, cfg.n_boot, cfg.seed, has_full=full_top is not None)
        paired.to_csv(self.dir / "paired_differences.csv", index=False)
        strata = report.strata_table(pq, b.queries, grades, focus_B=25 if 25 in cfg.budgets else cfg.budgets[0], ks=cfg.ks)
        strata.to_csv(self.dir / "query_strata.csv", index=False)
        geometry = {c: {k: pc.meta(c)[k] for k in ("orig_w", "orig_h", "resized_w", "resized_h", "grid_h", "grid_w", "n_tokens")}
                    for c in set(b.qrels.corpus_id.astype(int))}
        traces, pages = build_traces(cfg, b, results, full_top, geometry)
        (self.dir / "traces").mkdir(exist_ok=True)
        write_jsonl(self.dir / "traces" / "per_query_traces.jsonl", traces)
        pages.to_csv(self.dir / "page_level_traces.csv", index=False)
        self._storage_runtime(pc, qc, results)
        self._plots(agg, strata, pages, pc, full_top is not None)
        fb = 25 if 25 in cfg.budgets else cfg.budgets[0]
        n_ex = report.build_viewer(self.dir / "trace_viewer.html", traces, b, geometry, cfg, self.dir / "thumbs", fb, max(cfg.ks))
        self._summary(agg, pages, n_ex, full_top is not None)
        (self.dir / "FINDINGS.md").write_text(report.findings_markdown(agg, pages, paired, cfg, full_top is not None, cfg.mode))
        self.sync()
        return agg

    def _storage_runtime(self, pc, qc, results):
        keys = pc.done_keys()
        metas = [pc.meta(k) for k in keys]
        pages = pd.DataFrame(metas)
        pages[["corpus_id", "n_tokens", "bytes", "seconds", "orig_w", "orig_h", "resized_w", "resized_h", "grid_h", "grid_w"]].to_csv(
            self.dir / "page_token_counts.csv", index=False)
        n_tok = int(pages.n_tokens.sum())
        n_pages = len(pages)
        rows = [
            ("pages_embedded", n_pages, "count", ""),
            ("valid_page_vectors_total", n_tok, "vectors", "all attention-mask==1 rows; special/row tokens kept"),
            ("valid_page_vectors_mean", pages.n_tokens.mean(), "vectors/page", ""),
            ("valid_page_vectors_min", pages.n_tokens.min(), "vectors/page", ""),
            ("valid_page_vectors_max", pages.n_tokens.max(), "vectors/page", ""),
            ("computed_late_storage_fp32", n_tok * 128 * 4, "bytes", "sum(valid_tokens)*128*4"),
            ("computed_dense_storage_fp32", n_pages * 1024 * 4, "bytes", "pages*1024*4"),
            ("actual_page_cache_on_disk", dir_size(pc.dir), "bytes", "npz shards incl. dense, valid_index, input_ids, markers"),
            ("actual_query_cache_on_disk", dir_size(qc.dir), "bytes", ""),
            ("embed_seconds_per_page_mean", pages.seconds.mean(), "s", "measured, this device"),
            ("embed_seconds_per_page_median", pages.seconds.median(), "s", "measured, this device"),
            ("embed_seconds_total_sum_of_pages", pages.seconds.sum(), "s", "sum over page batches (cache resumes may span sessions)"),
            ("query_vectors_total", int(sum(qc.meta(k)["n_tokens"] for k in qc.done_keys())), "vectors", "includes <mask> expansion tokens"),
            ("free_disk_now", free_disk_gb(self.work_root) * 1e9, "bytes", ""),
            ("host_ram_available_now", host_ram_gb()["available_gb"] * 1e9, "bytes", ""),
        ]
        qd = self.work_root / f"qdrant_{self.run_id}"
        if qd.exists():
            rows.append(("actual_qdrant_dir_on_disk", dir_size(qd), "bytes", "storage + snapshots + logs"))
        for k, v in self.guard.summary().items():
            rows.append((k, v * 1e9, "bytes", "this process, since start (calibration/embedding only if run in this session)"))
        for k, v in self.manifest["timings_s"].items():
            rows.append((f"time_{k}", v, "s", "wall clock"))
        sr = pd.DataFrame(rows, columns=["metric", "value", "unit", "notes"])
        sr.to_csv(self.dir / "storage_and_runtime.csv", index=False)
        self.note("storage", {r.metric: r.value for r in sr.itertuples() if r.unit == "bytes" or r.metric.startswith("valid")})
        self._pages_df = pages

    def _plots(self, agg, strata, pages, pc, has_full):
        cfg = self.cfg
        pdir = self.dir / "plots"
        pdir.mkdir(exist_ok=True)
        report.plot_candidate(agg, pdir / "candidate_recall_coverage_vs_B.png")
        report.plot_final(agg, pdir / "final_topk_before_after_rerank.png", cfg.ks, has_full)
        fb = 25 if 25 in cfg.budgets else cfg.budgets[0]
        report.plot_strata(strata, pdir / "query_type_breakdown.png", fb, k=max(cfg.ks))
        report.plot_loss(pages, pdir / "loss_distribution.png", cfg.ks)
        dev = self.manifest.get("model", {}).get("device", "unknown device")
        gpu = self.manifest.get("environment", {}).get("gpu", {}).get("gpu_name")
        report.plot_storage_timing(self._pages_df, pdir / "storage_and_timing.png", gpu or dev)

    def _summary(self, agg, pages, n_ex, has_full):
        c = agg[agg.stage == "candidate"].sort_values("B")
        lines = [f"# Run summary: {self.run_id}", ""]
        if self.cfg.mode == "smoke":
            lines += ["> **SMOKE / ENGINEERING FIXTURE.** 8 queries and a 256-page sub-corpus that retains their positives (an easier, "
                      "qrel-aware setup). These numbers must NOT be compared with the full-corpus results or reported as a benchmark.", ""]
        lines += [f"- queries: {c.n_queries.max()} human-origin English; pages: {len(self.bundle.corpus_ids)}; preset `{self.cfg.preset}` (longest side {self.cfg.max_side}px)",
                  f"- full-corpus late reference: {'computed' if has_full else 'not computed'}", "", "## Candidate stage (macro, with 95% bootstrap CI over queries)",
                  "| B | recall | coverage |", "|---|---|---|"]
        for r in c.itertuples():
            lines.append(f"| {int(r.B)} | {r.macro_recall:.3f} [{r.recall_ci_lo:.3f}, {r.recall_ci_hi:.3f}] | {r.macro_coverage:.3f} [{r.coverage_ci_lo:.3f}, {r.coverage_ci_hi:.3f}] |")
        lines += ["", "Coverage of all annotated pages is not answer completeness. Unjudged pages are not established negatives.",
                  f"Trace viewer examples: {n_ex} (deterministic rules, see viewer)."]
        (self.dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n")

    def run_all(self, calibrate=True):
        """Every stage in order (used by the smoke cell). Always stops the Qdrant subprocess."""
        try:
            self.preflight(); self.load_data()
            if calibrate:
                self.calibrate()
            self.embed(); self.reverify_cache(); self.start_qdrant(); self.index(); self.parity(); self.evaluate()
            return self.analyze()
        finally:
            self.stop_qdrant()

    def export(self, out_zip=None):
        out = Path(out_zip or self.work_root / f"export_{self.run_id}.zip")
        root = Path(__file__).resolve().parents[2]
        report.export_zip(root, self.dir, out)
        return out
