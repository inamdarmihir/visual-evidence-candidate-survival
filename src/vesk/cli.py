"""python -m vesk.cli {smoke|full} --root DIR [--full-late] [--preset ...] [--stage ...]"""
from __future__ import annotations

import argparse
import sys

from .config import PRESETS, RunConfig
from .pipeline import Run

STAGES = ["preflight", "data", "calibrate", "embed", "reverify", "qdrant", "index", "parity", "evaluate", "analyze", "export"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vesk")
    ap.add_argument("mode", choices=["smoke", "full"])
    ap.add_argument("--root", default="./vesk_work", help="local working root (qdrant DB + cache live here)")
    ap.add_argument("--durable-root", default=None, help="optional durable copy of cache/results (e.g. Drive)")
    ap.add_argument("--preset", default="primary_1024", choices=sorted(PRESETS))
    ap.add_argument("--full-late", action="store_true", help="optional unrestricted late-interaction reference")
    ap.add_argument("--run-tag", default="", help="change to start a NEW run instead of reusing a cache")
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--doc-batch", type=int, default=1)
    ap.add_argument("--query-batch", type=int, default=4)
    a = ap.parse_args(argv)
    cfg = RunConfig(mode=a.mode, preset=a.preset, full_late_reference=a.full_late, run_tag=a.run_tag,
                    doc_batch_size=a.doc_batch, query_batch_size=a.query_batch)
    run = Run(cfg, a.root, a.durable_root)
    print("run id:", run.run_id, "dir:", run.dir)
    try:
        for s in a.stages:
            print(f"== {s}")
            if s == "preflight": run.preflight()
            elif s == "data": run.load_data()
            elif s == "calibrate": run.calibrate()
            elif s == "embed": run.embed()
            elif s == "reverify": run.reverify_cache()
            elif s == "qdrant": run.start_qdrant()
            elif s == "index": run.index()
            elif s == "parity": run.parity()
            elif s == "evaluate": run.evaluate()
            elif s == "analyze": run.analyze()
            elif s == "export": print(run.export())
    finally:
        run.stop_qdrant()
        run.save_manifest()


if __name__ == "__main__":
    sys.exit(main())
