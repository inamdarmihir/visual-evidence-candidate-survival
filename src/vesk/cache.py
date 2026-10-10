"""Individually addressable, checksummed embedding shards (one page / one query per file)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .util import atomic_write_bytes, atomic_write_json, sha256_file


class ShardCache:
    """<root>/<kind>/<id>.npz  + <id>.done.json (written last = completion marker)."""

    def __init__(self, root, kind: str, identity: dict):
        self.dir = Path(root) / kind
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        idf = self.dir / "CACHE_IDENTITY.json"
        if idf.exists():
            old = json.loads(idf.read_text())
            if old != json.loads(json.dumps(identity, sort_keys=True)):
                diff = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
                raise RuntimeError(
                    f"Refusing stale/incompatible cache at {self.dir}: identity differs in {diff}. "
                    "Use a new run (change run_tag / preset) instead of reusing it.")
        else:
            atomic_write_json(idf, identity)

    def _p(self, key, ext):
        return self.dir / f"{int(key):08d}.{ext}"

    def is_done(self, key) -> bool:
        return self._p(key, "done.json").exists() and self._p(key, "npz").exists()

    def done_keys(self):
        return sorted(int(p.name.split(".")[0]) for p in self.dir.glob("*.done.json")
                      if (self.dir / (p.name.split(".")[0] + ".npz")).exists())

    def write(self, key, arrays: dict, meta: dict):
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        atomic_write_bytes(self._p(key, "npz"), buf.getvalue())
        marker = dict(meta, key=int(key), sha256=sha256_file(self._p(key, "npz")),
                      bytes=self._p(key, "npz").stat().st_size)
        atomic_write_json(self._p(key, "done.json"), marker)  # marker last

    def meta(self, key) -> dict:
        return json.loads(self._p(key, "done.json").read_text())

    def read(self, key, verify=True) -> dict:
        if verify:
            self.verify(key)
        with np.load(self._p(key, "npz")) as z:
            return {k: z[k] for k in z.files}

    def verify(self, key):
        m = self.meta(key)
        got = sha256_file(self._p(key, "npz"))
        if got != m["sha256"]:
            raise RuntimeError(f"checksum mismatch for {self.kind}/{key}: cache corrupt; delete shard to recompute")

    def verify_all(self):
        for k in self.done_keys():
            self.verify(k)
        return len(self.done_keys())
