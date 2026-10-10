"""Atomic IO, hashing, jsonl helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def atomic_write_bytes(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=_default).encode())


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_jsonl(path, rows):
    atomic_write_bytes(path, ("".join(json.dumps(r, sort_keys=True, default=_default) + "\n" for r in rows)).encode())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def dir_size(path, allocated=True) -> int:
    """Bytes under path. allocated=True counts blocks actually on disk (sparse/preallocated files report a
    much larger apparent size)."""
    tot = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            st = p.stat()
            tot += st.st_blocks * 512 if allocated and hasattr(st, "st_blocks") else st.st_size
    return tot
