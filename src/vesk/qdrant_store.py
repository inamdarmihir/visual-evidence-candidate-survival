"""Pinned local Qdrant server (subprocess), collection schema, verified upserts, candidate reranking."""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import (QDRANT_ASSETS, QDRANT_SERVER_VERSION, RunConfig, corpus_point_id)
from .util import atomic_write_json, sha256_file

DENSE, LATE = "dense", "late"


def ensure_binary(dest_dir) -> dict:
    """Download once, verify against the pinned SHA256, extract. Returns provenance."""
    key = (platform.system(), platform.machine())
    if key not in QDRANT_ASSETS:
        raise RuntimeError(f"No pinned Qdrant v{QDRANT_SERVER_VERSION} asset for {key}; "
                           "the primary target is Linux x86_64 (Colab).")
    asset = QDRANT_ASSETS[key]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tgz = dest / Path(asset["url"]).name
    if not tgz.exists():
        urllib.request.urlretrieve(asset["url"], tgz)
    got = sha256_file(tgz)
    if got != asset["sha256"]:
        tgz.unlink()
        raise RuntimeError(f"SHA256 mismatch for {tgz.name}: {got} != pinned {asset['sha256']}")
    exe = dest / "bin" / "qdrant"
    if not exe.exists():
        (dest / "bin").mkdir(exist_ok=True)
        with tarfile.open(tgz) as t:
            m = next(x for x in t.getmembers() if x.isfile() and Path(x.name).name == "qdrant")
            m.name = "qdrant"
            t.extract(m, dest / "bin", filter="data")
        exe.chmod(0o755)
    return {"url": asset["url"], "sha256": got, "binary": str(exe), "platform": f"{key[0]}/{key[1]}"}


class QdrantServer:
    def __init__(self, binary, workdir, port=6333, grpc_port=6334):
        self.binary, self.work, self.port, self.grpc = str(binary), Path(workdir), port, grpc_port
        self.proc = None
        self.log_path = self.work / "qdrant.log"
        self.config_path = self.work / "config.yaml"

    def write_config(self):
        self.work.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            "log_level: INFO\n"
            f"storage:\n  storage_path: {self.work / 'storage'}\n  snapshots_path: {self.work / 'snapshots'}\n"
            f"service:\n  host: 127.0.0.1\n  http_port: {self.port}\n  grpc_port: {self.grpc}\n"
            "telemetry_disabled: true\n")

    def start(self, timeout=90):
        self.write_config()
        self.log = open(self.log_path, "ab")
        self.proc = subprocess.Popen([self.binary, "--config-path", str(self.config_path)], cwd=self.work,
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"qdrant exited early (code {self.proc.returncode}); see {self.log_path}")
            try:
                info = self.root_info()
                if info.get("version") != QDRANT_SERVER_VERSION:
                    raise RuntimeError(f"server version {info.get('version')} != pinned {QDRANT_SERVER_VERSION}")
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/readyz", timeout=2) as r:
                    if r.status == 200:
                        return info
            except (OSError, ValueError):
                time.sleep(0.5)
        raise TimeoutError(f"qdrant not ready after {timeout}s; see {self.log_path}")

    def root_info(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=2) as r:
            return json.loads(r.read())

    def stop(self, timeout=30):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "log", None):
            self.log.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()


def client(port=6333):
    from qdrant_client import QdrantClient
    return QdrantClient(host="127.0.0.1", port=port, prefer_grpc=False, timeout=600, check_compatibility=False)


def expected_schema():
    from qdrant_client import models as m
    return {
        DENSE: m.VectorParams(size=1024, distance=m.Distance.COSINE, datatype=m.Datatype.FLOAT32),
        LATE: m.VectorParams(size=128, distance=m.Distance.COSINE, datatype=m.Datatype.FLOAT32,
                             memory=m.Memory.COLD,  # supported cold placement; deprecated on_disk NOT used
                             hnsw_config=m.HnswConfigDiff(m=0),  # no per-token HNSW
                             multivector_config=m.MultiVectorConfig(comparator=m.MultiVectorComparator.MAX_SIM)),
    }


def collection_name(cfg: RunConfig) -> str:
    return "vesk_" + cfg.run_id().replace("-", "_")


def ensure_collection(c, cfg: RunConfig, manifest_hash: str) -> dict:
    """Create, or validate schema + manifest hash of an existing collection before reusing it."""
    name = collection_name(cfg)
    if c.collection_exists(name):
        info = c.get_collection(name)
        meta = (info.config.metadata or {})
        if meta.get("manifest_hash") != manifest_hash:
            raise RuntimeError(f"collection {name} manifest hash {meta.get('manifest_hash')} != {manifest_hash}; "
                               "refusing to reuse (start a new run).")
        validate_schema(info)
    else:
        c.create_collection(name, vectors_config=expected_schema(), metadata={"manifest_hash": manifest_hash})
    return schema_report(c.get_collection(name))


def validate_schema(info):
    vp = info.config.params.vectors
    dense, late = vp[DENSE], vp[LATE]
    errs = []
    if dense.size != 1024 or str(dense.distance.value).lower() != "cosine":
        errs.append("dense schema")
    if late.size != 128 or str(late.distance.value).lower() != "cosine":
        errs.append("late size/distance")
    if late.multivector_config is None or late.multivector_config.comparator.value != "max_sim":
        errs.append("late comparator")
    if late.hnsw_config is None or late.hnsw_config.m != 0:
        errs.append("late hnsw m != 0")
    if errs:
        raise RuntimeError("existing collection schema mismatch: " + ", ".join(errs))


def schema_report(info) -> dict:
    return json.loads(info.config.params.model_dump_json()) | {
        "points_count": info.points_count, "status": str(info.status), "indexed_vectors_count": info.indexed_vectors_count,
        "optimizer_status": str(info.optimizer_status)}


def upsert_pages(c, name, cache, bundle, cids, batch=4, progress=None, verify_atol=1e-6):
    """Idempotent upsert of a few pages at a time; each batch is read back and verified against the cache."""
    from qdrant_client import models as m
    bar = progress(total=len(cids), desc="upsert") if progress else None
    for s in range(0, len(cids), batch):
        chunk = cids[s:s + batch]
        pts, shards = [], {}
        for cid in chunk:
            z = cache.read(cid)
            shards[cid] = z
            meta = cache.meta(cid)
            pts.append(m.PointStruct(
                id=corpus_point_id(cid),
                vector={DENSE: z["dense"].tolist(), LATE: z["tokens"].tolist()},  # one page at a time
                payload={"corpus_id": int(cid), "doc_id": bundle.doc_of[cid],
                         "page_number_in_doc": bundle.page_of[cid], "n_tokens": meta["n_tokens"],
                         "row_offset": bundle.row_of[cid]}))
        c.upsert(name, points=pts, wait=True)
        got = c.retrieve(name, ids=[p.id for p in pts], with_vectors=[DENSE, LATE], with_payload=True)
        if len(got) != len(chunk):
            raise RuntimeError("upsert verification: missing points")
        byid = {g.id: g for g in got}
        for cid in chunk:
            g = byid[corpus_point_id(cid)]
            lv = np.asarray(g.vector[LATE], dtype=np.float32)
            dv = np.asarray(g.vector[DENSE], dtype=np.float32)
            z = shards[cid]
            if lv.shape != z["tokens"].shape or dv.shape != z["dense"].shape:
                raise RuntimeError(f"page {cid}: stored vector shape {lv.shape} != cached {z['tokens'].shape}")
            if not (np.allclose(lv, z["tokens"], atol=verify_atol) and np.allclose(dv, z["dense"], atol=verify_atol)):
                raise RuntimeError(f"page {cid}: stored vectors differ from cache")
            if g.payload["corpus_id"] != cid:
                raise RuntimeError(f"page {cid}: payload mismatch")
        del pts, shards, got
        if bar:
            bar.update(len(chunk))
    if bar:
        bar.close()
    n = c.count(name, exact=True).count
    if n != len(cids):
        raise RuntimeError(f"point count {n} != {len(cids)}")
    return n


def wait_green(c, name, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if str(c.get_collection(name).status).lower().endswith("green"):
            return True
        time.sleep(1)
    return False


def _pid_to_cid(points, pid_map):
    return [pid_map[str(p.id)] for p in points]


def dense_topk(c, name, qdense, pid_map, limit=100, exact=True):
    from qdrant_client import models as m
    r = c.query_points(name, query=qdense.tolist(), using=DENSE, limit=limit, with_payload=False,
                       search_params=m.SearchParams(exact=exact))
    return _pid_to_cid(r.points, pid_map), [float(p.score) for p in r.points]


def rerank_prefetch(c, name, qdense, qlate, B, pid_map, exact=True):
    """Server-side: dense prefetch (limit B, exact) then named late-vector MAX_SIM; score/n_q => MeanMaxSim."""
    from qdrant_client import models as m
    r = c.query_points(
        name, prefetch=m.Prefetch(query=qdense.tolist(), using=DENSE, limit=B,
                                  params=m.SearchParams(exact=exact)),
        query=qlate.tolist(), using=LATE, limit=B, with_payload=False)
    n_q = qlate.shape[0]
    return _pid_to_cid(r.points, pid_map), [float(p.score) / n_q for p in r.points]


def rerank_id_filter(c, name, qlate, candidate_cids, pid_map):
    """Fallback: late scoring restricted to explicit saved candidate IDs (cannot leave the set)."""
    from qdrant_client import models as m
    r = c.query_points(name, query=qlate.tolist(), using=LATE, limit=len(candidate_cids), with_payload=False,
                       query_filter=m.Filter(must=[m.HasIdCondition(has_id=[corpus_point_id(x) for x in candidate_cids])]))
    n_q = qlate.shape[0]
    return _pid_to_cid(r.points, pid_map), [float(p.score) / n_q for p in r.points]
