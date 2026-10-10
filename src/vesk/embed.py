"""One NeoMME forward pass -> dense + multivector embeddings, cached per page / per query."""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from .config import MODEL_ID, MODEL_REVISION, RunConfig
from .env import Guard, rss_gb


class Encoder:
    def __init__(self, cfg: RunConfig, device: str | None = None):
        from transformers import NeoMMEForRetrieval, NeoMMEProcessor  # fail clearly if unavailable
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if cfg.dtype == "float16":
            raise NotImplementedError("float16 path needs a separate parity-checked experiment; not enabled here")
        self.dtype = torch.float32
        self.processor = NeoMMEProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = NeoMMEForRetrieval.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=self.dtype)
        self.model.to(self.device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.patch = int(self.processor.image_processor.patch_size)

    # -- inputs ----------------------------------------------------------------------------------
    def _encode(self, messages, task, **extra):
        kw = {"padding": "longest", **extra}
        return self.processor.apply_chat_template(messages, task=task, tokenize=True, return_dict=True,
                                                  return_tensors="pt", processor_kwargs=kw)

    def effective_geometry(self, w, h):
        """Resized size / patch grid from the pinned implementation's own function (not re-derived)."""
        from transformers.models.neomme.image_processing_neomme import get_resize_output_size
        rh, rw = get_resize_output_size(h, w, self.cfg.max_side, None)
        gh, gw = -(-rh // self.patch), -(-rw // self.patch)
        return {"orig_w": w, "orig_h": h, "resized_w": rw, "resized_h": rh, "grid_h": gh, "grid_w": gw,
                "n_patches": gh * gw, "expected_tokens": 2 + gh * (gw + 1)}

    @torch.inference_mode()
    def _forward(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.embeddings, out.dense_embeddings, inputs["attention_mask"], inputs["input_ids"]

    @staticmethod
    def _split(emb, dense, mask, ids):
        res = []
        for i in range(emb.shape[0]):
            valid = mask[i].bool()
            idx = torch.nonzero(valid, as_tuple=True)[0]
            tok = emb[i, idx].float().cpu().numpy()
            # padding rows must have been zeroed by the model; check before discarding them
            if (~valid).any():
                assert float(emb[i, ~valid].abs().max()) == 0.0, "padding rows not zeroed"
            res.append({"tokens": tok, "dense": dense[i].float().cpu().numpy(),
                        "valid_index": idx.cpu().numpy().astype(np.int32),
                        "input_ids": ids[i, idx].cpu().numpy().astype(np.int32),
                        "padded_len": int(mask.shape[1])})
        return res

    def encode_pages(self, images):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}]}] for im in images]
        inputs = self._encode(msgs, "document", max_side=self.cfg.max_side)
        res = self._split(*self._forward(inputs))
        for im, r in zip(images, res):
            g = self.effective_geometry(*im.size)
            if r["tokens"].shape[0] != g["expected_tokens"]:
                raise RuntimeError(f"valid token count {r['tokens'].shape[0]} != layout {g['expected_tokens']}")
            r["geometry"] = g
        return res

    def encode_queries(self, texts):
        msgs = [[{"role": "user", "content": t}] for t in texts]
        return self._split(*self._forward(self._encode(msgs, "query")))


def check_embeddings(r: dict, what: str, tol=1e-3):
    t, d = r["tokens"], r["dense"]
    if not (np.isfinite(t).all() and np.isfinite(d).all()):
        raise FloatingPointError(f"{what}: non-finite embeddings")
    if t.ndim != 2 or t.shape[1] != 128 or d.shape != (1024,):
        raise ValueError(f"{what}: unexpected shapes {t.shape} {d.shape}")
    if np.abs(np.linalg.norm(t, axis=1) - 1).max() > tol or abs(np.linalg.norm(d) - 1) > tol:
        raise ValueError(f"{what}: embeddings not unit-normalised (tol {tol})")


def _arrays(r):
    return {"tokens": r["tokens"].astype(np.float32), "dense": r["dense"].astype(np.float32),
            "valid_index": r["valid_index"], "input_ids": r["input_ids"]}


def embed_pages(enc: Encoder, bundle, cache, guard: Guard, progress=None):
    """Resume-safe: skips pages whose shard + completion marker + checksum already verify."""
    todo = [c for c in bundle.corpus_ids if not cache.is_done(c)]
    bs = enc.cfg.doc_batch_size
    bar = progress(total=len(bundle.corpus_ids), initial=len(bundle.corpus_ids) - len(todo), desc="pages") if progress else None
    for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        images = [bundle.image(c) for c in chunk]            # lazy decode, one batch only
        t0 = time.perf_counter()
        res = enc.encode_pages(images)
        if enc.device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / len(chunk)
        for c, r in zip(chunk, res):
            check_embeddings(r, f"page {c}")
            cache.write(c, _arrays(r), {
                "corpus_id": int(c), "n_tokens": int(r["tokens"].shape[0]), "padded_len": r["padded_len"],
                "seconds": dt, "rss_gb": rss_gb(), "dtype": "float32", **r["geometry"]})
        del images, res
        guard.check()
        if bar:
            bar.update(len(chunk))
    if bar:
        bar.close()


def embed_queries(enc: Encoder, bundle, cache, guard: Guard, progress=None):
    q = bundle.queries
    todo = [(int(i), t) for i, t in zip(q.query_id, q["query"]) if not cache.is_done(i)]
    bs = enc.cfg.query_batch_size
    bar = progress(total=len(q), initial=len(q) - len(todo), desc="queries") if progress else None
    for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        t0 = time.perf_counter()
        res = enc.encode_queries([t for _, t in chunk])
        dt = (time.perf_counter() - t0) / len(chunk)
        for (qid, text), r in zip(chunk, res):
            check_embeddings(r, f"query {qid}")
            cache.write(qid, _arrays(r), {"query_id": qid, "n_tokens": int(r["tokens"].shape[0]),
                                          "padded_len": r["padded_len"], "seconds": dt, "dtype": "float32"})
        guard.check()
        if bar:
            bar.update(len(chunk))
    if bar:
        bar.close()


def calibrate(enc: Encoder, bundle, n=12, seed=42, guard: Guard | None = None):
    """Measured calibration on real pages (first page = warm-up, excluded). Not cached."""
    guard = guard or Guard()
    rng = random.Random(seed)
    cids = sorted(rng.sample(bundle.corpus_ids, min(n + 1, len(bundle.corpus_ids))))
    if enc.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    secs, toks = [], []
    for k, c in enumerate(cids):
        im = bundle.image(c)
        t0 = time.perf_counter()
        r = enc.encode_pages([im])[0]
        if enc.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        check_embeddings(r, f"calibration page {c}")
        guard.check()
        if k:  # skip warm-up
            secs.append(dt)
            toks.append(r["tokens"].shape[0])
    n_full = len(bundle.corpus_ids)
    mean_s = float(np.mean(secs))
    return {"device": str(enc.device), "pages": cids[1:], "warmup_page": cids[0],
            "seconds_per_page_mean": mean_s, "seconds_per_page_std": float(np.std(secs)),
            "mean_tokens": float(np.mean(toks)), **guard.summary(),
            "projected_full_corpus_pages": n_full,
            "projected_full_minutes": mean_s * n_full / 60,
            "assumptions": "projection = mean measured s/page x page count on THIS device and preset; "
                           "ignores image-decode variance, thermal/shared-GPU throttling and session limits; "
                           "no other device's throughput is borrowed"}
