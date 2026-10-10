"""Pinned constants, presets and the manifest-derived run identity."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field

MODEL_ID = "Hcompany/NeoMME-260M-Retriever"
MODEL_REVISION = "481083153ac22931afbd614e672212bffbcb666e"
QDRANT_SERVER_VERSION = "1.19.2"
QDRANT_CLIENT_VERSION = "1.19.1"

_CS_EXPECTED = {  # asserted against the pinned dataset; stop with an explanation otherwise
    "n_corpus": 1360, "n_queries_all": 1290, "n_human_english": 110, "n_sdg_english": 105,
    "n_qrels_positive": 545, "n_score1": 451, "n_score2": 94, "n_multi_page": 93,
    "n_with_score1": 90, "n_docs": 2,
    "doc_pages": {"Introduction_to_Computer_Science": 945, "Introduction_to_Python_Programming": 415},
    "query_types": {"extractive": 44, "enumerative": 29, "open-ended": 25, "compare-contrast": 21,
                    "multi-hop": 21, "boolean": 11, "numerical": 1},
}


# Dataset registry. Every entry is pinned to an exact revision; `expected` invariants are asserted on load.
# Only computer_science carries the full set of audited invariants (it is the primary, hand-audited dataset);
# the sibling domains pin the counts that were verified against their revisions and skip the rest.
DATASETS = {
    "computer_science": {"id": "vidore/vidore_v3_computer_science",
                         "revision": "d5cc75883d92e294f0c0fc2662551c9708a06ebc", "expected": _CS_EXPECTED},
    "hr": {"id": "vidore/vidore_v3_hr", "revision": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3",
           "expected": {"n_corpus": 1110, "n_queries_all": 1908, "n_human_english": 167,
                        "n_qrels_positive": 741, "n_score1": 598, "n_score2": 143, "n_docs": 14}},
    "physics": {"id": "vidore/vidore_v3_physics", "revision": "a0de276f515acc044b72cae8de53a44bb5a8f1f5",
                "expected": {"n_corpus": 1674, "n_queries_all": 1812, "n_human_english": 147,
                             "n_qrels_positive": 1082, "n_score1": 474, "n_score2": 608, "n_docs": 42}},
    "energy": {"id": "vidore/vidore_v3_energy", "revision": "caec06d3c73434d635f710f93bcd898331c59f20",
               "expected": {"n_corpus": 2225, "n_queries_all": 1848, "n_human_english": 194,
                            "n_qrels_positive": 694, "n_score1": 448, "n_score2": 246, "n_docs": 41}},
    "pharmaceuticals": {"id": "vidore/vidore_v3_pharmaceuticals",
                        "revision": "3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c",
                        "expected": {"n_corpus": 2313, "n_queries_all": 2184, "n_human_english": 182,
                                     "n_qrels_positive": 700, "n_score1": 563, "n_score2": 137, "n_docs": 52}},
}
DATASET_KEY = os.environ.get("VESK_DATASET", "computer_science")
if DATASET_KEY not in DATASETS:
    raise ValueError(f"VESK_DATASET={DATASET_KEY!r}; choose from {sorted(DATASETS)}")
DATASET_ID = DATASETS[DATASET_KEY]["id"]
DATASET_REVISION = DATASETS[DATASET_KEY]["revision"]
EXPECTED = DATASETS[DATASET_KEY]["expected"]

PIN_REQUIREMENTS = {"transformers": "5.19.0", "sentence-transformers": "6.1.0",
                    "qdrant-client": "1.19.1", "datasets": "5.1.0"}

# Official release assets; sha256 values are the digests published by GitHub for v1.19.2.
QDRANT_ASSETS = {
    ("Linux", "x86_64"): {
        "url": "https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-x86_64-unknown-linux-gnu.tar.gz",
        "sha256": "34a788a09a4cb278b5c6d2af96d9a6db30c8c7e88ae3550455f7a4d425bb8d4b"},
    # Development-only builds of the SAME server version, used when this repo is exercised off-Colab.
    ("Darwin", "arm64"): {
        "url": "https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-aarch64-apple-darwin.tar.gz",
        "sha256": "d24dfd25f4584684b337dd30ee016472cd25f53fc2501f1cd058ab3d7ee0b019"},
}

PRESETS = {
    "primary_1024": {"max_side": 1024, "note": "resource-bounded primary preset; NOT the model default (2048)"},
    "optional_2048": {"max_side": 2048, "note": "model default longest-side cap; separate cache/run identity"},
}

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"hf://datasets/{DATASET_ID}")


def corpus_point_id(corpus_id: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"corpus:{int(corpus_id)}"))


def query_point_id(query_id: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"query:{int(query_id)}"))


@dataclass(frozen=True)
class RunConfig:
    mode: str = "smoke"                    # "smoke" | "full"
    preset: str = "primary_1024"
    seed: int = 42
    budgets: tuple = (10, 25, 50, 100)
    ks: tuple = (5, 10)
    doc_batch_size: int = 1
    query_batch_size: int = 4
    dtype: str = "float32"
    allow_float16: bool = False             # separate explicit flag; needs parity check + distinct manifest
    n_boot: int = 2000
    smoke_n_queries: int = 8
    smoke_n_pages: int = 256
    dense_search_exact: bool = True
    full_late_reference: bool = False       # optional unrestricted late-interaction reference
    qdrant_port: int = 6333
    run_tag: str = ""                       # explicit new-run path: change the tag

    def __post_init__(self):
        if self.mode not in ("smoke", "full"):
            raise ValueError("mode must be 'smoke' or 'full'")
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}")
        if self.dtype not in ("float32", "float16"):
            raise ValueError("dtype must be float32 or float16 (bf16 is not offered on T4)")
        if self.dtype == "float16" and not self.allow_float16:
            raise ValueError("float16 requires allow_float16=True plus a float32 parity check; never silent")
        if self.budgets != tuple(sorted(self.budgets)) or max(self.budgets) < max(self.ks):
            raise ValueError("budgets must be ascending and >= max(ks)")

    @property
    def max_side(self) -> int:
        return PRESETS[self.preset]["max_side"]

    def identity(self) -> dict:
        """Everything that makes cached embeddings / results incompatible when changed."""
        return {
            "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "mode": self.mode, "preset": self.preset, "max_side": self.max_side,
            "dtype": self.dtype, "seed": self.seed,
            "smoke": ({"n_queries": self.smoke_n_queries, "n_pages": self.smoke_n_pages}
                      if self.mode == "smoke" else None),
            "qdrant_server": QDRANT_SERVER_VERSION, "pins": PIN_REQUIREMENTS,
            "valid_mask_policy": "keep every position with attention_mask==1; only padding trimmed",
            "run_tag": self.run_tag,
        }

    def run_id(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True).encode()
        return f"{self.mode}-{self.preset}-{hashlib.sha256(blob).hexdigest()[:10]}"

    def to_dict(self) -> dict:
        return asdict(self)
