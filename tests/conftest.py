"""Fixtures use ONLY the pinned real dataset (no synthetic records). Network/cache required; failures are real."""
import os
from pathlib import Path

import pytest

from vesk.config import DATASET_ID, DATASET_REVISION


@pytest.fixture(scope="session")
def tables():
    from datasets import load_dataset
    ld = lambda n: load_dataset(DATASET_ID, n, split="test", revision=DATASET_REVISION)
    return {"queries": ld("queries").to_pandas(), "qrels": ld("qrels").to_pandas(),
            "meta": ld("documents_metadata").to_pandas(),
            "corpus_ids": ld("corpus").select_columns(["corpus_id", "doc_id", "page_number_in_doc"]).to_pandas()}


@pytest.fixture(scope="session")
def run_dir():
    d = os.environ.get("VESK_RUN_DIR")
    if not d or not (Path(d) / "emb").exists():
        pytest.skip("set VESK_RUN_DIR to a completed run directory (with embeddings) to run cache-based tests")
    return Path(d)
