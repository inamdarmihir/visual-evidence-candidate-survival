"""Pinned dataset loading, audit, validation and deterministic smoke selection.

Only human-origin English queries are evaluated. Nothing here creates queries, labels or pages.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .config import DATASET_ID, DATASET_REVISION, EXPECTED, RunConfig


class DataValidationError(RuntimeError):
    pass


@dataclass
class Bundle:
    queries_all: pd.DataFrame      # all 1,290 rows (audit only)
    queries: pd.DataFrame          # evaluation queries (human, english [, smoke subset])
    qrels: pd.DataFrame            # positive qrels of the evaluation queries
    meta: pd.DataFrame             # documents_metadata
    corpus_ids: list               # corpus pages used in this run, sorted
    row_of: dict                   # corpus_id -> row offset in the corpus split (never assumed equal)
    doc_of: dict                   # corpus_id -> doc_id
    page_of: dict                  # corpus_id -> page_number_in_doc (0-based as stored)
    corpus_ds: object              # lazily decoded HF dataset (image column decoded on access only)
    audit: pd.DataFrame
    selection: dict                # exact chosen/excluded IDs

    def image(self, corpus_id):
        return self.corpus_ds[self.row_of[corpus_id]]["image"]


def _load(name):
    from datasets import load_dataset
    return load_dataset(DATASET_ID, name, split="test", revision=DATASET_REVISION)


def query_audit(q: pd.DataFrame) -> pd.DataFrame:
    a = q.groupby(["language", "query_generator"]).size().rename("n_rows").reset_index()
    a["used_for_primary_evaluation"] = (a.language == "english") & (a.query_generator == "human")
    return a


def assert_pinned_source(q: pd.DataFrame, qrels: pd.DataFrame, meta: pd.DataFrame, n_corpus: int):
    """Stop with an explanation if the pinned source does not match the verified invariants."""
    E = EXPECTED
    problems = []
    if n_corpus != E["n_corpus"]:
        problems.append(f"corpus rows {n_corpus} != {E['n_corpus']}")
    if len(q) != E["n_queries_all"]:
        problems.append(f"query rows {len(q)} != {E['n_queries_all']}")
    h = q[(q.language == "english") & (q.query_generator == "human")]
    s = q[(q.language == "english") & (q.query_generator == "sdg")]
    if len(h) != E["n_human_english"]:
        problems.append(f"human english queries {len(h)} != {E['n_human_english']}")
    if "n_sdg_english" in E and len(s) != E["n_sdg_english"]:
        problems.append(f"sdg english queries {len(s)} != {E['n_sdg_english']}")
    if not h.query_id.is_unique:
        problems.append("human english query_id not unique")
    r = qrels[qrels.query_id.isin(h.query_id)]
    vc = r.score.value_counts().to_dict()
    if len(r) != E["n_qrels_positive"] or vc.get(1) != E["n_score1"] or vc.get(2) != E["n_score2"]:
        problems.append(f"qrels {len(r)} / {vc} != {E['n_qrels_positive']} / 1:{E['n_score1']} 2:{E['n_score2']}")
    if r.duplicated(["query_id", "corpus_id"]).any():
        problems.append("duplicate (query_id, corpus_id) qrels")
    if r.query_id.nunique() != len(h):
        problems.append("some human english queries have no qrels (missing labels)")
    if "n_multi_page" in E and (r.groupby("query_id").corpus_id.nunique() > 1).sum() != E["n_multi_page"]:
        problems.append("multi-page query count differs")
    if "n_with_score1" in E and r.groupby("query_id").score.apply(lambda s: (s == 1).any()).sum() != E["n_with_score1"]:
        problems.append("queries with a score-1 page differ")
    qt = h.query_types.explode().value_counts().to_dict()
    if "query_types" in E and qt != E["query_types"]:
        problems.append(f"query_types {qt} != {E['query_types']}")
    pages = dict(zip(meta.doc_id, meta.page_number)) if "doc_id" in meta else dict(zip(meta.file_name.str.replace(".pdf", "", regex=False), meta.page_number))
    if len(meta) != E["n_docs"]:
        problems.append(f"documents_metadata rows {len(meta)} != {E['n_docs']}")
    if "doc_pages" in E and pages != E["doc_pages"]:
        problems.append(f"documents_metadata pages {pages} != {E['doc_pages']}")
    if problems:
        raise DataValidationError(
            "The pinned dataset revision does not match the verified invariants; refusing to run.\n  - "
            + "\n  - ".join(problems)
            + f"\nSource: {DATASET_ID}@{DATASET_REVISION}. Do not relabel generated queries as human-written.")


def select_smoke(human_ids, qrels, all_corpus_ids, cfg: RunConfig):
    """Deterministic, retrieval-score-blind: seeded sample of queries, all their positives, then
    seeded fill with OTHER real pages up to cfg.smoke_n_pages."""
    rng = random.Random(cfg.seed)
    qids = sorted(rng.sample(sorted(int(x) for x in human_ids), cfg.smoke_n_queries))
    pos = sorted(set(int(c) for c in qrels[qrels.query_id.isin(qids)].corpus_id))
    if len(pos) > cfg.smoke_n_pages:
        raise DataValidationError("smoke positives exceed smoke corpus size")
    others = sorted(set(int(c) for c in all_corpus_ids) - set(pos))
    fill = sorted(rng.sample(others, cfg.smoke_n_pages - len(pos)))
    return qids, sorted(pos + fill), {"positive_pages": pos, "filler_pages": fill}


def load_bundle(cfg: RunConfig) -> Bundle:
    q_all = _load("queries").to_pandas()
    qrels_all = _load("qrels").to_pandas()
    meta = _load("documents_metadata").to_pandas()
    corpus = _load("corpus")
    ids_ds = corpus.select_columns(["corpus_id", "doc_id", "page_number_in_doc"]).to_pandas()
    assert_pinned_source(q_all, qrels_all, meta, len(corpus))
    if not ids_ds.corpus_id.is_unique:
        raise DataValidationError("corpus_id not unique")
    audit = query_audit(q_all)

    human = q_all[(q_all.language == "english") & (q_all.query_generator == "human")].copy()
    qrels = qrels_all[qrels_all.query_id.isin(human.query_id)].copy()
    missing = set(qrels.corpus_id) - set(ids_ds.corpus_id)
    if missing:
        raise DataValidationError(f"qrels reference pages absent from the corpus: {sorted(missing)[:5]}")
    # page numbers are 0-based as stored: max index must equal page_count-1 per book
    for d, n in EXPECTED.get("doc_pages", {}).items():
        mx = int(ids_ds[ids_ds.doc_id == d].page_number_in_doc.max())
        if mx != n - 1:
            raise DataValidationError(f"{d}: max page_number_in_doc {mx} != {n - 1}")

    all_ids = sorted(int(c) for c in ids_ds.corpus_id)
    selection = {"mode": cfg.mode, "excluded_query_rows": int(len(q_all) - len(human)),
                 "all_human_english_query_ids": sorted(int(x) for x in human.query_id)}
    if cfg.mode == "smoke":
        qids, cids, extra = select_smoke(human.query_id, qrels, all_ids, cfg)
        human = human[human.query_id.isin(qids)]
        qrels = qrels[qrels.query_id.isin(qids)]
        selection.update(extra, smoke_query_ids=qids,
                         note="qrel-aware engineering fixture (positives retained); scores NOT comparable to full")
    else:
        cids = all_ids
    selection["eval_query_ids"] = sorted(int(x) for x in human.query_id)
    selection["corpus_ids"] = cids

    row_of = {int(c): i for i, c in enumerate(ids_ds.corpus_id)}
    return Bundle(
        queries_all=q_all, queries=human.sort_values("query_id").reset_index(drop=True),
        qrels=qrels.reset_index(drop=True), meta=meta, corpus_ids=cids, row_of=row_of,
        doc_of={int(c): d for c, d in zip(ids_ds.corpus_id, ids_ds.doc_id)},
        page_of={int(c): int(p) for c, p in zip(ids_ds.corpus_id, ids_ds.page_number_in_doc)},
        corpus_ds=corpus.select_columns(["image"]), audit=audit, selection=selection)


def qrels_by_query(qrels: pd.DataFrame) -> dict:
    """query_id -> {corpus_id: grade}"""
    out: dict = {}
    for r in qrels.itertuples(index=False):
        out.setdefault(int(r.query_id), {})[int(r.corpus_id)] = int(r.score)
    return out
