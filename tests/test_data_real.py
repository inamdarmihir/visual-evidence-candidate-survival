import pandas as pd
import pytest

from vesk.config import EXPECTED, RunConfig
from vesk.data import DataValidationError, assert_pinned_source, qrels_by_query, query_audit, select_smoke


def test_pinned_invariants_hold(tables):
    assert_pinned_source(tables["queries"], tables["qrels"], tables["meta"], len(tables["corpus_ids"]))


def test_audit_counts(tables):
    a = query_audit(tables["queries"])
    used = a[a.used_for_primary_evaluation]
    assert used.n_rows.tolist() == [110]
    assert a.n_rows.sum() == 1290
    eng = a[a.language == "english"].set_index("query_generator").n_rows
    assert eng["human"] == 110 and eng["sdg"] == 105


def test_stops_if_sdg_were_counted_as_human(tables):
    q = tables["queries"].copy()
    q.loc[(q.language == "english") & (q.query_generator == "sdg"), "query_generator"] = "human"   # relabelling attack on REAL rows
    with pytest.raises(DataValidationError):
        assert_pinned_source(q, tables["qrels"], tables["meta"], len(tables["corpus_ids"]))


def test_corpus_ids_not_assumed_to_be_row_offsets(tables):
    ids = tables["corpus_ids"].corpus_id.tolist()
    assert ids.count(ids[0]) == 1 and len(set(ids)) == len(ids)   # unique; offset mapping is stored explicitly


def test_smoke_selection_deterministic_and_blind(tables):
    human = tables["queries"].query(" language=='english' and query_generator=='human' ")
    qrels = tables["qrels"][tables["qrels"].query_id.isin(human.query_id)]
    cfg = RunConfig(mode="smoke")
    a = select_smoke(human.query_id, qrels, tables["corpus_ids"].corpus_id, cfg)
    b = select_smoke(human.query_id, qrels, tables["corpus_ids"].corpus_id, cfg)
    assert a == b
    qids, cids, extra = a
    assert len(qids) == 8 and len(cids) == 256 == len(set(cids))
    g = qrels_by_query(qrels[qrels.query_id.isin(qids)])
    assert {c for v in g.values() for c in v} <= set(cids)           # all positives retained
    assert not set(extra["filler_pages"]) & set(extra["positive_pages"])


def test_real_label_properties(tables):
    r = tables["qrels"]
    h = tables["queries"].query("language=='english' and query_generator=='human'")
    r = r[r.query_id.isin(h.query_id)]
    assert r.score.isin([1, 2]).all() and len(r) == EXPECTED["n_qrels_positive"]
