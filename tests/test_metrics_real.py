"""Metric checks built from REAL qrels: rankings are constructed from the real labels, never invented pages."""
import math

import pytest

from vesk.data import qrels_by_query
from vesk.metrics import loss_label, ndcg_at_k, survival


@pytest.fixture(scope="module")
def grades(tables):
    h = tables["queries"].query("language=='english' and query_generator=='human'")
    return qrels_by_query(tables["qrels"][tables["qrels"].query_id.isin(h.query_id)])


def test_perfect_ranking(grades):
    for g in grades.values():
        ideal = sorted(g, key=lambda c: (-g[c], c))
        s = survival(g, ideal)
        assert s["recall"] == 1 and s["coverage"] == 1
        assert ndcg_at_k(ideal, g, 10) == pytest.approx(1.0)


def test_empty_candidates_and_not_applicable_nan(grades):
    qid = next(q for q, g in grades.items() if all(v == 2 for v in g.values()))   # a real query with no score-1 page
    s = survival(grades[qid], [])
    assert s["recall"] == 0 and s["coverage"] == 0
    assert math.isnan(s["recall_s1"]) and s["n_s1"] == 0          # not computed, not zero


def test_partial_survival_multi_page(grades):
    qid = next(q for q, g in grades.items() if len(g) > 1)
    g = grades[qid]
    first = sorted(g)[0]
    s = survival(g, [first])
    assert s["recall"] == pytest.approx(1 / len(g)) and s["coverage"] == 0


def test_gain_convention_2_pow_score_minus_1(grades):
    qid = next(q for q, g in grades.items() if set(g.values()) == {1, 2})
    g = grades[qid]
    s2 = next(c for c, v in g.items() if v == 2)
    s1 = next(c for c, v in g.items() if v == 1)
    assert ndcg_at_k([s2], g, 1) > ndcg_at_k([s1], g, 1)        # gain 3 vs 1 at rank 1
    assert ndcg_at_k([s1], g, 1) == pytest.approx(1 / 3)


def test_loss_labels():
    assert loss_label(False, None, 10) == "candidate_loss"
    assert loss_label(True, 11, 10) == "reranking_loss"
    assert loss_label(True, 10, 10) == "survived_top_k"
