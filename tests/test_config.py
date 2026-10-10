import pytest

from vesk.config import PRESETS, RunConfig, corpus_point_id, query_point_id


def test_run_id_deterministic_and_preset_sensitive():
    a, b = RunConfig(mode="full"), RunConfig(mode="full")
    assert a.run_id() == b.run_id()
    assert a.run_id() != RunConfig(mode="full", preset="optional_2048").run_id()
    assert a.run_id() != RunConfig(mode="smoke").run_id()
    assert a.run_id() != RunConfig(mode="full", run_tag="again").run_id()


def test_primary_preset_is_1024_and_default_is_documented_2048():
    assert RunConfig().max_side == 1024
    assert PRESETS["optional_2048"]["max_side"] == 2048


def test_float16_never_silent():
    with pytest.raises(ValueError):
        RunConfig(dtype="float16")
    with pytest.raises(ValueError):
        RunConfig(dtype="bfloat16")


def test_budget_validation():
    with pytest.raises(ValueError):
        RunConfig(budgets=(25, 10))
    with pytest.raises(ValueError):
        RunConfig(budgets=(5,), ks=(5, 10))
    assert RunConfig().budgets == (10, 25, 50, 100) and RunConfig().ks == (5, 10)


def test_uuid5_ids_stable_and_namespaced():
    assert corpus_point_id(837) == corpus_point_id(837)
    assert corpus_point_id(5) != query_point_id(5)       # same integer, different namespace
    assert corpus_point_id(1) != corpus_point_id(2)


def test_bad_mode_and_preset():
    with pytest.raises(ValueError):
        RunConfig(mode="other")
    with pytest.raises(ValueError):
        RunConfig(preset="nope")
