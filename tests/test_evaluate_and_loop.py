import numpy as np
import pytest

from medusa.agent import canned
from medusa.config import DEFAULT_LOOP_CONFIG
from medusa.contract import reference_twin
from medusa.data import build
from medusa.harness.evaluate import evaluate_source


@pytest.fixture
def dataset(tmp_path):
    return build.build_synthetic(
        "synthetic-ecoli-fast",
        processed_dir=tmp_path,
        datasheet_path=tmp_path / "datasheet.md",
    )


def _source_of(module) -> str:
    import inspect

    return inspect.getsource(module)


def test_reference_twin_is_valid_and_beats_threshold(dataset):
    src = _source_of(reference_twin)
    res = evaluate_source(src, dataset, DEFAULT_LOOP_CONFIG)
    assert res.is_valid, res.error or res.checker_report
    assert res.family == "logistic"
    assert res.metrics["holdout_smape"] < 0.20


@pytest.mark.parametrize("src", canned.CANNED_SEQUENCE, ids=lambda s: s.split('FAMILY = "')[1][:12])
def test_canned_twins_all_evaluate_valid(src, dataset):
    res = evaluate_source(src, dataset, DEFAULT_LOOP_CONFIG)
    assert res.passed_checks and not res.crashed, res.error
    assert np.isfinite(res.metrics["holdout_smape"])


def test_broken_twin_is_flagged_not_raised(dataset):
    broken = canned.LOGISTIC.replace("return k / (1.0", "return 1 / 0 + (1.0")
    res = evaluate_source(broken, dataset, DEFAULT_LOOP_CONFIG)
    assert res.crashed and not res.is_valid


def test_loop_dry_run_produces_portfolio_and_scorecard(tmp_path):
    dataset = build.build_synthetic(
        "synthetic-ecoli-fast",
        processed_dir=tmp_path / "processed",
        datasheet_path=tmp_path / "processed" / "datasheet.md",
    )
    from medusa.agent.loop import run_loop
    import dataclasses

    cfg = dataclasses.replace(DEFAULT_LOOP_CONFIG, max_iters=4, diversity_nudge_every=2)
    result = run_loop(
        dataset,
        cfg,
        dry_run=True,
        runs_dir=tmp_path / "runs",
        processed_dir=tmp_path / "processed",
    )
    assert (result.run_dir / "trace.jsonl").exists()
    assert (result.run_dir / "portfolio" / "portfolio.md").exists()
    assert (result.run_dir / "scorecard.json").exists()
    assert result.scorecard.distinct_plausible_families >= 2
    assert result.scorecard.n_iters == 4
