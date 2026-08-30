from medusa.harness.archive import Archive
from medusa.harness.evaluate import EvalResult


def _valid(family: str, smape: float) -> EvalResult:
    return EvalResult(
        family=family,
        params={"a": 1.0},
        n_params=1,
        passed_checks=True,
        metrics={"holdout_smape": smape, "plausibility": 1.0},
    )


def _crashed() -> EvalResult:
    return EvalResult(family="broken", crashed=True, error="boom")


def test_best_per_family_keeps_lowest_smape():
    arc = Archive()
    arc.add(1, _valid("logistic", 0.5))
    arc.add(2, _valid("logistic", 0.2))
    arc.add(3, _valid("gompertz", 0.3))
    arc.add(4, _crashed())

    best = arc.best_per_family()
    assert set(best) == {"logistic", "gompertz"}
    assert best["logistic"].score == 0.2
    assert arc.distinct_valid_families == 2


def test_portfolio_is_ranked_and_capped():
    arc = Archive()
    for i, (fam, s) in enumerate(
        [("a", 0.4), ("b", 0.1), ("c", 0.3), ("d", 0.2)], start=1
    ):
        arc.add(i, _valid(fam, s))
    pf = arc.portfolio(k=3)
    assert [e.family for e in pf] == ["b", "d", "c"]


def test_summary_text_mentions_best_score():
    arc = Archive()
    arc.add(1, _valid("logistic", 0.25))
    txt = arc.summary_text()
    assert "logistic" in txt and "0.25" in txt
