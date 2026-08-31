import dataclasses

import numpy as np
import pytest

from medusa import domains
from medusa.agent.canned import SAAS_BASS
from medusa.config import DEFAULT_LOOP_CONFIG
from medusa.data import datasets
from medusa.domains.base import flow_balance, non_negative
from medusa.harness.evaluate import evaluate_source


def test_registry_has_builtin_domains():
    names = {d.name for d in domains.list_domains()}
    assert {"saas-seed", "ipb-ecoli", "ipb-ecoli-spatial", "synthetic-ecoli-fast"} <= names
    saas = domains.get("saas-seed")
    assert saas.kind == "business" and saas.constraints


def test_datasets_registry_delegates_to_domains():
    assert "saas-seed" in datasets.list_datasets()
    assert datasets.is_real("saas-seed")  # business counts as non-synthetic


@pytest.fixture
def saas(tmp_path):
    return datasets.build_dataset(
        "saas-seed", processed_dir=tmp_path, datasheet_path=tmp_path / "d.md"
    )


def test_saas_data_cash_bridge_holds_exactly(saas):
    # net_income and cash are constructed consistently from the (noisy) mrr, so the
    # cash-flow identity holds in the data even though other channels carry noise.
    o = saas.observations
    bridge = o.cash[1:] - o.cash[:-1] - o.net_income[1:] - o.capital_raised[1:]
    assert np.abs(bridge).max() < 1.0
    assert o.capital_raised[12] > 1e6  # the Series A


def test_canned_saas_twin_is_valid_and_consistent(saas):
    res = evaluate_source(SAAS_BASS, saas, DEFAULT_LOOP_CONFIG)
    assert res.is_valid, res.error or res.constraint_violation
    assert res.constraint_violation is None
    assert res.per_observable["mrr"] < 0.25
    assert set(res.per_observable) == set(saas.task.observables)


def test_constraint_violation_rejects_a_twin(saas):
    bad = SAAS_BASS.replace(
        'cash[i] = (cash[i-1] if i else cash0) + ni[i] + capital[i]',
        'cash[i] = (cash[i-1] if i else cash0) * 1.05 + ni[i]',
    )
    res = evaluate_source(bad, saas, DEFAULT_LOOP_CONFIG)
    assert not res.is_valid
    assert res.constraint_violation and "cash" in res.constraint_violation


def test_flow_balance_and_non_negative_helpers():
    ds = None
    ok = {"s": np.array([100.0, 110, 121]), "i": np.array([0.0, 10, 11])}
    assert flow_balance("s", "i")(ok, ds).ok
    bad = {"s": np.array([100.0, 200, 300]), "i": np.array([0.0, 10, 10])}
    assert not flow_balance("s", "i")(bad, ds).ok
    assert not non_negative("x")({"x": np.array([1.0, -3.0])}, ds).ok


def test_exogenous_lever_changes_the_forecast(saas):
    from medusa.harness.evaluate import _load_twin

    twin = _load_twin(SAAS_BASS)
    params = twin.fit(saas.fit)
    t = saas.observations.time_s
    base = {"marketing_spend": saas.observations.marketing_spend,
            "capital_raised": saas.observations.capital_raised}
    hi = {**base, "marketing_spend": base["marketing_spend"] * 2}
    y0 = twin.predict(params, t, base)["customers"][-1]
    y1 = twin.predict(params, t, hi)["customers"][-1]
    assert y1 > y0 * 1.1  # doubling the acquisition budget grows the customer base
