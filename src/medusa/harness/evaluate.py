"""Evaluate one candidate `twin.py` against a dataset. Deterministic; no LLM calls.

Runs in-process. `sandbox.py` calls this inside a subprocess for isolation + hard
timeout; tests and the reference-twin baseline call it directly.
"""

from __future__ import annotations

import dataclasses
import time
import traceback

import numpy as np

from medusa.config import LoopConfig
from medusa.contract.checker import check_twin_object, check_twin_source
from medusa.contract.interface import Observations
from medusa.data.build import Dataset
from medusa.harness import metrics

PLAUSIBILITY_FLOOR = 0.25  # below this, a candidate is disqualified from the portfolio
DEFAULT_TD_PLAUSIBLE_H = (0.15, 6.0)  # fallback doubling-time window when GT is unknown


@dataclasses.dataclass(slots=True)
class EvalResult:
    family: str | None = None
    params: dict | None = None
    n_params: int = 0
    passed_checks: bool = False
    crashed: bool = False
    nondeterministic: bool = False
    over_budget: bool = False
    error: str | None = None
    checker_report: str = ""
    runtime_s: float = 0.0
    metrics: dict = dataclasses.field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return bool(
            self.passed_checks
            and not self.crashed
            and not self.nondeterministic
            and not self.over_budget
            and self.metrics.get("plausibility", 0.0) >= PLAUSIBILITY_FLOOR
            and np.isfinite(self.metrics.get("holdout_smape", np.inf))
        )

    @property
    def score(self) -> float:
        """Ranking scalar, lower is better."""
        return self.metrics.get("holdout_smape", float("inf")) if self.is_valid else float("inf")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.params is not None:
            d["params"] = {k: float(v) for k, v in self.params.items()}
        d["metrics"] = {k: float(v) for k, v in self.metrics.items()}
        d["is_valid"] = self.is_valid
        d["score"] = self.score
        return d


def _load_twin(source: str):
    ns: dict = {}
    exec(compile(source, "twin.py", "exec"), ns)  # noqa: S102 - sandboxed candidate code
    twin_cls = ns.get("Twin")
    if twin_cls is None:
        raise RuntimeError("twin.py defines no `Twin` after exec")
    return twin_cls()


def evaluate_source(source: str, dataset: Dataset, cfg: LoopConfig) -> EvalResult:
    res = EvalResult()

    static = check_twin_source(source)
    res.checker_report = static.as_report()
    if not static.ok:
        res.error = "static checks failed"
        return res

    try:
        twin = _load_twin(source)
    except Exception:
        res.crashed = True
        res.error = f"import/instantiation error:\n{traceback.format_exc(limit=6)}"
        return res

    res.family = str(getattr(twin, "FAMILY", "") or "").strip() or None
    obj_check = check_twin_object(twin)
    res.checker_report = static.merge(obj_check).as_report()
    if not obj_check.ok:
        res.error = "twin object failed contract checks"
        return res
    res.n_params = len(getattr(twin, "PARAMS", {}))

    fit_obs, holdout_obs = dataset.fit, dataset.holdout
    t0 = time.perf_counter()
    try:
        params = dict(twin.fit(fit_obs))
        yhat_fit = np.asarray(twin.predict(params, fit_obs.time_s), dtype=float)
        yhat_hold = np.asarray(twin.predict(params, holdout_obs.time_s), dtype=float)
        yhat_hold_again = np.asarray(twin.predict(params, holdout_obs.time_s), dtype=float)
    except Exception:
        res.crashed = True
        res.error = f"fit/predict raised:\n{traceback.format_exc(limit=6)}"
        res.runtime_s = time.perf_counter() - t0
        return res
    res.runtime_s = time.perf_counter() - t0
    res.over_budget = res.runtime_s > cfg.twin_runtime_budget_s

    param_check = check_twin_object(twin, params=params)
    res.params = params
    if not param_check.ok:
        res.checker_report = static.merge(obj_check).merge(param_check).as_report()
        res.error = "fitted params violate the contract"
        return res

    if (
        yhat_hold.shape != holdout_obs.population_count.shape
        or yhat_fit.shape != fit_obs.population_count.shape
        or not np.all(np.isfinite(yhat_hold))
        or not np.all(np.isfinite(yhat_fit))
    ):
        res.crashed = True
        res.error = "predict returned wrong shape or non-finite values"
        return res

    res.nondeterministic = not np.allclose(yhat_hold, yhat_hold_again, rtol=0, atol=0)
    res.passed_checks = True

    gt = dataset.split.get("ground_truth", {}) or {}
    td_lo, td_hi = gt.get("td_plausible_h", DEFAULT_TD_PLAUSIBLE_H)
    implied_td = metrics.implied_doubling_time_h(twin.predict, params)
    fit_resid_log = np.log(np.clip(yhat_fit, 1e-9, None)) - np.log(
        np.clip(fit_obs.population_count, 1e-9, None)
    )

    res.metrics = {
        "holdout_smape": metrics.smape(holdout_obs.population_count, yhat_hold),
        "holdout_mase": metrics.mase(
            holdout_obs.population_count, yhat_hold, fit_obs.population_count
        ),
        "holdout_log_rmse": metrics.log_rmse(holdout_obs.population_count, yhat_hold),
        "fit_r2": metrics.r2(fit_obs.population_count, yhat_fit),
        "fit_log_rmse": metrics.log_rmse(fit_obs.population_count, yhat_fit),
        "aic": metrics.aic_from_residuals(fit_resid_log, res.n_params),
        "implied_doubling_h": implied_td,
        "plausibility": metrics.plausibility(implied_td, td_lo, td_hi),
    }
    return res
