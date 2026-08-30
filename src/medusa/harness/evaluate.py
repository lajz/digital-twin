"""Evaluate one candidate `twin.py` against a dataset. Deterministic; no LLM calls.

Dispatches on `dataset.task.mode`:
  - "series"  : twin.predict(params, times) -> array | {channel: array}
  - "spatial" : twin.simulate(params, init_cells, times, seed) -> [ {x,y,angle,length,width} ]
                reduced to spatial summary-statistic series, ensembled over seeds.

Runs in-process. `sandbox.py` calls this inside a subprocess for isolation + hard timeout.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
import types

import numpy as np

from medusa.config import LoopConfig
from medusa.contract.checker import check_twin_object, check_twin_source
from medusa.data.build import Dataset
from medusa.harness import metrics

PLAUSIBILITY_FLOOR = 0.25
DEFAULT_TD_PLAUSIBLE_H = (0.15, 6.0)
SPATIAL_SEEDS = (0, 1)
SPATIAL_DETERMINISM_FRAMES = 6  # compare only a short prefix for the determinism check


@dataclasses.dataclass(slots=True)
class EvalResult:
    family: str | None = None
    params: dict | None = None
    n_params: int = 0
    mode: str = "series"
    passed_checks: bool = False
    crashed: bool = False
    nondeterministic: bool = False
    over_budget: bool = False
    error: str | None = None
    checker_report: str = ""
    runtime_s: float = 0.0
    metrics: dict = dataclasses.field(default_factory=dict)
    per_observable: dict = dataclasses.field(default_factory=dict)

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
        return self.metrics.get("holdout_smape", float("inf")) if self.is_valid else float("inf")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.params is not None:
            d["params"] = {k: float(v) for k, v in self.params.items()}
        d["metrics"] = {k: float(v) for k, v in self.metrics.items()}
        d["per_observable"] = {k: float(v) for k, v in self.per_observable.items()}
        d["is_valid"] = self.is_valid
        d["score"] = self.score
        return d


def spatial_rollout(source: str, params: dict, dataset: Dataset, seed: int = 0):
    """Re-run a spatial twin in-process to get its rollout (for rendering, not scoring)."""
    from medusa.spatial.frames import SpatialFrames, rollout_to_frames

    twin = _load_twin(source)
    real = SpatialFrames.from_npz(dataset.frames_npz)
    init = {k: np.array(v) for k, v in real.frames[0].as_dict().items()}
    raw = twin.simulate(params, init, np.asarray(real.time_s, dtype=float), int(seed))
    return real, rollout_to_frames(raw, real.time_s)


def _load_twin(source: str):
    ns: dict = {}
    exec(compile(source, "twin.py", "exec"), ns)  # noqa: S102 - sandboxed candidate code
    twin_cls = ns.get("Twin")
    if twin_cls is None:
        raise RuntimeError("twin.py defines no `Twin` after exec")
    return twin_cls()


def _normalize_prediction(raw, times_s: np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(raw, dict):
        return {k: np.asarray(v, dtype=float) for k, v in raw.items()}
    arr = np.asarray(raw, dtype=float)
    return {"population_count": arr}


def evaluate_source(source: str, dataset: Dataset, cfg: LoopConfig) -> EvalResult:
    task = dataset.task
    res = EvalResult(mode=task.mode)

    static = check_twin_source(source, required_methods=task.required_methods)
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
    obj_check = check_twin_object(twin, required_methods=task.required_methods)
    res.checker_report = static.merge(obj_check).as_report()
    if not obj_check.ok:
        res.error = "twin object failed contract checks"
        return res
    res.n_params = len(getattr(twin, "PARAMS", {}))

    try:
        if task.mode == "spatial":
            _eval_spatial(twin, dataset, cfg, res)
        else:
            _eval_series(twin, dataset, cfg, res)
    except Exception:
        res.crashed = True
        res.error = f"evaluation raised:\n{traceback.format_exc(limit=8)}"
    return res


# --- series (population / structured) -------------------------------------------


def _eval_series(twin, dataset: Dataset, cfg: LoopConfig, res: EvalResult) -> None:
    task = dataset.task
    fit_obs, holdout_obs = dataset.fit, dataset.holdout

    t0 = time.perf_counter()
    params = dict(twin.fit(fit_obs))
    pred_fit = _normalize_prediction(twin.predict(params, fit_obs.time_s), fit_obs.time_s)
    pred_hold = _normalize_prediction(
        twin.predict(params, holdout_obs.time_s), holdout_obs.time_s
    )
    pred_hold2 = _normalize_prediction(
        twin.predict(params, holdout_obs.time_s), holdout_obs.time_s
    )
    res.runtime_s = time.perf_counter() - t0
    res.over_budget = res.runtime_s > cfg.twin_runtime_budget_s

    param_check = check_twin_object(twin, params=params, required_methods=task.required_methods)
    res.params = params
    if not param_check.ok:
        res.checker_report += "\n" + param_check.as_report()
        res.error = "fitted params violate the contract"
        return

    res.nondeterministic = any(
        k not in pred_hold2 or not np.allclose(pred_hold[k], pred_hold2[k], atol=0, rtol=0)
        for k in pred_hold
    )

    combined, per = metrics.multi_smape(pred_hold, holdout_obs, task.observables, task)
    res.per_observable = per
    _finish_metrics(res, dataset, combined, per, pred_fit, fit_obs, pred_hold, holdout_obs)
    res.passed_checks = True


# --- spatial (agent-based rollout) ---------------------------------------------


def _spatial_view(frames):
    return types.SimpleNamespace(
        time_s=frames.time_s,
        frames=[f.as_dict() for f in frames.frames],
        n_per_frame=[len(f) for f in frames.frames],
    )


def _eval_spatial(twin, dataset: Dataset, cfg: LoopConfig, res: EvalResult) -> None:
    from medusa.spatial.frames import SpatialFrames, rollout_to_frames
    from medusa.spatial.summarize import summary_series

    if dataset.frames_npz is None:
        raise RuntimeError("spatial dataset has no frames.npz")
    real = SpatialFrames.from_npz(dataset.frames_npz)
    t_split = dataset.split["t_split_s"]
    i_split = real.index_at_or_before(t_split) + 1
    fit_frames = real.slice_frames(0, i_split)
    init_cells = real.frames[0].as_dict()
    all_times = real.time_s

    task = dataset.task
    t0 = time.perf_counter()
    params = dict(twin.fit(_spatial_view(fit_frames)))

    def _sim(seed, times):
        return rollout_to_frames(
            twin.simulate(params, {k: np.array(v) for k, v in init_cells.items()},
                          np.asarray(times, dtype=float), int(seed)),
            times,
        )

    rollouts = [_sim(s, all_times) for s in SPATIAL_SEEDS]
    res.runtime_s = time.perf_counter() - t0
    res.over_budget = res.runtime_s > cfg.spatial_runtime_budget_s

    # determinism: same seed -> identical rollout (short prefix only, for speed)
    prefix = all_times[:SPATIAL_DETERMINISM_FRAMES]
    res.nondeterministic = not _same_rollout(
        _sim(SPATIAL_SEEDS[0], prefix),
        SpatialFrames(prefix, rollouts[0].frames[: len(prefix)]),
    )

    param_check = check_twin_object(twin, params=params, required_methods=task.required_methods)
    res.params = params
    if not param_check.ok:
        res.checker_report += "\n" + param_check.as_report()
        res.error = "fitted params violate the contract"
        return

    if any(len(r) < len(all_times) for r in rollouts):
        raise RuntimeError(
            f"simulate returned {min(len(r) for r in rollouts)} frames, need {len(all_times)}"
        )

    f0 = rollouts[0].frames[0]
    if len(f0) != len(real.frames[0]) or not np.allclose(
        np.sort(f0.x), np.sort(real.frames[0].x), atol=1.0
    ):
        res.error = (
            "the first rollout frame must be the given initial condition unchanged "
            f"(got {len(f0)} cells at different positions vs {len(real.frames[0])} given)"
        )
        return

    # ensemble-mean summary series, then split
    ens = _mean_summary([summary_series(r) for r in rollouts])
    split = dataset.split
    from medusa.data.build import split_observations

    pred_fit, pred_hold = split_observations(ens, split)
    combined, per = metrics.multi_smape(
        pred_hold.channels(), dataset.holdout, task.observables, task
    )
    res.per_observable = per
    _finish_metrics(
        res, dataset, combined, per,
        pred_fit.channels(), dataset.fit, pred_hold.channels(), dataset.holdout,
    )
    res.metrics["n_cells_final"] = float(len(rollouts[0].frames[-1]))
    res.passed_checks = True


def _same_rollout(a, b) -> bool:
    if len(a) != len(b):
        return False
    for fa, fb in zip(a.frames, b.frames):
        if len(fa) != len(fb) or not np.allclose(fa.x, fb.x, atol=0, rtol=0):
            return False
    return True


def _mean_summary(series_list):
    from medusa.contract.interface import Observations

    ref = series_list[0]
    chans = {}
    for name, _ in ref.channels().items():
        stack = np.vstack([s.channel(name) for s in series_list])
        chans[name] = np.mean(stack, axis=0)
    pop = chans.pop("population_count")
    return Observations(time_s=ref.time_s, population_count=pop, extra=chans)


# --- shared metric finalisation ----------------------------------------------


def _finish_metrics(res, dataset, combined, per, pred_fit, fit_obs, pred_hold, holdout_obs):
    biomass_key = "total_length_um" if "total_length_um" in pred_fit else "population_count"
    full_t = np.concatenate([fit_obs.time_s, holdout_obs.time_s])
    full_y = np.concatenate([pred_fit[biomass_key], pred_hold[biomass_key]])
    implied_td = metrics.doubling_time_from_series_h(full_t, full_y)

    gt = dataset.split.get("ground_truth", {}) or {}
    td_lo, td_hi = gt.get("td_plausible_h") or dataset.task.plausibility.get(
        "doubling_time_h", DEFAULT_TD_PLAUSIBLE_H
    )
    plaus = metrics.plausibility(implied_td, td_lo, td_hi)

    if "mean_length_um" in dataset.task.plausibility and "total_length_um" in pred_hold:
        ml = pred_hold["total_length_um"] / np.clip(pred_hold["population_count"], 1e-9, None)
        lo, hi = dataset.task.plausibility["mean_length_um"]
        plaus = min(plaus, metrics.plausibility(float(np.mean(ml)), lo, hi))

    pop_fit = pred_fit.get("population_count")
    fit_r2 = metrics.r2(fit_obs.population_count, pop_fit) if pop_fit is not None else float("nan")

    res.metrics.update(
        {
            "holdout_smape": combined,
            "holdout_smape_population": per.get("population_count", float("nan")),
            "fit_r2": fit_r2,
            "implied_doubling_h": implied_td,
            "plausibility": plaus,
        }
    )
