"""The `Domain` adapter interface + hard-constraint plumbing."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from medusa.contract.interface import Task
from medusa.data.build import Dataset


@dataclasses.dataclass(frozen=True, slots=True)
class ConstraintResult:
    ok: bool
    name: str
    detail: str = ""


# A constraint is a hard check on a twin's predicted trajectory (all scored + exogenous
# channels, fit+holdout concatenated). It encodes things that must be true by identity
# -- accounting balances, conservation laws, non-negativity -- so they cost no tuning.
Constraint = Callable[[dict[str, np.ndarray], Dataset], ConstraintResult]


@dataclasses.dataclass(frozen=True, slots=True)
class Domain:
    name: str
    task: Task
    build: Callable[..., Dataset]        # (*, fit_frac, processed_dir, datasheet_path)
    system_prompt: str                   # may contain "{twin_runtime_budget_s}"
    blurb: str = ""
    kind: str = "synthetic"              # synthetic | real | business
    constraints: tuple[Constraint, ...] = ()

    def check_constraints(
        self, predicted: dict[str, np.ndarray], dataset: Dataset
    ) -> ConstraintResult | None:
        """First failing constraint, or None if all pass / none defined."""
        for c in self.constraints:
            try:
                r = c(predicted, dataset)
            except Exception as exc:  # a constraint that errors = not satisfiable
                return ConstraintResult(False, getattr(c, "__name__", "constraint"),
                                        f"raised {exc!r}")
            if not r.ok:
                return r
        return None


# --- reusable constraint factories -------------------------------------------------


def non_negative(*channels: str) -> Constraint:
    def _c(pred, _ds):
        for ch in channels:
            v = pred.get(ch)
            if v is not None and np.min(v) < -1e-6:
                return ConstraintResult(False, f"non_negative[{ch}]",
                                        f"{ch} goes to {float(np.min(v)):.3g}")
        return ConstraintResult(True, "non_negative")
    _c.__name__ = "non_negative"
    return _c


def flow_balance(
    stock: str, inflow: str, outflow: str | None = None, *, rel_tol: float = 0.03,
    abs_tol: float = 0.0, extra_inflows: tuple[str, ...] = (),
) -> Constraint:
    """stock[t] == stock[t-1] + inflow[t] (- outflow[t]) (+ extra_inflows[t]), within tol.

    The canonical business identity: a cash balance is last month's cash plus net income
    plus capital raised; a customer count is last month's plus gross adds minus churn.
    """

    def _c(pred, _ds):
        s = pred.get(stock)
        i = pred.get(inflow)
        if s is None or i is None:
            return ConstraintResult(True, f"flow_balance[{stock}]", "channel absent")
        o = pred.get(outflow) if outflow else np.zeros_like(s)
        add = sum(pred.get(k, np.zeros_like(s)) for k in extra_inflows)
        expected = s[:-1] + i[1:] - (o[1:] if o is not None else 0.0) + (
            add[1:] if np.ndim(add) else 0.0
        )
        err = np.abs(s[1:] - expected)
        tol = rel_tol * np.abs(s[:-1]) + abs_tol
        bad = np.where(err > tol)[0]
        if bad.size:
            k = int(bad[0]) + 1
            return ConstraintResult(
                False, f"flow_balance[{stock}]",
                f"at t={k}: {stock}={s[k]:.3g} but {stock}[t-1]+{inflow}"
                f"{'-'+outflow if outflow else ''} = {expected[bad[0]]:.3g}",
            )
        return ConstraintResult(True, f"flow_balance[{stock}]")

    _c.__name__ = f"flow_balance[{stock}]"
    return _c


def ratio_within(numer: str, denom: str, lo: float, hi: float) -> Constraint:
    def _c(pred, _ds):
        a, b = pred.get(numer), pred.get(denom)
        if a is None or b is None:
            return ConstraintResult(True, f"ratio[{numer}/{denom}]", "channel absent")
        r = a / np.clip(b, 1e-9, None)
        if np.any(r < lo) or np.any(r > hi):
            return ConstraintResult(
                False, f"ratio[{numer}/{denom}]",
                f"{numer}/{denom} ranges {float(r.min()):.2f}..{float(r.max()):.2f}, "
                f"outside [{lo}, {hi}]",
            )
        return ConstraintResult(True, f"ratio[{numer}/{denom}]")

    _c.__name__ = f"ratio[{numer}/{denom}]"
    return _c


def bounded_step(channel: str, max_rel_change: float) -> Constraint:
    def _c(pred, _ds):
        v = pred.get(channel)
        if v is None or len(v) < 2:
            return ConstraintResult(True, f"bounded_step[{channel}]")
        rel = np.abs(np.diff(v)) / np.clip(np.abs(v[:-1]), 1e-9, None)
        if np.max(rel) > max_rel_change:
            k = int(np.argmax(rel)) + 1
            return ConstraintResult(
                False, f"bounded_step[{channel}]",
                f"{channel} jumps {float(rel.max())*100:.0f}% at t={k} "
                f"(cap {max_rel_change*100:.0f}%)",
            )
        return ConstraintResult(True, f"bounded_step[{channel}]")

    _c.__name__ = f"bounded_step[{channel}]"
    return _c
