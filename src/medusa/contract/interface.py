"""The data container passed to a twin, the task definition, and contract constants."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

REQUIRED_TWIN_ATTRS = ("FAMILY", "PARAMS", "fit", "predict")
PARAM_TRIPLE_LEN = 3  # (prior_low, prior_high, unit)

# Import allowlist for generated twin source (top-level module names).
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "numpy", "scipy", "math", "cmath", "statistics", "itertools", "functools",
        "collections", "dataclasses", "typing", "random", "warnings", "operator",
        "copy", "bisect", "heapq", "__future__",
    }
)
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "sklearn", "torch", "tensorflow", "jax", "keras", "statsmodels", "pandas",
        "os", "sys", "subprocess", "socket", "urllib", "requests", "httpx",
        "pathlib", "pickle", "importlib",
    }
)


@dataclasses.dataclass(slots=True)
class Observations:
    """A population growth signal as aligned 1-D time series.

    `time_s` and `population_count` are always present. Any other channel
    (`total_area_um2`, `total_length_um`, `mean_length_um`, `length_cv`,
    `colony_radius_um`, ...) rides in `extra` and is also reachable as an attribute,
    e.g. `obs.mean_length_um`.
    """

    time_s: np.ndarray
    population_count: np.ndarray
    extra: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.population_count = np.asarray(self.population_count, dtype=float)
        if self.time_s.ndim != 1 or self.time_s.shape != self.population_count.shape:
            raise ValueError("time_s and population_count must be matching 1-D arrays")
        order = np.argsort(self.time_s)
        self.time_s = self.time_s[order]
        self.population_count = self.population_count[order]
        clean = {}
        for k, v in self.extra.items():
            v = np.asarray(v, dtype=float)
            if v.shape != self.time_s.shape:
                raise ValueError(f"channel {k!r} shape mismatch")
            clean[k] = v[order]
        self.extra = clean

    def __len__(self) -> int:
        return int(self.time_s.size)

    def __getattr__(self, name: str) -> np.ndarray:
        # called only when normal lookup fails; `extra` is a real slot -> no recursion
        try:
            return object.__getattribute__(self, "extra")[name]
        except (AttributeError, KeyError) as exc:
            raise AttributeError(name) from exc

    @property
    def time_h(self) -> np.ndarray:
        return self.time_s / 3600.0

    def channel(self, name: str) -> np.ndarray:
        if name == "population_count":
            return self.population_count
        if name in self.extra:
            return self.extra[name]
        raise KeyError(f"observation channel {name!r} is not present")

    def channels(self) -> dict[str, np.ndarray]:
        return {"population_count": self.population_count, **self.extra}

    def slice_time(self, t0: float, t1: float) -> "Observations":
        mask = (self.time_s >= t0) & (self.time_s < t1)
        return Observations(
            time_s=self.time_s[mask],
            population_count=self.population_count[mask],
            extra={k: v[mask] for k, v in self.extra.items()},
        )

    def to_parquet(self, path: str | Path) -> None:
        import pandas as pd

        pd.DataFrame({"time_s": self.time_s, **self.channels()}).to_parquet(
            path, index=False
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> "Observations":
        import pandas as pd

        df = pd.read_parquet(path)
        extra = {
            c: df[c].to_numpy(dtype=float)
            for c in df.columns
            if c not in ("time_s", "population_count")
        }
        return cls(
            time_s=df["time_s"].to_numpy(dtype=float),
            population_count=df["population_count"].to_numpy(dtype=float),
            extra=extra,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """What a twin is scored on: which channels, their weights, the plausibility
    windows on the model's implied biology, and how the twin produces predictions."""

    name: str
    observables: tuple[str, ...] = ("population_count",)
    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    plausibility: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    mode: str = "series"  # "series" (twin.predict) | "spatial" (twin.simulate)
    # channels that are inputs, not scored outputs -- provided to the twin over the
    # full horizon (e.g. a business's planned marketing spend, capital raised)
    exogenous: tuple[str, ...] = ()
    # for plotting: the natural period and its label
    period_s: float = 3600.0
    period_label: str = "h"

    def weight(self, observable: str) -> float:
        return float(self.weights.get(observable, 1.0))

    @property
    def required_methods(self) -> tuple[str, ...]:
        return ("fit", "simulate") if self.mode == "spatial" else ("fit", "predict")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "observables": list(self.observables),
            "weights": dict(self.weights),
            "plausibility": {k: list(v) for k, v in self.plausibility.items()},
            "mode": self.mode,
            "exogenous": list(self.exogenous),
            "period_s": self.period_s,
            "period_label": self.period_label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            name=d["name"],
            observables=tuple(d.get("observables", ("population_count",))),
            weights=dict(d.get("weights", {})),
            plausibility={k: list(v) for k, v in d.get("plausibility", {}).items()},
            mode=d.get("mode", "series"),
            exogenous=tuple(d.get("exogenous", ())),
            period_s=d.get("period_s", 3600.0),
            period_label=d.get("period_label", "h"),
        )


POPULATION_TASK = Task(
    name="population",
    observables=("population_count",),
    plausibility={"doubling_time_h": [0.15, 6.0]},
)

STRUCTURED_TASK = Task(
    name="structured",
    observables=("total_length_um", "population_count", "mean_length_um", "length_cv"),
    weights={"length_cv": 0.5},
    plausibility={
        "doubling_time_h": [0.15, 6.0],
        "mean_length_um": [1.0, 12.0],
        "length_cv": [0.05, 0.6],
    },
)

SPATIAL_TASK = Task(
    name="spatial",
    mode="spatial",
    observables=(
        "total_length_um", "population_count", "colony_radius_um",
        "colony_aspect", "nematic_order", "mean_nn_dist_um",
    ),
    weights={"colony_aspect": 0.5, "nematic_order": 0.5},
    plausibility={
        "doubling_time_h": [0.15, 6.0],
        "mean_length_um": [1.0, 12.0],
    },
)
