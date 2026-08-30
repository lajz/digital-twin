"""The data container passed to a twin, and the constants describing the twin contract."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

REQUIRED_TWIN_ATTRS = ("FAMILY", "PARAMS", "fit", "predict")
PARAM_TRIPLE_LEN = 3  # (prior_low, prior_high, unit)

# Import allowlist for generated twin source (top-level module names).
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "numpy",
        "scipy",
        "math",
        "cmath",
        "statistics",
        "itertools",
        "functools",
        "collections",
        "dataclasses",
        "typing",
        "random",
        "warnings",
        "operator",
        "copy",
        "bisect",
        "heapq",
        "__future__",
    }
)
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "sklearn",
        "torch",
        "tensorflow",
        "jax",
        "keras",
        "statsmodels",
        "pandas",
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "pathlib",
        "pickle",
        "importlib",
    }
)


@dataclasses.dataclass(slots=True)
class Observations:
    """A well-mixed population growth signal. Times in seconds since the first frame."""

    time_s: np.ndarray
    population_count: np.ndarray
    total_area_um2: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.population_count = np.asarray(self.population_count, dtype=float)
        if self.total_area_um2 is not None:
            self.total_area_um2 = np.asarray(self.total_area_um2, dtype=float)
        if self.time_s.shape != self.population_count.shape:
            raise ValueError("time_s and population_count must have the same shape")
        if self.time_s.ndim != 1:
            raise ValueError("Observations must be 1-D")
        order = np.argsort(self.time_s)
        self.time_s = self.time_s[order]
        self.population_count = self.population_count[order]
        if self.total_area_um2 is not None:
            self.total_area_um2 = self.total_area_um2[order]

    def __len__(self) -> int:
        return int(self.time_s.size)

    @property
    def time_h(self) -> np.ndarray:
        return self.time_s / 3600.0

    def slice_time(self, t0: float, t1: float) -> "Observations":
        """Rows with t0 <= time_s < t1."""
        mask = (self.time_s >= t0) & (self.time_s < t1)
        return Observations(
            time_s=self.time_s[mask],
            population_count=self.population_count[mask],
            total_area_um2=None if self.total_area_um2 is None else self.total_area_um2[mask],
        )

    def to_parquet(self, path: str | Path) -> None:
        import pandas as pd

        cols = {"time_s": self.time_s, "population_count": self.population_count}
        if self.total_area_um2 is not None:
            cols["total_area_um2"] = self.total_area_um2
        pd.DataFrame(cols).to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "Observations":
        import pandas as pd

        df = pd.read_parquet(path)
        return cls(
            time_s=df["time_s"].to_numpy(dtype=float),
            population_count=df["population_count"].to_numpy(dtype=float),
            total_area_um2=(
                df["total_area_um2"].to_numpy(dtype=float)
                if "total_area_um2" in df.columns
                else None
            ),
        )
