"""Hand-written logistic twin: the baseline the agent must beat, and a test fixture.

Deliberately written to the same contract as a generated `twin.py` so the harness scores
it by the identical path.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


class Twin:
    FAMILY = "logistic"

    PARAMS = {
        "n0": (1.0, 1.0e5, "cells"),
        "mu_max": (0.05, 4.0, "1/h"),
        "carrying_capacity": (1.0e2, 1.0e9, "cells"),
    }

    METADATA = {
        "assumptions": [
            "well-mixed population",
            "single limiting resource folded into a fixed carrying capacity",
            "per-capita growth rate declines linearly with density",
        ],
        "state_vars": ["N"],
        "refs": ["Verhulst 1838", "Pearl & Reed 1920"],
    }

    @staticmethod
    def _curve(t_h: np.ndarray, n0: float, mu_max: float, k: float) -> np.ndarray:
        n0 = max(n0, 1e-9)
        k = max(k, n0 + 1e-9)
        ratio = (k - n0) / n0
        return k / (1.0 + ratio * np.exp(-mu_max * t_h))

    def fit(self, obs) -> dict:
        t_h = np.asarray(obs.time_s, dtype=float) / 3600.0
        y = np.asarray(obs.population_count, dtype=float)
        y = np.clip(y, 1e-9, None)

        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])

        y_last = float(np.max(y))
        p0 = np.array(
            [
                min(max(float(y[0]), lo[0]), hi[0]),
                0.5,
                min(max(y_last * 1.5, lo[2]), hi[2]),
            ]
        )

        # weak prior: when K is not yet identifiable from a pre-saturation window,
        # keep it near the largest observed count rather than running to the bound.
        k_anchor = np.log(max(y_last * 2.0, 10.0))

        def resid(p: np.ndarray) -> np.ndarray:
            pred = self._curve(t_h, *p)
            data_term = np.log(np.clip(pred, 1e-9, None)) - np.log(y)
            reg = np.array([0.15 * (np.log(max(p[2], 1e-9)) - k_anchor)])
            return np.concatenate([data_term, reg])

        sol = least_squares(
            resid, p0, bounds=(lo, hi), method="trf", max_nfev=4000
        )
        n0, mu_max, k = (float(v) for v in np.clip(sol.x, lo, hi))
        return {"n0": n0, "mu_max": mu_max, "carrying_capacity": k}

    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        return self._curve(
            t_h,
            params["n0"],
            params["mu_max"],
            params["carrying_capacity"],
        )
