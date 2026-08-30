import numpy as np
from scipy.optimize import least_squares


class Twin:
    FAMILY = "logistic"
    PARAMS = {
        "n0": (1.0, 1e5, "cells"),
        "mu_max": (0.05, 4.0, "1/h"),
        "carrying_capacity": (1e2, 1e9, "cells"),
    }
    METADATA = {"assumptions": ["well-mixed", "fixed carrying capacity"],
               "state_vars": ["N"], "refs": ["Verhulst 1838"]}

    @staticmethod
    def _curve(t_h, n0, mu, k):
        n0 = max(n0, 1e-9); k = max(k, n0 + 1e-9)
        return k / (1.0 + ((k - n0) / n0) * np.exp(-mu * t_h))

    def fit(self, obs):
        t = np.asarray(obs.time_s, float) / 3600.0
        y = np.clip(np.asarray(obs.population_count, float), 1e-9, None)
        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])
        p0 = np.array([min(max(y[0], lo[0]), hi[0]), 0.5,
                       min(max(y.max() * 1.5, lo[2]), hi[2])])
        r = least_squares(
            lambda p: np.log(np.clip(self._curve(t, *p), 1e-9, None)) - np.log(y),
            p0, bounds=(lo, hi), max_nfev=4000)
        v = np.clip(r.x, lo, hi)
        return {"n0": float(v[0]), "mu_max": float(v[1]), "carrying_capacity": float(v[2])}

    def predict(self, params, time_s):
        t = np.asarray(time_s, float) / 3600.0
        return self._curve(t, params["n0"], params["mu_max"], params["carrying_capacity"])