```python
import numpy as np
from scipy.optimize import least_squares


class Twin:
    FAMILY = "gompertz"
    PARAMS = {
        "n0": (1.0, 1e5, "cells"),
        "a": (1.0, 25.0, "ln(K/n0)"),
        "mu_max": (0.05, 4.0, "1/h"),
        "lag": (0.0, 24.0, "h"),
    }
    METADATA = {"assumptions": ["well-mixed", "sigmoidal log-growth", "explicit lag"],
               "state_vars": ["N"], "refs": ["Gompertz 1825", "Zwietering 1990"]}

    @staticmethod
    def _curve(t_h, n0, a, mu, lag):
        e = np.e
        expo = a * np.exp(-np.exp((mu * e / max(a, 1e-6)) * (lag - t_h) + 1.0))
        return n0 * np.exp(expo)

    def fit(self, obs):
        t = np.asarray(obs.time_s, float) / 3600.0
        y = np.clip(np.asarray(obs.population_count, float), 1e-9, None)
        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])
        p0 = np.array([min(max(y[0], lo[0]), hi[0]),
                       float(np.clip(np.log(y.max() / max(y[0], 1e-9)), lo[1], hi[1])),
                       0.5, 0.5])
        r = least_squares(
            lambda p: np.log(np.clip(self._curve(t, *p), 1e-9, None)) - np.log(y),
            p0, bounds=(lo, hi), max_nfev=5000)
        v = np.clip(r.x, lo, hi)
        return dict(zip(self.PARAMS, (float(x) for x in v)))

    def predict(self, params, time_s):
        t = np.asarray(time_s, float) / 3600.0
        return self._curve(t, params["n0"], params["a"], params["mu_max"], params["lag"])
```