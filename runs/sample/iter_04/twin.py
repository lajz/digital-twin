import numpy as np
from scipy.optimize import least_squares


class Twin:
    FAMILY = "richards"
    PARAMS = {
        "carrying_capacity": (1e2, 1e9, "cells"),
        "nu": (0.05, 8.0, "shape"),
        "mu_max": (0.05, 4.0, "1/h"),
        "t_infl": (0.0, 48.0, "h"),
    }
    METADATA = {"assumptions": ["well-mixed", "asymmetric sigmoid via shape nu"],
               "state_vars": ["N"], "refs": ["Richards 1959"]}

    @staticmethod
    def _curve(t_h, k, nu, mu, t_infl):
        nu = max(nu, 1e-3)
        z = np.clip(-mu * (t_h - t_infl), -50, 50)
        return k / np.power(1.0 + nu * np.exp(z), 1.0 / nu)

    def fit(self, obs):
        t = np.asarray(obs.time_s, float) / 3600.0
        y = np.clip(np.asarray(obs.population_count, float), 1e-9, None)
        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])
        p0 = np.array([min(max(y.max() * 1.5, lo[0]), hi[0]), 1.0, 0.5,
                       float(np.clip(t.mean(), lo[3], hi[3]))])
        r = least_squares(
            lambda p: np.log(np.clip(self._curve(t, *p), 1e-9, None)) - np.log(y),
            p0, bounds=(lo, hi), max_nfev=5000)
        v = np.clip(r.x, lo, hi)
        return dict(zip(self.PARAMS, (float(x) for x in v)))

    def predict(self, params, time_s):
        t = np.asarray(time_s, float) / 3600.0
        return self._curve(t, params["carrying_capacity"], params["nu"],
                           params["mu_max"], params["t_infl"])