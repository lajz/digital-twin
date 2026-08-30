"""Canned `twin.py` sources for --dry-run and tests.

Each is a self-contained, contract-compliant mechanistic twin in a different family.
The dry-run client cycles through these instead of calling the DeepSeek API.
"""

from __future__ import annotations

LOGISTIC = '''
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
'''

GOMPERTZ = '''
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
'''

BARANYI_ODE = '''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


class Twin:
    FAMILY = "baranyi-ode"
    PARAMS = {
        "n0": (1.0, 1e5, "cells"),
        "q0": (1e-4, 10.0, "dimensionless"),
        "mu_max": (0.05, 4.0, "1/h"),
        "nmax": (1e2, 1e9, "cells"),
    }
    METADATA = {"assumptions": ["well-mixed", "physiological-state lag variable Q",
                                "logistic ceiling"],
               "state_vars": ["N", "Q"], "refs": ["Baranyi & Roberts 1994"]}

    def _integrate(self, t_h, n0, q0, mu, nmax):
        def rhs(t, s):
            N, Q = s
            alpha = Q / (1.0 + Q)
            return [mu * alpha * (1.0 - N / nmax) * N, mu * Q]
        t_h = np.asarray(t_h, float)
        sol = solve_ivp(rhs, (0.0, float(t_h.max()) + 1e-6), [n0, q0],
                        t_eval=np.clip(t_h, 0.0, None), method="LSODA",
                        rtol=1e-6, atol=1e-6, max_step=0.5)
        if not sol.success:
            return np.full_like(t_h, n0)
        return sol.y[0]

    def fit(self, obs):
        t = np.asarray(obs.time_s, float) / 3600.0
        y = np.clip(np.asarray(obs.population_count, float), 1e-9, None)
        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])
        p0 = np.array([min(max(y[0], lo[0]), hi[0]), 0.1, 0.6,
                       min(max(y.max() * 1.3, lo[3]), hi[3])])

        def resid(p):
            pred = self._integrate(t, *p)
            return np.log(np.clip(pred, 1e-9, None)) - np.log(y)

        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=300)
        v = np.clip(r.x, lo, hi)
        return dict(zip(self.PARAMS, (float(x) for x in v)))

    def predict(self, params, time_s):
        t = np.asarray(time_s, float) / 3600.0
        return self._integrate(t, params["n0"], params["q0"],
                               params["mu_max"], params["nmax"])
'''

RICHARDS = '''
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
'''

CANNED_SEQUENCE = [LOGISTIC, GOMPERTZ, BARANYI_ODE, RICHARDS]


def wrapped_sequence() -> list[str]:
    """Canned sources fenced as the model would return them."""
    return [f"```python\n{s.strip()}\n```" for s in CANNED_SEQUENCE]
