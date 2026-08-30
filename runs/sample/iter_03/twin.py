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