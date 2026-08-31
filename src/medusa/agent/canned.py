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

STRUCTURED_COUPLED = '''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


class Twin:
    FAMILY = "coupled-odes"
    PARAMS = {
        "n0":         (1.0, 1e5, "cells"),
        "b0":         (1.0, 1e6, "um"),
        "elong_rate": (0.1, 3.0, "1/h"),
        "div_length": (2.0, 12.0, "um"),
        "div_sharp":  (1.0, 12.0, "dimensionless"),
        "cv0":        (0.05, 0.5, "fraction"),
    }
    METADATA = {"assumptions": ["exponential biomass growth",
                               "division flux rises steeply once mean length exceeds a target",
                               "constant length CV"],
               "state_vars": ["B", "N"], "refs": ["Fredrickson 1967", "Taheri-Araghi 2015"]}

    def fit(self, obs):
        t = np.asarray(obs.time_s, float) / 3600.0
        N = np.clip(np.asarray(obs.population_count, float), 1e-9, None)
        B = np.clip(np.asarray(obs.total_length_um, float), 1e-9, None)
        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])
        p0 = np.array([N[0], B[0], 1.0, float(np.clip(np.mean(B / N), 2.1, 11.9)), 4.0,
                       float(np.clip(np.std(B / N) / np.mean(B / N) + 0.25, 0.06, 0.49))])
        p0 = np.clip(p0, lo, hi)

        def resid(p):
            pr = self._predict(p, t)
            return np.concatenate([
                np.log(np.clip(pr["population_count"], 1e-9, None)) - np.log(N),
                np.log(np.clip(pr["total_length_um"], 1e-9, None)) - np.log(B),
            ])

        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=200)
        v = np.clip(r.x, lo, hi)
        return dict(zip(self.PARAMS, (float(x) for x in v)))

    def _predict(self, p, t_h):
        n0, b0, k, Ld, sharp, cv0 = p
        def rhs(t, s):
            B, N = np.maximum(s, 1e-9)
            meanL = B / N
            J = k * N * (meanL / Ld) ** sharp
            return [k * B, J]
        sol = solve_ivp(rhs, (0.0, float(np.max(t_h)) + 1e-6), [b0, n0],
                        t_eval=np.clip(t_h, 0.0, None), method="LSODA",
                        rtol=1e-7, atol=1e-7, max_step=0.25)
        B, N = sol.y if sol.success else (np.full_like(t_h, b0), np.full_like(t_h, n0))
        N = np.clip(N, 1e-9, None); B = np.clip(B, 1e-9, None)
        return {"total_length_um": B, "population_count": N,
                "mean_length_um": B / N, "length_cv": np.full_like(B, cv0)}

    def predict(self, params, time_s):
        p = np.array([params[k] for k in self.PARAMS])
        return self._predict(p, np.asarray(time_s, float) / 3600.0)
'''


SAAS_BASS = '''
import numpy as np
from scipy.optimize import least_squares

_MONTH_S = 30.0 * 86400.0


class Twin:
    FAMILY = "saturating-acquisition-churn"
    PARAMS = {
        "market":         (5e3, 2e5, "accounts"),
        "acq_per_dollar": (2e-3, 4e-2, "accounts/$"),
        "monthly_churn":  (4e-3, 0.15, "1/mo"),
        "arpa0":          (20.0, 400.0, "$/account/mo"),
        "arpa_growth":    (-0.01, 0.03, "1/mo"),
        "rev_per_head":   (8e4, 4e5, "$/yr"),
        "opex_per_head":  (6e3, 2e4, "$/mo"),
        "fixed_opex":     (0.0, 2e5, "$/mo"),
    }
    METADATA = {"assumptions": ["saturating paid acquisition", "constant logo churn",
                               "ARPA drifts geometrically", "headcount lags revenue (EMA)",
                               "books close exactly: cash is the running net-income total"],
               "state_vars": ["customers", "cash"], "refs": ["Bass 1969", "Skok LTV/CAC"]}

    def _run(self, p, months, spend, capital, c0, cash0, hc0):
        market, acq, churn, arpa0, ag, rph, oph, fx = p
        n = len(months)
        cust = np.zeros(n); newc = np.zeros(n); hc = np.zeros(n)
        mrr = np.zeros(n); ni = np.zeros(n); cash = np.zeros(n)
        cust[0] = c0; hc[0] = hc0
        for i in range(n):
            arpa = arpa0 * (1.0 + ag) ** i
            if i > 0:
                newc[i] = acq * spend[i] * max(0.0, 1.0 - cust[i-1] / market)
                cust[i] = cust[i-1] + newc[i] - churn * cust[i-1]
                tgt = cust[i] * arpa * 12.0 / rph
                hc[i] = hc[i-1] + 0.25 * (tgt - hc[i-1])
            else:
                newc[i] = acq * spend[i]
            mrr[i] = cust[i] * arpa
            ni[i] = mrr[i] - hc[i] * oph - fx - spend[i]
            cash[i] = (cash[i-1] if i else cash0) + ni[i] + capital[i]
        return {"mrr": mrr, "customers": cust, "new_customers": newc,
                "headcount": hc, "net_income": ni, "cash": cash}

    def fit(self, obs):
        t = np.asarray(obs.time_s, float) / _MONTH_S
        months = np.arange(len(t))
        spend = np.asarray(obs.marketing_spend, float)
        capital = np.asarray(obs.capital_raised, float)
        mrr = np.clip(np.asarray(obs.mrr, float), 1.0, None)
        cust = np.clip(np.asarray(obs.customers, float), 1.0, None)
        self._c0, self._cash0, self._hc0 = float(cust[0]), float(obs.cash[0]), float(obs.headcount[0])

        lo = np.array([b[0] for b in self.PARAMS.values()])
        hi = np.array([b[1] for b in self.PARAMS.values()])
        p0 = np.array([max(cust[-1] * 3, 6e3), 0.011, 0.03,
                       float(np.clip(mrr[0] / cust[0], 21, 399)), 0.004,
                       1.6e5, 1.1e4, 2e4])
        p0 = np.clip(p0, lo, hi)

        def resid(p):
            r = self._run(p, months, spend, capital, self._c0, self._cash0, self._hc0)
            return np.concatenate([
                np.log(np.clip(r["mrr"], 1.0, None)) - np.log(mrr),
                np.log(np.clip(r["customers"], 1.0, None)) - np.log(cust),
            ])

        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=400)
        v = np.clip(sol.x, lo, hi)
        return dict(zip(self.PARAMS, (float(x) for x in v)))

    def predict(self, params, time_s, exog=None):
        t = np.asarray(time_s, float) / _MONTH_S
        months = np.arange(len(t))
        exog = exog or {}
        spend = np.asarray(exog.get("marketing_spend", np.full(len(t), 1e5)), float)
        capital = np.asarray(exog.get("capital_raised", np.zeros(len(t))), float)
        p = np.array([params[k] for k in self.PARAMS])
        return self._run(p, months, spend, capital,
                         getattr(self, "_c0", 400.0), getattr(self, "_cash0", 8e5),
                         getattr(self, "_hc0", 8.0))
'''


CANNED_SEQUENCE = [LOGISTIC, GOMPERTZ, BARANYI_ODE, RICHARDS]
STRUCTURED_SEQUENCE = [STRUCTURED_COUPLED, GOMPERTZ, LOGISTIC]
SAAS_SEQUENCE = [SAAS_BASS]


def _spatial_sequence() -> list[str]:
    from medusa.spatial.sim_reference import SOURCE

    return [SOURCE]


def wrapped_sequence(task_name: str = "population") -> list[str]:
    """Canned sources for the given task, fenced as the model would return them."""
    seq = {
        "spatial": _spatial_sequence,
        "structured": lambda: STRUCTURED_SEQUENCE,
        "saas": lambda: SAAS_SEQUENCE,
    }.get(task_name, lambda: CANNED_SEQUENCE)()
    return [f"```python\n{s.strip()}\n```" for s in seq]
