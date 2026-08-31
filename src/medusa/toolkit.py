"""The seed helper library the inner twin can draw on.

These are self-contained numpy/scipy snippets for things the inner agent commonly
reinvents badly -- robust calibration, ODE integration, exact accounting. The inner
prompt shows `render_library()`; the agent COPIES what it needs into `twin.py` (which
stays a single self-contained file, no `medusa` import).

The meta-loop grows this library. `LoopConfig.helper_library` defaults to
`render_library()`; a genome may replace it. `tests/test_toolkit.py` pins every seed
snippet and is off-limits to the meta-agent.
"""

from __future__ import annotations

SNIPPETS: dict[str, str] = {
    "robust_log_fit": '''
def robust_log_fit(resid_log, p0, lo, hi, max_nfev=4000):
    """Least-squares in log space with bounds + a differential-evolution fallback.

    resid_log(p) -> array of (log model - log data). p0/lo/hi are 1-D arrays.
    Returns the fitted parameter vector, always finite and within [lo, hi].
    """
    import numpy as np
    from scipy.optimize import least_squares, differential_evolution

    p0 = np.clip(np.asarray(p0, float), lo, hi)
    try:
        sol = least_squares(resid_log, p0, bounds=(lo, hi), method="trf", max_nfev=max_nfev)
        x = sol.x
        if np.all(np.isfinite(x)) and np.sum(resid_log(x) ** 2) < np.sum(resid_log(p0) ** 2):
            return np.clip(x, lo, hi)
    except Exception:
        pass
    de = differential_evolution(
        lambda p: float(np.sum(np.asarray(resid_log(p)) ** 2)),
        list(zip(lo, hi)), seed=0, maxiter=60, tol=1e-8, polish=True, workers=1,
    )
    return np.clip(de.x, lo, hi)
''',
    "integrate_ode": '''
def integrate_ode(rhs, y0, t_eval_s, *, per_s=3600.0):
    """solve_ivp wrapper: integrate from 0, sample at t_eval_s (seconds), work in `per_s`
    units internally. Falls back to a constant trajectory if the solver fails."""
    import numpy as np
    from scipy.integrate import solve_ivp

    t = np.asarray(t_eval_s, float) / per_s
    sol = solve_ivp(rhs, (0.0, float(t.max()) + 1e-9), np.atleast_1d(y0),
                    t_eval=np.clip(t, 0.0, None), method="LSODA", rtol=1e-7, atol=1e-7)
    if not sol.success:
        return np.tile(np.atleast_1d(y0)[:, None], (1, t.size))
    return sol.y
''',
    "logistic_curve": '''
def logistic_curve(t_h, n0, mu_max, carrying_capacity):
    """Closed-form logistic. t_h in hours, mu_max per hour."""
    import numpy as np
    n0 = max(float(n0), 1e-9)
    k = max(float(carrying_capacity), n0 + 1e-9)
    return k / (1.0 + ((k - n0) / n0) * np.exp(-mu_max * np.asarray(t_h, float)))
''',
    "baranyi_lag": '''
def baranyi_lag(t_h, n0, mu_max, lag_h, carrying_capacity):
    """Baranyi-Roberts growth: lag -> exponential -> saturation. Hours / per-hour."""
    import numpy as np
    t = np.asarray(t_h, float)
    mu, lag = float(mu_max), max(float(lag_h), 0.0)
    h0 = mu * lag
    a = t + (1.0 / mu) * np.log(np.exp(-mu * t) + np.exp(-h0) - np.exp(-mu * t - h0))
    e = np.exp(mu * a)
    k = max(float(carrying_capacity), float(n0) + 1e-9)
    return float(n0) * e / (1.0 + (float(n0) * (e - 1.0)) / k)
''',
    "close_the_books": '''
def close_the_books(mrr, headcount, opex_per_head, fixed_opex, marketing_spend,
                    capital_raised, cash0):
    """Exact SaaS accounting. Returns (net_income, cash) arrays that satisfy
    cash[t] == cash[t-1] + net_income[t] + capital_raised[t] by construction."""
    import numpy as np
    mrr = np.asarray(mrr, float)
    net_income = mrr - np.asarray(headcount, float) * opex_per_head - fixed_opex \\
        - np.asarray(marketing_spend, float)
    cash = float(cash0) + np.cumsum(net_income + np.asarray(capital_raised, float))
    return net_income, cash
''',
}


def render_library(snippets: dict[str, str] | None = None) -> str:
    snippets = SNIPPETS if snippets is None else snippets
    parts = [
        "Vetted helpers you may copy verbatim into `twin.py` (keep it one self-contained "
        "file -- do NOT `import medusa`). Adapt as needed.\n"
    ]
    for name, code in snippets.items():
        parts.append(f"### `{name}`\n```python\n{code.strip()}\n```")
    return "\n\n".join(parts)
