```python
import numpy as np
from scipy.optimize import least_squares

class Twin:
    FAMILY = "monod-consumer"
    
    PARAMS = {
        "n0":     (1.0, 1e4, "cells"),
        "mu_max": (0.2, 2.0, "1/h"),
        "K_s":    (1e-6, 1e3, "substrate units"),
        "s0":     (1e-3, 1e6, "substrate units"),
        "Y":      (1e-8, 1e-2, "cells/substrate unit"),
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting resource (substrate S)",
            "Monod kinetics for growth: mu = mu_max * S/(K_s + S)",
            "substrate consumption coupled to growth: dS/dt = -(1/Y)*dN/dt",
            "no maintenance, no death, no saturation in cell count over window",
            "counts are continuous approximation of integer cells"
        ],
        "state_vars": ["N", "S"],
        "refs": ["Monod 1949", "Kovárová-Kovar & Egli 1998"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        logy = np.log(y)
        
        # Model: dN/dt = mu_max * S/(K_s+S) * N
        #        dS/dt = -(1/Y) * dN/dt
        # Integral form: N + Y*S = const => S = S0 - (N-N0)/Y
        # So dN/dt = mu_max * (S0 - (N-N0)/Y)/(K_s + S0 - (N-N0)/Y) * N
        # Let A = Y*S0, B = Y*K_s, then S0 - (N-N0)/Y = (A - (N-N0))/Y
        # dN/dt = mu_max * (A - N + N0)/(B + A - N + N0) * N
        # Define C = A + N0 = Y*S0 + N0. Then dN/dt = mu_max * (C - N)/(B + C - N) * N
        # This is a generalized logistic-like model. Solve ODE numerically.
        
        def model(theta, t):
            n0, mu, K_s, s0, Y = theta
            # Solve ODE
            from scipy.integrate import solve_ivp
            def rhs(t, z):
                N = z[0]
                S = z[1]
                dN = mu * S/(K_s + S) * N
                dS = -(1.0/Y) * dN
                return [dN, dS]
            z0 = [n0, s0]
            sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)
            return sol.y[0]
        
        # Initial guess: fit pure exponential to late data
        late_mask = t_h > 1.0
        if np.sum(late_mask) > 2:
            A_mat = np.vstack([np.ones_like(t_h[late_mask]), t_h[late_mask]]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, logy[late_mask], rcond=None)
            mu_init = np.clip(coef[1], 0.2, 2.0)
            n0_init = np.clip(np.exp(coef[0] - mu_init*1.0), 10, 1e4)
        else:
            A_mat = np.vstack([np.ones_like(t_h), t_h]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, logy, rcond=None)
            n0_init = np.clip(np.exp(coef[0]), 10, 1e4)
            mu_init = np.clip(coef[1], 0.2, 2.0)
        
        # For Monod, we want S >> K_s over the window so growth is near exponential.
        # Choose s0 large, K_s small, Y such that substrate depletion is negligible.
        # But we allow some slight curvature if needed.
        s0_init = 1e4
        K_s_init = 1e-2
        Y_init = 1e-5  # so that total cell mass change ~ (1160-110)*1e-5 ~ 0.01 substrate units, negligible vs s0
        
        lb = [1.0, 0.2, 1e-6, 1e-3, 1e-8]
        ub = [1e4, 2.0, 1e3, 1e6, 1e-2]
        p0 = [n0_init, mu_init, K_s_init, s0_init, Y_init]
        
        def residual(theta):
            try:
                N_pred = model(theta, t_h)
            except Exception:
                return np.ones_like(logy)*1e6
            N_pred = np.maximum(N_pred, 1e-6)
            return logy - np.log(N_pred)
        
        # Use a few random restarts to avoid local minima
        best_res = None
        best_theta = None
        rng = np.random.default_rng(42)
        for restart in range(5):
            # perturb initial guess in log space
            if restart > 0:
                p0_pert = [
                    np.clip(n0_init * np.exp(rng.normal(0, 0.1)), lb[0], ub[0]),
                    np.clip(mu_init * np.exp(rng.normal(0, 0.1)), lb[1], ub[1]),
                    np.clip(10**rng.uniform(-6, 2), lb[2], ub[2]),
                    np.clip(10**rng.uniform(-2, 5), lb[3], ub[3]),
                    np.clip(10**rng.uniform(-8, -2), lb[4], ub[4]),
                ]
            else:
                p0_pert = p0
            try:
                res = least_squares(residual, p0_pert, bounds=(lb, ub), max_nfev=2000)
                if best_res is None or res.cost < best_res.cost:
                    best_res = res
                    best_theta = res.x
            except Exception:
                continue
        
        if best_theta is None:
            # fallback to simple exponential
            best_theta = [n0_init, mu_init, 1e-2, 1e4, 1e-5]
        
        n0, mu, K_s, s0, Y = best_theta
        # Clamp to bounds
        n0 = float(np.clip(n0, lb[0], ub[0]))
        mu = float(np.clip(mu, lb[1], ub[1]))
        K_s = float(np.clip(K_s, lb[2], ub[2]))
        s0 = float(np.clip(s0, lb[3], ub[3]))
        Y = float(np.clip(Y, lb[4], ub[4]))
        
        return {
            "n0": n0,
            "mu_max": mu,
            "K_s": K_s,
            "s0": s0,
            "Y": Y,
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        mu = params["mu_max"]
        K_s = params["K_s"]
        s0 = params["s0"]
        Y = params["Y"]
        t_h = time_s / 3600.0
        
        from scipy.integrate import solve_ivp
        def rhs(t, z):
            N = z[0]
            S = z[1]
            dN = mu * S/(K_s + S) * N
            dS = -(1.0/Y) * dN
            return [dN, dS]
        z0 = [n0, s0]
        sol = solve_ivp(rhs, [t_h[0], t_h[-1]], z0, t_eval=t_h, method='LSODA', rtol=1e-8, atol=1e-10)
        N = sol.y[0]
        # Ensure finite and non-negative
        N = np.maximum(N, 0)
        return N
```