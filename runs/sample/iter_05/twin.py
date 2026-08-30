import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

class Twin:
    # Mechanistic: two-reservoir model with explicit lag via "maturation" of a
    # bottleneck enzyme. Cells transition from lag to exponential growth when a
    # critical enzyme pool is built. Growth is then limited by a Monod-type
    # resource depletion, but we model the resource implicitly via a carrying
    # capacity with a smooth switch. This is distinct from Baranyi-Richards
    # because the lag is driven by a separate state variable (enzyme) with its
    # own dynamics, and the growth term uses a Monod form with a time-varying
    # substrate that is consumed proportionally to growth.
    FAMILY = "two-reservoir-monod"
    
    PARAMS = {
        "n0": (0.5, 100.0, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "lag": (0.0, 3.0, "h"),
        "carrying_capacity": (1e3, 1e6, "cells"),
        "k_s": (1e2, 1e6, "cells"),  # Monod half-saturation in terms of equivalent population
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting resource",
            "lag phase due to accumulation of a bottleneck enzyme",
            "growth rate follows Monod kinetics on a resource that is depleted proportionally to biomass",
            "resource pool is implicit in carrying capacity",
            "no death, no maintenance"
        ],
        "state_vars": ["N", "E", "S"],
        "refs": ["Monod 1949", "Baranyi & Roberts 1994 (lag concept)"],
    }
    
    def fit(self, obs):
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        y_log = np.log(y + 1.0)
        
        bounds = [
            self.PARAMS["n0"][:2],
            self.PARAMS["mu_max"][:2],
            self.PARAMS["lag"][:2],
            self.PARAMS["carrying_capacity"][:2],
            self.PARAMS["k_s"][:2],
        ]
        
        # Initial guesses
        n0_guess = max(1.0, y[0])
        # Estimate mu from early exponential phase (before saturation)
        mu_est = 2.0
        if len(t_h) > 2:
            slopes = []
            for i in range(len(t_h)):
                window = (t_h >= t_h[i]) & (t_h <= t_h[i] + 1.0)
                if window.sum() >= 3:
                    t_sub = t_h[window]
                    y_sub = np.maximum(y[window], 1.0)
                    if np.all(np.isfinite(y_sub)) and np.ptp(t_sub) > 0:
                        A = np.vstack([np.ones_like(t_sub), t_sub]).T
                        coeff, _, _, _ = np.linalg.lstsq(A, np.log(y_sub), rcond=None)
                        if np.isfinite(coeff[1]):
                            slopes.append(coeff[1])
            if slopes:
                mu_est = max(0.5, min(3.5, np.max(slopes)))
        
        # Lag estimate: time when log(y) first exceeds log(n0)+0.5
        lag_guess = 0.5
        threshold = np.log(n0_guess) + 0.5
        idx = np.where(np.log(y+1) > threshold)[0]
        if len(idx) > 0:
            lag_guess = max(0.0, t_h[idx[0]] - 0.2)
        lag_guess = np.clip(lag_guess, *self.PARAMS["lag"][:2])
        
        K_guess = np.max(y) * 1.2
        K_guess = np.clip(K_guess, *self.PARAMS["carrying_capacity"][:2])
        k_s_guess = K_guess * 0.1
        k_s_guess = np.clip(k_s_guess, *self.PARAMS["k_s"][:2])
        
        x0 = [n0_guess, mu_est, lag_guess, K_guess, k_s_guess]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def residuals(params):
            n0, mu_max, lag, K, k_s = params
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            lag = np.clip(lag, *self.PARAMS["lag"][:2])
            K = np.clip(K, *self.PARAMS["carrying_capacity"][:2])
            k_s = np.clip(k_s, *self.PARAMS["k_s"][:2])
            
            p = {
                "n0": n0, "mu_max": mu_max, "lag": lag,
                "carrying_capacity": K, "k_s": k_s
            }
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 1e-6)
                pred_log = np.log(pred)
                # Weighted log residuals, more weight on early low counts
                return (pred_log - y_log) / np.sqrt(y + 1.0)
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # Multi-start least_squares for robustness
        best_res = np.inf
        best_x = None
        rng = np.random.default_rng(42)
        starts = [x0]
        for _ in range(5):
            perturb = np.array([
                rng.uniform(0.8, 1.2) for _ in range(5)
            ])
            starts.append(np.clip(x0 * perturb, [b[0] for b in bounds], [b[1] for b in bounds]))
        
        for start in starts:
            try:
                result = least_squares(
                    residuals, start,
                    bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                    max_nfev=500,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                )
                if result.cost < best_res:
                    best_res = result.cost
                    best_x = result.x
            except Exception:
                continue
        
        if best_x is None:
            # Fallback to simple grid search on key params
            mu_grid = np.linspace(0.5, 3.5, 10)
            lag_grid = np.linspace(0.0, 2.0, 10)
            K_grid = np.geomspace(1e3, 1e6, 10)
            best_res = np.inf
            best_x = x0
            for mu in mu_grid:
                for lag in lag_grid:
                    for K in K_grid:
                        params = [n0_guess, mu, lag, K, k_s_guess]
                        res = np.sum(residuals(params)**2)
                        if res < best_res:
                            best_res = res
                            best_x = params
        
        best_x = np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best_x[0],
            "mu_max": best_x[1],
            "lag": best_x[2],
            "carrying_capacity": best_x[3],
            "k_s": best_x[4],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        lag = params["lag"]
        K = params["carrying_capacity"]
        k_s = params["k_s"]
        
        # Two-reservoir model:
        # dN/dt = mu_max * (E/(E+1)) * (S/(S+k_s)) * N
        # dE/dt = mu_max * E  (exponential buildup of enzyme)
        # dS/dt = - (1/Y) * dN/dt  but we use a Monod term with S starting at K
        # To keep it simple, we model S as a resource reservoir that is depleted
        # proportionally to growth: S(t) = K - N(t) (since N consumes resource)
        # But then Monod term becomes (K-N)/(K-N+k_s). This is equivalent to
        # a logistic-like term but with a different shape. We'll use that.
        #
        # Enzyme E starts at E0 = 1/(exp(mu_max*lag)-1) so that E/(E+1) ~ 0
        # initially and reaches ~1 after lag.
        
        E0 = 1.0 / (np.exp(mu_max * lag) - 1.0) if lag > 0 else 1e10
        
        def rhs(t, state):
            N, E = state
            if N <= 0 or E <= 0:
                return [0.0, 0.0]
            N_safe = min(N, K * 1.1)
            S = max(K - N_safe, 0.0)
            monod = S / (S + k_s) if S + k_s > 0 else 0.0
            enzyme_factor = E / (E + 1.0)
            growth = mu_max * enzyme_factor * monod * N_safe
            dE = mu_max * E
            return [growth, dE]
        
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        sol = solve_ivp(
            rhs, t_span, [n0, E0],
            t_eval=t_h,
            method='LSODA',
            rtol=1e-10, atol=1e-10,
            max_step=0.02,
        )
        
        if not sol.success:
            # Fallback: logistic with lag shift
            t_shift = t_h - lag
            t_shift = np.maximum(t_shift, 0.0)
            N_pred = K / (1 + (K/n0 - 1) * np.exp(-mu_max * t_shift))
            return np.maximum(N_pred, 0.0)
        
        N = sol.y[0]
        N = np.maximum(N, 0.0)
        N = np.minimum(N, K * 1.01)
        return N