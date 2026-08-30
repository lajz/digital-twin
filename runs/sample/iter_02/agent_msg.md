```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

class Twin:
    FAMILY = "baranyi-robust-fixed"
    
    PARAMS = {
        "n0": (0.5, 100, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "lag": (0.0, 3.0, "h"),
        "carrying_capacity": (1e3, 1e6, "cells"),
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed", 
            "single limiting nutrient",
            "Baranyi-Roberts lag phase with adjustment function",
            "log-normal multiplicative noise on observations"
        ],
        "state_vars": ["N"],
        "refs": ["Baranyi & Roberts 1994", "Buchanan et al. 1997"],
    }
    
    def fit(self, obs):
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        # use log-space residuals to handle range of magnitudes
        y_log = np.log(y + 1.0)
        
        bounds = [
            self.PARAMS["n0"][:2],
            self.PARAMS["mu_max"][:2],
            self.PARAMS["lag"][:2],
            self.PARAMS["carrying_capacity"][:2],
        ]
        
        # Initial guess from data: n0 ~ first point, mu_max from early growth, lag ~ time to double
        n0_guess = max(1.0, y[0])
        # find where population starts growing significantly (first time > 2*n0)
        idx = np.where(y > 2*n0_guess)[0]
        if len(idx) > 0:
            t_growth = t_h[idx[0]]
            lag_guess = max(0.0, t_growth - 0.3)
        else:
            lag_guess = 0.5
        # estimate mu_max from log-linear part (between lag and saturation)
        mu_max_guess = 2.0
        K_guess = np.max(y) * 1.5
        
        x0 = [n0_guess, mu_max_guess, lag_guess, K_guess]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def residuals(params):
            n0, mu_max, lag, K = params
            # clamp to bounds for safety
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            lag = np.clip(lag, *self.PARAMS["lag"][:2])
            K = np.clip(K, *self.PARAMS["carrying_capacity"][:2])
            
            p = {"n0": n0, "mu_max": mu_max, "lag": lag, "carrying_capacity": K}
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 0.0)
                pred_log = np.log(pred + 1.0)
                return pred_log - y_log
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # First local optimization from good initial guess
        result_local = minimize(
            lambda p: np.sum(residuals(p)**2),
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12},
        )
        
        if result_local.success:
            best = result_local.x
        else:
            # fallback to differential evolution
            result = differential_evolution(
                lambda p: np.sum(residuals(p)**2),
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True,
                workers=1,
            )
            best = result.x
        
        # ensure within bounds
        best = np.clip(best, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best[0],
            "mu_max": best[1],
            "lag": best[2],
            "carrying_capacity": best[3],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        lag = params["lag"]
        K = params["carrying_capacity"]
        
        # Baranyi-Roberts model with adjustment function
        # Use explicit lag with smooth transition: alpha = 0 for t<lag, then 1
        # Smooth transition via logistic with width ~0.05h
        def rhs(t, N):
            # smooth adjustment function, sharp transition
            alpha = 1.0 / (1.0 + np.exp(-(t - lag) * 40.0))
            if N <= 0:
                return 0.0
            return mu_max * alpha * N * (1.0 - N / K)
        
        # integrate from t=0 to max time
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        # use dense output for arbitrary times
        sol = solve_ivp(
            rhs, t_span, [n0], 
            t_eval=t_h, 
            method='LSODA',
            rtol=1e-8, atol=1e-8,
            max_step=0.05,
        )
        
        if not sol.success:
            # fallback: return constant
            return np.full_like(t_h, n0)
        
        N = sol.y[0]
        # ensure non-negative
        N = np.maximum(N, 0.0)
        return N
```