```python
import numpy as np
from scipy.optimize import least_squares

class Twin:
    FAMILY = "exponential"
    
    PARAMS = {
        "n0":     (1.0, 1e4, "cells"),
        "mu_max": (0.33, 1.5, "1/h"),
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "no lag, no saturation in window",
            "exponential growth (Malthusian)",
            "counts are integer but modeled as continuous",
            "constant division rate over observed range"
        ],
        "state_vars": ["N"],
        "refs": ["Malthus 1798; Monod 1949"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        
        # Fit N(t) = n0 * exp(mu * t) in log space
        # log y = log n0 + mu * t
        # Use linear regression on log(y) for initial guess, then refine with least_squares
        logy = np.log(y)
        
        # Initial guess via linear regression on log(y) vs t
        A = np.vstack([np.ones_like(t_h), t_h]).T
        coef, _, _, _ = np.linalg.lstsq(A, logy, rcond=None)
        log_n0_init = coef[0]
        mu_init = coef[1]
        
        # Clamp to plausible ranges
        n0_lo, n0_hi, _ = self.PARAMS["n0"]
        mu_lo, mu_hi, _ = self.PARAMS["mu_max"]
        
        n0_init = np.clip(np.exp(log_n0_init), n0_lo, n0_hi)
        mu_init = np.clip(mu_init, mu_lo, mu_hi)
        
        # Optimize in log-space to keep positive and bounded
        # Transform: n0 = n0_lo + (n0_hi - n0_lo) * sigmoid(x0)
        #           mu = mu_lo + (mu_hi - mu_lo) * sigmoid(x1)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        # Inverse transform for initial guess
        x0_init = np.log((n0_init - n0_lo) / (n0_hi - n0_init))
        x1_init = np.log((mu_init - mu_lo) / (mu_hi - mu_init))
        p0 = np.array([x0_init, x1_init])
        
        def residual(x):
            n0 = n0_lo + (n0_hi - n0_lo) * sigmoid(x[0])
            mu = mu_lo + (mu_hi - mu_lo) * sigmoid(x[1])
            N_pred = n0 * np.exp(mu * t_h)
            return logy - np.log(N_pred)
        
        # Use least_squares (unconstrained on x, but sigmoid keeps params in bounds)
        res = least_squares(residual, p0, max_nfev=2000)
        x_opt = res.x
        
        n0 = n0_lo + (n0_hi - n0_lo) * sigmoid(x_opt[0])
        mu = mu_lo + (mu_hi - mu_lo) * sigmoid(x_opt[1])
        
        # Ensure within bounds (numerical safety)
        n0 = float(np.clip(n0, n0_lo, n0_hi))
        mu = float(np.clip(mu, mu_lo, mu_hi))
        
        return {
            "n0": n0,
            "mu_max": mu,
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        mu = params["mu_max"]
        t_h = time_s / 3600.0
        N = n0 * np.exp(mu * t_h)
        return N
```