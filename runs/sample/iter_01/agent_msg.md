```python
import numpy as np
from scipy.optimize import least_squares

class Twin:
    FAMILY = "logistic"
    
    PARAMS = {
        "n0":     (1.0, 1e4, "cells"),
        "mu_max": (0.05, 3.0, "1/h"),
        "carrying_capacity": (1e3, 1e8, "cells"),
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting resource",
            "logistic growth (density-dependent slowing)",
            "no lag or death phase in window",
            "counts are integer but modeled as continuous"
        ],
        "state_vars": ["N"],
        "refs": ["Verhulst 1838; Peleg & Corradini 2011"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        
        # Fit in log space to handle orders of magnitude
        # Model: N(t) = K / (1 + (K/N0 - 1) * exp(-mu*t))
        # We'll fit parameters n0, mu_max, K using least_squares on log(y)
        
        # Initial guesses from data
        n0_guess = y[0]
        K_guess = max(y[-1] * 1.5, n0_guess * 2)
        # Estimate mu from early exponential slope
        early_mask = t_h < 0.5
        if early_mask.sum() >= 3:
            slope = np.polyfit(t_h[early_mask], np.log(y[early_mask]), 1)[0]
            mu_guess = max(0.1, min(2.0, slope))
        else:
            mu_guess = 0.5
        
        # Bounds from PARAMS
        n0_lo, n0_hi, _ = self.PARAMS["n0"]
        mu_lo, mu_hi, _ = self.PARAMS["mu_max"]
        K_lo, K_hi, _ = self.PARAMS["carrying_capacity"]
        
        # Use log-transform for parameters to keep them positive and bounded
        # We'll optimize in unbounded space, then clip to bounds
        p0 = np.array([
            np.log(n0_guess - n0_lo + 1e-9),  # log-shift to keep > lo
            np.log(mu_guess - mu_lo + 1e-9),
            np.log(K_guess - K_lo + 1e-9),
        ])
        
        def residual_logp(logp):
            # Transform back
            n0 = n0_lo + np.exp(logp[0])
            mu = mu_lo + np.exp(logp[1])
            K = K_lo + np.exp(logp[2])
            # Clip to upper bounds
            n0 = min(n0, n0_hi)
            mu = min(mu, mu_hi)
            K = min(K, K_hi)
            # Predict
            # Avoid overflow in exp(-mu*t)
            exp_term = np.exp(-mu * t_h)
            # Handle K/N0 - 1 potentially large
            factor = (K / n0 - 1.0) * exp_term
            # For numerical stability: if factor > 1e6, N ~ K * exp(mu*t)/ (K/n0-1) ~ n0*exp(mu*t)
            # but we can just compute directly with np.expm1 for safety
            denom = 1.0 + factor
            # Avoid division by zero / negative
            denom = np.maximum(denom, 1e-12)
            N_pred = K / denom
            # Log residual
            return np.log(y) - np.log(N_pred + 1e-12)
        
        # Use least_squares with bounds on logp? We'll just use unconstrained and clip later.
        res = least_squares(residual_logp, p0, method='lm', max_nfev=2000)
        logp_opt = res.x
        
        n0 = n0_lo + np.exp(logp_opt[0])
        mu = mu_lo + np.exp(logp_opt[1])
        K = K_lo + np.exp(logp_opt[2])
        
        # Clip to hard bounds
        n0 = float(np.clip(n0, n0_lo, n0_hi))
        mu = float(np.clip(mu, mu_lo, mu_hi))
        K = float(np.clip(K, K_lo, K_hi))
        
        # Refine with a direct bounded least_squares on the actual parameters
        # (sometimes helps)
        def residual(p):
            n0, mu, K = p
            exp_term = np.exp(-mu * t_h)
            factor = (K / n0 - 1.0) * exp_term
            denom = 1.0 + factor
            denom = np.maximum(denom, 1e-12)
            N_pred = K / denom
            return np.log(y) - np.log(N_pred + 1e-12)
        
        p0_direct = np.array([n0, mu, K])
        bounds = ([n0_lo, mu_lo, K_lo], [n0_hi, mu_hi, K_hi])
        res2 = least_squares(residual, p0_direct, bounds=bounds, max_nfev=2000)
        n0, mu, K = res2.x
        
        return {
            "n0": float(n0),
            "mu_max": float(mu),
            "carrying_capacity": float(K),
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        mu = params["mu_max"]
        K = params["carrying_capacity"]
        t_h = time_s / 3600.0
        
        # Logistic solution
        exp_term = np.exp(-mu * t_h)
        factor = (K / n0 - 1.0) * exp_term
        denom = 1.0 + factor
        denom = np.maximum(denom, 1e-12)
        N = K / denom
        # Ensure non-negative, finite
        N = np.maximum(N, 0.0)
        return N
```