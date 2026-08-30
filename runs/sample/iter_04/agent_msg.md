```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

class Twin:
    # Two-stage Baranyi-Roberts model with explicit lag phase and curvature parameter
    # Mechanistic: lag is driven by intracellular "work-to-build" (q), then exponential
    # growth limited by carrying capacity. Distinct from plain Baranyi by using a
    # Richards-type curvature parameter (nu) for sharper/smoother transition.
    FAMILY = "baranyi-richards"
    
    PARAMS = {
        "n0": (0.5, 100.0, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "lag": (0.0, 3.0, "h"),
        "carrying_capacity": (1e3, 1e6, "cells"),
        "nu": (0.1, 5.0, "dimensionless"),  # Richards curvature: >1 sharper, <1 smoother
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting resource",
            "Baranyi-Roberts lag via intracellular work (q)",
            "Richards curvature parameter nu for asymmetric sigmoid",
            "no death, no maintenance",
            "carrying capacity is constant"
        ],
        "state_vars": ["N", "q"],
        "refs": ["Baranyi & Roberts 1994", "Richards 1959"],
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
            self.PARAMS["nu"][:2],
        ]
        
        # Initial guesses from data
        n0_guess = max(1.0, y[0])
        # Estimate mu_max from early exponential phase (before saturation)
        # Use max log-slope over any window of 1h
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
        nu_guess = 1.0  # Baranyi-like
        
        x0 = [n0_guess, mu_est, lag_guess, K_guess, nu_guess]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def residuals(params):
            n0, mu_max, lag, K, nu = params
            # clamp
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            lag = np.clip(lag, *self.PARAMS["lag"][:2])
            K = np.clip(K, *self.PARAMS["carrying_capacity"][:2])
            nu = np.clip(nu, *self.PARAMS["nu"][:2])
            
            p = {
                "n0": n0, "mu_max": mu_max, "lag": lag,
                "carrying_capacity": K, "nu": nu
            }
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 1e-6)
                pred_log = np.log(pred)
                # Use relative error in log space
                return (pred_log - y_log) / np.sqrt(y + 1.0)
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # Multi-start local optimization for robustness
        best_res = np.inf
        best_x = None
        rng = np.random.default_rng(42)
        starts = [x0]
        # Add random perturbations around guess
        for _ in range(5):
            perturb = np.array([
                rng.uniform(0.8, 1.2) for _ in range(5)
            ])
            starts.append(np.clip(x0 * perturb, [b[0] for b in bounds], [b[1] for b in bounds]))
        
        for start in starts:
            result = minimize(
                lambda p: np.sum(residuals(p)**2),
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8},
            )
            if result.success and result.fun < best_res:
                best_res = result.fun
                best_x = result.x
        
        if best_x is None:
            # Fallback to differential evolution
            result = differential_evolution(
                lambda p: np.sum(residuals(p)**2),
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-12,
                polish=True,
                workers=1,
            )
            best_x = result.x
        
        best_x = np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best_x[0],
            "mu_max": best_x[1],
            "lag": best_x[2],
            "carrying_capacity": best_x[3],
            "nu": best_x[4],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        lag = params["lag"]
        K = params["carrying_capacity"]
        nu = params["nu"]
        
        # Baranyi-Roberts with Richards curvature:
        # dN/dt = mu_max * (q/(q+1)) * N * (1 - (N/K)^nu)
        # dq/dt = mu_max * q   (q0 chosen so that lag = ln(1+1/q0)/mu_max)
        # q0 = 1/(exp(mu_max*lag)-1)
        
        q0 = 1.0 / (np.exp(mu_max * lag) - 1.0) if lag > 0 else 1e10
        
        def rhs(t, state):
            N, q = state
            if N <= 0 or q <= 0:
                return [0.0, 0.0]
            # Clamp N to avoid overflow
            N_safe = min(N, K * 1.1)
            # Logistic term with Richards curvature
            if nu < 1e-6:
                # avoid division by zero
                logistic = 1.0 - (N_safe / K)
            else:
                logistic = 1.0 - (N_safe / K) ** nu
            logistic = max(logistic, -1e-6)
            growth = mu_max * (q / (q + 1.0)) * N_safe * logistic
            dq = mu_max * q
            return [growth, dq]
        
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        # Use dense output with fine max_step for accuracy
        sol = solve_ivp(
            rhs, t_span, [n0, q0],
            t_eval=t_h,
            method='LSODA',
            rtol=1e-10, atol=1e-10,
            max_step=0.02,
        )
        
        if not sol.success:
            # Fallback: pure logistic with same K and lag via time shift
            t_shift = t_h - lag
            t_shift = np.maximum(t_shift, 0.0)
            N_pred = K / (1 + (K/n0 - 1) * np.exp(-mu_max * t_shift))
            return np.maximum(N_pred, 0.0)
        
        N = sol.y[0]
        N = np.maximum(N, 0.0)
        # Ensure N doesn't exceed K significantly
        N = np.minimum(N, K * 1.01)
        return N
```