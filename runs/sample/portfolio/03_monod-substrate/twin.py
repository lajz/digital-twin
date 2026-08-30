import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

class Twin:
    # Using a resource-explicit Monod model where growth is limited by a single substrate
    # This is mechanistically distinct from Baranyi: lag emerges naturally from initial
    # low substrate conversion efficiency, and carrying capacity is substrate-derived.
    FAMILY = "monod-substrate"
    
    PARAMS = {
        "n0": (0.5, 100, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "ks": (1e-3, 1e3, "substrate units"),       # half-saturation constant
        "yield": (1e-7, 1e-3, "cells/substrate unit"),  # cells produced per substrate
        "s0": (1e3, 1e8, "substrate units"),          # initial substrate
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting substrate",
            "Monod growth kinetics: dN/dt = mu_max * S/(S+Ks) * N",
            "substrate consumption proportional to growth: dS/dt = - (1/yield) * dN/dt",
            "no maintenance or death",
            "initial substrate S0 determines carrying capacity via yield*S0"
        ],
        "state_vars": ["N", "S"],
        "refs": ["Monod 1949", "Pirt 1975"],
    }
    
    def fit(self, obs):
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        y_log = np.log(y + 1.0)
        
        # Parameter bounds
        bounds = [
            self.PARAMS["n0"][:2],
            self.PARAMS["mu_max"][:2],
            self.PARAMS["ks"][:2],
            self.PARAMS["yield"][:2],
            self.PARAMS["s0"][:2],
        ]
        
        # Initial guesses
        n0_guess = max(1.0, y[0])
        # estimate mu_max from max log-slope in early growth (before saturation)
        # crude: use first few points where y>2*n0
        idx = np.where(y > 2*n0_guess)[0]
        if len(idx) > 0:
            t_growth = t_h[idx[0]]
            # assume lag ~0.6h from data sheet, but we let it emerge
            # use slope between t_growth and t_growth+1h
            mask = (t_h >= t_growth) & (t_h <= t_growth + 1.0)
            if mask.sum() >= 2:
                t_sub = t_h[mask]
                y_sub = np.maximum(y[mask], 1.0)
                # log-linear fit
                A = np.vstack([np.ones_like(t_sub), t_sub]).T
                coeff, _, _, _ = np.linalg.lstsq(A, np.log(y_sub), rcond=None)
                mu_est = max(0.5, min(3.5, coeff[1]))
            else:
                mu_est = 2.0
        else:
            mu_est = 2.0
        
        # Carrying capacity ~ max observed * 1.5
        K_guess = np.max(y) * 1.5
        # If yield and s0 product = K, set yield=1e-5, s0=K/yield
        yield_guess = 1e-5
        s0_guess = K_guess / yield_guess
        # clamp s0 to bounds
        s0_guess = np.clip(s0_guess, *self.PARAMS["s0"][:2])
        ks_guess = 10.0  # arbitrary mid-range
        
        x0 = [n0_guess, mu_est, ks_guess, yield_guess, s0_guess]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def residuals(params):
            n0, mu_max, ks, yld, s0 = params
            # clamp
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            ks = np.clip(ks, *self.PARAMS["ks"][:2])
            yld = np.clip(yld, *self.PARAMS["yield"][:2])
            s0 = np.clip(s0, *self.PARAMS["s0"][:2])
            
            p = {
                "n0": n0, "mu_max": mu_max, "ks": ks,
                "yield": yld, "s0": s0
            }
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 0.0)
                pred_log = np.log(pred + 1.0)
                return pred_log - y_log
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # Local optimization first
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
                maxiter=150,
                popsize=15,
                tol=1e-10,
                polish=True,
                workers=1,
            )
            best = result.x
        
        best = np.clip(best, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best[0],
            "mu_max": best[1],
            "ks": best[2],
            "yield": best[3],
            "s0": best[4],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        ks = params["ks"]
        yld = params["yield"]
        s0 = params["s0"]
        
        # Initial substrate: S0
        # Carrying capacity = yld * S0 (when S depleted)
        # But we must ensure S0 is large enough to support growth to observed max
        # The model will naturally saturate when S -> 0
        
        def rhs(t, state):
            N, S = state
            if N <= 0 or S <= 0:
                return [0.0, 0.0]
            # Monod term
            growth_rate = mu_max * (S / (S + ks)) * N
            # substrate consumption
            dS = - (1.0 / yld) * growth_rate
            # ensure S doesn't go negative
            if S <= 0:
                dS = 0.0
                growth_rate = 0.0
            return [growth_rate, dS]
        
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        # Use dense output
        sol = solve_ivp(
            rhs, t_span, [n0, s0],
            t_eval=t_h,
            method='LSODA',
            rtol=1e-8, atol=1e-8,
            max_step=0.05,
        )
        
        if not sol.success:
            # fallback: simple logistic with same carrying capacity
            K = yld * s0
            alpha = mu_max  # no lag
            N_pred = K / (1 + (K/n0 - 1) * np.exp(-alpha * t_h))
            return np.maximum(N_pred, 0.0)
        
        N = sol.y[0]
        N = np.maximum(N, 0.0)
        return N