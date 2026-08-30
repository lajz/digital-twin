import numpy as np
from scipy.optimize import least_squares

class Twin:
    FAMILY = "two-stage-lag"
    
    PARAMS = {
        "n0":     (1.0, 1e4, "cells"),
        "mu_max": (0.2, 2.0, "1/h"),
        "lag":    (0.0, 2.0, "h"),
        "k":      (0.01, 10.0, "1/h"),  # rate of transition from lag to exponential
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting resource",
            "two-stage growth: cells in lag phase with zero growth, then switch to exponential",
            "transition rate k controls gradual activation (first-order kinetics)",
            "no saturation in window",
            "counts are continuous approximation of integer cells"
        ],
        "state_vars": ["N", "f"],  # f = fraction of cells in exponential state
        "refs": ["Buchanan et al. 1997 (three-phase linear model)", "McKellar & Lu 2003"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        logy = np.log(y)
        
        # Model: two subpopulations: lagging (no growth) and active (exponential)
        # Let f(t) be fraction of active cells, governed by df/dt = k*(1-f)
        # f(0) = f0, but we can absorb into lag parameter: f(t) = 1 - exp(-k*(t-lag)) for t>lag
        # Then dN/dt = mu_max * f(t) * N
        # N(t) = n0 * exp(mu_max * integral_0^t f(s) ds)
        # For t <= lag: f=0 => N=n0
        # For t > lag: integral = (t-lag) - (1/k)*(1-exp(-k*(t-lag)))
        # So N(t) = n0 * exp(mu_max * [(t-lag) - (1/k)*(1-exp(-k*(t-lag)))])
        
        def model(theta, t):
            n0, mu, lag, k = theta
            t_eff = t - lag
            N = np.empty_like(t_eff)
            mask = t_eff > 0
            if np.any(mask):
                tt = t_eff[mask]
                # compute exponent safely
                exp_kt = np.exp(-k * tt)
                exp_kt = np.where(k*tt > 700, 0.0, exp_kt)
                exponent = mu * (tt - (1.0/k)*(1.0 - exp_kt))
                # guard against overflow
                exponent = np.clip(exponent, -700, 700)
                N[mask] = n0 * np.exp(exponent)
            N[~mask] = n0
            return N
        
        # Initial guess: fit exponential to late data (after 1h) to get mu
        late_mask = t_h > 1.0
        if np.sum(late_mask) > 2:
            A_mat = np.vstack([np.ones_like(t_h[late_mask]), t_h[late_mask]]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, logy[late_mask], rcond=None)
            mu_init = np.clip(coef[1], 0.2, 2.0)
            n0_init = np.clip(np.exp(coef[0] - mu_init*1.0), 10, 1e4)  # back-extrapolate to t=0
        else:
            # fallback to full fit
            A_mat = np.vstack([np.ones_like(t_h), t_h]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, logy, rcond=None)
            n0_init = np.clip(np.exp(coef[0]), 10, 1e4)
            mu_init = np.clip(coef[1], 0.2, 2.0)
        
        lag_init = 0.2  # small lag from data (early points show slow growth)
        k_init = 5.0    # fast transition
        
        lb = [1.0, 0.2, 0.0, 0.01]
        ub = [1e4, 2.0, 2.0, 10.0]
        p0 = [n0_init, mu_init, lag_init, k_init]
        
        def residual(theta):
            N_pred = model(theta, t_h)
            # avoid log(0) or negative
            N_pred = np.maximum(N_pred, 1e-6)
            return logy - np.log(N_pred)
        
        res = least_squares(residual, p0, bounds=(lb, ub), max_nfev=5000)
        n0, mu, lag, k = res.x
        # clamp to bounds
        n0 = float(np.clip(n0, lb[0], ub[0]))
        mu = float(np.clip(mu, lb[1], ub[1]))
        lag = float(np.clip(lag, lb[2], ub[2]))
        k = float(np.clip(k, lb[3], ub[3]))
        
        return {
            "n0": n0,
            "mu_max": mu,
            "lag": lag,
            "k": k,
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        mu = params["mu_max"]
        lag = params["lag"]
        k = params["k"]
        t_h = time_s / 3600.0
        
        t_eff = t_h - lag
        N = np.empty_like(t_eff)
        mask = t_eff > 0
        if np.any(mask):
            tt = t_eff[mask]
            exp_kt = np.exp(-k * tt)
            exp_kt = np.where(k*tt > 700, 0.0, exp_kt)
            exponent = mu * (tt - (1.0/k)*(1.0 - exp_kt))
            exponent = np.clip(exponent, -700, 700)
            N[mask] = n0 * np.exp(exponent)
        N[~mask] = n0
        return N