```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class Twin:
    # Mechanistic family: two subpopulations with stochastic switching between
    # fast and slow growing states (phenotypic heterogeneity / persistence-like).
    # This is a different mechanism from the previous "two-stage-lag-adaptation"
    # which used a deterministic transition from lag to active. Here both states
    # grow, but with different rates, and cells switch back and forth, producing
    # a smooth acceleration that can capture subtle deviations from pure exponential.
    FAMILY = "two-state-switching"
    
    PARAMS = {
        "n0":        (1.0, 1e4, "cells"),
        "mu_fast":   (0.2, 2.0, "1/h"),
        "mu_slow":   (0.0, 0.5, "1/h"),
        "switch_fs": (0.01, 5.0, "1/h"),  # rate from fast->slow
        "switch_sf": (0.01, 5.0, "1/h"),  # rate from slow->fast
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "two phenotypic states: fast-growing and slow-growing (persister-like)",
            "cells switch between states with first-order kinetics",
            "both states grow exponentially with their own rate",
            "no death, no resource limitation in window",
            "counts are continuous approximation of integer cells",
            "switching rates are constant (no adaptation, just stochastic phenotype switching)"
        ],
        "state_vars": ["N_fast", "N_slow"],
        "refs": ["Balaban et al. 2004", "Kussell & Leibler 2005"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        logy = np.log(y)
        
        # Model:
        # dN_fast/dt = mu_fast*N_fast - switch_fs*N_fast + switch_sf*N_slow
        # dN_slow/dt = mu_slow*N_slow + switch_fs*N_fast - switch_sf*N_slow
        # Initial: N_fast(0) = n0, N_slow(0) = 0 (all start fast, but quickly equilibrate)
        
        def model(theta, t):
            n0, mu_f, mu_s, w_fs, w_sf = theta
            def rhs(t, z):
                Nf, Ns = z
                dNf = mu_f*Nf - w_fs*Nf + w_sf*Ns
                dNs = mu_s*Ns + w_fs*Nf - w_sf*Ns
                return [dNf, dNs]
            z0 = [n0, 0.0]
            sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t,
                           method='LSODA', rtol=1e-8, atol=1e-10)
            Ntot = sol.y[0] + sol.y[1]
            return Ntot
        
        # Initial guess: use late-time exponential slope as mu_fast, and small slow fraction
        # First, fit a straight line to the last 20% of data
        n_late = max(3, int(0.2*len(t_h)))
        late_mask = np.ones_like(t_h, dtype=bool)
        late_mask[:len(t_h)-n_late] = False
        if np.sum(late_mask) > 2:
            A = np.vstack([np.ones_like(t_h[late_mask]), t_h[late_mask]]).T
            coef, _, _, _ = np.linalg.lstsq(A, logy[late_mask], rcond=None)
            mu_init = np.clip(coef[1], 0.2, 1.5)
            n0_init = np.clip(np.exp(coef[0] - mu_init*0.0), 10, 1e4)
        else:
            A = np.vstack([np.ones_like(t_h), t_h]).T
            coef, _, _, _ = np.linalg.lstsq(A, logy, rcond=None)
            n0_init = np.clip(np.exp(coef[0]), 10, 1e4)
            mu_init = np.clip(coef[1], 0.2, 1.5)
        
        # For switching model, initial guess: mu_fast ~ late slope, mu_slow ~ 0.1*mu_fast
        # switching rates ~ 0.1-1 per hour
        mu_slow_init = np.clip(mu_init*0.1, 0.0, 0.5)
        w_fs_init = 0.5
        w_sf_init = 0.5
        
        lb = [1.0, 0.2, 0.0, 0.01, 0.01]
        ub = [1e4, 2.0, 0.5, 5.0, 5.0]
        p0 = [n0_init, mu_init, mu_slow_init, w_fs_init, w_sf_init]
        
        # Ensure mu_slow < mu_fast (physically meaningful)
        # We'll enforce by clipping in residual if needed
        
        def residual(theta):
            n0, mu_f, mu_s, w_fs, w_sf = theta
            if mu_s >= mu_f:
                return np.ones_like(logy)*1e6
            try:
                N_pred = model(theta, t_h)
            except Exception:
                return np.ones_like(logy)*1e6
            N_pred = np.maximum(N_pred, 1e-6)
            return logy - np.log(N_pred)
        
        best_res = None
        best_theta = None
        rng = np.random.default_rng(123)
        for restart in range(15):
            if restart > 0:
                p0_pert = [
                    np.clip(n0_init * np.exp(rng.normal(0, 0.2)), lb[0], ub[0]),
                    np.clip(mu_init * np.exp(rng.normal(0, 0.1)), lb[1], ub[1]),
                    np.clip(mu_slow_init * np.exp(rng.normal(0, 0.3)), lb[2], ub[2]),
                    np.clip(w_fs_init * np.exp(rng.normal(0, 0.5)), lb[3], ub[3]),
                    np.clip(w_sf_init * np.exp(rng.normal(0, 0.5)), lb[4], ub[4]),
                ]
                # ensure mu_s < mu_f
                if p0_pert[2] >= p0_pert[1]:
                    p0_pert[2] = np.clip(p0_pert[1]*0.5, lb[2], ub[2])
            else:
                p0_pert = p0
            try:
                res = least_squares(residual, p0_pert, bounds=(lb, ub), max_nfev=4000)
                if best_res is None or res.cost < best_res.cost:
                    best_res = res
                    best_theta = res.x
            except Exception:
                continue
        
        if best_theta is None:
            best_theta = [n0_init, mu_init, mu_slow_init, w_fs_init, w_sf_init]
        
        n0, mu_f, mu_s, w_fs, w_sf = best_theta
        n0 = float(np.clip(n0, lb[0], ub[0]))
        mu_f = float(np.clip(mu_f, lb[1], ub[1]))
        mu_s = float(np.clip(mu_s, lb[2], ub[2]))
        w_fs = float(np.clip(w_fs, lb[3], ub[3]))
        w_sf = float(np.clip(w_sf, lb[4], ub[4]))
        # enforce mu_s < mu_f
        if mu_s >= mu_f:
            mu_s = mu_f * 0.5
        
        return {
            "n0": n0,
            "mu_fast": mu_f,
            "mu_slow": mu_s,
            "switch_fs": w_fs,
            "switch_sf": w_sf,
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        mu_f = params["mu_fast"]
        mu_s = params["mu_slow"]
        w_fs = params["switch_fs"]
        w_sf = params["switch_sf"]
        t_h = time_s / 3600.0
        
        def rhs(t, z):
            Nf, Ns = z
            dNf = mu_f*Nf - w_fs*Nf + w_sf*Ns
            dNs = mu_s*Ns + w_fs*Nf - w_sf*Ns
            return [dNf, dNs]
        
        z0 = [n0, 0.0]
        sol = solve_ivp(rhs, [t_h[0], t_h[-1]], z0, t_eval=t_h,
                       method='LSODA', rtol=1e-8, atol=1e-10)
        Ntot = sol.y[0] + sol.y[1]
        Ntot = np.maximum(Ntot, 0)
        return Ntot
```