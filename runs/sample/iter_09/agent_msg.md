```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class Twin:
    # Mechanistic family: population growth with explicit cell-cycle structure.
    # Instead of a single "growth rate" for the whole population, we track cells
    # through two phases: a "pre-replication" phase (G1-like) and a "division-ready"
    # phase (S/G2/M-like). Cells progress through the cycle with first-order kinetics,
    # and only division-ready cells divide. This captures potential delays between
    # biomass accumulation and division events, producing wavy or delayed growth
    # patterns that pure exponential or two-state models cannot represent.
    FAMILY = "cell-cycle-two-phase"
    
    PARAMS = {
        "n0":        (1.0, 1e4, "cells"),
        "k_g1":      (0.1, 3.0, "1/h"),   # rate of G1 -> S/G2/M progression
        "k_division":(0.1, 3.0, "1/h"),   # rate of division (S/G2/M -> 2 G1 cells)
        "frac_s":    (0.01, 0.99, "dimensionless"),  # initial fraction in S/G2/M phase
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "cell cycle simplified to two phases: G1-like and S/G2/M-like",
            "cells progress G1 -> S/G2/M with first-order kinetics (rate k_g1)",
            "S/G2/M cells divide with first-order kinetics (rate k_division)",
            "division produces two G1 cells (binary fission)",
            "no death, no resource limitation in window",
            "counts are continuous approximation of integer cells",
            "cell-cycle times are exponentially distributed (memoryless)",
            "initial population has a fraction frac_s already in S/G2/M"
        ],
        "state_vars": ["N_G1", "N_S"],
        "refs": ["Smith & Martin 1973", "Cooper 1991"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        logy = np.log(y)
        
        # Model equations:
        # dN_G1/dt = -k_g1*N_G1 + 2*k_division*N_S
        # dN_S/dt  =  k_g1*N_G1 - k_division*N_S
        # Initial: N_G1(0) = n0*(1-frac_s), N_S(0) = n0*frac_s
        
        def model(theta, t):
            n0, k_g1, k_div, frac_s = theta
            def rhs(t, z):
                Ng, Ns = z
                dNg = -k_g1*Ng + 2*k_div*Ns
                dNs =  k_g1*Ng - k_div*Ns
                return [dNg, dNs]
            z0 = [n0*(1-frac_s), n0*frac_s]
            sol = solve_ivp(rhs, [t[0], t[-1]], z0, t_eval=t,
                           method='LSODA', rtol=1e-8, atol=1e-10)
            return sol.y[0] + sol.y[1]
        
        # Initial guess: effective growth rate ~ late-time slope, but cell cycle
        # rates should be roughly 2-3x the effective growth rate (since each division
        # requires progression through both phases).
        n_late = max(3, int(0.2*len(t_h)))
        late_mask = np.ones_like(t_h, dtype=bool)
        late_mask[:len(t_h)-n_late] = False
        if np.sum(late_mask) > 2:
            A = np.vstack([np.ones_like(t_h[late_mask]), t_h[late_mask]]).T
            coef, _, _, _ = np.linalg.lstsq(A, logy[late_mask], rcond=None)
            mu_eff = np.clip(coef[1], 0.2, 1.5)
        else:
            A = np.vstack([np.ones_like(t_h), t_h]).T
            coef, _, _, _ = np.linalg.lstsq(A, logy, rcond=None)
            mu_eff = np.clip(coef[1], 0.2, 1.5)
        
        n0_init = np.clip(y[0], 10, 1e4)
        k_init = np.clip(mu_eff * 2.0, 0.3, 2.0)
        frac_init = 0.3
        
        lb = [1.0, 0.1, 0.1, 0.01]
        ub = [1e4, 3.0, 3.0, 0.99]
        p0 = [n0_init, k_init, k_init, frac_init]
        
        def residual(theta):
            n0, k_g1, k_div, frac_s = theta
            if k_g1 <= 0 or k_div <= 0 or frac_s <= 0 or frac_s >= 1:
                return np.ones_like(logy)*1e6
            try:
                N_pred = model(theta, t_h)
            except Exception:
                return np.ones_like(logy)*1e6
            N_pred = np.maximum(N_pred, 1e-6)
            return logy - np.log(N_pred)
        
        best_res = None
        best_theta = None
        rng = np.random.default_rng(456)
        for restart in range(20):
            if restart > 0:
                p0_pert = [
                    np.clip(n0_init * np.exp(rng.normal(0, 0.15)), lb[0], ub[0]),
                    np.clip(k_init * np.exp(rng.normal(0, 0.2)), lb[1], ub[1]),
                    np.clip(k_init * np.exp(rng.normal(0, 0.2)), lb[2], ub[2]),
                    np.clip(frac_init * np.exp(rng.normal(0, 0.3)), lb[3], ub[3]),
                ]
            else:
                p0_pert = p0
            try:
                res = least_squares(residual, p0_pert, bounds=(lb, ub), max_nfev=5000)
                if best_res is None or res.cost < best_res.cost:
                    best_res = res
                    best_theta = res.x
            except Exception:
                continue
        
        if best_theta is None:
            best_theta = [n0_init, k_init, k_init, frac_init]
        
        n0, k_g1, k_div, frac_s = best_theta
        n0 = float(np.clip(n0, lb[0], ub[0]))
        k_g1 = float(np.clip(k_g1, lb[1], ub[1]))
        k_div = float(np.clip(k_div, lb[2], ub[2]))
        frac_s = float(np.clip(frac_s, lb[3], ub[3]))
        
        # Ensure physically meaningful: both rates positive, frac in (0,1)
        if k_g1 <= 0 or k_div <= 0:
            k_g1 = max(k_g1, 0.1)
            k_div = max(k_div, 0.1)
        frac_s = min(max(frac_s, 0.01), 0.99)
        
        return {
            "n0": n0,
            "k_g1": k_g1,
            "k_division": k_div,
            "frac_s": frac_s,
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        k_g1 = params["k_g1"]
        k_div = params["k_division"]
        frac_s = params["frac_s"]
        t_h = time_s / 3600.0
        
        def rhs(t, z):
            Ng, Ns = z
            dNg = -k_g1*Ng + 2*k_div*Ns
            dNs =  k_g1*Ng - k_div*Ns
            return [dNg, dNs]
        
        z0 = [n0*(1-frac_s), n0*frac_s]
        sol = solve_ivp(rhs, [t_h[0], t_h[-1]], z0, t_eval=t_h,
                       method='LSODA', rtol=1e-8, atol=1e-10)
        Ntot = sol.y[0] + sol.y[1]
        Ntot = np.maximum(Ntot, 0)
        return Ntot
```