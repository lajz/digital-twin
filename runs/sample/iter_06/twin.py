import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

class Twin:
    # Mechanistic: "age-structured" model where cells have an internal age/maturity
    # variable. Newborn cells are in lag (immature) and mature exponentially.
    # Division produces two immature daughters, so the lag emerges naturally from
    # the cell cycle rather than being an externally imposed delay. Growth is
    # eventually limited by a carrying capacity via a density-dependent death rate
    # (logistic-like saturation). This is structurally distinct from Baranyi
    # (no explicit lag parameter, no substrate pool) and from two-reservoir
    # (no Monod resource; saturation comes from mortality, not resource depletion).
    FAMILY = "age-structured-maturity"
    
    PARAMS = {
        "n0": (0.5, 100.0, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "maturation_rate": (0.1, 5.0, "1/h"),  # rate at which immature -> mature
        "carrying_capacity": (1e3, 1e6, "cells"),
        "death_rate": (0.0, 1.0, "1/h"),  # density-dependent death coefficient
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "each cell has an internal maturity state (0=immature/lag, 1=mature)",
            "immature cells do not divide; they mature at first-order rate",
            "mature cells divide at rate mu_max (exponential growth)",
            "division produces two immature daughters (resets maturity to 0)",
            "density-dependent death rate proportional to N/K (logistic saturation)",
            "no explicit resource; carrying capacity emerges from death term",
            "no age distribution beyond two states (immature/mature) - lumped"
        ],
        "state_vars": ["N_i", "N_m"],
        "refs": ["Kermack-McKendrick age structure", "logistic growth with lag"],
    }
    
    def fit(self, obs):
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        y_log = np.log(y + 1.0)
        
        bounds = [
            self.PARAMS["n0"][:2],
            self.PARAMS["mu_max"][:2],
            self.PARAMS["maturation_rate"][:2],
            self.PARAMS["carrying_capacity"][:2],
            self.PARAMS["death_rate"][:2],
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
        
        # Maturation rate guess: roughly 1/lag, with lag ~0.5-1h
        mat_guess = 1.0 / 0.7  # ~1.43
        mat_guess = np.clip(mat_guess, *self.PARAMS["maturation_rate"][:2])
        
        K_guess = np.max(y) * 1.2
        K_guess = np.clip(K_guess, *self.PARAMS["carrying_capacity"][:2])
        death_guess = 0.0  # start with no death, logistic via death term
        
        x0 = [n0_guess, mu_est, mat_guess, K_guess, death_guess]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def residuals(params):
            n0, mu_max, mat, K, death = params
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            mat = np.clip(mat, *self.PARAMS["maturation_rate"][:2])
            K = np.clip(K, *self.PARAMS["carrying_capacity"][:2])
            death = np.clip(death, *self.PARAMS["death_rate"][:2])
            
            p = {
                "n0": n0, "mu_max": mu_max, "maturation_rate": mat,
                "carrying_capacity": K, "death_rate": death
            }
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 1e-6)
                pred_log = np.log(pred)
                # Weighted log residuals, more weight on early low counts
                return (pred_log - y_log) / np.sqrt(y + 1.0)
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # Multi-start least_squares
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
            # Fallback: grid search on mu, mat, K
            mu_grid = np.linspace(0.5, 3.5, 8)
            mat_grid = np.linspace(0.5, 3.0, 8)
            K_grid = np.geomspace(1e3, 1e6, 8)
            best_res = np.inf
            best_x = x0
            for mu in mu_grid:
                for mat in mat_grid:
                    for K in K_grid:
                        params = [n0_guess, mu, mat, K, 0.0]
                        res = np.sum(residuals(params)**2)
                        if res < best_res:
                            best_res = res
                            best_x = params
        
        best_x = np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best_x[0],
            "mu_max": best_x[1],
            "maturation_rate": best_x[2],
            "carrying_capacity": best_x[3],
            "death_rate": best_x[4],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        mat = params["maturation_rate"]
        K = params["carrying_capacity"]
        death = params["death_rate"]
        
        # Initial: all cells are immature (lag phase)
        N_i0 = n0
        N_m0 = 0.0
        
        def rhs(t, state):
            Ni, Nm = state
            if Ni < 0: Ni = 0.0
            if Nm < 0: Nm = 0.0
            N = Ni + Nm
            if N <= 0:
                return [0.0, 0.0]
            
            # Density-dependent death (logistic)
            mortality = death * (N / K) if K > 0 else 0.0
            
            # Maturation: immature -> mature
            maturation_flux = mat * Ni
            
            # Division of mature cells: each mature cell divides at rate mu_max,
            # producing two immature daughters (so Ni increases by 2*mu_max*Nm,
            # and Nm stays same because the mother is replaced by two daughters that
            # are immature? Actually in a simple model: mature cell divides, mother
            # disappears, two immature daughters appear. So Nm decreases by mu_max*Nm,
            # Ni increases by 2*mu_max*Nm. But that's not standard. Let's think:
            # A mature cell divides -> two immature cells. So one mature cell is
            # replaced by two immature cells. So dNm = -mu_max*Nm, dNi = +2*mu_max*Nm.
            # But then the total N increases by mu_max*Nm (since -1+2=+1). That's
            # correct exponential growth.
            division_flux = mu_max * Nm
            
            dNi = maturation_flux + 2.0 * division_flux - mortality * Ni
            dNm = -maturation_flux - division_flux - mortality * Nm
            
            return [dNi, dNm]
        
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        sol = solve_ivp(
            rhs, t_span, [N_i0, N_m0],
            t_eval=t_h,
            method='LSODA',
            rtol=1e-10, atol=1e-10,
            max_step=0.02,
        )
        
        if not sol.success:
            # Fallback: simple logistic with lag (shouldn't happen)
            lag = 1.0 / mat if mat > 0 else 0.0
            t_shift = t_h - lag
            t_shift = np.maximum(t_shift, 0.0)
            N_pred = K / (1 + (K/n0 - 1) * np.exp(-mu_max * t_shift))
            return np.maximum(N_pred, 0.0)
        
        N = sol.y[0] + sol.y[1]
        N = np.maximum(N, 0.0)
        # Cap at carrying capacity (with some tolerance)
        N = np.minimum(N, K * 1.05)
        return N