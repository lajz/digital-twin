import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class Twin:
    FAMILY = "baranyi-adaptation"
    
    PARAMS = {
        "n0":     (1.0, 1e4, "cells"),
        "mu_max": (0.2, 2.0, "1/h"),
        "lag_time": (0.0, 1.5, "h"),
        "adapt_rate": (0.1, 5.0, "1/h"),
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "two physiological states: lag (N_l) and active (N_a)",
            "cells transition from lag to active with rate adapt_rate",
            "active cells grow exponentially with rate mu_max",
            "lag phase duration controlled by both lag_time and adapt_rate",
            "no death, no resource limitation in window",
            "counts are continuous approximation of integer cells"
        ],
        "state_vars": ["N_l", "N_a"],
        "refs": ["Baranyi & Roberts 1994", "Buchanan 1997"],
    }
    
    def fit(self, obs) -> dict[str, float]:
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        logy = np.log(y)
        
        # Model: dN_l/dt = -adapt_rate * N_l
        #        dN_a/dt = adapt_rate * N_l + mu_max * N_a
        # Initial: N_l(0) = n0, N_a(0) = 0 (all cells start in lag)
        # But we also have lag_time as a pre-incubation: start at t=-lag_time
        # with all cells in lag, then integrate forward.
        # This gives a proper lag phase with smooth transition.
        
        def model(theta, t):
            n0, mu, lag, adapt = theta
            t_start = -lag
            t_full = np.concatenate([[t_start], t])
            
            def rhs(t, z):
                N_l, N_a = z
                dN_l = -adapt * N_l
                dN_a = adapt * N_l + mu * N_a
                return [dN_l, dN_a]
            
            z0 = [n0, 0.0]
            sol = solve_ivp(rhs, [t_start, t_full[-1]], z0, t_eval=t_full,
                           method='LSODA', rtol=1e-8, atol=1e-10)
            N_total = sol.y[0] + sol.y[1]
            # Remove the first point (t_start)
            N_total = N_total[1:]
            return N_total
        
        # Initial guess from late-time exponential fit
        late_mask = t_h > 1.0
        if np.sum(late_mask) > 2:
            A_mat = np.vstack([np.ones_like(t_h[late_mask]), t_h[late_mask]]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, logy[late_mask], rcond=None)
            mu_init = np.clip(coef[1], 0.2, 2.0)
            n0_init = np.clip(np.exp(coef[0] - mu_init*1.0), 10, 1e4)
        else:
            A_mat = np.vstack([np.ones_like(t_h), t_h]).T
            coef, _, _, _ = np.linalg.lstsq(A_mat, logy, rcond=None)
            n0_init = np.clip(np.exp(coef[0]), 10, 1e4)
            mu_init = np.clip(coef[1], 0.2, 2.0)
        
        # Initial guess for lag and adaptation
        lag_init = 0.1
        adapt_init = 1.0
        
        lb = [1.0, 0.2, 0.0, 0.1]
        ub = [1e4, 2.0, 1.5, 5.0]
        p0 = [n0_init, mu_init, lag_init, adapt_init]
        
        def residual(theta):
            n0, mu, lag, adapt = theta
            try:
                N_pred = model(theta, t_h)
            except Exception:
                return np.ones_like(logy)*1e6
            N_pred = np.maximum(N_pred, 1e-6)
            return logy - np.log(N_pred)
        
        best_res = None
        best_theta = None
        rng = np.random.default_rng(42)
        for restart in range(10):
            if restart > 0:
                p0_pert = [
                    np.clip(n0_init * np.exp(rng.normal(0, 0.2)), lb[0], ub[0]),
                    np.clip(mu_init * np.exp(rng.normal(0, 0.1)), lb[1], ub[1]),
                    np.clip(lag_init * np.exp(rng.normal(0, 0.5)), lb[2], ub[2]),
                    np.clip(adapt_init * np.exp(rng.normal(0, 0.5)), lb[3], ub[3]),
                ]
            else:
                p0_pert = p0
            try:
                res = least_squares(residual, p0_pert, bounds=(lb, ub), max_nfev=3000)
                if best_res is None or res.cost < best_res.cost:
                    best_res = res
                    best_theta = res.x
            except Exception:
                continue
        
        if best_theta is None:
            best_theta = [n0_init, mu_init, 0.05, 1.0]
        
        n0, mu, lag, adapt = best_theta
        n0 = float(np.clip(n0, lb[0], ub[0]))
        mu = float(np.clip(mu, lb[1], ub[1]))
        lag = float(np.clip(lag, lb[2], ub[2]))
        adapt = float(np.clip(adapt, lb[3], ub[3]))
        
        return {
            "n0": n0,
            "mu_max": mu,
            "lag_time": lag,
            "adapt_rate": adapt,
        }
    
    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        n0 = params["n0"]
        mu = params["mu_max"]
        lag = params["lag_time"]
        adapt = params["adapt_rate"]
        t_h = time_s / 3600.0
        
        t_start = -lag
        t_full = np.concatenate([[t_start], t_h])
        
        def rhs(t, z):
            N_l, N_a = z
            dN_l = -adapt * N_l
            dN_a = adapt * N_l + mu * N_a
            return [dN_l, dN_a]
        
        z0 = [n0, 0.0]
        sol = solve_ivp(rhs, [t_start, t_full[-1]], z0, t_eval=t_full,
                       method='LSODA', rtol=1e-8, atol=1e-10)
        N_total = sol.y[0] + sol.y[1]
        N_total = N_total[1:]
        N_total = np.maximum(N_total, 0)
        return N_total