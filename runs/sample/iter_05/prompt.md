# SYSTEM

You are a computational systems-biology modeller. Your job: write a Python file
`twin.py` that is a *digital twin* of a well-mixed single-cell microbial population --
a mechanistic model of how the cell count grows over time -- plus the code that
calibrates it to data.

You will be given real observations (time, population count) for a FIT window only.
A held-out future window is used to score your forecast; you never see it. Each round
you get your previous attempt and its scores, and a summary of model families tried so
far. Improve the forecast, and explore genuinely different mechanistic families over
the run.

## Hard requirements (a violation = your submission is rejected, unscored)

1. Output EXACTLY ONE fenced ```python code block and nothing else. It is the whole
   content of `twin.py`.
2. Imports: only the standard library, `numpy`, and `scipy`. NO `sklearn`, `torch`,
   `tensorflow`, `jax`, `keras`, `statsmodels`, `pandas`, network access, or file I/O.
3. The model must be MECHANISTIC: every parameter has a physical/biological meaning,
   a unit, and a plausible prior range. No black-box regressors, no fitting an
   arbitrary polynomial / spline / neural net to the curve.
4. Deterministic: `predict(params, t)` called twice with the same arguments returns
   identical arrays. Seed any RNG you use.
5. `fit` + `predict` together must run in under 5 seconds.

## Required interface

```python
import numpy as np

class Twin:
    # short slug naming the mechanistic family, e.g. "logistic", "gompertz",
    # "baranyi", "richards", "monod-consumer", "two-stage-lag". Used to group results.
    FAMILY = "logistic"

    # name -> (prior_low, prior_high, unit)
    PARAMS = {
        "n0":     (1.0, 1e4, "cells"),
        "mu_max": (0.05, 3.0, "1/h"),
        "carrying_capacity": (1e3, 1e8, "cells"),
    }

    METADATA = {
        "assumptions": ["well-mixed", "single limiting resource", "..."],
        "state_vars": ["N"],
        "refs": ["Verhulst 1838", "..."],
    }

    def fit(self, obs) -> dict[str, float]:
        # obs.time_s : np.ndarray[float]  seconds since first frame
        # obs.population_count : np.ndarray[float]
        # obs.total_area_um2 : np.ndarray[float] | None
        # Return a dict with EXACTLY the keys of PARAMS, each within its prior range.
        ...

    def predict(self, params: dict, time_s: np.ndarray) -> np.ndarray:
        # Return predicted population_count at the given times. Same length as time_s.
        ...
```

Notes:
- `time_s` is in seconds; growth rates are conventionally per hour -- convert.
- You own the calibration. Use `scipy.optimize` (least_squares, differential_evolution,
  minimize) and/or ODE integration (`scipy.integrate.solve_ivp`). Respect PARAMS bounds.
- Prefer fitting in log space or with relative error -- counts span orders of magnitude.
- Make `fit` robust: clamp to bounds, guard against overflow, return finite values.


# USER

## Dataset

# Datasheet: synthetic-ecoli-fast

- **Source:** synthetic (medusa.data.synthetic), Baranyi-Roberts model + Poisson/lognormal noise
- **Organism:** E. coli (synthetic)
- **Frame interval:** 180 s
- **Duration:** 13.0 h
- **Reduction:** direct model output; `population_count` is the simulated cell count,
  `total_area_um2` a biomass proxy (count x mean cell area).
- **Ground-truth doubling time:** 0.35 h
  (mu_max = 2.000 / h); plausible range for the metric: [0.21, 0.55] h
- **Ground-truth carrying capacity:** 4e+04 cells
- **Ground-truth lag:** 0.60 h
- **Note:** synthetic data is for pipeline validation and the benchmark suite; the
  headline demo runs on real public microscopy (see data/fetch.py).

## Observations (FIT window only, downsampled)

| time_h | population_count | total_area_um2 |
|---|---|---|
| 0.000 | 1 | 3.0 |
| 0.100 | 3 | 9.0 |
| 0.250 | 2 | 6.0 |
| 0.350 | 1 | 3.0 |
| 0.500 | 1 | 3.0 |
| 0.650 | 4 | 12.0 |
| 0.750 | 4 | 12.0 |
| 0.900 | 3 | 9.0 |
| 1.050 | 6 | 18.0 |
| 1.150 | 6 | 18.0 |
| 1.300 | 10 | 30.0 |
| 1.400 | 9 | 27.0 |
| 1.550 | 11 | 33.0 |
| 1.700 | 22 | 66.0 |
| 1.800 | 29 | 87.0 |
| 1.950 | 28 | 84.0 |
| 2.100 | 37 | 111.0 |
| 2.200 | 53 | 159.0 |
| 2.350 | 55 | 165.0 |
| 2.450 | 94 | 281.9 |
| 2.600 | 104 | 311.9 |
| 2.750 | 144 | 431.8 |
| 2.850 | 189 | 566.6 |
| 3.000 | 266 | 797.3 |
| 3.150 | 324 | 970.8 |
| 3.250 | 371 | 1111.3 |
| 3.400 | 449 | 1344.3 |
| 3.500 | 743 | 2223.6 |
| 3.650 | 877 | 2622.4 |
| 3.800 | 1125 | 3360.2 |
| 3.900 | 1428 | 4261.2 |
| 4.050 | 1918 | 5713.2 |
| 4.200 | 2711 | 8056.4 |
| 4.300 | 2848 | 8447.1 |
| 4.450 | 4256 | 12577.5 |
| 4.550 | 4899 | 14434.9 |
| 4.700 | 6835 | 20031.2 |
| 4.850 | 7876 | 22928.8 |
| 4.950 | 9310 | 26962.7 |
| 5.100 | 11427 | 32798.2 |
| 5.250 | 14543 | 41315.1 |
| 5.350 | 14273 | 40246.7 |
| 5.500 | 18121 | 50496.4 |
| 5.600 | 20747 | 57347.6 |
| 5.750 | 24102 | 65821.6 |
| 5.900 | 24451 | 66009.3 |
| 6.000 | 32014 | 85810.0 |
| 6.150 | 32494 | 86253.3 |
| 6.300 | 29314 | 77163.9 |
| 6.400 | 31746 | 83165.4 |
| 6.550 | 36581 | 95249.7 |
| 6.650 | 36850 | 95626.7 |
| 6.800 | 36009 | 93056.1 |
| 6.950 | 34633 | 89210.4 |
| 7.050 | 42769 | 109976.7 |
| 7.200 | 40260 | 103309.6 |
| 7.350 | 40123 | 102795.5 |
| 7.450 | 35538 | 90973.0 |
| 7.600 | 36214 | 92612.3 |
| 7.750 | 44965 | 114907.0 |

## Model families tried so far

| family | best holdout sMAPE | plausibility | status (latest) |
|---|---|---|---|
| baranyi-robust | -- | -- | over time budget |
| baranyi-robust-fixed | 0.060 | 1.00 | ok |
| monod-substrate | 0.158 | 1.00 | ok |
| baranyi-richards | 0.244 | 1.00 | ok |

Best valid holdout sMAPE so far: 0.060. Distinct valid families: 3.

## Your previous attempt

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

family: baranyi-richards
valid: True  |  holdout sMAPE: 0.2436  |  fit R^2: 0.5657  |  implied doubling time: 0.35 h  |  plausibility: 1.00



Write the next `twin.py`. One ```python block, nothing else.
