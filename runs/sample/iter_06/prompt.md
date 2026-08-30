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
| two-reservoir-monod | 0.096 | 1.00 | ok |

Best valid holdout sMAPE so far: 0.060. Distinct valid families: 4.

## Your previous attempt

```python
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

class Twin:
    # Mechanistic: two-reservoir model with explicit lag via "maturation" of a
    # bottleneck enzyme. Cells transition from lag to exponential growth when a
    # critical enzyme pool is built. Growth is then limited by a Monod-type
    # resource depletion, but we model the resource implicitly via a carrying
    # capacity with a smooth switch. This is distinct from Baranyi-Richards
    # because the lag is driven by a separate state variable (enzyme) with its
    # own dynamics, and the growth term uses a Monod form with a time-varying
    # substrate that is consumed proportionally to growth.
    FAMILY = "two-reservoir-monod"
    
    PARAMS = {
        "n0": (0.5, 100.0, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "lag": (0.0, 3.0, "h"),
        "carrying_capacity": (1e3, 1e6, "cells"),
        "k_s": (1e2, 1e6, "cells"),  # Monod half-saturation in terms of equivalent population
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed",
            "single limiting resource",
            "lag phase due to accumulation of a bottleneck enzyme",
            "growth rate follows Monod kinetics on a resource that is depleted proportionally to biomass",
            "resource pool is implicit in carrying capacity",
            "no death, no maintenance"
        ],
        "state_vars": ["N", "E", "S"],
        "refs": ["Monod 1949", "Baranyi & Roberts 1994 (lag concept)"],
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
            self.PARAMS["k_s"][:2],
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
        
        # Lag estimate: time when log(y) first exceeds log(n0)+0.5
        lag_guess = 0.5
        threshold = np.log(n0_guess) + 0.5
        idx = np.where(np.log(y+1) > threshold)[0]
        if len(idx) > 0:
            lag_guess = max(0.0, t_h[idx[0]] - 0.2)
        lag_guess = np.clip(lag_guess, *self.PARAMS["lag"][:2])
        
        K_guess = np.max(y) * 1.2
        K_guess = np.clip(K_guess, *self.PARAMS["carrying_capacity"][:2])
        k_s_guess = K_guess * 0.1
        k_s_guess = np.clip(k_s_guess, *self.PARAMS["k_s"][:2])
        
        x0 = [n0_guess, mu_est, lag_guess, K_guess, k_s_guess]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def residuals(params):
            n0, mu_max, lag, K, k_s = params
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            lag = np.clip(lag, *self.PARAMS["lag"][:2])
            K = np.clip(K, *self.PARAMS["carrying_capacity"][:2])
            k_s = np.clip(k_s, *self.PARAMS["k_s"][:2])
            
            p = {
                "n0": n0, "mu_max": mu_max, "lag": lag,
                "carrying_capacity": K, "k_s": k_s
            }
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 1e-6)
                pred_log = np.log(pred)
                # Weighted log residuals, more weight on early low counts
                return (pred_log - y_log) / np.sqrt(y + 1.0)
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # Multi-start least_squares for robustness
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
            # Fallback to simple grid search on key params
            mu_grid = np.linspace(0.5, 3.5, 10)
            lag_grid = np.linspace(0.0, 2.0, 10)
            K_grid = np.geomspace(1e3, 1e6, 10)
            best_res = np.inf
            best_x = x0
            for mu in mu_grid:
                for lag in lag_grid:
                    for K in K_grid:
                        params = [n0_guess, mu, lag, K, k_s_guess]
                        res = np.sum(residuals(params)**2)
                        if res < best_res:
                            best_res = res
                            best_x = params
        
        best_x = np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best_x[0],
            "mu_max": best_x[1],
            "lag": best_x[2],
            "carrying_capacity": best_x[3],
            "k_s": best_x[4],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        lag = params["lag"]
        K = params["carrying_capacity"]
        k_s = params["k_s"]
        
        # Two-reservoir model:
        # dN/dt = mu_max * (E/(E+1)) * (S/(S+k_s)) * N
        # dE/dt = mu_max * E  (exponential buildup of enzyme)
        # dS/dt = - (1/Y) * dN/dt  but we use a Monod term with S starting at K
        # To keep it simple, we model S as a resource reservoir that is depleted
        # proportionally to growth: S(t) = K - N(t) (since N consumes resource)
        # But then Monod term becomes (K-N)/(K-N+k_s). This is equivalent to
        # a logistic-like term but with a different shape. We'll use that.
        #
        # Enzyme E starts at E0 = 1/(exp(mu_max*lag)-1) so that E/(E+1) ~ 0
        # initially and reaches ~1 after lag.
        
        E0 = 1.0 / (np.exp(mu_max * lag) - 1.0) if lag > 0 else 1e10
        
        def rhs(t, state):
            N, E = state
            if N <= 0 or E <= 0:
                return [0.0, 0.0]
            N_safe = min(N, K * 1.1)
            S = max(K - N_safe, 0.0)
            monod = S / (S + k_s) if S + k_s > 0 else 0.0
            enzyme_factor = E / (E + 1.0)
            growth = mu_max * enzyme_factor * monod * N_safe
            dE = mu_max * E
            return [growth, dE]
        
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        sol = solve_ivp(
            rhs, t_span, [n0, E0],
            t_eval=t_h,
            method='LSODA',
            rtol=1e-10, atol=1e-10,
            max_step=0.02,
        )
        
        if not sol.success:
            # Fallback: logistic with lag shift
            t_shift = t_h - lag
            t_shift = np.maximum(t_shift, 0.0)
            N_pred = K / (1 + (K/n0 - 1) * np.exp(-mu_max * t_shift))
            return np.maximum(N_pred, 0.0)
        
        N = sol.y[0]
        N = np.maximum(N, 0.0)
        N = np.minimum(N, K * 1.01)
        return N
```

family: two-reservoir-monod
valid: True  |  holdout sMAPE: 0.0964  |  fit R^2: 0.9857  |  implied doubling time: 0.36 h  |  plausibility: 1.00

**This round: propose a mechanistic family NOT yet in the list above** (or a materially different structure), even if it scores a little worse -- we want a diverse portfolio of candidate twins, not one family re-tuned.

Write the next `twin.py`. One ```python block, nothing else.
