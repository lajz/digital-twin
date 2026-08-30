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

Distinct valid families: 0.

## Your previous attempt

```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

class Twin:
    FAMILY = "baranyi-robust"
    
    PARAMS = {
        "n0": (0.5, 100, "cells"),
        "mu_max": (0.5, 3.5, "1/h"),
        "lag": (0.0, 3.0, "h"),
        "carrying_capacity": (1e3, 1e6, "cells"),
    }
    
    METADATA = {
        "assumptions": [
            "well-mixed", 
            "single limiting nutrient",
            "Baranyi-Roberts lag phase with adjustment function",
            "log-normal multiplicative noise on observations"
        ],
        "state_vars": ["N"],
        "refs": ["Baranyi & Roberts 1994", "Buchanan et al. 1997"],
    }
    
    def fit(self, obs):
        t_h = obs.time_s / 3600.0
        y = obs.population_count.astype(float)
        # use log-space residuals to handle range of magnitudes
        y_log = np.log(y + 1.0)
        
        bounds = [
            self.PARAMS["n0"][:2],
            self.PARAMS["mu_max"][:2],
            self.PARAMS["lag"][:2],
            self.PARAMS["carrying_capacity"][:2],
        ]
        
        def residuals(params):
            n0, mu_max, lag, K = params
            # clamp to bounds for safety
            n0 = np.clip(n0, *self.PARAMS["n0"][:2])
            mu_max = np.clip(mu_max, *self.PARAMS["mu_max"][:2])
            lag = np.clip(lag, *self.PARAMS["lag"][:2])
            K = np.clip(K, *self.PARAMS["carrying_capacity"][:2])
            
            p = {"n0": n0, "mu_max": mu_max, "lag": lag, "carrying_capacity": K}
            try:
                pred = self.predict(p, t_h * 3600.0)
                pred = np.maximum(pred, 0.0)
                pred_log = np.log(pred + 1.0)
                return pred_log - y_log
            except Exception:
                return np.ones_like(y_log) * 1e6
        
        # differential evolution for global search, then refine with local
        result = differential_evolution(
            lambda p: np.sum(residuals(p)**2),
            bounds,
            seed=42,
            maxiter=300,
            popsize=15,
            tol=1e-10,
            polish=True,
            workers=1,
        )
        
        best = result.x
        # ensure within bounds
        best = np.clip(best, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return {
            "n0": best[0],
            "mu_max": best[1],
            "lag": best[2],
            "carrying_capacity": best[3],
        }
    
    def predict(self, params, time_s):
        t_h = np.asarray(time_s, dtype=float) / 3600.0
        n0 = params["n0"]
        mu_max = params["mu_max"]
        lag = params["lag"]
        K = params["carrying_capacity"]
        
        # Baranyi-Roberts model with adjustment function
        # dN/dt = mu_max * alpha(t) * N * (1 - N/K)
        # alpha(t) = 1/(1 + exp(-q(t))), q(t) = q0 - mu_max*t
        # q0 = -mu_max * lag  (so alpha ~ 0 until t≈lag)
        # But for simplicity, use explicit lag: alpha = 0 for t<lag, then 1
        # More robust: use smooth transition via logistic
        # alpha(t) = 1/(1 + exp(-(t-lag)*10))  # sharp but smooth
        
        # Use ODE integration for accuracy
        def rhs(t, N):
            # smooth adjustment function, sharp transition
            alpha = 1.0 / (1.0 + np.exp(-(t - lag) * 20.0))
            if N <= 0:
                return 0.0
            return mu_max * alpha * N * (1.0 - N / K)
        
        # integrate from t=0 to max time
        t_span = (0.0, t_h[-1] if len(t_h) > 0 else 0.0)
        if t_span[1] <= 0:
            return np.array([n0])
        
        # use dense output for arbitrary times
        sol = solve_ivp(
            rhs, t_span, [n0], 
            t_eval=t_h, 
            method='LSODA',
            rtol=1e-8, atol=1e-8,
            max_step=0.05,
        )
        
        if not sol.success:
            # fallback: return constant
            return np.full_like(t_h, n0)
        
        N = sol.y[0]
        # ensure non-negative
        N = np.maximum(N, 0.0)
        return N
```

family: baranyi-robust
valid: False  |  holdout sMAPE: 0.0599  |  fit R^2: 0.9907  |  implied doubling time: 0.36 h  |  plausibility: 1.00



Write the next `twin.py`. One ```python block, nothing else.
