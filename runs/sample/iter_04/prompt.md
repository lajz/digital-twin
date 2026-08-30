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

Best valid holdout sMAPE so far: 0.060. Distinct valid families: 2.

## Your previous attempt

```python
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
```

family: monod-substrate
valid: True  |  holdout sMAPE: 0.1584  |  fit R^2: 0.9551  |  implied doubling time: 0.37 h  |  plausibility: 1.00



Write the next `twin.py`. One ```python block, nothing else.
