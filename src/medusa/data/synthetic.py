"""Synthetic growth datasets with known ground truth.

Used to (a) bootstrap and test the whole pipeline before wiring a real download, and
(b) provide labelled entries for the benchmark suite. Each preset carries the true
doubling time so the plausibility metric has a target even for synthetic data.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from medusa.contract.interface import Observations


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticPreset:
    name: str
    organism: str
    n0: float
    mu_max_per_h: float  # true max specific growth rate
    carrying_capacity: float
    lag_h: float
    duration_h: float
    frame_interval_s: float
    noise_cv: float  # multiplicative noise coefficient of variation
    seed: int

    @property
    def doubling_time_h(self) -> float:
        return float(np.log(2.0) / self.mu_max_per_h)


PRESETS: dict[str, SyntheticPreset] = {
    "synthetic-ecoli-fast": SyntheticPreset(
        name="synthetic-ecoli-fast",
        organism="E. coli (synthetic)",
        n0=2.0,
        mu_max_per_h=2.0,
        carrying_capacity=4.0e4,
        lag_h=0.6,
        duration_h=13.0,
        frame_interval_s=180.0,
        noise_cv=0.08,
        seed=1,
    ),
    "synthetic-yeast-slow": SyntheticPreset(
        name="synthetic-yeast-slow",
        organism="S. cerevisiae (synthetic)",
        n0=5.0,
        mu_max_per_h=0.45,
        carrying_capacity=1.2e4,
        lag_h=2.0,
        duration_h=44.0,
        frame_interval_s=600.0,
        noise_cv=0.10,
        seed=2,
    ),
    "synthetic-bsub-mid": SyntheticPreset(
        name="synthetic-bsub-mid",
        organism="B. subtilis (synthetic)",
        n0=1.0,
        mu_max_per_h=1.1,
        carrying_capacity=8.0e4,
        lag_h=1.0,
        duration_h=24.0,
        frame_interval_s=300.0,
        noise_cv=0.09,
        seed=3,
    ),
}


def _baranyi(t_h: np.ndarray, preset: SyntheticPreset) -> np.ndarray:
    """Baranyi-Roberts growth: lag -> exponential -> saturation (mechanistic, smooth)."""
    mu = preset.mu_max_per_h
    n0, k, lag = preset.n0, preset.carrying_capacity, preset.lag_h
    # adjustment function A(t): time spent adjusting out of lag
    h0 = mu * lag
    a = t_h + (1.0 / mu) * np.log(np.exp(-mu * t_h) + np.exp(-h0) - np.exp(-mu * t_h - h0))
    exp_term = np.exp(mu * a)
    y = n0 * exp_term / (1.0 + (n0 * (exp_term - 1.0)) / k)
    return y


def generate(preset: SyntheticPreset) -> Observations:
    rng = np.random.default_rng(preset.seed)
    t_s = np.arange(0.0, preset.duration_h * 3600.0 + 1.0, preset.frame_interval_s)
    t_h = t_s / 3600.0
    true_counts = _baranyi(t_h, preset)

    # counting noise: Poisson on the integer count + a small multiplicative component
    multiplicative = rng.lognormal(mean=0.0, sigma=preset.noise_cv, size=t_s.shape)
    noisy = rng.poisson(np.clip(true_counts * multiplicative, 0.0, None)).astype(float)
    noisy = np.clip(noisy, 1.0, None)

    # crude biomass proxy: mean cell area ~ constant with mild crowding shrinkage
    mean_area = 3.0 * (1.0 - 0.15 * true_counts / preset.carrying_capacity)
    total_area = noisy * mean_area

    return Observations(
        time_s=t_s, population_count=noisy, extra={"total_area_um2": total_area}
    )


def datasheet(preset: SyntheticPreset) -> str:
    return f"""\
# Datasheet: {preset.name}

- **Source:** synthetic (medusa.data.synthetic), Baranyi-Roberts model + Poisson/lognormal noise
- **Organism:** {preset.organism}
- **Frame interval:** {preset.frame_interval_s:.0f} s
- **Duration:** {preset.duration_h:.1f} h
- **Reduction:** direct model output; `population_count` is the simulated cell count,
  `total_area_um2` a biomass proxy (count x mean cell area).
- **Ground-truth doubling time:** {preset.doubling_time_h:.2f} h
  (mu_max = {preset.mu_max_per_h:.3f} / h); plausible range for the metric: \
[{preset.doubling_time_h * 0.6:.2f}, {preset.doubling_time_h * 1.6:.2f}] h
- **Ground-truth carrying capacity:** {preset.carrying_capacity:.3g} cells
- **Ground-truth lag:** {preset.lag_h:.2f} h
- **Note:** synthetic data is for pipeline validation and the benchmark suite; the
  headline demo runs on real public microscopy (see data/fetch.py).
"""
