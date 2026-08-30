"""Forecast-vs-actual plots for a candidate twin."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from medusa.data.build import Dataset  # noqa: E402
from medusa.harness.evaluate import _load_twin  # noqa: E402


def sanity_plot(dataset: Dataset, out_path: str | Path) -> Path:
    obs = dataset.observations
    t_split = dataset.split["t_split_s"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(obs.time_h, obs.population_count, "o-", ms=3, lw=1, color="#1b4965")
    ax.axvline(t_split / 3600.0, color="#888", ls="--", lw=1, label="fit / holdout split")
    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("population count")
    ax.set_title(f"{dataset.name}: canonical growth signal")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def forecast_plot(
    source: str,
    params: dict,
    dataset: Dataset,
    out_path: str | Path,
    *,
    title: str = "",
) -> Path:
    twin = _load_twin(source)
    obs = dataset.observations
    t_split = dataset.split["t_split_s"]

    dense_s = np.linspace(float(obs.time_s[0]), float(obs.time_s[-1]), 400)
    curve = np.asarray(twin.predict(params, dense_s), dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fit_mask = obs.time_s < t_split
    ax.scatter(
        obs.time_h[fit_mask], obs.population_count[fit_mask],
        s=14, color="#1b4965", alpha=0.55, label="observed (fit)", zorder=2,
    )
    ax.scatter(
        obs.time_h[~fit_mask], obs.population_count[~fit_mask],
        s=14, color="#c1121f", alpha=0.55, label="observed (holdout)", zorder=2,
    )
    ax.plot(dense_s / 3600.0, curve, color="#5fa8d3", lw=2.4, label="twin", zorder=4)
    ax.axvline(t_split / 3600.0, color="#888", ls="--", lw=1, label="fit / holdout split")

    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("population count")
    ax.set_title(title or f"{dataset.name}: forecast vs actual")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
