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


def _predict_channels(twin, params, times_s, exog=None) -> dict:
    import inspect

    t = np.asarray(times_s, dtype=float)
    try:
        n = len(inspect.signature(twin.predict).parameters)
    except (TypeError, ValueError):
        n = 2
    raw = twin.predict(params, t, exog or {}) if n >= 3 else twin.predict(params, t)
    if isinstance(raw, dict):
        return {k: np.asarray(v, dtype=float) for k, v in raw.items()}
    return {"population_count": np.asarray(raw, dtype=float)}


def forecast_plot(
    source: str,
    params: dict,
    dataset: Dataset,
    out_path: str | Path,
    *,
    title: str = "",
) -> Path:
    """Forecast-vs-actual, one panel per scored channel (series tasks only)."""
    twin = _load_twin(source)
    obs = dataset.observations
    t_split = dataset.split["t_split_s"]
    per_s, per_lbl = dataset.task.period_s, dataset.task.period_label
    channels = [c for c in dataset.task.observables if c in obs.channels()]
    if not channels:
        channels = ["population_count"]

    if dataset.task.exogenous:  # stateful w/ external inputs: predict on the real grid
        dense_s = obs.time_s
        exog = {k: obs.channel(k) for k in dataset.task.exogenous if k in obs.channels()}
    else:
        dense_s = np.linspace(float(obs.time_s[0]), float(obs.time_s[-1]), 300)
        exog = None
    pred_dense = _predict_channels(twin, params, dense_s, exog)

    ncol = min(len(channels), 2)
    nrow = (len(channels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.6 * nrow), squeeze=False)
    fit_mask = obs.time_s < t_split

    for ax, ch in zip(axes.ravel(), channels):
        y = obs.channel(ch)
        ax.scatter((obs.time_s[fit_mask] / per_s), y[fit_mask], s=13, color="#1b4965",
                   alpha=0.55, label="observed (fit)", zorder=2)
        ax.scatter((obs.time_s[~fit_mask] / per_s), y[~fit_mask], s=13, color="#c1121f",
                   alpha=0.55, label="observed (holdout)", zorder=2)
        if ch in pred_dense:
            ax.plot(dense_s / per_s, pred_dense[ch], color="#5fa8d3", lw=2.2,
                    label="twin", zorder=4)
        ax.axvline(t_split / per_s, color="#888", ls="--", lw=1)
        if ch in ("population_count", "total_length_um", "total_area_um2", "mrr"):
            ax.set_yscale("log")
        ax.set_xlabel(f"time ({per_lbl})")
        ax.set_ylabel(ch)
        ax.legend(frameon=False, fontsize=7)
    for ax in axes.ravel()[len(channels):]:
        ax.set_visible(False)

    fig.suptitle(title or f"{dataset.name}: forecast vs actual", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=118)
    plt.close(fig)
    return out_path
