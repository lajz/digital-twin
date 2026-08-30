"""Reduce a dataset to the canonical growth signal and write the processed artifacts.

Canonical outputs (in data/processed/ by default):
  - observations.parquet : time_s, population_count, [total_area_um2]
  - split.json           : fit / holdout time cutoff
  - ../datasheet.md       : provenance + ground-truth params for the plausibility metric
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from medusa import config
from medusa.contract.interface import Observations
from medusa.data import synthetic


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    observations: Observations
    fit: Observations
    holdout: Observations
    split: dict
    datasheet: str


def make_split(obs: Observations, fit_frac: float = 0.6) -> dict:
    if not 0.2 <= fit_frac <= 0.9:
        raise ValueError("fit_frac should be within [0.2, 0.9]")
    t0, t1 = float(obs.time_s[0]), float(obs.time_s[-1])
    t_split = t0 + fit_frac * (t1 - t0)
    n_fit = int(np.count_nonzero(obs.time_s < t_split))
    if n_fit < 4:
        raise ValueError("fit window has < 4 points; need a longer series or larger frac")
    if len(obs) - n_fit < 2:
        raise ValueError("holdout window has < 2 points")
    return {
        "fit_frac": fit_frac,
        "t_split_s": t_split,
        "n_fit": n_fit,
        "n_holdout": len(obs) - n_fit,
        "t_start_s": t0,
        "t_end_s": t1,
    }


def split_observations(obs: Observations, split: dict) -> tuple[Observations, Observations]:
    t_split = split["t_split_s"]
    fit = obs.slice_time(obs.time_s[0] - 1.0, t_split)
    holdout = obs.slice_time(t_split, obs.time_s[-1] + 1.0)
    return fit, holdout


def write(
    obs: Observations,
    datasheet: str,
    *,
    name: str,
    ground_truth: dict | None = None,
    fit_frac: float = 0.6,
    processed_dir: Path | None = None,
    datasheet_path: Path | None = None,
) -> Dataset:
    processed_dir = processed_dir or config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    datasheet_path = datasheet_path or config.DATASHEET_MD
    datasheet_path.parent.mkdir(parents=True, exist_ok=True)

    split = make_split(obs, fit_frac)
    split["dataset"] = name
    split["ground_truth"] = ground_truth or {}
    obs.to_parquet(processed_dir / "observations.parquet")
    (processed_dir / "split.json").write_text(json.dumps(split, indent=2))
    datasheet_path.write_text(datasheet)

    fit, holdout = split_observations(obs, split)
    return Dataset(name, obs, fit, holdout, split, datasheet)


def build_synthetic(preset_name: str, *, fit_frac: float = 0.6, **kw) -> Dataset:
    preset = synthetic.PRESETS[preset_name]
    obs = synthetic.generate(preset)
    td = preset.doubling_time_h
    ground_truth = {
        "doubling_time_h": td,
        "td_plausible_h": [round(td * 0.6, 4), round(td * 1.6, 4)],
        "carrying_capacity": preset.carrying_capacity,
        "lag_h": preset.lag_h,
    }
    return write(
        obs,
        synthetic.datasheet(preset),
        name=preset_name,
        ground_truth=ground_truth,
        fit_frac=fit_frac,
        **kw,
    )


def load(
    processed_dir: Path | None = None, datasheet_path: Path | None = None
) -> Dataset:
    """Load the previously written processed dataset."""
    processed_dir = processed_dir or config.PROCESSED_DIR
    datasheet_path = datasheet_path or config.DATASHEET_MD
    obs = Observations.from_parquet(processed_dir / "observations.parquet")
    split = json.loads((processed_dir / "split.json").read_text())
    fit, holdout = split_observations(obs, split)
    datasheet = datasheet_path.read_text() if datasheet_path.exists() else ""
    return Dataset(split.get("dataset", "unknown"), obs, fit, holdout, split, datasheet)
