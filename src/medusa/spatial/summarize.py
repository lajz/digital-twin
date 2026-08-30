"""Reduce rod-cell frames to spatial summary-statistic time series.

These summaries -- not pixels -- are what a spatial twin is scored on. All are
translation/rotation-invariant and robust to a few missing cells.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from medusa.contract.interface import Observations
from medusa.spatial.frames import CellFrame, SpatialFrames

SPATIAL_CHANNELS = (
    "population_count",   # number of cells
    "total_length_um",    # summed rod length (biomass proxy)
    "colony_radius_um",   # radius of gyration of cell centres
    "colony_aspect",      # sqrt(major/minor) of the centre covariance (>= 1)
    "nematic_order",      # |<exp(2 i theta)>| of rod orientations, in [0, 1]
    "mean_nn_dist_um",    # mean nearest-neighbour distance between centres
)


def summarize_frame(f: CellFrame) -> dict[str, float]:
    n = len(f)
    if n == 0:
        return dict.fromkeys(SPATIAL_CHANNELS, 0.0)

    out = {"population_count": float(n), "total_length_um": float(np.sum(f.length))}

    cx, cy = f.x - f.x.mean(), f.y - f.y.mean()
    out["colony_radius_um"] = float(np.sqrt(np.mean(cx**2 + cy**2)))

    if n >= 2:
        cov = np.cov(np.vstack([cx, cy]))
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 1e-9, None)
        out["colony_aspect"] = float(np.sqrt(ev[-1] / ev[0]))
    else:
        out["colony_aspect"] = 1.0

    out["nematic_order"] = float(np.abs(np.mean(np.exp(2j * f.angle))))

    if n >= 2:
        pts = np.column_stack([f.x, f.y])
        d, _ = cKDTree(pts).query(pts, k=2)
        out["mean_nn_dist_um"] = float(np.mean(d[:, 1]))
    else:
        out["mean_nn_dist_um"] = 0.0

    return out


def summarize_frames(frames: SpatialFrames) -> dict[str, np.ndarray]:
    rows = [summarize_frame(f) for f in frames.frames]
    return {c: np.array([r[c] for r in rows], dtype=float) for c in SPATIAL_CHANNELS}


def summary_series(frames: SpatialFrames) -> Observations:
    """Spatial summaries as an Observations time series."""
    s = summarize_frames(frames)
    extra = {c: s[c] for c in SPATIAL_CHANNELS if c != "population_count"}
    return Observations(
        time_s=frames.time_s, population_count=s["population_count"], extra=extra
    )
