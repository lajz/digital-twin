"""Render rod-cell frames, and build side-by-side model-vs-reality comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from medusa.spatial.frames import CellFrame, SpatialFrames  # noqa: E402

_REAL_C = "#1b4965"
_TWIN_C = "#5fa8d3"


def draw_frame(ax, frame: CellFrame, *, color: str, bounds=None) -> None:
    n = len(frame)
    if n:
        ux, uy = np.cos(frame.angle), np.sin(frame.angle)
        half = 0.5 * np.maximum(frame.length - frame.width, 0.0)
        segs = np.stack(
            [
                np.column_stack([frame.x - ux * half, frame.y - uy * half]),
                np.column_stack([frame.x + ux * half, frame.y + uy * half]),
            ],
            axis=1,
        )
        lc = LineCollection(
            segs, colors=color,
            linewidths=np.clip(np.median(frame.width) * 2.2, 1.0, 4.0),
            capstyle="round", alpha=0.9,
        )
        ax.add_collection(lc)
    if bounds:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.95, f"n={n}", transform=ax.transAxes, va="top", fontsize=8,
            color=color, weight="bold")


def _common_bounds(*frames: CellFrame):
    xs = np.concatenate([f.x for f in frames if len(f)] or [np.array([0.0])])
    ys = np.concatenate([f.y for f in frames if len(f)] or [np.array([0.0])])
    pad = 4.0
    return (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)


def comparison_figure(
    real: SpatialFrames,
    twin: SpatialFrames,
    *,
    split_time_s: float,
    out_path: str | Path,
    title: str = "",
    n_cols: int = 5,
) -> Path:
    """Grid: columns = time points across the movie, rows = [reality, twin]."""
    n = min(len(real), len(twin))
    idx = np.unique(np.linspace(0, n - 1, n_cols).astype(int))
    fig, axes = plt.subplots(2, len(idx), figsize=(2.5 * len(idx), 5.4), squeeze=False)

    for col, i in enumerate(idx):
        b = _common_bounds(real.frames[i], twin.frames[i])
        held = real.time_s[i] >= split_time_s
        draw_frame(axes[0][col], real.frames[i], color=_REAL_C, bounds=b)
        draw_frame(axes[1][col], twin.frames[i], color=_TWIN_C, bounds=b)
        tag = "holdout" if held else "fit"
        axes[0][col].set_title(f"t = {real.time_h[i]:.2f} h\n({tag})", fontsize=8,
                               color="#c1121f" if held else "#333")
    axes[0][0].set_ylabel("reality", fontsize=11, color=_REAL_C, weight="bold")
    axes[1][0].set_ylabel("twin", fontsize=11, color=_TWIN_C, weight="bold")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
