"""L2 spatial layer: per-cell rod configurations, spatial summary statistics, rendering.

A spatial twin generates a *rollout* -- a list of cell configurations over time,
initialised from the first observed frame. It is scored on spatial summary-statistic
time series over the holdout window (never on pixels, never on held-out frames shown to
the agent), and its rollout is rendered side by side with reality.
"""

from medusa.spatial.frames import CellFrame, SpatialFrames
from medusa.spatial.summarize import SPATIAL_CHANNELS, summarize_frames, summary_series

__all__ = [
    "CellFrame",
    "SpatialFrames",
    "SPATIAL_CHANNELS",
    "summarize_frames",
    "summary_series",
]
