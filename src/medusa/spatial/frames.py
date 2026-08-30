"""Per-frame rod-cell configurations and their on-disk (npz) form.

A "cell" is a rod (a capsule): centre (x, y) in microns, orientation angle in radians,
length in microns (pole-to-pole of the cylinder axis), width in microns.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

CELL_KEYS = ("x", "y", "angle", "length", "width")


@dataclasses.dataclass(slots=True)
class CellFrame:
    """One time point: arrays of length n_cells."""

    x: np.ndarray
    y: np.ndarray
    angle: np.ndarray
    length: np.ndarray
    width: np.ndarray

    def __post_init__(self) -> None:
        arrs = [np.asarray(getattr(self, k), dtype=float).ravel() for k in CELL_KEYS]
        n = arrs[0].size
        if any(a.size != n for a in arrs):
            raise ValueError("CellFrame arrays must all have the same length")
        for k, a in zip(CELL_KEYS, arrs):
            setattr(self, k, a)

    def __len__(self) -> int:
        return int(self.x.size)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in CELL_KEYS}

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> "CellFrame":
        w = d.get("width")
        if w is None:
            w = np.full_like(np.asarray(d["x"], dtype=float), 1.0)
        return cls(x=d["x"], y=d["y"], angle=d["angle"], length=d["length"], width=w)

    def bounds(self, pad: float = 2.0) -> tuple[float, float, float, float]:
        if len(self) == 0:
            return (-10.0, 10.0, -10.0, 10.0)
        r = 0.5 * self.length.max() + pad
        return (
            float(self.x.min() - r), float(self.x.max() + r),
            float(self.y.min() - r), float(self.y.max() + r),
        )


@dataclasses.dataclass(slots=True)
class SpatialFrames:
    """A time-lapse of CellFrames. `time_s` is seconds since the first frame."""

    time_s: np.ndarray
    frames: list[CellFrame]
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float).ravel()
        if self.time_s.size != len(self.frames):
            raise ValueError("time_s length must match number of frames")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def time_h(self) -> np.ndarray:
        return self.time_s / 3600.0

    def counts(self) -> np.ndarray:
        return np.array([len(f) for f in self.frames], dtype=float)

    def index_at_or_before(self, t_s: float) -> int:
        return int(np.searchsorted(self.time_s, t_s, side="right") - 1)

    def slice_frames(self, i0: int, i1: int) -> "SpatialFrames":
        return SpatialFrames(self.time_s[i0:i1], self.frames[i0:i1], dict(self.meta))

    # --- npz io --------------------------------------------------------------

    def to_npz(self, path: str | Path) -> None:
        flat: dict[str, np.ndarray] = {"time_s": self.time_s}
        flat["n_per_frame"] = np.array([len(f) for f in self.frames], dtype=int)
        for k in CELL_KEYS:
            flat[k] = np.concatenate(
                [getattr(f, k) for f in self.frames] or [np.array([])]
            )
        np.savez_compressed(path, meta_keys=np.array(list(self.meta)),
                            meta_vals=np.array([str(v) for v in self.meta.values()]),
                            **flat)

    @classmethod
    def from_npz(cls, path: str | Path) -> "SpatialFrames":
        z = np.load(path, allow_pickle=False)
        n_per = z["n_per_frame"]
        offs = np.concatenate([[0], np.cumsum(n_per)])
        frames = []
        for i in range(len(n_per)):
            s, e = offs[i], offs[i + 1]
            frames.append(CellFrame(**{k: z[k][s:e] for k in CELL_KEYS}))
        meta = dict(zip(z["meta_keys"].tolist(), z["meta_vals"].tolist())) \
            if "meta_keys" in z else {}
        return cls(z["time_s"], frames, meta)


def rollout_to_frames(rollout: list, time_s: np.ndarray) -> SpatialFrames:
    """Normalise whatever a twin's `simulate` returned into SpatialFrames."""
    frames = []
    for item in rollout:
        if isinstance(item, CellFrame):
            frames.append(item)
        elif isinstance(item, dict):
            frames.append(CellFrame.from_dict(item))
        else:
            raise TypeError(f"rollout frame must be a dict or CellFrame, got {type(item)}")
    return SpatialFrames(np.asarray(time_s, dtype=float)[: len(frames)], frames)
