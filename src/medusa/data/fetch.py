"""Acquire a real public microscopy time-lapse and reduce it to the canonical signal.

Real datasets need a browser click-through or large downloads, so this module does NOT
auto-download. It (a) prints where to get each shortlisted dataset, and (b) reduces a
raw file you drop into data/raw/ into data/processed/observations.parquet.

Shortlist (see memory / plan):
  1. DeLTA 2.0 agar-pad microcolony  -- Zenodo, from doi:10.1101/2021.08.10.455795
  2. Cell Tracking Challenge bacterial 2D sets -- celltrackingchallenge.net (TRA GT)
  3. Tanouchi et al. 2015 / Scientific Data 2017 -- PMC5369309 (mother machine; fallback)

The generic reducer below expects a CSV with one row per tracked-cell-per-frame OR one
row per frame with a count column. Point it at the right columns with --time-col etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from medusa import config
from medusa.contract.interface import Observations
from medusa.data import build

INSTRUCTIONS = __doc__


def reduce_csv(
    csv_path: str | Path,
    *,
    time_col: str,
    time_unit: str = "s",
    count_col: str | None = None,
    frame_col: str | None = None,
    area_col: str | None = None,
) -> Observations:
    """Reduce a raw tracking CSV to (time_s, population_count, [total_area_um2]).

    - If `count_col` is given, each row is a frame with that count.
    - Else rows are (cell, frame) records: population = rows per `frame_col` (or per time).
    """
    df = pd.read_csv(csv_path)
    scale = {"s": 1.0, "min": 60.0, "h": 3600.0, "frame": None}.get(time_unit)

    if count_col is not None:
        g = df[[time_col, count_col]].dropna().groupby(time_col, as_index=False).max()
        times = g[time_col].to_numpy(dtype=float)
        counts = g[count_col].to_numpy(dtype=float)
        areas = None
        if area_col is not None:
            a = df[[time_col, area_col]].dropna().groupby(time_col, as_index=False).sum()
            areas = a[area_col].to_numpy(dtype=float)
    else:
        key = frame_col or time_col
        grp = df.groupby(key)
        counts = grp.size().to_numpy(dtype=float)
        times = np.asarray(sorted(df[key].unique()), dtype=float)
        if frame_col is not None and time_col in df.columns:
            times = grp[time_col].first().to_numpy(dtype=float)
        areas = (
            grp[area_col].sum().to_numpy(dtype=float) if area_col in df.columns else None
        )

    if scale is None:  # frame index -> assume uniform; caller should pass real times
        times = times - times.min()
    else:
        times = (times - times.min()) * scale

    return Observations(time_s=times, population_count=counts, total_area_um2=areas)


def build_from_raw(
    csv_path: str | Path,
    *,
    name: str,
    datasheet: str,
    ground_truth: dict | None = None,
    fit_frac: float = 0.6,
    **reduce_kw,
) -> build.Dataset:
    obs = reduce_csv(csv_path, **reduce_kw)
    return build.write(
        obs, datasheet, name=name, ground_truth=ground_truth, fit_frac=fit_frac
    )


# --- a concrete real dataset: Ingalls lab monolayer benchmark, E. coli ----------

_IPB_BASE = "https://raw.githubusercontent.com/ingallslab/ImageProcessing-Benchmarking/main"
IPB_ECOLI_IMAGE_CSV = (
    f"{_IPB_BASE}/CellProfiler-Omnipose/E.coli-larger-dataset/Image.csv"
)
IPB_ECOLI_RAW = "ipb_ecoli_cpomnipose_image.csv"
IPB_ECOLI_FRAME_INTERVAL_S = 90.0  # "large population dataset", 90 s intervals

IPB_ECOLI_DATASHEET = """\
# Datasheet: ipb-ecoli

- **Source:** Ahmadi et al., "A benchmarked comparison of software packages for
  time-lapse image processing of monolayer bacterial population dynamics",
  Microbiology Spectrum 2024 (doi:10.1128/spectrum.00032-24).
  Repo: github.com/ingallslab/ImageProcessing-Benchmarking (CC-BY 4.0).
- **This signal:** CellProfiler + Omnipose pipeline, `E.coli-larger-dataset`,
  `Image.csv` column `Count_FilterObjects` (segmented+filtered cell count) per frame.
- **Organism:** Escherichia coli K-12, growing as a monolayer microcolony on an agar pad.
- **Frames:** 100, at {interval:.0f} s intervals (~150 min total).
- **Reduction:** `population_count` = `Count_FilterObjects` per `ImageNumber`;
  `time_s` = (ImageNumber - 1) x {interval:.0f}. No lag or saturation phase is captured
  (the crop starts from an established ~110-cell field), so this is a near-pure
  exponential-growth window: ~110 -> ~1160 cells, ~3.4 doublings.
- **Ground-truth doubling time:** not provided per-experiment; literature range for
  E. coli on agar pads is ~25-60 min. Plausible range for the metric: [0.33, 1.5] h.
- **Caveat:** this is one automated pipeline's count, not a manual ground truth. The
  benchmark's other pipelines (DeLTA, FAST, SuperSegger) agree to within a few percent
  on this dataset.
""".format(interval=IPB_ECOLI_FRAME_INTERVAL_S)


def fetch_ipb_ecoli(dest_dir: Path | None = None) -> Path:
    """Download the CellProfiler-Omnipose Image.csv for the large E. coli dataset."""
    import urllib.request

    dest_dir = dest_dir or config.RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / IPB_ECOLI_RAW
    if not dest.exists():
        urllib.request.urlretrieve(IPB_ECOLI_IMAGE_CSV, dest)  # noqa: S310 - fixed https URL
    return dest


def build_ipb_ecoli(*, fit_frac: float = 0.6, **kw) -> build.Dataset:
    csv_path = config.RAW_DIR / IPB_ECOLI_RAW
    if not csv_path.exists():
        csv_path = fetch_ipb_ecoli()

    df = pd.read_csv(csv_path).sort_values("ImageNumber")
    counts = df["Count_FilterObjects"].to_numpy(dtype=float)
    time_s = (df["ImageNumber"].to_numpy(dtype=float) - 1.0) * IPB_ECOLI_FRAME_INTERVAL_S
    obs = Observations(time_s=time_s, population_count=counts, total_area_um2=None)

    ground_truth = {
        "doubling_time_h": None,
        "td_plausible_h": [0.33, 1.5],
        "note": "literature E. coli agar-pad doubling time; not measured for this well",
    }
    return build.write(
        obs, IPB_ECOLI_DATASHEET, name="ipb-ecoli", ground_truth=ground_truth,
        fit_frac=fit_frac, **kw,
    )


def print_instructions() -> None:
    print(INSTRUCTIONS)
    raw = config.RAW_DIR
    print(f"\nDrop raw files into: {raw}")
    if raw.exists():
        found = sorted(p.name for p in raw.iterdir() if p.is_file())
        print("Currently present:", found or "(none)")
