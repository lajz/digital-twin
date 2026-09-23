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

    extra = {} if areas is None else {"total_area_um2": np.asarray(areas, float)}
    return Observations(time_s=times, population_count=counts, extra=extra)


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
    obs = Observations(time_s=time_s, population_count=counts)

    ground_truth = {
        "doubling_time_h": None,
        "td_plausible_h": [0.33, 1.5],
        "note": "literature E. coli agar-pad doubling time; not measured for this well",
    }
    return build.write(
        obs, IPB_ECOLI_DATASHEET, name="ipb-ecoli", ground_truth=ground_truth,
        fit_frac=fit_frac, **kw,
    )


# --- L1 structured: size-distribution summaries for the same E. coli movie ---------

IPB_ECOLI_STRUCTURED_DATASHEET = """\
# Datasheet: ipb-ecoli-structured (L1)

Same movie as `ipb-ecoli`, reduced from the per-object table `FilterObjects.csv` to a
size-structured signal: per frame, cell count, summed rod length (biomass proxy), mean
single-cell length, and the length coefficient of variation.

- **Organism:** E. coli K-12 monolayer microcolony, agar pad. 100 frames at 90 s.
- **Pixel size:** self-calibrated from median cell width == ~1.0 um (~{px:.3f} um/px).
- **Scored channels:** total_length_um, population_count, mean_length_um, length_cv.
- Mean cell length runs ~3.5 -> 4.3 -> 4.0 um (a non-monotone bump a count-only model
  cannot explain); length CV stays ~0.28-0.31.
- Plausible doubling time (E. coli, agar pad): [0.33, 1.5] h; mean length [1, 12] um.
"""


def build_ipb_ecoli_structured(
    *, fit_frac: float = 0.6, processed_dir=None, datasheet_path=None
) -> build.Dataset:
    from medusa.contract.interface import STRUCTURED_TASK

    csv_path = config.RAW_DIR / IPB_ECOLI_OBJECTS_RAW
    if not csv_path.exists():
        csv_path = fetch_ipb_ecoli_objects()
    df = pd.read_csv(csv_path)
    px_um = 1.0 / float(df["AreaShape_MinorAxisLength"].median())
    df["len_um"] = df["AreaShape_MajorAxisLength"] * px_um
    g = df.groupby("ImageNumber")
    times = (np.array(sorted(df["ImageNumber"].unique()), float) - 1.0) * IPB_ECOLI_FRAME_INTERVAL_S
    obs = Observations(
        time_s=times,
        population_count=g.size().to_numpy(float),
        extra={
            "total_length_um": g["len_um"].sum().to_numpy(float),
            "mean_length_um": g["len_um"].mean().to_numpy(float),
            "length_cv": (g["len_um"].std() / g["len_um"].mean()).to_numpy(float),
        },
    )
    ground_truth = {"td_plausible_h": [0.33, 1.5], "px_um": px_um}
    return build.write(
        obs, IPB_ECOLI_STRUCTURED_DATASHEET.format(px=px_um), name="ipb-ecoli-structured",
        ground_truth=ground_truth, fit_frac=fit_frac, task=STRUCTURED_TASK,
        processed_dir=processed_dir, datasheet_path=datasheet_path,
    )


# --- L2 spatial: per-cell rod configurations for the same E. coli movie -----------

IPB_ECOLI_OBJECTS_CSV = (
    f"{_IPB_BASE}/CellProfiler-Omnipose/E.coli-larger-dataset/FilterObjects.csv"
)
IPB_ECOLI_OBJECTS_RAW = "ipb_ecoli_cpomnipose_objects.csv"

IPB_ECOLI_SPATIAL_DATASHEET = """\
# Datasheet: ipb-ecoli-spatial (L2)

- **Source:** same movie as `ipb-ecoli` (Ahmadi et al. 2024,
  github.com/ingallslab/ImageProcessing-Benchmarking, CC-BY 4.0), but the per-object
  table `FilterObjects.csv` -- one row per segmented cell per frame.
- **Organism:** E. coli K-12 monolayer microcolony on an agar pad.
- **Frames:** 100 at 90 s. **Per-cell fields used:** `Location_Center_X/Y`,
  `AreaShape_Orientation`, `AreaShape_MajorAxisLength` (length),
  `AreaShape_MinorAxisLength` (width).
- **Pixel size:** self-calibrated -- E. coli width is tightly regulated at ~1.0 um, so
  1 px = 1.0 / median(MinorAxisLength_px) um (~0.12 um/px here).
- **What the twin sees:** the fit-window frames for calibration, and the *first* frame
  as the simulation's initial condition. It is scored on spatial summary-statistic time
  series over the holdout window only: cell count, total rod length, radius of gyration,
  colony aspect ratio, nematic order, mean nearest-neighbour distance.
- **Caveat:** automated segmentation, not manual ground truth; positions/orientations
  carry a few-percent error.
- Plausible doubling time (E. coli, agar pad): [0.33, 1.5] h.
"""


def fetch_ipb_ecoli_objects(dest_dir: Path | None = None) -> Path:
    import urllib.request

    dest_dir = dest_dir or config.RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / IPB_ECOLI_OBJECTS_RAW
    if not dest.exists():
        urllib.request.urlretrieve(IPB_ECOLI_OBJECTS_CSV, dest)  # noqa: S310
    return dest


def build_ipb_ecoli_spatial(
    *, fit_frac: float = 0.6, processed_dir=None, datasheet_path=None
) -> build.Dataset:
    from medusa.contract.interface import SPATIAL_TASK
    from medusa.spatial.frames import CellFrame, SpatialFrames
    from medusa.spatial.summarize import summary_series

    processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = config.RAW_DIR / IPB_ECOLI_OBJECTS_RAW
    if not csv_path.exists():
        csv_path = fetch_ipb_ecoli_objects()
    df = pd.read_csv(csv_path)

    px_um = 1.0 / float(df["AreaShape_MinorAxisLength"].median())
    df = df.sort_values(["ImageNumber", "ObjectNumber"])
    frames, times = [], []
    for img, g in df.groupby("ImageNumber"):
        frames.append(
            CellFrame(
                x=g["Location_Center_X"].to_numpy(float) * px_um,
                y=g["Location_Center_Y"].to_numpy(float) * px_um,
                angle=np.deg2rad(g["AreaShape_Orientation"].to_numpy(float)),
                length=g["AreaShape_MajorAxisLength"].to_numpy(float) * px_um,
                width=g["AreaShape_MinorAxisLength"].to_numpy(float) * px_um,
            )
        )
        times.append((int(img) - 1) * IPB_ECOLI_FRAME_INTERVAL_S)
    sf = SpatialFrames(np.array(times, float), frames,
                       meta={"px_um": f"{px_um:.5f}", "organism": "E. coli"})
    sf.to_npz(processed_dir / "frames.npz")

    obs = summary_series(sf)
    ground_truth = {"td_plausible_h": [0.33, 1.5], "px_um": px_um}
    return build.write(
        obs, IPB_ECOLI_SPATIAL_DATASHEET, name="ipb-ecoli-spatial",
        ground_truth=ground_truth, fit_frac=fit_frac, task=SPATIAL_TASK,
        frames_npz=processed_dir / "frames.npz",
        processed_dir=processed_dir, datasheet_path=datasheet_path,
    )


# --- a concrete real dataset: Hudson's Bay Co. lynx-hare pelt counts (predator-prey) --

LYNX_HARE_CSV = (
    "https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/"
    "lotka-volterra/hudson-bay-lynx-hare.csv"
)
LYNX_HARE_RAW = "hudson_bay_lynx_hare.csv"
YEAR_S = 365.25 * 86400.0  # Julian year, in seconds

LYNX_HARE_DATASHEET = """\
# Datasheet: lynx-hare

- **Source:** Hudson's Bay Company snowshoe hare / Canada lynx pelt-trading counts,
  1900-1920 -- the classic series that motivated the Lotka-Volterra predator-prey
  equations. This 21-point series is the one the Stan development team curates for
  their own Lotka-Volterra case study (Bob Carpenter, "Predator-Prey Population
  Dynamics: the Lotka-Volterra Model in Stan", mc-stan.org/learn-stan/case-studies/
  lotka-volterra-predator-prey.html; data file: github.com/stan-dev/example-models,
  knitr/lotka-volterra/hudson-bay-lynx-hare.csv). That file's own header attributes it
  to http://www.math.tamu.edu/~phoward/m442/modbasics.pdf (P. Howard, "Modeling
  Basics", Texas A&M lecture notes), downloaded 15 Oct 2017 -- this dataset circulates
  in the ecology/applied-math literature without one single canonical primary
  citation; several textbooks (including Howard's) reproduce the same 21 numbers.
- **Units:** counts are in **thousands of pelts traded per year** -- a proxy for
  population, not a census. Trapping effort, fur prices, and reporting practices all
  confound the pelt count vs. the true population; treat this as a noisy indirect
  observable, the same caveat `ipb-ecoli`'s datasheet makes about its automated cell
  count not being a manual ground truth.
- **Years:** 1900-1920 inclusive, one point per year (21 points total). Small by the
  standards of this project's other datasets (~12 fit / ~9 holdout at fit_frac=0.6) --
  don't be surprised by noisy holdout scores. A longer 1845-1935 series exists
  (`ecostudy` R package, `lynxhare` dataset, Stevens 2009, "A Primer of Ecology with
  R") but was not used here; documenting the choice per the task brief.
- **Scored channels:** `hare` (prey) and `lynx` (predator), both thousands of pelts.
  No exogenous levers -- unlike `saas-seed`, nothing is handed to the twin beyond the
  two time series; the whole point is two mutually-driven populations with no
  external forcing.
- **Shape:** oscillatory, non-monotonic -- hare and lynx both rise, peak, crash, and
  recover on a roughly 9-10 year cycle, with the lynx peak lagging the hare peak by a
  year or two (predators multiply only once prey is abundant, then overshoot and crash
  it). No lag phase, no saturation, no accounting identity -- structurally unlike every
  other domain in this project.
- **Plausibility:** intentionally left empty (`PREDATOR_PREY_TASK.plausibility == {}`).
  The harness's plausibility scorer (`harness/evaluate.py::_finish_metrics`) only knows
  how to check a fixed set of hardcoded derived quantities (an implied exponential
  doubling time, mean cell length, ARPA, monthly churn) -- none of which has a
  Lotka-Volterra analog, and an "implied doubling time" isn't even a coherent concept
  for a series that goes up AND down. Wiring in a genuine LV-parameter plausibility
  check would mean extending the harness itself -- out of scope for this
  generalization probe. For reference only (NOT enforced as a score gate): fitting
  this series with Stan's own Lotka-Volterra model
  (`dH/dt = (alpha - beta*L)*H`, `dL/dt = (-gamma + delta*H)*L`), the posterior means
  are alpha~=0.55/yr, beta~=0.028/(yr*'000 pelts), gamma~=0.80/yr,
  delta~=0.024/(yr*'000 pelts) (Carpenter's case study; closely matches Howard's
  independent point estimates alpha*=0.55, beta*=0.028, gamma*=0.84, delta*=0.026).
  These numbers inform the PARAMS prior ranges suggested in the system prompt, but
  are not checked by the harness.
"""


def fetch_lynx_hare(dest_dir: Path | None = None) -> Path:
    """Download the Hudson's Bay Company lynx/hare pelt-count series."""
    import urllib.request

    dest_dir = dest_dir or config.RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / LYNX_HARE_RAW
    if not dest.exists():
        urllib.request.urlretrieve(LYNX_HARE_CSV, dest)  # noqa: S310 - fixed https URL
    return dest


def build_lynx_hare(*, fit_frac: float = 0.6, **kw) -> build.Dataset:
    from medusa.domains.predator_prey import PREDATOR_PREY_TASK

    csv_path = config.RAW_DIR / LYNX_HARE_RAW
    if not csv_path.exists():
        csv_path = fetch_lynx_hare()

    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values("Year")
    years = df["Year"].to_numpy(dtype=float)
    hare = df["Hare"].to_numpy(dtype=float)
    lynx = df["Lynx"].to_numpy(dtype=float)
    time_s = (years - years[0]) * YEAR_S

    # population_count is a mandatory field on Observations regardless of what's
    # scored (contract/interface.py); duplicate hare into it and into extra, exactly
    # how saas.py's generate() duplicates `customers` into both slots.
    obs = Observations(
        time_s=time_s, population_count=hare, extra={"hare": hare, "lynx": lynx},
    )
    ground_truth = {
        "lv_params_ref_not_enforced": {
            "alpha_per_yr": 0.55, "beta_per_yr_kpelt": 0.028,
            "gamma_per_yr": 0.80, "delta_per_yr_kpelt": 0.024,
        },
        "note": "Stan case-study posterior means (Carpenter); reference only, see datasheet",
    }
    return build.write(
        obs, LYNX_HARE_DATASHEET, name="lynx-hare", ground_truth=ground_truth,
        fit_frac=fit_frac, task=PREDATOR_PREY_TASK, **kw,
    )


def print_instructions() -> None:
    print(INSTRUCTIONS)
    raw = config.RAW_DIR
    print(f"\nDrop raw files into: {raw}")
    if raw.exists():
        found = sorted(p.name for p in raw.iterdir() if p.is_file())
        print("Currently present:", found or "(none)")
