import numpy as np
import pytest

from medusa.contract.interface import SPATIAL_TASK, STRUCTURED_TASK
from medusa.spatial.frames import CellFrame, SpatialFrames
from medusa.spatial.summarize import SPATIAL_CHANNELS, summary_series


def _demo_frames(n_frames=8, seed=0):
    rng = np.random.default_rng(seed)
    frames, times = [], []
    n = 20
    for i in range(n_frames):
        n = int(n * 1.15)
        frames.append(
            CellFrame(
                x=rng.uniform(0, 40, n), y=rng.uniform(0, 40, n),
                angle=rng.uniform(0, np.pi, n),
                length=rng.normal(3.5, 0.6, n).clip(1.5, 8),
                width=np.full(n, 1.0),
            )
        )
        times.append(i * 90.0)
    return SpatialFrames(np.array(times), frames)


def test_summary_series_has_all_channels():
    obs = summary_series(_demo_frames())
    for c in SPATIAL_CHANNELS:
        assert c in obs.channels()
    assert obs.population_count[-1] > obs.population_count[0]
    assert np.all(np.isfinite(obs.channel("nematic_order")))


def test_spatialframes_npz_roundtrip(tmp_path):
    sf = _demo_frames()
    sf.to_npz(tmp_path / "f.npz")
    back = SpatialFrames.from_npz(tmp_path / "f.npz")
    assert len(back) == len(sf)
    assert np.allclose(back.frames[3].x, sf.frames[3].x)


def test_reference_spatial_twin_runs_and_scores(tmp_path):
    pytest.importorskip("scipy")
    import dataclasses

    from medusa.config import DEFAULT_LOOP_CONFIG
    from medusa.data import build
    from medusa.harness.evaluate import evaluate_source
    from medusa.spatial.sim_reference import SOURCE

    # tiny synthetic spatial dataset so the test is fast
    sf = _demo_frames(n_frames=12)
    sf.to_npz(tmp_path / "frames.npz")
    obs = summary_series(sf)
    ds = build.write(
        obs, "synthetic spatial", name="spatial-test", task=SPATIAL_TASK,
        frames_npz=tmp_path / "frames.npz", processed_dir=tmp_path,
        datasheet_path=tmp_path / "d.md",
    )
    cfg = dataclasses.replace(DEFAULT_LOOP_CONFIG, spatial_runtime_budget_s=60.0)
    res = evaluate_source(SOURCE, ds, cfg)
    assert res.mode == "spatial"
    assert res.passed_checks and not res.crashed, res.error
    assert set(res.per_observable) <= set(SPATIAL_TASK.observables)


def test_multi_smape_weighted_over_channels():
    from medusa.contract.interface import Observations, Task
    from medusa.harness import metrics

    obs = Observations(
        time_s=np.arange(5.0), population_count=np.array([1.0, 2, 4, 8, 16]),
        extra={"mean_length_um": np.full(5, 3.0)},
    )
    task = Task("t", observables=("population_count", "mean_length_um"),
               weights={"mean_length_um": 0.0})
    pred = {"population_count": obs.population_count, "mean_length_um": np.full(5, 99.0)}
    combined, per = metrics.multi_smape(pred, obs, task.observables, task)
    assert per["population_count"] == 0.0
    assert per["mean_length_um"] > 1.0
    assert combined == 0.0  # mean_length weight is zero
