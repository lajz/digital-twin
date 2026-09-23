import numpy as np
import pytest

from medusa import config
from medusa.data import build, fetch, synthetic
from medusa.data.build import make_split, split_observations


def _obs():
    return synthetic.generate(synthetic.PRESETS["synthetic-bsub-mid"])


def test_split_partitions_observations():
    obs = _obs()
    split = make_split(obs, fit_frac=0.6)
    fit, holdout = split_observations(obs, split)
    assert len(fit) + len(holdout) == len(obs)
    assert fit.time_s.max() < holdout.time_s.min()
    assert len(fit) == split["n_fit"]


def test_split_rejects_extreme_fractions():
    obs = _obs()
    with pytest.raises(ValueError):
        make_split(obs, fit_frac=0.95)


def test_build_synthetic_roundtrip(tmp_path):
    ds = build.build_synthetic(
        "synthetic-yeast-slow",
        processed_dir=tmp_path,
        datasheet_path=tmp_path / "datasheet.md",
    )
    reloaded = build.load(processed_dir=tmp_path, datasheet_path=tmp_path / "datasheet.md")
    assert reloaded.name == ds.name
    assert np.allclose(reloaded.observations.population_count, ds.observations.population_count)
    assert reloaded.split["ground_truth"]["doubling_time_h"] > 0


def test_synthetic_growth_is_monotone_ish_and_saturates():
    obs = _obs()
    assert obs.population_count[-1] > obs.population_count[0] * 100
    late = obs.population_count[-5:].mean()
    preset = synthetic.PRESETS["synthetic-bsub-mid"]
    assert late > 0.5 * preset.carrying_capacity


def test_build_lynx_hare_from_a_local_fixture_csv(tmp_path, monkeypatch):
    # a small fixture standing in for the real (network-fetched) CSV, same header
    # shape (comment lines + leading-space column names) as the real Stan file.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / fetch.LYNX_HARE_RAW).write_text(
        "# Data from http://www.math.tamu.edu/~phoward/m442/modbasics.pdf\n"
        "# Downloaded 15 October 2017\n"
        "Year, Lynx, Hare\n"
        "1900, 4.0, 30.0\n"
        "1901, 6.1, 47.2\n"
        "1902, 9.8, 70.2\n"
        "1903, 35.2, 77.4\n"
        "1904, 59.4, 36.3\n"
        "1905, 41.7, 20.6\n"
        "1906, 19.0, 18.1\n"
    )
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)

    ds = fetch.build_lynx_hare(
        fit_frac=0.6, processed_dir=tmp_path / "processed", datasheet_path=tmp_path / "d.md",
    )

    assert ds.name == "lynx-hare"
    assert ds.task.name == "predator-prey"
    assert len(ds.observations) == 7
    assert ds.split["n_fit"] == 4 and ds.split["n_holdout"] == 3
    assert ds.observations.time_s[0] == 0.0
    np.testing.assert_allclose(ds.observations.population_count, ds.observations.hare)
    np.testing.assert_allclose(
        ds.observations.hare, [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1]
    )
    np.testing.assert_allclose(
        ds.observations.lynx, [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0]
    )
