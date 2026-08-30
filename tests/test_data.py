import numpy as np
import pytest

from medusa.data import build, synthetic
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
