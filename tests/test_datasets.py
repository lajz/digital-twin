import pytest

from medusa.data import datasets


def test_registry_lists_synthetic_and_real():
    names = datasets.list_datasets()
    assert "synthetic-ecoli-fast" in names
    assert "ipb-ecoli" in names
    assert "lynx-hare" in names
    assert datasets.is_real("ipb-ecoli")
    assert datasets.is_real("lynx-hare")
    assert not datasets.is_real("synthetic-ecoli-fast")


def test_build_unknown_dataset_raises():
    with pytest.raises(ValueError):
        datasets.build_dataset("does-not-exist")


def test_build_synthetic_via_registry(tmp_path):
    ds = datasets.build_dataset(
        "synthetic-bsub-mid", processed_dir=tmp_path, datasheet_path=tmp_path / "d.md"
    )
    assert ds.name == "synthetic-bsub-mid"
    assert len(ds.fit) + len(ds.holdout) == len(ds.observations)
