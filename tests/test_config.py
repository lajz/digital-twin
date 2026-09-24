import subprocess

from medusa import config
from medusa.config import DEFAULT_LOOP_CONFIG
from medusa.data import fetch
from medusa.harness import sandbox


def test_runtime_budget_for_default_falls_back_to_base_field():
    cfg = DEFAULT_LOOP_CONFIG
    assert cfg.runtime_budget_for("population") == cfg.twin_runtime_budget_s
    assert cfg.runtime_budget_for("unknown-task") == cfg.twin_runtime_budget_s


def test_runtime_budget_for_spatial_and_predator_prey():
    cfg = DEFAULT_LOOP_CONFIG
    assert cfg.runtime_budget_for("spatial") == cfg.spatial_runtime_budget_s
    assert cfg.runtime_budget_for("predator-prey") == cfg.twin_runtime_budget_predator_prey


def test_candidate_timeout_for_default_falls_back_to_base_field():
    cfg = DEFAULT_LOOP_CONFIG
    assert cfg.candidate_timeout_for("population") == cfg.candidate_timeout_s
    assert cfg.candidate_timeout_for("unknown-task") == cfg.candidate_timeout_s


def test_candidate_timeout_for_spatial_and_predator_prey():
    cfg = DEFAULT_LOOP_CONFIG
    assert cfg.candidate_timeout_for("spatial") == cfg.spatial_candidate_timeout_s
    assert cfg.candidate_timeout_for("predator-prey") == cfg.candidate_timeout_predator_prey_s


def test_run_candidate_uses_predator_prey_timeout(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / fetch.LYNX_HARE_RAW).write_text(
        "Year, Lynx, Hare\n1900, 4.0, 30.0\n1901, 6.1, 47.2\n1902, 9.8, 70.2\n"
        "1903, 35.2, 77.4\n1904, 59.4, 36.3\n1905, 41.7, 20.6\n1906, 19.0, 18.1\n"
    )
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    processed_dir = tmp_path / "processed"
    fetch.build_lynx_hare(
        fit_frac=0.6, processed_dir=processed_dir, datasheet_path=tmp_path / "d.md",
    )

    captured = {}

    def fake_run(args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(args, 0, stdout='{"crashed": true}\n', stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    sandbox.run_candidate("class Twin: pass", DEFAULT_LOOP_CONFIG, processed_dir=processed_dir)

    assert captured["timeout"] == DEFAULT_LOOP_CONFIG.candidate_timeout_predator_prey_s


def test_run_candidate_falls_back_to_base_timeout_when_dataset_load_fails(monkeypatch):
    def fake_load(*, processed_dir):
        raise FileNotFoundError("no dataset here")

    monkeypatch.setattr(sandbox.build, "load", fake_load)

    captured = {}

    def fake_run(args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(args, 0, stdout='{"crashed": true}\n', stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    sandbox.run_candidate("class Twin: pass", DEFAULT_LOOP_CONFIG, processed_dir=config.PROCESSED_DIR)

    assert captured["timeout"] == DEFAULT_LOOP_CONFIG.candidate_timeout_s
