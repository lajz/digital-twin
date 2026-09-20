import json

import numpy as np
import pytest

from medusa import critic
from medusa.config import DEFAULT_LOOP_CONFIG
from medusa.meta.archive import GenomeScore, MetaArchive, GenomeEntry
from medusa.meta.genome import ENUM_SPECS, KNOB_SPECS, Genome, clamp_knob, seed_genome


def test_genome_apply_and_roundtrip():
    g = seed_genome().child(
        component={"diversity_nudge_text": "be bold"},
        knob={"temperature": 5.0},  # out of range -> clamps to 1.2
        rationale="x", generation=1,
    )
    cfg = g.apply(DEFAULT_LOOP_CONFIG)
    assert cfg.diversity_nudge_text == "be bold"
    assert cfg.temperature == 1.2
    back = Genome.from_dict(json.loads(json.dumps(g.to_dict())))
    assert back.apply(DEFAULT_LOOP_CONFIG).temperature == 1.2
    assert back.touched in ("diversity_nudge_text", "temperature")


def test_knob_clamping():
    assert clamp_knob("min_families", 99) == 5
    assert clamp_knob("min_families", 1) == 2
    assert clamp_knob("thinking", 1) is True
    assert isinstance(clamp_knob("temperature", 0.5), float)


def test_critic_stance_is_an_enum_knob_not_a_numeric_range():
    assert "critic_stance" not in KNOB_SPECS
    assert ENUM_SPECS["critic_stance"] == ("coach", "skeptic")
    assert clamp_knob("critic_stance", "skeptic") == "skeptic"
    assert clamp_knob("critic_stance", "cynic") == "coach"  # invalid -> falls back


def test_genome_critic_stance_roundtrip_and_syncs_prompt():
    g = seed_genome().child(
        knob={"critic_stance": "skeptic"}, rationale="ipb-ecoli A/B favors skeptic",
        generation=1,
    )
    assert g.touched == "critic_stance"
    ok, msg = g.validates()
    assert ok, msg

    cfg = g.apply(DEFAULT_LOOP_CONFIG)
    assert cfg.critic_stance == "skeptic"
    assert cfg.critic_prompt == critic.SKEPTIC_PROMPT  # prompt follows the stance switch

    back = Genome.from_dict(json.loads(json.dumps(g.to_dict())))
    assert back.knobs["critic_stance"] == "skeptic"
    back_cfg = back.apply(DEFAULT_LOOP_CONFIG)
    assert back_cfg.critic_stance == "skeptic"
    assert back_cfg.critic_prompt == critic.SKEPTIC_PROMPT
    assert back.genome_id == g.genome_id


def test_genome_stance_switch_defers_to_an_explicit_critic_prompt_edit():
    # a single child() call that sets both -- the explicit component wins, not the sync
    g = seed_genome().child(
        knob={"critic_stance": "skeptic"}, component={"critic_prompt": "custom text"},
    )
    assert g.apply(DEFAULT_LOOP_CONFIG).critic_prompt == "custom text"


def test_genome_rejects_bad_components():
    assert not Genome(components={"nope": "x"}).validates()[0]
    assert seed_genome().child(component={"nope": "x"}).touched == ""  # filtered, no-op
    bad_tmpl = seed_genome().child(component={"iteration_template": "just {datasheet}"})
    assert not bad_tmpl.validates()[0]  # dropped required slots
    bad_lib = seed_genome().child(
        component={"helper_library": "```python\nthis is not valid python\n```"}
    )
    assert not bad_lib.validates()[0]


def _score(mean, worst, cost, fam=3.0, valid=1.0):
    return GenomeScore(mean, worst, cost, fam, valid, 2, "x")


def test_pareto_front_and_constraints():
    arc = MetaArchive()
    arc.add(GenomeEntry(seed_genome(), _score(0.3, 0.5, 0.01), _score(0.30, 0.50, 0.01)))
    arc.add(GenomeEntry(seed_genome().child(knob={"max_iters": 10}),
                        _score(0.2, 0.4, 0.02), _score(0.20, 0.40, 0.02)))   # dominates none fully
    arc.add(GenomeEntry(seed_genome().child(knob={"max_iters": 12}),
                        _score(0.4, 0.6, 0.03), _score(0.40, 0.60, 0.03)))   # dominated
    arc.add(GenomeEntry(seed_genome().child(knob={"max_iters": 14}),
                        _score(0.1, 0.3, 0.05), _score(0.10, 0.30, 0.05, fam=1.0)))  # infeasible
    front = arc.front()
    assert all(e.val.feasible for e in front)
    ids = {e.genome.genome_id for e in front}
    # the 0.40/0.60/0.03 entry is dominated by 0.20/0.40/0.02
    assert len(front) == 2
    best = arc.best_on_val()
    assert best.val.mean_best_smape == 0.20  # lowest feasible


def test_digest_smoke(tmp_path):
    from medusa.meta.digest import reflect
    # a minimal fake bench dir: one run with a trace
    run = tmp_path / "20260101-000000-synthetic-ecoli-fast"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"dataset": "synthetic-ecoli-fast"}))
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"iter": 1, "family": "logistic", "status": "ok", "is_valid": True, "holdout_smape": 0.2},
        {"iter": 2, "family": "logistic", "status": "over time budget", "is_valid": False},
        {"iter": 3, "family": None, "status": "no_code_block", "is_valid": False},
    ]))
    out = reflect(tmp_path)
    assert "synthetic-ecoli-fast" in out
    assert "budget" in out.lower() and "code block" in out.lower()


def test_dry_run_meta_loop_one_generation(tmp_path):
    from medusa.meta.loop import run_meta_loop

    res = run_meta_loop(generations=1, dry_run=True, runs_dir=tmp_path)
    assert res.checkpoint
    assert (res.meta_dir / "meta_checkpoint.md").exists()
    assert (res.meta_dir / "seed.json").exists()
    assert len(res.archive.entries) == 2  # seed + child
