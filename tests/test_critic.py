"""The in-run critic: rendering, wiring through the loop, cost accounting, meta wiring."""

import dataclasses
import json

import pytest

from medusa import critic
from medusa.config import DEFAULT_LOOP_CONFIG, PRICE_PER_MTOK_IN, PRICE_PER_MTOK_OUT
from medusa.data import build
from medusa.harness.scorecard import loop_scorecard
from medusa.meta.genome import clamp_knob, seed_genome


@pytest.fixture
def dataset(tmp_path):
    return build.build_synthetic(
        "synthetic-ecoli-fast",
        processed_dir=tmp_path / "processed",
        datasheet_path=tmp_path / "processed" / "datasheet.md",
    )


def _run(tmp_path, **overrides):
    from medusa.agent.loop import run_loop

    ds = build.build_synthetic(
        "synthetic-ecoli-fast",
        processed_dir=tmp_path / "processed",
        datasheet_path=tmp_path / "processed" / "datasheet.md",
    )
    cfg = dataclasses.replace(DEFAULT_LOOP_CONFIG, **overrides)
    return run_loop(ds, cfg, dry_run=True, runs_dir=tmp_path / "runs",
                    processed_dir=tmp_path / "processed")


def _rows(run_dir):
    return [json.loads(l) for l in (run_dir / "trace.jsonl").read_text().splitlines() if l.strip()]


# --- unit -------------------------------------------------------------------


def test_render_critic_context_smoke():
    out = critic.render_critic_context(
        critic.CRITIC_CONTEXT_TEMPLATE,
        iteration=3,
        datasheet="a synthetic E. coli growth curve",
        obs_table="| time_h | population_count |\n|---|---|\n| 0 | 1000 |",
        constraint_names=["non_negative", "flow_balance[cash]"],
        family="logistic",
        status="ok",
        holdout_smape=0.1234,
        plausibility=0.87,
        per_observable={"population_count": 0.1234},
        archive_summary="| family | ... |",
        candidate_source="class Twin:\n    FAMILY = 'logistic'\n",
        recent_trace=[{"iter": 2, "family": "gompertz", "status": "ok", "holdout_smape": 0.2}],
        stance="coach",
    )
    assert "logistic" in out
    assert "population_count" in out
    assert "0.1234" in out
    assert "non_negative" in out
    assert "coach" in out
    assert "gompertz" in out  # recent trace rendered


def test_render_critic_context_degrades_without_trajectory():
    out = critic.render_critic_context(
        critic.CRITIC_CONTEXT_TEMPLATE,
        iteration=1, datasheet="", obs_table="", constraint_names=[],
        family=None, status="crashed", holdout_smape=None, plausibility=None,
        per_observable={}, archive_summary="", candidate_source=None,
        recent_trace=None, stance="skeptic",
    )
    assert "n/a" in out
    assert "no scored trajectory" in out
    assert "(no code produced)" in out


def test_format_note():
    assert critic.format_note("try a lag phase", "coach") == (
        "## Critic feedback (coach)\n\ntry a lag phase"
    )
    assert critic.format_note("", "coach") == ""
    assert critic.format_note("   \n ", "skeptic") == ""


def test_prompt_for_stance():
    assert critic.prompt_for_stance("skeptic") == critic.SKEPTIC_PROMPT
    assert critic.prompt_for_stance("coach") == critic.COACH_PROMPT
    assert critic.prompt_for_stance("nonsense") == critic.COACH_PROMPT
    assert set(critic.STANCES) == {"coach", "skeptic"}


def test_resolve_enabled_domain_default():
    assert critic.resolve_enabled(None, "real") is True
    assert critic.resolve_enabled(None, "business") is False
    assert critic.resolve_enabled(None, "synthetic") is False


def test_resolve_enabled_explicit_forces_either_way():
    assert critic.resolve_enabled(True, "business") is True
    assert critic.resolve_enabled(True, "synthetic") is True
    assert critic.resolve_enabled(False, "real") is False


# --- through the loop -----------------------------------------------------------


def test_canned_critic_smoke_through_loop(tmp_path):
    res = _run(tmp_path, max_iters=4, diversity_nudge_every=0,
               critic_enabled=True, critic_every=1)
    rd = res.run_dir

    assert (rd / "iter_01" / "critic.md").exists()
    assert (rd / "iter_03" / "critic.md").exists()

    p2 = (rd / "iter_02" / "prompt.md").read_text()
    assert "## Critic feedback (coach)" in p2
    assert "carrying-capacity term" in p2  # a substring of the first canned note

    scored = [r for r in _rows(rd) if r.get("status") != "no_code_block" and r["iter"] < 4]
    assert scored and all(r["critic_prompt_tokens"] > 0 and r["critic_response_tokens"] > 0
                          for r in scored)

    sc = json.loads((rd / "scorecard.json").read_text())
    rows = _rows(rd)
    assert sc["total_response_tokens"] == (
        sum(r.get("response_tokens", 0) for r in rows)
        + sum(r.get("critic_response_tokens", 0) for r in rows)
    )
    assert sc["critic_response_tokens"] == sum(r.get("critic_response_tokens", 0) for r in rows)


def test_critic_charges_usd_cost(tmp_path):
    off = _run(tmp_path / "a", max_iters=4, diversity_nudge_every=0)
    on = _run(tmp_path / "b", max_iters=4, diversity_nudge_every=0,
              critic_enabled=True, critic_every=1)

    cp = on.scorecard.critic_prompt_tokens
    crp = on.scorecard.critic_response_tokens
    assert cp > 0 and crp > 0
    delta = cp / 1e6 * PRICE_PER_MTOK_IN + crp / 1e6 * PRICE_PER_MTOK_OUT

    # enabling the critic costs at least its own tokens (the injected note also grows the
    # main-agent prompt slightly, so this is a lower bound), and never less.
    assert on.scorecard.usd_cost >= off.scorecard.usd_cost + delta

    # usd_cost is internally consistent with the (critic-inclusive) token totals it reports
    assert on.scorecard.usd_cost == pytest.approx(
        on.scorecard.total_prompt_tokens / 1e6 * PRICE_PER_MTOK_IN
        + on.scorecard.total_response_tokens / 1e6 * PRICE_PER_MTOK_OUT
    )


def test_critic_disabled_is_strict_noop(tmp_path):
    res = _run(tmp_path, max_iters=4, diversity_nudge_every=0)
    rd = res.run_dir

    assert not list(rd.glob("iter_*/critic.md"))
    for p in rd.glob("iter_*/prompt.md"):
        assert "Critic feedback" not in p.read_text()
    for r in _rows(rd):
        assert r.get("critic_prompt_tokens", 0) == 0
        assert r.get("critic_response_tokens", 0) == 0

    assert res.scorecard.critic_prompt_tokens == 0
    rows = _rows(rd)
    plain = (sum(r.get("prompt_tokens", 0) for r in rows) / 1e6 * PRICE_PER_MTOK_IN
             + sum(r.get("response_tokens", 0) for r in rows) / 1e6 * PRICE_PER_MTOK_OUT)
    assert res.scorecard.usd_cost == pytest.approx(plain)


def test_critic_every_gates_cadence(tmp_path):
    res = _run(tmp_path, max_iters=5, diversity_nudge_every=0,
               critic_enabled=True, critic_every=2)
    rd = res.run_dir

    assert not (rd / "iter_01" / "critic.md").exists()
    assert (rd / "iter_02" / "critic.md").exists()
    assert (rd / "iter_04" / "critic.md").exists()

    assert "Critic feedback" in (rd / "iter_03" / "prompt.md").read_text()
    assert "Critic feedback" not in (rd / "iter_02" / "prompt.md").read_text()


class _FakeDomain:
    def __init__(self, kind):
        self.kind = kind
        self.constraints = ()


def test_critic_defaults_on_for_real_domain(tmp_path, monkeypatch):
    from medusa.agent import loop as loop_mod

    monkeypatch.setattr(loop_mod, "_domain_for", lambda dataset: _FakeDomain("real"))
    res = _run(tmp_path, max_iters=2, diversity_nudge_every=0)  # critic_enabled left at default
    assert (res.run_dir / "iter_01" / "critic.md").exists()


def test_critic_defaults_off_for_business_and_synthetic_domains(tmp_path, monkeypatch):
    from medusa.agent import loop as loop_mod

    for kind in ("business", "synthetic"):
        monkeypatch.setattr(loop_mod, "_domain_for", lambda dataset, _k=kind: _FakeDomain(_k))
        res = _run(tmp_path / kind, max_iters=2, diversity_nudge_every=0)
        assert not list(res.run_dir.glob("iter_*/critic.md"))


def test_critic_explicit_enabled_forces_on_for_business_domain(tmp_path, monkeypatch):
    from medusa.agent import loop as loop_mod

    monkeypatch.setattr(loop_mod, "_domain_for", lambda dataset: _FakeDomain("business"))
    res = _run(tmp_path, max_iters=2, diversity_nudge_every=0, critic_enabled=True)
    assert (res.run_dir / "iter_01" / "critic.md").exists()


def test_critic_explicit_disabled_forces_off_for_real_domain(tmp_path, monkeypatch):
    from medusa.agent import loop as loop_mod

    monkeypatch.setattr(loop_mod, "_domain_for", lambda dataset: _FakeDomain("real"))
    res = _run(tmp_path, max_iters=2, diversity_nudge_every=0, critic_enabled=False)
    assert not list(res.run_dir.glob("iter_*/critic.md"))


def test_cli_critic_flags_force_override():
    from medusa.cli import _cfg_from_args, build_parser

    parser = build_parser()
    assert _cfg_from_args(parser.parse_args(["run"])).critic_enabled is None
    assert _cfg_from_args(parser.parse_args(["run", "--critic"])).critic_enabled is True
    assert _cfg_from_args(parser.parse_args(["run", "--no-critic"])).critic_enabled is False


def test_registered_domain_kinds_match_critic_defaults():
    """The domains that actually get the critic-on-by-default treatment: real data only.
    (Cheap metadata check -- doesn't build ipb-ecoli, which needs the raw CSV / network.)"""
    from medusa import domains

    assert domains.get("ipb-ecoli").kind == "real"
    assert domains.get("saas-seed").kind == "business"
    assert domains.get("synthetic-ecoli-fast").kind == "synthetic"


# --- pure scorecard -----------------------------------------------------------


def test_scorecard_sums_critic_tokens(tmp_path):
    run = tmp_path / "20260101-000000-synthetic-ecoli-fast"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"dataset": "synthetic-ecoli-fast"}))
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"iter": 1, "family": "logistic", "status": "ok", "is_valid": True,
         "holdout_smape": 0.2, "prompt_tokens": 1000, "response_tokens": 200,
         "critic_prompt_tokens": 500, "critic_response_tokens": 50, "wall_s": 0.1},
        {"iter": 2, "family": "gompertz", "status": "ok", "is_valid": True,
         "holdout_smape": 0.15, "prompt_tokens": 1000, "response_tokens": 200,
         "critic_prompt_tokens": 0, "critic_response_tokens": 0, "wall_s": 0.1},
    ]))
    m = loop_scorecard(run, price_in=1.0, price_out=1.0)
    assert m.total_prompt_tokens == 2500
    assert m.total_response_tokens == 450
    assert m.critic_prompt_tokens == 500
    assert m.critic_response_tokens == 50
    assert m.usd_cost == pytest.approx((2500 + 450) / 1e6)


# --- meta wiring ------------------------------------------------------------


def test_genome_can_toggle_and_evolve_critic():
    child = seed_genome().child(knob={"critic_enabled": 1, "critic_every": 2})
    ok, msg = child.validates()
    assert ok, msg
    cfg = child.apply(DEFAULT_LOOP_CONFIG)
    assert cfg.critic_enabled is True and cfg.critic_every == 2

    comp = seed_genome().child(component={"critic_prompt": "be a sharper coach"})
    assert comp.touched == "critic_prompt"
    assert comp.apply(DEFAULT_LOOP_CONFIG).critic_prompt == "be a sharper coach"

    # untouched -> knob absent from the genome -> the per-domain default applies (None),
    # not a hardcoded off
    assert seed_genome().apply(DEFAULT_LOOP_CONFIG).critic_enabled is None
    assert not seed_genome().child(component={"critic_prompt": "   "}).validates()[0]
    assert not seed_genome().child(component={"critic_prompt": "x" * 6001}).validates()[0]


def test_critic_every_lo_is_one():
    assert clamp_knob("critic_every", 0) == 1
    assert clamp_knob("critic_every", 99) == 4
