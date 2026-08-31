"""The meta-loop: reflective Pareto evolution of the inner loop's configuration.

Stops after one generation by default (`generations=1`) and writes `meta_checkpoint.md`
so we can confirm the hill-climb has real signal before spending a long run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np

from medusa import config
from medusa.agent.cache import SHARED
from medusa.bench import suite
from medusa.config import DEFAULT_LOOP_CONFIG, MetaConfig
from medusa.meta import report
from medusa.meta.archive import GenomeEntry, MetaArchive
from medusa.meta.digest import reflect
from medusa.meta.evaluator import evaluate_genome
from medusa.meta.genome import Genome, seed_genome
from medusa.meta import prompts as mp


@dataclasses.dataclass(slots=True)
class MetaResult:
    meta_dir: Path
    archive: MetaArchive
    checkpoint: bool


def _meta_client(dry_run: bool, meta_cfg: MetaConfig):
    if dry_run:
        return mp.CannedMetaClient()
    from medusa.agent.deepseek import DeepSeekClient

    return DeepSeekClient(DEFAULT_LOOP_CONFIG.replace(
        model=meta_cfg.model, temperature=meta_cfg.temperature, thinking=False,
    ))


def _score_line(s) -> str:
    return (f"mean {s.mean_best_smape:.3f} | worst {s.worst_best_smape:.3f} | "
            f"fam {s.mean_distinct_families:.1f} | ${s.mean_usd_cost:.3f} | "
            f"valid {s.mean_valid_rate:.0%} | feasible {s.feasible}")


def run_meta_loop(
    meta_cfg: MetaConfig | None = None,
    *,
    generations: int | None = None,
    dry_run: bool = False,
    runs_dir: Path | None = None,
) -> MetaResult:
    meta_cfg = meta_cfg or MetaConfig()
    gens = generations if generations is not None else meta_cfg.generations
    runs_dir = runs_dir or config.RUNS_DIR
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    meta_dir = runs_dir / f"meta-{stamp}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    train, val, test = suite.META_TRAIN, suite.META_VAL, suite.META_TEST
    (meta_dir / "meta_config.json").write_text(json.dumps(
        {**dataclasses.asdict(meta_cfg), "generations": gens, "dry_run": dry_run,
         "train": train, "val": val, "test": test}, indent=2, default=str))

    base = DEFAULT_LOOP_CONFIG
    archive = MetaArchive()

    def _eval(g: Genome, names, tag, **kw):
        d = meta_dir / tag
        d.mkdir(exist_ok=True)
        return evaluate_genome(g, names, meta_cfg, runs_dir=d, dry_run=dry_run, **kw)

    # --- seed + its noise band ------------------------------------------------
    seed = seed_genome()
    seed_train = _eval(seed, train, "seed_train")
    seed_val = _eval(seed, val, "seed_val")
    seed_val_rerun = _eval(seed, val, "seed_val_rerun", nocache=True)
    archive.add(GenomeEntry(seed, seed_train, seed_val))
    noise_band = abs(seed_val.mean_best_smape - seed_val_rerun.mean_best_smape)
    (meta_dir / "seed.json").write_text(json.dumps({
        "train": seed_train.to_dict(), "val": seed_val.to_dict(),
        "val_rerun": seed_val_rerun.to_dict(), "noise_band": noise_band,
    }, indent=2, default=str))

    client = _meta_client(dry_run, meta_cfg)
    best_val = seed_val.mean_best_smape
    stall = 0
    last_child: GenomeEntry | None = None

    for gen in range(1, gens + 1):
        parent_entry = archive.sample_parent(rng)
        digest = reflect(parent_entry.train.bench_dir)

        sys_prompt = mp.META_SYSTEM_PROMPT.format(
            component_menu=mp.component_menu(base, parent_entry.genome),
            knob_menu=mp.knob_menu(base, parent_entry.genome),
        )
        user_prompt = mp.META_ITERATION_TEMPLATE.format(
            generation=parent_entry.genome.generation,
            genome_state=mp.genome_state(base, parent_entry.genome),
            train_mean=parent_entry.train.mean_best_smape,
            train_worst=parent_entry.train.worst_best_smape,
            train_fam=parent_entry.train.mean_distinct_families,
            train_cost=parent_entry.train.mean_usd_cost,
            train_valid=parent_entry.train.mean_valid_rate,
            digest=digest,
            front=_front_text(archive, base),
        )
        completion = client.complete(sys_prompt, user_prompt)
        proposal = mp.parse_proposal(completion.text)

        gdir = meta_dir / f"gen_{gen:02d}"
        gdir.mkdir(exist_ok=True)
        (gdir / "digest.md").write_text(digest)
        (gdir / "agent_msg.md").write_text(completion.text or "(empty)")
        (gdir / "prompt.md").write_text(f"# SYSTEM\n{sys_prompt}\n\n# USER\n{user_prompt}")

        if proposal is None:
            (gdir / "outcome.json").write_text(json.dumps({"rejected": "no parseable proposal"}))
            continue
        child = parent_entry.genome.child(
            component=proposal.get("component"), knob=proposal.get("knob"),
            rationale=proposal.get("rationale", ""), generation=gen,
        )
        ok, msg = child.validates()
        if child.touched == "" or child.genome_id == parent_entry.genome.genome_id:
            ok, msg = False, "proposal changed nothing (unknown component/knob, or a no-op)"
        if not ok:
            (gdir / "outcome.json").write_text(json.dumps({"rejected": msg,
                                                           "proposal": proposal}))
            continue

        ct = _eval(child, train, f"gen_{gen:02d}_train")
        cv = _eval(child, val, f"gen_{gen:02d}_val")
        entry = GenomeEntry(child, ct, cv)
        archive.add(entry)
        last_child = entry
        (gdir / "genome.json").write_text(json.dumps(child.to_dict(), indent=2))
        (gdir / "outcome.json").write_text(json.dumps({
            "touched": child.touched, "diff": child.diff_summary(base),
            "train": ct.to_dict(), "val": cv.to_dict(),
        }, indent=2, default=str))

        if gens == 1:  # the review gate -- stop here, write the checkpoint
            report.write_checkpoint(
                meta_dir, base, seed, seed_val, seed_val_rerun, noise_band,
                digest, child, entry, msg="ok",
            )
            return MetaResult(meta_dir, archive, checkpoint=True)

        if cv.mean_best_smape < best_val - 1e-4:
            best_val, stall = cv.mean_best_smape, 0
        else:
            stall += 1
        spent = sum(e.train.mean_usd_cost * e.train.n_datasets +
                    e.val.mean_usd_cost * e.val.n_datasets for e in archive.entries)
        if stall >= meta_cfg.patience or spent > meta_cfg.usd_budget:
            break

    # --- finish: touch META_TEST once, write the report ---------------------
    best = archive.best_on_val()
    if best is not None:
        best.test = _eval(best.genome, test, "best_test")
    archive.write_portfolio(meta_dir / "portfolio")
    (meta_dir / "leaderboard.json").write_text(archive.to_json())
    report.write_report(meta_dir, base, archive, seed_val, seed_val_rerun, noise_band)
    return MetaResult(meta_dir, archive, checkpoint=False)


def _front_text(archive: MetaArchive, base) -> str:
    rows = archive.front()
    if not rows:
        return "(empty)"
    return "\n".join(
        f"- {e.genome.genome_id} (gen {e.genome.generation}, touched {e.genome.touched or 'seed'}): "
        f"val {_score_line(e.val)}"
        for e in sorted(rows, key=lambda e: e.val.mean_best_smape)
    )
