"""Evaluate a genome: apply it, run the inner loop across a suite, reduce to a score."""

from __future__ import annotations

from pathlib import Path

from medusa.agent.cache import SHARED
from medusa.bench.runner import run_bench
from medusa.config import DEFAULT_LOOP_CONFIG, MetaConfig, quick_config
from medusa.meta.archive import GenomeScore
from medusa.meta.genome import Genome


def evaluate_genome(
    genome: Genome,
    suite_names: tuple[str, ...],
    meta_cfg: MetaConfig,
    *,
    runs_dir: Path,
    dry_run: bool = False,
    nocache: bool = False,
) -> GenomeScore:
    cfg = genome.apply(quick_config(DEFAULT_LOOP_CONFIG, max_iters=meta_cfg.quick_max_iters))

    prev = SHARED.enabled
    if nocache:
        SHARED.enabled = False
    try:
        br = run_bench(cfg, dry_run=dry_run, datasets=list(suite_names), runs_dir=runs_dir)
    finally:
        SHARED.enabled = prev

    return GenomeScore.from_bench(br)
