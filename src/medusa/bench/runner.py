"""Run the inner loop across the benchmark suite and aggregate LoopMetrics."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import statistics
import tempfile
from pathlib import Path

from medusa import config
from medusa.agent.loop import run_loop
from medusa.bench import suite
from medusa.config import LoopConfig
from medusa.harness.scorecard import LoopMetrics


@dataclasses.dataclass(slots=True)
class BenchResult:
    bench_dir: Path
    per_dataset: dict[str, LoopMetrics]
    aggregate: dict


def _aggregate(per: dict[str, LoopMetrics]) -> dict:
    scored = [m for m in per.values() if m.best_holdout_smape is not None]
    smapes = [m.best_holdout_smape for m in scored]
    return {
        "datasets": len(per),
        "datasets_with_valid_twin": len(scored),
        "mean_best_holdout_smape": statistics.fmean(smapes) if smapes else None,
        "worst_best_holdout_smape": max(smapes) if smapes else None,
        "mean_improvement_auc": statistics.fmean(m.improvement_auc for m in per.values())
        if per
        else 0.0,
        "mean_distinct_families": statistics.fmean(
            m.distinct_plausible_families for m in per.values()
        )
        if per
        else 0.0,
        "mean_valid_rate": statistics.fmean(m.valid_rate for m in per.values())
        if per
        else 0.0,
        "total_usd_cost": sum(m.usd_cost for m in per.values()),
        "total_iters": sum(m.n_iters for m in per.values()),
    }


def run_bench(
    cfg: LoopConfig,
    *,
    dry_run: bool = False,
    datasets: list[str] | None = None,
    runs_dir: Path | None = None,
) -> BenchResult:
    runs_dir = runs_dir or config.RUNS_DIR
    names = datasets or suite.DEFAULT_SUITE
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_dir = runs_dir / f"bench-{stamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    per: dict[str, LoopMetrics] = {}
    for name in names:
        entry = suite.SUITE[name]
        with tempfile.TemporaryDirectory(prefix=f"medusa-bench-{name}-") as tmp:
            processed_dir = Path(tmp)
            dataset = entry.materialize(processed_dir)
            result = run_loop(
                dataset,
                cfg,
                dry_run=dry_run,
                runs_dir=bench_dir,
                processed_dir=processed_dir,
            )
            per[name] = result.scorecard

    aggregate = _aggregate(per)
    (bench_dir / "bench.json").write_text(
        json.dumps(
            {
                "config": json.loads(cfg.to_json()),
                "dry_run": dry_run,
                "per_dataset": {k: v.to_dict() for k, v in per.items()},
                "aggregate": aggregate,
            },
            indent=2,
            default=str,
        )
    )
    return BenchResult(bench_dir=bench_dir, per_dataset=per, aggregate=aggregate)
