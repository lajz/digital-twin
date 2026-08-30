"""`medusa` command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from medusa import config
from medusa.config import DEFAULT_LOOP_CONFIG, LoopConfig
from medusa.data import build, fetch
from medusa.data.synthetic import PRESETS


def _cfg_from_args(args: argparse.Namespace) -> LoopConfig:
    overrides = {}
    for field in ("model", "temperature", "max_iters", "target_smape",
                  "min_families", "diversity_nudge_every"):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if getattr(args, "thinking", False):
        overrides["thinking"] = True
    return dataclasses.replace(DEFAULT_LOOP_CONFIG, **overrides)


def _ensure_dataset(args: argparse.Namespace) -> build.Dataset:
    if config.OBSERVATIONS_PARQUET.exists() and not getattr(args, "rebuild", False):
        return build.load()
    preset = getattr(args, "synthetic", None) or "synthetic-ecoli-fast"
    print(f"No processed dataset found -- building synthetic '{preset}'.")
    return build.build_synthetic(preset, fit_frac=getattr(args, "fit_frac", 0.6))


# --- subcommands ---------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from medusa.harness.plots import sanity_plot

    if args.synthetic:
        ds = build.build_synthetic(args.synthetic, fit_frac=args.fit_frac)
    else:
        ds = build.build_synthetic("synthetic-ecoli-fast", fit_frac=args.fit_frac)
    out = config.PROCESSED_DIR / "sanity.png"
    sanity_plot(ds, out)
    print(f"dataset:        {ds.name}")
    print(f"points:         {len(ds.observations)} ({ds.split['n_fit']} fit / "
          f"{ds.split['n_holdout']} holdout)")
    print(f"span:           {ds.observations.time_h[-1]:.1f} h")
    print(f"count range:    {ds.observations.population_count.min():.0f} .. "
          f"{ds.observations.population_count.max():.0f}")
    print(f"ground truth:   {json.dumps(ds.split.get('ground_truth', {}))}")
    print(f"written:        {config.OBSERVATIONS_PARQUET}")
    print(f"sanity plot:    {out}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    fetch.print_instructions()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from medusa.agent.loop import run_loop

    ds = _ensure_dataset(args)
    cfg = _cfg_from_args(args)
    print(f"running loop on '{ds.name}'  (model={cfg.model}, max_iters={cfg.max_iters}, "
          f"dry_run={args.dry_run})")
    result = run_loop(ds, cfg, dry_run=args.dry_run)

    sc = result.scorecard
    print(f"\nrun dir: {result.run_dir}")
    print(f"best holdout sMAPE:      {sc.best_holdout_smape}")
    print(f"distinct valid families: {sc.distinct_plausible_families}")
    print(f"valid rate:              {sc.valid_rate:.2f}")
    print(f"improvement AUC:         {sc.improvement_auc:.3f}")
    print(f"iters to target:         {sc.iters_to_target}")
    print(f"tokens (prompt/resp):    {sc.total_prompt_tokens} / {sc.total_response_tokens}")
    print(f"est. cost:               ${sc.usd_cost:.4f}")
    print(f"\nportfolio: {result.run_dir / 'portfolio' / 'portfolio.md'}")
    for e in result.archive.portfolio():
        print(f"  #{e.family:<14} holdout sMAPE {e.score:.4f}  (iter {e.iteration})")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from medusa.bench.runner import run_bench

    cfg = _cfg_from_args(args)
    result = run_bench(cfg, dry_run=args.dry_run, datasets=args.datasets or None)
    print(f"\nbench dir: {result.bench_dir}")
    for name, m in result.per_dataset.items():
        print(f"  {name:<24} best sMAPE {m.best_holdout_smape}  "
              f"families {m.distinct_plausible_families}  AUC {m.improvement_auc:.3f}")
    print("\naggregate:")
    print(json.dumps(result.aggregate, indent=2, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run()
    if run_dir is None or not run_dir.exists():
        print("no run found")
        return 1
    sc = run_dir / "scorecard.json"
    pf = run_dir / "portfolio" / "portfolio.md"
    print(f"# {run_dir}\n")
    if sc.exists():
        print("## scorecard\n")
        print(json.dumps(json.loads(sc.read_text()), indent=2))
    if pf.exists():
        print("\n" + pf.read_text())
    return 0


def _latest_run() -> Path | None:
    if not config.RUNS_DIR.exists():
        return None
    runs = [p for p in config.RUNS_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


# --- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medusa", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    presets = sorted(PRESETS)

    b = sub.add_parser("build", help="build a dataset -> data/processed/")
    b.add_argument("--synthetic", choices=presets, help="synthetic preset to build")
    b.add_argument("--fit-frac", type=float, default=0.6)
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("fetch", help="how to obtain a real public dataset")
    f.set_defaults(func=cmd_fetch)

    def add_loop_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dry-run", action="store_true", help="use canned twins, no API")
        sp.add_argument("--iters", dest="max_iters", type=int)
        sp.add_argument("--model")
        sp.add_argument("--temperature", type=float)
        sp.add_argument("--target-smape", dest="target_smape", type=float)
        sp.add_argument("--min-families", dest="min_families", type=int)
        sp.add_argument("--diversity-nudge-every", dest="diversity_nudge_every", type=int)
        sp.add_argument("--thinking", action="store_true")

    r = sub.add_parser("run", help="run the feedback loop on the processed dataset")
    add_loop_args(r)
    r.add_argument("--synthetic", choices=presets, help="build+use this preset first")
    r.add_argument("--rebuild", action="store_true")
    r.add_argument("--fit-frac", type=float, default=0.6)
    r.set_defaults(func=cmd_run)

    bn = sub.add_parser("bench", help="run the loop across the benchmark suite")
    add_loop_args(bn)
    bn.add_argument("--datasets", nargs="*", help="subset of suite dataset names")
    bn.set_defaults(func=cmd_bench)

    rp = sub.add_parser("report", help="print a run's scorecard + portfolio")
    rp.add_argument("run_dir", nargs="?")
    rp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
