"""`medusa` command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from medusa import config
from medusa.config import DEFAULT_LOOP_CONFIG, LoopConfig
from medusa.data import build, datasets, fetch


def _cfg_from_args(args: argparse.Namespace) -> LoopConfig:
    overrides = {}
    for field in ("model", "temperature", "max_iters", "target_smape",
                  "min_families", "diversity_nudge_every", "critic_every"):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if getattr(args, "thinking", False):
        overrides["thinking"] = True
    stance = getattr(args, "critic_stance", None)
    if stance:
        from medusa import critic

        overrides["critic_stance"] = stance
        overrides["critic_prompt"] = critic.prompt_for_stance(stance)
    if getattr(args, "critic", False) or stance:
        overrides["critic_enabled"] = True
    return dataclasses.replace(DEFAULT_LOOP_CONFIG, **overrides)


def _ensure_dataset(args: argparse.Namespace) -> build.Dataset:
    name = getattr(args, "dataset", None)
    if name:
        print(f"building dataset '{name}'.")
        return datasets.build_dataset(name, fit_frac=getattr(args, "fit_frac", 0.6))
    if config.OBSERVATIONS_PARQUET.exists() and not getattr(args, "rebuild", False):
        return build.load()
    print("No processed dataset found -- building 'synthetic-ecoli-fast'.")
    return datasets.build_dataset(
        "synthetic-ecoli-fast", fit_frac=getattr(args, "fit_frac", 0.6)
    )


# --- subcommands ---------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from medusa.harness.plots import sanity_plot

    ds = datasets.build_dataset(
        args.dataset or "synthetic-ecoli-fast", fit_frac=args.fit_frac
    )
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


def cmd_domains(args: argparse.Namespace) -> int:
    from medusa import domains

    for d in domains.list_domains():
        n_con = f"  [{len(d.constraints)} constraints]" if d.constraints else ""
        print(f"  {d.name:<24} {d.kind:<9} {d.blurb}{n_con}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    if getattr(args, "dataset", None) == "ipb-ecoli":
        path = fetch.fetch_ipb_ecoli()
        print(f"downloaded: {path}")
        print("now: uv run medusa build --dataset ipb-ecoli")
        return 0
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
    if sc.critic_response_tokens:
        print(f"  of which critic:       {sc.critic_prompt_tokens} / {sc.critic_response_tokens}")
    print(f"est. cost:               ${sc.usd_cost:.4f}")
    print(f"\nportfolio: {result.run_dir / 'portfolio' / 'portfolio.md'}")
    for e in result.archive.portfolio():
        print(f"  #{e.family:<20} holdout sMAPE {e.score:.4f}  (iter {e.iteration})")
    demo = result.run_dir / "demo.html"
    if demo.exists():
        print(f"\ndemo:  open {demo}")
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


def cmd_demo(args: argparse.Namespace) -> int:
    from medusa.demo import build_demo

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run()
    if run_dir is None or not run_dir.exists():
        print("no run found")
        return 1
    out = build_demo(run_dir)
    print(f"wrote {out}" if out else "no renders to build a demo from")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    from medusa.data import datasets
    from medusa.spatial.player import build_player

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run()
    if run_dir is None or not run_dir.exists():
        print("no run found")
        return 1
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("task", {}).get("name") != "spatial":
        print(f"'{meta.get('dataset')}' is not a spatial run; nothing to animate")
        return 1

    src, params, family = _resolve_twin(run_dir, args)
    if src is None:
        print("could not resolve a twin to play")
        return 1

    ds = datasets.build_dataset(meta["dataset"])
    out = build_player(ds, src, params, run_dir / "player.html", family=family)
    print(f"wrote {out}  (open in a browser)")
    return 0


def _resolve_twin(run_dir: Path, args: argparse.Namespace):
    if getattr(args, "iter", None):
        idir = run_dir / f"iter_{args.iter:02d}"
        if (idir / "twin.py").exists():
            m = json.loads((idir / "metrics.json").read_text())
            return ((idir / "twin.py").read_text(), m.get("params") or {},
                    m.get("family") or f"iter {args.iter}")
    pdir = run_dir / "portfolio"
    fams = sorted(pdir.glob("[0-9][0-9]_*")) if pdir.exists() else []
    if getattr(args, "family", None):
        fams = [d for d in fams if d.name.split("_", 1)[1] == args.family] or fams
    if fams:
        d = fams[0]
        return (d / "twin.py").read_text(), json.loads((d / "params.json").read_text()), \
            d.name.split("_", 1)[1]
    return None, {}, ""


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


def _latest_meta() -> Path | None:
    if not config.RUNS_DIR.exists():
        return None
    ms = [p for p in config.RUNS_DIR.glob("meta-*") if p.is_dir()]
    return max(ms, key=lambda p: p.stat().st_mtime) if ms else None


def cmd_meta(args: argparse.Namespace) -> int:
    from medusa.config import MetaConfig
    from medusa.meta.loop import run_meta_loop

    mc = MetaConfig(
        generations=args.generations,
        usd_budget=args.budget if args.budget is not None else MetaConfig().usd_budget,
    )
    print(f"meta-loop: generations={args.generations}, dry_run={args.dry_run}, "
          f"budget=${mc.usd_budget}")
    res = run_meta_loop(mc, generations=args.generations, dry_run=args.dry_run)
    print(f"\nmeta dir: {res.meta_dir}")
    if res.checkpoint:
        print("\n" + (res.meta_dir / "meta_checkpoint.md").read_text())
    else:
        print("\n" + (res.meta_dir / "meta_report.md").read_text())
    return 0


def cmd_meta_report(args: argparse.Namespace) -> int:
    from medusa.meta.report import build_meta_html

    meta_dir = Path(args.meta_dir) if args.meta_dir else _latest_meta()
    if meta_dir is None or not meta_dir.exists():
        print("no meta run found")
        return 1
    html = build_meta_html(meta_dir)  # (re)build the dashboard, works on partial runs
    for name in ("meta_checkpoint.md", "meta_report.md"):
        p = meta_dir / name
        if p.exists():
            print(p.read_text())
            print(f"\ndashboard: open {html}")
            return 0
    print(f"{meta_dir}: no checkpoint/report yet\ndashboard: open {html}")
    return 0


def cmd_meta_promote(args: argparse.Namespace) -> int:
    import tomllib

    from medusa.meta.genome import Genome

    genome = Genome.from_dict(json.loads(Path(args.genome).read_text()))
    ok, msg = genome.validates()
    if not ok:
        print(f"refusing to promote: {msg}")
        return 1
    table = {**genome.components, **genome._clamped_knobs()}
    toml_path = config.REPO_ROOT / "medusa.toml"
    existing = tomllib.loads(toml_path.read_text()).get("loop", {}) if toml_path.exists() else {}

    print("This will write [loop] overrides to medusa.toml:")
    for k, v in table.items():
        old = existing.get(k, "<default>")
        shown = v if not isinstance(v, str) else f"<{len(v)} chars>"
        print(f"  {k}: {old if not isinstance(old, str) else f'<{len(old)} chars>'} -> {shown}")
    if not args.yes:
        print("\nre-run with --yes to apply")
        return 0

    lines = ["# written by `medusa meta-promote`", "[loop]"]
    for k, v in table.items():
        lines.append(f"{k} = {_toml_value(v)}")
    toml_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {toml_path}")
    return 0


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v)  # a JSON string is a valid TOML basic string (newlines -> \n)


# --- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medusa", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    names = datasets.list_datasets()

    b = sub.add_parser("build", help="build a dataset -> data/processed/")
    b.add_argument("--dataset", choices=names, metavar="NAME",
                   help=f"dataset to build ({', '.join(names)})")
    b.add_argument("--fit-frac", type=float, default=0.6)
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("fetch", help="download / explain a real public dataset")
    f.add_argument("--dataset", choices=datasets.REAL_DATASETS, metavar="NAME",
                   help="download this real dataset's raw file into data/raw/")
    f.set_defaults(func=cmd_fetch)

    sub.add_parser("domains", help="list registered domain adapters").set_defaults(
        func=cmd_domains
    )

    def add_loop_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dry-run", action="store_true", help="use canned twins, no API")
        sp.add_argument("--iters", dest="max_iters", type=int)
        sp.add_argument("--model")
        sp.add_argument("--temperature", type=float)
        sp.add_argument("--target-smape", dest="target_smape", type=float)
        sp.add_argument("--min-families", dest="min_families", type=int)
        sp.add_argument("--diversity-nudge-every", dest="diversity_nudge_every", type=int)
        sp.add_argument("--thinking", action="store_true")
        sp.add_argument("--critic", action="store_true",
                        help="enable the in-run critic (soft NL feedback into each prompt)")
        sp.add_argument("--critic-stance", dest="critic_stance", choices=("coach", "skeptic"),
                        help="critic stance to seed (implies --critic)")
        sp.add_argument("--critic-every", dest="critic_every", type=int,
                        help="run the critic every Nth scored iteration")

    r = sub.add_parser("run", help="run the feedback loop on the processed dataset")
    add_loop_args(r)
    r.add_argument("--dataset", choices=names, metavar="NAME",
                   help="build + use this dataset instead of the processed one")
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

    dm = sub.add_parser("demo", help="(re)build demo.html for a run from its renders")
    dm.add_argument("run_dir", nargs="?")
    dm.set_defaults(func=cmd_demo)

    pl = sub.add_parser("play", help="build player.html: animate reality vs a spatial twin")
    pl.add_argument("run_dir", nargs="?")
    pl.add_argument("--family", help="portfolio family to play (default: #1)")
    pl.add_argument("--iter", type=int, help="play a specific iteration's twin instead")
    pl.set_defaults(func=cmd_play)

    mt = sub.add_parser("meta", help="hill-climb the loop's config + tooling (meta-loop)")
    mt.add_argument("--generations", type=int, default=1,
                    help="1 (default) = one generation then a checkpoint; run longer only "
                         "after the checkpoint verdict is green")
    mt.add_argument("--dry-run", action="store_true", help="canned inner + meta agents, no API")
    mt.add_argument("--budget", type=float, help="USD cap (default 3.0)")
    mt.set_defaults(func=cmd_meta)

    mr = sub.add_parser("meta-report", help="print the latest (or given) meta run's report")
    mr.add_argument("meta_dir", nargs="?")
    mr.set_defaults(func=cmd_meta_report)

    mpr = sub.add_parser("meta-promote", help="write a genome's overrides to medusa.toml")
    mpr.add_argument("genome", help="path to a genome_*.json")
    mpr.add_argument("--yes", action="store_true", help="apply (otherwise just show the diff)")
    mpr.set_defaults(func=cmd_meta_promote)

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
