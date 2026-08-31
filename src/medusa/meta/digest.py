"""Reflect on a bench directory -> a specific, actionable natural-language diagnosis.

Deterministic (no LLM). This is GEPA's core feedback signal: the meta-agent learns far
faster from "41% of first candidates blew the runtime budget, all using
differential_evolution" than from a scalar score.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path


def _runs(bench_dir: Path) -> list[Path]:
    return sorted(p.parent for p in bench_dir.glob("*/trace.jsonl"))


def _trace(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines() if l.strip()]


def reflect(bench_dir: str | Path) -> str:
    bench_dir = Path(bench_dir)
    runs = _runs(bench_dir)
    if not runs:
        return "(no runs to reflect on)"

    n_runs = len(runs)
    total_iters = 0
    first_iter_over_budget = 0
    over_budget = 0
    no_code = 0
    status_ct: collections.Counter = collections.Counter()
    constraint_ct: collections.Counter = collections.Counter()
    de_in_over_budget = 0
    per_run = []
    families_all: collections.Counter = collections.Counter()

    for run in runs:
        rows = _trace(run)
        total_iters += len(rows)
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        ds = meta.get("dataset", run.name)
        fams = {r.get("family") for r in rows if r.get("is_valid") and r.get("family")}
        families_all.update(fams)
        best = min(
            (r["holdout_smape"] for r in rows
             if r.get("is_valid") and isinstance(r.get("holdout_smape"), (int, float))),
            default=None,
        )
        for k, r in enumerate(rows):
            st = str(r.get("status", "?"))
            status_ct[st.split(":")[0]] += 1
            if st.startswith("constraint"):
                constraint_ct[st.split("constraint:")[-1].strip().split(":")[0]] += 1
            if st == "no_code_block":
                no_code += 1
            if st == "over time budget":
                over_budget += 1
                if k == 0:
                    first_iter_over_budget += 1
                tw = (run / f"iter_{r['iter']:02d}" / "twin.py")
                if tw.exists() and "differential_evolution" in tw.read_text():
                    de_in_over_budget += 1
        per_run.append({"dataset": ds, "n_iters": len(rows), "families": len(fams),
                        "best_smape": best})

    lines = [f"## Reflection on {n_runs} inner runs / {total_iters} candidate iterations", ""]
    for pr in per_run:
        bs = f"{pr['best_smape']:.3f}" if isinstance(pr["best_smape"], (int, float)) else "no valid twin"
        lines.append(f"- **{pr['dataset']}**: {pr['n_iters']} iters, {pr['families']} valid "
                     f"famil{'y' if pr['families']==1 else 'ies'}, best sMAPE {bs}")
    lines.append("")

    valid = status_ct.get("ok", 0)
    lines.append(f"Valid rate: {valid}/{total_iters} ({valid/max(total_iters,1):.0%}). "
                 f"Status mix: " + ", ".join(f"{k} {v}" for k, v in status_ct.most_common()))
    if over_budget:
        note = f" -- {de_in_over_budget} of them still call differential_evolution" if de_in_over_budget else ""
        lines.append(f"Over the runtime budget: {over_budget} candidates "
                     f"({first_iter_over_budget} were the very first attempt){note}.")
    if no_code:
        lines.append(f"No usable code block: {no_code} responses.")
    if constraint_ct:
        lines.append("Constraint failures: " +
                     ", ".join(f"{k} x{v}" for k, v in constraint_ct.most_common()))

    thin = [pr["dataset"] for pr in per_run if pr["families"] < 2]
    if thin:
        lines.append(f"Never reached a 2nd model family on: {', '.join(thin)}.")
    unsolved = [pr["dataset"] for pr in per_run if not isinstance(pr["best_smape"], (int, float))]
    if unsolved:
        lines.append(f"No valid twin at all on: {', '.join(unsolved)}.")

    lines += ["", "### Where to push next", _suggestions(
        status_ct, de_in_over_budget, no_code, thin, unsolved, per_run
    )]
    return "\n".join(lines)


def _suggestions(status_ct, de_over, no_code, thin, unsolved, per_run) -> str:
    tips = []
    if de_over:
        tips.append("- The prompt says don't use `differential_evolution` in `fit`, but "
                    "candidates still do. Make the constraint sharper or point at the "
                    "`robust_log_fit` helper explicitly.")
    if no_code:
        tips.append("- Some responses have no code block. Tighten the output-format line.")
    if thin:
        tips.append(f"- {', '.join(thin)} stay on one family. Raise `diversity_nudge_every` "
                    "or make `diversity_nudge_text` more forceful.")
    if unsolved:
        tips.append(f"- {', '.join(unsolved)}: no valid twin. The task prompt or a helper "
                    "for that regime is missing.")
    scored = [pr for pr in per_run if isinstance(pr["best_smape"], (int, float))]
    hard = [pr for pr in scored if pr["best_smape"] > 0.15]
    if hard:
        tips.append("- Above sMAPE 0.15: " +
                    ", ".join(f"{pr['dataset']} ({pr['best_smape']:.2f})" for pr in hard) +
                    ". The mechanism these need is likely not being proposed -- add it to "
                    "the relevant system prompt or a helper snippet.")
    if scored and not tips:
        worst = max(scored, key=lambda pr: pr["best_smape"])
        many_fam = [pr for pr in scored if pr["families"] >= pr["n_iters"]]
        if many_fam:
            tips.append(f"- {', '.join(pr['dataset'] for pr in many_fam)} used a new family "
                        "every iteration and never converged -- the loop explores but "
                        "doesn't refine. Lower `diversity_nudge_every` pressure, or tell the "
                        "agent to re-tune a promising family before switching.")
        tips.append(f"- Weakest result: {worst['dataset']} at sMAPE {worst['best_smape']:.3f}. "
                    "A small clarification in its system prompt about the dominant dynamic "
                    "(lag / saturation / regime change) is the cheapest lever.")
    return "\n".join(tips) or "- Nothing obviously broken; a small prompt clarification."
