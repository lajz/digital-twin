"""The feedback loop: prompt -> candidate twin.py -> harness score -> feedback -> repeat.

Everything tunable lives in the passed `LoopConfig`; this module holds no literals that
a meta-loop would want to change.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np

from medusa import config, prompts
from medusa.agent.deepseek import get_client
from medusa.config import LoopConfig
from medusa.data.build import Dataset
from medusa.harness import plots
from medusa.harness.archive import Archive
from medusa.harness.sandbox import run_candidate
from medusa.harness.scorecard import LoopMetrics, loop_scorecard

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclasses.dataclass(slots=True)
class RunResult:
    run_dir: Path
    archive: Archive
    scorecard: LoopMetrics


def extract_code_block(text: str) -> str | None:
    blocks = _CODE_BLOCK.findall(text or "")
    if not blocks:
        stripped = (text or "").strip()
        return stripped if "class Twin" in stripped else None
    return max(blocks, key=len).strip()


def obs_table(obs, max_points: int) -> str:
    n = len(obs)
    idx = np.unique(np.linspace(0, n - 1, min(max_points, n)).astype(int))
    cols = ["time_h", *obs.channels()]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i in idx:
        vals = [f"{obs.time_s[i] / 3600.0:.3f}"]
        for _, arr in obs.channels().items():
            v = arr[i]
            vals.append(f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}")
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _previous_section(source: str, res) -> str:
    m = res.metrics
    summary = (
        f"family: {res.family}   mode: {res.mode}\n"
        f"valid: {res.is_valid}  |  combined holdout sMAPE: "
        f"{m.get('holdout_smape', float('nan')):.4f}  |  implied doubling time: "
        f"{m.get('implied_doubling_h', float('nan')):.2f} h  |  plausibility: "
        f"{m.get('plausibility', float('nan')):.2f}\n"
    )
    if res.per_observable:
        summary += "per-observable sMAPE: " + ", ".join(
            f"{k}={v:.3f}" for k, v in res.per_observable.items()
        ) + "\n"
    if res.runtime_s:
        summary += f"runtime: {res.runtime_s:.1f} s (over budget: {res.over_budget})\n"
    if res.error:
        summary += f"error: {res.error}\n"
    if res.checker_report and res.checker_report != "OK":
        summary += f"checker:\n{res.checker_report}\n"
    return f"```python\n{source}\n```\n\n{summary}"


def _render_iter(idir: Path, demo_dir, source: str, res, dataset: Dataset, i: int) -> None:
    """Forecast plot (series) or rollout + side-by-side comparison (spatial)."""
    if not res.is_valid:
        return
    try:
        if dataset.task.mode == "spatial":
            from medusa.harness.evaluate import spatial_rollout
            from medusa.spatial.render import comparison_figure

            real, twin_roll = spatial_rollout(source, res.params, dataset, seed=0)
            twin_roll.to_npz(idir / "rollout.npz")
            fig = comparison_figure(
                real, twin_roll,
                split_time_s=dataset.split["t_split_s"],
                out_path=idir / "comparison.png",
                title=f"iter {i}: {res.family}  (holdout sMAPE {res.score:.3f})",
            )
            if demo_dir is not None:
                demo_dir.mkdir(exist_ok=True)
                (demo_dir / f"iter_{i:02d}.png").write_bytes(Path(fig).read_bytes())
        else:
            plots.forecast_plot(
                source, res.params, dataset, idir / "forecast.png",
                title=f"iter {i}: {res.family} (holdout sMAPE {res.score:.3f})",
            )
    except Exception as exc:  # pragma: no cover - rendering is best-effort
        (idir / "render_error.txt").write_text(repr(exc))


def _write_portfolio(run_dir: Path, archive: Archive, sources: dict[int, str], dataset: Dataset) -> None:
    pdir = run_dir / "portfolio"
    pdir.mkdir(exist_ok=True)
    lines = [f"# Portfolio -- {dataset.name}  (task: {dataset.task.name})", ""]
    for rank, entry in enumerate(archive.portfolio(), start=1):
        fam_dir = pdir / f"{rank:02d}_{entry.family}"
        fam_dir.mkdir(exist_ok=True)
        src = sources.get(entry.iteration, "")
        (fam_dir / "twin.py").write_text(src)
        (fam_dir / "params.json").write_text(json.dumps(entry.params, indent=2))
        (fam_dir / "metrics.json").write_text(json.dumps(entry.metrics, indent=2, default=str))
        _render_iter(fam_dir, None, src, _AsResult(entry), dataset, entry.iteration)
        m = entry.metrics
        lines += [
            f"## {rank}. {entry.family}  (iteration {entry.iteration})",
            "",
            f"- combined holdout sMAPE: **{entry.score:.4f}**",
            "- per-observable: " + ", ".join(
                f"{k} {v:.3f}" for k, v in entry.per_observable.items()
            ),
            f"- implied doubling time: {m.get('implied_doubling_h', float('nan')):.2f} h "
            f"(plausibility {m.get('plausibility', float('nan')):.2f})",
            f"- parameters: `{json.dumps(entry.params)}`",
            "",
        ]
    (pdir / "portfolio.md").write_text("\n".join(lines))


class _AsResult:
    """Adapt an ArchiveEntry to the subset of EvalResult that _render_iter reads."""

    def __init__(self, entry) -> None:
        self.is_valid = True
        self.params = entry.params
        self.family = entry.family
        self.score = entry.score
        self.metrics = entry.metrics
        self.per_observable = entry.per_observable


def run_loop(
    dataset: Dataset,
    cfg: LoopConfig,
    *,
    dry_run: bool = False,
    runs_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> RunResult:
    runs_dir = runs_dir or config.RUNS_DIR
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"{stamp}-{dataset.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = run_dir / "demo"

    (run_dir / "config.json").write_text(cfg.to_json())
    (run_dir / "meta.json").write_text(
        json.dumps(
            {"dataset": dataset.name, "task": dataset.task.to_dict(), "started": stamp,
             "dry_run": dry_run, "split": dataset.split},
            indent=2, default=str,
        )
    )
    trace = (run_dir / "trace.jsonl").open("w")

    client = get_client(cfg, dry_run=dry_run, task_name=dataset.task.name)
    archive = Archive()
    sources: dict[int, str] = {}
    budget = cfg.runtime_budget_for(dataset.task.name)

    from medusa import domains

    dom = domains.get(dataset.name)
    base_prompt = cfg.system_prompt_override or (
        dom.system_prompt if dom else cfg.system_prompt_for(dataset.task.name)
    )
    system_prompt = base_prompt.replace("{twin_runtime_budget_s}", f"{budget:g}")
    fit_table = obs_table(dataset.fit, cfg.context_obs_max_points)
    previous_section = prompts.FIRST_ITERATION_PREVIOUS
    best_seen = float("inf")
    iters_since_improve = 0

    for i in range(1, cfg.max_iters + 1):
        nudge = (
            prompts.DIVERSITY_NUDGE
            if cfg.diversity_nudge_every and i % cfg.diversity_nudge_every == 0
            else ""
        )
        user_prompt = prompts.render_iteration(
            cfg.iteration_template,
            datasheet=dataset.datasheet or f"dataset: {dataset.name}",
            obs_table=fit_table,
            archive_summary=archive.summary_text(cfg.archive_summary_top_k),
            previous_section=previous_section,
            diversity_nudge=nudge,
        )

        completion = client.complete(system_prompt, user_prompt)
        source = extract_code_block(completion.text)

        idir = run_dir / f"iter_{i:02d}"
        idir.mkdir(exist_ok=True)
        (idir / "prompt.md").write_text(f"# SYSTEM\n\n{system_prompt}\n\n# USER\n\n{user_prompt}")
        msg = completion.text or "(empty response)"
        if completion.reasoning:
            msg += f"\n\n---\n## reasoning (finish: {completion.finish_reason})\n\n{completion.reasoning}"
        (idir / "agent_msg.md").write_text(msg)

        base_row = {
            "iter": i,
            "finish_reason": completion.finish_reason,
            "prompt_tokens": completion.prompt_tokens,
            "response_tokens": completion.response_tokens,
        }

        if source is None:
            (idir / "metrics.json").write_text(json.dumps({"status": "no code block"}, indent=2))
            trace.write(json.dumps({**base_row, "family": None, "status": "no_code_block",
                                    "is_valid": False, "holdout_smape": None,
                                    "wall_s": 0.0}) + "\n")
            trace.flush()
            previous_section = (
                "Your previous response contained no usable ```python code block. "
                "Return exactly one fenced python block."
            )
            continue

        (idir / "twin.py").write_text(source)
        sources[i] = source

        res = run_candidate(source, cfg, processed_dir=processed_dir)
        entry = archive.add(i, res)
        (idir / "metrics.json").write_text(json.dumps(res.to_dict(), indent=2, default=str))
        _render_iter(idir, demo_dir, source, res, dataset, i)

        trace.write(json.dumps({
            **base_row,
            "family": res.family,
            "mode": res.mode,
            "status": entry.status,
            "is_valid": res.is_valid,
            "score": None if res.score == float("inf") else res.score,
            "holdout_smape": res.metrics.get("holdout_smape"),
            "per_observable": res.per_observable,
            "plausibility": res.metrics.get("plausibility"),
            "implied_doubling_h": res.metrics.get("implied_doubling_h"),
            "wall_s": res.runtime_s,
        }, default=str) + "\n")
        trace.flush()

        previous_section = _previous_section(source, res)

        if res.is_valid and res.score < best_seen - 1e-4:
            best_seen = res.score
            iters_since_improve = 0
        else:
            iters_since_improve += 1

        if (
            archive.distinct_valid_families >= cfg.min_families
            and archive.best_score <= cfg.target_smape
            and iters_since_improve >= cfg.plateau_patience
        ):
            break

    trace.close()
    (run_dir / "leaderboard.json").write_text(archive.to_json())
    _write_portfolio(run_dir, archive, sources, dataset)

    if dataset.task.mode == "spatial" and archive.portfolio():
        try:
            from medusa.spatial.player import build_player

            top = archive.portfolio()[0]
            build_player(
                dataset, sources[top.iteration], top.params, run_dir / "player.html",
                family=top.family or "twin",
            )
        except Exception:  # pragma: no cover - best effort
            pass

    scorecard = loop_scorecard(run_dir, target_smape=cfg.target_smape)
    (run_dir / "scorecard.json").write_text(json.dumps(scorecard.to_dict(), indent=2, default=str))
    try:
        from medusa.demo import build_demo

        build_demo(run_dir)
    except Exception:  # pragma: no cover
        pass

    return RunResult(run_dir=run_dir, archive=archive, scorecard=scorecard)
