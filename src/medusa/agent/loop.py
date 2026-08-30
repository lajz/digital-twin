"""The feedback loop: prompt -> candidate twin.py -> harness score -> feedback -> repeat.

Everything tunable lives in the passed `LoopConfig`; this module holds no literals that
a meta-loop would want to change.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import shutil
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
    rows = ["| time_h | population_count | total_area_um2 |", "|---|---|---|"]
    for i in idx:
        area = "" if obs.total_area_um2 is None else f"{obs.total_area_um2[i]:.1f}"
        rows.append(f"| {obs.time_s[i] / 3600.0:.3f} | {obs.population_count[i]:.0f} | {area} |")
    return "\n".join(rows)


def _previous_section(source: str, res) -> str:
    m = res.metrics
    summary = (
        f"family: {res.family}\n"
        f"valid: {res.is_valid}  |  holdout sMAPE: {m.get('holdout_smape', float('nan')):.4f}  "
        f"|  fit R^2: {m.get('fit_r2', float('nan')):.4f}  "
        f"|  implied doubling time: {m.get('implied_doubling_h', float('nan')):.2f} h  "
        f"|  plausibility: {m.get('plausibility', float('nan')):.2f}\n"
    )
    if res.error:
        summary += f"error: {res.error}\n"
    if res.checker_report and res.checker_report != "OK":
        summary += f"checker:\n{res.checker_report}\n"
    return f"```python\n{source}\n```\n\n{summary}"


def _write_portfolio(run_dir: Path, archive: Archive, sources: dict[int, str], dataset: Dataset) -> None:
    pdir = run_dir / "portfolio"
    pdir.mkdir(exist_ok=True)
    lines = [f"# Portfolio -- {dataset.name}", ""]
    for rank, entry in enumerate(archive.portfolio(), start=1):
        fam_dir = pdir / f"{rank:02d}_{entry.family}"
        fam_dir.mkdir(exist_ok=True)
        src = sources.get(entry.iteration, "")
        (fam_dir / "twin.py").write_text(src)
        (fam_dir / "params.json").write_text(json.dumps(entry.params, indent=2))
        (fam_dir / "metrics.json").write_text(json.dumps(entry.metrics, indent=2, default=str))
        try:
            plots.forecast_plot(
                src, entry.params, dataset, fam_dir / "forecast.png",
                title=f"#{rank} {entry.family} (holdout sMAPE {entry.score:.3f})",
            )
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            (fam_dir / "forecast_error.txt").write_text(str(exc))
        m = entry.metrics
        lines += [
            f"## {rank}. {entry.family}  (iteration {entry.iteration})",
            "",
            f"- holdout sMAPE: **{entry.score:.4f}**",
            f"- holdout MASE: {m.get('holdout_mase', float('nan')):.3f}",
            f"- fit R^2: {m.get('fit_r2', float('nan')):.4f}",
            f"- implied doubling time: {m.get('implied_doubling_h', float('nan')):.2f} h "
            f"(plausibility {m.get('plausibility', float('nan')):.2f})",
            f"- parameters: `{json.dumps(entry.params)}`",
            "",
        ]
    (pdir / "portfolio.md").write_text("\n".join(lines))


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

    (run_dir / "config.json").write_text(cfg.to_json())
    (run_dir / "meta.json").write_text(
        json.dumps(
            {"dataset": dataset.name, "started": stamp, "dry_run": dry_run,
             "split": dataset.split},
            indent=2, default=str,
        )
    )
    trace = (run_dir / "trace.jsonl").open("w")

    client = get_client(cfg, dry_run=dry_run)
    archive = Archive()
    sources: dict[int, str] = {}
    system_prompt = cfg.system_prompt.replace(
        "{twin_runtime_budget_s}", f"{cfg.twin_runtime_budget_s:g}"
    )
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
        (idir / "agent_msg.md").write_text(completion.text or "(empty response)")

        if source is None:
            res_dict = {"status": "no code block", "is_valid": False}
            (idir / "metrics.json").write_text(json.dumps(res_dict, indent=2))
            trace.write(json.dumps({
                "iter": i, "family": None, "status": "no_code_block", "is_valid": False,
                "holdout_smape": None, "plausibility": None,
                "prompt_tokens": completion.prompt_tokens,
                "response_tokens": completion.response_tokens, "wall_s": 0.0,
            }) + "\n")
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

        if res.is_valid:
            try:
                plots.forecast_plot(
                    source, res.params, dataset, idir / "forecast.png",
                    title=f"iter {i}: {res.family} (holdout sMAPE {res.score:.3f})",
                )
            except Exception:  # pragma: no cover
                pass

        trace.write(json.dumps({
            "iter": i,
            "family": res.family,
            "status": entry.status,
            "is_valid": res.is_valid,
            "score": None if res.score == float("inf") else res.score,
            "holdout_smape": res.metrics.get("holdout_smape"),
            "plausibility": res.metrics.get("plausibility"),
            "implied_doubling_h": res.metrics.get("implied_doubling_h"),
            "prompt_tokens": completion.prompt_tokens,
            "response_tokens": completion.response_tokens,
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

    scorecard = loop_scorecard(run_dir, target_smape=cfg.target_smape)
    (run_dir / "scorecard.json").write_text(json.dumps(scorecard.to_dict(), indent=2, default=str))

    return RunResult(run_dir=run_dir, archive=archive, scorecard=scorecard)
