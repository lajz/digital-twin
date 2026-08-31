"""Score the *loop itself*: reduce a run's trace.jsonl to LoopMetrics.

This is the substrate the Milestone 7 meta-loop optimizes against. Keep it a pure
function of the on-disk trace so any run (past or present) can be re-scored.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from medusa import config

_SMAPE_CAP = 1.0  # invalid / missing candidates count as this bad for the AUC


@dataclasses.dataclass(slots=True)
class LoopMetrics:
    dataset: str
    n_iters: int
    best_holdout_smape: float | None
    iters_to_target: int | None
    distinct_plausible_families: int
    valid_rate: float
    candidate_crash_rate: float
    improvement_auc: float
    total_prompt_tokens: int
    total_response_tokens: int
    usd_cost: float
    wall_s_total: float
    critic_prompt_tokens: int = 0       # of the totals above, the share spent on the in-run critic
    critic_response_tokens: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _read_trace(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "trace.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def loop_scorecard(
    run_dir: str | Path,
    *,
    target_smape: float = 0.10,
    price_in: float | None = None,
    price_out: float | None = None,
) -> LoopMetrics:
    run_dir = Path(run_dir)
    rows = _read_trace(run_dir)
    price_in = config.PRICE_PER_MTOK_IN if price_in is None else price_in
    price_out = config.PRICE_PER_MTOK_OUT if price_out is None else price_out

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    n = len(rows)
    valid = [r for r in rows if r.get("is_valid")]
    crashed = [r for r in rows if r.get("status") == "crashed"]

    best_curve: list[float] = []
    best = math.inf
    iters_to_target: int | None = None
    for i, r in enumerate(rows, start=1):
        s = r.get("holdout_smape")
        if r.get("is_valid") and s is not None and math.isfinite(s):
            best = min(best, s)
            if iters_to_target is None and best <= target_smape:
                iters_to_target = i
        best_curve.append(min(best, _SMAPE_CAP) if math.isfinite(best) else _SMAPE_CAP)

    auc = (
        sum(max(0.0, 1.0 - b) for b in best_curve) / len(best_curve)
        if best_curve
        else 0.0
    )

    fam = {r.get("family") for r in valid if r.get("family")}
    crit_prompt_tok = sum(int(r.get("critic_prompt_tokens") or 0) for r in rows)
    crit_resp_tok = sum(int(r.get("critic_response_tokens") or 0) for r in rows)
    prompt_tok = sum(int(r.get("prompt_tokens") or 0) for r in rows) + crit_prompt_tok
    resp_tok = sum(int(r.get("response_tokens") or 0) for r in rows) + crit_resp_tok

    return LoopMetrics(
        dataset=meta.get("dataset", "unknown"),
        n_iters=n,
        best_holdout_smape=None if best == math.inf else best,
        iters_to_target=iters_to_target,
        distinct_plausible_families=len(fam),
        valid_rate=len(valid) / n if n else 0.0,
        candidate_crash_rate=len(crashed) / n if n else 0.0,
        improvement_auc=auc,
        total_prompt_tokens=prompt_tok,
        total_response_tokens=resp_tok,
        usd_cost=prompt_tok / 1e6 * price_in + resp_tok / 1e6 * price_out,
        wall_s_total=sum(float(r.get("wall_s") or 0.0) for r in rows),
        critic_prompt_tokens=crit_prompt_tok,
        critic_response_tokens=crit_resp_tok,
    )
