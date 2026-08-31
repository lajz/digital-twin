"""Track candidates across iterations: per-family best, portfolio ranking, prompt summary."""

from __future__ import annotations

import dataclasses
import json
import math

from medusa.harness.evaluate import EvalResult


@dataclasses.dataclass(slots=True)
class ArchiveEntry:
    iteration: int
    family: str | None
    score: float
    is_valid: bool
    n_params: int
    metrics: dict
    params: dict | None
    status: str  # "ok" | short failure reason
    per_observable: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_eval(cls, iteration: int, res: EvalResult) -> "ArchiveEntry":
        if res.is_valid:
            status = "ok"
        elif res.crashed:
            status = "crashed"
        elif res.nondeterministic:
            status = "nondeterministic"
        elif res.over_budget:
            status = "over time budget"
        elif res.constraint_violation:
            status = f"constraint: {res.constraint_violation}"
        elif not res.passed_checks:
            status = "failed contract checks"
        else:
            status = "implausible"
        return cls(
            iteration=iteration,
            family=res.family,
            score=res.score,
            is_valid=res.is_valid,
            n_params=res.n_params,
            metrics=dict(res.metrics),
            params=res.params,
            status=status,
            per_observable=dict(res.per_observable),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Archive:
    def __init__(self) -> None:
        self.entries: list[ArchiveEntry] = []

    def add(self, iteration: int, res: EvalResult) -> ArchiveEntry:
        entry = ArchiveEntry.from_eval(iteration, res)
        self.entries.append(entry)
        return entry

    # --- queries -----------------------------------------------------------------

    def best_per_family(self) -> dict[str, ArchiveEntry]:
        best: dict[str, ArchiveEntry] = {}
        for e in self.entries:
            if not e.is_valid or e.family is None:
                continue
            cur = best.get(e.family)
            if cur is None or e.score < cur.score:
                best[e.family] = e
        return best

    def portfolio(self, k: int = 3) -> list[ArchiveEntry]:
        ranked = sorted(self.best_per_family().values(), key=lambda e: e.score)
        return ranked[:k]

    @property
    def best_score(self) -> float:
        valid = [e.score for e in self.entries if e.is_valid]
        return min(valid) if valid else math.inf

    @property
    def distinct_valid_families(self) -> int:
        return len(self.best_per_family())

    def families_tried(self) -> set[str]:
        return {e.family for e in self.entries if e.family}

    # --- rendering --------------------------------------------------------------

    def summary_text(self, top_k: int = 6) -> str:
        if not self.entries:
            return "(nothing tried yet)"
        best = self.best_per_family()
        lines = ["| family | best holdout sMAPE | plausibility | status (latest) |",
                 "|---|---|---|---|"]
        # group latest status per family (incl. invalid ones so the agent avoids repeats)
        seen: dict[str, ArchiveEntry] = {}
        for e in self.entries:
            key = e.family or f"(unnamed @ iter {e.iteration})"
            seen[key] = e
        for key, latest in list(seen.items())[:top_k]:
            b = best.get(key)
            smape = f"{b.score:.3f}" if b else "--"
            plaus = f"{b.metrics.get('plausibility', float('nan')):.2f}" if b else "--"
            lines.append(f"| {key} | {smape} | {plaus} | {latest.status} |")
        extra = []
        if self.best_score < math.inf:
            extra.append(f"Best valid holdout sMAPE so far: {self.best_score:.3f}.")
        extra.append(f"Distinct valid families: {self.distinct_valid_families}.")
        return "\n".join(lines) + "\n\n" + " ".join(extra)

    def leaderboard(self) -> list[dict]:
        return [e.to_dict() for e in sorted(self.entries, key=lambda e: e.score)]

    def to_json(self) -> str:
        return json.dumps(
            {
                "best_score": None if self.best_score == math.inf else self.best_score,
                "distinct_valid_families": self.distinct_valid_families,
                "portfolio": [e.to_dict() for e in self.portfolio()],
                "leaderboard": self.leaderboard(),
            },
            indent=2,
            default=str,
        )
