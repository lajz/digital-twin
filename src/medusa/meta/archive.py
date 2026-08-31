"""Multi-objective archive of genomes: a Pareto front, not a single best."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from medusa.bench.runner import BenchResult
from medusa.meta.genome import Genome

_SMAPE_CAP = 1.0
MIN_FAMILIES = 2          # fixed meta constraint -- a genome can't lower this
MIN_VALID_RATE = 0.5


@dataclasses.dataclass(slots=True)
class GenomeScore:
    mean_best_smape: float          # over datasets; a dataset w/ no valid twin -> cap
    worst_best_smape: float
    mean_usd_cost: float
    mean_distinct_families: float
    mean_valid_rate: float
    n_datasets: int
    bench_dir: str

    @property
    def feasible(self) -> bool:
        return (self.mean_distinct_families >= MIN_FAMILIES
                and self.mean_valid_rate >= MIN_VALID_RATE)

    @property
    def objectives(self) -> tuple[float, float, float]:
        """All lower-is-better."""
        return (self.mean_best_smape, self.worst_best_smape, self.mean_usd_cost)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["feasible"] = self.feasible
        return d

    @classmethod
    def from_bench(cls, br: BenchResult) -> "GenomeScore":
        per = list(br.per_dataset.values())
        smapes = [m.best_holdout_smape if m.best_holdout_smape is not None else _SMAPE_CAP
                  for m in per]
        return cls(
            mean_best_smape=sum(smapes) / len(smapes),
            worst_best_smape=max(smapes),
            mean_usd_cost=sum(m.usd_cost for m in per) / len(per),
            mean_distinct_families=sum(m.distinct_plausible_families for m in per) / len(per),
            mean_valid_rate=sum(m.valid_rate for m in per) / len(per),
            n_datasets=len(per),
            bench_dir=str(br.bench_dir),
        )


@dataclasses.dataclass(slots=True)
class GenomeEntry:
    genome: Genome
    train: GenomeScore
    val: GenomeScore
    test: GenomeScore | None = None

    def to_dict(self) -> dict:
        return {
            "genome": self.genome.to_dict(),
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict() if self.test else None,
        }


def _dominates(a: GenomeScore, b: GenomeScore) -> bool:
    oa, ob = a.objectives, b.objectives
    return all(x <= y for x, y in zip(oa, ob)) and any(x < y for x, y in zip(oa, ob))


class MetaArchive:
    def __init__(self) -> None:
        self.entries: list[GenomeEntry] = []

    def add(self, entry: GenomeEntry) -> None:
        self.entries.append(entry)

    # --- queries ------------------------------------------------------------

    def feasible(self) -> list[GenomeEntry]:
        return [e for e in self.entries if e.val.feasible]

    def front(self) -> list[GenomeEntry]:
        feas = self.feasible() or self.entries
        return [e for e in feas
                if not any(_dominates(o.val, e.val) for o in feas if o is not e)]

    def best_on_val(self) -> GenomeEntry | None:
        feas = self.feasible() or self.entries
        if not feas:
            return None
        return min(feas, key=lambda e: (e.val.mean_best_smape, e.val.worst_best_smape,
                                        e.val.mean_usd_cost))

    def sample_parent(self, rng) -> GenomeEntry:
        front_ids = {id(e) for e in self.front()}
        weights = []
        for i, e in enumerate(self.entries):
            w = 3.0 if id(e) in front_ids else 1.0
            w *= 1.0 + 0.15 * i / max(len(self.entries) - 1, 1)  # mild recency bonus
            weights.append(w)
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for e, w in zip(self.entries, weights):
            acc += w
            if r <= acc:
                return e
        return self.entries[-1]

    # --- io ---------------------------------------------------------------

    def leaderboard(self) -> list[dict]:
        rows = sorted(self.entries, key=lambda e: e.val.mean_best_smape)
        return [e.to_dict() for e in rows]

    def to_json(self) -> str:
        best = self.best_on_val()
        return json.dumps(
            {
                "n_genomes": len(self.entries),
                "front": [e.to_dict() for e in self.front()],
                "best_on_val": best.to_dict() if best else None,
                "leaderboard": self.leaderboard(),
            },
            indent=2, default=str,
        )

    def write_portfolio(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(self.front(), key=lambda e: e.val.mean_best_smape)
        for i, e in enumerate(ranked, start=1):
            (out_dir / f"genome_{i:02d}.json").write_text(
                json.dumps(e.genome.to_dict(), indent=2)
            )
        (out_dir / "front.json").write_text(
            json.dumps([e.to_dict() for e in ranked], indent=2, default=str)
        )
