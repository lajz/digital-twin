"""Datasets the loop is benchmarked against, and the meta-loop's train/val/test splits.

A benchmark entry materializes its dataset into a given processed dir, so each loop run
in the suite is fully isolated. SUITE is populated from the domain registry (spatial
domains excluded -- too slow/expensive for a bench sweep).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from medusa.data import datasets
from medusa.data.build import Dataset


@dataclasses.dataclass(frozen=True, slots=True)
class BenchEntry:
    name: str
    kind: str  # synthetic | real | business
    materialize: Callable[[Path], Dataset]


def _mk(name: str) -> Callable[[Path], Dataset]:
    def _build(processed_dir: Path) -> Dataset:
        return datasets.build_dataset(
            name, processed_dir=processed_dir, datasheet_path=processed_dir / "datasheet.md"
        )

    return _build


def _build_suite() -> dict[str, BenchEntry]:
    from medusa import domains

    out: dict[str, BenchEntry] = {}
    for d in domains.list_domains():
        if d.task.mode == "spatial":
            continue
        out[d.name] = BenchEntry(d.name, d.kind, _mk(d.name))
    return out


SUITE: dict[str, BenchEntry] = _build_suite()

# non-real (no network) fast series datasets, for `medusa bench` with no args
DEFAULT_SUITE = [n for n, e in SUITE.items() if e.kind == "synthetic"]

# --- meta-loop suite splits -------------------------------------------------------
# Disjoint. TRAIN is what the meta-loop evolves against; VAL ranks + early-stops;
# TEST is touched exactly once, at the end, for the honest number. Spans bacteria +
# business and several failure modes (long lag, sparse noise, market plateau).

# disjoint. saas-plateau (the flakiest) sits in TRAIN, where its variance only adds
# exploration pressure; VAL is the stabler datasets so the ranking signal is clean.
META_TRAIN = ("synthetic-ecoli-fast", "synthetic-lag-heavy", "saas-plateau")
META_VAL = ("synthetic-bsub-mid", "synthetic-yeast-slow", "saas-seed")
META_TEST = ("synthetic-noisy-sparse",)
