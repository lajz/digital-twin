"""Registry of datasets the loop is benchmarked against.

A benchmark entry knows how to materialize its dataset into a given processed dir, so
each loop run in the suite is fully isolated.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from medusa.data import build
from medusa.data.build import Dataset


@dataclasses.dataclass(frozen=True, slots=True)
class BenchEntry:
    name: str
    kind: str  # "synthetic" | "real"
    materialize: Callable[[Path], Dataset]


def _synthetic(preset: str) -> Callable[[Path], Dataset]:
    def _mk(processed_dir: Path) -> Dataset:
        return build.build_synthetic(
            preset,
            processed_dir=processed_dir,
            datasheet_path=processed_dir / "datasheet.md",
        )

    return _mk


SUITE: dict[str, BenchEntry] = {
    "synthetic-ecoli-fast": BenchEntry(
        "synthetic-ecoli-fast", "synthetic", _synthetic("synthetic-ecoli-fast")
    ),
    "synthetic-yeast-slow": BenchEntry(
        "synthetic-yeast-slow", "synthetic", _synthetic("synthetic-yeast-slow")
    ),
    "synthetic-bsub-mid": BenchEntry(
        "synthetic-bsub-mid", "synthetic", _synthetic("synthetic-bsub-mid")
    ),
}

DEFAULT_SUITE = list(SUITE)
