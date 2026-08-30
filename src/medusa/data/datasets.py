"""Unified dataset registry: synthetic presets + real downloadable datasets.

`build_dataset(name)` is the single entry point used by the CLI and the bench suite.
"""

from __future__ import annotations

from pathlib import Path

from medusa.data import build, fetch, synthetic

REAL_DATASETS = ("ipb-ecoli", "ipb-ecoli-structured", "ipb-ecoli-spatial")


def list_datasets() -> list[str]:
    return [*sorted(synthetic.PRESETS), *REAL_DATASETS]


def is_real(name: str) -> bool:
    return name in REAL_DATASETS


def build_dataset(
    name: str,
    *,
    fit_frac: float = 0.6,
    processed_dir: Path | None = None,
    datasheet_path: Path | None = None,
) -> build.Dataset:
    kw = {}
    if processed_dir is not None:
        kw["processed_dir"] = processed_dir
        kw["datasheet_path"] = datasheet_path or (processed_dir / "datasheet.md")

    if name in synthetic.PRESETS:
        return build.build_synthetic(name, fit_frac=fit_frac, **kw)
    if name == "ipb-ecoli":
        return fetch.build_ipb_ecoli(fit_frac=fit_frac, **kw)
    if name == "ipb-ecoli-structured":
        return fetch.build_ipb_ecoli_structured(fit_frac=fit_frac, **kw)
    if name == "ipb-ecoli-spatial":
        return fetch.build_ipb_ecoli_spatial(fit_frac=fit_frac, **kw)
    raise ValueError(
        f"unknown dataset {name!r}; choose from {', '.join(list_datasets())}"
    )
