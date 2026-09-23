"""Unified dataset registry: synthetic presets + real downloadable datasets.

`build_dataset(name)` is the single entry point used by the CLI and the bench suite.
"""

from __future__ import annotations

from pathlib import Path

from medusa.data import build, fetch, synthetic

REAL_DATASETS = ("ipb-ecoli", "ipb-ecoli-structured", "ipb-ecoli-spatial", "lynx-hare")


def list_datasets() -> list[str]:
    from medusa import domains

    return sorted({d.name for d in domains.list_domains()} | set(synthetic.PRESETS))


def is_real(name: str) -> bool:
    from medusa import domains

    dom = domains.get(name)
    return name in REAL_DATASETS or (dom is not None and dom.kind != "synthetic")


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

    from medusa import domains

    dom = domains.get(name)
    if dom is not None:
        return dom.build(fit_frac=fit_frac, **kw)

    if name in synthetic.PRESETS:
        return build.build_synthetic(name, fit_frac=fit_frac, **kw)
    if name == "ipb-ecoli":
        return fetch.build_ipb_ecoli(fit_frac=fit_frac, **kw)
    raise ValueError(
        f"unknown dataset {name!r}; choose from {', '.join(list_datasets())}"
    )
