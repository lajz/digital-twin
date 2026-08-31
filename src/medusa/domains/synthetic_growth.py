"""Register the synthetic bacterial-growth presets as domains (thin wrappers)."""

from __future__ import annotations

from medusa import prompts
from medusa.contract.interface import POPULATION_TASK
from medusa.data import build
from medusa.data.synthetic import PRESETS
from medusa.domains.base import Domain


def _register() -> None:
    from medusa.domains import register

    for name, preset in PRESETS.items():
        def _build(*, _n=name, **kw):
            return build.build_synthetic(_n, **kw)

        register(Domain(
            name=name,
            task=POPULATION_TASK,
            build=_build,
            system_prompt=prompts.SYSTEM_PROMPT,
            blurb=f"synthetic {preset.organism} growth curve (Baranyi + noise)",
            kind="synthetic",
        ))


_register()
