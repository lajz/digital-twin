"""Register the real E. coli microscopy datasets as domains (L0 / L1 / L2)."""

from __future__ import annotations

from medusa import prompts
from medusa.contract.interface import POPULATION_TASK, SPATIAL_TASK, STRUCTURED_TASK
from medusa.data import fetch
from medusa.domains.base import Domain


def _register() -> None:
    from medusa.domains import register

    register(Domain(
        name="ipb-ecoli",
        task=POPULATION_TASK,
        build=lambda **kw: fetch.build_ipb_ecoli(**kw),
        system_prompt=prompts.SYSTEM_PROMPT,
        blurb="L0 -- real E. coli microcolony, cell count vs time",
        kind="real",
    ))
    register(Domain(
        name="ipb-ecoli-structured",
        task=STRUCTURED_TASK,
        build=lambda **kw: fetch.build_ipb_ecoli_structured(**kw),
        system_prompt=prompts.STRUCTURED_SYSTEM_PROMPT,
        blurb="L1 -- real E. coli, size-structured (count + biomass + mean length + CV)",
        kind="real",
    ))
    register(Domain(
        name="ipb-ecoli-spatial",
        task=SPATIAL_TASK,
        build=lambda **kw: fetch.build_ipb_ecoli_spatial(**kw),
        system_prompt=prompts.SPATIAL_SYSTEM_PROMPT,
        blurb="L2 -- real E. coli, per-cell rods (agent-based; scored on spatial stats)",
        kind="real",
    ))


_register()
