"""The meta-loop: an agent that hill-climbs the inner loop and its in-loop tooling.

Mirrors the inner loop one level up -- the meta-agent generates a `Genome` (prompt /
knob / helper-library components, all `LoopConfig` fields), scored by how well the inner
loop then does across a benchmark suite of domains. Reflective Pareto evolution
(GEPA-style): a natural-language digest of the inner loop's failure modes feeds the next
proposal; an archive keeps a Pareto front, not just the best.
"""

from medusa.meta.genome import COMPONENT_FIELDS, KNOB_SPECS, Genome
from medusa.meta.archive import GenomeEntry, GenomeScore, MetaArchive

__all__ = [
    "Genome",
    "COMPONENT_FIELDS",
    "KNOB_SPECS",
    "GenomeScore",
    "GenomeEntry",
    "MetaArchive",
]
