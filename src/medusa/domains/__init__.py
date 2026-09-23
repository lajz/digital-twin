"""Domain adapters: everything that makes the loop specific to a subject.

Only three things are domain-specific -- the data reduction, what's scored, and the
agent's instructions. A `Domain` bundles them (plus optional hard constraints and a
renderer) behind one interface, so applying the loop to a new subject is registering an
adapter, not editing the harness.

    from medusa.domains import get, list_domains
    ds = get("saas-seed").build()

Adapters register themselves on import; importing this package imports the built-ins.
"""

from __future__ import annotations

from medusa.domains.base import Constraint, ConstraintResult, Domain

_REGISTRY: dict[str, Domain] = {}


def register(domain: Domain) -> Domain:
    _REGISTRY[domain.name] = domain
    return domain


def get(name: str) -> Domain | None:
    return _REGISTRY.get(name)


def list_domains() -> list[Domain]:
    return sorted(_REGISTRY.values(), key=lambda d: (d.kind, d.name))


# --- built-in adapters (import for side effects: registration) ------------------

from medusa.domains import ecoli, predator_prey, saas, synthetic_growth  # noqa: E402,F401

__all__ = [
    "Domain",
    "Constraint",
    "ConstraintResult",
    "register",
    "get",
    "list_domains",
]
