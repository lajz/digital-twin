"""Register the Hudson's Bay Company lynx-hare pelt-count series as a `Domain`.

Two coupled, oscillating populations (predator-prey) -- the classic dataset that
inspired the Lotka-Volterra equations. Structurally unlike every other domain in this
project: no lag phase, no saturation, no accounting identity, and both channels are
non-monotonic with a delayed reciprocal causation between them instead. This exists as
a generalization probe for the already-tuned loop (system prompts, critic, toolkit),
not a new domain to hill-climb -- see the PR that introduced this file.
"""

from __future__ import annotations

from medusa import prompts
from medusa.contract.interface import Task
from medusa.data import fetch
from medusa.data.fetch import YEAR_S  # single definition, shared with build_lynx_hare
from medusa.domains.base import Domain

PREDATOR_PREY_TASK = Task(
    name="predator-prey",
    observables=("hare", "lynx"),
    # sMAPE is already scale-invariant (a % error per channel, see
    # harness/metrics.py::smape), so equal weight treats hare and lynx as equally
    # important to explain rather than upweighting whichever has the larger raw count.
    weights={"hare": 1.0, "lynx": 1.0},
    mode="series",
    exogenous=(),  # unlike saas-seed, no external levers -- the two populations only
    # drive each other. plausibility intentionally omitted: the harness's plausibility
    # scorer only knows a fixed set of hardcoded derived quantities (doubling time,
    # mean cell length, ARPA, monthly churn) -- none has a Lotka-Volterra analog, and
    # "doubling time" isn't a coherent concept for a series that goes up AND down. See
    # data/fetch.py::LYNX_HARE_DATASHEET for the literature LV-parameter reference
    # values (used only to inform the system prompt's PARAMS priors, not scored).
    period_s=YEAR_S,
    period_label="yr",
)


def _register() -> None:
    from medusa.domains import register

    register(Domain(
        name="lynx-hare",
        task=PREDATOR_PREY_TASK,
        build=lambda **kw: fetch.build_lynx_hare(**kw),
        system_prompt=prompts.PREDATOR_PREY_SYSTEM_PROMPT,
        blurb="real Hudson's Bay Company lynx/hare pelt counts, 1900-1920 -- coupled "
              "predator-prey population dynamics",
        kind="real",
    ))


_register()
