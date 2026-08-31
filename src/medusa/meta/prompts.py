"""Prompts for the meta-agent, plus proposal parsing and a canned dry-run client."""

from __future__ import annotations

import json
import re
import textwrap

from medusa.config import LoopConfig
from medusa.meta.genome import COMPONENT_FIELDS, KNOB_SPECS, Genome

META_SYSTEM_PROMPT = """\
You tune `medusa`, a feedback loop that writes mechanistic digital twins. Each inner run:
an agent writes `twin.py` (+ its calibration), a harness scores its forecast on a
held-out window, the score feeds back; over ~8 iterations it returns a portfolio of
model families. You are improving the loop's own configuration so it does this better
across many domains (bacterial growth curves AND SaaS businesses).

You change EXACTLY ONE thing per generation, a SMALL move, grounded in the reflection
you are given. You may change one **component** (a piece of text the inner loop uses):

{component_menu}

...or nudge one **knob**:

{knob_menu}

You may NOT change the model, timeouts, runtime budgets, or token limits.

## Output format -- follow EXACTLY

Line 1:  `TOUCHED: <component-or-knob name>`
Line 2:  `WHY: <2-3 sentences tying this to the reflection>`
Then, for a **knob**, line 3:  `KNOB: <number or true/false>`
or, for a **component**, a single fenced code block with the COMPLETE new text of
that component (any characters allowed inside -- do not escape anything):

```
<the entire new component text>
```

Nothing else. When you rewrite a component, start from its CURRENT text (shown below in
full) and make a SMALL, surgical edit -- do not invent a new contract.
"""

META_ITERATION_TEMPLATE = """\
## The components you may edit -- CURRENT TEXT IN FULL

{components_full}

## Knobs (current values)

{knobs_full}

## This genome's overrides so far (generation {generation})

{genome_state}

## Its scorecard on the training suite

mean best sMAPE {train_mean:.3f} | worst-case {train_worst:.3f} | \
mean families {train_fam:.1f} | mean cost ${train_cost:.3f} | valid rate {train_valid:.0%}

## Reflection on the last training run

{digest}

## Pareto front so far

{front}

Propose the next single change.
"""

_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_:.-]*\s*\n(.*?)```", re.DOTALL)

# components worth showing the meta-agent for a series-only meta-suite
MENU_COMPONENTS = (
    "helper_library", "iteration_template", "diversity_nudge_text",
    "first_iteration_text", "system_prompt", "system_prompt_saas",
)


def _effective(base: LoopConfig, genome: Genome, name: str) -> str:
    return genome.components.get(name, getattr(base, name, ""))


def component_menu(base: LoopConfig, genome: Genome) -> str:
    return "\n".join(
        f"- `{name}` ({len(_effective(base, genome, name))} chars)"
        for name in MENU_COMPONENTS
    )


def components_full(base: LoopConfig, genome: Genome) -> str:
    out = []
    for name in MENU_COMPONENTS:
        out.append(f"### `{name}`\n```\n{_effective(base, genome, name)}\n```")
    return "\n\n".join(out)


def knob_menu(base: LoopConfig, genome: Genome) -> str:
    out = []
    for name, (lo, hi, kind) in KNOB_SPECS.items():
        cur = genome._clamped_knobs().get(name, getattr(base, name))
        out.append(f"- `{name}` = {cur}  (allowed {lo}..{hi}, {kind})")
    return "\n".join(out)


def genome_state(base: LoopConfig, genome: Genome, *, full_components: bool = True) -> str:
    if not genome.components and not genome.knobs:
        return "(the committed baseline -- no overrides yet)"
    parts = []
    for name, text in genome.components.items():
        body = text if full_components else textwrap.shorten(text, 400, placeholder=" …")
        parts.append(f"### component `{name}`\n{body}")
    for name, val in genome._clamped_knobs().items():
        parts.append(f"### knob `{name}` = {val}")
    return "\n\n".join(parts)


def parse_proposal(text: str) -> dict | None:
    """Parse the TOUCHED / WHY / (KNOB | fenced block) format. Robust to any content."""
    text = text or ""
    m_touched = re.search(r"TOUCHED:\s*`?([A-Za-z_]+)`?", text)
    if not m_touched:
        return None
    touched = m_touched.group(1)
    why = ""
    m_why = re.search(r"WHY:\s*(.+?)(?:\n[A-Z]{3,}:|\n```|\Z)", text, re.DOTALL)
    if m_why:
        why = m_why.group(1).strip()

    if touched in KNOB_SPECS:
        m_knob = re.search(r"KNOB:\s*([-\d.]+|true|false|True|False)", text)
        if not m_knob:
            return None
        raw = m_knob.group(1)
        val = raw.lower() == "true" if raw.lower() in ("true", "false") else float(raw)
        return {"touched": touched, "rationale": why, "knob": {touched: val}}

    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        return None
    return {"touched": touched, "rationale": why,
            "component": {touched: max(blocks, key=len).strip()}}


# --- canned meta client (dry-run: no API) --------------------------------------

_CANNED = [
    "TOUCHED: diversity_nudge_every\nWHY: canned -- push families sooner\nKNOB: 2",
    "TOUCHED: temperature\nWHY: canned -- cool it for consistency\nKNOB: 0.45",
    ("TOUCHED: diversity_nudge_text\nWHY: canned -- blunter diversity ask\n\n```\n"
     "THIS ROUND: a mechanistic family not yet listed -- different state variables or a "
     "different closure, not the same model re-tuned.\n```"),
]


class CannedMetaClient:
    model = "dry-run-meta"

    def __init__(self) -> None:
        self._i = 0

    def complete(self, system: str, user: str):  # noqa: ARG002
        from medusa.agent.deepseek import Completion

        text = _CANNED[self._i % len(_CANNED)]
        self._i += 1
        return Completion(text=text, prompt_tokens=len(system) // 4 + len(user) // 4,
                          response_tokens=len(text) // 4, model=self.model)
