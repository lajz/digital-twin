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

Reply with ONE fenced ```json block and nothing else:

```json
{{"touched": "<component-or-knob name>",
  "rationale": "<2-3 sentences tying this to the reflection>",
  "component": {{"<name>": "<the COMPLETE new text for that component>"}}}}
```

or, for a knob:

```json
{{"touched": "temperature", "rationale": "...", "knob": {{"temperature": 0.5}}}}
```
"""

META_ITERATION_TEMPLATE = """\
## Current genome (generation {generation})

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

_JSON_BLOCK = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def component_menu(base: LoopConfig, genome: Genome) -> str:
    out = []
    for name in COMPONENT_FIELDS:
        cur = genome.components.get(name, getattr(base, name, ""))
        preview = textwrap.shorten(cur.replace("\n", " "), width=160, placeholder=" …")
        out.append(f"- `{name}` ({len(cur)} chars): {preview}")
    return "\n".join(out)


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
    blocks = _JSON_BLOCK.findall(text or "")
    if not blocks:
        blocks = re.findall(r"(\{.*\})", text or "", re.DOTALL)
    for raw in reversed(blocks):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and ("component" in d or "knob" in d):
            return d
    return None


# --- canned meta client (dry-run: no API) --------------------------------------

_CANNED = [
    {"touched": "temperature", "rationale": "canned: cool it down for consistency",
     "knob": {"temperature": 0.55}},
    {"touched": "diversity_nudge_every",
     "rationale": "canned: push families sooner", "knob": {"diversity_nudge_every": 2}},
    {"touched": "diversity_nudge_text",
     "rationale": "canned: make the diversity ask blunter",
     "component": {"diversity_nudge_text":
                   "THIS ROUND: a mechanistic family not yet listed. Different state "
                   "variables or a different closure -- not the same model re-tuned."}},
]


class CannedMetaClient:
    model = "dry-run-meta"

    def __init__(self) -> None:
        self._i = 0

    def complete(self, system: str, user: str):  # noqa: ARG002
        from medusa.agent.deepseek import Completion

        payload = _CANNED[self._i % len(_CANNED)]
        self._i += 1
        text = f"```json\n{json.dumps(payload)}\n```"
        return Completion(text=text, prompt_tokens=len(system) // 4 + len(user) // 4,
                          response_tokens=len(text) // 4, model=self.model)
