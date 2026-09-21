"""Prompts for the meta-agent, plus proposal parsing and a canned dry-run client."""

from __future__ import annotations

import json
import re
import textwrap

from medusa.config import LoopConfig
from medusa.meta.genome import COMPONENT_FIELDS, ENUM_SPECS, KNOB_SPECS, Genome

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

One lever pairs up: `critic_enabled` / `critic_every` turn on an in-run critic that
appends a short natural-language note to every Nth iteration's prompt, and `critic_prompt`
is that critic's instructions. It already defaults on for real-data domains and off for
synthetic/business ones (a measured A/B, not a guess); touching `critic_enabled` FORCES it
the other way for every domain in this run, so only do it if the reflection shows a
specific domain plateauing or stuck on one family despite (or for lack of) its default.
It costs extra tokens each time it fires (charged to the cost objective).

Separately, `critic_stance` (coach | skeptic) picks which of two canned system prompts the
critic runs with -- a ships-coach that suggests the next mechanism, or a skeptic that
red-teams the twin's stated assumptions. A same-day A/B on `ipb-ecoli` (real E. coli data,
critic already on) found the stance is not interchangeable: coach -> holdout sMAPE 0.065,
skeptic -> 0.043, both far below critic-off's 0.283 -- the skeptic caught a segmentation-
pipeline artifact (detection efficiency improving as cells separate) that the coach's
next-mechanism framing structurally can't flag. Touching this knob re-seeds `critic_prompt`
to the chosen stance's canned text, so don't touch both `critic_stance` and `critic_prompt`
in the same generation -- the stance change would just be overwritten right back.

## Output format -- follow EXACTLY

Line 1:  `TOUCHED: <component-or-knob name>`
Line 2:  `WHY: <2-3 sentences tying this to the reflection>`

Then, for a **knob**:
`KNOB: <number, true/false, or the stance name for critic_stance>`

For a **component**, a SURGICAL find/replace using these exact sentinel lines (NOT code
fences -- the content may itself contain ``` ):

<<<FIND
<an EXACT substring of the component's current text, shown below in full>
<<<REPLACE
<what that substring becomes>
<<<END

To ADD text, make FIND a short unique anchor from the current text and repeat that
anchor inside REPLACE. Change ONE small thing; keep it well under a 40% size change;
never touch an interface/contract line.
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
    "first_iteration_text", "critic_prompt", "system_prompt", "system_prompt_saas",
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
    for name, choices in ENUM_SPECS.items():
        cur = genome._clamped_knobs().get(name, getattr(base, name))
        out.append(f"- `{name}` = {cur}  (allowed {'|'.join(choices)}, enum)")
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

    if touched in ENUM_SPECS:
        m_knob = re.search(r"KNOB:\s*([A-Za-z_]+)", text)
        if not m_knob or m_knob.group(1) not in ENUM_SPECS[touched]:
            return None
        return {"touched": touched, "rationale": why, "knob": {touched: m_knob.group(1)}}

    # component: a surgical FIND / REPLACE using sentinel lines (robust to ``` in content)
    fr = re.search(
        r"<<<FIND[ \t]*\n(.*?)\n<<<REPLACE[ \t]*\n(.*?)\n<<<END", text, re.DOTALL
    )
    if fr:
        return {"touched": touched, "rationale": why,
                "edit": {"find": fr.group(1), "replace": fr.group(2)}}
    return None


# --- canned meta client (dry-run: no API) --------------------------------------

_CANNED = [
    "TOUCHED: archive_summary_top_k\nWHY: canned -- show fewer families back\nKNOB: 4",
    "TOUCHED: temperature\nWHY: canned -- cool it for consistency\nKNOB: 0.45",
    ("TOUCHED: diversity_nudge_text\nWHY: canned -- blunter diversity ask\n\n"
     "<<<FIND\nmaterially different structure\n<<<REPLACE\n"
     "materially different structure (new state variables, not a re-tune)\n<<<END"),
    ("TOUCHED: critic_enabled\nWHY: canned -- the loop is plateauing on one family; "
     "try the in-run critic\nKNOB: true"),
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
