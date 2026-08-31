"""A `Genome` -- the set of `LoopConfig` components the meta-agent may change."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

from medusa.config import LoopConfig

# Textual components. One generation rewrites at most one of these.
COMPONENT_FIELDS = (
    "helper_library",
    "iteration_template",
    "diversity_nudge_text",
    "first_iteration_text",
    "system_prompt",
    "system_prompt_structured",
    "system_prompt_spatial",
    "system_prompt_saas",
)

# Scalar knobs -> (lo, hi, kind). Frozen-for-fairness fields (model, *_timeout_s,
# *_budget_s, max_response_tokens) are deliberately absent.
KNOB_SPECS: dict[str, tuple] = {
    "temperature": (0.2, 1.2, "float"),
    "diversity_nudge_every": (0, 6, "int"),
    "context_obs_max_points": (20, 120, "int"),
    "archive_summary_top_k": (3, 10, "int"),
    "min_families": (2, 5, "int"),
    "plateau_patience": (2, 8, "int"),
    "max_iters": (6, 25, "int"),
    "thinking": (0, 1, "bool"),
}

# the iteration template must keep these slots or render_iteration loses the data
_REQUIRED_TEMPLATE_SLOTS = {"obs_table", "archive_summary", "previous_section"}
_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def clamp_knob(name: str, value) -> float | int | bool:
    lo, hi, kind = KNOB_SPECS[name]
    if kind == "bool":
        return bool(value)
    v = max(lo, min(hi, value))
    return int(round(v)) if kind == "int" else float(v)


@dataclasses.dataclass(slots=True)
class Genome:
    components: dict[str, str] = dataclasses.field(default_factory=dict)
    knobs: dict[str, object] = dataclasses.field(default_factory=dict)
    parent_id: str = "-"
    generation: int = 0
    rationale: str = ""
    touched: str = ""  # which component/knob this genome changed vs its parent

    # ---- identity -------------------------------------------------------------

    @property
    def genome_id(self) -> str:
        blob = json.dumps({"c": self.components, "k": self._clamped_knobs()}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def _clamped_knobs(self) -> dict:
        return {k: clamp_knob(k, v) for k, v in self.knobs.items() if k in KNOB_SPECS}

    # ---- apply / round-trip -------------------------------------------------

    def apply(self, base: LoopConfig) -> LoopConfig:
        overrides = {k: v for k, v in self.components.items() if k in COMPONENT_FIELDS}
        overrides.update(self._clamped_knobs())
        return base.replace(**overrides)

    def child(self, *, component: dict | None = None, knob: dict | None = None,
              rationale: str = "", generation: int = 0) -> "Genome":
        comps = dict(self.components)
        knobs = dict(self.knobs)
        touched = ""
        for name, text in (component or {}).items():
            if name in COMPONENT_FIELDS:
                comps[name] = text
                touched = name
        for name, val in (knob or {}).items():
            if name in KNOB_SPECS:
                knobs[name] = clamp_knob(name, val)
                touched = name
        return Genome(comps, knobs, parent_id=self.genome_id, generation=generation,
                      rationale=rationale, touched=touched)

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id, "parent_id": self.parent_id,
            "generation": self.generation, "touched": self.touched,
            "rationale": self.rationale, "components": self.components,
            "knobs": self._clamped_knobs(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(
            components=dict(d.get("components", {})),
            knobs=dict(d.get("knobs", {})),
            parent_id=d.get("parent_id", "-"),
            generation=int(d.get("generation", 0)),
            rationale=d.get("rationale", ""),
            touched=d.get("touched", ""),
        )

    # ---- validation --------------------------------------------------------

    def validates(self) -> tuple[bool, str]:
        for name in self.components:
            if name not in COMPONENT_FIELDS:
                return False, f"unknown component {name!r}"
        for name in self.knobs:
            if name not in KNOB_SPECS:
                return False, f"unknown knob {name!r}"

        tmpl = self.components.get("iteration_template")
        if tmpl is not None:
            import string

            slots = {n for _, n, _, _ in string.Formatter().parse(tmpl) if n}
            missing = _REQUIRED_TEMPLATE_SLOTS - slots
            if missing:
                return False, f"iteration_template dropped required slots: {sorted(missing)}"

        lib = self.components.get("helper_library")
        if lib is not None:
            if lib.count("```") % 2 != 0:
                return False, "helper_library has unbalanced ``` fences"
            blocks = [b for b in _CODE_BLOCK.findall(lib) if b.strip()]
            if len(blocks) < 2:
                return False, "helper_library lost its snippets (<2 non-empty code blocks)"
            import ast
            for block in blocks:
                try:
                    tree = ast.parse(block)
                except SyntaxError as exc:
                    return False, f"helper_library snippet does not parse: {exc}"
                if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           for n in tree.body):
                    return False, "a helper_library snippet defines no function"
        return True, "ok"

    def diff_summary(self, base: LoopConfig) -> str:
        lines = []
        for name, text in self.components.items():
            old = getattr(base, name, "")
            lines.append(f"- component `{name}`: {len(old)} -> {len(text)} chars")
        for name, val in self._clamped_knobs().items():
            lines.append(f"- knob `{name}`: {getattr(base, name)} -> {val}")
        return "\n".join(lines) or "(identical to base)"


def seed_genome() -> Genome:
    return Genome(parent_id="-", generation=0, touched="", rationale="baseline")
