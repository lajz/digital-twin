"""Paths, environment loading, and the single source of loop configuration.

`LoopConfig` holds *every* tunable knob for the feedback loop. `agent/loop.py` reads
only from a `LoopConfig` instance -- it contains no literals of its own -- so a future
meta-loop (Milestone 7) can mutate a config and re-run the loop without touching code.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from medusa import prompts, toolkit

# --- paths -------------------------------------------------------------------

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = REPO_ROOT / "runs"

OBSERVATIONS_PARQUET = PROCESSED_DIR / "observations.parquet"
SPLIT_JSON = PROCESSED_DIR / "split.json"
DATASHEET_MD = DATA_DIR / "datasheet.md"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Does not overwrite existing env vars."""
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --- API / pricing ---------------------------------------------------------------

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

# Placeholder USD prices per 1M tokens for cost accounting in the scorecard.
# deepseek-chat/deepseek-reasoner aliases retired 2026-07-24; v4-flash is the cheap tier.
# Override with DEEPSEEK_PRICE_IN / DEEPSEEK_PRICE_OUT once exact numbers are confirmed.
PRICE_PER_MTOK_IN = float(os.environ.get("DEEPSEEK_PRICE_IN", "0.28"))
PRICE_PER_MTOK_OUT = float(os.environ.get("DEEPSEEK_PRICE_OUT", "0.42"))


# --- the loop configuration ----------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class LoopConfig:
    """Every knob the feedback loop can turn. Serializable for the meta-loop."""

    # model
    model: str = "deepseek-v4-flash"
    temperature: float = 0.7
    thinking: bool = False
    max_response_tokens: int = 12000

    # loop control
    max_iters: int = 20
    target_smape: float = 0.10
    min_families: int = 3
    plateau_patience: int = 4  # stop if best score unimproved this many iters (after target)

    # prompt shaping
    system_prompt: str = prompts.SYSTEM_PROMPT
    system_prompt_structured: str = prompts.STRUCTURED_SYSTEM_PROMPT
    system_prompt_spatial: str = prompts.SPATIAL_SYSTEM_PROMPT
    system_prompt_saas: str = prompts.SAAS_SYSTEM_PROMPT
    # set to override the domain's own system prompt (the meta-loop's hook)
    system_prompt_override: str | None = None
    iteration_template: str = prompts.ITERATION_TEMPLATE
    diversity_nudge_text: str = prompts.DIVERSITY_NUDGE
    first_iteration_text: str = prompts.FIRST_ITERATION_PREVIOUS
    helper_library: str = dataclasses.field(default_factory=toolkit.render_library)
    diversity_nudge_every: int = 3  # every Nth iter, push for an unexplored family
    context_obs_max_points: int = 60  # downsample fit-window table to <= this many rows
    archive_summary_top_k: int = 6  # families to describe back to the agent

    # harness
    candidate_timeout_s: float = 20.0
    twin_runtime_budget_s: float = 5.0
    # spatial (L2) tasks: agent-based rollouts are stochastic + slower
    spatial_candidate_timeout_s: float = 150.0
    spatial_runtime_budget_s: float = 90.0

    def system_prompt_for(self, task_name: str) -> str:
        return {
            "structured": self.system_prompt_structured,
            "spatial": self.system_prompt_spatial,
            "saas": self.system_prompt_saas,
        }.get(task_name, self.system_prompt)

    def runtime_budget_for(self, task_name: str) -> float:
        return self.spatial_runtime_budget_s if task_name == "spatial" else self.twin_runtime_budget_s

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def replace(self, **overrides) -> "LoopConfig":
        return dataclasses.replace(self, **overrides)

    @classmethod
    def load(cls, toml_path: Path | None = None) -> "LoopConfig":
        """Defaults merged with an optional `medusa.toml` [loop] table (promoted genome)."""
        import tomllib

        toml_path = toml_path or (REPO_ROOT / "medusa.toml")
        base = cls()
        if not toml_path.exists():
            return base
        table = tomllib.loads(toml_path.read_text()).get("loop", {})
        return base.from_dict({**dataclasses.asdict(base), **table})


DEFAULT_LOOP_CONFIG = LoopConfig.load()  # committed defaults + optional medusa.toml


def quick_config(base: LoopConfig | None = None, *, max_iters: int = 8) -> LoopConfig:
    """A LoopConfig for meta-evaluation: fewer inner iterations, faster stop."""
    base = base or DEFAULT_LOOP_CONFIG
    return dataclasses.replace(
        base, max_iters=min(base.max_iters, max_iters), plateau_patience=2,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class MetaConfig:
    """Knobs for the meta-loop (`medusa/meta/`)."""

    model: str = "deepseek-v4-flash"
    generations: int = 10
    usd_budget: float = 3.0
    patience: int = 3                # stop after this many generations w/o META_VAL gain
    quick_max_iters: int = 8         # inner iterations per meta-eval run
    train_suite: tuple[str, ...] = ()   # filled from bench.suite.META_TRAIN if empty
    val_suite: tuple[str, ...] = ()
    test_suite: tuple[str, ...] = ()
    temperature: float = 0.8


DEFAULT_META_CONFIG = MetaConfig()
