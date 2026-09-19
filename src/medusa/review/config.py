"""Config precedence: built-in defaults < .medusa-review.json (committed) < environment
< explicit overrides (CLI flags)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from medusa import config as medusa_config
from medusa.review.types import SEVERITIES, ReviewConfig, Severity


def _read_file_config(root: Path) -> dict[str, Any]:
    path = root / ".medusa-review.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _severity(value: Any, fallback: Severity) -> Severity:
    return value if value in SEVERITIES else fallback


def load_config(overrides: dict[str, Any] | None = None, root: Path | None = None) -> ReviewConfig:
    medusa_config.load_dotenv()
    overrides = overrides or {}
    root = root or medusa_config.REPO_ROOT
    file_cfg = _read_file_config(root)
    env = os.environ
    defaults = ReviewConfig()

    def pick(key: str, env_key: str, cast=str):
        if overrides.get(key) is not None:
            return overrides[key]
        if env.get(env_key) is not None:
            return cast(env[env_key])
        if file_cfg.get(key) is not None:
            return file_cfg[key]
        return None

    def pick_or(key: str, env_key: str, fallback, cast=str):
        # `pick(...) or fallback` would silently replace an explicit 0 -- keep None
        # the only trigger for the fallback.
        value = pick(key, env_key, cast)
        return fallback if value is None else value

    passes = file_cfg.get("passes", {})
    fail_on_raw = pick("fail_on", "MEDUSA_REVIEW_FAIL_ON")
    fail_on = _severity(fail_on_raw, "high") if fail_on_raw else None

    return ReviewConfig(
        # Every field below goes through the same `pick` precedence chain (overrides <
        # env < file_cfg) -- no field-specific `or` chains, so there's nothing for a
        # future pass over this function to accidentally make asymmetric.
        base_ref=pick("base_ref", "MEDUSA_REVIEW_BASE"),
        head_ref=pick("head_ref", "MEDUSA_REVIEW_HEAD"),
        base_url=pick_or("base_url", "MEDUSA_REVIEW_BASE_URL", defaults.base_url),
        model=pick_or("model", "MEDUSA_REVIEW_MODEL", defaults.model),
        max_diff_bytes=pick_or("max_diff_bytes", "MEDUSA_REVIEW_MAX_DIFF_BYTES", defaults.max_diff_bytes, int),
        min_severity=_severity(
            pick("min_severity", "MEDUSA_REVIEW_MIN_SEVERITY"), defaults.min_severity
        ),
        max_tokens=pick_or("max_tokens", "MEDUSA_REVIEW_MAX_TOKENS", defaults.max_tokens, int),
        retries=pick_or("retries", "MEDUSA_REVIEW_RETRIES", defaults.retries, int),
        review_pass=overrides.get("review_pass", passes.get("review", defaults.review_pass)),
        security_pass=overrides.get("security_pass", passes.get("security", defaults.security_pass)),
        fail_on=fail_on,
    )


def api_key() -> str | None:
    """The review model's key: a dedicated override, else the same key the loop uses."""
    return os.environ.get("MEDUSA_REVIEW_API_KEY") or os.environ.get(medusa_config.DEEPSEEK_API_KEY_ENV)
