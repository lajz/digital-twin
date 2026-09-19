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


def _as_bool(value: Any, fallback: bool) -> bool:
    """A JSON bool comes through as a real `bool` already; this only guards a
    hand-edited `.medusa-review.json` where someone quoted it (`"false"`), which
    Python would otherwise treat as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return fallback


def load_config(overrides: dict[str, Any] | None = None, root: Path | None = None) -> ReviewConfig:
    medusa_config.load_dotenv()
    overrides = overrides or {}
    root = root or medusa_config.REPO_ROOT
    file_cfg = _read_file_config(root)
    env = os.environ
    defaults = ReviewConfig()

    def pick(key: str, env_key: str, cast=str):
        # `cast` applies no matter which source wins -- a quoted number in
        # .medusa-review.json (`"max_diff_bytes": "400000"`) must come out an int
        # here, not slip through as a str and blow up later at a numeric comparison.
        if overrides.get(key) is not None:
            value = overrides[key]
        elif env.get(env_key) is not None:
            value = env[env_key]
        elif file_cfg.get(key) is not None:
            value = file_cfg[key]
        else:
            return None
        return cast(value)

    def pick_or(key: str, env_key: str, fallback, cast=str):
        # `pick(...) or fallback` would silently replace an explicit 0 -- keep None
        # the only trigger for the fallback.
        value = pick(key, env_key, cast)
        return fallback if value is None else value

    passes = file_cfg.get("passes")
    if not isinstance(passes, dict):  # malformed .medusa-review.json -- fall back
        passes = {}
    fail_on_raw = pick("fail_on", "MEDUSA_REVIEW_FAIL_ON")
    # An unrecognized value fails closed (the strictest gate) rather than silently
    # disabling the gate a typo was trying to set.
    fail_on = _severity(fail_on_raw, "high") if fail_on_raw else None

    return ReviewConfig(
        # Every scalar field below goes through the same `pick` precedence chain
        # (overrides < env < file_cfg) -- no field-specific `or` chains. The two
        # booleans just below are the deliberate exception: `passes` is a nested
        # dict, not a flat env-backed key, so they read straight from
        # overrides/file_cfg via `_as_bool` instead.
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
        review_pass=_as_bool(
            overrides.get("review_pass", passes.get("review")), defaults.review_pass
        ),
        security_pass=_as_bool(
            overrides.get("security_pass", passes.get("security")), defaults.security_pass
        ),
        fail_on=fail_on,
    )


def api_key() -> str | None:
    """The review model's key: a dedicated override, else the same key the loop uses."""
    return os.environ.get("MEDUSA_REVIEW_API_KEY") or os.environ.get(medusa_config.DEEPSEEK_API_KEY_ENV)
