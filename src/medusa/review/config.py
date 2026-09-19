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

    passes = file_cfg.get("passes", {})
    fail_on_raw = pick("fail_on", "MEDUSA_REVIEW_FAIL_ON")
    fail_on = _severity(fail_on_raw, "high") if fail_on_raw else None

    return ReviewConfig(
        base_ref=overrides.get("base_ref") or env.get("MEDUSA_REVIEW_BASE") or file_cfg.get("base_ref"),
        head_ref=overrides.get("head_ref") or env.get("MEDUSA_REVIEW_HEAD"),
        base_url=pick("base_url", "MEDUSA_REVIEW_BASE_URL") or defaults.base_url,
        model=pick("model", "MEDUSA_REVIEW_MODEL") or defaults.model,
        max_diff_bytes=pick("max_diff_bytes", "MEDUSA_REVIEW_MAX_DIFF_BYTES", int) or defaults.max_diff_bytes,
        min_severity=_severity(
            pick("min_severity", "MEDUSA_REVIEW_MIN_SEVERITY"), defaults.min_severity
        ),
        max_tokens=pick("max_tokens", "MEDUSA_REVIEW_MAX_TOKENS", int) or defaults.max_tokens,
        retries=pick("retries", "MEDUSA_REVIEW_RETRIES", int) or defaults.retries,
        review_pass=overrides.get("review_pass", passes.get("review", defaults.review_pass)),
        security_pass=overrides.get("security_pass", passes.get("security", defaults.security_pass)),
        fail_on=fail_on,
    )


def api_key() -> str | None:
    """The review model's key: a dedicated override, else the same key the loop uses."""
    return os.environ.get("MEDUSA_REVIEW_API_KEY") or os.environ.get(medusa_config.DEEPSEEK_API_KEY_ENV)
