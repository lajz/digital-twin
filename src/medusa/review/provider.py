"""Model call for a review pass: an OpenAI-compatible chat completion against DeepSeek,
plus tolerant JSON extraction from its response."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from medusa import config as medusa_config
from medusa.review import config as review_config
from medusa.review.types import ReviewConfig


def _client(cfg: ReviewConfig):
    from openai import OpenAI

    key = review_config.api_key()
    if not key:
        raise RuntimeError(
            f"No review API key set. Put {medusa_config.DEEPSEEK_API_KEY_ENV} (or "
            "MEDUSA_REVIEW_API_KEY for a separate budget) in .env or the environment."
        )
    return OpenAI(api_key=key, base_url=cfg.base_url)


def _scrub(text: str, key: str | None) -> str:
    """The final failure message embeds the raw SDK exception; some providers echo
    request details on error, so don't let the key itself end up in a printed log."""
    return text.replace(key, "***") if key else text


def complete(system: str, user: str, cfg: ReviewConfig) -> str:
    client = _client(cfg)
    key = review_config.api_key()
    last_exc: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                # Fixed, not a ReviewConfig field: this is the one call site, and low
                # temperature is what makes findings reproducible run to run -- not a
                # knob a caller should need to reach for.
                temperature=0.2,
                max_tokens=cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # deepseek-v4-flash reasons by default and can burn the whole token
                # budget on reasoning before emitting any content -- must disable it
                # explicitly (medusa.agent.deepseek carries the same note).
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content or ""
        # Broad on purpose, matching medusa.agent.deepseek.DeepSeekClient's identical
        # retry loop: this call site sees the same mix of transient (rate limit,
        # timeout) and permanent (bad key, bad model name) failures, and the repo's
        # existing convention is to retry-then-surface rather than special-case each
        # exception type.
        except Exception as exc:  # noqa: BLE001 - broad retry, surfaced on final failure
            last_exc = exc
            if attempt < cfg.retries:
                time.sleep(min(2**attempt, 20))
    message = _scrub(str(last_exc), key)
    raise RuntimeError(f"review model call failed after {cfg.retries + 1} attempts: {message}")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Models routinely wrap JSON in a markdown fence or add stray prose; pull the
    object out rather than requiring an exact match. Always returns a dict -- a
    top-level JSON array or scalar is treated the same as unparsable text, since
    callers do `.get("findings")` on the result."""
    text = text.strip()
    match = _FENCE_RE.search(text)
    candidate = match.group(1).strip() if match else text
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}
