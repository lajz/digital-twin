"""Thin wrapper over the DeepSeek chat API (OpenAI-compatible) + a dry-run stand-in."""

from __future__ import annotations

import dataclasses
import os
import time

from medusa import config
from medusa.agent import canned
from medusa.agent.cache import SHARED, ResponseCache
from medusa.config import LoopConfig


@dataclasses.dataclass(slots=True)
class Completion:
    text: str
    prompt_tokens: int
    response_tokens: int
    model: str
    reasoning: str | None = None
    finish_reason: str | None = None


class DryRunClient:
    """Cycles through canned twin sources for a task. No network, no cost."""

    model = "dry-run"

    def __init__(self, sources: list[str] | None = None, *, task_name: str = "population") -> None:
        self._sources = sources or canned.wrapped_sequence(task_name)
        self._i = 0

    def complete(self, system: str, user: str) -> Completion:  # noqa: ARG002
        text = self._sources[self._i % len(self._sources)]
        self._i += 1
        return Completion(
            text=text,
            prompt_tokens=len(system) // 4 + len(user) // 4,
            response_tokens=len(text) // 4,
            model=self.model,
        )


class DeepSeekClient:
    def __init__(
        self, cfg: LoopConfig, *, max_retries: int = 4, cache: ResponseCache | None = SHARED
    ) -> None:
        from openai import OpenAI

        config.load_dotenv()
        api_key = os.environ.get(config.DEEPSEEK_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"{config.DEEPSEEK_API_KEY_ENV} is not set (put it in .env or the "
                f"environment). Use --dry-run to exercise the loop without an API key."
            )
        self._client = OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)
        self._cfg = cfg
        self._max_retries = max_retries
        self._cache = cache
        self.model = cfg.model

    def _sig(self) -> str:
        return f"{self._cfg.model}|t={self._cfg.temperature:g}|think={int(self._cfg.thinking)}"

    def complete(self, system: str, user: str) -> Completion:
        if self._cache is not None:
            hit = self._cache.get(self._sig(), system, user)
            if hit is not None:
                return Completion(**hit)
        completion = self._complete_uncached(system, user)
        if self._cache is not None:
            self._cache.put(self._sig(), system, user, completion)
        return completion

    def _complete_uncached(self, system: str, user: str) -> Completion:
        # DeepSeek V4: thinking is a request-level toggle, not a model name. It must be
        # sent explicitly -- v4-flash reasons by default and can burn the whole token
        # budget before emitting any content.
        thinking = "enabled" if self._cfg.thinking else "disabled"
        max_tokens = self._cfg.max_response_tokens
        if self._cfg.thinking:
            max_tokens = max(max_tokens, 24000)  # leave room for reasoning + answer

        kwargs: dict = {
            "model": self._cfg.model,
            "temperature": self._cfg.temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "extra_body": {"thinking": {"type": thinking}},
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                usage = resp.usage
                choice = resp.choices[0]
                return Completion(
                    text=choice.message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    response_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    model=resp.model or self._cfg.model,
                    reasoning=getattr(choice.message, "reasoning_content", None),
                    finish_reason=choice.finish_reason,
                )
            except Exception as exc:  # noqa: BLE001 - broad retry
                last_exc = exc
                time.sleep(min(2**attempt, 20))
        raise RuntimeError(f"DeepSeek API failed after {self._max_retries} attempts: {last_exc}")


def get_client(
    cfg: LoopConfig, *, dry_run: bool, task_name: str = "population",
    cache: ResponseCache | None = SHARED,
):
    return DryRunClient(task_name=task_name) if dry_run else DeepSeekClient(cfg, cache=cache)
