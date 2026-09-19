"""Shared types for the review tool."""

from __future__ import annotations

import dataclasses
from typing import Literal

Severity = Literal["high", "medium", "low", "nit"]
SEVERITIES: tuple[Severity, ...] = ("high", "medium", "low", "nit")


@dataclasses.dataclass(slots=True)
class Finding:
    severity: Severity
    file: str
    line: int | None
    title: str
    detail: str
    suggestion: str
    pass_name: str


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewConfig:
    base_ref: str | None = None
    head_ref: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    max_diff_bytes: int = 400_000
    min_severity: Severity = "nit"
    max_tokens: int = 16_000
    retries: int = 2
    review_pass: bool = True
    security_pass: bool = True
    fail_on: Severity | None = None
