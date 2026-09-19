"""Runs one named review pass over a diff and turns the model's raw JSON into
deduplicated `Finding`s."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from medusa.review.provider import complete, extract_json
from medusa.review.types import SEVERITIES, Finding, ReviewConfig, Severity

PROMPT_DIR = Path(__file__).parent / "prompts"
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def _severity(value: Any) -> Severity:
    return value if value in SEVERITIES else "low"


def _line(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _str(value: Any, fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def normalize(raw: list[Any], pass_name: str) -> list[Finding]:
    findings = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        detail = _str(item.get("detail"))
        if not detail:
            continue
        findings.append(
            Finding(
                severity=_severity(item.get("severity")),
                file=_str(item.get("file"), "(unspecified)"),
                line=_line(item.get("line")),
                title=_str(item.get("title"), "Unlabeled finding"),
                detail=detail,
                suggestion=_str(item.get("suggestion")),
                pass_name=pass_name,
            )
        )
    return findings


def _slug(title: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", title.lower()))


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Collapse near-duplicate findings -- same issue described twice, or once per
    pass. Two findings merge only when they clearly name the same thing: same file +
    same line, or same file + same title. The more severe one wins.

    Deliberately ignores `pass_name`: this is a straight port of the source tool's
    own dedupe semantics (same file+line from two different passes is treated as one
    finding, not two), not an oversight. A same-line collision between an unrelated
    review and security finding is an accepted, known trade-off of that design --
    already re-flagged by this tool across several of its own review runs."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: _RANK[f.severity]):
        dup = any(
            k.file == f.file
            and ((f.line is not None and k.line == f.line) or _slug(k.title) == _slug(f.title))
            for k in kept
        )
        if not dup:
            kept.append(f)
    return kept


KNOWN_PASSES = ("review", "security")


def run_pass(name: str, diff: str, cfg: ReviewConfig) -> list[Finding]:
    if name not in KNOWN_PASSES:
        raise RuntimeError(f"unknown review pass '{name}' (expected one of {KNOWN_PASSES})")
    prompt_path = PROMPT_DIR / f"{name}.md"
    if not prompt_path.exists():
        raise RuntimeError(f"missing prompt file for pass '{name}': {prompt_path}")

    system = prompt_path.read_text()
    user = f"Here is the unified diff to review:\n\n{diff}"
    content = complete(system, user, cfg)
    parsed = extract_json(content)
    raw = parsed.get("findings")
    return normalize(raw if isinstance(raw, list) else [], name)
