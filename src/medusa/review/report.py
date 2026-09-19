"""Prints findings to the terminal, grouped most-severe first."""

from __future__ import annotations

from medusa.review.types import SEVERITIES, Finding, ReviewConfig

_ICON = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e1", "nit": "⚪"}
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def print_findings(findings: list[Finding], cfg: ReviewConfig) -> None:
    shown = [f for f in findings if _RANK[f.severity] <= _RANK[cfg.min_severity]]
    if not shown:
        suffix = "" if cfg.min_severity == "nit" else f" at or above '{cfg.min_severity}'"
        print(f"medusa-review: no findings{suffix}")
        return

    for f in sorted(shown, key=lambda f: _RANK[f.severity]):
        loc = f"{f.file}:{f.line}" if f.line else f.file
        print(f"\n{_ICON[f.severity]} [{f.severity}] {f.pass_name} — {loc}")
        print(f"  {f.title}")
        print(f"  {f.detail}")
        if f.suggestion:
            print(f"  suggestion: {f.suggestion}")
    print()
