"""`medusa-review` command-line entry point: advisory AI review of the current branch
vs its base. Wired into `.githooks/pre-push` and safe to run manually."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from medusa.review.config import api_key, load_config
from medusa.review.diff import collect_diff
from medusa.review.passes import dedupe, run_pass
from medusa.review.report import print_findings
from medusa.review.types import SEVERITIES, Finding

EPILOGUE = """
Model config comes from .env / environment: DEEPSEEK_API_KEY (shared with the rest of
medusa) or MEDUSA_REVIEW_API_KEY for a separate key; MEDUSA_REVIEW_MODEL (default
deepseek-v4-flash) and MEDUSA_REVIEW_BASE_URL (default https://api.deepseek.com).
See .medusa-review.json for committed defaults.
"""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="medusa-review",
        description="Advisory AI review of the current branch vs its base.",
        epilog=EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base", dest="base_ref", help="diff against this ref (default: auto-detected)"
    )
    p.add_argument("--head", dest="head_ref", help="head of the diff (default: HEAD)")
    p.add_argument(
        "--min", dest="min_severity", choices=SEVERITIES, help="lowest severity to print"
    )
    only = p.add_mutually_exclusive_group()
    only.add_argument("--review-only", action="store_true", help="skip the security pass")
    only.add_argument("--security-only", action="store_true", help="skip the general review pass")
    p.add_argument(
        "--fail-on",
        dest="fail_on",
        choices=SEVERITIES,
        help="exit non-zero if a finding at or above this severity exists",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    overrides: dict[str, Any] = {
        "base_ref": args.base_ref,
        "head_ref": args.head_ref,
        "min_severity": args.min_severity,
        "fail_on": args.fail_on,
    }
    if args.review_only:
        overrides["review_pass"], overrides["security_pass"] = True, False
    if args.security_only:
        overrides["review_pass"], overrides["security_pass"] = False, True

    cfg = load_config(overrides)

    try:
        collected = collect_diff(cfg)
    except RuntimeError as exc:
        print(f"medusa-review: {exc} -- skipping")
        return 0
    if collected is None:
        print("medusa-review: no reviewable changes vs base -- skipping")
        return 0

    if not api_key():
        print("medusa-review: no API key set -- skipping AI review (see --help)")
        return 0

    diff_for_model = collected.diff
    if collected.truncated:
        print(f"medusa-review: diff exceeds {cfg.max_diff_bytes} bytes, truncated")
        # Tell the model the diff is partial so it doesn't report a spurious finding
        # about the abrupt cutoff.
        diff_for_model += (
            f"\n\n[diff truncated at {cfg.max_diff_bytes} bytes -- changes past this "
            "point are not shown]"
        )
    print(
        f"medusa-review: reviewing {len(collected.changed_files)} file(s) "
        f"({collected.base_ref}..{collected.head_ref}) with {cfg.model} ..."
    )

    findings: list[Finding] = []
    for name, enabled in (("review", cfg.review_pass), ("security", cfg.security_pass)):
        if not enabled:
            continue
        try:
            findings += run_pass(name, diff_for_model, cfg)
        except RuntimeError as exc:
            # One pass failing (a transient API error, a missing prompt file) must
            # not skip the other -- each pass covers different ground.
            print(f"medusa-review: {name} pass failed ({exc}) -- skipping it")
    findings = dedupe(findings)

    print_findings(findings, cfg)

    if cfg.fail_on:
        # SEVERITIES is ordered most-severe-first, so a lower rank is more severe.
        rank = {s: i for i, s in enumerate(SEVERITIES)}
        if any(rank[f.severity] <= rank[cfg.fail_on] for f in findings):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
