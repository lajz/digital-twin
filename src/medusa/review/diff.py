"""Collects the unified diff between a base ref and head for the review passes."""

from __future__ import annotations

import dataclasses
import subprocess

from medusa.review.types import ReviewConfig

_FALLBACK_BASES = ("origin/main", "origin/master", "master")


@dataclasses.dataclass(slots=True)
class DiffResult:
    diff: str
    changed_files: list[str]
    base_ref: str
    head_ref: str
    truncated: bool


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def _ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref], capture_output=True
        ).returncode
        == 0
    )


def _default_base() -> str:
    try:
        symbolic = _run(["git", "symbolic-ref", "refs/remotes/origin/HEAD"]).strip()
        ref = symbolic.removeprefix("refs/remotes/")
        if ref and _ref_exists(ref):
            return ref
    except subprocess.CalledProcessError:
        pass
    for ref in _FALLBACK_BASES:
        if _ref_exists(ref):
            return ref
    return "HEAD"


def collect_diff(cfg: ReviewConfig) -> DiffResult | None:
    """None means there's nothing to review (head == base, or the merge-base diff is empty).
    Raises RuntimeError with the underlying git error for a bad ref or non-git cwd --
    the caller decides whether that should still let the push through."""
    base = cfg.base_ref or _default_base()
    head = cfg.head_ref or "HEAD"

    try:
        diff = _run(["git", "diff", "--merge-base", base, head])
        if not diff.strip():
            return None
        names = _run(["git", "diff", "--merge-base", base, head, "--name-only"])
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"git diff {base}..{head} failed: {stderr or exc}") from exc

    changed_files = [f for f in names.splitlines() if f]

    truncated = False
    encoded = diff.encode()
    if len(encoded) > cfg.max_diff_bytes:
        # Cut on a line boundary so we never hand the model a diff sliced mid-hunk
        # (or mid multi-byte UTF-8 sequence).
        clipped = encoded[: cfg.max_diff_bytes].decode(errors="ignore")
        diff = clipped.rsplit("\n", 1)[0] + "\n"
        truncated = True

    return DiffResult(
        diff=diff, changed_files=changed_files, base_ref=base, head_ref=head, truncated=truncated
    )
