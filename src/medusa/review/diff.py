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
    the caller decides whether that should still let the push through.

    base_ref/head_ref come from local config/env/CLI flags -- the developer's own
    machine, not untrusted input -- so they're passed to git as-is. Revisit this if
    a remote sink is ever added that runs against a PR-supplied ref."""
    head = cfg.head_ref or "HEAD"

    try:
        # _default_base() also shells out to git -- keep it inside the try so a
        # missing git binary is reported the same way as a failure further down.
        base = cfg.base_ref or _default_base()
        diff = _run(["git", "diff", "--merge-base", base, head])
        if not diff.strip():
            return None
        names = _run(["git", "diff", "--merge-base", base, head, "--name-only"])
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"git diff {base}..{head} failed: {stderr or exc}") from exc
    except OSError as exc:
        # e.g. FileNotFoundError if `git` isn't on PATH -- CalledProcessError doesn't
        # cover that case, but callers only expect RuntimeError from this function.
        raise RuntimeError(f"could not run git: {exc}") from exc

    changed_files = [f for f in names.splitlines() if f]

    truncated = False
    encoded = diff.encode()
    if len(encoded) > cfg.max_diff_bytes:
        # Cut on a line boundary so we never hand the model a diff sliced mid-hunk (or
        # mid multi-byte UTF-8 sequence). Only ever take a PREFIX of the byte-limited
        # slice -- appending a newline instead could push a tiny budget back over it.
        clipped = encoded[: cfg.max_diff_bytes].decode(errors="ignore")
        last_newline = clipped.rfind("\n")
        if last_newline == -1:
            # Not even one complete line fits the budget -- there's nothing sane to
            # send the model (a mid-line slice would look like noise, not a diff).
            return None
        diff = clipped[: last_newline + 1]
        truncated = True

    return DiffResult(
        diff=diff, changed_files=changed_files, base_ref=base, head_ref=head, truncated=truncated
    )
