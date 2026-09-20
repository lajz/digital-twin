"""The AI review tool's pure logic: JSON extraction, finding normalization/dedup,
and config precedence. No network calls."""

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest

from medusa.review import cli
from medusa.review.config import load_config
from medusa.review.diff import DiffResult, collect_diff
from medusa.review.passes import dedupe, normalize
from medusa.review.provider import _scrub, extract_json
from medusa.review.types import Finding, ReviewConfig


# --- extract_json -------------------------------------------------------------


def test_extract_json_plain():
    assert extract_json('{"findings": []}') == {"findings": []}


def test_extract_json_fenced():
    text = '```json\n{"findings": [{"severity": "low"}]}\n```'
    assert extract_json(text) == {"findings": [{"severity": "low"}]}


def test_extract_json_handles_literal_braces_inside_values():
    text = '{"findings": [{"detail": "use {placeholder} syntax"}]}'
    assert extract_json(text) == {"findings": [{"detail": "use {placeholder} syntax"}]}


def test_extract_json_top_level_array_is_not_a_dict():
    # run_pass does `.get("findings")` on the result -- a bare array must not crash it.
    assert extract_json("[1, 2, 3]") == {}


def test_extract_json_top_level_scalar_is_not_a_dict():
    assert extract_json("42") == {}


# --- _scrub -----------------------------------------------------------------------


def test_scrub_removes_the_key_from_an_error_message():
    assert _scrub("401 for key sk-secret123", "sk-secret123") == "401 for key ***"


def test_scrub_is_a_noop_without_a_key():
    assert _scrub("some error", None) == "some error"


def test_extract_json_with_surrounding_prose():
    text = 'Sure, here you go:\n\n{"findings": []}\n\nLet me know if you need more.'
    assert extract_json(text) == {"findings": []}


def test_extract_json_garbage_returns_empty_dict():
    assert extract_json("not json at all") == {}


# --- normalize ------------------------------------------------------------------


def test_normalize_drops_findings_without_detail():
    raw = [{"severity": "high", "file": "a.py", "detail": ""}, {"severity": "high"}]
    assert normalize(raw, "review") == []


def test_normalize_clamps_bad_severity_and_line():
    raw = [{"severity": "catastrophic", "file": "a.py", "line": "not-a-number", "detail": "x"}]
    [f] = normalize(raw, "review")
    assert f.severity == "low"
    assert f.line is None
    assert f.pass_name == "review"


def test_normalize_fills_fallbacks():
    [f] = normalize([{"detail": "something is wrong"}], "security")
    assert f.file == "(unspecified)"
    assert f.title == "Unlabeled finding"
    assert f.suggestion == ""


def test_normalize_ignores_non_dict_entries():
    assert normalize(["oops", None, 3], "review") == []


# --- dedupe -----------------------------------------------------------------------


def _finding(**overrides) -> Finding:
    base = dict(
        severity="low", file="a.py", line=1, title="Some issue",
        detail="detail", suggestion="", pass_name="review",
    )
    base.update(overrides)
    return Finding(**base)


def test_dedupe_merges_same_file_and_line_keeping_more_severe():
    findings = [_finding(severity="low"), _finding(severity="high", title="different title")]
    [kept] = dedupe(findings)
    assert kept.severity == "high"


def test_dedupe_merges_same_file_and_title_slug():
    findings = [
        _finding(line=1, title="Off by one error"),
        _finding(line=2, title="off-by-one-error"),
    ]
    assert len(dedupe(findings)) == 1


def test_dedupe_keeps_distinct_findings():
    findings = [_finding(line=1), _finding(line=2, title="unrelated")]
    assert len(dedupe(findings)) == 2


def test_dedupe_does_not_merge_two_untitled_findings_in_the_same_file():
    # normalize() gives every titleless finding the same fallback title -- that must
    # not make the title-slug merge treat them as the same issue.
    findings = [
        _finding(line=None, title="Unlabeled finding", detail="issue A"),
        _finding(line=None, title="Unlabeled finding", detail="issue B"),
    ]
    assert len(dedupe(findings)) == 2


# --- config precedence -----------------------------------------------------------


def test_load_config_defaults(tmp_path):
    cfg = load_config(root=tmp_path)
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.review_pass is True
    assert cfg.security_pass is True
    assert cfg.fail_on is None


def test_load_config_file_overrides_defaults(tmp_path):
    (tmp_path / ".medusa-review.json").write_text(
        json.dumps({"model": "deepseek-v4-pro", "passes": {"security": False}})
    )
    cfg = load_config(root=tmp_path)
    assert cfg.model == "deepseek-v4-pro"
    assert cfg.security_pass is False
    assert cfg.review_pass is True


def test_load_config_env_overrides_file(tmp_path, monkeypatch):
    (tmp_path / ".medusa-review.json").write_text(json.dumps({"model": "deepseek-v4-pro"}))
    monkeypatch.setenv("MEDUSA_REVIEW_MODEL", "qwen3-coder:30b")
    cfg = load_config(root=tmp_path)
    assert cfg.model == "qwen3-coder:30b"


def test_load_config_cli_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDUSA_REVIEW_MODEL", "qwen3-coder:30b")
    cfg = load_config({"model": "explicit-override"}, root=tmp_path)
    assert cfg.model == "explicit-override"


def test_load_config_review_only_override(tmp_path):
    cfg = load_config({"review_pass": True, "security_pass": False}, root=tmp_path)
    assert cfg.review_pass is True
    assert cfg.security_pass is False


def test_load_config_falsy_numeric_overrides_are_not_swallowed(tmp_path):
    # `pick(...) or default` would treat an explicit 0 as "unset" -- it must not.
    cfg = load_config({"retries": 0, "max_tokens": 0}, root=tmp_path)
    assert cfg.retries == 0
    assert cfg.max_tokens == 0


def test_load_config_casts_a_quoted_number_from_the_file(tmp_path):
    # json.loads would hand back a str for a quoted number; pick() must still cast
    # it to int rather than letting a str reach ReviewConfig.max_diff_bytes.
    (tmp_path / ".medusa-review.json").write_text(
        json.dumps({"max_diff_bytes": "1234", "retries": "3"})
    )
    cfg = load_config(root=tmp_path)
    assert cfg.max_diff_bytes == 1234
    assert cfg.retries == 3


def test_load_config_malformed_passes_value_falls_back_to_defaults(tmp_path):
    (tmp_path / ".medusa-review.json").write_text(json.dumps({"passes": True}))
    cfg = load_config(root=tmp_path)  # must not raise AttributeError on `passes.get`
    assert cfg.review_pass is True
    assert cfg.security_pass is True


def test_load_config_coerces_a_quoted_bool_in_passes(tmp_path):
    # A hand-edited .medusa-review.json with `"security": "false"` is a Python-truthy
    # string; it must still disable the pass, not silently enable it.
    (tmp_path / ".medusa-review.json").write_text(
        json.dumps({"passes": {"review": "true", "security": "false"}})
    )
    cfg = load_config(root=tmp_path)
    assert cfg.review_pass is True
    assert cfg.security_pass is False


# --- collect_diff -----------------------------------------------------------------


# Invoked from inside a `pre-push` hook, git sets GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE
# for the *outer* repo; inherited by any subprocess (this fixture's own `git` calls, and
# `collect_diff`'s internally), they'd redirect git away from the throwaway repo this
# fixture creates. Strip them for the whole test via monkeypatch, not just `_git`.
_GIT_REPO_ENV_KEYS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES",
    "GIT_PREFIX",
)


def _git(*args: str, cwd) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key in _GIT_REPO_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "T", cwd=root)
    (root / "a.txt").write_text("one\n")
    _git("add", "a.txt", cwd=root)
    _git("commit", "-q", "-m", "base", cwd=root)
    return root


def test_collect_diff_none_when_head_equals_base(repo, monkeypatch):
    monkeypatch.chdir(repo)
    result = collect_diff(ReviewConfig(base_ref="HEAD", head_ref="HEAD"))
    assert result is None


def test_collect_diff_detects_changed_file(repo, monkeypatch):
    (repo / "a.txt").write_text("one\ntwo\n")
    _git("commit", "-aqm", "add a line", cwd=repo)
    monkeypatch.chdir(repo)
    result = collect_diff(ReviewConfig(base_ref="HEAD~1", head_ref="HEAD"))
    assert result is not None
    assert result.changed_files == ["a.txt"]
    assert not result.truncated
    assert "+two" in result.diff


def test_collect_diff_truncates_on_a_line_boundary(repo, monkeypatch):
    (repo / "a.txt").write_text("\n".join(f"line{i}" for i in range(200)) + "\n")
    _git("commit", "-aqm", "grow the file", cwd=repo)
    monkeypatch.chdir(repo)
    cfg = ReviewConfig(base_ref="HEAD~1", head_ref="HEAD")
    full = collect_diff(cfg)
    truncated = collect_diff(dataclasses.replace(cfg, max_diff_bytes=200))

    assert truncated is not None
    assert truncated.truncated
    assert len(truncated.diff.encode()) <= 200
    assert truncated.diff.endswith("\n")
    # every line kept is a complete line from the untruncated diff -- none sliced mid-way
    full_lines = set(full.diff.splitlines())
    assert all(line in full_lines for line in truncated.diff.splitlines())


def test_collect_diff_raises_clear_error_on_bad_ref(repo, monkeypatch):
    monkeypatch.chdir(repo)
    with pytest.raises(RuntimeError, match="git diff"):
        collect_diff(ReviewConfig(base_ref="not-a-real-ref", head_ref="HEAD"))


def test_collect_diff_auto_detects_base_via_fallback_list(repo, monkeypatch):
    # No origin remote in this repo, so `_default_base()` falls through
    # `_FALLBACK_BASES` to a local "master" branch.
    _git("branch", "master", cwd=repo)
    (repo / "a.txt").write_text("one\ntwo\n")
    _git("commit", "-aqm", "add a line", cwd=repo)
    monkeypatch.chdir(repo)
    result = collect_diff(ReviewConfig())  # base_ref unset
    assert result is not None
    assert result.base_ref == "master"


def test_collect_diff_wraps_missing_git_binary_during_auto_detect(repo, monkeypatch):
    monkeypatch.chdir(repo)

    def _boom(*args, **kwargs):
        raise FileNotFoundError("git: command not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    # base_ref unset -> collect_diff must route the _default_base() call through the
    # same try/except as everything else, not raise a bare FileNotFoundError.
    with pytest.raises(RuntimeError, match="could not run git"):
        collect_diff(ReviewConfig())


def test_collect_diff_tiny_budget_yields_none_not_a_mid_line_slice(repo, monkeypatch):
    # A budget smaller than even the first line has no complete line to send the
    # model -- that's "nothing reviewable," same as an empty diff, not a mid-line
    # fragment (which would look like noise rather than a diff) and not a result
    # that exceeds the budget by appending a trailing newline.
    (repo / "a.txt").write_text("a-long-first-line-with-no-early-break\nsecond\n")
    _git("commit", "-aqm", "grow", cwd=repo)
    monkeypatch.chdir(repo)
    cfg = ReviewConfig(base_ref="HEAD~1", head_ref="HEAD", max_diff_bytes=5)
    assert collect_diff(cfg) is None


# --- cli.main -----------------------------------------------------------------

_DIFF = DiffResult(
    diff="+ x", changed_files=["a.py"], base_ref="origin/main", head_ref="HEAD", truncated=False
)


def _finding(**overrides) -> Finding:
    base = dict(
        severity="low", file="a.py", line=1, title="Some issue",
        detail="detail", suggestion="", pass_name="review",
    )
    base.update(overrides)
    return Finding(**base)


def test_main_skips_when_no_reviewable_changes(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_diff", lambda cfg: None)
    assert cli.main([]) == 0
    assert "no reviewable changes" in capsys.readouterr().out


def test_main_skips_without_an_api_key(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_diff", lambda cfg: _DIFF)
    monkeypatch.setattr(cli, "api_key", lambda: None)
    assert cli.main([]) == 0
    assert "no API key set" in capsys.readouterr().out


def test_main_fail_on_gates_the_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "collect_diff", lambda cfg: _DIFF)
    monkeypatch.setattr(cli, "api_key", lambda: "k")
    monkeypatch.setattr(cli, "run_pass", lambda name, diff, cfg: [_finding(severity="high")])

    assert cli.main(["--fail-on", "high"]) == 1
    assert cli.main([]) == 0  # same findings, but no --fail-on -> still exits 0


def test_main_handles_run_pass_failure_without_crashing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_diff", lambda cfg: _DIFF)
    monkeypatch.setattr(cli, "api_key", lambda: "k")

    def _boom(name, diff, cfg):
        raise RuntimeError("model call failed")

    monkeypatch.setattr(cli, "run_pass", _boom)
    assert cli.main([]) == 0
    assert "pass failed" in capsys.readouterr().out


def test_main_one_pass_failing_does_not_skip_the_other(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_diff", lambda cfg: _DIFF)
    monkeypatch.setattr(cli, "api_key", lambda: "k")
    calls: list[str] = []

    def _fake(name, diff, cfg):
        calls.append(name)
        if name == "review":
            raise RuntimeError("boom")
        return [_finding(pass_name="security")]

    monkeypatch.setattr(cli, "run_pass", _fake)
    assert cli.main([]) == 0
    assert calls == ["review", "security"]  # security still ran despite review's failure
    assert "review pass failed" in capsys.readouterr().out


def test_main_appends_truncation_notice_for_the_model(monkeypatch):
    truncated_diff = dataclasses.replace(_DIFF, truncated=True)
    monkeypatch.setattr(cli, "collect_diff", lambda cfg: truncated_diff)
    monkeypatch.setattr(cli, "api_key", lambda: "k")

    seen: list[str] = []

    def _capture(name, diff, cfg):
        seen.append(diff)
        return []

    monkeypatch.setattr(cli, "run_pass", _capture)
    cli.main([])
    assert seen and "truncated at" in seen[0]
