"""The AI review tool's pure logic: JSON extraction, finding normalization/dedup,
and config precedence. No network calls."""

from __future__ import annotations

import json

from medusa.review.config import load_config
from medusa.review.passes import dedupe, normalize
from medusa.review.provider import extract_json
from medusa.review.types import Finding


# --- extract_json -------------------------------------------------------------


def test_extract_json_plain():
    assert extract_json('{"findings": []}') == {"findings": []}


def test_extract_json_fenced():
    text = '```json\n{"findings": [{"severity": "low"}]}\n```'
    assert extract_json(text) == {"findings": [{"severity": "low"}]}


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
