"""User guardrail rules: parse, skip-on-invalid, and the shipped default."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from local_operator.guardrails.rules import DEFAULTS_DIR, load_rules, parse_rule


def _rule(body: str) -> str:
    return f"---\n{body}\n---\nGuidance body.\n"


def test_block_review_warn_parse() -> None:
    for action in ("block", "review", "warn"):
        rule = parse_rule(
            _rule(f"name: r\nscope: tool:bash\naction: {action}\ncondition: ['git push']"),
            source="r.md",
        )
        assert rule is not None and rule.action == action and rule.body == "Guidance body."
        assert rule.matches("bash", {"command": "git push origin"})
        assert not rule.matches("bash", {"command": "git status"})
        assert not rule.matches("write", {"command": "git push"})


def test_wildcard_scope_matches_any_tool_arguments() -> None:
    rule = parse_rule(
        _rule("name: r\nscope: 'tool:*'\naction: block\ncondition: ['secrets/']"), source="r.md"
    )
    assert rule is not None and rule.matches("write", {"path": "~/.local-operator/secrets/x"})


def test_invalid_frontmatter_skipped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_rule("no frontmatter", source="bad.md") is None
        assert parse_rule(_rule("name: r\nscope: tool:bash\naction: nuke\ncondition: [x]"), source="a.md") is None
        assert parse_rule(_rule("name: r\nscope: text\naction: block\ncondition: [x]"), source="s.md") is None
    assert "bad.md" in caplog.text


def test_bad_regex_skipped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        rule = parse_rule(
            _rule("name: r\nscope: tool:bash\naction: block\ncondition: ['(unclosed', 'ok']"),
            source="r.md",
        )
    assert rule is not None and len(rule.conditions) == 1
    assert "(unclosed" in caplog.text
    with caplog.at_level(logging.WARNING):
        assert parse_rule(
            _rule("name: r\nscope: tool:bash\naction: block\ncondition: ['(unclosed']"), source="r.md"
        ) is None


def test_user_file_cannot_claim_builtin_resolver() -> None:
    text = _rule("name: x\nscope: tool:bash\naction: warn\nresolver: destructive-paths")
    assert parse_rule(text, source="user.md") is None
    assert parse_rule(text, source="shipped.md", shipped=True) is not None


def test_shipped_default_is_dangerous_delete(tmp_path: Path) -> None:
    rules = load_rules(tmp_path / "absent")
    names = [r.name for r in rules.builtin]
    assert names == ["dangerous-delete"] and rules.user == []
    assert (DEFAULTS_DIR / "dangerous-delete.md").is_file()


def test_user_rules_loaded_from_dir(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(_rule("name: a\nscope: tool:bash\naction: warn\ncondition: [x]"))
    (tmp_path / "broken.md").write_text("junk")
    rules = load_rules(tmp_path)
    assert [r.name for r in rules.user] == ["a"]
