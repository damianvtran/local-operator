"""Manual ``$name`` skill invocation: parsing, the money guard, rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.skills.discovery import Skill
from local_operator.skills.invoke import (
    invocation_name,
    parse_invocation,
    render_invocation,
)


def _resolved(text: str, skills: dict[str, Skill]):
    """Parse and assert a match, so a regression fails HERE and stays typed.

    ``parse_invocation`` is deliberately ``Optional`` — "this is prose" is half
    its job — so the positive-path tests would otherwise each need their own
    ``assert is not None`` before touching a field.
    """
    result = parse_invocation(text, skills)
    assert result is not None, f"expected {text!r} to resolve to a skill"
    return result


def _skill(name: str, *, hide: bool = False) -> Skill:
    return Skill(
        name=name,
        description=f"The {name} skill.",
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        source="/skills",
        hide=hide,
    )


@pytest.fixture
def skills() -> dict[str, Skill]:
    return {
        "research": _skill("research"),
        "code-review": _skill("code-review"),
        "secret_audit": _skill("secret_audit", hide=True),
    }


class TestParseInvocation:
    def test_bare_name_resolves_with_empty_request(self, skills):
        result = parse_invocation("$research", skills)
        assert result is not None
        assert result.skill.name == "research"
        assert result.request == ""
        assert result.token == "$research"

    def test_name_and_request_splits_on_the_token(self, skills):
        result = parse_invocation("$research fix the login bug", skills)
        assert result is not None
        assert result.skill.name == "research"
        assert result.request == "fix the login bug"

    def test_hyphenated_and_underscored_names_resolve(self, skills):
        assert _resolved("$code-review this MR", skills).skill.name == "code-review"
        assert _resolved("$secret_audit", skills).skill.name == "secret_audit"

    def test_hidden_skills_are_invocable(self, skills):
        """`hide` suppresses semantic ROUTING; naming it is the explicit case."""
        result = parse_invocation("$secret_audit check the repo", skills)
        assert result is not None
        assert result.skill.hide is True

    def test_case_insensitive_but_resolves_the_discovery_name(self, skills):
        result = parse_invocation("$Research look into this", skills)
        assert result is not None
        assert result.skill.name == "research"
        assert result.request == "look into this"

    def test_leading_whitespace_is_tolerated(self, skills):
        assert _resolved("  $research go", skills).skill.name == "research"


class TestNotAnInvocation:
    """The vocabulary is the guard: anything that is not a skill name is prose."""

    @pytest.mark.parametrize(
        "text",
        [
            "$100 for the redesign",
            "$5/unit at scale",
            "$PATH is not set",
            "$",
            "$ research",
            "$unknown-skill do the thing",
            "please run $research later",
            "cost is $research",
            "",
            "   ",
            "just a normal message",
            "/research",
        ],
    )
    def test_returns_none(self, skills, text):
        assert parse_invocation(text, skills) is None

    def test_money_amount_never_matches_even_with_numeric_skill(self, skills):
        """A numeric-named skill still cannot capture a currency amount."""
        assert parse_invocation("$100 for the redesign", skills) is None

    def test_trailing_punctuation_ends_the_token(self, skills):
        result = parse_invocation("$research, then report", skills)
        assert result is not None
        assert result.skill.name == "research"
        assert result.request == ", then report"

    def test_empty_vocabulary_matches_nothing(self):
        assert parse_invocation("$research go", {}) is None


class TestInvocationName:
    """The picker's question: is the caret in a `$` token, and what is typed."""

    def test_bare_sigil_opens_the_full_list(self):
        assert invocation_name("$") == ""

    def test_partial_name_is_the_query(self):
        assert invocation_name("$res") == "res"

    def test_unknown_name_still_queries(self):
        """Three characters in, nothing is valid yet — the list must stay up."""
        assert invocation_name("$zzz") == "zzz"

    def test_space_terminates_the_token(self):
        assert invocation_name("$research go") is None

    def test_trailing_space_closes_the_list(self):
        assert invocation_name("$research ") is None

    def test_newline_terminates(self):
        assert invocation_name("$research\nmore") is None

    @pytest.mark.parametrize("text", ["", "hello", "/team", "a $research"])
    def test_non_tokens(self, text):
        assert invocation_name(text) is None


class TestRenderInvocation:
    def test_body_and_request_both_reach_the_model(self, skills):
        invocation = _resolved("$research fix the login bug", skills)
        rendered = render_invocation(invocation, "# Research\nDo it well.")
        assert "Do it well." in rendered
        assert "fix the login bug" in rendered
        assert 'name="research"' in rendered
        assert "</skill>" in rendered

    def test_bare_invocation_carries_no_invented_request(self, skills):
        invocation = _resolved("$research", skills)
        rendered = render_invocation(invocation, "# Research\nDo it well.")
        assert rendered.rstrip().endswith("</skill>")

    def test_names_the_skill_in_the_imperative(self, skills):
        invocation = _resolved("$code-review this MR", skills)
        rendered = render_invocation(invocation, "body")
        assert "`code-review`" in rendered
