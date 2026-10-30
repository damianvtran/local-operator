"""``expand_invocation``: the pure half both submitting surfaces share.

The TUI and the mobile relay each own their cwd (a process directory versus a
session record), but the RULE — which names fire, what a body becomes, what a
message sends when one body is unusable — is one function so the two can be
compared byte for byte. These cases pin that rule without a filesystem: the
caller hands in the reader, which is the only I/O the function is allowed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from local_operator.sigils import SKILL_BLOCK_OPEN, skill_block_spans
from local_operator.skills.discovery import Skill
from local_operator.skills.invoke import (
    SKILL_BODY_BUDGET_BYTES,
    ExpandedInvocation,
    SkillPayloadTooLarge,
    expand_invocation,
    render_skill_block,
)


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
        "secret-audit": _skill("secret-audit", hide=True),
    }


def reader(bodies: dict[str, str]) -> Callable[[str], str]:
    """A reader that fails the way the real resolver does for an unknown name."""

    def read(name: str) -> str:
        return bodies[name]

    return read


def test_one_match_appends_the_same_block_a_session_would_persist(skills) -> None:
    """The payload IS the user's text plus the frozen block, byte for byte."""
    expanded = expand_invocation(
        "$research fix the login bug",
        skills,
        reader({"research": "Read primary sources."}),
    )
    assert expanded.text == (
        "$research fix the login bug\n\n"
        + render_skill_block([("research", "Read primary sources.")])
    )
    assert expanded.attached == ("research",)
    assert expanded.skipped == ()
    # The user's own line stays the prefix, so replay strips only the block.
    assert expanded.text.startswith("$research fix the login bug")
    assert skill_block_spans(expanded.text)


def test_two_matches_keep_text_order_not_dict_order(skills) -> None:
    expanded = expand_invocation(
        "$code-review then $research please",
        skills,
        reader({"code-review": "Check the diff.", "research": "Read primary sources."}),
    )
    assert expanded.attached == ("code-review", "research")
    assert expanded.text.index('name="code-review"') < expanded.text.index('name="research"')


def test_a_hidden_skill_is_invocable_by_name() -> None:
    """``hide`` suppresses routing; typing the name is the case it leaves open."""
    skills = {"secret-audit": _skill("secret-audit", hide=True)}
    expanded = expand_invocation(
        "$secret-audit check the repo", skills, reader({"secret-audit": "Scan for secrets."})
    )
    assert expanded.attached == ("secret-audit",)


@pytest.mark.parametrize(
    "text", ["no invocation here", "$PATH is set", "costs$5 today", "$nope please"]
)
def test_prose_and_shell_variables_expand_to_themselves(skills, text: str) -> None:
    expanded = expand_invocation(text, skills, reader({}))
    assert expanded.text == text
    assert expanded.attached == ()
    assert expanded.skipped == ()


def test_an_empty_body_skips_that_skill_and_sends_the_text(skills) -> None:
    expanded = expand_invocation(
        "$research fix the login bug", skills, reader({"research": "---\nname: research\n---\n"})
    )
    assert expanded.text == "$research fix the login bug"
    assert expanded.attached == ()
    assert [(skip.name, skip.reason) for skip in expanded.skipped] == [("research", "empty")]


def test_a_raising_reader_skips_only_that_skill(skills) -> None:
    """One unreadable body must not cost the other one, or the message."""

    def read(name: str) -> str:
        if name == "research":
            raise OSError("SKILL.md is unreadable")
        return "Check the diff."

    expanded = expand_invocation("$research then $code-review", skills, read)
    assert expanded.attached == ("code-review",)
    assert [(skip.name, skip.reason) for skip in expanded.skipped] == [("research", "unreadable")]
    assert expanded.text.startswith("$research then $code-review\n\n")
    assert expanded.text.count(SKILL_BLOCK_OPEN) == 1


def test_every_body_skipped_means_the_text_is_untouched(skills) -> None:
    # An empty map makes the reader raise for both names, which is the
    # "unreadable" arm; the point here is that the text still goes.
    expanded = expand_invocation("$research and $code-review", skills, reader({}))
    assert expanded.text == "$research and $code-review"
    assert expanded.attached == ()
    assert [skip.name for skip in expanded.skipped] == ["research", "code-review"]


def test_a_rendered_block_is_not_attached_twice(skills) -> None:
    """A payload replayed through the expander must not double its bodies."""
    once = expand_invocation("$research go", skills, reader({"research": "Read primary sources."}))
    twice = expand_invocation(once.text, skills, reader({"research": "Read primary sources."}))
    assert twice.text == once.text
    assert twice.attached == ()


def test_the_budget_is_measured_over_the_bodies_and_refuses_the_message(skills) -> None:
    body = "x" * 64
    bodies = {"research": body, "code-review": body}
    expanded = expand_invocation("$research $code-review", skills, reader(bodies), body_budget=128)
    assert expanded.attached == ("research", "code-review")

    with pytest.raises(SkillPayloadTooLarge) as raised:
        expand_invocation("$research $code-review", skills, reader(bodies), body_budget=127)
    assert raised.value.size == 128
    assert raised.value.budget == 127


def test_the_default_budget_is_the_documented_bound() -> None:
    """384 KiB: a bound the relay's frame can carry, not a measurement."""
    assert SKILL_BODY_BUDGET_BYTES == 384 * 1024


def test_an_expansion_with_nothing_attached_is_still_a_receipt() -> None:
    """The caller reports skips; an untouched text is a value, not ``None``."""
    expanded = ExpandedInvocation(text="plain")
    assert expanded.attached == () and expanded.skipped == ()
