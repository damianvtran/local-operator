"""The rules the trimmed system prompt may never lose.

The 2026-10-08 context diet cut ``prompts_md/system.md`` roughly in half by
moving mechanics into the guides and ``tool://`` docs. What could NOT move is
the set of rules a model needs before it knows there is anything to look up:
a guide it has not read cannot stop it from installing a browser engine or
treating an ``ask`` receipt as consent. This file pins exactly that set, one
property per test, each against the arm of the template where it applies, so
a later trim fails here by name rather than as a behaviour regression in a
live session.

Phrases are matched on whitespace-normalised text: the source is hard-wrapped
and a rewrap must not read as a deletion.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.prompts_api import render_template

_FULL = {"has_browser": True, "no_browser": False, "has_console": True, "no_console": False}


def _flat(data: dict[str, Any]) -> str:
    return " ".join(render_template("system.md", data).split())


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> str:
    from local_operator.asks import policy

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    return _flat(_FULL)


def test_ask_is_the_last_resort(queued: str) -> None:
    assert "Deciding is your job; `ask` is the exception" in queued
    assert "`ask` is the only channel" in queued


def test_a_receipt_is_not_consent(queued: str) -> None:
    assert "Treat every `ask` as QUEUED" in queued
    assert "a receipt is never consent" in queued
    assert "run nothing the ask was meant to authorise" in queued


def test_destructive_actions_need_explicit_named_approval(queued: str) -> None:
    assert "require explicit user approval" in queued
    assert "never extends to a destructive or irreversible step by implication" in queued


def test_browser_cleanup_and_no_engine_install(queued: str) -> None:
    assert "action=close" in queued and "BEFORE the final response" in queued
    assert "Never close another session's tab" in queued
    assert "Never install or script a browser engine" in queued


def test_a_browserless_session_still_may_not_install_an_engine() -> None:
    text = _flat({"has_browser": False, "no_browser": True})
    assert "Never install or script a browser engine" in text
    assert "guide://browser" in text


def test_safety_rules_survive(queued: str) -> None:
    """Review round 1, F2: each of these deletions left every suite green."""
    assert "require explicit user approval before you run them" in queued
    assert "If an approval request is declined, stop that action and say so" in queued
    assert "Treat unknown files as the user's work" in queued
    assert "never print credentials, tokens or keys" in queued
    assert "Respect denials of a prompted write or command" in queued
    assert "never end a turn with pending items" in queued


def test_ask_triggers_stay_an_exhaustive_last_resort_list(queued: str) -> None:
    """F2(e): "only when:" is the brake. "whenever unsure, e.g. when:" keeps both
    headline sentences and inverts the rule, so the brake itself is pinned."""
    assert "Reach for `ask` only when: the action is destructive or irreversible" in queued


def test_a_consoleless_session_may_not_fake_a_terminal() -> None:
    text = _flat({"has_console": False, "no_console": True})
    assert "never script a terminal emulator or treat another window's terminal as this one" in text


def test_rules_restored_in_review_round_one(queued: str) -> None:
    """F5/N1/minor: rules the first trim dropped instead of moving."""
    assert "when the user asks to be reminded" in queued
    assert "arm one when the user asks you to watch, poll, or be told" in queued
    assert "end it with no reply, and don't notify" in queued
    assert "Never write lettered options into your reply; ask everything in one call" in queued
    assert "for an urgent one, delegate the question to a `task` subagent" in queued
    assert "before browser, generic API, or local-config discovery" in queued
    assert "never force-activate a tab or raise a window" in queued
    assert "NEVER find or inspect skills with bash" in queued
    assert "grepping code instead is how you end up editing a file nothing reads" in queued


def test_peer_messaging_never_shells_out(queued: str) -> None:
    assert "never shell out to `lop send`, cmux, or another multiplexer" in queued


def test_guides_skills_and_mcp_are_read_through_their_schemes(queued: str) -> None:
    assert "you MUST `read guide://<name>` BEFORE acting" in queued
    assert "`skill://<name>`" in queued and "`skill://<name>/<relpath>`" in queued
    assert "NEVER find or inspect skills with bash (`find`, `ls`), glob or grep" in queued
    assert "`mcp://<server>`" in queued and "`mcp://<server>/<tool>`" in queued
    assert "`tool://<name>`" in queued


@pytest.mark.parametrize(
    "guide",
    ["browser", "console", "sessions", "peer-messaging", "scratchpad", "mcp", "agents"],
)
def test_every_guide_the_moved_prose_now_lives_in_is_pointed_at(queued: str, guide: str) -> None:
    """A trim that moves detail into a guide must leave the pointer behind."""
    assert f"guide://{guide}" in queued
