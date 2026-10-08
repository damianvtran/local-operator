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


def test_peer_messaging_never_shells_out(queued: str) -> None:
    assert "never shell out to `lop send`, cmux, or another multiplexer" in queued


def test_guides_skills_and_mcp_are_read_through_their_schemes(queued: str) -> None:
    assert "you MUST `read guide://<name>` BEFORE acting" in queued
    assert "`skill://<name>`" in queued and "`skill://<name>/<relpath>`" in queued
    assert "never locate skills with bash, glob or grep" in queued
    assert "`mcp://<server>`" in queued and "`mcp://<server>/<tool>`" in queued
    assert "`tool://<name>`" in queued


@pytest.mark.parametrize(
    "guide",
    ["browser", "console", "sessions", "peer-messaging", "scratchpad", "mcp", "agents"],
)
def test_every_guide_the_moved_prose_now_lives_in_is_pointed_at(queued: str, guide: str) -> None:
    """A trim that moves detail into a guide must leave the pointer behind."""
    assert f"guide://{guide}" in queued
