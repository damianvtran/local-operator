"""The Librarian's quiet-day banner veto.

The reported defect: the settle-time veto that turns a check-in's declared
``notify=True`` back into silence for the sentinel was Aida-only, so a quiet
day in the Library would raise a completion banner. These cells drive the real
``Session._finalize_attention_notify`` over a real Librarian session (state.json
names it, so ``_librarian_duty`` is resolved the way the runtime resolves it),
with the run's delivery facts set exactly as the wake pump sets them.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    TextContent,
)
from local_operator.librarian import bootstrap, proactive, state
from local_operator.resume import write_session_attachment
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream

MODEL = ModelSpec(provider="test", model_id="librarian-veto-model", context_window=100_000)
SESSION_ID = "feedc0de2222"


def _reply(text: str) -> ScriptedStream:
    return ScriptedStream([[StreamEndEvent(stop_reason="stop")]])


def _event(text: str | None) -> AgentEndEvent:
    messages = (
        cast("list[AgentMessage]", [Message(role="assistant", content=[TextContent(text=text)])])
        if text
        else []
    )
    return AgentEndEvent(messages=messages)


def _make_session(root: Path) -> Session:
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    write_session_attachment(directory, team="", agent="librarian", goal="")
    state.update_state(root, session_id=SESSION_ID)
    return Session(
        model=MODEL,
        model_source="config",
        stream_fn=_reply(proactive.QUIET_REPLY),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        cwd=str(root),
    )


def _set_checkin(session: Session, wake_id: str = proactive.CADENCE_ID) -> None:
    session._run_triggers = {"wake_prompt"}
    session._run_wake_ids = {wake_id}
    session._run_notify_requested = True


@pytest.mark.asyncio
async def test_the_quiet_sentinel_silences_her_banner(isolated_root: Path) -> None:
    assert await bootstrap.ensure_session(isolated_root) is not None
    session = _make_session(isolated_root)
    try:
        assert session._librarian_duty is True
        assert session._aida_duty is False
        _set_checkin(session)
        assert session._finalize_attention_notify(_event(proactive.QUIET_REPLY)) is False
        # The shapes a model actually writes around a sentinel.
        assert session._finalize_attention_notify(_event(f"  `{proactive.QUIET_REPLY}`  ")) is False
        # An empty reply is equally evidence of a quiet day.
        assert session._finalize_attention_notify(_event(None)) is False
        # A real pass keeps its banner.
        assert (
            session._finalize_attention_notify(_event("Renamed 4 assets and added 2 tags.")) is True
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_extra_pass_is_covered_and_user_intent_is_not(isolated_root: Path) -> None:
    assert await bootstrap.ensure_session(isolated_root) is not None
    session = _make_session(isolated_root)
    try:
        _set_checkin(session, wake_id="librarian-extra-1")
        assert session._finalize_attention_notify(_event(proactive.QUIET_REPLY)) is False

        # A user-armed wake keeps normal semantics: user intent is never
        # silenced by a daily-pass rule.
        _set_checkin(session, wake_id="w1")
        assert session._finalize_attention_notify(_event(proactive.QUIET_REPLY)) is True

        # A mixed run (a typed message beside the delivery) likewise.
        _set_checkin(session)
        session._run_triggers = {"wake_prompt", "user"}
        assert session._finalize_attention_notify(_event(proactive.QUIET_REPLY)) is True
    finally:
        await session.dispose()
