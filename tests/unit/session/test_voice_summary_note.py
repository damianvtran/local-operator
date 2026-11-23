"""The spoken-summary turn note: who gets it, and that no row shows it.

Gate (docs/design/voice-spoken-summary.md, narrowed 2026-11-23): only a prompt
that was DICTATED, in the session that HOLDS the voice slot. A typed prompt, a
prompt in another session, a wake and a harness-injected turn get nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.rows import user_row_text
from local_operator.harness.types import StreamEndEvent
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.voice import slot as voice_slot
from local_operator.voice.summary_note import VOICE_SUMMARY_NOTE, note_spans, with_note
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _session(tmp_path: Path, *, holds_slot: bool) -> Any:
    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]] * 3))
    session._config_dir = tmp_path / "config"
    if holds_slot:
        voice_slot.enable(session._config_dir, session._session_id)
    return session


def _first_user_text(session: Any) -> str:
    return next(m.text for m in session._context.messages if m.role == "user")


@pytest.mark.asyncio
async def test_a_dictated_prompt_in_the_voice_session_carries_the_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt("what changed today", input_mode="dictated", input_path="qwen_s2s")
    text = _first_user_text(session)
    assert text == f"what changed today\n\n{VOICE_SUMMARY_NOTE}"
    # The row a person reads is exactly what they said.
    assert user_row_text(text) == "what changed today"
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "typed", "mixed"])
async def test_a_prompt_that_was_not_dictated_gets_no_note(tmp_path: Path, mode: Any) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt("typed words", input_mode=mode)
    assert _first_user_text(session) == "typed words"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_dictated_prompt_without_the_slot_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=False)
    await session.prompt("spoken but voice is off", input_mode="dictated")
    assert _first_user_text(session) == "spoken but voice is off"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_dictated_prompt_while_another_session_holds_the_slot_gets_no_note(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, holds_slot=False)
    voice_slot.enable(session._config_dir, "some-other-session")
    await session.prompt("not my slot", input_mode="dictated")
    assert _first_user_text(session) == "not my slot"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_harness_injected_dictated_turn_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    await session.prompt("chrome", input_mode="dictated", harness_injected=True)
    assert _first_user_text(session) == "chrome"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_wake_delivery_in_the_voice_session_gets_no_note(tmp_path: Path) -> None:
    session = _session(tmp_path, holds_slot=True)
    schedule = WakeSchedule(id="w1", message="check the build", next_due_at=0, created_at=0)
    await session._deliver_wake(
        DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    )
    await wait_for(lambda: any("check the build" in str(m) for m in session._context.messages))
    await session.dispose()
    assert not any(VOICE_SUMMARY_NOTE in str(m) for m in session._context.messages)


def test_the_note_asks_for_the_label_the_server_selects() -> None:
    from local_operator.voice import speakable

    assert "`Spoken:`" in VOICE_SUMMARY_NOTE
    # A reply that follows the note's instruction is exactly what the server selects.
    reply = "Long answer.\n\nSpoken: Done. Two things changed."
    assert speakable.select_spoken(reply) == "Done. Two things changed."


def test_a_row_hides_every_complete_note_and_only_those() -> None:
    body = "first\n\n" + VOICE_SUMMARY_NOTE
    assert user_row_text(body) == "first"
    assert user_row_text(with_note(with_note("twice"))) == "twice"
    # Quoting a fragment, or the label alone, is the user's own text: nothing is hidden.
    quoted = "what does `Spoken:` mean in (The user dictated this message by voice"
    assert user_row_text(quoted) == quoted
    assert note_spans("no note here") == []
