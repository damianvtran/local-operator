"""``journal_incident`` must not restamp the session's activity clock.

The regression test for the reported defect. A user resumed a session, an
expired MCP OAuth grant journalled two boot-time ``session_incident`` entries,
and the ``/resume`` picker then displayed the session as worked on hours more
recently than it was. Session ``965426f4d60d`` showed ``3.06 h`` when work had
actually stopped ``8.14 h`` earlier — a 5.1 h lie written by an incident append
— and 19 of 509 rows in that store displayed an age wrong by more than a
minute, worst case 13.0 h (``FINDING-resume-clock.md``).

These tests drive the real ``retention.session_activity`` rather than a stat of
the file, so they fail if the clock's own definition ever stops agreeing with
what the writer preserves. The second half is as load-bearing as the first: the
fix must ignore ONE KIND of write, not freeze the clock.
"""

from __future__ import annotations

import os

import pytest

from local_operator.harness.types import StreamEndEvent
from local_operator.session.retention import session_activity
from local_operator.session.session import (
    SESSION_INCIDENT_MESSAGE_TYPE,
    CustomMessage,
    _default_convert_to_llm,
)

from .test_session import MODEL, ScriptedStream, make_session

#: Backdated far enough that no timestamp granularity can mask a restamp.
PAST = 1_700_000_000.0


async def _worked_in_session(tmp_path, turns: int = 1):
    """A session with real work in its transcript and its clock wound back.

    Built through ``make_session`` — the construction the rest of this suite
    uses — so the transcript is the one the product writes.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")] for _ in range(turns + 1)])
    session = make_session(tmp_path, stream, model=MODEL)
    for index in range(turns):
        await session.prompt(f"real work {index}")
    session_dir = tmp_path / "sess"
    os.utime(session_dir / "transcript.jsonl", (PAST, PAST))
    return session, session_dir


def _last_custom_type(session_dir) -> str:
    import json

    lines = (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])["payload"].get("custom_type", "")


@pytest.mark.asyncio
async def test_journalling_an_incident_leaves_session_activity_unchanged(tmp_path):
    """THE regression test. Fails on ``origin/main``."""
    session, session_dir = await _worked_in_session(tmp_path)
    before = session_activity(session_dir)
    assert before == pytest.approx(PAST, abs=1e-6)

    try:
        await session.journal_incident("provider returned 401 unauthorized")
    finally:
        await session.dispose()

    assert session_activity(session_dir) == pytest.approx(before, abs=1e-6)
    assert _last_custom_type(session_dir) == SESSION_INCIDENT_MESSAGE_TYPE


@pytest.mark.asyncio
async def test_a_user_turn_after_an_incident_still_advances_the_clock(tmp_path):
    """The fix ignores one kind of write; it does not stop the clock."""
    session, session_dir = await _worked_in_session(tmp_path)
    before = session_activity(session_dir)
    assert before is not None

    try:
        await session.journal_incident("provider returned 401 unauthorized")
        assert session_activity(session_dir) == pytest.approx(before, abs=1e-6)
        await session.prompt("back to work")
    finally:
        await session.dispose()

    after = session_activity(session_dir)
    assert after is not None and after > before


@pytest.mark.asyncio
async def test_two_boot_incidents_in_a_row_do_not_move_the_clock(tmp_path):
    """The reproduced case is TWO incidents — one expired ``linear`` grant and
    one expired ``notion`` grant journalled at the same boot. This is the exact
    shape of ``965426f4d60d``.
    """
    session, session_dir = await _worked_in_session(tmp_path)
    before = session_activity(session_dir)
    transcript_path = session_dir / "transcript.jsonl"
    lines_before = len(transcript_path.read_text(encoding="utf-8").splitlines())

    try:
        await session.journal_incident("MCP server 'linear': MCP authorization failed")
        await session.journal_incident("MCP server 'notion': MCP authorization failed")
    finally:
        await session.dispose()

    assert session_activity(session_dir) == pytest.approx(before, abs=1e-6)
    lines_after = len(transcript_path.read_text(encoding="utf-8").splitlines())
    assert lines_after == lines_before + 2


@pytest.mark.asyncio
async def test_the_incident_is_still_in_the_transcript_for_the_model(tmp_path):
    """Pins the refusal of the sidecar alternative.

    The incident is read back by the model on the next turn AND on resume
    replay (``_default_convert_to_llm`` renders it as an injected user
    message), by the mobile fold, and by compaction's kept window. Moving the
    entry out of ``transcript.jsonl`` to dodge the clock would silently stop
    the model learning why its last run failed; this test fails loudly if
    anyone later does that.
    """
    session, session_dir = await _worked_in_session(tmp_path)

    try:
        await session.journal_incident("provider returned 401 unauthorized")
    finally:
        await session.dispose()

    from local_operator.session.transcript import Transcript

    replayed = Transcript(session_dir).build_llm_history()
    assert any(
        isinstance(message, CustomMessage) and message.custom_type == SESSION_INCIDENT_MESSAGE_TYPE
        for message in replayed
    )

    converted = _default_convert_to_llm(replayed)
    assert any(
        message.role == "user" and "401 unauthorized" in _text_of(message) for message in converted
    )


def _text_of(message) -> str:
    return " ".join(getattr(part, "text", "") for part in getattr(message, "content", []) or [])
