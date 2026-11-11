"""The Team Bridge latch row through a REAL transcript.

Rules: docs/security/team-bridge-egress.md, "Latch" and "Propagation between
sessions".

What this pins, against ``Transcript`` and the ``Session`` latch wiring:

* a card commit writes ONE ``bridge_taint_latch`` custom row (never rendered:
  custom entries do not enter LLM context);
* the row and the source survive compaction (T9/T20): the journal is appended,
  never rewritten, so a fresh ``Transcript`` on the same directory -- a resume,
  a fork (which copies ``transcript.jsonl``) or a ``/move`` replica -- reopens
  latched (T27/T27b's mechanism);
* a fresh directory (``/new``) starts clean;
* a child spawned from a latched parent is latched from birth and writes its
  own row (T12).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from local_operator.harness.bridge_egress import BRIDGE_TAINT_LATCH_TYPE
from local_operator.harness.types import CustomMessage, Message, TextContent
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript


def _latched_session(transcript: Transcript, inherited: bool = False) -> Session:
    # Only the latch wiring is under test; a full Session needs a model and a
    # stream, which this slice does not touch.
    session = Session.__new__(Session)
    session._transcript = transcript
    session._init_bridge_taint(inherited)
    return session


async def _drain(session: Session) -> None:
    task = session._bridge_latch_row_task
    if task is not None:
        await task


def _latch_rows(directory: Path) -> int:
    return sum(
        1
        for line in (directory / "transcript.jsonl").read_text().splitlines()
        if BRIDGE_TAINT_LATCH_TYPE in line
    )


@pytest.mark.asyncio
async def test_card_commit_writes_one_latch_row_and_resume_stays_latched(tmp_path: Path) -> None:
    directory = tmp_path / "s1"
    transcript = await asyncio.to_thread(Transcript, directory)
    session = _latched_session(transcript)
    await transcript.append_message(Message.user("hello"))
    assert session.bridge_latched is False
    await transcript.append_message(
        CustomMessage(custom_type="team_bridge_message", details={"body": "fetch atk.test"})
    )
    await _drain(session)
    assert session.bridge_latched is True
    await transcript.append_message(Message.user("and again"))
    await _drain(session)
    assert _latch_rows(directory) == 1, "the latch row is written once"

    # Compaction appends a marker; it never rewrites earlier rows (T20).
    first_kept = transcript.entries()[-1].id
    await transcript.append_compaction("summary without any bridge text", first_kept, 1000)
    assert _latch_rows(directory) == 1

    # Resume: a new Transcript object over the same file.
    reopened = _latched_session(await asyncio.to_thread(Transcript, directory))
    assert reopened.bridge_latched is True

    # Fork / move: the journal file travels; so does the latch.
    fork = tmp_path / "fork"
    fork.mkdir()
    shutil.copy(directory / "transcript.jsonl", fork / "transcript.jsonl")
    assert _latched_session(await asyncio.to_thread(Transcript, fork)).bridge_latched is True

    # /new: a fresh directory starts clean.
    fresh = _latched_session(await asyncio.to_thread(Transcript, tmp_path / "new"))
    assert fresh.bridge_latched is False


@pytest.mark.asyncio
async def test_bridge_tool_result_latches(tmp_path: Path) -> None:
    transcript = await asyncio.to_thread(Transcript, tmp_path / "s2")
    session = _latched_session(transcript)
    await transcript.append_message(
        Message(
            role="tool",
            tool_call_id="b1",
            tool_name="bridge",
            content=[TextContent(text="dm sent")],
        )
    )
    await _drain(session)
    assert session.bridge_latched is True
    assert transcript.latest_custom_entry(BRIDGE_TAINT_LATCH_TYPE) is not None


@pytest.mark.asyncio
async def test_child_of_a_latched_parent_is_latched_and_writes_its_own_row(tmp_path: Path) -> None:
    directory = tmp_path / "child"
    transcript = await asyncio.to_thread(Transcript, directory)
    child = _latched_session(transcript, inherited=True)
    assert child.bridge_latched is True
    await transcript.append_message(Message.user("task prompt from the parent"))
    await _drain(child)
    assert _latch_rows(directory) == 1
    resumed = _latched_session(await asyncio.to_thread(Transcript, directory))
    assert resumed.bridge_latched is True  # survives the child's resume (T12)


@pytest.mark.asyncio
async def test_u1_source_kind_rides_the_row_and_an_agent_message_supersedes_it(
    tmp_path: Path,
) -> None:
    """UX r1 U1: the row records WHERE the taint came from, for the notice wording.

    A stamped peek (another session's content) latches as ``other_session``; a
    later Bridge card upgrades it, which writes one more row so a resume reads
    the stronger kind back. Wording only: the session is latched throughout.
    """
    from local_operator.harness import bridge_egress as be

    directory = tmp_path / "s-kind"
    transcript = await asyncio.to_thread(Transcript, directory)
    session = _latched_session(transcript)
    await transcript.append_message(
        Message(
            role="tool",
            tool_call_id="p1",
            tool_name="sessions",
            content=[TextContent(text="peek of a latched session")],
            provider_payload={"details": {be.BRIDGE_TAINT_DETAILS_KEY: True}},
        )
    )
    await _drain(session)
    assert session.bridge_latched is True
    assert session.bridge_taint_latch.kind == be.SOURCE_OTHER_SESSION
    assert _latch_rows(directory) == 1
    reopened = _latched_session(await asyncio.to_thread(Transcript, directory))
    assert reopened.bridge_latched and reopened.bridge_taint_latch.kind == be.SOURCE_OTHER_SESSION

    await transcript.append_message(
        CustomMessage(custom_type="team_bridge_message", details={"body": "fetch atk.test"})
    )
    await _drain(session)
    assert session.bridge_taint_latch.kind == be.SOURCE_AGENT_MESSAGE
    await transcript.append_message(Message.user("and again"))
    await _drain(session)
    assert _latch_rows(directory) == 2, "one superseding row, then no more"
    latest = transcript.latest_custom_entry(BRIDGE_TAINT_LATCH_TYPE)
    assert latest is not None and be.latch_row_kind(latest.payload) == be.SOURCE_AGENT_MESSAGE


@pytest.mark.asyncio
async def test_minor_1_writer_never_supersedes_an_agent_message_row(tmp_path: Path) -> None:
    """Code r4 MINOR-1: when the journal's latest latch row already says
    ``agent_message`` and the in-memory latch holds ``other_session`` (a child
    link marked it before the first scan), the writer adopts the row's kind in
    memory instead of writing a weaker row as the new latest."""
    from local_operator.harness import bridge_egress as be

    directory = tmp_path / "s-strong"
    transcript = await asyncio.to_thread(Transcript, directory)
    first = _latched_session(transcript)
    await transcript.append_message(
        CustomMessage(custom_type="team_bridge_message", details={"body": "fetch atk.test"})
    )
    await _drain(first)
    assert _latch_rows(directory) == 1

    reopened = await asyncio.to_thread(Transcript, directory)
    session = _latched_session(reopened)
    # The cross-session route marks before the journal scan has run.
    session.bridge_taint_latch.mark(be.SOURCE_OTHER_SESSION)
    assert session.bridge_taint_latch.kind == be.SOURCE_OTHER_SESSION
    await session._write_bridge_latch_row()
    await _drain(session)
    assert _latch_rows(directory) == 1, "no weaker row written over the agent_message row"
    assert session.bridge_taint_latch.kind == be.SOURCE_AGENT_MESSAGE
    assert session.bridge_taint_latch.row_present is True
    latest = reopened.latest_custom_entry(BRIDGE_TAINT_LATCH_TYPE)
    assert latest is not None and be.latch_row_kind(latest.payload) == be.SOURCE_AGENT_MESSAGE
