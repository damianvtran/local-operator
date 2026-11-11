"""The Team Bridge latch row through a REAL transcript (memo §3.1, §3.5).

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
