"""The community read flag through a REAL transcript (RFC v2.4 §5.9, review R3-1).

"Once the chief has read community posts, every later turn in that session
falls back to Ask me each time — including after compaction … until a new
session starts. The harness records this as a flag in the session journal, so a
resumed session keeps it and a forked session inherits it."

Pinned against ``Transcript`` and the ``Session`` wiring (the bridge latch test's
harness, ``tests/unit/session/test_bridge_taint_latch.py``):

* a ``community_read`` result commit taints the session and writes ONE
  ``community_read_taint`` row;
* compaction appends a marker and never lifts it;
* resume (a new ``Transcript`` on the same directory) and fork (a copied
  ``transcript.jsonl``) reopen tainted — including when the tool result row's
  payload was blanked, because the custom row carries the fact on its own;
* a fresh directory (``/new``) starts clean;
* a private-team bridge result does NOT set this flag (the card must not claim
  the chief read community posts), and the flag reaches the per-turn context.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from local_operator.community.taint import (
    COMMUNITY_CONTENT_DETAILS_KEY,
    COMMUNITY_TAINT_ROW_TYPE,
    session_tainted,
    standing_grant_usable,
)
from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript


def _session(transcript: Transcript) -> Session:
    session = Session.__new__(Session)
    session._transcript = transcript
    session._init_bridge_taint(False)
    session._init_community_taint()
    return session


async def _drain(session: Session) -> None:
    for task in (session._community_taint_row_task, session._bridge_latch_row_task):
        if task is not None:
            await task


def _rows(directory: Path) -> int:
    return sum(
        1
        for line in (directory / "transcript.jsonl").read_text().splitlines()
        if f'"{COMMUNITY_TAINT_ROW_TYPE}"' in line
    )


def _read_result(details: dict) -> Message:
    return Message.tool_result(
        ToolResult(
            tool_call_id="c1",
            tool_name="bridge",
            content=[TextContent(text="[COMMUNITY CONTENT — untrusted] …")],
            details=details,
        )
    )


@pytest.mark.asyncio
async def test_read_taints_survives_compaction_resume_fork_and_new_is_clean(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "s1"
    transcript = await asyncio.to_thread(Transcript, directory)
    session = _session(transcript)
    await transcript.append_message(Message.user("summarize #help"))
    assert session.community_tainted is False
    await transcript.append_message(
        _read_result(
            {"ok": True, "state": "ok", "content": "…", COMMUNITY_CONTENT_DETAILS_KEY: True}
        )
    )
    await _drain(session)
    assert session.community_tainted is True
    await transcript.append_message(Message.user("thanks"))
    await _drain(session)
    assert _rows(directory) == 1, "the flag row is written once"

    first_kept = transcript.entries()[-1].id
    await transcript.append_compaction("summary that drops the posts", first_kept, 1000)
    assert _rows(directory) == 1
    assert _session(await asyncio.to_thread(Transcript, directory)).community_tainted is True

    # Fork whose copy lost the stamped tool row (a prune blanked it): the
    # custom row alone keeps the fork tainted.
    fork = tmp_path / "fork"
    fork.mkdir()
    kept = [
        line
        for line in (directory / "transcript.jsonl").read_text().splitlines()
        if COMMUNITY_CONTENT_DETAILS_KEY not in line or COMMUNITY_TAINT_ROW_TYPE in line
    ]
    (fork / "transcript.jsonl").write_text("\n".join(kept) + "\n")
    assert _session(await asyncio.to_thread(Transcript, fork)).community_tainted is True

    full_fork = tmp_path / "fork2"
    full_fork.mkdir()
    shutil.copy(directory / "transcript.jsonl", full_fork / "transcript.jsonl")
    assert _session(await asyncio.to_thread(Transcript, full_fork)).community_tainted is True

    fresh = _session(await asyncio.to_thread(Transcript, tmp_path / "new"))
    assert fresh.community_tainted is False


@pytest.mark.asyncio
async def test_private_bridge_result_latches_the_bridge_but_not_the_community_flag(
    tmp_path: Path,
) -> None:
    transcript = await asyncio.to_thread(Transcript, tmp_path / "s2")
    session = _session(transcript)
    await transcript.append_message(_read_result({"ok": True, "state": "ok", "content": "{}"}))
    await _drain(session)
    assert session.bridge_latched is True
    assert session.community_tainted is False
    assert transcript.latest_custom_entry(COMMUNITY_TAINT_ROW_TYPE) is None


@pytest.mark.asyncio
async def test_flag_reaches_the_tool_context_and_fails_closed(tmp_path: Path) -> None:
    transcript = await asyncio.to_thread(Transcript, tmp_path / "s3")
    session = _session(transcript)
    holder = type("Ctx", (), {"community_taint": session._community_taint})()
    assert session_tainted(holder) is False and standing_grant_usable(holder) is True
    session._community_taint.mark()
    assert standing_grant_usable(holder) is False
    # No flag object at all (a bare host): nothing proves it clean.
    assert standing_grant_usable(object()) is False


@pytest.mark.asyncio
async def test_row_payload_is_content_free(tmp_path: Path) -> None:
    directory = tmp_path / "s4"
    transcript = await asyncio.to_thread(Transcript, directory)
    session = _session(transcript)
    await transcript.append_message(_read_result({"ok": True, COMMUNITY_CONTENT_DETAILS_KEY: True}))
    await _drain(session)
    row = transcript.latest_custom_entry(COMMUNITY_TAINT_ROW_TYPE)
    assert row is not None
    assert json.loads(json.dumps(row.payload.get("details"))) == {"version": 1, "inherited": False}


@pytest.mark.asyncio
async def test_real_fork_snapshot_inherits_the_flag_even_cut_before_the_read(
    tmp_path: Path,
) -> None:
    """Through ``Transcript.fork_snapshot`` itself (``/fork``), not a file copy.

    The custom row is copied wholesale (journal rows are not conversation rows),
    so a fork cut BEFORE the read is still tainted: fail-closed, matching the RFC
    ("a forked session inherits it") rather than reasoning about where the
    model's context begins.
    """
    parent = await asyncio.to_thread(Transcript, tmp_path / "sessions" / "parent000001")
    session = _session(parent)
    first = Message.user("summarize #help")
    await parent.append_message(first)
    # A paired call/result: fork refuses an orphaned tool result.
    call = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[ToolCall(id="c1", name="bridge", arguments={"op": "community_read"})],
    )
    await parent.append_messages(
        [call, _read_result({"ok": True, "state": "ok", COMMUNITY_CONTENT_DETAILS_KEY: True})]
    )
    await _drain(session)
    await parent.append_message(Message.user("thanks"))
    for cut in (None, first.id):
        fork_id, _, _ = await parent.fork_snapshot(through_entry_id=cut)
        child = await asyncio.to_thread(Transcript, tmp_path / "sessions" / fork_id)
        assert _session(child).community_tainted is True, cut
