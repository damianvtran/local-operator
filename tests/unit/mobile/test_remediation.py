"""Regression tests for agent review round 1 (PR #188). Each test names the
finding it pins."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile import registry
from local_operator.mobile.daemon import MobileDaemon, SessionEntry
from local_operator.mobile.types import SessionRecord


def make_record(port: int) -> SessionRecord:
    import os

    return SessionRecord(
        pid=os.getpid(),  # alive, so scan classifies it live/wedged, never stale
        kind="tui",
        session_id="s1",
        conversation_name="demo",
        cwd="/tmp",
        model_label="test/model",
        control_port=port,
        control_key="k" * 64,
    )


@pytest.mark.asyncio
async def test_f1_degraded_sessions_are_redialed(tmp_path) -> None:  # noqa: ANN001
    """F1: a refused dial must not starve reconnects — degraded is 'we owe
    this session a redial', not a reason to skip it."""
    record = make_record(port=1)  # port 1 refuses immediately
    registry.publish(record, root=tmp_path)

    daemon = MobileDaemon(port=0, password="pw")
    # Point the registry at the tmp root for this scan.
    import local_operator.mobile.daemon as daemon_module

    original_scan = registry.scan
    daemon_module.registry.scan = lambda: original_scan(tmp_path)  # type: ignore[assignment]
    try:
        await daemon._scan_once()
        entry = daemon.table.entries[record.pid]
        # Let the first dial fail and mark the entry degraded.
        task = daemon._dial_tasks[record.pid]
        await task
        assert entry.degraded is True

        entry.next_dial_at = 0  # backoff elapsed
        await daemon._scan_once()
        assert record.pid in daemon._dial_tasks
        # The redial is a NEW attempt against the refused port.
        await daemon._dial_tasks[record.pid]
    finally:
        daemon_module.registry.scan = original_scan  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_f3_fanout_never_drops_a_repaint() -> None:
    """F3: a full subscriber queue evicts-then-retries until the put lands —
    a repaint (maybe the approval card) is never silently lost."""
    from local_operator.mobile.daemon import _fan_out
    from local_operator.mobile.types import SessionProjection

    record = make_record(port=1)
    entry = SessionEntry(record)
    entry.projection = SessionProjection(session_id="s1", pid=1)

    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait({"stale": True})  # exactly full at entry
    entry.subscribers.add(queue)

    _fan_out(entry)
    assert queue.qsize() == 1
    frame = queue.get_nowait()
    assert frame.get("session_id") == "s1"  # the NEW frame, not the stale one


def test_f9_password_newlines_rejected() -> None:
    """F9: a newline would split the security -i mini-shell command."""
    from local_operator.mobile.auth import store_password

    with pytest.raises(ValueError, match="newline"):
        store_password("good\nbad")


def test_f7_compaction_rows_tracked_by_id() -> None:
    """F7: the end event finalizes the row the START event opened, by id —
    never a reverse-scan guess that could land on a later compaction's row."""
    from local_operator.harness.types import CompactionEndEvent, CompactionStartEvent
    from local_operator.mobile.projection import ProjectionFold
    from local_operator.mobile.types import SessionProjection

    fold = ProjectionFold(SessionProjection(session_id="s1", pid=1))
    fold.fold_event(CompactionStartEvent(reason="manual"))
    first_id = fold._open_compaction_id
    assert first_id is not None
    fold.fold_event(CompactionStartEvent(reason="manual"))  # a second one opens
    fold.fold_event(
        CompactionEndEvent(reason="manual", success=True, tokens_before=100, tokens_after=50)
    )
    rows = [e for e in fold.projection.transcript if e.kind == "compaction"]
    # The SECOND row was finalized (the one the end event pairs with).
    assert rows[-1].final is True
    assert "100 → 50" in rows[-1].text
    assert rows[0].final is False  # the first stays open; its end never came


def test_f6_history_fold_prunes_dead_tool_bookkeeping() -> None:
    """F6: after the tail cut, correlation maps keep only surviving rows."""
    from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult
    from local_operator.mobile.projection import ProjectionFold
    from local_operator.mobile.types import (
        PROJECTION_TRANSCRIPT_LIMIT,
        SessionProjection,
    )

    fold = ProjectionFold(SessionProjection(session_id="s1", pid=1))
    history = []
    for i in range(PROJECTION_TRANSCRIPT_LIMIT + 30):
        call = ToolCall(id=f"c{i}", name="read", arguments={"path": f"/f{i}.py"})
        history.append(Message.assistant("", tool_calls=[call]))
        history.append(
            Message.tool_result(
                ToolResult(tool_call_id=f"c{i}", content=[TextContent(text="body")])
            )
        )
    fold.fold_history(history)
    assert len(fold.projection.transcript) == PROJECTION_TRANSCRIPT_LIMIT
    surviving_rows = {e.id for e in fold.projection.transcript}
    assert set(fold._tool_rows.values()) <= surviving_rows
    assert set(fold._tool_args) <= set(fold._tool_rows)
