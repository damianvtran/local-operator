"""The runner end to end: a real session, a real handle, a real journal row (memo §2.1/§2.9).

WHY THE HANDLE IS IN THE TEST AT ALL. The subscriber that feeds the runner is installed by the
runtime at boot (``RuntimeServer`` calls ``handle.subscribe`` before the first heartbeat), not
by the session. A test that called ``runner.on_agent_end`` directly would assert the wiring
into existence; this one drives ``session.prompt`` through the same subscription the daemon
uses and reads the row the job actually journaled.

The fake ``write`` tool is a real tool: the pre-filter keys on the tool NAME and its ``path``
argument, so the turn's evidence is produced the way production produces it (a tool call the
model made, its result persisted, the file on disk).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentTool,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE
from tests.unit.session.test_session import ScriptedStream, wait_for

pytestmark = pytest.mark.asyncio

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


def _write_tool(root: Path, calls: list[str]) -> AgentTool:
    async def execute(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        path = root / args["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("the report\n" + "x" * 200)
        calls.append(args["path"])
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="write",
            content=[TextContent(text=f"Wrote {args['path']}")],
        )

    return AgentTool(
        name="write",
        # A REAL schema: the loop validates the model's arguments against it, and an
        # undeclared ``path`` would be a planning failure whose synthetic result marks the
        # call errored -- which the pre-filter then (correctly) refuses to treat as evidence.
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        interruptible=True,
        execute=execute,
    )


def _report_stream(target: str) -> ScriptedStream:
    return ScriptedStream(
        [
            [
                StreamToolCallDelta(
                    index=0,
                    id="c1",
                    name="write",
                    argument_delta='{"path": "' + target + '"}',
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Wrote the report."), StreamEndEvent(stop_reason="stop")],
        ]
    )


def _make_session(directory: Path, stream: Any, tools: list[Any], cwd: Path) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools,
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        cwd=str(cwd),
    )


async def _await_row(transcript: Transcript) -> dict[str, Any]:
    await wait_for(lambda: transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE) is not None, timeout=10)
    row = transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE)
    assert row is not None
    return row


async def test_a_deliverable_turn_journals_a_done_row_under_the_answer(tmp_path: Path) -> None:
    calls: list[str] = []
    stream = _report_stream("reports/q3.md")
    session = _make_session(
        tmp_path / "sessions" / "s1",
        stream,
        [_write_tool(tmp_path, calls)],
        tmp_path,
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle.subscribe(lambda: None)
    try:
        await session.prompt("write me the Q3 report")
        row = await _await_row(session.transcript)
        assert calls == ["reports/q3.md"]
        assert row["state"] == "done"
        assert [f["path"] for f in row["files"]] == ["reports/q3.md"]
        assert row["files"][0]["name"] == "q3.md"
        assert row["files"][0]["why"] == "written by write"
        assert row["components"] == [] and row["decision"]["vendor"] == "heuristic"
        assert row["decision"]["skipped"] is None
        # The anchor is the final assistant message, and it is durable.
        assert session.transcript.has_entry(row["anchor"])
        # The row lands AFTER the answer and the attention marker, and the turn is settled.
        kinds = [
            (e.payload.get("custom_type") or e.payload.get("role"))
            for e in session.transcript.entries()
        ]
        assert session._turns_settled == 1
        assert kinds[-1] == SUPPLEMENT_CUSTOM_TYPE
        assert "attention_started" in [
            e.payload.get("custom_type") for e in session.transcript.entries()
        ]
    finally:
        await handle.dispose()
        await session.dispose()


async def test_the_attention_marker_and_the_answer_precede_the_row(tmp_path: Path) -> None:
    """Memo §2.1 ordering, asserted by EVENT SEQUENCE (never by clock): the end event the
    subscriber saw, the attention marker, then the row."""
    stream = _report_stream("reports/order.md")
    session = _make_session(
        tmp_path / "sessions" / "s2",
        stream,
        [_write_tool(tmp_path, [])],
        tmp_path,
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle.subscribe(lambda: None)
    sequence: list[str] = []
    from local_operator.harness.types import AgentEndEvent, MessageEndEvent

    def handler(event: Any) -> None:
        if isinstance(event, AgentEndEvent):
            sequence.append("agent_end")
        elif isinstance(event, MessageEndEvent) and event.message.role == "assistant":
            sequence.append("answer")

    session.subscribe(handler)
    try:
        await session.prompt("write the report")
        row = await _await_row(session.transcript)
        # The turn emits one assistant message per model call (the tool call, then the
        # answer); what the ordering claim needs is that NOTHING follows the end, and that
        # the answer preceded it.
        assert sequence[-1] == "agent_end" and sequence.count("agent_end") == 1, sequence
        assert sequence.index("answer") < sequence.index("agent_end"), sequence
        assert session._attention_run_settled, "the attention outcome must be published first"
        # ...and in the JOURNAL: the attention marker and the answer both precede the row.
        from local_operator.session.attention import ATTENTION_CUSTOM_TYPE

        journal = [
            (e.payload.get("custom_type"), e.payload.get("role"), e.id)
            for e in session.transcript.entries()
        ]
        row_index = next(
            i for i, (kind, _r, _id) in enumerate(journal) if kind == SUPPLEMENT_CUSTOM_TYPE
        )
        attention_index = max(
            i for i, (kind, _r, _id) in enumerate(journal) if kind == ATTENTION_CUSTOM_TYPE
        )
        anchor_index = next(i for i, (_k, role, eid) in enumerate(journal) if eid == row["anchor"])
        assert anchor_index < row_index and attention_index < row_index, journal
        assert row["anchor"] in {m.id for m in session.transcript.build_llm_history()}
    finally:
        await handle.dispose()
        await session.dispose()


async def test_a_running_job_never_reads_as_busy(tmp_path: Path) -> None:
    """The reaper and a build refresh may cut it: the job is NOT in ``_background_tasks``."""
    stream = _report_stream("reports/hold.md")
    session = _make_session(
        tmp_path / "sessions" / "s3",
        stream,
        [_write_tool(tmp_path, [])],
        tmp_path,
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle.subscribe(lambda: None)
    release = asyncio.Event()
    original = handle._supplements._write

    async def hold(anchor: str, decision: Any) -> None:
        await release.wait()
        await original(anchor, decision)

    handle._supplements._write = hold  # type: ignore[method-assign]
    try:
        await session.prompt("write the report")
        await wait_for(lambda: handle._supplements.running, timeout=10)
        assert handle.is_busy() is False, "a supplement job must not hold the session busy"
        assert handle._supplements._task not in getattr(session, "_background_tasks", set())
        release.set()
        await _await_row(session.transcript)
    finally:
        release.set()
        await handle.dispose()
        await session.dispose()


async def test_a_new_turn_supersedes_a_running_job(tmp_path: Path) -> None:
    stream = _report_stream("reports/first.md")
    session = _make_session(
        tmp_path / "sessions" / "s4",
        stream,
        [_write_tool(tmp_path, [])],
        tmp_path,
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle.subscribe(lambda: None)
    release = asyncio.Event()
    original = handle._supplements._write
    gate = {"first": True}

    async def hold(anchor: str, decision: Any) -> None:
        if gate["first"]:
            gate["first"] = False
            await release.wait()
        await original(anchor, decision)

    handle._supplements._write = hold  # type: ignore[method-assign]
    try:
        await session.prompt("write the first report")
        await wait_for(lambda: handle._supplements.running, timeout=10)
        first_task = handle._supplements._task
        # A second eligible turn cancels the first job. The session's own stream script has to
        # answer the second turn too.
        stream.turns.append(
            [StreamTextDelta(delta="Second answer."), StreamEndEvent(stop_reason="stop")]
        )
        await session.prompt("and now answer this instead")
        await wait_for(lambda: first_task is not None and first_task.cancelled(), timeout=10)
        release.set()
        await wait_for(
            lambda: handle._supplements._task is not None and handle._supplements._task.done(),
            timeout=10,
        )
    finally:
        release.set()
        await handle.dispose()
        await session.dispose()


async def test_the_kill_switch_stops_the_job_before_any_work(tmp_path: Path) -> None:
    from local_operator.supplements import policy

    stream = _report_stream("reports/off.md")
    session = _make_session(
        tmp_path / "sessions" / "s5",
        stream,
        [_write_tool(tmp_path, [])],
        tmp_path,
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle.subscribe(lambda: None)
    try:
        policy.SUPPLEMENTS = False
        await session.prompt("write me a report")
        # Nothing is scheduled on the off path (the refusal is synchronous in the event
        # fan-out, before ``prompt`` returns), so these are direct assertions rather than
        # a wait on the clock -- round-1 R7; the file's own doctrine is "wait on the
        # event, never on the clock", and there is no event here to wait on.
        assert handle._supplements._task is None
        assert session.transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE) is None
    finally:
        policy.SUPPLEMENTS = True
        await handle.dispose()
        await session.dispose()


async def test_the_settings_snapshot_is_read_at_build_not_per_turn(tmp_path: Path) -> None:
    """Round-1 R3: the runner reads ``values.supplements`` ONCE, at build (memo §2.12).

    The section's scope is ``NEW_SESSIONS``, and the scope claim is exactly this: an edit
    lands on the NEXT session. A per-job read made an edit land on the next eligible turn
    of an EXISTING session -- LIVE behaviour under a NEW_SESSIONS label. The turn below
    is the discriminator: the trigger still schedules its job, and under the old shape
    that job read the EDITED config and journalled a row.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.session.runtime.supplements import SupplementRunner

    config_dir_path = tmp_path / "cfg"
    config_dir_path.mkdir()
    manager = ConfigManager(config_dir_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["supplements.files"], False)
    calls: list[str] = []
    stream = _report_stream("reports/snap.md")
    session = _make_session(
        tmp_path / "sessions" / "snap", stream, [_write_tool(tmp_path, calls)], tmp_path
    )
    handle = ServingSessionHandle(
        session,
        asyncio.get_running_loop(),
        install_gates=False,
        cwd=str(tmp_path),
        config_dir=config_dir_path,
    )
    handle.subscribe(lambda: None)
    try:
        # The build-time snapshot saw ``files: false``: inactive, master on by default.
        snapshot = handle._supplements._settings
        assert snapshot.enabled is True and snapshot.files is False and not snapshot.active
        # Edit the config mid-session through the same facade the settings page uses...
        settings_io.write_setting(manager, settings_io.BY_KEY["supplements.files"], True)
        await session.prompt("write me the report")
        # ...the job runs on the SNAPSHOT and no-ops: no row, while the turn was real.
        await wait_for(
            lambda: handle._supplements._task is not None and handle._supplements._task.done(),
            timeout=10,
        )
        assert session.transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE) is None
        assert calls == ["reports/snap.md"]
        # A runner built AFTER the edit reads it: the edit lands on the next session.
        fresh = SupplementRunner(session, cwd=str(tmp_path), config_dir=config_dir_path)
        assert fresh._settings.files is True and fresh._settings.active
    finally:
        await handle.dispose()
        await session.dispose()


async def test_dispose_cancels_a_running_job(tmp_path: Path) -> None:
    stream = _report_stream("reports/dispose.md")
    session = _make_session(
        tmp_path / "sessions" / "s6",
        stream,
        [_write_tool(tmp_path, [])],
        tmp_path,
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle.subscribe(lambda: None)
    release = asyncio.Event()

    async def hold(anchor: str, decision: Any) -> None:
        await release.wait()

    handle._supplements._write = hold  # type: ignore[method-assign]
    try:
        await session.prompt("write the report")
        await wait_for(lambda: handle._supplements.running, timeout=10)
        task = handle._supplements._task
        await handle.dispose()
        assert task is not None and task.cancelled(), "dispose must reap the supplement job"
    finally:
        release.set()
        await session.dispose()
