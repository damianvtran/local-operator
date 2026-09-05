"""Fork a real owner through its socket and navigate one real OperatorApp.

Only the model is scripted. The transcript, tool execution, gate, wire protocol,
viewer, and navigation are production objects. Never open native terminals from
this matrix: even an isolated HOME inherits the developer's terminal markers.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tools.builtin import build_write_tool
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    text_turn,
    tool_call_turn,
    transcript_text,
    user_message,
    wait_for_adoption,
)
from tests.e2e.watchdog import bounded


async def _capture(app, pilot, name: str) -> None:
    destination = os.environ.get("LO_FORK_EVIDENCE_DIR")
    if not destination:
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for frame in range(2):
        await pilot.pause()
        from scripts.visual_capture import save_capture

        save_capture(app, root / f"{name}-{frame}.svg")
        screen = app.screen
        print(
            json.dumps(
                {
                    "capture": name,
                    "frame": frame,
                    "size": tuple(screen.size),
                    "virtual_size": tuple(screen.virtual_size),
                    "scrollbar": screen.show_vertical_scrollbar,
                    "editor": tuple(app._editor().size),
                }
            )
        )
        assert screen.virtual_size == screen.size
        assert not screen.show_vertical_scrollbar


async def _never_take_over() -> Any:
    raise AssertionError("a fork viewer must not take over its original")


async def _pump(pilot, predicate) -> None:
    # The outer C-thread watchdog bounds a true deadlock; loop turns, not a
    # timing assertion, drive ordinary asynchronous publication under CI load.
    for _ in range(1000):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("fork condition was not published")


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("size", [(100, 30), (60, 24)])
async def test_fork_keeps_original_tool_and_gate_then_returns_without_restart(
    headless_tui_env: Path, workspace: Path, monkeypatch, approval: bool, size: tuple[int, int]
) -> None:
    from local_operator.spawn import registry

    def forbid_window(**kwargs):
        raise AssertionError("default switch must never open a native terminal")

    monkeypatch.setattr(registry, "active_backend", forbid_window)
    # HOME isolation does not remove inherited native terminal targets. Exercise
    # that environment deliberately, but intercept the subprocess boundary so a
    # regression fails here instead of pinning a developer's real sidebar title.
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("CMUX_SURFACE_ID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr("local_operator.multiplexer.cmux._cmux_binary", lambda: "/bin/cmux")
    native_calls = []
    run = subprocess.run

    def intercept_native(argv, *args, **kwargs):
        if isinstance(argv, list) and Path(str(argv[0])).name == "cmux":
            native_calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")
        return run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", intercept_native)
    # Naming is a separate provider call and would consume this finite model
    # script before the actual tool turn. It is not part of fork admission.
    monkeypatch.setattr(OwnedSessionHandle, "_maybe_name_conversation", lambda *_: None)
    config = headless_tui_env
    (config / "config.yml").write_text("values:\n  runtime:\n    background_on_resume: false\n")
    parent_dir = config / "sessions" / "forkparent01"
    await seed_transcript(
        parent_dir, [user_message("original question"), assistant_message("saved answer")]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    cancelled: list[bool] = []
    writes: list[str] = []
    target = workspace / "original-result.txt"
    tool = build_write_tool()
    execute = tool.execute

    async def held_write(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
            result = await execute(*args, **kwargs)
            writes.append(target.read_text())
            return result
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    tool.execute = held_write
    parent_stream = ScriptedStream(
        [
            tool_call_turn(
                text="working on original",
                tool_name="write",
                tool_call_id="original-write",
                arguments={"path": str(target), "content": "original completed"},
            ),
            text_turn("original tool completed"),
            text_turn("original scheduled wake completed"),
        ]
    )
    owner = build_session(parent_dir, parent_stream, tools=[tool], cwd=workspace)
    owner._yolo = not approval
    from local_operator.harness.wake import build_wake_schedule

    scheduled = build_wake_schedule(
        {"message": "original scheduled follow-up", "in": "1h"}, [], int(time.time() * 1000)
    )
    assert "schedule" in scheduled
    wake = scheduled["schedule"]
    await owner.wake_scheduler.update([wake])
    child_started, child_release = asyncio.Event(), asyncio.Event()

    async def child_stream(request, signal=None):
        child_started.set()
        await child_release.wait()
        for event in text_turn("child completed independently"):
            yield event

    child = build_session(config / "sessions" / "forkchild001", child_stream, cwd=workspace)

    async def child_run(job_id, signal, progress):
        await child.prompt("continue child work")
        return "child completed independently"

    child_job = owner.jobs.register("task", "original child", child_run, agent_id="forkchild001")
    await asyncio.wait_for(child_started.wait(), 20)
    job = owner.jobs.get(child_job)
    assert job is not None
    owner.subscribe(lambda event: finished.set() if event.type == "agent_end" else None)
    handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(workspace))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    runtimes = {owner.session_id: (owner, handle, server)}
    branch_stream = ScriptedStream([text_turn("fork opened exactly once")])
    viewers = []

    async def resume(sid):
        if sid not in runtimes:
            branch = build_session(config / "sessions" / sid, branch_stream, cwd=workspace)
            branch_handle = OwnedSessionHandle(
                branch, asyncio.get_running_loop(), cwd=str(workspace)
            )
            branch_server = RuntimeServer(branch_handle, kind="daemon")
            await branch_server.start_in_process()
            runtimes[sid] = branch, branch_handle, branch_server
        current, _, runtime = runtimes[sid]
        viewer = await RemoteSession.connect(
            runtime._record, sid, config_dir=config, takeover_factory=_never_take_over
        )
        viewers.append(viewer)
        return viewer

    async def factory():
        return await resume(owner.session_id)

    app = OperatorApp(factory, resume_factory=resume)
    try:
        with bounded(90, "fork preserves original tool and returns to the same owner"):
            async with app.run_test(size=size) as pilot:
                await wait_for_adoption(app, pilot)
                app._set_approve_all(not approval)
                assert app._session is not None
                from local_operator.tui.terminal_title import TerminalTitle, osc_title

                # Capture the production OSC writer without enabling the headless
                # driver's output. The same status band survives each adoption.
                title_wire: list[str] = []
                title = TerminalTitle(title_wire.append)
                assert app._status is not None
                app._status.set_terminal_title(title)
                title.start()
                pending: tuple[str, ...] = ()
                pending_id = ""
                await app._session.prompt("continue original")
                if approval:
                    await _pump(pilot, lambda: app._approval is not None)
                    pending = tuple(handle._pending_futures)
                    assert len(pending) == 1
                    pending_id = next(iter(handle._pending_futures))
                    assert not entered.is_set()
                else:
                    await asyncio.wait_for(entered.wait(), 20)
                await _capture(
                    app, pilot, f"original-{'approval' if approval else 'tool'}-{size[0]}"
                )
                app._cmd_fork("try another route", app._notice)
                await _pump(
                    pilot,
                    lambda: app._session is not None
                    and app._session.session_id != owner.session_id,
                )
                assert app._session is not None
                fork_id = app._session.session_id
                await _pump(pilot, lambda: len(branch_stream.requests) == 1)
                assert owner.is_streaming
                assert job.status == "running"
                assert not owner.wake_scheduler.disposed
                assert owner.wake_scheduler.schedules == (wake,)
                assert not runtimes[fork_id][0].wake_scheduler.schedules
                assert not runtimes[fork_id][0].jobs.list()
                assert not cancelled
                assert not target.exists()
                painted = " ".join(transcript_text(app).split())
                assert "Return to original: /resume forkparent01" in painted
                assert "Work still in progress was not copied" in painted
                await _capture(app, pilot, f"after-{'approval' if approval else 'tool'}-{size[0]}")
                assert "Original question" in title.current
                assert osc_title(title.current) in title_wire
                assert native_calls == [], "headless fork leaked a native cmux mutation"
                app._run_slash_command("/rename Divergent work")
                await _pump(
                    pilot,
                    lambda: runtimes[fork_id][0].conversation_name == "Divergent work"
                    and "Divergent work" in title.current,
                )
                assert "Divergent work" in title.current
                assert osc_title(title.current) in title_wire
                assert native_calls == [], "fork rename must use OSC, not a user override"
                assert (
                    "original-write"
                    not in (config / "sessions" / fork_id / "transcript.jsonl").read_text()
                )
                if approval:
                    assert tuple(handle._pending_futures) == pending
                    assert not handle._pending_futures[pending_id].done()
                app._resume_session(owner.session_id, app._notice)
                await _pump(
                    pilot,
                    lambda: app._session is not None
                    and app._session.session_id == owner.session_id,
                )
                assert runtimes[owner.session_id][0] is owner
                assert "Divergent work" not in title.current
                assert osc_title(title.current) in title_wire
                assert len(runtimes) == 2
                if approval:
                    await _pump(pilot, lambda: app._approval is not None)
                    await _capture(app, pilot, f"returned-approval-{size[0]}")
                    app._answer_live_approval_as_allowed()
                    await asyncio.wait_for(entered.wait(), 20)
                release.set()
                await asyncio.wait_for(finished.wait(), 20)
                assert not cancelled
                assert writes == ["original completed"]
                assert target.read_text() == "original completed"
                child_release.set()
                await asyncio.wait_for(owner.jobs.settled_event(child_job).wait(), 20)
                assert job.status == "completed"
                finished.clear()
                assert await owner.wake_scheduler.pump(wake.next_due_at) == 1
                await asyncio.wait_for(finished.wait(), 20)
                assert any(
                    isinstance(m, Message) and m.text == "original scheduled wake completed"
                    for m in owner.history()
                )
                print(
                    json.dumps(
                        {
                            "size": size,
                            "approval": approval,
                            "original_owner_reused": True,
                            "parent_cancelled": cancelled,
                            "writes": writes,
                            "fork_model_calls": len(branch_stream.requests),
                        }
                    )
                )
                app._resume_session(fork_id, app._notice)
                await _pump(
                    pilot, lambda: app._session is not None and app._session.session_id == fork_id
                )
                assert len(branch_stream.requests) == 1
                assert "Divergent work" in title.current
                assert osc_title(title.current) in title_wire
                assert native_calls == [], "resuming a fork must not pin a native title"
                await _capture(
                    app, pilot, f"renamed-{'approval' if approval else 'tool'}-{size[0]}"
                )
                print(
                    json.dumps(
                        {
                            "terminal_title": title.current,
                            "osc": title_wire[-1],
                            "native_calls": native_calls,
                        }
                    )
                )
                title.stop()
                assert (
                    sum(
                        m.text == "try another route"
                        for m in runtimes[fork_id][0].history()
                        if isinstance(m, Message) and m.role == "user"
                    )
                    == 1
                )
    finally:
        release.set()
        child_release.set()
        await child.dispose()
        for viewer in viewers:
            await viewer.dispose()
        for current, _, runtime in runtimes.values():
            await runtime.aclose()
            await current.dispose()


@pytest.mark.asyncio
async def test_owner_snapshot_rejects_foreign_or_path_selecting_requests(
    headless_tui_env: Path,
) -> None:
    from local_operator.mobile.attach_client import AttachClient

    directory = headless_tui_env / "sessions" / "forkscope001"
    await seed_transcript(directory, [user_message("owner only")])
    owner = build_session(directory, ScriptedStream([]))
    handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    viewer = await RemoteSession.connect(
        server._record,
        owner.session_id,
        config_dir=headless_tui_env,
        takeover_factory=_never_take_over,
    )
    try:
        assert isinstance(viewer._client, AttachClient)
        for payload in (
            {"message": 42},
            {"message": "", "parent_id": "someone-else"},
            {"message": "", "config_dir": "/tmp"},
        ):
            with pytest.raises(RuntimeError):
                await viewer._client._request_payload("fork_snapshot", **payload)
        with pytest.raises(ValueError, match="session's machine"):
            await server._dispatch_payload("fork_snapshot", {"message": ""}, locality="remote")
        assert len(list((headless_tui_env / "sessions").iterdir())) == 1
        response = await viewer.fork_snapshot()
        assert response["parent_id"] == owner.session_id
        assert (headless_tui_env / "sessions" / response["fork_id"] / "transcript.jsonl").exists()
    finally:
        await viewer.dispose()
        await server.aclose()
        await owner.dispose()
