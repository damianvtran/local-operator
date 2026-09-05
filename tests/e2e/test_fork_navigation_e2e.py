"""A create belongs to the request, not the viewer or its departing ledger."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    seed_transcript,
    transcript_text,
    user_message,
    wait_for_adoption,
)
from tests.e2e.test_fork_e2e import _capture, _never_take_over, _pump
from tests.e2e.watchdog import bounded


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (60, 24)])
@pytest.mark.parametrize("timing", ["before", "after", "failed", "cancelled", "snapshot-error"])
async def test_snapshot_reply_survives_navigation(
    headless_tui_env: Path,
    workspace: Path,
    monkeypatch,
    size: tuple[int, int],
    timing: str,
) -> None:
    config = headless_tui_env
    (config / "config.yml").write_text("values:\n  runtime:\n    background_on_resume: false\n")
    parent_id, destination = "navparent001", "unrelated001"
    owners = {}
    servers = []
    viewers = []
    for sid, text in [(parent_id, "Original context"), (destination, "Unrelated conversation")]:
        directory = config / "sessions" / sid
        await seed_transcript(directory, [user_message(text)])
        owner = build_session(directory, ScriptedStream([]), cwd=workspace)
        handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(workspace))
        server = RuntimeServer(handle, kind="daemon")
        await server.start_in_process()
        owners[sid] = (owner, server)
        servers.append(server)
    entered_copy, release_copy = asyncio.Event(), asyncio.Event()
    entered_adoption, release_adoption = asyncio.Event(), asyncio.Event()
    original, source_server = owners[parent_id]
    snapshot = original.fork_snapshot
    calls = []

    async def held_snapshot(message=""):
        calls.append(message)
        entered_copy.set()
        await release_copy.wait()
        if timing == "snapshot-error":
            raise ValueError("synthetic snapshot refusal")
        return await snapshot(message)

    monkeypatch.setattr(original, "fork_snapshot", held_snapshot)

    async def connect(sid):
        server = owners[sid][1]
        viewer = await RemoteSession.connect(
            server._record,
            sid,
            config_dir=config,
            takeover_factory=_never_take_over,
        )
        viewers.append(viewer)
        return viewer

    async def factory():
        return await connect(parent_id)

    async def resume(sid):
        assert sid == destination, "a stale fork result must not hijack navigation"
        entered_adoption.set()
        await release_adoption.wait()
        if timing == "failed":
            raise ConnectionError("synthetic unrelated startup failure")
        if timing == "cancelled":
            raise asyncio.CancelledError
        return await connect(sid)

    app = OperatorApp(factory, resume_factory=resume)
    try:
        with bounded(60, "fork response lease and incoming ledger race"):
            async with app.run_test(size=size) as pilot:
                await wait_for_adoption(app, pilot)
                source = viewers[0]
                client = source._client
                assert client is not None
                app._cmd_fork("", app._notice)
                await entered_copy.wait()
                app._resume_session(destination, app._notice)
                await entered_adoption.wait()
                assert source._disposed
                assert client.connected
                assert source._snapshot_clients
                assert not original._disposed
                if timing == "after":
                    release_adoption.set()
                    await _pump(pilot, lambda: not app._session_transition_pending)
                release_copy.set()
                await _pump(pilot, lambda: not app._fork_in_progress)
                assert not client.connected
                assert not source._snapshot_clients
                if timing != "after":
                    assert app._pending_fork_outcome is not None
                    release_adoption.set()
                await _pump(pilot, lambda: not app._session_transition_pending)
                await pilot.pause()
                forks = [p for p in (config / "sessions").iterdir() if p.name not in owners]
                text = " ".join(transcript_text(app).split())
                assert len(calls) == 1
                assert not original._disposed
                if timing == "snapshot-error":
                    assert not forks
                    assert "fork failed: synthetic snapshot refusal" in text
                else:
                    assert len(forks) == 1
                    assert text.count("fork saved:") == 1
                    assert f"Open it with /resume {forks[0].name}" in text
                    assert "fork failed:" not in text
                if timing not in ("failed", "cancelled"):
                    assert app._session is not None
                    assert app._session.session_id == destination
                    assert "Unrelated conversation" in text
                assert app._pending_fork_outcome is None
                await _capture(app, pilot, f"navigation-{timing}-{size[0]}")
    finally:
        release_copy.set()
        release_adoption.set()
        for viewer in viewers:
            await viewer.dispose()
        for server in servers:
            await server.aclose()
        for owner, _ in owners.values():
            await owner.dispose()
