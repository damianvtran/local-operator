"""Replace real viewers while a real socket owner retains its turn and gates."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.reexec import REEXEC_CODE, take_plan
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tools.builtin import build_write_tool
from local_operator.tui.app import OperatorApp
from local_operator.update import MobileRefresh, VersionCheck
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    seed_transcript,
    text_turn,
    tool_call_turn,
    transcript_text,
    user_message,
    wait_for_adoption,
)
from tests.e2e.test_fork_e2e import _capture, _never_take_over, _pump
from tests.e2e.watchdog import bounded


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/reload", "/update"])
@pytest.mark.parametrize("approval", [False, True, "answered"])
async def test_relaunch_preserves_owner_turn_and_gate(
    headless_tui_env: Path, workspace: Path, monkeypatch, command: str, approval: bool | str
) -> None:
    config = headless_tui_env
    monkeypatch.setattr(OwnedSessionHandle, "_maybe_name_conversation", lambda *_: None)
    monkeypatch.setattr(
        "local_operator.update.check_latest",
        lambda **_: VersionCheck(installed="0.1.0", latest="0.2.0", behind=True),
    )
    monkeypatch.setattr("local_operator.update.perform_upgrade", lambda **_: "0.2.0")
    monkeypatch.setattr(
        "local_operator.update.refresh_mobile_after_upgrade",
        lambda **_: MobileRefresh(kind="skipped"),
    )
    directory = config / "sessions" / "reloadowner01"
    await seed_transcript(directory, [user_message("Saved original context")])
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled = []
    writes = []
    target = workspace / "result.txt"
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
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Writing while the terminal reloads",
                tool_name="write",
                tool_call_id="reload-write",
                arguments={"path": str(target), "content": "completed once"},
            ),
            text_turn("The original turn completed"),
        ]
    )
    owner = build_session(directory, stream, tools=[tool], cwd=workspace)
    owner._yolo = not approval
    handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(workspace))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    viewers = []

    async def factory():
        viewer = await RemoteSession.connect(
            server._record,
            owner.session_id,
            config_dir=config,
            takeover_factory=_never_take_over,
        )
        viewers.append(viewer)
        return viewer

    pending = ()
    try:
        with bounded(90, "reload preserves owner and unanswered approval"):
            app = OperatorApp(factory)
            async with app.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(app, pilot)
                app._set_approve_all(not approval)
                await app._session.prompt("Continue original work")
                if approval:
                    await _pump(pilot, lambda: app._approval is not None)
                    pending = tuple(handle._pending_futures.values())
                    assert len(pending) == 1
                    if approval == "answered":
                        app._answer_live_approval_as_allowed()
                        await entered.wait()
                        pending = ()
                else:
                    await entered.wait()
                pid = viewers[0]._runtime_pid
                await _capture(app, pilot, f"reload-before-{command[1:]}-{approval}")
                app._run_slash_command(command)
                await _pump(pilot, lambda: app.return_code == REEXEC_CODE)
                assert app._restart_plan.resume_id == owner.session_id
            assert not owner._disposed
            assert not cancelled
            assert all(not future.done() for future in pending)
            assert not target.exists()
            replacement = OperatorApp(factory)
            async with replacement.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(replacement, pilot)
                replacement._set_approve_all(not approval)
                assert viewers[-1]._runtime_pid == pid
                assert replacement._session.session_id == owner.session_id
                assert "Saved original context" in transcript_text(replacement)
                if approval is True:
                    await _pump(pilot, lambda: replacement._approval is not None)
                await _capture(replacement, pilot, f"reload-after-{command[1:]}-{approval}")
                if approval is True:
                    replacement._answer_live_approval_as_allowed()
                    await entered.wait()
                release.set()
                await _pump(pilot, lambda: not owner.is_streaming and bool(writes))
                await _pump(
                    pilot,
                    lambda: "The original turn completed" in transcript_text(replacement),
                )
                assert writes == ["completed once"]
                assert not cancelled
                await _capture(replacement, pilot, f"reload-settled-{command[1:]}-{approval}")
                print(
                    {
                        "command": command,
                        "approval": approval,
                        "owner_pid": pid,
                        "writes": writes,
                        "cancelled": cancelled,
                        "pending_survived": len(pending),
                    }
                )
    finally:
        release.set()
        take_plan()
        for viewer in viewers:
            await viewer.dispose()
        await server.aclose()
        await owner.dispose()
