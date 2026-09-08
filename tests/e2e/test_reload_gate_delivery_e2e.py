"""Committed answers drain; unanswered successor questions survive relaunch."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.reexec import REEXEC_CODE, take_plan
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    seed_transcript,
    user_message,
    wait_for_adoption,
)
from tests.e2e.test_fork_e2e import _never_take_over, _pump
from tests.e2e.watchdog import bounded


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval", "ask"])
@pytest.mark.parametrize("timing", ["same-tick", "sending", "sidebar"])
async def test_committed_gate_drains_before_viewer_disposal(
    headless_tui_env: Path, workspace: Path, monkeypatch, kind: str, timing: str
) -> None:
    config = headless_tui_env
    directory = config / "sessions" / "replyowner01"
    await seed_transcript(directory, [user_message("Preserve my answer")])
    owner = build_session(directory, ScriptedStream([]), cwd=workspace)
    handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(workspace))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    viewer = await RemoteSession.connect(
        server._record, owner.session_id, config_dir=config, takeover_factory=_never_take_over
    )
    client = viewer._client
    assert client is not None
    send_entered, release_send, detach_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancelled = []
    attempts = []
    original = getattr(client, f"{kind}_answer")

    async def delayed_answer(*args, **kwargs):
        attempts.append(args)
        send_entered.set()
        try:
            await release_send.wait()
            return await original(*args, **kwargs)
        except asyncio.CancelledError:
            # Owner state may retire a delivered gate before the wire ACK comes
            # back. Cancellation BEFORE release is the data-loss window here.
            if not release_send.is_set():
                cancelled.append(True)
            raise

    monkeypatch.setattr(client, f"{kind}_answer", delayed_answer)
    detach = viewer.detach_viewer_gates

    async def observed_detach(**kwargs):
        detach_entered.set()
        await detach(**kwargs)

    monkeypatch.setattr(viewer, "detach_viewer_gates", observed_detach)

    async def factory():
        return viewer

    async def deliver_during_shutdown():
        await detach_entered.wait()
        await send_entered.wait()
        assert not cancelled
        release_send.set()

    answer_task = None
    delivery = None
    try:
        with bounded(60, "committed gate delivery through relaunch"):
            app = OperatorApp(factory)
            async with app.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(app, pilot)
                app._set_approve_all(False)
                if kind == "approval":
                    handle._auto_approve = False
                    answer_task = asyncio.create_task(
                        handle._approval_gate("write", "Save one record")
                    )
                    await _pump(pilot, lambda: app._approval is not None)
                    app._answer_live_approval_as_allowed()
                else:
                    questions = [
                        AskQuestion(
                            id="choice",
                            question="Choose a destination",
                            options=[AskOption(label="Here"), AskOption(label="There")],
                        ),
                        AskQuestion(
                            id="next",
                            question="Confirm the next step",
                            options=[AskOption(label="Proceed"), AskOption(label="Wait")],
                        ),
                    ]
                    answer_task = asyncio.create_task(handle._ask_gate(questions))
                    await _pump(pilot, lambda: app._ask_screen is not None)
                    assert app._ask_screen is not None
                    assert viewer.pending_gate is not None
                    app._ask_screen.settle({viewer.pending_gate.request_id: ["Here"]})
                if timing != "same-tick":
                    await send_entered.wait()
                if timing == "sidebar":
                    app._suspend_sidebar_gates(app._interaction)
                    assert viewer.has_pending_gate_reply
                delivery = asyncio.create_task(deliver_during_shutdown())
                app._run_slash_command("/reload")
                assert app.return_code == REEXEC_CODE
            await delivery
            assert answer_task is not None
            if kind == "ask":
                # Draining Q1 must not auto-answer/cancel Q2 while the old DOM
                # disappears. A fresh viewer receives the still-pending question.
                assert not answer_task.done()
                successor = await RemoteSession.connect(
                    server._record,
                    owner.session_id,
                    config_dir=config,
                    takeover_factory=_never_take_over,
                )

                async def replacement_factory():
                    return successor

                replacement = OperatorApp(replacement_factory)
                async with replacement.run_test(size=(120, 36)) as pilot:
                    await wait_for_adoption(replacement, pilot)
                    await _pump(pilot, lambda: replacement._ask_screen is not None)
                    assert successor.pending_gate is not None
                    assert successor.pending_gate.question_index == 1
                    assert replacement._ask_screen is not None
                    replacement._ask_screen.settle({successor.pending_gate.request_id: ["Proceed"]})
                    await _pump(pilot, answer_task.done)
                await successor.dispose()
            result = await answer_task
            assert result == (
                True if kind == "approval" else {"choice": ["Here"], "next": ["Proceed"]}
            )
            assert len(attempts) == 1
            assert not cancelled
            assert not owner._disposed
            print(
                {
                    "kind": kind,
                    "timing": timing,
                    "attempts": len(attempts),
                    "answer": result,
                    "cancelled_before_delivery": cancelled,
                }
            )
    finally:
        release_send.set()
        take_plan()
        if delivery is not None:
            await asyncio.gather(delivery, return_exceptions=True)
        if answer_task is not None and not answer_task.done():
            answer_task.cancel()
            await asyncio.gather(answer_task, return_exceptions=True)
        await viewer.dispose()
        await server.aclose()
        await owner.dispose()
