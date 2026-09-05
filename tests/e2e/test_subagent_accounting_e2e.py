"""The default background runtime, not an in-process widget double, owns money."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    Usage,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR, Session
from local_operator.session.transcript import Transcript
from local_operator.tui.widgets.subagent_panel import job_stats
from tests.e2e.test_viewer_attach_e2e import _never_take_over, _wait_for_record

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_background_bills_survive_failure_resume_sweep_and_restart(
    headless_tui_env: Path, workspace: Path
) -> None:
    """A billed real bash call fails on its next model call, then resumes.

    The provider boundary is scripted; child construction, tool execution,
    persistence, socket attach and resume are all the production path. A
    renderer with a cold model cache must use the owner's price facts.
    """
    directory = headless_tui_env / "sessions" / "billingsess1"
    effect = workspace / "accounted.txt"
    calls = 0

    async def stream(request, signal=None):
        nonlocal calls
        # Completion delivery prompts the parent independently; do not let that
        # normal runtime turn consume the child's deterministic provider script.
        is_child = any(
            "BILLING_CHILD" in getattr(message, "text", "") for message in request.messages
        )
        if not is_child:
            yield StreamTextDelta(delta="Acknowledged child outcome")
            yield StreamEndEvent(stop_reason="stop")
            return
        calls += 1
        if calls == 1:
            yield StreamToolCallDelta(
                index=0,
                id="write-bill",
                name="bash",
                argument_delta=json.dumps(
                    {"command": f"printf billed > {effect}", "i": "Writing billed evidence"}
                ),
            )
            yield StreamEndEvent(
                stop_reason="toolUse",
                usage=Usage(input_tokens=1000, usd_cost=0.125, context_tokens=1000),
            )
        elif calls == 2:
            raise RuntimeError("controlled failure after billed tool")
        else:
            yield StreamTextDelta(delta="Finished billing child")
            yield StreamEndEvent(
                stop_reason="stop",
                usage=Usage(input_tokens=1000, usd_cost=0.125, context_tokens=1000),
            )

    async def approve(*_args, **_kwargs):
        return True

    def build():
        return Session(
            model=ModelSpec(provider="test", model_id="billing", context_window=100000),
            stream_fn=stream,
            tools=[],
            transcript=Transcript(directory),
            system_blocks_provider=lambda *_: [],
            yolo=True,
            cwd=str(workspace),
            request_approval=approve,
        )

    owner = build()
    handle = OwnedSessionHandle(
        owner, asyncio.get_running_loop(), cwd=str(workspace), auto_approve=True
    )
    server = RuntimeServer(handle, kind="daemon")
    viewer = None
    await server.start_in_process()
    try:
        record = await _wait_for_record(headless_tui_env, owner.session_id)
        viewer = await RemoteSession.connect(
            record, owner.session_id, config_dir=headless_tui_env, takeover_factory=_never_take_over
        )
        job_id = owner._launch_subagent("accounted", "BILLING_CHILD")
        await asyncio.wait_for(owner.jobs.settled_event(job_id).wait(), 20)
        assert owner.jobs.get(job_id).status == "failed"
        assert effect.read_text() == "billed"
        assert owner.frontend_state.cumulative_cost == 0.125
        new_id, error = owner.subagent_comms.resume(job_id, "Continue BILLING_CHILD")
        assert error is None and new_id is not None
        await asyncio.wait_for(owner.jobs.settled_event(new_id).wait(), 20)
        owner.refresh_frontend_state()
        assert owner.frontend_state.cumulative_cost == 0.25
        # Wait for a canonical socket frame rather than an elapsed paint budget.
        async with asyncio.timeout(20):
            while viewer.frontend_state.cumulative_cost != 0.25:
                await asyncio.sleep(0.01)
        stats = await asyncio.to_thread(job_stats, viewer.jobs.get(new_id))
        assert stats.cost == 0.125  # Row remains current-attempt/self-only.
        assert stats.context_tokens == 1000
        owner.jobs._retention_ms = 0
        owner.jobs._sweep_due()
        assert owner.jobs.list() == []
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await owner.dispose()

    checkpoint = json.loads((directory / SUBAGENT_ROSTER_SIDECAR).read_text())
    assert checkpoint["jobs"] == []
    assert sum(row["usd_cost"] or 0 for row in checkpoint["accounting"]) == 0.25
    restarted = build()
    try:
        await restarted.async_init()
        restarted.refresh_frontend_state()
        assert restarted.frontend_state.cumulative_cost == 0.25
        new_id = restarted._launch_subagent("after sweep", "BILLING_CHILD")
        await asyncio.wait_for(restarted.jobs.settled_event(new_id).wait(), 20)
        restarted.refresh_frontend_state()
        assert restarted.frontend_state.cumulative_cost == 0.375
    finally:
        await restarted.dispose()
