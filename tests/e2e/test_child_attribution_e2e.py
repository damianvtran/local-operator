"""A child's stop attribution must survive a parent restart (design §4.4, e2e).

The unit cells drive the same code paths through real ``Session``s, but they
build the parent directly. This cell is the ASSEMBLED shape: a real
``ServingSessionHandle`` behind a real ``RuntimeServer``, a child launched
through it, stopped through the serving handle's own mobile-stop path (the
``_cancel_children`` caller this PR gives an actor token), and then a fresh
parent booted over the same transcript directory reading the roster back.

The claim under test is the one the whole workstream exists for: after the
process that stopped the child is gone, the next boot can still say WHO ended
it and WHY — which is what the stop receipt is for.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR, Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import execute_hub

pytestmark = pytest.mark.e2e

MODEL = ModelSpec(provider="test", model_id="attr", context_window=100_000)


def _tool_call(name: str, args: dict[str, Any]) -> Any:
    async def gen():
        yield StreamToolCallDelta(
            index=0, id=f"call-{name}", name=name, argument_delta=json.dumps(args)
        )
        yield StreamEndEvent(stop_reason="toolUse")

    return gen()


def _text(body: str) -> Any:
    async def gen():
        yield StreamTextDelta(delta=body)
        yield StreamEndEvent(stop_reason="stop")

    return gen()


def _provider(*, child_marker: str):
    """Parent answers in prose; the CHILD parks in one long tool call.

    The marker keeps the two roles apart on one scripted stream — the child is
    recognised by its own prompt reaching the provider, exactly as the existing
    accounting e2e does.

    THE CHILD MUST ACTUALLY PARK, and that is a correctness requirement of this
    cell, not decoration (reviewer MAJOR 1 / QA Q2). A bare ``sleep 3600`` is
    REFUSED by the sleep guard — six identical refusals then trip the loop's
    no-progress guard, so the child settles itself about 50 ms after attaching
    and the cell only passed when ``abort()`` won that race (3/3 red at load
    20-33, 6/6 green at load ~11.5). On the green runs it also passed for the
    wrong reason: it asserted the torn-down race rather than the shape this
    workstream exists for. The guard's OWN inline escape is used instead, so the
    child sits in a real, blocking ``bash`` call until something stops it —
    which is what a wedged lane looks like in the field.
    """

    def stream(request, signal=None):
        is_child = any(child_marker in getattr(message, "text", "") for message in request.messages)
        if is_child:
            return _tool_call(
                "bash",
                {
                    "command": "LOCAL_OPERATOR_ALLOW_LONG_SLEEP=1 sleep 3600",
                    "i": "working",
                },
            )
        return _text("parent acknowledged")

    return stream


@pytest.mark.asyncio
async def test_a_cancelled_childs_attribution_survives_a_parent_restart(
    headless_tui_env: Path, workspace: Path
) -> None:
    directory = headless_tui_env / "sessions" / "attribsess1"
    marker = "ATTRIBUTION_CHILD"

    async def approve(*_args: Any, **_kwargs: Any) -> bool:
        return True

    def build() -> Session:
        return Session(
            model=MODEL,
            stream_fn=_provider(child_marker=marker),
            tools=[],
            transcript=Transcript(directory),
            system_blocks_provider=lambda *_: [],
            yolo=True,
            cwd=str(workspace),
            request_approval=approve,
        )

    owner = build()
    await owner.async_init()
    handle = ServingSessionHandle(
        owner, asyncio.get_running_loop(), cwd=str(workspace), auto_approve=True
    )
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    child_dir: Path | None = None
    try:
        job_id = owner._launch_subagent("attrib-child", marker)
        for _ in range(400):
            await asyncio.sleep(0.05)
            child_dir = owner.subagent_comms.session_dir_of(job_id)
            if child_dir is not None:
                break
        assert child_dir is not None, "the child never attached"

        # WAIT UNTIL IT IS GENUINELY PARKED, and assert it is: a lane receipt on
        # disk (staged at attach) and a runner still in flight. Without this the
        # cell can pass on a child that had already settled itself, which is the
        # failure mode the reviewer reproduced.
        from local_operator.session import subagent_ledger as ledger

        for _ in range(200):
            job = owner.jobs.get(job_id)
            if job is not None and job.status == "running" and ledger.read_lane_receipts(child_dir):
                break
            await asyncio.sleep(0.05)
        parked = owner.jobs.get(job_id)
        assert parked is not None and parked.status == "running", (
            f"the child did not stay parked (status={getattr(parked, 'status', None)}); "
            "a child that settles itself makes this cell assert a race, not a wedge"
        )
        assert ledger.read_lane_receipts(child_dir), "the lane receipt was never staged"

        # The MOBILE path: this is ``serving._cancel_children``, whose actor token
        # this PR adds. It cancels every running child of the session.
        await handle.abort()
        for _ in range(200):
            await asyncio.sleep(0.05)
            job = owner.jobs.get(job_id)
            if job is not None and job.status != "running":
                break
    finally:
        server.close()
        await owner.dispose()

    # The stop receipt is on disk in the CHILD's directory, and it names the phone.
    receipts = sorted(child_dir.glob("subagent-stop-*.v1.json"))
    assert receipts, f"no stop receipt survived in {child_dir}"
    payload = json.loads(receipts[0].read_text())
    assert payload["actor"] == "mobile-stop"
    assert payload["deliberate"] is True

    # A FRESH parent over the same transcript directory: the boot reconcile pass
    # is what turns the receipt back into a roster attribution.
    restarted = build()
    try:
        await restarted.async_init()
        info = next(
            (row for row in restarted.subagent_comms.roster() if row.label == "attrib-child"),
            None,
        )
        assert info is not None, "the restarted parent lost the child's record"
        assert info.ended_by == "mobile-stop", info

        from local_operator.harness.types import ToolContext

        listed = await execute_hub(
            "c",
            {"op": "list"},
            None,
            None,
            ToolContext(
                cwd=str(workspace),
                subagent_comms=restarted.subagent_comms,
                jobs=restarted.jobs,
            ),
        )
        text = listed.content[0].text  # type: ignore[union-attr]
        assert "ended by the phone" in text, text
    finally:
        await restarted.dispose()

    # The sidecar was written by the restarted boot, so the claim is durable.
    assert (directory / SUBAGENT_ROSTER_SIDECAR).exists()
