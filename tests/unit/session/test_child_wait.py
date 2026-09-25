"""A non-delegating child keeps ``wait``, and that ``wait`` hears its parent.

The defect: a coder child (live session ``f7318cc06bdd``, 2026-09-24) polled a
background pytest run with foreground ``sleep 1500; tail log`` calls for hours.
Its role does not delegate, so the prune in ``harness.subagent`` had removed
``wait`` along with ``task`` — leaving ``jobs`` (which cannot BLOCK) and a
foreground sleep (which a hub note cannot interrupt) as its only options. Each
of its parent's three notes was read only when a sleep ran out.

These drive the REAL child construction (``_build_child_session``), the REAL
comms channel a parent's ``hub send`` uses, and the child's REAL ``wait`` and
``bash`` tools, so they pin the three properties that make the spare safe:
the tool is there, it wakes on a hub note, and it cannot reach a sibling.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from local_operator.agent_profiles import AgentProfile
from local_operator.harness import subagent as subagent_mod
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The coder shape from the evidence: full toolset, does not delegate.
CODER = AgentProfile(name="coder", description="implements a slice", may_delegate=False)


class _OneShot:
    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _parent(tmp_path) -> Session:
    return Session(
        model=MODEL,
        stream_fn=_OneShot(),
        tools=[],
        transcript=Transcript(tmp_path / "parent"),
        system_blocks_provider=lambda: ["stable", "env"],
    )


async def _coder_child(parent: Session, job_id: str) -> Session:
    child = await subagent_mod._build_child_session(
        label="backend-catalogue-paging",
        prompt="implement the slice",
        parent_session=parent,
        model_spec=None,
        job_id=job_id,
        agent="coder",
        profile=CODER,
    )
    # What ``run_subagent`` does once the child is live: a parent's
    # ``hub send`` reaches the child only through this attached record.
    parent.subagent_comms.record_launch(job_id, "backend-catalogue-paging")
    parent.subagent_comms.attach(job_id, child, child._transcript.directory)
    return child


def _tool(session: Session, name: str):
    return next(tool for tool in session._tools if tool.name == name)


@pytest.mark.asyncio
async def test_a_coder_child_holds_wait_but_not_task_or_wake(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _parent(tmp_path)
    child = await _coder_child(parent, "job-coder")
    try:
        names = {tool.name for tool in child._tools}
        assert {"bash", "jobs", "wait"} <= names
        assert names.isdisjoint({"task", "wake"})
    finally:
        await child.dispose()
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_hub_note_wakes_the_childs_wait_on_its_own_background_job(
    tmp_path, monkeypatch
) -> None:
    """The replacement for ``sleep 1500; tail log``: background the work, block
    in ``wait``, and a parent's note ends the block in well under a second
    instead of when the sleep runs out."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _parent(tmp_path)
    job_id = "job-coder"
    child = await _coder_child(parent, job_id)
    context = child._build_tool_context()
    bg_id = ""
    try:
        started = await _tool(child, "bash").execute(
            "bg", {"command": "sleep 60; echo done", "background": True}, None, None, context
        )
        assert started.is_error is False, started.text
        assert started.details is not None
        bg_id = started.details["job_id"]

        async def note_soon() -> None:
            await asyncio.sleep(0.3)
            delivery = parent.subagent_comms.send(job_id, "Manager, decision: stop the suite.")
            assert delivery.outcome == "injected", delivery

        sender = asyncio.ensure_future(note_soon())
        t0 = time.perf_counter()
        result = await asyncio.wait_for(
            _tool(child, "wait").execute(
                "w", {"job_id": bg_id, "wait_ms": 600_000}, None, None, context
            ),
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        await sender

        assert elapsed < 10.0, f"the wait heard the note after {elapsed:.1f}s"
        assert result.details is not None
        assert result.details["interrupted_by"] == "hub_message", result.details
        assert result.details["status"] == "running"
        # The job the child was waiting on is untouched: the note redirected the
        # child's attention, it did not stop its work.
        row = child.jobs.get(bg_id)
        assert row is not None and row.status == "running"
    finally:
        if bg_id:
            await child.jobs.cancel(bg_id)
        await child.dispose()
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_childs_wait_refuses_a_sibling_subagent(tmp_path, monkeypatch) -> None:
    """The spare must not widen reach. The comms registry is SHARED down the
    tree (a child holds its parent's instance), so a sibling's id and label DO
    resolve through it — the refusal comes from the child's own job manager,
    which never held the sibling. Pinned by id and by label."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _parent(tmp_path)

    async def forever(job_id, signal, report_progress) -> str:
        await asyncio.sleep(3600)
        return "never"

    sibling_id = parent.jobs.register("task", "sibling-qa", forever)
    parent.subagent_comms.record_launch(sibling_id, "sibling-qa")
    child = await _coder_child(parent, "job-coder")
    context = child._build_tool_context()
    try:
        for target in (sibling_id, "sibling-qa"):
            t0 = time.perf_counter()
            result = await _tool(child, "wait").execute(
                "w", {"job_id": target, "wait_ms": 600_000}, None, None, context
            )
            assert time.perf_counter() - t0 < 5.0
            assert result.is_error is True, (target, result.text)
            assert f"unknown job {target}" in result.text
        row = parent.jobs.get(sibling_id)
        assert row is not None and row.status == "running"
    finally:
        await parent.jobs.cancel(sibling_id)
        await child.dispose()
        await parent.dispose()
