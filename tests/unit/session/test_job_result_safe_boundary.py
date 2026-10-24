"""Safe-boundary delivery for canonical background-job results."""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any

import pytest

import local_operator.session.session as session_module
from local_operator.harness.comms import SubagentComms
from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE, JobType
from local_operator.harness.loop import AgentLoop
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentMessage,
    AgentTool,
    ChatRequest,
    CustomMessage,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import build_bash_tool, execute_wait
from tests.unit.session.test_session import make_session, wait_for


def _wire(request: ChatRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), default=str)


def _job_result_rows(session: Any) -> list[Any]:
    return [
        entry
        for entry in session._transcript.entries()
        if entry.type == "message" and entry.payload.get("custom_type") == JOB_RESULT_MESSAGE_TYPE
    ]


def _job(session: Any, job_id: str) -> Any:
    job = session.jobs.get(job_id)
    assert job is not None
    return job


@pytest.mark.asyncio
async def test_job_result_reaches_tool_boundary_before_parent_final(tmp_path: Path) -> None:
    """A result settled during a real tool is input to the next request.

    The hub report and terminal result are independent events: report A must not
    consume or textually deduplicate final B. The existing turn, rather than a
    stale post-final delivery turn, must carry both to the provider.
    """
    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute(call_id, args, signal, on_update, context):
        tool_entered.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name="slow",
            content=[TextContent(text="tool completed without cancellation")],
        )

    tool = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)
            request_number = len(self.requests)

            async def generate():
                if request_number == 1:
                    yield StreamToolCallDelta(
                        index=0,
                        id="slow-call",
                        name="slow",
                        argument_delta="{}",
                    )
                    yield StreamEndEvent(stop_reason="tool_calls")
                else:
                    yield StreamTextDelta(delta=f"parent final {request_number}")
                    yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream, tools=[tool])
    comms = SubagentComms(session)
    parent = asyncio.create_task(session.prompt("do work"))
    try:
        await asyncio.wait_for(tool_entered.wait(), 5)

        async def runner(job_id, signal, report_progress):
            comms.reply_to_parent(job_id, "interim or report A")
            return "distinct final B"

        job_id = session.jobs.register("task", "research", runner)
        await wait_for(lambda: bool(_job_result_rows(session)))
        assert len(_job_result_rows(session)) == 1
        assert _job(session, job_id).consumed is False

        release_tool.set()
        await parent
        # The finally fallback is spawned synchronously before prompt returns.
        # Drain that task when the broken implementation creates it; the fixed
        # path has no automatic post-final delivery task to wait for.
        await wait_for(lambda: not session._background_tasks)

        assert len(stream.requests) == 2, "the result must not buy a stale post-final turn"
        second_request = _wire(stream.requests[1])
        assert "subagent-message" in second_request
        assert "interim or report A" in second_request
        assert second_request.count("background job 'research' completed") == 1
        assert second_request.count("distinct final B") == 1
        assert "tool completed without cancellation" in second_request
        assert len(_job_result_rows(session)) == 1
        assert _job(session, job_id).consumed is True
    finally:
        release_tool.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_writes, expected_requests, fallback_owns_result",
    [
        pytest.param(1, 2, False, id="first-safe-retry"),
        pytest.param(2, 3, False, id="later-safe-retry"),
        pytest.param(4, 3, True, id="finally-fallback-after-safe-retries"),
    ],
)
async def test_result_commits_only_after_its_canonical_row_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_writes: int,
    expected_requests: int,
    fallback_owns_result: bool,
) -> None:
    """Transient journal failures retry before ANY automatic context commit.

    One failure is repaired at the first safe boundary; two move recovery to a
    later safe boundary in the same run. Four exhaust every safe retry, so the
    existing finally fallback must own the pending result before delivering it.
    """
    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()
    rows_at_request: list[int] = []

    async def execute(call_id, args, signal, on_update, context):
        tool_entered.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name="slow",
            content=[TextContent(text="slow completed")],
        )

    tool = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            rows_at_request.append(len(_job_result_rows(session)))
            self.requests.append(request)

            async def generate():
                if len(self.requests) == 1:
                    yield StreamToolCallDelta(
                        index=0, id="slow-call", name="slow", argument_delta="{}"
                    )
                    yield StreamEndEvent(stop_reason="tool_calls")
                else:
                    yield StreamTextDelta(delta="parent response")
                    yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream, tools=[tool])
    real_fallback = session._deliver_deferred_job_results
    fallback_pending_snapshots: list[list[str]] = []

    async def observe_fallback() -> None:
        fallback_pending_snapshots.append(list(session._deferred_job_results))
        await real_fallback()

    monkeypatch.setattr(session, "_deliver_deferred_job_results", observe_fallback)
    real_append = session._transcript.append_message
    failures_left = failed_writes

    async def fail_job_row_writes(message, *args, **kwargs):
        nonlocal failures_left
        if (
            isinstance(message, CustomMessage)
            and message.custom_type == JOB_RESULT_MESSAGE_TYPE
            and failures_left > 0
        ):
            failures_left -= 1
            raise OSError("injected transient job-row write failure")
        return await real_append(message, *args, **kwargs)

    monkeypatch.setattr(session._transcript, "append_message", fail_job_row_writes)
    parent = asyncio.create_task(session.prompt("preserve the canonical result"))
    try:
        await asyncio.wait_for(tool_entered.wait(), 5)

        async def runner(job_id, signal, report_progress):
            return "durable before delivery"

        job_id = session.jobs.register("task", "transient-journal-failure", runner)
        await wait_for(lambda: job_id in session._deferred_job_results)
        assert _job_result_rows(session) == []

        release_tool.set()
        await parent
        await wait_for(lambda: not session._background_tasks)

        assert failures_left == 0
        assert len(stream.requests) == expected_requests
        assert rows_at_request == [0] * (expected_requests - 1) + [1]
        for request in stream.requests[1:-1]:
            assert "durable before delivery" not in _wire(request)
        assert "durable before delivery" in _wire(stream.requests[-1])
        assert len(_job_result_rows(session)) == 1
        assert not session._deferred_job_results
        if fallback_owns_result:
            assert fallback_pending_snapshots == [[job_id], []]
        else:
            assert fallback_pending_snapshots == [[]]
            assert _job(session, job_id).consumed is True
    finally:
        release_tool.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


@pytest.mark.asyncio
async def test_result_after_last_yield_uses_the_finally_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely late result still gets one later delivery turn."""
    real_collect_yield = AgentLoop._collect_yield_injections
    job_ids: list[str] = []

    async def settle_after_last_yield(config):
        late = await real_collect_yield(config)
        if not late and not job_ids:
            # The yield snapshot is already fixed. Settling now is after this
            # run's LAST safe boundary but while Session is still streaming, so
            # only its finally fallback can own the pending result.
            async def runner(job_id, signal, report_progress):
                return "genuinely post-last-boundary result"

            job_ids.append(session.jobs.register("task", "last-boundary", runner))
            await wait_for(lambda: job_ids[0] in session._deferred_job_results)
            assert session._is_streaming
        return late

    monkeypatch.setattr(
        AgentLoop, "_collect_yield_injections", staticmethod(settle_after_last_yield)
    )

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)

            async def generate():
                yield StreamTextDelta(delta="response")
                yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream)
    try:
        await session.prompt("do work")
        await wait_for(lambda: not session._background_tasks)

        assert len(stream.requests) == 2
        assert "genuinely post-last-boundary result" not in _wire(stream.requests[0])
        assert "genuinely post-last-boundary result" in _wire(stream.requests[1])
        assert len(_job_result_rows(session)) == 1
        assert not session._deferred_job_results
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_peer_run_records_job_origin_only_after_result_commit(
    tmp_path: Path,
) -> None:
    """A job incorporated into a peer run retains the job's loud completion policy."""
    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()
    ends: list[AgentEndEvent] = []

    async def execute(call_id, args, signal, on_update, context):
        tool_entered.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name="slow",
            content=[TextContent(text="slow completed")],
        )

    tool = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)

            async def generate():
                if len(self.requests) == 1:
                    yield StreamToolCallDelta(
                        index=0, id="slow-call", name="slow", argument_delta="{}"
                    )
                    yield StreamEndEvent(stop_reason="tool_calls")
                else:
                    yield StreamTextDelta(delta="important final B was incorporated")
                    yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream, tools=[tool])
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    try:
        await session.receive_peer_message(
            "Continue the outstanding research",
            mode="mailbox",
            wake=True,
            sender={"pid": 42, "conversation_name": "synthetic-peer"},
        )
        await asyncio.wait_for(tool_entered.wait(), 5)

        async def runner(job_id, signal, report_progress):
            return "distinct important final B"

        job_id = session.jobs.register("task", "research", runner)
        await wait_for(lambda: job_id in session._deferred_job_results)
        release_tool.set()
        await wait_for(
            lambda: bool(ends) and not session.is_streaming and not session._background_tasks
        )

        assert len(stream.requests) == 2
        assert "distinct important final B" in _wire(stream.requests[1])
        assert len(ends) == 1
        assert ends[0].notify is True
        assert session._run_input_types == {"peer_message", "job_result"}
        assert _job(session, job_id).consumed is True
        assert (await session.refresh_attention())["unseen"] is True
    finally:
        release_tool.set()
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("job_type", ["task", "video"])
async def test_job_result_reaches_no_tool_yield_boundary(
    tmp_path: Path, job_type: JobType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-tool response re-enters at the yield boundary with the result.

    Video uses the same generic manager completion carrier as task/bash. This is
    deliberately a controlled manager result, not a claim of live video-provider
    coverage.
    """
    boundary_entered = asyncio.Event()
    release_boundary = asyncio.Event()
    real_loop_config = session_module.LoopConfig

    def with_yield_gate(**kwargs: Any):
        config = real_loop_config(**kwargs)
        config.on_before_yield = gated_boundary
        return config

    monkeypatch.setattr(session_module, "LoopConfig", with_yield_gate)

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)
            request_number = len(self.requests)

            async def generate():
                yield StreamTextDelta(delta=f"parent response {request_number}")
                yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream)

    async def gated_boundary() -> None:
        # The post-response drain has already taken its snapshot. Settling here
        # makes the YIELD collector carry the result, not that earlier drain.
        boundary_entered.set()
        await release_boundary.wait()

    parent = asyncio.create_task(session.prompt("work without tools"))
    try:
        await asyncio.wait_for(boundary_entered.wait(), 5)
        result_text = f"settled {job_type} result"

        async def runner(job_id, signal, report_progress):
            return result_text

        job_id = session.jobs.register(job_type, f"{job_type}-completion", runner)
        await wait_for(lambda: bool(_job_result_rows(session)))
        release_boundary.set()
        await parent
        await wait_for(lambda: not session._background_tasks)

        assert len(stream.requests) == 2
        second_request = _wire(stream.requests[1])
        assert result_text in second_request
        assert f"background job '{job_type}-completion' completed" in second_request
        assert len(_job_result_rows(session)) == 1
        assert _job(session, job_id).consumed is True
    finally:
        release_boundary.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


@pytest.mark.asyncio
async def test_wait_wins_against_a_materialized_result_carrier(tmp_path: Path) -> None:
    """A real wait result suppresses the already-materialized automatic copy."""
    stream = ScriptedEmptyStream()
    session = make_session(tmp_path, stream)
    session._is_streaming = True

    async def runner(job_id, signal, report_progress):
        return "returned through wait"

    job_id = session.jobs.register("task", "wait-race", runner)
    try:
        await wait_for(lambda: bool(_job_result_rows(session)))
        carriers: list[AgentMessage] = [
            message
            for message in await session._drain_asides()
            if isinstance(message, CustomMessage) and message.custom_type == JOB_RESULT_MESSAGE_TYPE
        ]
        assert len(carriers) == 1
        carrier = carriers[0]
        assert _job(session, job_id).consumed is False

        result = await execute_wait(
            "wait-call",
            {"job_id": job_id, "wait_ms": 1000},
            None,
            None,
            ToolContext(cwd=str(tmp_path), session_id="wait-race", jobs=session.jobs),
        )
        assert "returned through wait" in result.text
        assert _job(session, job_id).consumed is True

        AgentLoop._drain_pending([carrier], session._context)
        assert carrier not in session._context.messages
        assert job_id not in session._deferred_job_results
        assert len(_job_result_rows(session)) == 1

        session._is_streaming = False
        await session._deliver_deferred_job_results()
        await wait_for(lambda: not session._background_tasks)
        assert not stream.requests
    finally:
        session._is_streaming = False
        await session.dispose()


@pytest.mark.asyncio
async def test_discarded_result_is_requeued_without_false_acknowledgement(tmp_path: Path) -> None:
    """The loop's budget/abort discard hook retains the canonical pending row."""
    stream = ScriptedEmptyStream()
    session = make_session(tmp_path, stream)
    session._is_streaming = True

    async def runner(job_id, signal, report_progress):
        return "not yet acknowledged"

    job_id = session.jobs.register("task", "discard-race", runner)
    try:
        await wait_for(lambda: bool(_job_result_rows(session)))
        carriers: list[AgentMessage] = [
            message
            for message in await session._drain_asides()
            if isinstance(message, CustomMessage) and message.custom_type == JOB_RESULT_MESSAGE_TYPE
        ]
        assert len(carriers) == 1
        carrier = carriers[0]

        AgentLoop._discard_pending_custom([carrier])
        assert _job(session, job_id).consumed is False
        assert job_id in session._deferred_job_results
        assert carrier not in session._context.messages

        # A later safe boundary can offer the exact same object again; no twin
        # row or replacement message is needed after a discarded attempt.
        retried = [
            message
            for message in await session._drain_asides()
            if isinstance(message, CustomMessage) and message.custom_type == JOB_RESULT_MESSAGE_TYPE
        ]
        assert retried == [carrier]
        assert len(_job_result_rows(session)) == 1

        # Departure still wins over a pending retry and buys no provider turn.
        session.retire_job_deliveries_to_transcript()
        session._is_streaming = False
        await session._deliver_deferred_job_results()
        assert not stream.requests
        assert _job(session, job_id).consumed is False
        assert not session._deferred_job_results
        assert len(_job_result_rows(session)) == 1
    finally:
        session._is_streaming = False
        await session.dispose()


@pytest.mark.asyncio
async def test_boundary_commit_uses_the_retained_job_after_manager_sweep(
    tmp_path: Path,
) -> None:
    """Manager retention cannot invalidate an already-journaled pending result."""
    session = make_session(tmp_path, ScriptedEmptyStream())
    session._is_streaming = True

    async def runner(job_id, signal, report_progress):
        return "retained after sweep"

    job_id = session.jobs.register("task", "retention-race", runner)
    try:
        await wait_for(lambda: bool(_job_result_rows(session)))
        job = session.jobs.get(job_id)
        assert job is not None
        job.settled_at = 1
        assert session.jobs.list() == [], "precondition: the read-time retention sweep ran"
        assert session.jobs.get(job_id) is None

        carriers: list[AgentMessage] = [
            message
            for message in await session._drain_asides()
            if isinstance(message, CustomMessage) and message.custom_type == JOB_RESULT_MESSAGE_TYPE
        ]
        assert len(carriers) == 1
        AgentLoop._drain_pending(carriers, session._context)

        assert job.consumed is True
        assert carriers == [
            message
            for message in session._context.messages
            if isinstance(message, CustomMessage) and message.custom_type == JOB_RESULT_MESSAGE_TYPE
        ]
        assert not session._deferred_job_results
        assert len(_job_result_rows(session)) == 1
    finally:
        session._is_streaming = False
        await session.dispose()


@pytest.mark.asyncio
async def test_budget_exhaustion_leaves_result_for_the_finally_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real zero-budget branch discards without consuming or losing."""
    real_loop_config = session_module.LoopConfig

    def no_aside_continuations(**kwargs: Any):
        config = real_loop_config(**kwargs)
        config.max_paused_turn_continuations = 0
        config.on_before_yield = gated_boundary
        return config

    monkeypatch.setattr(session_module, "LoopConfig", no_aside_continuations)
    boundary_entered = asyncio.Event()
    release_boundary = asyncio.Event()
    fallback_entered = asyncio.Event()
    release_fallback = asyncio.Event()

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)
            request_number = len(self.requests)

            async def generate():
                yield StreamTextDelta(delta=f"response {request_number}")
                yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream)

    async def gated_boundary() -> None:
        # Settlement AFTER the post-response drain snapshot is what makes the
        # yield collector, rather than the earlier carrier, own this result.
        boundary_entered.set()
        await release_boundary.wait()

    real_fallback = session._deliver_deferred_job_results
    fallback_observation: dict[str, Any] = {}

    async def gated_fallback() -> None:
        fallback_observation["pending"] = dict(session._deferred_job_results)
        fallback_observation["context_ids"] = {message.id for message in session._context.messages}
        fallback_entered.set()
        await release_fallback.wait()
        await real_fallback()

    monkeypatch.setattr(session, "_deliver_deferred_job_results", gated_fallback)
    parent = asyncio.create_task(session.prompt("hit the aside budget"))
    try:
        await asyncio.wait_for(boundary_entered.wait(), 5)

        async def runner(job_id, signal, report_progress):
            return "result survives budget exhaustion"

        job_id = session.jobs.register("task", "budget-exhaustion", runner)
        await wait_for(lambda: bool(_job_result_rows(session)))
        release_boundary.set()
        await asyncio.wait_for(fallback_entered.wait(), 5)

        job = _job(session, job_id)
        assert job.consumed is False
        assert list(fallback_observation["pending"]) == [job_id]
        pending_message = fallback_observation["pending"][job_id][2]
        assert pending_message.id not in fallback_observation["context_ids"]
        assert len(stream.requests) == 1
        assert len(_job_result_rows(session)) == 1

        release_fallback.set()
        await parent
        await wait_for(lambda: not session._background_tasks)
        # Continuation exhaustion is stopped work: the existing pre-abort
        # residue guard holds the fallback rather than buying a provider turn.
        assert len(stream.requests) == 1
        assert "result survives budget exhaustion" not in _wire(stream.requests[0])
        assert job.consumed is False
        assert not session._deferred_job_results
        assert len(_job_result_rows(session)) == 1
    finally:
        release_boundary.set()
        release_fallback.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["abort", "leaving"])
async def test_departure_after_materialization_does_not_acknowledge(
    tmp_path: Path, departure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A latch between materialization and commit buys NO provider continuation."""
    boundary_entered = asyncio.Event()
    release_boundary = asyncio.Event()
    real_loop_config = session_module.LoopConfig

    def with_yield_gate(**kwargs: Any):
        config = real_loop_config(**kwargs)
        config.on_before_yield = gated_boundary
        return config

    monkeypatch.setattr(session_module, "LoopConfig", with_yield_gate)
    real_drain = AgentLoop._drain_pending
    armed: list[bool] = []

    def arm_then_drain(pending: list[AgentMessage], context: Any) -> int:
        if any(
            isinstance(message, CustomMessage) and message.custom_type == JOB_RESULT_MESSAGE_TYPE
            for message in pending
        ):
            armed.append(True)
            if departure == "abort":
                session.abort("abort after materialization")
            else:
                session.retire_job_deliveries_to_transcript()
        return real_drain(pending, context)

    monkeypatch.setattr(AgentLoop, "_drain_pending", staticmethod(arm_then_drain))

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)

            async def generate():
                yield StreamTextDelta(delta="unexpected post-latch continuation")
                yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream)
    real_fallback = session._deliver_deferred_job_results
    fallback_entered = asyncio.Event()
    release_fallback = asyncio.Event()

    async def gated_fallback() -> None:
        fallback_entered.set()
        await release_fallback.wait()
        await real_fallback()

    monkeypatch.setattr(session, "_deliver_deferred_job_results", gated_fallback)

    async def gated_boundary() -> None:
        boundary_entered.set()
        await release_boundary.wait()

    parent = asyncio.create_task(session.prompt("depart at the yield commit"))
    try:
        await asyncio.wait_for(boundary_entered.wait(), 5)

        async def runner(job_id, signal, report_progress):
            return "pending through departure"

        job_id = session.jobs.register("task", f"{departure}-after-materialization", runner)
        await wait_for(lambda: job_id in session._deferred_job_results)
        release_boundary.set()
        await asyncio.wait_for(fallback_entered.wait(), 5)

        assert armed == [True]
        assert len(stream.requests) == 1
        assert "pending through departure" not in _wire(stream.requests[0])
        assert _job(session, job_id).consumed is False
        assert job_id in session._deferred_job_results
        assert "job_result" not in session._run_input_types
        assert len(_job_result_rows(session)) == 1

        release_fallback.set()
        await parent
        await wait_for(lambda: not session._background_tasks)
        assert len(stream.requests) == 1
        assert _job(session, job_id).consumed is False
        assert not session._deferred_job_results
        assert len(_job_result_rows(session)) == 1
    finally:
        release_boundary.set()
        release_fallback.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


@pytest.mark.asyncio
async def test_abort_before_boundary_keeps_result_unacknowledged(tmp_path: Path) -> None:
    """Aborting the running tool cannot turn a pending result into a delivery."""
    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute(call_id, args, signal, on_update, context):
        tool_entered.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name="slow",
            content=[TextContent(text="tool finished")],
        )

    tool = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)

            async def generate():
                if len(self.requests) == 1:
                    yield StreamToolCallDelta(
                        index=0, id="slow-call", name="slow", argument_delta="{}"
                    )
                    yield StreamEndEvent(stop_reason="tool_calls")
                else:
                    yield StreamTextDelta(delta="unexpected post-abort delivery")
                    yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream, tools=[tool])
    parent = asyncio.create_task(session.prompt("abort this turn"))
    try:
        await asyncio.wait_for(tool_entered.wait(), 5)

        async def runner(job_id, signal, report_progress):
            return "pending through abort"

        job_id = session.jobs.register("task", "abort-race", runner)
        await wait_for(lambda: bool(_job_result_rows(session)))
        session.abort("test abort before boundary")
        release_tool.set()
        await parent
        await wait_for(lambda: not session._background_tasks)

        assert len(stream.requests) == 1
        assert _job(session, job_id).consumed is False
        assert not session._deferred_job_results
        assert "pending through abort" not in _wire(stream.requests[0])
        assert len(_job_result_rows(session)) == 1
    finally:
        release_tool.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


@pytest.mark.asyncio
async def test_real_background_bash_reaches_the_next_tool_boundary(tmp_path: Path) -> None:
    """A real bash wrapper completion uses the same in-turn safe carrier."""
    slow_entered = asyncio.Event()
    release_slow = asyncio.Event()
    release_bash = tmp_path / "release-background-bash"
    command = (
        f"while [ ! -f {shlex.quote(str(release_bash))} ]; do sleep 0.02; done; "
        "printf 'real bash final'"
    )

    async def execute_slow(call_id, args, signal, on_update, context):
        slow_entered.set()
        await release_slow.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name="slow",
            content=[TextContent(text="slow tool finished")],
        )

    slow = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}},
        execute=execute_slow,
    )

    class Stream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)
            request_number = len(self.requests)

            async def generate():
                if request_number == 1:
                    yield StreamToolCallDelta(
                        index=0,
                        id="bash-call",
                        name="bash",
                        argument_delta=json.dumps({"command": command, "background": True}),
                    )
                    yield StreamEndEvent(stop_reason="tool_calls")
                elif request_number == 2:
                    yield StreamToolCallDelta(
                        index=0,
                        id="slow-call",
                        name="slow",
                        argument_delta="{}",
                    )
                    yield StreamEndEvent(stop_reason="tool_calls")
                else:
                    yield StreamTextDelta(delta="parent received bash")
                    yield StreamEndEvent(stop_reason="stop")

            return generate()

    stream = Stream()
    session = make_session(tmp_path, stream, tools=[build_bash_tool(), slow])
    parent = asyncio.create_task(session.prompt("run background bash"))
    try:
        await asyncio.wait_for(slow_entered.wait(), 10)
        bash_jobs = [job for job in session.jobs.list() if job.type == "bash"]
        context_text = "\n".join(
            getattr(message, "text", "") for message in session._context.messages
        )
        assert len(bash_jobs) == 1, context_text
        bash_job = bash_jobs[0]
        assert bash_job.status == "running"

        release_bash.write_text("run", encoding="utf-8")
        await wait_for(lambda: bool(_job_result_rows(session)), timeout=10)
        assert bash_job.status == "completed"
        release_slow.set()
        await parent
        await wait_for(lambda: not session._background_tasks)

        assert len(stream.requests) == 3
        third_request = _wire(stream.requests[2])
        assert "real bash final" in third_request
        assert "slow tool finished" in third_request
        assert len(_job_result_rows(session)) == 1
        assert bash_job.consumed is True
    finally:
        release_bash.write_text("run", encoding="utf-8")
        release_slow.set()
        if not parent.done():
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
        await session.dispose()


class ScriptedEmptyStream:
    """A stream that records any unexpected automatic provider delivery."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def generate():
            yield StreamTextDelta(delta="automatic delivery")
            yield StreamEndEvent(stop_reason="stop")

        return generate()
