"""The process progress signal must preserve a child parked in a provider call.

The model stream owns the counter, and this test connects that real counter to
``process._step_in_flight``. A parent request is kept parked beside the child so
it cannot accidentally masquerade as child activity after the child finishes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
)
from local_operator.model.configure import SessionStreamFn
from local_operator.session.runtime import process


def _request() -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(provider="openrouter", model_id="deepseek/deepseek-v4.1-flash"),
        messages=[Message.user("progress signal test")],
    )


def _stream() -> SessionStreamFn:
    stream = SessionStreamFn(MagicMock(), {}, "process-child-progress")
    # The test observes only the provider-request state; avoid starting the
    # analytics recorder, whose asynchronous storage is outside this contract.
    stream._record_usage = MagicMock()  # type: ignore[method-assign]
    return stream


def _handle(stream: SessionStreamFn) -> SimpleNamespace:
    session = SimpleNamespace(_stream_fn=stream, _context=None, _compacting=False)
    return SimpleNamespace(_session=session)


async def _consume(
    stream: SessionStreamFn, source: AsyncIterator[StreamEvent]
) -> list[StreamEvent]:
    return [event async for event in stream._record_stream(_request(), source)]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["success", "error", "cancel"])
async def test_child_provider_request_is_process_progress_until_terminal(terminal: str) -> None:
    """Parked child calls count; terminal paths clear the signal; parent calls do not."""
    parent = _stream()
    child = parent.fork("child")
    handle = _handle(parent)
    parent_started, child_started = asyncio.Event(), asyncio.Event()
    parent_release, child_release = asyncio.Event(), asyncio.Event()

    async def parked_parent() -> AsyncIterator[StreamEvent]:
        parent_started.set()
        await parent_release.wait()
        yield StreamEndEvent(stop_reason="stop")

    async def parked_child() -> AsyncIterator[StreamEvent]:
        child_started.set()
        await child_release.wait()
        if terminal == "error":
            raise RuntimeError("child provider stream failed")
        yield StreamEndEvent(stop_reason="stop")

    parent_task = asyncio.create_task(_consume(parent, parked_parent()))
    child_task = asyncio.create_task(_consume(child, parked_child()))
    try:
        await asyncio.gather(parent_started.wait(), child_started.wait())
        # Both lanes are parked in provider streams, but only the fork is child work.
        assert process._step_in_flight(handle) is True

        if terminal == "cancel":
            child_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await child_task
        else:
            child_release.set()
            if terminal == "error":
                with pytest.raises(RuntimeError, match="child provider stream failed"):
                    await child_task
            else:
                await child_task

        # The manager's own request is still pending, but its child-only counter is clear.
        assert process._step_in_flight(handle) is False
        parent_release.set()
        await parent_task
    finally:
        for release in (parent_release, child_release):
            release.set()
        for task in (parent_task, child_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(parent_task, child_task, return_exceptions=True)
        await child.close()
        await parent.close()


def test_unreadable_child_request_signal_fails_closed() -> None:
    class BrokenStream:
        @property
        def child_model_requests_in_flight(self) -> bool:
            raise RuntimeError("counter unavailable")

    handle = _handle(BrokenStream())  # type: ignore[arg-type]
    assert process._step_in_flight(handle) is True


def test_missing_child_request_signal_preserves_idle_result() -> None:
    """Legacy/custom stream functions without the optional signal remain usable."""
    handle = _handle(SimpleNamespace())  # type: ignore[arg-type]
    assert process._step_in_flight(handle) is False
