"""The ``wait`` budget is sized to the work, not to a poll interval.

The measured problem (this host, 30 h of Anthropic transcripts): the old
300 000 ms ceiling made an agent awaiting CI (8-20 min) or a long subagent
(30-90 min) poll every five minutes — 1 488 wait-only model calls in 877
consecutive chains, 434 of them two or more polls deep. Each poll re-sends the
whole context, and any poll landing after the provider's 5-minute prompt-cache
TTL rewrites the prefix. A 60-minute ceiling lets one call cover the job.

The ceiling only raises the budget; it must not change what wakes the wait.
The settle / peer / abort early-returns are pinned at a full-hour budget here
so a regression that made a long wait sleep to its deadline is caught by a
test that runs in well under a second, and the existing suites keep their
short-budget coverage:

- ``test_wait_settles_on_events.py`` — settle, dispose, abort at 300 000 ms.
- ``test_wait_wakes_on_peer_message.py`` — peer message and steer cancel.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import ValidationError

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools.builtin import WaitParams, build_wait_tool, execute_wait

ONE_HOUR_MS = 3_600_000


def _runner(delay: float, result: str = "done") -> Any:
    async def run(job_id: str, signal: Any, report_progress: Any) -> str:
        await asyncio.sleep(delay)
        return result

    return run


def test_wait_ms_accepts_a_full_hour_and_rejects_more() -> None:
    """3 600 000 validates; one millisecond over is a schema error."""
    assert WaitParams(job_id="j", wait_ms=ONE_HOUR_MS).wait_ms == ONE_HOUR_MS
    with pytest.raises(ValidationError):
        WaitParams(job_id="j", wait_ms=ONE_HOUR_MS + 1)
    with pytest.raises(ValidationError):
        WaitParams(job_id="j", wait_ms=0)


def test_wait_ms_default_spans_a_typical_ci_run() -> None:
    """The default is ten minutes: one call covers a typical pipeline."""
    assert WaitParams(job_id="j").wait_ms == 600_000


def test_wait_schema_advertises_the_sizing_rule() -> None:
    """The model reads the JSON schema, not this module: the ceiling, the
    default and the sizing instruction must all reach it through the tool."""
    tool = build_wait_tool(ToolContext(cwd=".", jobs=AsyncJobManager()))
    assert tool is not None
    field = tool.parameters["properties"]["wait_ms"]
    assert field["maximum"] == ONE_HOUR_MS
    assert field["default"] == 600_000
    assert "SIZE THE BUDGET TO THE WORK" in field["description"]
    assert "SIZE THE BUDGET TO THE WORK" in tool.description


@pytest.mark.asyncio
async def test_an_hour_long_wait_still_returns_on_settle() -> None:
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "quick", _runner(0.2, "reviewed"))

    started = time.perf_counter()
    result = await execute_wait(
        "wc", {"job_id": job_id, "wait_ms": ONE_HOUR_MS}, None, None, context
    )
    elapsed = time.perf_counter() - started

    assert "reviewed" in result.text
    assert elapsed < 2.0, f"waited {elapsed:.2f}s for a job that took 0.2s"
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_hour_long_wait_still_returns_on_abort() -> None:
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "slow", _runner(30.0))
    signal = AbortSignal()
    asyncio.get_running_loop().call_later(0.1, signal.abort, "user")

    started = time.perf_counter()
    result = await execute_wait(
        "wc", {"job_id": job_id, "wait_ms": ONE_HOUR_MS}, signal, None, context
    )
    elapsed = time.perf_counter() - started

    assert "aborted" in result.text
    assert elapsed < 3.0, f"abort took {elapsed:.2f}s to be noticed"
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_hour_long_wait_still_returns_on_peer_message() -> None:
    class _Peer:
        def __init__(self) -> None:
            self._event = asyncio.Event()
            self._count = 0

        def event(self) -> asyncio.Event:
            return self._event

        def count(self) -> int:
            return self._count

        def mark(self) -> None:
            self._count += 1
            self._event.set()

    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    job_id = manager.register("task", "slow", _runner(30.0))
    asyncio.get_running_loop().call_later(0.1, peer.mark)

    started = time.perf_counter()
    result = await execute_wait(
        "wc", {"job_id": job_id, "wait_ms": ONE_HOUR_MS}, None, None, context
    )
    elapsed = time.perf_counter() - started

    assert result.details is not None
    assert result.details["interrupted_by"] == "peer_message"
    assert elapsed < 3.0, f"peer message took {elapsed:.2f}s to be noticed"
    await manager.dispose()
