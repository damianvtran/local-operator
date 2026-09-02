"""The ``wait`` budget is sized to the work, not to a poll interval.

The measured problem (this host, 30 h of Anthropic transcripts): the old
300 000 ms ceiling made an agent awaiting CI (8-20 min) or a long subagent
(30-90 min) poll every five minutes — 1 488 wait-only model calls in 877
consecutive chains, 434 of them two or more polls deep. Each poll re-sends the
whole context, and any poll landing after the provider's 5-minute prompt-cache
TTL rewrites the prefix. A 60-minute ceiling lets one call cover the job.

The ceiling only raises the budget; it must not change what wakes the wait.
EVERY early-return is pinned at a full-hour budget here — settle, abort, steer
cancel, and each kind of inbound message (peer, scheduled wake, hub note) — so
a regression that made a long wait sleep to its deadline is caught by a test
that runs in well under a second. The existing suites keep their short-budget
coverage:

- ``test_wait_settles_on_events.py`` — settle, dispose, abort at 300 000 ms.
- ``test_wait_wakes_on_peer_message.py`` — peer message and steer cancel.

The inbound kinds matter because two of them were NOT wake sources before
review round 1 of this change: a scheduled wake landing mid-turn is courtesy
steering (never urgent, never cancels a tool) and a child's ``hub`` note is an
aside thunk, so both inherited the whole budget as their delivery latency.
The session-side half — that each producer actually calls ``mark`` — is
pinned in ``tests/unit/session/test_session_peer.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import ValidationError

from local_operator.harness.comms import HUB_MESSAGE_TYPE
from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.tools.builtin import (
    _ARRIVAL_NOTES,
    WaitParams,
    build_wait_tool,
    execute_wait,
)
from tests.unit.tools.test_wait_wakes_on_peer_message import _Peer

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


def test_wait_schema_advertises_the_sizing_rule_once() -> None:
    """The model reads the JSON schema, not this module: the ceiling, the
    default and the sizing instruction must all reach it through the tool.
    ONCE — the tool description is the one full statement (system.md points
    at it, the agents guide carries the why); a copy in the field description
    ships in the same schema and only re-bills the rule on every turn."""
    tool = build_wait_tool(ToolContext(cwd=".", jobs=AsyncJobManager()))
    assert tool is not None
    field = tool.parameters["properties"]["wait_ms"]
    assert field["maximum"] == ONE_HOUR_MS
    assert field["default"] == 600_000
    assert "SIZE THE BUDGET TO THE WORK" in tool.description
    assert "SIZE THE BUDGET TO THE WORK" not in field["description"]
    assert "3600000" in field["description"]


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


def test_arrival_notes_name_every_producer_kind() -> None:
    """The wording table is keyed by literal strings (importing the constants
    would be a cycle), so a renamed message type would silently fall through
    to the generic "a <kind> message arrived" fallback. Pin the keys."""
    assert set(_ARRIVAL_NOTES) == {
        PEER_MESSAGE_MESSAGE_TYPE,
        WAKE_PROMPT_MESSAGE_TYPE,
        HUB_MESSAGE_TYPE,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "phrase"),
    [
        (PEER_MESSAGE_MESSAGE_TYPE, "another session"),
        (WAKE_PROMPT_MESSAGE_TYPE, "scheduled wake"),
        (HUB_MESSAGE_TYPE, "hub message"),
    ],
)
async def test_an_hour_long_wait_still_returns_on_each_inbound_kind(kind: str, phrase: str) -> None:
    """A peer message, a scheduled wake and a hub note each cut a 60-minute
    wait short in ~0 s, and the model is told WHICH one did — a reminder the
    user scheduled must not read as "a message from another session"."""
    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    job_id = manager.register("task", "slow", _runner(30.0))
    asyncio.get_running_loop().call_later(0.1, peer.mark, kind)

    started = time.perf_counter()
    result = await execute_wait(
        "wc", {"job_id": job_id, "wait_ms": ONE_HOUR_MS}, None, None, context
    )
    elapsed = time.perf_counter() - started

    assert result.details is not None
    assert result.details["interrupted_by"] == kind
    assert result.details["arrivals"] == {kind: 1}
    assert result.details["job_id"] == job_id
    assert phrase in result.text
    assert elapsed < 3.0, f"{kind} took {elapsed:.2f}s to be noticed"
    # Nothing was cancelled or consumed: the job keeps running for the re-wait.
    row = manager.get(job_id)
    assert row is not None and row.status == "running"
    await manager.dispose()


@pytest.mark.asyncio
async def test_several_kinds_at_once_report_inbound_with_a_breakdown() -> None:
    """A wake and a peer message landing in the same park: ``interrupted_by``
    stays the neutral ``inbound`` (no single kind is THE cause) and the
    per-kind breakdown carries both, so the text names each one."""
    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    job_id = manager.register("task", "slow", _runner(30.0))

    def both() -> None:
        peer.mark(WAKE_PROMPT_MESSAGE_TYPE)
        peer.mark(PEER_MESSAGE_MESSAGE_TYPE)

    asyncio.get_running_loop().call_later(0.1, both)
    result = await execute_wait(
        "wc", {"job_id": job_id, "wait_ms": ONE_HOUR_MS}, None, None, context
    )

    assert result.details is not None
    assert result.details["interrupted_by"] == "inbound"
    assert result.details["arrivals"] == {WAKE_PROMPT_MESSAGE_TYPE: 1, PEER_MESSAGE_MESSAGE_TYPE: 1}
    assert "scheduled wake" in result.text and "another session" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_hour_long_wait_still_returns_on_steer_cancel() -> None:
    """`lop send --now` / a typed steer cancels the tool task; the wait must
    absorb it and hand back the still-running payload at the full budget,
    the same as it does at 300 000 ms in the peer-message suite."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=_Peer())
    job_id = manager.register("task", "slow", _runner(30.0))
    # A steer always arrives with a LIVE, non-aborted signal (it rides
    # interruptible_runner); signal=None would be a plain cancel and re-raise.
    task = asyncio.ensure_future(
        execute_wait("wc", {"job_id": job_id, "wait_ms": ONE_HOUR_MS}, AbortSignal(), None, context)
    )
    await asyncio.sleep(0.1)

    started = time.perf_counter()
    task.cancel()  # what interruptible_runner does on a steer
    result = await task
    elapsed = time.perf_counter() - started

    assert "still running" in result.text
    assert result.details is not None
    assert result.details["job_id"] == job_id
    assert result.details["interrupted_by"] == "cancelled"
    assert elapsed < 3.0, f"steer cancel took {elapsed:.2f}s to be absorbed"
    row = manager.get(job_id)
    assert row is not None and row.status == "running"
    await manager.dispose()
