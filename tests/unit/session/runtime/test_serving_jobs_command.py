"""The runtime's ``jobs`` command: one cancel, inside the owning session.

This is the ladder word the desktop's job-cancel route reaches
(``route_shared_slash("jobs", …)``) when a live runtime holds the session, and
the reason it exists is that ``AsyncJobManager`` state — the roster, the abort
signals, the runner tasks — lives only in the process that owns the session:
there is no file or index an external writer could reach, so the cancel has to
run where the manager does.

What this file pins is the contract between that word and the route: it
understands exactly one op (``cancel``), the cancel it runs is the SAME call
``jobs op='cancel'`` makes — ``AsyncJobManager.cancel`` against a REAL
manager, so the abort-signal/task-cancel mechanism is the live implementation
— and every refusal carries a machine-readable code (the route maps the code
to a status; the prose is only for a human). The unknown-id refusal speaks the
frozen ``not_found`` phrase from the engine's wording map, exactly, because
that phrase is what the rail's design renders for it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.serving import ServingSessionHandle

#: The frozen phrase (``tools/image_tool._CANCEL_PHRASES["not_found"]``), pinned
#: as a literal because it is quoted copy: a change to it must fail a test, not
#: silently move the words the rail renders.
FROZEN_NOT_FOUND = "not cancelled — the provider no longer knows the job"


class _Session:
    """The one thing the slash reads off a session: its job manager."""

    def __init__(self, manager: AsyncJobManager | None) -> None:
        self.jobs = manager


def _handle() -> ServingSessionHandle:
    """The handler without its constructor. Reaching the ladder branch is the
    whole test; standing up a socket, a process and a session to reach one
    ``if`` would test the harness instead (the ``_monitor_slash`` suite's own
    shortcut)."""
    return ServingSessionHandle.__new__(ServingSessionHandle)


async def _run(session: Any, payload: dict[str, Any]) -> SlashResult:
    return await _handle()._jobs_slash(session, json.dumps(payload), SlashResult)


def _running_job(manager: AsyncJobManager, observed: list[bool]) -> tuple[str, asyncio.Event]:
    """A synthetic detached job: runs until cancelled, exactly as a video
    runner does between its submit and its settle. The ``finally`` records the
    mechanism the runner's bounded ``best_effort_cancel`` keys on — the abort
    signal fired BEFORE the task dies — and the ``started`` event lets a test
    wait until the runner is actually live, because a cancel that lands before
    the task's first step takes the manager's never-entered teardown path
    instead (a real job is always past that by the time a person can press)."""
    started = asyncio.Event()

    async def runner(job_id: str, signal: Any, progress: Any) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            observed.append(signal.aborted)

    return manager.register("bash", "synthetic job", runner), started


def _row(manager: AsyncJobManager, job_id: str) -> Any:
    """The job's row, asserted present — ``get`` is optional by signature."""
    job = manager.get(job_id)
    assert job is not None, f"job {job_id} is not on the roster"
    return job


async def _settle(manager: AsyncJobManager, job_id: str) -> None:
    for _ in range(200):
        if _row(manager, job_id).status != "running":
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the synthetic job did not settle")


@pytest.mark.asyncio
async def test_cancel_reaches_the_manager_and_aborts_the_signal() -> None:
    manager = AsyncJobManager()
    observed: list[bool] = []
    job_id, started = _running_job(manager, observed)
    await asyncio.wait_for(started.wait(), timeout=5.0)
    session = _Session(manager)

    result = await _run(session, {"op": "cancel", "job_id": job_id})

    assert result.kind == "notice"
    assert result.data == {"job_id": job_id, "cancelled": True}
    # The receipt's own words: the tool's sentence for the same act.
    assert result.text == f"cancelled job {job_id} (synthetic job)"
    # The manager row settled cancelled, and the runner saw the abort — the
    # exact pair ``AsyncJobManager.cancel`` produces on the tool path.
    assert _row(manager, job_id).status == "cancelled"
    assert observed == [True]


@pytest.mark.asyncio
async def test_cancel_is_the_same_manager_call_the_tool_makes() -> None:
    """Manager parity: the word must reach ``AsyncJobManager.cancel`` — the
    call ``jobs op='cancel'`` makes — and nothing else on the manager."""
    manager = AsyncJobManager()
    observed: list[bool] = []
    job_id, started = _running_job(manager, observed)
    await asyncio.wait_for(started.wait(), timeout=5.0)
    session = _Session(manager)
    calls: list[str] = []
    original = manager.cancel

    async def spy(cancelled_id: str, **kwargs: Any) -> bool:
        calls.append(cancelled_id)
        return await original(cancelled_id, **kwargs)

    manager.cancel = spy  # type: ignore[method-assign]

    result = await _run(session, {"op": "cancel", "job_id": job_id})

    assert result.kind == "notice"
    assert calls == [job_id]


@pytest.mark.asyncio
async def test_an_unknown_job_speaks_the_frozen_not_found_phrase() -> None:
    manager = AsyncJobManager()
    session = _Session(manager)

    result = await _run(session, {"op": "cancel", "job_id": "nosuchjob000"})

    assert result.kind == "error"
    assert result.data["code"] == "jobs_not_found"
    assert result.text == FROZEN_NOT_FOUND


@pytest.mark.asyncio
async def test_a_settled_job_is_refused_in_the_tools_own_words() -> None:
    """The race and the retry: the row is there, but not running. The press
    asked for a stop that cannot happen and must not be painted as one — the
    tool's own sentence, so both callers report one thing."""
    manager = AsyncJobManager()

    async def runner(job_id: str, signal: Any, progress: Any) -> str:
        return "done"

    job_id = manager.register("bash", "quick", runner)
    await _settle(manager, job_id)
    session = _Session(manager)

    result = await _run(session, {"op": "cancel", "job_id": job_id})

    assert _row(manager, job_id).status == "completed"
    assert result.kind == "error"
    assert result.data["code"] == "jobs_not_cancelled"
    assert result.text == f"job {job_id} was not cancelled (status: completed)"


@pytest.mark.asyncio
async def test_a_session_without_a_manager_is_jobs_unavailable() -> None:
    result = await _run(_Session(None), {"op": "cancel", "job_id": "abc"})

    assert result.kind == "error"
    assert result.data["code"] == "jobs_unavailable"
    assert "no job manager attached" in result.text


@pytest.mark.asyncio
async def test_malformed_payloads_and_unknown_ops_are_jobs_invalid() -> None:
    manager = AsyncJobManager()
    session = _Session(manager)

    unreadable = await _handle()._jobs_slash(session, "not json", SlashResult)
    unknown_op = await _run(session, {"op": "peek", "job_id": "abc"})

    assert unreadable.kind == "error" and unreadable.data["code"] == "jobs_invalid"
    assert unknown_op.kind == "error" and unknown_op.data["code"] == "jobs_invalid"
