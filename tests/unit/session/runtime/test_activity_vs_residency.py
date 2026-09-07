"""The record's activity bit is about the CONVERSATION, not about residency.

**This file exists because a session that had finished talking still wore a
spinner in the sidebar.** The reported symptom was a "done" conversation
("Article-search-svc…") that kept showing the working indicator; the measured
cause was that `OwnedSessionHandle.is_busy()` — the *reaper's* residency
predicate — was published verbatim as `SessionRecord.busy`, which every
surface renders as "a turn is running".

Those are two different questions and they were answered by one predicate:

* **Residency** ("may this runtime exit?") must be MAXIMALLY inclusive. A
  backgrounded `bash` job, a still-running subagent, a queued prompt and a
  parked gate all forbid exit, because exiting would destroy work. Fail-closed
  is correct there: an over-broad residency answer costs an idle process.
* **Activity** ("is this conversation working right now?") is what a spinner
  claims. A user who reads a spinner expects tokens to be moving. A detached
  background job the user themself launched with `background=true` is
  deliberately outliving its turn, so it is exactly the thing that must NOT
  animate a row.

Measured on this host before the fix: 8 of 8 live sessions published
`busy=True`, including one idle for 25.6 minutes whose transcript's last entry
was a completed turn — because a background `bash` job and a stale subagent row
each held `is_busy()` True forever. The bit was not stale; it was answering the
wrong question, so no amount of republishing could have fixed it.

The split introduced here is `is_conversationally_active()`: the terms of
`is_busy()` that mean "the conversation itself is moving", with the
work-retention terms (background jobs, subagents, background tasks) removed.
`is_busy()` is untouched, because the reaper's fail-closed behaviour is
correct and this must not make a runtime exit under live work.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from tests.e2e.harness import ScriptedStream, build_session, text_turn


async def _rig(directory: Path, replies: int = 2) -> tuple[Any, OwnedSessionHandle, RuntimeServer]:
    """A real Session under the production handle and server.

    Same rig as ``test_busy_settles``: the seam under test is the one
    production runs, so the handle and the server are the real classes and only
    the provider stream is scripted.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stream = ScriptedStream([text_turn(f"reply {i}") for i in range(replies)])
    session = build_session(directory, stream)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    handle.subscribe(server._schedule_push)
    return session, handle, server


async def _running_job(session: Any, job_type: str) -> str:
    """Register a REAL job on the session's own manager and leave it running.

    Deliberately not a manager double. The predicate under test reads job rows
    through ``session.jobs.list()``, and a double would let the test agree with
    an implementation that production data never produces — the shape of a
    running row (``status``, ``type``, ``queued``) is exactly what is being
    asserted about. The runner parks on an Event the caller never sets, which
    is what a backgrounded `bash` job or a live subagent looks like to the
    manager: admitted, started, and not finished.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def run(job_id: str, signal: Any, emit: Any) -> str:
        # The manager's ``JobRunFn`` contract: (job_id, abort signal, emit).
        started.set()
        await release.wait()
        return "done"

    job_id = session.jobs.register(job_type, f"{job_type}-job", run)
    await asyncio.wait_for(started.wait(), 5.0)
    _RELEASES.append(release)
    return job_id


#: Every parked runner, released in teardown so no test leaves a pending task
#: behind for the next one to trip over.
_RELEASES: list[asyncio.Event] = []


@pytest.fixture(autouse=True)
def _release_parked_jobs():
    _RELEASES.clear()
    yield
    for event in _RELEASES:
        event.set()
    _RELEASES.clear()


@pytest.mark.asyncio
async def test_a_background_job_holds_residency_but_is_not_conversation_activity(
    tmp_path: Path,
) -> None:
    """THE REPORTED BUG, as an assertion.

    A backgrounded `bash` job is running; the conversation has said everything
    it is going to say. The runtime must stay resident (the job's output would
    be destroyed by an exit) and the row must NOT animate.

    On the unfixed tree ``is_conversationally_active`` does not exist and the
    published bit is ``is_busy()``, which is True here.
    """
    session, handle, server = await _rig(tmp_path / "sess")
    try:
        await _running_job(session, "bash")
        assert handle.is_busy() is True, "a running job must still pin residency"
        assert handle.is_conversationally_active() is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_running_subagent_holds_residency_but_is_not_conversation_activity(
    tmp_path: Path,
) -> None:
    """Delegated work is the same case as a background job.

    A `task` subagent outlives the turn that launched it by design — the parent
    is explicitly free to keep talking, or to say nothing at all. Measured on
    this host: three live sessions carried a `running` roster row with the
    parent idle for 8-25 minutes, and all three wore the spinner.
    """
    session, handle, server = await _rig(tmp_path / "sess")
    try:
        await _running_job(session, "task")
        assert handle.is_busy() is True
        assert handle.is_conversationally_active() is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_live_turn_is_conversation_activity(tmp_path: Path) -> None:
    """The signal must still be TRUE for the case the spinner exists for.

    The guard against fixing the false positive by making the bit always False:
    while a turn streams, both predicates hold.
    """
    session, handle, server = await _rig(tmp_path / "sess")
    seen: list[bool] = []

    def on_event(event: Any) -> None:
        # Sampled from INSIDE the turn: after it settles the predicate is False
        # again, so a sample taken afterwards would prove nothing. Guarded
        # because the session logs and swallows a raising handler, which would
        # otherwise turn "the predicate does not exist yet" into the much less
        # informative "no events were produced".
        probe = getattr(handle, "is_conversationally_active", None)
        seen.append(bool(probe()) if callable(probe) else False)

    session.subscribe(on_event)
    try:
        await handle.prompt("hello", wait_complete=True)
        assert seen, "the turn produced no events to sample"
        assert any(seen), "a streaming turn must read as conversation activity"
        # And it must SETTLE: a predicate stuck True is the bug in the other
        # direction, and is exactly what shipped.
        await asyncio.sleep(0)
        assert handle.is_conversationally_active() is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_parked_gate_is_conversation_activity(tmp_path: Path) -> None:
    """A gate waiting on a person belongs to a turn that is mid-flight.

    It renders as the needs-you marker rather than the spinner (that
    precedence lives in ``row_state_mark``), but the underlying turn IS
    running: the tool slot is held and the conversation is not finished. So
    the activity predicate says True and the marker layer decides what to draw.
    """
    session, handle, server = await _rig(tmp_path / "sess")
    try:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        handle._pending_futures["req-1"] = future
        assert handle.is_conversationally_active() is True
    finally:
        handle._pending_futures.clear()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_published_record_bit_follows_activity_not_residency(
    tmp_path: Path,
) -> None:
    """End to end: what a background job does to the RECORD the sidebar reads.

    ``_publish_busy`` is the one seam between the handle and the record, so
    this asserts the thing the user actually sees rather than the predicate
    behind it.
    """
    session, handle, server = await _rig(tmp_path / "sess")
    try:
        await _running_job(session, "bash")
        handle._publish_busy()
        assert server._busy is False, "an idle conversation must not publish busy"
        assert handle.is_busy() is True, "residency must be unaffected by the split"
    finally:
        await session.dispose()
