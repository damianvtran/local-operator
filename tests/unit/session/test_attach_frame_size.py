"""Attach frames must fit the control socket, however busy the session is.

The runtime's reader drops any line past ``server._MAX_LINE_BYTES`` (1 MiB),
and a job's retained trajectory is bounded in COUNT (``TRAJECTORY_CAP`` = 500
events) but not in BYTES — each event holds a whole tool result. Ten children
at the cap serialize to ~3.1 MB, so before trajectories were taken out of the
snapshot the first frame of a busy session could not be sent at all and the
session simply could not be attached to: 12 of 17 sessions on the reference
machine failed exactly this way.

These are hard size assertions rather than "it worked" assertions, because the
failure they guard is silent — an oversized frame is a dropped line, not an
error anybody reports.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    JobState,
    filter_update_trajectories,
    sync_wire_payload,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import _MAX_LINE_BYTES, RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle

#: A tool result big enough to be realistic. The point of the cap is that ONE
#: event carries an unbounded payload, so a small filler would test the row
#: count rather than the thing that overflows.
_RESULT_CHARS = 400


def _event(index: int) -> dict[str, Any]:
    return {
        "type": "tool_execution_end",
        "generation": 1,
        "tool_call_id": f"call_{index:06d}",
        "tool_name": "bash",
        "intent": "Checking something moderately descriptive here",
        "result": {"content": [{"type": "text", "text": "x" * _RESULT_CHARS}]},
        "_traj_seq": index,
    }


def _jobs(count: int, rows: int, *, start: int = 0) -> list[JobState]:
    return [
        JobState(
            id=f"job{index}",
            type="task",
            label=f"child {index}",
            status="running",
            trajectory=[_event(row) for row in range(start, start + rows)],
        )
        for index in range(count)
    ]


def _line_bytes(frame: dict[str, Any]) -> int:
    """Exactly what the socket writes: one JSON line plus its delimiter."""
    return len(json.dumps(frame).encode()) + 1


async def _record(root: Path):  # noqa: ANN202
    for _ in range(100):
        rows = registry.scan(root)
        if rows and rows[0][1] == "live":
            return rows[0][0]
        await asyncio.sleep(0.02)
    raise AssertionError("record did not publish")


async def _never():
    raise AssertionError("takeover was not expected")


def test_ten_jobs_at_the_cap_overflow_the_line_limit_without_the_fix() -> None:
    """The regression this guards is real, not hypothetical.

    Asserting the UNFIXED size keeps the other tests meaningful: if a future
    change made trajectories small enough to fit anyway, the fix's own tests
    would pass for the wrong reason and this one would fail loudly instead.
    """
    store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", jobs=_jobs(10, 500))
    )
    subscription = store.subscribe(lambda _update: None)
    naive = _line_bytes({"op": "frontend_sync", "data": subscription.sync.model_dump(mode="json")})
    assert naive > _MAX_LINE_BYTES, (
        "the fixture no longer reproduces the oversized frame; "
        f"{naive} bytes is under the {_MAX_LINE_BYTES} limit"
    )


def test_sync_for_ten_jobs_at_the_cap_fits_the_line_limit() -> None:
    store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", jobs=_jobs(10, 500))
    )
    subscription = store.subscribe(lambda _update: None)
    frame = {"op": "frontend_sync", "data": sync_wire_payload(subscription.sync)}
    assert _line_bytes(frame) < _MAX_LINE_BYTES

    jobs = frame["data"]["snapshot"]["jobs"]
    assert len(jobs) == 10
    # The rows are gone but the COUNT survives, which is what lets the viewer
    # say "loading 500 events" instead of rendering the child as empty.
    assert all(job["trajectory"] == [] for job in jobs)
    assert [job["trajectory_length"] for job in jobs] == [500] * 10


def test_delta_burst_across_unwatched_jobs_fits_the_line_limit() -> None:
    """The snapshot is only half the budget; a mid-turn burst is the other.

    Ten children each appending 200 events in one tick overflows the limit as
    surely as the snapshot did, and a viewer reading ONE child's page must not
    pay for the other nine.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1", jobs=_jobs(10, 0)))
    store.subscribe(lambda _update: None)
    update = store.mutate(jobs=_jobs(10, 200))
    assert update is not None
    payload = update.model_dump(mode="json")

    unfiltered = _line_bytes({"op": "frontend_update", "data": payload})
    assert unfiltered > _MAX_LINE_BYTES, "the burst fixture no longer overflows"

    watching_one = filter_update_trajectories(payload, {"job3"}.__contains__)
    assert _line_bytes({"op": "frontend_update", "data": watching_one}) < _MAX_LINE_BYTES
    assert list(watching_one["job_trajectory_appends"]) == ["job3"]

    watching_none = filter_update_trajectories(payload, lambda _job_id: False)
    assert _line_bytes({"op": "frontend_update", "data": watching_none}) < _MAX_LINE_BYTES
    assert watching_none["job_trajectory_appends"] == {}
    # Row counts still ride the roster, so a page opened later knows what to
    # fetch rather than resuming from a hole.
    assert [job["trajectory_length"] for job in watching_none["changes"]["jobs"]] == [200] * 10


def test_unfiltered_update_is_returned_unchanged_when_nothing_needs_dropping() -> None:
    """The common delta carries no trajectories and must not pay for a copy."""
    payload = {"epoch": "e1", "sequence": 3, "changes": {"streaming": True}}
    assert filter_update_trajectories(payload, lambda _job_id: False) is payload


@pytest.mark.asyncio
async def test_attach_succeeds_against_a_session_that_exceeded_the_old_limit(
    tmp_path: Path, monkeypatch
) -> None:
    """The end-to-end claim: this session could not be attached to before.

    Drives the real ``RuntimeServer`` over a real socket with a roster whose
    naive snapshot is ~3 MB, then fetches one child's window on demand and
    checks the rows arrive intact.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    handle._frontend.mutate(jobs=_jobs(10, 500))
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        # Attach itself is the assertion: an oversized frame never arrives, so
        # before the fix this connect timed out waiting for the sync.
        assert len(remote.frontend_state.jobs) == 10
        job = remote.jobs.get("job4")
        assert job is not None
        assert list(job.trajectory) == []
        assert job.trajectory_length == 500

        assert await remote.load_job_trajectory("job4") is True
        loaded = remote.jobs.get("job4")
        assert loaded is not None
        assert len(loaded.trajectory) == 500
        assert loaded.trajectory[0]["tool_call_id"] == "call_000000"
        assert loaded.trajectory[-1]["tool_call_id"] == "call_000499"
        # One page cannot carry 500 rows; the loader must have paged.
        pages = [call for call in handle.calls if call[0] == "job_trajectory"]
        assert len(pages) > 1
        # Unopened children stay unfetched: the whole point is that a viewer
        # pays for the page it is reading and nothing else.
        unopened = remote.jobs.get("job5")
        assert unopened is not None
        assert list(unopened.trajectory) == []
        assert unopened.trajectory_length == 500
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_live_appends_reach_only_the_watched_job(tmp_path: Path, monkeypatch) -> None:
    """``watch_job`` is what makes the delta stream affordable."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    handle._frontend.mutate(jobs=_jobs(2, 1))
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        assert await remote.load_job_trajectory("job0") is True

        handle._frontend.mutate(jobs=_jobs(2, 4))
        for _ in range(100):
            watched = remote.jobs.get("job0")
            if watched is not None and len(watched.trajectory) == 4:
                break
            await asyncio.sleep(0.02)

        watched = remote.jobs.get("job0")
        assert watched is not None
        assert len(watched.trajectory) == 4, "appends for the open page must arrive"
        unwatched = remote.jobs.get("job1")
        assert unwatched is not None
        assert list(unwatched.trajectory) == [], "an unopened page must cost nothing"
        # The count still tells the truth for the job nobody is watching.
        assert unwatched.trajectory_length == 4
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()
