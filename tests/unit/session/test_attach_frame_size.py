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
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, Usage
from local_operator.session.frontend_state import (
    USAGE_COMPONENT_CAP,
    _folded_components,
    FrontendSessionState,
    FrontendStateStore,
    JobState,
    filter_update_trajectories,
    oversized_frame_report,
    sync_wire_payload,
)
from local_operator.session.remote import RemoteSession
from local_operator.tui.costs import job_cost
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


# ---------------------------------------------------------------------------
# The CLASS, not the next instance.
#
# Trajectories were the first unbounded per-turn list to overflow this frame.
# ``usage_components`` was the second, and it shipped a release in which the
# reference machine's largest sessions could not be attached to at all. The
# tests below are written against the SHAPE — "an attach frame stays under the
# cap however long the conversation ran and however many children it launched"
# — so a third such field fails CI here rather than in a user's terminal.
# ---------------------------------------------------------------------------


def _identity() -> dict[str, str]:
    """The serving identity folding keys on."""
    return {"provider": "anthropic", "model_id": "claude-opus-4-8-20260101"}


def _priced_spec() -> ModelSpec:
    """A spec the cost table can price, so receipts actually accrue.

    ``accrue_usage`` only appends to ``usage_components`` when it can put a
    number on the call; an unpriceable model records PARTIAL knowledge and
    appends nothing, which would make the cap assertions below pass trivially.
    """
    return ModelSpec(
        provider="anthropic",
        model_id="claude-opus-4-8-20260101",
        display_name="Opus",
        context_window=1_000_000,
        max_output_tokens=64_000,
    )


def _receipt(index: int) -> Usage:
    """One provider receipt, the size the real ones are.

    Real receipts carry the serving identity and a per-call price, which is
    what makes them ~275 bytes each rather than a handful. A tiny filler would
    test the row count instead of the thing that overflows.
    """
    return Usage(
        input_tokens=12_000 + index,
        output_tokens=800 + index,
        cache_read_tokens=9_000,
        cache_write_tokens=1_200,
        context_tokens=180_000 + index,
        usd_cost=0.0123,
        provider="anthropic",
        model_id="claude-opus-4-8-20260101",
    )


def test_a_long_conversations_receipts_are_capped_at_accumulation() -> None:
    """The list must not grow without bound as turns accumulate.

    Asserted on the STORE rather than on a byte size: the property that makes
    the frame fit is that the list stops growing, and a byte assertion would
    pass for the wrong reason the day a receipt gets smaller.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    session = SimpleNamespace(effective_model=_priced_spec())
    for index in range(USAGE_COMPONENT_CAP * 3):
        store.accrue_usage(session, _receipt(index))

    components = store.state.usage_components
    assert len(components) == USAGE_COMPONENT_CAP
    # Newest-wins, oldest evicted — the same discipline AsyncJob.trajectory
    # uses. The most recent receipts are the ones a mixed-provider aggregate
    # needs, so keeping the head instead would keep the useless half.
    assert components[-1].input_tokens == 12_000 + (USAGE_COMPONENT_CAP * 3 - 1)


def test_capping_receipts_cannot_move_a_number_the_ui_paints() -> None:
    """The cap is only safe because the painted figures are running totals.

    ``cumulative_parent_cost`` accrues per call and is never re-derived by
    summing ``usage_components``. If that ever changed, this fails and the cap
    has to grow a running aggregate beside the bounded tail.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    session = SimpleNamespace(effective_model=_priced_spec())
    for index in range(USAGE_COMPONENT_CAP * 2):
        store.accrue_usage(session, _receipt(index))

    state = store.state
    # Every receipt carried a provider-reported price, so the lifetime cost is
    # the full count's worth even though only the tail survives.
    assert len(state.usage_components) == USAGE_COMPONENT_CAP
    assert state.cumulative_parent_cost == pytest.approx(0.0123 * USAGE_COMPONENT_CAP * 2)
    # Occupancy is a LEVEL, not a sum, and comes from the newest receipt.
    assert state.context_tokens == 180_000 + (USAGE_COMPONENT_CAP * 2 - 1)


def test_a_restored_fat_checkpoint_is_capped_on_the_way_in() -> None:
    """Every transcript written before the cap still carries the fat list.

    This is the case the operator's machine is actually in: 4,910 sessions
    whose newest checkpoint holds thousands of receipts. Without capping on
    RESTORE, the first resume of such a session rebuilds the oversized state in
    memory and re-emits exactly the frame that could not be sent.
    """

    class _Transcript:
        def latest_custom(self, _custom_type: str) -> dict[str, Any]:
            fat = FrontendSessionState(
                session_id="s1",
                epoch="old",
                usage_components=[_receipt(index) for index in range(2_685)],
            )
            return {"state": fat.model_dump(mode="json")}

    session = SimpleNamespace(session_id="s1", _transcript=_Transcript())
    store = FrontendStateStore.from_checkpoint(session)

    assert len(store.state.usage_components) == USAGE_COMPONENT_CAP
    frame = {"op": "frontend_sync", "data": sync_wire_payload(store.subscribe(lambda _u: None).sync)}
    assert _line_bytes(frame) < _MAX_LINE_BYTES


def test_the_attach_frame_fits_for_a_session_that_ran_all_year() -> None:
    """The class guard: everything unbounded, at once, still under the cap.

    A long conversation (receipts), a deep roster (jobs, each with retained
    events and its own per-call receipts), a full model catalogue, many todos
    and long names — the union of every field that grows with use. The
    reference machine's real session serialized 1,052,296 bytes here.
    """
    jobs = [
        job.model_copy(
            update={
                # The per-job twin of usage_components: a child's own folded
                # receipts, which is what kept 18 stripped-trajectory jobs at
                # 196 KB on the reference machine.
                "usage": Usage(
                    input_tokens=1_000,
                    cost_components=[_receipt(index) for index in range(400)],
                ),
                "result_text": "r" * 4_000,
                "prompt": "p" * 4_000,
            }
        )
        for job in _jobs(20, 500)
    ]
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        jobs=jobs,
        usage_components=[_receipt(index) for index in range(5_000)],
        child_costs={f"job{index}": 1.25 for index in range(500)},
        todos=[],
        conversation_title="a" * 500,
        goal="g" * 2_000,
        cwd="/" + "d" * 500,
        model_catalogue=[
            {"provider": "p", "model_id": f"model-{index}", "context_window": 200_000}
            for index in range(300)
        ],
    )
    store = FrontendStateStore(state)
    frame = {"op": "frontend_sync", "data": sync_wire_payload(store.subscribe(lambda _u: None).sync)}

    size = _line_bytes(frame)
    assert size < _MAX_LINE_BYTES, (
        f"the attach frame is {size:,} bytes, over the {_MAX_LINE_BYTES:,} limit. "
        "Some field in FrontendSessionState grows without bound and is not "
        "capped at accumulation or stripped in sync_wire_payload. "
        f"{oversized_frame_report(frame, _MAX_LINE_BYTES)}"
    )


def test_folding_a_jobs_receipts_does_not_change_what_it_cost() -> None:
    """The per-job list is priced, so it is folded rather than capped.

    ``job_cost`` sums ``usage.cost_components``: dropping a row there would
    undercount a child's spend, which is a wrong number on screen rather than
    a large frame. Folding by serving identity is lossless because each
    component is priced independently and then summed — verified here against
    the mixed reported/estimated case, which is the one that could drift.

    Measured against the reference machine's real roster while writing this:
    14 jobs, 104 components folding to 1, worst cost difference $0.00.
    """
    reported = [
        Usage(input_tokens=1_000, output_tokens=100, usd_cost=0.25, **_identity())
        for _ in range(40)
    ]
    # No usd_cost: priced from tokens at the model's rate, which only folds
    # correctly if the tokens are summed rather than the prices.
    estimated = [
        Usage(input_tokens=2_000, output_tokens=300, **_identity()) for _ in range(40)
    ]
    usage = Usage(input_tokens=1, cost_components=[*reported, *estimated])
    label = "anthropic/claude-opus-4-8-20260101"

    before = job_cost(SimpleNamespace(usage=usage, model_label=label), default_model_label=label)
    folded = usage.model_copy(update={"cost_components": _folded_components(usage.cost_components)})
    after = job_cost(SimpleNamespace(usage=folded, model_label=label), default_model_label=label)

    assert before is not None
    assert after == pytest.approx(before)
    # Bounded by DISTINCT IDENTITIES (reported and estimated are two buckets),
    # not by call count — which is what makes it survive a deep roster.
    assert len(folded.cost_components) == 2


def test_an_oversized_frame_is_reported_with_the_field_responsible() -> None:
    """The diagnosis, not just the refusal.

    An oversized frame used to present as a slow owner: unreadable line, dead
    pump, 15 s timeout, silent degrade. The report is what makes the next
    occurrence one log line to find instead of a profiling session, so it must
    name the offending field rather than only the size.
    """
    fits = {"op": "frontend_sync", "data": {"snapshot": {"todos": []}}}
    assert oversized_frame_report(fits, _MAX_LINE_BYTES) is None

    huge = {
        "op": "frontend_sync",
        "data": {"snapshot": {"usage_components": ["x" * 40] * 40_000, "cwd": "/tmp"}},
    }
    report = oversized_frame_report(huge, _MAX_LINE_BYTES)
    assert report is not None
    assert "usage_components" in report
    assert "n=40000" in report
    assert "1,048,576" in report


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
