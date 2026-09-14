"""A roster tick must cost the DELTA, not the retained window.

WHY THIS FILE EXISTS. ``Session._schedule_frontend_jobs`` coalesces roster churn
onto ``FrontendStateStore.refresh_jobs`` every 50 ms while a turn runs, and that
tick used to rebuild every job's whole retained window: ``JobState.from_job``
copied every row, ``_freeze_job`` rebuilt every row's frozen containers, and
``_jobs_equal`` then deep-compared them -- ~50-57 ms per tick with ~87 GC
collections on a 5-child roster at the 500-row cap, re-measured at 88 ms/tick on
this host (``docs/evidence/frame-cost-loop-starvation/``). That is more than one
core demanded at 20 Hz, on the loop that also drives the record's heartbeat, which
is why a busy session read as wedged everywhere.

Every assertion here is STRUCTURAL -- what ran, not how long it took -- per
AGENTS.md's "Timing, flakes" section: copies and freezes counted through spies,
and object identity where the point is that one object was reused rather than an
equal one rebuilt. The cache's soundness rests on the writer's invariant
(:func:`test_the_relay_writer_only_appends_and_trims_the_front`) plus a
fingerprint, so each way the fingerprint can be fooled has a test that the memo
REFUSES to reuse rather than one that it reuses.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY, AsyncJob
from local_operator.harness.subagent import TRAJECTORY_CAP, _make_relay
from local_operator.harness.types import (
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.session import frontend_state as module
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    FrontendSync,
    FrontendUsage,
    JobState,
    sync_wire_payload,
)

ROWS = 8


def _row(seq: int, *, stamped: bool = True, text: str = "x" * 64) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "tool_execution_end",
        "tool_call_id": f"call_{seq}",
        "tool_name": "bash",
        "result": {"content": [{"type": "text", "text": text}]},
    }
    if stamped:
        row[TRAJECTORY_SEQ_KEY] = seq
    return row


def _rows(count: int, *, text: str = "x" * 64) -> list[dict[str, Any]]:
    return [_row(seq, text=text) for seq in range(count)]


def _job(job_id: str, rows: list[dict[str, Any]] | None = None) -> AsyncJob:
    return AsyncJob(
        id=job_id,
        type="task",
        label=job_id,
        status="running",
        start_time=1_700_000_000.0,
        trajectory=_rows(ROWS) if rows is None else rows,
        latest_details={"progress": "thinking"},
        prompt="the child's launch prompt",
    )


def _trajectory(job: AsyncJob) -> list[dict[str, Any]]:
    """The job's retained rows, asserted present: ``None`` means "none recorded"."""
    rows = job.trajectory
    assert rows is not None
    return rows


def _store(jobs: list[AsyncJob]) -> FrontendStateStore:
    """A store seeded the way a session seeds it: from the same live rows."""
    return FrontendStateStore(
        FrontendSessionState(
            session_id="frame-cost",
            epoch="e1",
            jobs=[JobState.from_job(job) for job in jobs],
        )
    )


def _session(jobs: list[AsyncJob]) -> Any:
    """The minimum a roster refresh reads: a job manager and nothing else."""
    return SimpleNamespace(
        jobs=SimpleNamespace(
            list=lambda: list(jobs), accounting_components=lambda: [], get=lambda _id: None
        ),
        model=None,
    )


def _is_row_shaped(value: Any) -> bool:
    """True for a retained trajectory ROW rather than any other mapping.

    Keyed on ``tool_call_id``, which every relayed row carries and no nested value
    does: a row's own ``result.content`` blocks are dicts too, so keying on
    ``type`` would count each row twice. Read through ``keys()`` rather than
    ``in``, because a frozen row is a tuple subclass whose membership test walks
    its (key, value) PAIRS -- ``"tool_call_id" in frozen_row`` is False, and a
    spy that used it would count nothing on exactly the rows that matter.
    """
    if not isinstance(value, Mapping):
        return False
    return any(key == "tool_call_id" for key in value.keys())


class _RowWork:
    """Counts the per-ROW work one or more ticks do.

    The bug is not a duration: it is that an idle tick copies and freezes every
    retained row to conclude nothing changed. Counting that is both flake-free and
    precise about the claim.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.copies = 0
        self.freezes = 0
        original_deepcopy = copy.deepcopy
        original_freeze_value = module._freeze_value

        def counted_deepcopy(value: Any, *args: Any, **kwargs: Any) -> Any:
            if _is_row_shaped(value):
                self.copies += 1
            return original_deepcopy(value, *args, **kwargs)

        def counted_freeze_value(value: Any) -> Any:
            if _is_row_shaped(value):
                self.freezes += 1
            return original_freeze_value(value)

        monkeypatch.setattr(copy, "deepcopy", counted_deepcopy)
        monkeypatch.setattr(module, "_freeze_value", counted_freeze_value)

    def reset(self) -> None:
        self.copies = 0
        self.freezes = 0


def _warm(store: FrontendStateStore, session: Any, monkeypatch: pytest.MonkeyPatch) -> _RowWork:
    """One tick to populate the memo, then a spy scoped to what follows."""
    store.refresh_jobs(session)
    return _RowWork(monkeypatch)


def test_an_unchanged_roster_tick_copies_and_freezes_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect: an idle tick was O(retained window) to publish nothing."""
    jobs = [_job(f"child-{index}") for index in range(3)]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    assert store.refresh_jobs(session) is None, "an unchanged roster published a frame"
    assert (work.copies, work.freezes) == (0, 0), "an idle tick still copied retained rows"

    # And the state it concluded was unchanged is still the same window, by
    # identity: nothing was rebuilt into an equal copy.
    assert store._state.jobs[0].trajectory == jobs[0].trajectory


def test_a_tick_that_appended_one_row_freezes_exactly_that_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [_job(f"child-{index}") for index in range(3)]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    appended = _row(ROWS)
    _trajectory(jobs[0]).append(appended)
    update = store.refresh_jobs(session)

    assert update is not None and update.job_trajectory_appends == {jobs[0].id: [appended]}
    assert work.freezes == 1, "an appended row did not cost exactly one row of freezing"
    assert work.copies == 0, "the appended row was copied as well as frozen"
    assert len(store._state.jobs[0].trajectory) == ROWS + 1


def test_the_retained_window_is_the_same_object_across_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity, not equality: a rebuilt-but-equal window is the cost being removed."""
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    _warm(store, session, monkeypatch)
    first = store._state.jobs[0].trajectory

    assert store.refresh_jobs(session) is None
    assert store._state.jobs[0].trajectory is first, "the window was rebuilt into an equal copy"
    # The memo must hand back the object canonical state HOLDS, not its own equal
    # copy of it: ``mutate`` keeps the state's shells when a refresh proves them
    # unchanged, so a memo pointing anywhere else makes the next tick re-prove a
    # whole window equal instead of recognising it by identity.
    assert store._retained_windows()._by_job["child-0"].rows is first

    _trajectory(jobs[0]).append(_row(ROWS))
    update = store.refresh_jobs(session)

    assert update is not None
    assert update.job_trajectory_appends == {"child-0": [_trajectory(jobs[0])[-1]]}
    assert len(store._state.jobs[0].trajectory) == ROWS + 1
    assert store._state.jobs[0].trajectory[ROWS] == _trajectory(jobs[0])[-1]


def test_cap_rotation_ships_exactly_one_replacement() -> None:
    """Past the cap the window rotates; the memo must not turn that into appends."""
    rows = _rows(TRAJECTORY_CAP)
    jobs = [_job("child-0", rows)]
    session = _session(jobs)
    store = _store(jobs)
    store.refresh_jobs(session)

    _trajectory(jobs[0]).append(_row(TRAJECTORY_CAP))
    del _trajectory(jobs[0])[:1]
    update = store.refresh_jobs(session)

    assert update is not None
    assert update.job_trajectory_replacements == ["child-0"]
    # One replacement, and the front really moved: the window is the NEWEST cap
    # rows. (The delta also carries the rotated rows as appends -- pre-existing
    # behaviour of this reducer, unchanged here.)
    assert list(store._state.jobs[0].trajectory) == jobs[0].trajectory
    assert store._state.jobs[0].trajectory_length == TRAJECTORY_CAP
    assert store._state.jobs[0].trajectory[0][TRAJECTORY_SEQ_KEY] == 1


def test_an_unstamped_row_drops_the_memo_instead_of_reusing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row from an older release carries no stamp, so nothing can match on it."""
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    _trajectory(jobs[0]).append(_row(ROWS, stamped=False))
    store.refresh_jobs(session)
    assert work.freezes == ROWS + 1, "a window with an unstamped row was partly reused"

    work.reset()
    store.refresh_jobs(session)
    assert work.freezes == ROWS + 1, "an unstamped window was cached and reused"
    assert list(store._state.jobs[0].trajectory) == jobs[0].trajectory


def test_a_rebound_row_list_is_not_matched_to_the_previous_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``subagent.runner`` replaces ``job.trajectory`` and the stamps restart at 0.

    This is the case a fingerprint over (count, first stamp, last stamp) alone
    cannot see: a second attempt at the same job id can present exactly those
    numbers while its rows are different. The row LIST identity is what separates
    them, and the shipped rows must be the new attempt's.
    """
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    # The runner's rebind, then a fresh attempt: same ids, same stamps, different work.
    jobs[0].trajectory = []
    store.refresh_jobs(session)
    jobs[0].trajectory = [_row(seq, text=f"second attempt {seq}") for seq in range(ROWS)]
    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == ROWS, "a rebound row list was reused instead of frozen"
    assert store._state.jobs[0].trajectory == jobs[0].trajectory
    assert store._state.jobs[0].trajectory[0]["result"]["content"][0]["text"] == (
        "second attempt 0"
    )


def test_a_reordered_row_list_is_not_matched_to_the_previous_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same rows, same stamps, same first/last, new list: the shape only LIST identity catches.

    A fingerprint over the stamps would accept this: the count, the first and last
    stamps and even the tail row object are unchanged, because the rows were
    reordered rather than replaced. Nothing in the writer does this today; the
    check exists so that a rebuild which happens to agree on both ends cannot be
    mistaken for an unchanged window.
    """
    jobs = [_job("child-0", _rows(4))]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    reordered = [_trajectory(jobs[0])[0], _trajectory(jobs[0])[2], _trajectory(jobs[0])[1]]
    reordered.append(_trajectory(jobs[0])[3])
    jobs[0].trajectory = reordered
    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == 4, "a reordered window was reused rather than frozen"
    assert list(store._state.jobs[0].trajectory) == reordered


def test_a_row_list_refilled_in_place_is_not_matched_to_its_previous_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same LIST, emptied and refilled: count, first and last stamps all agree.

    The rows are new objects, so the tail object is what separates the two
    windows -- the guard a fingerprint over stamps alone would leave open.
    """
    jobs = [_job("child-0", _rows(4))]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    rows = _trajectory(jobs[0])
    rows.clear()
    rows.extend(_row(seq, text=f"second attempt {seq}") for seq in range(4))
    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == 4, "a refilled window was reused rather than frozen"
    assert list(store._state.jobs[0].trajectory) == rows


def test_a_rewritten_front_row_is_not_matched_to_the_previous_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The front of the window moved without the window growing or shrinking.

    A front trim takes rows off the front and the first row's stamp moves forward
    by exactly as many rows as were evicted; a first row that is somewhere else
    entirely is not that, and there is nothing to reconstruct from.
    """
    jobs = [_job("child-0", _rows(4))]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    _trajectory(jobs[0])[0] = _row(99, text="a front row from nowhere")
    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == 4, "a rewritten front was reused rather than frozen"
    assert list(store._state.jobs[0].trajectory) == jobs[0].trajectory


def test_a_job_that_left_the_roster_stops_being_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """The memo holds the raw row list, so it may not outlive the job."""
    jobs = [_job("child-0"), _job("child-1")]
    session = _session(jobs)
    store = _store(jobs)
    _warm(store, session, monkeypatch)

    del jobs[0]
    store.refresh_jobs(session)

    assert "child-0" not in store._retained_windows()._by_job
    assert "child-1" in store._retained_windows()._by_job


def test_an_initial_refresh_drops_the_memo_before_it_reads_the_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full rescan is where rows are rebuilt rather than appended.

    Asserted on the memo the refresh actually saw, not on the tick after it: the
    refresh re-reads the rows it is dropping the memo for, so it ends with a
    populated memo either way. What the invalidation buys is that the payload it
    installs is built from rows frozen for THIS scan.
    """
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    _warm(store, session, monkeypatch)
    assert store._retained_windows()._by_job, "the fixture is supposed to have a memo"

    entries_seen: list[int] = []
    original_window = module._TrajectoryWindows.window

    def recording_window(self: Any, job_id: str, rows: Any) -> Any:
        entries_seen.append(len(self._by_job))
        return original_window(self, job_id, rows)

    monkeypatch.setattr(module._TrajectoryWindows, "window", recording_window)
    store.refresh_from_session(session, initial=True)

    assert entries_seen, "the initial refresh never read the rows"
    assert entries_seen[0] == 0, "the initial refresh read its rows against a live memo"


def test_replace_starts_from_no_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """``replace()`` re-seats canonical state from a payload, same reason."""
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    store.replace(store.state)
    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == ROWS, "the memo survived a replace()"


def _rich_sync() -> FrontendSync:
    """A snapshot with every field the wire builder bounds: rows, todos, usage.

    Built through a real store, so the job rows carry the frozen wrappers a
    runtime's canonical state carries -- a plain-list fixture would serialise
    without walking a single row and measure nothing.
    """
    jobs = [
        JobState(
            id=f"child-{index}",
            type="task",
            status="running",
            label=f"child {index}",
            agent="coder",
            intent="pin the event-loop hotspot",
            trajectory=_rows(3),
            trajectory_length=3,
            todos=[{"name": "Plan", "items": [{"text": "step", "status": "pending"}]}],
            prompt="prompt " * 20,
            result_text="result " * 20,
            model_label="deepseek/deepseek-flash",
            start_time=1_700_000_000.0,
            usage=FrontendUsage(
                input_tokens=10,
                output_tokens=5,
                context_tokens=100,
                usd_cost=0.1,
                cost_components=[FrontendUsage(input_tokens=1, usd_cost=0.01)],
            ),
        )
        for index in range(2)
    ]
    snapshot = FrontendStateStore(
        FrontendSessionState(
            session_id="frame-cost-sync",
            epoch="e1",
            sequence=3,
            cwd="/tmp",
            conversation_title="frame cost",
            usage_components=[FrontendUsage(input_tokens=7, usd_cost=0.5)],
            live_events=[
                {"type": "tool_execution_start", "tool_call_id": "c1", "tool_name": "bash"},
            ],
            jobs=jobs,
            model_catalogue=[{"name": "deepseek/deepseek-flash", "context_window": 1000}],
        )
    ).state
    return FrontendSync(epoch="e1", sequence=3, snapshot=snapshot)


def test_sync_wire_payload_bytes_are_identical_when_the_rows_leave_the_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retained rows are omitted from the THAW; not one byte of the frame may move.

    The whole point of doing this in the serializer instead of in the dumped
    result is that a job row keeps its keys and their order, so this is an A/B
    against the pre-change path: the same function with the dump's serialization
    context dropped is exactly what it did before, since everything after that
    dump is unchanged.
    """
    sync = _rich_sync()
    after = json.dumps(sync_wire_payload(sync))

    original_dump = FrontendSync.model_dump

    def dump_without_context(self: FrontendSync, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("context", None)
        return original_dump(self, *args, **kwargs)

    monkeypatch.setattr(FrontendSync, "model_dump", dump_without_context)
    before = json.dumps(sync_wire_payload(sync))

    assert after == before, "the sync frame's bytes changed"
    # And the rows really are gone, so the equality is not the trivially-empty case.
    payload = json.loads(after)
    assert [job["trajectory"] for job in payload["snapshot"]["jobs"]] == [[], []]
    assert [job["trajectory_length"] for job in payload["snapshot"]["jobs"]] == [3, 3]


def test_sync_wire_payload_does_not_thaw_the_retained_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost half of the byte-identity gate above: the rows are never touched.

    The bytes not moving is only half the claim -- the other half is that the
    frame stops walking ~100 MiB of retained tool results to ship 16 KiB of
    roster. ``_wire_value`` is the single funnel each retained row went through
    on the way to JSON, so counting the row-shaped calls to it counts exactly the
    work the omission removes.
    """
    sync = _rich_sync()
    thawed = 0
    original_wire_value = module._wire_value

    def counted_wire_value(value: Any) -> Any:
        nonlocal thawed
        if _is_row_shaped(value):
            thawed += 1
        return original_wire_value(value)

    monkeypatch.setattr(module, "_wire_value", counted_wire_value)
    payload = sync_wire_payload(sync)

    assert thawed == 0, "the sync frame still serialized its retained rows"
    assert payload["snapshot"]["jobs"][0]["trajectory"] == []


@pytest.mark.asyncio
async def test_the_relay_writer_only_appends_and_trims_the_front() -> None:
    """The ONE behaviour no fingerprint can catch is a row revised in place.

    The cache therefore rests on the writer: ``_make_relay`` stamps a fresh row
    per event, appends it, never revises a retained row, and trims only from the
    front past the cap. This drives the real relay past the cap and asserts each
    of those facts on the row objects themselves, so a future change that starts
    rewriting rows fails here instead of shipping stale rows to a viewer.
    """
    job = _job("child-0", [])
    relay = _make_relay(
        "child-0",
        "child-0",
        job,
        cast(Any, SimpleNamespace(_notify_roster_change=lambda: None)),
        _noop_emit,
        lambda _progress: None,
        {},
    )

    await relay(ToolExecutionStartEvent(tool_call_id="c0", tool_name="bash", intent="run"))
    assert [row[TRAJECTORY_SEQ_KEY] for row in _trajectory(job)] == [0]
    retained = _trajectory(job)[0]

    for index in range(1, TRAJECTORY_CAP + 5):
        await relay(
            ToolExecutionStartEvent(tool_call_id=f"c{index}", tool_name="bash", intent="run")
        )
        stamps = [row[TRAJECTORY_SEQ_KEY] for row in _trajectory(job)]
        assert stamps == sorted(stamps), "a retained row's stamp was not monotone"
        assert len(set(stamps)) == len(stamps), "a stamp was reused"

    assert len(_trajectory(job)) == TRAJECTORY_CAP
    assert _trajectory(job)[0] is not retained, "past the cap the window did not trim the front"
    # The first row of this run was evicted from the FRONT, and nothing retained
    # was replaced: rows are removed by count from position 0, never rewritten.
    assert [row[TRAJECTORY_SEQ_KEY] for row in _trajectory(job)] == list(
        range(5, TRAJECTORY_CAP + 5)
    )


async def _noop_emit(event: Any) -> None:
    return None


def test_the_end_event_the_relay_records_is_a_row_the_cache_can_stamp() -> None:
    """A guard on the fixture itself: the row shape below is the real one."""
    row = ToolExecutionEndEvent(
        tool_call_id="c1",
        tool_name="bash",
        result=ToolResult(tool_call_id="c1", tool_name="bash", content=[]),
    ).model_dump(mode="json")
    assert "type" in row
    assert _is_row_shaped(row)
