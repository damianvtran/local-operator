"""A roster tick must cost the DELTA, not the retained window.

WHY THIS FILE EXISTS. ``Session._schedule_frontend_jobs`` coalesces roster churn
onto ``FrontendStateStore.refresh_jobs`` every 50 ms while a turn runs, and that
tick used to rebuild every job's whole retained window: ``JobState.from_job``
copied every row, ``_freeze_job`` rebuilt every row's frozen containers, and
``_jobs_equal`` then deep-compared them -- ~50-57 ms per tick with ~87 GC
collections on a 5-child roster at the 500-row cap, re-measured at 88 ms/tick on
this host (``scripts/bench_roster_tick.py``). That is more than one
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

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY, AsyncJob
from local_operator.harness.subagent import TRAJECTORY_CAP, _make_relay
from local_operator.harness.types import (
    ModelSpec,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    Usage,
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


#: A trajectory already in canonical-state containers -- exactly what
#: ``JobState.trajectory`` holds. ``_FrozenMapping`` is deliberately NOT a ``dict``
#: subclass, so a frozen row is not a raw job row: ``from_job`` says so on every
#: tree, and the memo has to agree with it in both directions (round 3, F7).
_FROZEN_ROWS = module._FrozenSequence(module._freeze_value(row) for row in _rows(2))


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


def _store(jobs: list[Any]) -> FrontendStateStore:
    """A store seeded the way a session seeds it: from the same live rows."""
    return FrontendStateStore(
        FrontendSessionState(
            session_id="frame-cost",
            epoch="e1",
            jobs=[JobState.from_job(job) for job in jobs],
        )
    )


def _session(jobs: list[Any]) -> Any:
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


def test_lineage_stamping_keeps_the_window_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production path always stamps lineage onto the row; it must not rebuild rows.

    ``_with_lineage`` runs whenever the session has a comms registry, which in a
    live session is always -- and it merges the parent/child identity through
    ``model_copy``, so the retained window has to survive it by identity.
    """
    jobs = [_job("child-0")]
    session = _session(jobs)
    session._subagent_comms = SimpleNamespace(
        job_rows=lambda: list(jobs),
        node=lambda _job_id: SimpleNamespace(
            session_id=None,
            live=False,
            session_dir=None,
            parent_job_id="parent-0",
            launch_message_id="",
            launch_prompts=None,
            attempt_aliases=(),
        ),
    )
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)
    first = store._state.jobs[0].trajectory

    assert store.refresh_jobs(session) is None
    assert (work.copies, work.freezes) == (0, 0), "lineage stamping re-materialised the rows"
    assert store._state.jobs[0].trajectory is first
    assert store._state.jobs[0].parent_job_id == "parent-0"


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


def test_cap_rotation_ships_the_appended_tail_and_no_replacement() -> None:
    """Past the cap the window rotates; the delta is the TAIL, not the window.

    WHY THIS EXPECTATION MOVED (it asserted ``["child-0"]`` in
    ``job_trajectory_replacements`` until ``_capped_overlap_tail`` landed). The
    marker was a PROXY for the invariant that matters -- a rotation must be
    reduced correctly and must not be silently mis-shipped -- and it was never the
    invariant itself: the reducer sends that marker on every full-cap rotation,
    including the provable ones, which is why a long-running child cost all 500
    rows per frame. With the rotation proven row for row the wire form is the
    appended tail and ZERO replacements, and the receiver's own rule,
    ``(old + tail)[-TRAJECTORY_CAP:]``, lands on this window exactly. The guard
    the marker was carrying now lives in
    :func:`test_an_unprovable_rotation_still_ships_a_replacement`, which is the
    case it was really protecting.
    """
    rows = _rows(TRAJECTORY_CAP)
    jobs = [_job("child-0", rows)]
    session = _session(jobs)
    store = _store(jobs)
    store.refresh_jobs(session)
    before = list(store._state.jobs[0].trajectory)

    _trajectory(jobs[0]).append(_row(TRAJECTORY_CAP))
    del _trajectory(jobs[0])[:1]
    update = store.refresh_jobs(session)

    assert update is not None
    assert update.job_trajectory_replacements == []
    tail = update.job_trajectory_appends["child-0"]
    # The DELTA, not the window: one row, the one the writer appended. Shipping
    # the rotated window here IS the ~634 KB/frame defect, so the count is the
    # assertion rather than a detail of it.
    assert len(tail) == 1
    assert tail[0][TRAJECTORY_SEQ_KEY] == TRAJECTORY_CAP
    # SUFFICIENT, not merely different: apply the receiver's rule to the window
    # this delta replaced and it reproduces the owner's rows. Compared by STAMP
    # because a delta thaws its rows at the wire boundary while canonical state
    # holds frozen ones, and the two are deliberately not equal.
    assert [row[TRAJECTORY_SEQ_KEY] for row in (*before, *tail)[-TRAJECTORY_CAP:]] == [
        row[TRAJECTORY_SEQ_KEY] for row in _trajectory(jobs[0])
    ]
    # The front really moved: the window is the NEWEST cap rows.
    assert list(store._state.jobs[0].trajectory) == jobs[0].trajectory
    assert store._state.jobs[0].trajectory_length == TRAJECTORY_CAP
    assert store._state.jobs[0].trajectory[0][TRAJECTORY_SEQ_KEY] == 1


def test_an_unprovable_rotation_still_ships_a_replacement() -> None:
    """A rotation whose overlap cannot be PROVEN keeps the full replacement.

    The protective intent behind the replacement marker, and the reason it may not
    simply be deleted now that a rotation usually ships a tail: a tail the reducer
    cannot reconstruct would hand a viewer the WRONG rows, which is worse than
    handing it too many. ``_capped_overlap_tail`` proves an overlap row for row
    rather than trusting agreeing endpoint stamps -- ``_lo_seq`` counts RELAYS, so
    a row revised in place leaves the first and last stamps identical and an
    interior edit can pass a stamp-only test unnoticed. Here the rotation is real
    and row 250 was revised, so no prefix of the new window equals a suffix of the
    old and the classifier must fall back to the replacement it has always sent.
    """
    rows = _rows(TRAJECTORY_CAP)
    jobs = [_job("child-0", rows)]
    session = _session(jobs)
    store = _store(jobs)
    store.refresh_jobs(session)
    before = store._state.jobs[0].trajectory

    _trajectory(jobs[0]).append(_row(TRAJECTORY_CAP))
    del _trajectory(jobs[0])[:1]
    # A NEW list object, so the roster memo re-freezes the window instead of
    # serving the one it cached under the old list's identity: a row revised in
    # place is the single change that memo's fingerprint cannot see, which is the
    # subject of ``test_the_relay_writer_only_appends_and_trims_the_front`` and
    # not what this cell is here to test.
    revised = list(_trajectory(jobs[0]))
    revised[250] = _row(251, text="revised in place")
    jobs[0].trajectory = revised

    update = store.refresh_jobs(session)

    assert update is not None
    # Pinned at the rule, on the same canonical shapes the classifier sees: the
    # stamps say "499 rows of overlap" and the element-wise comparison refuses it.
    assert module._capped_overlap_tail(before, store._state.jobs[0].trajectory) is None
    assert update.job_trajectory_replacements == ["child-0"]
    shipped = update.job_trajectory_appends["child-0"]
    assert len(shipped) == TRAJECTORY_CAP, "an unprovable rotation shipped a partial tail"
    assert shipped[250]["result"]["content"][0]["text"] == "revised in place"
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


def test_a_non_list_trajectory_is_frozen_rather_than_emptied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sequence the memo cannot fingerprint takes the FULL freeze, never an empty window.

    ``JobState.from_job`` has always materialised any iterable -- an iterable
    trajectory is exactly the host- or extension-provided row its defensive loop
    exists for -- so a tuple must ship its rows as the pre-change tree shipped
    them. The guard that read "rows I cannot fingerprint" as "no rows" emptied a
    job that must not be emptied; being unprovable may cost the memo, never a row.
    """
    rows = (_row(0), _row(1))
    job = SimpleNamespace(id="child-0", type="task", status="running", trajectory=rows, prompt="p")
    session = _session([job])
    store = _store([])
    work = _RowWork(monkeypatch)

    store.refresh_jobs(session)
    assert list(store._state.jobs[0].trajectory) == list(rows)
    assert store._state.jobs[0].trajectory_length == 2
    assert work.freezes == 2

    # And nothing about it is memoised, so the next tick freezes it again rather
    # than handing back a window no identity could ever prove.
    work.reset()
    store.refresh_jobs(session)
    assert work.freezes == 2
    assert "child-0" not in store._retained_windows()._by_job


async def _no_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
    """A stream double that never yields: these tests never run a real turn.

    An async GENERATOR rather than a plain callable, because that is what
    ``stream_fn``'s annotation requires -- a stub that only returns ``None`` would
    raise the first time a turn ran, which is the kind of double that hides a
    broken test.
    """
    return
    yield  # pragma: no cover -- the ``yield`` is what makes this a generator


def _real_session(directory: Path) -> Any:
    """A real Session, for the one cell that must drive the real coalescer."""
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    return Session(
        model=ModelSpec(provider="test", model_id="mock"),
        stream_fn=_no_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
    )


@pytest.mark.parametrize(
    "shape",
    [{"_lo_seq": 0, "type": "x"}, "abc", b"abc", (1, 2), ("a", "b")],
    ids=["mapping", "str", "bytes", "tuple-of-non-rows", "tuple-of-strs"],
)
def test_a_sequence_that_holds_no_rows_keeps_the_pre_change_answer(shape: Any) -> None:
    """A non-list sequence the writer never produces must not raise out of the tick.

    The fallback mirrors ``from_job``'s row predicate (``_retained_row``), so a
    sequence whose items are not rows gets the answer the pre-change tree gave it:
    no rows, and no exception. Freezing its ITEMS instead puts non-rows into
    canonical state, where ``FrontendUpdate(job_trajectory_appends=...)`` rejects
    them -- a ``ValidationError`` out of ``refresh_jobs`` and, on the roster path,
    out of the pump's bare ``call_later`` callback (round 2, F6/Q5).
    """
    job = SimpleNamespace(id="child-0", type="task", status="running", trajectory=shape, prompt="p")
    store = _store([])

    store.refresh_jobs(_session([job]))

    assert list(store.state.jobs[0].trajectory) == []
    assert store.state.jobs[0].trajectory_length == 0
    assert "child-0" not in store._retained_windows()._by_job


@pytest.mark.asyncio
async def test_a_sequence_that_holds_no_rows_does_not_escape_the_roster_pump(
    tmp_path: Path,
) -> None:
    """The loop-handler half: nothing reaches the exception handler on a real tick.

    ``Session._schedule_frontend_jobs`` runs the refresh from a bare ``call_later``
    callback, so anything ``refresh_jobs`` raises there is reported to the loop's
    exception handler and lost. Measured before the predicate was shared: three
    handler exceptions (mapping, str, tuple of strings) against zero on the
    merge-base; this asserts the zero, with the pump and the roster intact.
    """
    loop = asyncio.get_running_loop()
    seen: list[str] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: seen.append(str(context.get("exception") or context.get("message")))
    )
    try:
        session = _real_session(tmp_path)
        session.jobs.set_max_running(4)
        gate = asyncio.Event()

        async def run(job_id: str, signal: Any, progress: Any) -> None:
            await gate.wait()

        job_id = session.jobs.register("task", "child", run)
        job = session.jobs.get(job_id)
        job.trajectory = [{"type": "x", "_lo_seq": 0, "tool_call_id": "c", "tool_name": "bash"}]
        store = session._frontend_state_store
        store.subscribe(lambda _update: None)  # a live viewer: the pump runs

        session._schedule_frontend_jobs()
        await asyncio.sleep(0.25)
        assert [row.id for row in store.state.jobs] == [job_id]

        # Every shape from the round-3 matrix, including the `list` cells the
        # writer's declared type allows with wrong items: each one must reach the
        # coalescer, leave the roster intact and raise nothing into the loop.
        for shape in (
            {"_lo_seq": 0, "type": "x"},
            "abc",
            b"abc",
            ("a", "b"),
            (1, 2),
            ["a", "b"],
            [_row(0), "x", _row(1)],
            [_row(0), _row(1), "x"],
            _FROZEN_ROWS,
        ):
            job.trajectory = shape
            session._schedule_frontend_jobs()
            await asyncio.sleep(0.25)
            assert [row.id for row in store.state.jobs] == [job_id]

        assert seen == [], f"a malformed trajectory reached the loop: {seen[:1]}"
    finally:
        loop.set_exception_handler(previous)


def test_rebind_drops_the_previous_lineages_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The epoch is part of what a window is valid for.

    Pinned against the memo rather than through the store on purpose: every public
    path that moves the store's epoch also clears the memo outright
    (``replace()``, ``refresh_from_session(initial=True)``), so a store-level test
    would stay green with this guard deleted -- which is what round 1 found.
    """
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)
    windows = store._retained_windows()
    assert "child-0" in windows._by_job, "the fixture is supposed to have a memo"

    windows.rebind("e2")

    assert windows._by_job == {}, "a lineage move kept the previous epoch's window"
    work.reset()
    windows.window("child-0", getattr(jobs[0], "trajectory", None))
    assert work.freezes == ROWS, "the first read of a new lineage reused the old window"


class _JobThatFailsLate:
    """A job whose row builds once and then fails the way a malformed row does.

    The failure has to land AFTER the memo has answered, or the memo's own
    ``pop`` is what releases the entry and the ordering under test is masked: an
    unchanged trajectory is a memo HIT, so nothing in ``window`` touches the entry.
    """

    def __init__(self, job_id: str, rows: list[dict[str, Any]]) -> None:
        self.id = job_id
        self.type = "task"
        self.status = "running"
        self.prompt = "p"
        self.healthy = True
        self._rows = rows

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        return self._rows

    @property
    def latest_details(self) -> dict[str, Any]:
        if not self.healthy:
            raise RuntimeError("a malformed extension row")
        return {"progress": "thinking"}


def _seen_rows(shape: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """(rows on the first tick, on the second tick, and what a plain rebuild holds).

    The second tick is the one that can consult a memo, so the pair is THE
    transparency check: a window may neither keep a row the rebuild path drops nor
    drop a row it keeps. Generators are excluded by callers -- a generator is
    single-shot, so a read after the tick that materialised it is empty on every
    tree including the merge-base, which is a property of the shape and not of the
    memo.
    """
    job = SimpleNamespace(id="child-0", type="task", status="running", trajectory=shape, prompt="p")
    session = _session([job])
    store = _store([])
    store.refresh_jobs(session)
    first_tick = list(store.state.jobs[0].trajectory)
    store.refresh_jobs(session)
    second_tick = list(store.state.jobs[0].trajectory)
    return first_tick, second_tick, list(JobState.from_job(job).trajectory)


@pytest.mark.parametrize(
    ("shape", "merge_base_rows"),
    [
        (["a", "b"], 0),
        ([_row(0), "x", _row(1)], 2),
        ([_row(0), _row(1), "x"], 2),
    ],
    ids=["list-of-strs", "list-with-a-str-in-the-middle", "list-with-a-trailing-str"],
)
def test_a_list_holding_non_rows_answers_what_the_merge_base_answered(
    shape: Any, merge_base_rows: int
) -> None:
    """A ``list`` whose ITEMS are wrong answers 0 rows quietly, as it did before the memo.

    This is the shape the declared type allows and the writer's own type does not
    guarantee: ``list[dict]`` with an item that is not a dict. Before the memo the
    only reader materialised the rows it recognised and shipped the rest as absent;
    from ``0854ac6ac`` on, the fast path froze every item, so the non-row reached
    canonical state and ``FrontendUpdate(job_trajectory_appends=...)`` raised -- once
    per tick, out of the roster pump's bare ``call_later`` callback, and with the two
    readers disagreeing for the same job. The fingerprint cannot see an item in the
    middle, so such a list is simply not memoised (see :func:`_row_items`).
    """
    first_tick, second_tick, rebuild = _seen_rows(shape)

    assert second_tick == first_tick == rebuild
    assert len(second_tick) == merge_base_rows


def test_a_list_holding_only_rows_is_still_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard above must not cost the common path its memo.

    ``_row_items`` returns an all-rows list BY IDENTITY, so the tick after the first
    still answers from the entry instead of freezing the window again -- the whole
    point of this file. Pinned structurally: a second unchanged tick freezes nothing.
    """
    jobs = [_job("child-0")]
    session = _session(jobs)
    store = _store(jobs)
    work = _warm(store, session, monkeypatch)

    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == 0, "a clean list stopped being memoised"
    assert "child-0" in store._retained_windows()._by_job


@pytest.mark.parametrize(
    "shape",
    [
        _rows(2),
        (_row(0), _row(1)),
        [_row(0), "x", _row(1)],
        ["a", "b"],
        {"_lo_seq": 0, "type": "x"},
        "abc",
        b"abc",
        ("a", "b"),
        (1, 2),
        _FROZEN_ROWS,
    ],
    ids=[
        "list-of-dicts",
        "tuple-of-dicts",
        "list-with-a-str-in-the-middle",
        "list-of-strs",
        "mapping",
        "str",
        "bytes",
        "tuple-of-strs",
        "tuple-of-non-rows",
        "frozen-sequence",
    ],
)
def test_the_memo_serves_exactly_what_a_plain_rebuild_would(shape: Any) -> None:
    """Transparency for every shape, in both directions.

    THE invariant the memo's soundness rests on: what a warm read (second tick, memo
    in play) serves equals what a cold read serves, which equals what the one-off
    reader builds. Stated as an equality rather than as a literal count, so the
    adjudicated ``_FrozenSequence`` answer (0 rows: canonical-state containers are
    not raw job rows, and ``from_job`` says so on every tree) is locked to its reason
    instead of to the number -- and so the next agent cannot flip either direction
    while the malformed cells stay green.
    """
    first_tick, second_tick, rebuild = _seen_rows(shape)

    assert second_tick == first_tick == rebuild


def test_the_shared_predicate_refuses_a_canonical_container() -> None:
    """The REASON the ``frozen-sequence`` cell ships 0 rows, pinned where the rule lives.

    A ``_FrozenMapping`` is what canonical state already holds -- deliberately not a
    ``dict`` subclass -- so accepting it would be a second, drifting notion of what a
    raw job row is, and ``JobState.from_job`` has refused it since the merge-base.
    Locking this here means the transparency cell above cannot be flipped to the
    other direction without a test saying so.
    """
    assert module._is_retained_row(_FROZEN_ROWS[0]) is False
    assert module._is_retained_row(_row(0)) is True
    assert (
        module._is_retained_row(
            ToolExecutionEndEvent(
                tool_call_id="c0",
                tool_name="bash",
                result=ToolResult(tool_call_id="c0", tool_name="bash", content=[]),
            )
        )
        is True
    )


def test_a_row_that_fails_to_build_does_not_keep_its_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A skipped row must not pin the raw row list its memo entry holds by reference.

    ``_jobs`` deliberately skips one malformed extension row rather than erasing
    the whole roster, and ``retain`` bounds the memo to the live roster precisely
    so an entry cannot outlive the job it describes. A job that is skipped is not
    live, so it must not be handed to ``retain`` as seen.
    """
    job_a = _JobThatFailsLate("child-0", _rows(4))
    job_b = _JobThatFailsLate("child-1", _rows(4))
    session = _session([job_a, job_b])
    store = _store([])
    work = _warm(store, session, monkeypatch)
    assert {"child-0", "child-1"} <= set(store._retained_windows()._by_job)

    job_b.healthy = False
    work.reset()
    store.refresh_jobs(session)

    assert work.freezes == 0, "the fixture was supposed to be a memo HIT, not a freeze"
    assert "child-1" not in store._retained_windows()._by_job, "a skipped row kept its memo entry"
    assert "child-0" in store._retained_windows()._by_job
    assert [job.id for job in store._state.jobs] == ["child-0"]


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


# ---------------------------------------------------------------------------
# Released rows: identity survives, the tick stops paying for it
#
# The window memo above makes a tick cost the DELTA for a child that is still
# working. It does nothing for a child that has FINISHED: ``comms.job_rows()``
# re-adds every settled child the execution ledger already swept (through
# ``_ChildRecord.job_ref``, deliberately -- the publish is a follower's only
# handle on a swept child), so a long session's roster only grows, and every
# one of those rows was rebuilt, re-validated and re-frozen on every 50 ms
# tick even though nothing about it can change again.
#
# Measured on the operator's own wedged session: 48 rows, 42 hours, the loop
# pinned at 100 % of a core with the roster generation not advancing. These
# pin the two halves of the fix -- the row is projected without its window and
# stamped ``roster_released``, and it is then reused by identity rather than
# rebuilt.
# ---------------------------------------------------------------------------


def _settled(job: AsyncJob, *, ago: float = 3600.0) -> AsyncJob:
    """The job as the ledger leaves it once retention has let it go."""
    job.status = "completed"
    job.settled_at = __import__("time").time() - ago
    return job


def _released_session(jobs: list[Any], *, retention_ms: float = 5 * 60_000) -> Any:
    """A session whose manager publishes a retention window, as a live one does."""
    session = _session(jobs)
    session.jobs.retention_ms = retention_ms
    session._subagent_comms = SimpleNamespace(
        job_rows=lambda: list(jobs),
        nodes=lambda: [],
        node=lambda _job_id: None,
    )
    return session


def test_a_released_row_keeps_its_identity_and_says_it_was_released() -> None:
    """Released means "not a current member", never "gone"."""
    live, done = _job("child-live"), _settled(_job("child-done"))
    session = _released_session([live, done])
    store = _store([live, done])
    store.refresh_jobs(session)

    rows = {row.id: row for row in store.state.jobs}
    assert set(rows) == {"child-live", "child-done"}, "a released child vanished from the roster"
    assert rows["child-done"].roster_released is True
    assert rows["child-done"].status == "completed"
    assert rows["child-done"].label == "child-done"
    # The live child is untouched by any of this.
    assert rows["child-live"].roster_released is False
    assert len(rows["child-live"].trajectory) == ROWS


def test_a_released_row_sheds_the_retained_window_it_can_no_longer_change() -> None:
    """The per-ROW half: a settled child stops carrying 500 rows on every tick."""
    done = _settled(_job("child-done"))
    session = _released_session([done])
    store = _store([done])
    store.refresh_jobs(session)

    row = store.state.jobs[0]
    assert row.trajectory == (), "a released row still carried its retained window"
    # The rows are not lost -- they are on disk in the child's own transcript,
    # which is what the subagent page reads. The COUNT still rides along so the
    # page can say how many events there were.
    assert row.trajectory_length == ROWS


def test_a_running_child_is_never_released_whatever_its_stamps_say() -> None:
    """``retention_expired`` refuses a running row; this must inherit that."""
    running = _job("child-live")
    running.settled_at = 0.0  # an ancient stamp, but it is still running
    session = _released_session([running])
    store = _store([running])
    store.refresh_jobs(session)

    row = store.state.jobs[0]
    assert row.roster_released is False, "a RUNNING child was released"
    assert len(row.trajectory) == ROWS


def test_a_settled_child_inside_the_window_is_not_released_yet() -> None:
    """Retention is a window, and a child that just finished is still in it."""
    fresh = _settled(_job("child-done"), ago=1.0)
    session = _released_session([fresh])
    store = _store([fresh])
    store.refresh_jobs(session)

    assert store.state.jobs[0].roster_released is False
    assert len(store.state.jobs[0].trajectory) == ROWS


def test_a_released_row_is_reused_by_identity_across_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-JOB half, and the one the wedge actually turned on.

    Shedding the window removes the per-row cost; this removes the per-row-ROW
    cost that remained -- validating and freezing a fresh ``JobState`` for every
    settled child, on every tick, forever. Asserted by IDENTITY rather than by
    timing, per this file's header: the same object, not an equal rebuild.
    """
    done = [_settled(_job(f"child-{index}")) for index in range(4)]
    live = _job("child-live")
    session = _released_session([live, *done])
    store = _store([live, *done])
    store.refresh_jobs(session)

    # Read CANONICAL state, not the ``state`` property: that one deep-copies
    # every row on the way out by design (it must never share an owning model),
    # so it can never answer an identity question about what the store holds.
    before = {row.id: row for row in store._state.jobs if row.roster_released}
    assert len(before) == 4, "the settled children were not released"

    # A tick driven by the LIVE child appending, which is what a real tick is.
    _trajectory(live).append(_row(ROWS))
    store.refresh_jobs(session)

    after = {row.id: row for row in store._state.jobs if row.roster_released}
    assert set(after) == set(before)
    for job_id, row in after.items():
        assert row is before[job_id], f"{job_id} was rebuilt on a tick it cannot have changed"


def test_a_released_row_is_rebuilt_when_its_terminal_facts_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memo is a fingerprint, not a latch: a changed row must not be served stale."""
    done = _settled(_job("child-done"))
    session = _released_session([done])
    store = _store([done])
    store.refresh_jobs(session)
    # CANONICAL state, not the `state` property: that one runs every row through
    # `_public_job`, which `model_copy`s unconditionally, so two `state` reads
    # are never the same object and an identity assertion across them is
    # vacuously true. The value assertion below is what carried this test; the
    # identity one now means something too.
    first = store._state.jobs[0]

    done.result_text = "the answer the child came back with"
    store.refresh_jobs(session)
    second = store._state.jobs[0]

    assert second is not first, "a released row was served from a stale memo"
    assert second.result_text == "the answer the child came back with"


def test_a_released_row_is_rebuilt_when_its_derived_inputs_move() -> None:
    """The key covers what ``_released_row`` DERIVES, not only terminal text.

    AGENT REVIEW ROUND 1, S1. ``_released_row`` prices the row from ``job.usage``
    and ``job.descendant_usage``, carries ``trajectory_length`` off the retained
    window and reads ``model_label`` -- and the fingerprint covered none of
    them. A released row whose accounting moved was therefore served the SAME
    object for ever, with no later tick able to repair it, because the mark
    never moves again. Each leg below moves exactly one derived input and
    asserts a REBUILD, which is the only observable a memo can offer.

    Read CANONICAL state for the identity legs (``_public_job`` copies on every
    ``state`` read, so an identity assertion across two of those is vacuous).
    """
    done = _settled(_job("child-done"))
    session = _released_session([done])
    store = _store([done])
    store.refresh_jobs(session)
    current = store._state.jobs[0]
    assert current.roster_released is True, "the premise is unmet: the row was not released"
    assert current.trajectory_length == ROWS

    # EACH LEG RE-BASELINES AND CARRIES BOTH ASSERTIONS, deliberately. Comparing
    # every leg back to the FIRST row would let an earlier leg's rebuild satisfy
    # a later leg's identity check, and an identity check with no value check
    # says nothing about what the row actually holds -- both were true of the
    # first draft of this test, and a per-element sabotage sweep is what showed
    # it (each element below must be able to fail the file ON ITS OWN).

    # (a) accounting is accumulated IN PLACE on the job, so an identity check on
    # the usage object would never fire -- the counters have to be keyed.
    done.usage = Usage(input_tokens=11, output_tokens=22)
    store.refresh_jobs(session)
    after_usage = store._state.jobs[0]
    assert after_usage is not current, "a released row priced from a moved usage was reused"
    assert after_usage.usage is not None and after_usage.usage.output_tokens == 22
    current = after_usage

    # (b) the retained window: the length is the cheap discriminator, and it is
    # what ``trajectory_length`` on the row is built from.
    _trajectory(done).append(_row(ROWS))
    store.refresh_jobs(session)
    after_window = store._state.jobs[0]
    assert after_window is not current, "a released row's trajectory_length moved unnoticed"
    assert after_window.trajectory_length == ROWS + 1
    current = after_window

    # (c) settled-descendant accounting, replaced wholesale at the ownership
    # boundary by ``detach_child_manager``.
    done.descendant_usage = [Usage(input_tokens=3, output_tokens=4)]
    store.refresh_jobs(session)
    after_descendants = store._state.jobs[0]
    assert after_descendants is not current, "a released row's descendants moved unnoticed"
    assert len(after_descendants.descendant_usage) == 1


def test_the_release_key_covers_every_element_that_can_move_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: each element added for S1 must be able to fail this file ON ITS OWN.

    Round 2's per-element sweep found the first test pinned only the usage and
    descendant elements: ``model_label`` and the plan could also be dropped with
    the file still green, and the trajectory pair was falsifiable only as a PAIR
    until the front-deletion leg below gave the length element a shape of its
    own. (``context_window`` had no row in that sweep because it was not in the
    key at all -- that was round 2's R1, not a gap in the pinning; review round
    3, T5.)

    Driven through ``_ReleasedRows.row`` directly rather than through
    ``refresh_jobs``, because one of these legs -- a rotation at constant
    length -- is VALUE-EQUAL for everything a released row projects, so
    ``_jobs_equal`` keeps the old row and discards the rebuild. A store-level
    identity assertion cannot see that element work; the memo can.
    """
    from local_operator.tools.builtin import TODO_STORE

    session_id = "abcdef123456"
    memo = module._ReleasedRows("e1")
    job = _settled(_job("child-done"))
    comms = SimpleNamespace(
        node=lambda _job_id: SimpleNamespace(
            session_id=session_id,
            live=False,
            session_dir=None,
            parent_job_id=None,
            launch_message_id="",
            launch_prompts=None,
            attempt_aliases=(),
        )
    )

    def row() -> JobState:
        return memo.row("child-done", job, comms)

    current = row()

    # (d) the relay's ``ModelChangeEvent`` arm writes ``model_label`` and
    # ``context_window`` on consecutive lines of the same block; round 2 (R1)
    # caught the key carrying only the first while the row reads BOTH.
    job.model_label = "other-provider/other-model"
    after_label = row()
    assert after_label is not current, "model_label moved and the row was reused"
    assert after_label.model_label == "other-provider/other-model"
    current = after_label

    job.context_window = 200_000
    after_context = row()
    assert after_context is not current, "context_window moved and the row was reused"
    assert after_context.context_window == 200_000
    current = after_context

    # (e) a rotation at CONSTANT length: an append that evicts the front, which
    # is the shape a length-only key cannot see.
    rows = _trajectory(job)
    rows.pop(0)
    rows.append(_row(ROWS))
    assert len(rows) == ROWS, "the rotation must leave the length unchanged"
    after_rotation = row()
    assert after_rotation is not current, "a rotation at constant length was not noticed"
    current = after_rotation

    # (e2) a FRONT deletion on its own: the length moves while the newest stamp
    # does NOT, which is the one shape the relay-stamp element cannot see. This
    # leg is what makes the length element falsifiable without its pair -- a
    # sweep with only (e) leaves it green, because both elements catch an
    # append-plus-evict and only the length catches a bare front drop.
    rows.pop(0)
    assert len(rows) == ROWS - 1, "the front drop must shorten the window"
    after_front_drop = row()
    assert after_front_drop is not current, "a front deletion was not noticed"
    assert after_front_drop.trajectory_length == ROWS - 1
    current = after_front_drop

    # (f) a plan APPEARING. ``todos`` rides onto the row from ``TODO_STORE`` via
    # ``_with_lineage``, so no node-derived element can cover it.
    monkeypatch.setitem(
        TODO_STORE,
        session_id,
        [{"name": "Todos", "items": [{"text": "ship it", "status": "pending"}]}],
    )
    after_plan = row()
    assert after_plan is not current, "a plan appearing did not move the key"
    assert after_plan.todos, "the key moved but the row did not gain the plan"


def test_a_released_row_keeps_the_lineage_its_transcript_page_needs() -> None:
    """A swept child's page is reachable ONLY through the lineage on its row.

    ``_with_lineage`` is what stamps ``session_id``/``session_dir``, and the
    released projection has to run it too -- skipping it is what
    ``test_a_swept_child_keeps_its_durable_identity_on_the_roster`` catches at
    the session level, pinned here at the unit the projection lives in.
    """
    done = _settled(_job("child-done"))
    session = _released_session([done])
    session._subagent_comms = SimpleNamespace(
        job_rows=lambda: [done],
        nodes=lambda: [],
        node=lambda _job_id: SimpleNamespace(
            session_id="abcdef123456",
            live=False,
            session_dir=Path("/tmp/sessions/abcdef123456"),
            parent_job_id="parent-0",
            launch_message_id="subagent-launch:child-done",
            launch_prompts=None,
            attempt_aliases=(),
        ),
    )
    store = _store([done])
    store.refresh_jobs(session)

    row = store.state.jobs[0]
    assert row.roster_released is True
    assert row.session_id == "abcdef123456"
    assert row.session_dir == "/tmp/sessions/abcdef123456"
    assert row.parent_job_id == "parent-0"


def test_a_host_that_publishes_no_retention_window_releases_nothing() -> None:
    """Fail CLOSED: an unknown window costs the old work, never a wrong release."""
    done = _settled(_job("child-done"))
    session = _session([done])  # no ``retention_ms`` on the manager at all
    session._subagent_comms = SimpleNamespace(
        job_rows=lambda: [done], nodes=lambda: [], node=lambda _job_id: None
    )
    store = _store([done])
    store.refresh_jobs(session)

    row = store.state.jobs[0]
    assert row.roster_released is False
    assert len(row.trajectory) == ROWS


def test_a_paused_child_is_not_released_however_long_it_has_been_parked() -> None:
    """A pause is mechanically a cancel; the roster window exempts it deliberately."""
    parked = _settled(_job("child-paused"))
    parked.status = "cancelled"
    session = _released_session([parked])
    session._subagent_comms = SimpleNamespace(
        job_rows=lambda: [parked],
        nodes=lambda: [SimpleNamespace(job_id="child-paused", status="paused")],
        node=lambda _job_id: None,
    )
    store = _store([parked])
    store.refresh_jobs(session)

    assert store.state.jobs[0].roster_released is False, "a PAUSED child was released"


def test_an_unresolved_failure_is_not_released_on_the_quiet_clock() -> None:
    """Retention is a timer on quiet resolutions, not on unfinished business."""
    failed = _settled(_job("child-failed"))
    failed.status = "failed"
    session = _released_session([failed])
    store = _store([failed])
    store.refresh_jobs(session)

    assert store.state.jobs[0].roster_released is False, "a FAILED child was released"


def test_the_released_flag_does_not_ride_the_wire_when_it_is_false() -> None:
    """One key per row at roster scale is what the attach frame guard measures."""
    live, done = _job("child-live"), _settled(_job("child-done"))
    session = _released_session([live, done])
    store = _store([live, done])
    store.refresh_jobs(session)

    payload = sync_wire_payload(
        FrontendSync(epoch=store.state.epoch, sequence=store.state.sequence, snapshot=store.state)
    )
    rows = {row["id"]: row for row in payload["snapshot"]["jobs"]}
    assert "roster_released" not in rows["child-live"], "the default bought wire bytes"
    assert rows["child-done"]["roster_released"] is True, "the informative value was dropped"


def test_a_released_row_holds_only_frozen_containers() -> None:
    """A released row must obey the SAME immutability contract as any other.

    The near-miss this pins. ``_released_row`` freezes, but ``_with_lineage``
    runs after it and re-stamps ``launch_prompts`` (a ``dict``),
    ``attempt_aliases`` and ``todos`` (``list``s) straight off the comms node,
    so the row it returns is not frozen however frozen its input was. An
    earlier draft MARKED that result frozen instead of freezing it -- and since
    ``_freeze_job`` early-returns on a marked row, the raw containers then
    survived every later tick: ``_public_job`` shares a non-``BaseModel`` field
    by reference, so canonical state was reachable and mutable through the
    public ``state`` accessor, and the row raised on ``hash()``.

    Asserted on the CONTAINER TYPES rather than on behaviour because that is
    the invariant: the earlier released-row tests all passed against the broken
    version, since an empty ``dict`` and an empty ``_FrozenMapping`` compare
    equal and only a NON-EMPTY one can tell them apart.
    """
    done = _settled(_job("child-done"))
    session = _released_session([done])
    session._subagent_comms = SimpleNamespace(
        job_rows=lambda: [done],
        nodes=lambda: [],
        node=lambda _job_id: SimpleNamespace(
            session_id="abcdef123456",
            live=False,
            session_dir=Path("/tmp/sessions/abcdef123456"),
            parent_job_id="parent-0",
            launch_message_id="subagent-launch:child-done",
            # Non-empty on purpose: the empty case cannot distinguish a raw
            # container from a frozen one.
            launch_prompts={"subagent-launch:child-done": "go do a thing"},
            attempt_aliases=["older-attempt"],
        ),
    )
    store = _store([done])
    store.refresh_jobs(session)

    row = store._state.jobs[0]
    assert row.roster_released is True
    assert isinstance(row.launch_prompts, module._FrozenMapping), "launch_prompts was not frozen"
    assert isinstance(row.attempt_aliases, module._FrozenSequence), "attempt_aliases was not frozen"
    # The whole point of the frozen containers: the row is a value, so it
    # hashes and can be shared without a defensive copy.
    assert isinstance(hash(row), int)

    # And it STAYS frozen: ``_freeze_job`` early-returns on a row it recognises,
    # so a row that slipped through unfrozen once would never be repaired.
    _trajectory(done)  # the released row sheds its window; the job still has one
    store.refresh_jobs(session)
    again = store._state.jobs[0]
    assert isinstance(again.launch_prompts, module._FrozenMapping)
    assert isinstance(again.attempt_aliases, module._FrozenSequence)
