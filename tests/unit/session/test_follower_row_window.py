"""A follower's ``jobs`` delta must cost the DELTA, not the window it lands in.

WHY THIS FILE EXISTS. ``FrontendStateStore.apply_update`` is the reducer every
attached viewer runs on the event loop, once per canonical delta per session
(``AttachedSession._on_frontend_update``). For a ``jobs`` delta it rebuilt every
job from scratch — ``list(prior.trajectory)`` extended with
``job_trajectory_appends``, then ``FrontendSessionState.model_validate``
re-validating every retained row, then ``_freeze_state_jobs`` re-freezing every
row. The coupling audit measured that at 6.8-9.3 ms per delta on a 6-child
roster at a 500-row retained window: 14-19 % of a core per attached session
streaming at the producer's 20/s cadence, on the loop that also paints the frame
and reads every other session's socket.

The fix is the follower half of the memo the open producer-side PR #1123 adds
one process upstream: a job's frozen retained window is kept per job and reused
BY IDENTITY, and only the appended tail is frozen. It is a separate cache with a
separate proof — see ``_FollowerTrajectoryWindows`` — because the follower holds
an immutable frozen window where the producer holds the mutable raw row list.

Every assertion here is STRUCTURAL — what ran, not how long it took, per
AGENTS.md's "Timing, flakes": rows frozen through a spy, pydantic's view of the
delta captured, and object identity where the point is that one object was
REUSED rather than an equal one rebuilt.

The proof's own soundness rests on identity plus a set of refusals, so each way
it could be fooled has a test that the memo REFUSES to reuse rather than one
that it reuses. Two of those (the epoch move, the tail witness) cannot be
produced through the reducer today because the retained container is immutable;
they are exercised at the memo's own boundary, which is where the guard lives.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import pytest

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY
from local_operator.harness.subagent import TRAJECTORY_CAP
from local_operator.session import frontend_state as module
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
)


def _row(index: int, *, text: str = "x" * 16) -> dict[str, Any]:
    return {
        "type": "value",
        "tool_call_id": f"call_{index}",
        "tool_name": "bash",
        "index": index,
        "result": {"content": [{"type": "text", "text": text}]},
    }


def _state(children: int, rows: int, *, epoch: str = "e1") -> FrontendSessionState:
    return FrontendSessionState(
        session_id="follower-window",
        epoch=epoch,
        jobs=[
            JobState(
                id=f"child-{child}",
                type="task",
                label=f"child-{child}",
                status="running",
                trajectory=[_row(index) for index in range(rows)],
                trajectory_length=rows,
            )
            for child in range(children)
        ],
    )


class _Wire:
    """Frames from a REAL producer, so the follower sees the frames the wire carries.

    The delta the follower is asked to reduce is not hand-built here: it comes
    out of a second ``FrontendStateStore`` driven exactly as a session drives it
    (``mutate``), which is what makes the appends/replacements on these frames
    the ones production produces rather than the shapes a test hoped for.
    """

    def __init__(self, children: int, rows: int) -> None:
        self.start = _state(children, rows)
        self.state = self.start
        self.owner = FrontendStateStore(self.start)
        self.jobs = list(self.start.jobs)

    def _publish(self, jobs: list[JobState]) -> FrontendUpdate:
        self.jobs = jobs
        update = self.owner.mutate(jobs=jobs)
        assert update is not None, "the producer published no frame"
        assert update.epoch == self.start.epoch
        self.state = self.owner.state
        return update

    def append_rows(self, count: int = 1) -> FrontendUpdate:
        """One appended row per child: the shape the audit measured."""
        jobs = [
            job.model_copy(
                update={
                    "trajectory": [*job.trajectory, *[_row(1000 + step) for step in range(count)]]
                }
            )
            for job in self.jobs
        ]
        return self._publish(jobs)

    def touch_status(self, status: str = "completed") -> FrontendUpdate:
        """A roster delta that appends nothing — the proof must survive it."""
        return self._publish([job.model_copy(update={"status": status}) for job in self.jobs])

    def replace_window(self, rows: int) -> FrontendUpdate:
        """A window that is not a suffix of the previous one: a replacement."""
        return self._publish(
            [
                job.model_copy(
                    update={
                        "trajectory": [
                            _row(9000 + index, text="second attempt") for index in range(rows)
                        ],
                        "trajectory_length": rows,
                    }
                )
                for job in self.jobs
            ]
        )


def _is_row_shaped(value: Any) -> bool:
    """True for a retained trajectory ROW rather than any other mapping.

    Keyed on ``tool_call_id``, which every row carries and no nested value does:
    a row's own ``result.content`` blocks are dicts too, so keying on ``type``
    would count each row twice. Read through ``keys()`` rather than ``in``,
    because a frozen row is a tuple subclass whose membership test walks its
    (key, value) PAIRS — ``"tool_call_id" in frozen_row`` is False, so a spy that
    used it would count nothing on exactly the rows that matter.
    """
    if not isinstance(value, Mapping):
        return False
    return any(key == "tool_call_id" for key in value.keys())


class _RowWork:
    """Counts the rows one or more deltas freeze, and the jobs they rebuild."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows = 0
        self.jobs = 0
        original_freeze_value = module._freeze_value
        original_freeze_job = module._freeze_job

        def counted_freeze_value(value: Any) -> Any:
            if _is_row_shaped(value):
                self.rows += 1
            return original_freeze_value(value)

        def counted_freeze_job(job: Any) -> Any:
            self.jobs += 1
            return original_freeze_job(job)

        monkeypatch.setattr(module, "_freeze_value", counted_freeze_value)
        monkeypatch.setattr(module, "_freeze_job", counted_freeze_job)

    def reset(self) -> None:
        self.rows = 0
        self.jobs = 0


def _validated_job_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The job rows pydantic is asked to validate, captured from the real call."""
    seen: list[dict[str, Any]] = []
    original = FrontendSessionState.model_validate

    def spy(payload: Any, **kwargs: Any) -> Any:
        if isinstance(payload, dict) and "jobs" in payload:
            seen.extend(payload["jobs"])
        return original(payload, **kwargs)

    monkeypatch.setattr(FrontendSessionState, "model_validate", spy)
    return seen


class _Producer(Protocol):
    """What a follower needs from a wire: the two states and the writer.

    Both fixtures satisfy this — `_Wire` grows its window past the cap, `_CappedWire`
    rotates it at the cap — so the follower tests can be written once against the
    shape rather than duplicated per producer.
    """

    start: FrontendSessionState
    state: FrontendSessionState
    owner: FrontendStateStore


def _follower(wire: _Producer) -> FrontendStateStore:
    """A viewer seeded the way a viewer is: from the same starting state."""
    return FrontendStateStore(wire.start)


def test_a_proven_delta_freezes_only_the_appended_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect: 6.8-9.3 ms per delta for an appended row per child."""
    wire = _Wire(children=3, rows=40)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())  # warm: the first frame has no proof
    # The frame is built BEFORE the spy is installed: a producer rebuilds its own
    # roster while publishing, and counting that would measure the producer's cost
    # (the defect PR #1123 fixes one process upstream) rather than the follower's.
    update = wire.append_rows()
    work = _RowWork(monkeypatch)

    follower.apply_update(update)

    assert work.rows == 3, "a delta appending one row per child froze more than those rows"
    assert work.jobs == 3, "each roster row's shell is still rebuilt, and only its shell"
    assert len(follower._state.jobs[0].trajectory) == 42


def test_a_proven_delta_never_hands_the_retained_window_to_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the cost: re-validating rows whose value is discarded."""
    wire = _Wire(children=2, rows=40)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    update = wire.append_rows()
    seen = _validated_job_rows(monkeypatch)

    follower.apply_update(update)

    assert len(seen) == 2, "the roster was not validated as two job rows"
    assert all("trajectory" not in row for row in seen), (
        "a retained window reached model_validate; its 40 rows would be re-validated "
        "only to be overwritten by the reducer's own window"
    )


def test_the_retained_rows_survive_a_proven_delta_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity, not equality: an equal-but-rebuilt row is the cost being removed."""
    wire = _Wire(children=1, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    before = follower._state.jobs[0].trajectory

    follower.apply_update(wire.append_rows())
    after = follower._state.jobs[0].trajectory

    assert after is not before
    assert len(after) == 8
    assert all(
        after[index] is before[index] for index in range(6)
    ), "the retained prefix was rebuilt rather than reused"


def test_a_roster_tick_that_appends_nothing_keeps_the_window_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status-only tick must not invalidate the proof for the NEXT delta.

    Reusing the window is not enough here: the canonical job has to keep the very
    object the memo describes, or every delta after a scalar tick pays a full
    re-freeze.
    """
    wire = _Wire(children=2, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    window = follower._state.jobs[0].trajectory
    update = wire.touch_status()
    work = _RowWork(monkeypatch)

    follower.apply_update(update)

    assert follower._state.jobs[0].status == "completed"
    assert follower._state.jobs[0].trajectory is window, "the window was rebuilt"
    assert work.rows == 0, "a tick that appended nothing froze retained rows"

    follow_up = wire.append_rows()
    work.reset()
    follower.apply_update(follow_up)
    assert work.rows == 2, "the proof was lost across the scalar tick"


def test_a_re_seated_page_of_the_same_rows_is_not_handed_back_older(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The identity proof itself, through a path a viewer really takes.

    A viewer refetches a child's page on reconnect and re-seeds it, and the
    fetched rows can be the ones the window already held. The seeded window is
    then EQUAL to the memo's but is not the object canonical state holds, and the
    proof is identity rather than equality for that reason: the memo's contract
    is "hand back what canonical state holds, or nothing", which is what keeps
    the next delta's proof O(1) instead of a whole-window comparison — the cost
    this memo exists to remove — and what stops it holding an object no state
    holds.
    """
    wire = _Wire(children=1, rows=6)
    follower = _follower(wire)

    follower.apply_update(wire.append_rows())
    rows = [dict(row) for row in follower._state.jobs[0].trajectory]
    assert follower.seed_job_trajectory("child-0", rows)
    seeded = follower._state.jobs[0].trajectory
    assert seeded == follower._follower_windows._by_job["child-0"].rows
    assert seeded is not follower._follower_windows._by_job["child-0"].rows

    update = wire.append_rows()
    work = _RowWork(monkeypatch)
    follower.apply_update(update)

    assert (
        work.rows == len(seeded) + 1
    ), "the memo handed back a window canonical state no longer holds"


def test_the_follower_matches_the_owner_row_for_row() -> None:
    """Equivalence with the retained path, at a shape worth comparing."""
    wire = _Wire(children=4, rows=50)
    follower = _follower(wire)
    for _ in range(6):
        follower.apply_update(wire.append_rows())
        assert follower._state.jobs[0].trajectory == wire.owner.state.jobs[0].trajectory
    follower.apply_update(wire.touch_status())
    follower.apply_update(wire.replace_window(rows=7))
    assert [list(job.trajectory) for job in follower._state.jobs] == [
        list(job.trajectory) for job in wire.owner.state.jobs
    ]
    assert follower._state.jobs[0].trajectory_length == len(wire.owner.state.jobs[0].trajectory)


def test_the_front_trim_is_expressed_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AT the cap the window rotates on every delta; refusing the memo there
    would leave the memo useless at exactly the shape this work was measured
    against. The trim is the rebuild path's own ``del rows[:len(rows) - CAP]``,
    so the surviving rows are the same objects in the same order."""
    wire = _Wire(children=1, rows=TRAJECTORY_CAP)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    before = follower._state.jobs[0].trajectory
    assert len(before) == TRAJECTORY_CAP
    update = wire.append_rows()
    work = _RowWork(monkeypatch)

    follower.apply_update(update)
    after = follower._state.jobs[0].trajectory

    assert len(after) == TRAJECTORY_CAP, "the cap was not honoured"
    assert after[: TRAJECTORY_CAP - 1] == before[1:], "the front did not move by one row"
    assert after[TRAJECTORY_CAP - 1] == update.job_trajectory_appends["child-0"][0]
    assert after[0] is before[1], "the rotated window was rebuilt rather than re-sliced"
    assert work.rows == 1, "the whole window was re-frozen instead of the appended row"


def test_a_replacement_never_reuses_the_previous_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime's own 'this is not a suffix' marker, checked before the memo."""
    wire = _Wire(children=1, rows=8)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    assert "child-0" in follower._follower_windows._by_job
    update = wire.replace_window(rows=5)
    assert update.job_trajectory_replacements == ["child-0"]
    work = _RowWork(monkeypatch)

    follower.apply_update(update)

    assert list(follower._state.jobs[0].trajectory) == update.job_trajectory_appends["child-0"]
    assert all(
        row["result"]["content"][0]["text"] == "second attempt"
        for row in follower._state.jobs[0].trajectory
    )
    assert work.rows == 5, "the replacement froze something other than its own rows"


def test_a_seeded_page_is_not_matched_to_the_window_it_replaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``seed_job_trajectory`` installs a freshly frozen page, not an append."""
    wire = _Wire(children=1, rows=8)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    seeded = [_row(500 + index, text="fetched page") for index in range(4)]
    assert follower.seed_job_trajectory("child-0", seeded)
    update = wire.append_rows()
    work = _RowWork(monkeypatch)

    follower.apply_update(update)

    assert work.rows == 5, "the seeded window was extended without being frozen"
    assert list(follower._state.jobs[0].trajectory[:4]) == seeded


def test_a_replace_reseat_invalidates_every_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``replace``/``replace_and_notify`` build rows from a PAYLOAD."""
    wire = _Wire(children=2, rows=8)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    assert follower._follower_windows._by_job

    follower.replace(wire.state)
    assert not follower._follower_windows._by_job, "a payload re-seat kept a stale window"

    seeded_lengths = [len(job.trajectory) for job in follower._state.jobs]
    update = wire.append_rows()
    work = _RowWork(monkeypatch)
    follower.apply_update(update)
    # Every retained row is frozen again, which is the point: a delta after a
    # re-seat has no proof to build on, so the pre-seat window cannot be
    # extended even for one row. The count is the whole NEW window (the rows the
    # re-seat installed plus the delta's own appends), which is exactly the cost
    # the memo removes on every other delta.
    assert seeded_lengths == [9, 9]
    assert work.rows == sum(
        len(job.trajectory) for job in follower._state.jobs
    ), "a pre-seat window was reused after a re-seat"


def test_a_replaced_epoch_cannot_contribute_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lineage witness, at the memo's own boundary.

    Unreachable through ``apply_update`` today (an epoch move is a
    ``replace()``, which clears the memo outright), which is why it is driven
    here: the guard is what keeps a FUTURE path that re-seats ``_state`` directly
    from extending a previous lineage's window.
    """
    wire = _Wire(children=1, rows=8)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    prior = follower._state.jobs[0]
    assert follower._follower_windows.prove("child-0", prior, epoch="e1") is not None

    assert follower._follower_windows.prove("child-0", prior, epoch="e2") is None
    assert not follower._follower_windows._by_job
    assert follower._follower_windows._epoch == "e2"


def test_a_tail_witness_that_disagrees_refuses_the_window() -> None:
    """The second witness, at the memo's own boundary.

    ``_FrozenSequence`` is immutable, so a window whose identity matches cannot
    have a different last row — which is why this cannot be produced through the
    reducer. The witness exists for the reducer that installs a MUTABLE row
    container, where container identity alone stops being a proof; this pins that
    it refuses rather than trusting the container.
    """
    from local_operator.session.frontend_state import (
        _FollowerRowWindow,
        _FrozenSequence,
    )

    wire = _Wire(children=1, rows=8)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    prior = follower._state.jobs[0]

    entry = follower._follower_windows._by_job["child-0"]
    assert entry.rows is prior.trajectory
    follower._follower_windows._by_job["child-0"] = _FollowerRowWindow(
        entry.rows, _FrozenSequence(())  # a tail that is not this window's last row
    )

    assert follower._follower_windows.prove("child-0", prior, epoch="e1") is None
    assert "child-0" not in follower._follower_windows._by_job


def test_a_job_entering_the_roster_is_never_given_another_jobs_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A roster row with no previous canonical job extends nothing — while the
    children that ARE proven keep paying only their own appended row.

    The mixed roster is what makes this structural: a memo that had reused the
    proven prefix for the newcomer would freeze one row for it, and the two
    counts below separate that from the correct answer.
    """
    wire = _Wire(children=2, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    windows = [job.trajectory for job in follower._state.jobs]

    jobs = [*wire.jobs, JobState(id="newcomer", type="task", trajectory=[_row(77)])]
    update = wire.owner.mutate(jobs=jobs)
    assert update is not None
    work = _RowWork(monkeypatch)
    follower.apply_update(update)

    newcomer = next(job for job in follower._state.jobs if job.id == "newcomer")
    assert list(newcomer.trajectory) == [_row(77)]
    # The frame appends nothing to the proven children, so they freeze nothing,
    # and the newcomer pays only its own row — a memo that had reused a proven
    # child's prefix for it would freeze nothing at all here while shipping the
    # WRONG rows (which the content assertion above refuses).
    assert work.rows == 1, "the newcomer inherited a window or the proof was lost"
    assert newcomer.trajectory is not windows[0]
    assert len(follower._state.jobs) == 3


def test_a_job_leaving_the_roster_drops_its_entry() -> None:
    """Otherwise the memo pins a departed job's rows for the life of the store."""
    wire = _Wire(children=3, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    assert set(follower._follower_windows._by_job) == {"child-0", "child-1", "child-2"}

    update = wire.owner.mutate(jobs=wire.jobs[:2])
    assert update is not None
    follower.apply_update(update)

    assert set(follower._follower_windows._by_job) == {"child-0", "child-1"}
    assert [job.id for job in follower._state.jobs] == ["child-0", "child-1"]


def test_a_degraded_delta_drops_the_memo() -> None:
    """A shed body leaves the local window known-incomplete; the proof is not
    what makes it safe, so the memo is dropped rather than carried across a
    lineage the follower has declared untrustworthy."""
    wire = _Wire(children=2, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    assert follower._follower_windows._by_job

    shed = FrontendUpdate(
        epoch=wire.start.epoch,
        sequence=follower._state.sequence + 1,
        changes={},
        degraded=True,
        degraded_reason="oversized frame",
    )
    follower.apply_update(shed)

    assert not follower._follower_windows._by_job, "a shed frame kept a window"
    assert len(follower._state.jobs[0].trajectory) == 7, "the degraded arm changed state"


def test_a_follower_that_has_no_older_runtime_count_derives_it_from_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``JobState``'s before-validator derivation, kept meaning the local window.

    The wire normally carries the runtime's own count; an older runtime does not,
    and the reducer used to derive the count from the rebuilt rows. Rows no
    longer reach the validator, so the same derivation is done from the window
    that is actually installed.
    """
    wire = _Wire(children=1, rows=6)
    follower = _follower(wire)
    frame = wire.append_rows()
    stripped = frame.model_copy(
        update={
            "changes": {
                **frame.changes,
                "jobs": [
                    {k: v for k, v in row.items() if k != "trajectory_length"}
                    for row in frame.changes["jobs"]
                ],
            }
        }
    )
    follower.apply_update(stripped)

    assert follower._state.jobs[0].trajectory_length == len(follower._state.jobs[0].trajectory) == 7


@pytest.mark.parametrize("invalid_count", [None, "not-a-count", [], {}])
def test_invalid_explicit_count_refuses_entire_delta(invalid_count: Any) -> None:
    """Skipping old rows must not skip shell validation or commit half a delta."""
    wire = _Wire(children=2, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    previous = follower.state
    frame = wire.append_rows()
    valid_count = frame.changes["jobs"][1]["trajectory_length"]
    frame.changes["jobs"][1]["trajectory_length"] = invalid_count
    with pytest.raises(ValueError):
        follower.apply_update(frame)
    assert follower.state == previous
    assert follower._state.sequence == previous.sequence

    # A refused packet must leave the same sequence available for a corrected
    # retry, including the cached windows of jobs processed before the bad one.
    frame.changes["jobs"][1]["trajectory_length"] = valid_count
    follower.apply_update(frame)
    assert follower.state == wire.owner.state


def test_mutating_delta_and_public_shell_cannot_change_retained_rows() -> None:
    """The identity proof is sound only when neither input nor readers own rows."""
    wire = _Wire(children=1, rows=6)
    follower = _follower(wire)
    follower.apply_update(wire.append_rows())
    frame = wire.append_rows()
    follower.apply_update(frame)
    expected = follower.state
    frame.job_trajectory_appends["child-0"][-1]["result"]["content"][0]["text"] = "changed"
    public = follower.state
    with pytest.raises(ValueError, match="frozen"):
        public.jobs[0].label = "changed"
    with pytest.raises(TypeError):
        public.jobs[0].trajectory[-1]["result"]["content"][0]["text"] = "changed"
    assert follower.state == expected


# --------------------------------------------------------------------------
# The producer's append/replacement classifier at the CAP.
#
# `_Wire` grows its window forever, so the prefix test always held and the
# replacement arm was never exercised against a real rotation. A child that has
# been running a while is exactly the case that matters: eviction breaks the
# prefix on EVERY append, so a prefix-only classifier ships all 500 rows per job
# per frame (measured at ~634 KB/frame against 6.5 KB for the uncapped shape).
# --------------------------------------------------------------------------


def _stamped_row(index: int, stamp: int, *, text: str = "x" * 16) -> dict[str, Any]:
    """One retained row carrying the writer's own append stamp."""
    return {**_row(index, text=text), TRAJECTORY_SEQ_KEY: stamp}


def _stamped_state(children: int, rows: int) -> FrontendSessionState:
    return FrontendSessionState(
        session_id="capped-wire",
        epoch="e1",
        jobs=[
            JobState(
                id=f"child-{child}",
                type="task",
                label=f"child-{child}",
                status="running",
                trajectory=[_stamped_row(index, index) for index in range(rows)],
                trajectory_length=rows,
            )
            for child in range(children)
        ],
    )


class _CappedWire:
    """A producer whose window is FULL, so every append evicts from the front.

    The shape ``harness/subagent.py`` actually writes — stamp the event, append
    it, drop the overflow — driven through the production writer
    (``FrontendStateStore.mutate``) exactly as ``_Wire`` is, so the frames under
    test are the ones the wire carries rather than hand-built deltas.
    """

    def __init__(self, children: int, rows: int = TRAJECTORY_CAP) -> None:
        self.start = _stamped_state(children, rows)
        self.state = self.start
        self.owner = FrontendStateStore(self.start)
        self.jobs = list(self.start.jobs)
        self.relayed = [rows] * children

    def _publish(self, jobs: list[JobState]) -> FrontendUpdate:
        self.jobs = jobs
        update = self.owner.mutate(jobs=jobs)
        assert update is not None, "the producer published no frame"
        assert update.epoch == self.start.epoch
        self.state = self.owner.state
        return update

    def rotate(self, count: int = 1) -> FrontendUpdate:
        """Append ``count`` stamped rows per child and evict the overflow."""
        rotated = []
        for child, job in enumerate(self.jobs):
            rows = list(job.trajectory)
            for _ in range(count):
                rows.append(_stamped_row(10_000 + self.relayed[child], self.relayed[child]))
                self.relayed[child] += 1
            if len(rows) > TRAJECTORY_CAP:
                del rows[: len(rows) - TRAJECTORY_CAP]
            rotated.append(
                job.model_copy(update={"trajectory": rows, "trajectory_length": len(rows)})
            )
        return self._publish(rotated)


def test_a_capped_rotation_ships_the_appended_row_not_the_whole_window() -> None:
    """The defect this classifier exists for: a full window costs a tail."""
    wire = _CappedWire(children=3)
    follower = _follower(wire)
    follower.apply_update(wire.rotate())

    update = wire.rotate()

    assert update.job_trajectory_replacements == [], (
        "a provable cap rotation was reported as a replacement, so every frame "
        "carries the whole retained window"
    )
    assert all(len(rows) == 1 for rows in update.job_trajectory_appends.values())
    assert (
        len(update.model_dump_json()) < 8_000
    ), "the frame still carries the whole window rather than the appended row"


def test_a_hydrated_follower_equals_the_producer_through_cap_rotations() -> None:
    """Reconstruction is EXACT, not merely close — checked past the first tick."""
    wire = _CappedWire(children=2)
    follower = _follower(wire)
    for _ in range(8):
        follower.apply_update(wire.rotate())

    assert follower.state == wire.owner.state

    for count in (1, 2, 50, TRAJECTORY_CAP - 1):
        rotated = wire.rotate(count)
        follower.apply_update(rotated)
        assert follower.state == wire.owner.state, f"a +{count} rotation diverged"
        assert rotated.job_trajectory_replacements == []

    follower.apply_update(wire.rotate())
    assert follower._state.jobs[0].trajectory == wire.owner.state.jobs[0].trajectory


def test_a_follower_that_kept_no_memo_reconstructs_the_tail() -> None:
    """The existing receiver needs no new field: it already trims at the cap.

    A cold follower has no proof to consult, so it takes the ordinary rebuild
    path — appends applied to what it holds, then trimmed — which is the code an
    older release runs for every delta. Dropping the memo between ticks is the
    faithful way to exercise it without checking out a second tree.
    """
    wire = _CappedWire(children=2)
    follower = _follower(wire)
    for _ in range(3):
        follower.apply_update(wire.rotate())
        follower._follower_windows.reset(follower._state.epoch)

    update = wire.rotate()
    follower._follower_windows.reset(follower._state.epoch)
    follower.apply_update(update)

    assert update.job_trajectory_replacements == []
    assert follower._state.jobs[0].trajectory == wire.owner.state.jobs[0].trajectory


def test_a_capped_follower_keeps_the_rotated_rows_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transfer that matters: the suffix win now applies at the cap too."""
    wire = _CappedWire(children=1)
    follower = _follower(wire)
    follower.apply_update(wire.rotate())
    before = follower._state.jobs[0].trajectory
    # The frame is BUILT first and the counter installed after it, so only the
    # FOLLOWER's work is measured: building a frame runs the producer, whose own
    # freezing would otherwise land in this total and make a memo hit look like
    # a rebuild. Measured on the SECOND rotation because the first has no proof
    # to consult and pays one full freeze by design.
    update = wire.rotate()
    work = _RowWork(monkeypatch)

    follower.apply_update(update)

    after = follower._state.jobs[0].trajectory
    assert work.rows == 1, "a provable rotation re-froze the whole window"
    assert after[0] is before[1], "the rotated window was rebuilt rather than re-sliced"
    assert list(after) == list(wire.owner.state.jobs[0].trajectory)


def _capped_tail(old: list[Any], new: list[Any]) -> list[Any] | None:
    return module._capped_overlap_tail(old, new)


@pytest.mark.parametrize(
    ("prior", "appended"),
    [
        (TRAJECTORY_CAP, 1),
        (TRAJECTORY_CAP, 2),
        (TRAJECTORY_CAP, 50),
        (TRAJECTORY_CAP - 1, 2),
    ],
)
def test_the_classifier_proves_the_rotations_the_runtime_produces(
    prior: int, appended: int
) -> None:
    """Each cell is replayed through the receiver's own rule and compared.

    A rotation lands on a window of exactly the cap, so the overlap is
    ``CAP - appended`` and the tail is the rows the owner appended. The last
    cell CROSSES the cap: a 499-row window plus two appends is a full 500-row
    window, and requiring the prior window to be full as well would refuse a
    rotation the receiver handles exactly.
    """
    overlap = TRAJECTORY_CAP - appended
    old = [_stamped_row(index, index) for index in range(prior)]
    new = [
        _stamped_row(prior - overlap + step, prior - overlap + step)
        for step in range(TRAJECTORY_CAP)
    ]
    assert new[:overlap] == old[prior - overlap :]

    tail = _capped_tail(old, new)

    assert tail is not None, f"a provable rotation ({prior}->{len(new)}) was refused"
    assert tail == new[overlap:]
    assert len(tail) == appended
    assert list(list(old) + list(tail))[-TRAJECTORY_CAP:] == new


@pytest.mark.parametrize("prior", [TRAJECTORY_CAP, TRAJECTORY_CAP - 1])
def test_a_window_shorter_than_the_cap_is_still_a_replacement(prior: int) -> None:
    """A short window cannot be rebuilt by append+trim even as an equal suffix.

    The cap-equality guard is load-bearing rather than incidental: with fewer
    rows than the receiver keeps, its trim has nothing to drop, so the rows a
    front deletion removed would stay missing and the reconstruction would be a
    silently short window. Replace instead.
    """
    old = [_stamped_row(index, index) for index in range(prior)]
    new = old[3:] + [_stamped_row(30_000, 10_000)]
    assert len(new) < TRAJECTORY_CAP

    assert _capped_tail(old, new) is None


def test_an_unprovable_rotation_keeps_the_replacement() -> None:
    """Refusals are conservative, and a refusal is never the only correct answer.

    Each cell pins what the classifier owes its caller: a WRONG tail is
    unacceptable, so an overlap it cannot prove must fall back to the replacement
    it has always sent. Where a rotation happens to be provable anyway (a reorder
    that still leaves a genuine overlap), the tail it returns must reconstruct
    the new window exactly -- refusing a provable frame is allowed to cost bytes,
    never correctness, so both answers are checked against the receiver's rule
    rather than against a preferred branch.
    """
    old = [_stamped_row(index, index) for index in range(TRAJECTORY_CAP)]
    rotated = old[1:] + [_stamped_row(30_000, TRAJECTORY_CAP)]

    # An interior edit with both endpoints agreeing: the stamp distance still
    # proposes the right offset, and only the row comparison can refuse it.
    edited = [dict(row) for row in rotated]
    edited[10] = _stamped_row(10, 10, text="edited")
    assert _capped_tail(old, edited) is None, "an interior edit was accepted as a rotation"

    # Stamps that cannot propose an offset at all.
    unstamped = [
        {key: value for key, value in row.items() if key != TRAJECTORY_SEQ_KEY} for row in rotated
    ]
    assert _capped_tail(old, unstamped) is None
    assert _capped_tail(old, [{**rotated[0], TRAJECTORY_SEQ_KEY: True}, *rotated[1:]]) is None
    assert _capped_tail(old, [{**rotated[0], TRAJECTORY_SEQ_KEY: "7"}, *rotated[1:]]) is None

    # A reset or repeated stamp sequence is a hint that cannot be trusted, and
    # the offset it proposes fails the comparison.
    reset = [{**row, TRAJECTORY_SEQ_KEY: 0} for row in rotated]
    assert _capped_tail(old, reset) is None

    # No eviction at all, and nothing appended: there is no tail to ship.
    assert _capped_tail(old, old) is None
    assert _capped_tail([], rotated) is None

    # A prior window SHORTER than the cap is legal — the guard is on the NEW
    # window, because that is the one the receiver trims against — as is a
    # reorder that still leaves a genuine overlap. Both may be accepted, and an
    # accepted tail must land on the new window under the receiver's own rule.
    for name, prior, candidate in (
        ("short prior", old[:100], rotated),
        ("reorder", old, [*rotated[1:], rotated[0]]),
    ):
        tail = _capped_tail(prior, candidate)
        assert tail is not None, f"{name}: a provable rotation was refused"
        assert list(list(prior) + list(tail))[-TRAJECTORY_CAP:] == candidate, name


def test_a_long_running_child_costs_one_row_per_frame_for_thousands_of_events() -> None:
    """The horizon this exists for: a child past the cap, for a long time.

    A cap rotation is not a transient — a working child sits at the cap for the
    whole of its run, so the per-frame cost has to stay the appended row rather
    than the window, for every frame and not just the first few.
    """
    wire = _CappedWire(children=3)
    follower = _follower(wire)
    follower.apply_update(wire.rotate())
    emitted = 0
    for _ in range(20):
        update = wire.rotate(50)
        assert update.job_trajectory_replacements == []
        assert sum(len(rows) for rows in update.job_trajectory_appends.values()) == 150
        emitted += sum(len(rows) for rows in update.job_trajectory_appends.values())
        follower.apply_update(update)
        assert follower.state == wire.owner.state

    assert emitted == 3 * 1000, emitted
    assert follower._state.jobs[0].trajectory == wire.owner.state.jobs[0].trajectory


def test_rows_of_equal_value_still_prove_an_overlap_by_position() -> None:
    """Equal ROW VALUES must not let the classifier pick a different offset.

    A child can emit the same event twice, so rows can agree field for field. When
    they do, a comparison alone cannot choose between offsets — every candidate
    compares equal — so the stamp is what pins the offset and the reconstruction is
    then exact. The property this pins is that the returned tail starts at the
    offset the stamps name: a classifier free to slide to a larger, equally-equal
    offset would still "pass" a value comparison and hand the receiver a window
    that is right by luck rather than by proof.
    """
    old = [_stamped_row(index, index, text="identical") for index in range(TRAJECTORY_CAP)]
    # A normal one-row rotation in which every value is interchangeable.
    new = [_stamped_row(index, index, text="identical") for index in range(1, TRAJECTORY_CAP + 1)]

    tail = _capped_tail(old, new)

    assert tail is not None
    assert len(tail) == 1, "the classifier slid to a larger, equally-equal overlap"
    assert tail == new[TRAJECTORY_CAP - 1 :]
    assert list(list(old) + list(tail))[-TRAJECTORY_CAP:] == new


def test_a_job_that_leaves_the_roster_takes_its_window_with_it() -> None:
    """A returning job must not be handed a window the memo kept for it.

    The memo is keyed by job id, and a job id can come back — a re-added child,
    a re-used slot. `retain` drops entries for jobs no longer on the roster, so a
    returning job pays a fresh freeze instead of inheriting rows from a window
    that was never its own.
    """
    wire = _CappedWire(children=2)
    follower = _follower(wire)
    for _ in range(3):
        follower.apply_update(wire.rotate())
    before = follower._state.jobs[0].trajectory

    # The job leaves the roster, then a different job arrives with the same id.
    wire.jobs = [wire.jobs[1]]
    follower.apply_update(wire._publish(list(wire.jobs)))
    assert "child-0" not in {job.id for job in follower._state.jobs}

    returned = [_stamped_row(40_000 + index, 5_000 + index) for index in range(TRAJECTORY_CAP)]
    wire.jobs = [
        JobState(
            id="child-0",
            type="task",
            label="child-0",
            status="running",
            trajectory=returned,
            trajectory_length=len(returned),
        ),
        *wire.jobs,
    ]
    follower.apply_update(wire._publish(list(wire.jobs)))

    assert follower._state.jobs[0].trajectory == wire.owner.state.jobs[0].trajectory
    assert (
        follower._state.jobs[0].trajectory != before
    ), "a returning job was handed the window of the job that used to hold its id"


def test_a_window_cleared_to_empty_is_a_replacement_not_a_rotation() -> None:
    """An emptied window has no suffix, and the receiver must be told so."""
    wire = _CappedWire(children=1)
    follower = _follower(wire)
    follower.apply_update(wire.rotate())

    wire.jobs = [wire.jobs[0].model_copy(update={"trajectory": [], "trajectory_length": 0})]
    update = wire._publish(list(wire.jobs))
    follower.apply_update(update)

    assert update.job_trajectory_replacements == ["child-0"]
    assert follower._state.jobs[0].trajectory == ()
    assert follower._state.jobs[0].trajectory_length == 0


def test_a_window_shortened_to_the_cap_minus_one_is_not_rebuilt_by_append() -> None:
    """Below-cap shortening must not be sold as a rotation, even as a suffix."""
    old = [_stamped_row(index, index) for index in range(TRAJECTORY_CAP)]
    shortened = old[1:]
    assert len(shortened) == TRAJECTORY_CAP - 1

    assert _capped_tail(old, shortened) is None
    assert _capped_tail(old, [*shortened, _stamped_row(30_000, TRAJECTORY_CAP)]) is not None


def test_a_full_window_never_loses_a_row_to_the_classifier() -> None:
    """A replacement and a proven tail must land on the same canonical state.

    The comparison is a replacement AGAINST a tail rather than two tails, so the
    two arms are proven equivalent rather than merely both passing.
    """
    tail_wire = _CappedWire(children=1)
    tail_follower = _follower(tail_wire)
    for _ in range(4):
        tail_follower.apply_update(tail_wire.rotate())
    tail_follower.apply_update(tail_wire.rotate())

    # The same sequence, expressed as the frame a REFUSING classifier emits: the
    # appended rows ARE the whole new window and the marker is set. That is the
    # shape the replacement arm has always carried, so the two frames differ only
    # in how they were classified.
    replacement_wire = _CappedWire(children=1)
    replacement_follower = _follower(replacement_wire)
    for _ in range(4):
        replacement_follower.apply_update(replacement_wire.rotate())
    refused = replacement_wire.rotate()
    window = replacement_wire.owner.state.jobs[0].trajectory
    refused.job_trajectory_appends["child-0"] = [module._wire_value(row) for row in window]
    refused.job_trajectory_replacements = ["child-0"]
    replacement_follower.apply_update(refused)

    assert replacement_follower.state == tail_follower.state
    assert replacement_wire.owner.state == tail_wire.owner.state
