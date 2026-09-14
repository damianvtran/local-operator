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
from typing import Any

import pytest

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


def _follower(wire: _Wire) -> FrontendStateStore:
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
