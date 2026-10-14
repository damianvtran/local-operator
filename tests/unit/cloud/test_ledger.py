"""``CloudLedger``: the write-once rule, the journal, and the record's mapping.

The store discipline here is ``network/move_queue.py``'s (§3.6) and the two
properties that make it worth copying are proved on THIS record rather than
asserted: a terminal state is WRITE-ONCE, the transition journal is APPEND-ONLY,
and a read-modify-write holds the record's cross-process lock for the whole of
itself — not merely for the write, which is the difference between a lock and a
decoration.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.cloud import ledger as ledger_mod
from local_operator.cloud.fake import FakeComputeProvider
from local_operator.cloud.ledger import (
    CloudLedger,
    lock_for,
    read_request,
    request_path,
)
from local_operator.cloud.provider import CloudRefused
from local_operator.cloud.types import (
    TASK_STATES,
    ComputeTask,
    PoolRequest,
    TaskState,
    member_lifecycle_for,
)
from local_operator.wakes.lock import WakeLockBusy

#: §3.4's table, transcribed. `ever_ran` is what decides the three collapse
#: states: a task that never reached `running` never produced a member row, so
#: there is nothing to expire and the answer is None rather than a lifecycle.
MEMBER_TABLE: dict[str, str | None] = {
    "queued": None,
    "provisioning": "provisioning",
    "running": "active",
    "draining": "draining",
    "expired": "expired",
    "failed": None,
    "cancelled": None,
    "lost": None,
}

#: The three states that mean "it ran, then it ended somehow".
COLLAPSED = ("failed", "cancelled", "lost")


def _lock_is_held(root: Path, request_id: str) -> bool:
    """Whether a peer currently holds this record's lock — a probe, not a sleep.

    A second ``open`` gets its own file description, so an exclusive ``flock``
    from here fails even against a lock this very process holds, which is what
    makes this a real probe rather than a same-process coincidence.
    """
    fd = os.open(lock_for(root, request_id).path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        # Closed on every path: an unclosed description would keep a lock
        # contender alive past the test that took it.
        os.close(fd)


def test_terminal_states_are_write_once(
    tmp_path: Path, request_factory: Callable[..., PoolRequest]
) -> None:
    """``expired`` after ``failed`` is a no-op, with the reason preserved."""
    ledger = CloudLedger(tmp_path)
    task = ledger.create(request_factory())
    assert ledger.transition(task.request_id, "provisioning") is not None
    assert (
        ledger.transition(
            task.request_id,
            "failed",
            detail="the container exited 1",
            reason="failed",
            exit_code=1,
        )
        is not None
    )
    failed = ledger.read(task.request_id)
    assert failed is not None
    assert (failed.state, failed.stop_reason, failed.exit_code) == ("failed", "failed", 1)

    # The mark of the write-once rule: nothing advanced, so the caller is told so.
    assert ledger.transition(task.request_id, "expired", detail="drained", reason="drained") is None
    # A SAME-state call is refused as well. That is what makes the explicit guard
    # load-bearing rather than decorative: without it the same-state path would
    # stamp ``last_seen_at`` and write the file, so a terminal record would still
    # be changing after it had ended.
    assert ledger.transition(task.request_id, "failed", detail="again") is None
    after = ledger.read(task.request_id)
    assert after is not None
    assert after.state == "failed"
    assert after.stop_reason == "failed", "a later transition may not overwrite the reason"
    assert after.stopped_at == failed.stopped_at
    assert after.last_seen_at == failed.last_seen_at
    assert after.events == failed.events, "a refused transition may not journal anything"


def test_transition_journal_is_append_only(
    tmp_path: Path, request_factory: Callable[..., PoolRequest]
) -> None:
    """``events`` grows by one per accepted transition and never edits an entry."""
    ledger = CloudLedger(tmp_path)
    task = ledger.create(request_factory())
    first = ledger.read(task.request_id)
    assert first is not None
    assert [event["phase"] for event in first.events] == ["queued"]

    steps: tuple[tuple[TaskState, str], ...] = (
        ("provisioning", "submit returned a ref"),
        ("running", ""),
        ("draining", "asked to stop"),
        ("expired", "checkpointed"),
    )
    for state, detail in steps:
        before = ledger.read(task.request_id)
        assert before is not None
        assert ledger.transition(task.request_id, state, detail=detail) is not None
        after = ledger.read(task.request_id)
        assert after is not None
        assert len(after.events) == len(before.events) + 1, state
        # The earlier entries are byte-for-byte what they were: an append that
        # rewrote a prefix would hide the sequence the journal exists to show.
        assert after.events[: len(before.events)] == before.events

    final = ledger.read(task.request_id)
    assert final is not None
    events = final.events
    assert [event["phase"] for event in events] == [
        "queued",
        "provisioning",
        "running",
        "draining",
        "expired",
    ]
    assert [event["detail"].get("message", "") for event in events] == [
        "",
        "submit returned a ref",
        "",
        "asked to stop",
        "checkpointed",
    ]
    assert [event["at"] for event in events] == sorted(event["at"] for event in events)


@pytest.mark.parametrize("state", sorted(TASK_STATES))
def test_member_lifecycle_mapping_matches_the_table_in_the_design(state: str) -> None:
    """§3.4's mapping, over all eight states, on both sides of ``ever_ran``."""
    assert set(MEMBER_TABLE) == set(TASK_STATES), "the table under test is not the whole vocabulary"
    expected = MEMBER_TABLE[state]
    assert member_lifecycle_for(state) == expected
    if state in COLLAPSED:
        assert member_lifecycle_for(state, ever_ran=True) == "expired"
    else:
        assert member_lifecycle_for(state, ever_ran=True) == expected


def test_two_processes_cannot_interleave_a_transition(
    tmp_path: Path, request_factory: Callable[..., PoolRequest], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock is held ACROSS the read-modify-write, and a peer's hold is refused."""
    writer = CloudLedger(tmp_path)
    task = writer.create(request_factory(request_id="req_5f5f5f5f"))
    peer = CloudLedger(tmp_path, lock_timeout_s=0.1)

    held_during_read: list[bool] = []
    real_read = ledger_mod.read_task

    def probing_read(root: Any, request_id: str) -> ComputeTask | None:
        held_during_read.append(_lock_is_held(Path(root), request_id))
        return real_read(root, request_id)

    monkeypatch.setattr(ledger_mod, "read_task", probing_read)
    assert writer.transition(task.request_id, "provisioning") is not None
    monkeypatch.undo()
    assert held_during_read == [True], (
        "the record's lock must be held while the record is READ, not only while it "
        "is written: a bare read-modify-write would let two transitions read the "
        "same state and one of them silently vanish"
    )

    with lock_for(tmp_path, task.request_id, timeout_s=0.1):
        with pytest.raises(WakeLockBusy):
            peer.transition(task.request_id, "running")
    current = writer.read(task.request_id)
    assert current is not None
    assert current.state == "provisioning", "the refused writer may not have written"


def test_cancel_reports_why_it_could_not_stop_the_task(
    tmp_path: Path, request_factory: Callable[..., PoolRequest]
) -> None:
    """``cancel``'s vocabulary is ``move_queue``'s, because the question is the same."""
    ledger = CloudLedger(tmp_path)
    task = ledger.create(request_factory())
    cancelled, outcome = ledger.cancel(task.request_id)
    assert (outcome, cancelled is not None and cancelled.state) == ("cancelled", "cancelled")

    # A second press is not "too late" — it has not started, it already ended.
    _, again = ledger.cancel(task.request_id)
    assert again == "already"

    assert ledger.cancel("req_there_is_no_such_record") == (None, "absent")

    late = ledger.create(request_factory(request_id="req_11112222"))
    for state in ("provisioning", "running", "draining"):
        ledger.transition(late.request_id, state)
    held, outcome = ledger.cancel(late.request_id)
    assert outcome == "too_late"
    assert held is not None and held.state == "draining"


def test_expire_stalled_fails_a_task_that_never_reached_running(
    tmp_path: Path, request_factory: Callable[..., PoolRequest]
) -> None:
    """§3.4's provisioning deadline, and the one task it must never touch."""
    ledger = CloudLedger(tmp_path)
    stalled = ledger.create(request_factory(request_id="req_aaaa0001"))
    ledger.transition(stalled.request_id, "provisioning")
    moved = ledger.create(request_factory(request_id="req_aaaa0002"))
    ledger.transition(moved.request_id, "provisioning")
    ledger.transition(moved.request_id, "running")

    assert ledger.expire_stalled(deadline_s=0.0) == [stalled.request_id]
    expired = ledger.read(stalled.request_id)
    assert expired is not None
    assert (expired.state, expired.stop_reason) == ("failed", "provision_deadline")
    live = ledger.read(moved.request_id)
    assert live is not None and live.state == "running", "a running task is not stalled"


@pytest.mark.asyncio
async def test_reconcile_folds_a_live_task_and_refuses_to_guess_when_the_provider_is_down(
    tmp_path: Path, request_factory: Callable[..., PoolRequest]
) -> None:
    """Re-read and report; never re-drive, and never guess (§9.2)."""
    ledger = CloudLedger(tmp_path)
    task = ledger.create(request_factory(request_id="req_beef0001"))
    provider = FakeComputeProvider()
    ref = await provider.submit(request_factory(request_id=task.request_id))
    assert ledger.transition(
        task.request_id, "provisioning", provider=provider.name, provider_ref=ref
    )
    recorded = ledger.read(task.request_id)
    assert recorded is not None and recorded.provider_ref == ref, (
        "the provider reference arrives WITH the transition submit() caused, so it "
        "is written by the same writer rather than by a second one"
    )

    provider.mark(ref, "running")
    assert (await ledger.reconcile(provider))[task.request_id] == "running"
    reconciled = ledger.read(task.request_id)
    assert reconciled is not None and reconciled.state == "running"

    class _Down(FakeComputeProvider):
        async def status(self, provider_ref: str) -> Any:
            raise CloudRefused("provider_unavailable", "the ECS API is not answering")

    outcomes = await ledger.reconcile(_Down())
    assert outcomes[task.request_id].startswith("unavailable:")
    untouched = ledger.read(task.request_id)
    assert untouched is not None
    assert untouched.state == "running", "an unreachable provider changes nothing"


def test_list_prunes_the_oldest_terminal_records(
    tmp_path: Path, request_factory: Callable[..., PoolRequest], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record log is bounded, and a prune takes the intent with the task."""
    monkeypatch.setattr(ledger_mod, "MAX_CLOUD_TASKS", 2)
    ledger = CloudLedger(tmp_path)
    ids: list[str] = []
    for index in range(3):
        task = ledger.create(request_factory(request_id=f"req_prune{index:03d}"))
        ledger.transition(task.request_id, "failed")
        ids.append(task.request_id)

    kept = [task.request_id for task in ledger.list_tasks()]
    assert ids[0] not in kept, "the oldest terminal record is the one that goes"
    assert ids[1:] == sorted(kept)
    assert read_request(tmp_path, ids[0]) is None
    assert not request_path(tmp_path, ids[0]).exists()
