"""The bounded residency of the fail-closed busy probe.

The measured state this closes (2026-09-17): 34 of 57 live runtimes on the
reference host had no discovery record in any namespace, several were
``ppid=1`` and had been alive over ten hours. A runtime's own reaper is the only
party that may end it, and its first term is FAIL-CLOSED: ``_work_in_flight``
answers "work is in flight, stay resident" for every sample whose probe raises.
That answer is right for one tick and wrong for the life of the machine, and
nothing outside could contradict it.

So the fail-closed answer STAYS for the immediate decision and is COUNTED. These
tests pin the three properties that make the count safe rather than merely
finite: it advances only on consecutive failures, a healthy sample clears it,
and a runtime that is legitimately wanted — a viewer attached, or a wake inside
the warm window — is never spent against the bound. The last test is the one
that matters most: the exit NAMES itself, because an exit that says nothing
about itself is the failure the whole of ``process``'s instrumentation exists to
remove.
"""

from __future__ import annotations

import time

import pytest

from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import (
    PROBE_DEFER_BOUND,
    _idle_exit_reason,
    _should_exit,
    _work_in_flight,
)


class Raises:
    """A handle whose busy probe is unusable — the pinned state, in one object."""

    def __init__(self, error: str = "counter gone") -> None:
        self.error = error
        self.calls = 0

    def is_busy(self) -> bool:
        self.calls += 1
        raise RuntimeError(self.error)


class Answers:
    def __init__(self, busy: bool = False, *, next_wake_ms: int | None = None) -> None:
        self._busy = busy
        self._next_wake_ms = next_wake_ms

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._next_wake_ms


class Watcher:
    """A runtime with ``attaches`` follower terminals."""

    def __init__(self, attaches: int = 0) -> None:
        self._attaches = attaches

    def attach_clients(self) -> int:
        return self._attaches


@pytest.fixture(autouse=True)
def _clean_streak():
    # The streak is PROCESS state, because the thing it counts is one runtime per
    # process; a test that leaves it advanced would hand the next test a runtime
    # that is already at the bound.
    child_mod._probe_defers.reset()
    yield
    child_mod._probe_defers.reset()


def test_fail_closed_answer_is_unchanged_for_a_single_failure() -> None:
    # The immediate decision must not move: uncertainty still means "stay".
    assert _work_in_flight(Raises()) is True
    assert _should_exit(Raises(), Watcher()) is False


def test_streak_advances_only_to_the_bound_then_the_runtime_may_leave() -> None:
    handle, runtime = Raises(), Watcher()
    for _ in range(PROBE_DEFER_BOUND - 1):
        assert _should_exit(handle, runtime) is False
    # The bound-th visit is the verdict: a probe that has answered nothing for the
    # whole streak is broken, not busy.
    assert _should_exit(handle, runtime) is True


def test_a_healthy_sample_clears_the_streak() -> None:
    # A probe that ANSWERS — either way — is not pinning anything, so a later
    # streak starts from zero rather than accumulating credit across a healthy
    # window.
    for _ in range(PROBE_DEFER_BOUND - 1):
        _should_exit(Raises(), Watcher())
    assert _should_exit(Answers(busy=True), Watcher()) is False
    assert child_mod._probe_defers.streak == 0
    # Almost a full streak again, and still held: nothing carried over.
    for _ in range(PROBE_DEFER_BOUND - 1):
        assert _should_exit(Raises(), Watcher()) is False


def test_an_attached_viewer_spends_no_streak() -> None:
    # A viewer is a reason to stay that has nothing to do with the probe, so the
    # bound must not be spent on it: otherwise a broken probe plus a person at
    # the keyboard would end the runtime they are using.
    handle = Raises()
    for _ in range(PROBE_DEFER_BOUND * 2):
        assert _should_exit(handle, Watcher(attaches=1)) is False


def test_an_imminent_wake_spends_no_streak() -> None:
    # Term 2 is the same asymmetry: a runtime about to fire its own wake is
    # wanted, and the streak is reset rather than counted through it.
    handle = Raises(error="handled elsewhere")
    handle.next_wake_due_at = lambda: int(time.time() * 1000) + 60_000  # type: ignore
    for _ in range(PROBE_DEFER_BOUND * 2):
        assert _should_exit(handle, Watcher()) is False


def test_a_legitimate_hold_between_failures_resets_the_streak() -> None:
    # Measured through a viewer, which is the one hold a test can raise and lower.
    handle = Raises()
    for _ in range(PROBE_DEFER_BOUND - 1):
        assert _should_exit(handle, Watcher()) is False
    assert _should_exit(handle, Watcher(attaches=2)) is False
    assert child_mod._probe_defers.streak == 0
    for _ in range(PROBE_DEFER_BOUND - 1):
        assert _should_exit(handle, Watcher()) is False


def test_the_bound_is_a_minute_at_the_reapers_own_cadence() -> None:
    # The count is a LINEAR clock because the reaper is the only sampler: at
    # REAP_CHECK_S per sample the bound is the documented ~60 s. Asserted as a
    # range so it stays honest if the cadence moves, without pinning the exact
    # figure in a test that does not own the constant.
    assert 30.0 <= PROBE_DEFER_BOUND * child_mod.REAP_CHECK_S <= 120.0


def test_exit_reason_names_the_probe_when_the_bound_held_it() -> None:
    # THE EXIT MUST BE ATTRIBUTABLE. A quiet exit that cannot say which term held
    # it is the failure the journal and the reason argument exist to remove.
    assert _idle_exit_reason() == "idle-exit"
    for _ in range(PROBE_DEFER_BOUND):
        _should_exit(Raises(error="scheduler torn"), Watcher())
    reason = _idle_exit_reason()
    assert reason != "idle-exit"
    assert "busy probe unusable" in reason
    assert str(PROBE_DEFER_BOUND) in reason
    assert "scheduler torn" in reason
