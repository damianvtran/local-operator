"""PR A of the session-survival design must be provably inert on residency.

THE GUARD THIS FILE IS. ``docs/design-session-survival.md`` adds instrumentation
(a turn journal, a boot record, a classifier preference) and explicitly does NOT
change the residency policy: ``_should_exit`` and all three of its terms,
``DEFAULT_GRACE_S``, ``WARM_WINDOW_S`` and the build-watch timings stay exactly
as they are, and ``docs/design-idle-reap.md``'s decision about who may initiate a
reap is untouched. That is a claim about a diff, and a claim about a diff cannot
be re-checked by reading it: it is re-checked by PINNING THE VALUES AND THE
PREDICATE'S OWN BEHAVIOUR here, so the next change that quietly turns
instrumentation into a lifetime change fails on the value it moved.

WHAT IS PINNED, AND WHY EACH HALF IS WORTH A TEST
-------------------------------------------------
* the CONSTANTS, as literals. A tuning change is legal on its own merits and
  illegal in this PR; the literals make the intent explicit rather than implied,
  and a genuine retune has to come here and say so.
* the PREDICATE'S TRUTH TABLE, through the three terms in their documented
  order. Term 1 alone decides whether work is destroyed (a viewer leaving never
  aborts a turn), and each term below it can only make the answer "stay". A
  change from "all three" to "any of them" would keep every constant listed
  above and still end sessions under a running turn, which is why the behaviour
  is pinned and not just the numbers.
* the PROBE DIRECTIONS. Terms 2 and 3 fail OPEN (a broken accessor must not pin
  a runtime) while term 1 fails CLOSED (an unreadable work probe must never be
  the thing that ends a turn). Both directions are load-bearing and both are
  invisible to a values-only test.

The instruments themselves are covered by ``test_turn_journal.py``; nothing here
touches them — and the last test in this file asserts that they stay out of the
predicate, because a residency check that consults the journal stops detecting a
leak between the two.
"""

from __future__ import annotations

import time

from local_operator import buildwatch
from local_operator.session.runtime import process
from local_operator.session.runtime.types import SIGNAL_DRAIN_S


class _Handle:
    """A handle answering the two probes the predicate actually makes.

    ``next_wake_due_at`` is read from the HANDLE, not the runtime
    (``buildwatch.wake_within_window``), in epoch milliseconds — a double that
    put it on the runtime would make term 2 permanently false and the truth
    table below would pass while testing two terms out of three.
    """

    def __init__(self, *, busy: bool = False, wake_in_ms: int | None = None) -> None:
        self.busy = busy
        self._wake_in_ms = wake_in_ms

    def is_busy(self) -> bool:
        return self.busy

    def next_wake_due_at(self) -> int | None:
        if self._wake_in_ms is None:
            return None
        return int(time.time() * 1000) + self._wake_in_ms


class _Runtime:
    """A runtime whose only input is the interactive viewer count."""

    def __init__(self, *, viewers: int = 0) -> None:
        self._viewers = viewers

    def attach_clients(self) -> int:
        return self._viewers


def test_the_grace_and_warm_window_constants_are_unchanged() -> None:
    """The literals the design froze, asserted as literals."""
    assert process.DEFAULT_GRACE_S == 3.0
    assert process.REAP_CHECK_S == 0.25
    assert process.WARM_WINDOW_S == 90.0
    assert buildwatch.WARM_WINDOW_S == 90.0, "the residency term moved module, not value"


def test_the_build_watch_timings_are_unchanged() -> None:
    """``design-runtime-autorefresh``'s timings are untouched by this PR."""
    assert process.BUILD_CHECK_S == 5.0
    assert process.BUILD_SETTLE_S == 10.0
    assert process.BUILD_STAGGER_S == 20.0
    assert buildwatch.BUILD_CHECK_S == 5.0
    assert buildwatch.BUILD_SETTLE_S == 10.0
    assert buildwatch.BUILD_STAGGER_S == 20.0


def test_the_signal_drain_bound_is_unchanged() -> None:
    """The receiver-side work-aware signal path is not part of this PR."""
    assert SIGNAL_DRAIN_S == 120.0


def test_should_exit_requires_all_three_terms() -> None:
    """Idle, no wake due and no viewer: the ONLY combination that exits."""
    assert process._should_exit(_Handle(), _Runtime()) is True

    # Term 1 is work, and it alone holds the process.
    assert process._should_exit(_Handle(busy=True), _Runtime()) is False
    # Term 2 is a wake about to fire, read through the handle.
    assert process._should_exit(_Handle(wake_in_ms=1_000), _Runtime()) is False
    # A wake OUTSIDE the warm window does not hold it.
    assert process._should_exit(_Handle(wake_in_ms=int(91.0 * 1000)), _Runtime()) is True
    # Term 3 is an interactive viewer.
    assert process._should_exit(_Handle(), _Runtime(viewers=1)) is False
    # Work outranks every other term: busy, a wake due and a viewer attached
    # still stays.
    assert process._should_exit(_Handle(busy=True, wake_in_ms=1_000), _Runtime(viewers=1)) is False


def test_work_alone_decides_whether_a_turn_could_be_lost() -> None:
    """Term 1 is checked FIRST and ALONE, and it fails CLOSED."""
    assert process._should_exit(_Handle(busy=True), _Runtime(viewers=1)) is False

    class _Broken:
        def is_busy(self) -> bool:
            raise RuntimeError("probe exploded")

    # An unreadable probe means work: uncertainty must never license an exit
    # under a running turn.
    assert process._work_in_flight(_Broken()) is True
    # An absent probe is the long-standing treatment of a reduced handle.
    assert process._work_in_flight(object()) is False


def test_the_readiness_terms_fail_open() -> None:
    """A broken wake/viewer accessor must not pin the runtime it was asked about."""

    class _BrokenWake:
        def is_busy(self) -> bool:
            return False

        def next_wake_due_at(self) -> int:
            raise RuntimeError("wake accessor exploded")

    class _BrokenViewers:
        def attach_clients(self) -> int:
            raise RuntimeError("viewer probe exploded")

    assert process._should_exit(_BrokenWake(), _Runtime()) is True
    assert process._should_exit(_Handle(), _BrokenViewers()) is True


def test_no_journal_binding_reaches_the_residency_predicate() -> None:
    """The instruments are not wired into the predicate, structurally.

    A source assertion rather than a behavioural one, and deliberately so:
    ``_should_exit`` reads exactly two objects (the handle and the runtime), so a
    reference to the journal inside it could only arrive by someone editing the
    predicate itself — which is the change this file exists to catch. Named
    imports of the instrument modules are what it looks for, because a `getattr`
    probe spelled out at a call site is not how this module reaches its inputs.
    """
    import inspect

    source = inspect.getsource(process)
    start = source.index("def _work_in_flight")
    end = source.index("async def _clean_exit")
    assert (
        "journal" not in source[start:end]
    ), "the residency predicate and the work probe must not consult the instruments"
