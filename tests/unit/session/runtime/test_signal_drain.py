"""A termination signal must not destroy work the runtime would not lose.

The incident these tests exist for, measured on 2026-09-14: a single broadcast
sweep SIGTERM'd 21 live runtimes within 6 ms and cut 32 turns off, every one
recorded ``kind=error, cause=runtime-shutdown``. The graceful paths were always
work-aware — the reaper consults ``may_refresh``/``_should_exit`` and refuses to
exit under a live turn — while ``amain``'s signal handler set the stop event
outright. The asymmetry WAS the bug: the same event the reaper defers to a safe
boundary was fatal when it arrived as a signal.

What is pinned here:

* one work predicate (``_work_in_flight``), shared by the reaper's exit
  predicate and the signal drain, failing closed on an unreadable probe;
* a signal with work in flight COMMITS at the signal — announces the departure
  and latches against new work — and then waits for the boundary, which is the
  one commit point it shares with the build-replaced drain
  (``_commit_to_leaving``);
* the commit is NOT a cut-off note: ``begin_retire`` (the latch that writes the
  cause ``Session._classify_cut_off`` consumes) is reached only at the boundary,
  so a turn that COMPLETES inside the drain is never published as an error;
* the wait is BOUNDED, and expiry disposes exactly as the signal path always
  did, with no retirement latched for an exit that did interrupt work.

The wiring itself (a real SIGTERM to a real runtime parked in a real tool) is
``tests/e2e/test_signal_drain_e2e.py``; these are the unit-level properties.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from local_operator.session.runtime import process
from local_operator.session.runtime.types import LEAVING_ON_SIGNAL
from local_operator.session.runtime.process import (
    _commit_to_leaving,
    _drain_for_signal,
    _should_exit,
    _work_in_flight,
)


class _WorkHandle:
    """A handle whose work predicate is a switch, and which records its latches.

    Both halves of the drain's decision are switches: ``is_busy`` (would the
    dispose lose work?) is what the signal handler asks, and ``may_refresh``
    (may this runtime act now?) is what the drain's boundary asks. They are
    modelled separately because the production handle answers them from
    different predicates and a stub that conflated them would hide exactly the
    drift this module's one-work-predicate rule exists to prevent.
    """

    def __init__(
        self,
        *,
        busy: bool = False,
        events: list[str] | None = None,
        drains: bool = True,
    ) -> None:
        self.busy = busy
        self.events = events if events is not None else []
        self.probes = 0
        self.drains: bool = drains
        self.drain_latches: list[tuple[str, str]] = []
        self.retires: list[tuple[str, str]] = []
        #: The production handle's own published state: True once a drain has
        #: been latched, which is what a second trigger must respect.
        self._draining = False

    def is_busy(self) -> bool:
        self.probes += 1
        return self.busy

    def may_refresh(self) -> str:
        return "busy" if self.busy else ""

    def begin_drain(self, cause: str, detail: str = "") -> bool:
        self.drain_latches.append((cause, detail))
        self.events.append(f"drain:{cause}")
        if not self.drains:
            return False
        self._draining = True
        return True

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.retires.append((cause, detail))
        self.events.append(f"latch:{cause}")
        return True


class _NoDrainHandle:
    """A handle that never grew the drain latch (an older or reduced host).

    Spelled out rather than subclassed: ``begin_drain`` has to be ABSENT, and an
    inherited method cannot be deleted from an instance.
    """

    def __init__(self, *, busy: bool = False) -> None:
        self.busy = busy

    def is_busy(self) -> bool:
        return self.busy

    def may_refresh(self) -> str:
        return "busy" if self.busy else ""


class _RaisingHandle:
    """A handle whose probe is broken — the shape ``is_busy`` fails closed on."""

    def is_busy(self) -> bool:
        raise RuntimeError("probe exploded")


class _RecordingRuntime:
    """A runtime that records the retirement announcements it is asked to send."""

    def __init__(self, events: list[str] | None = None) -> None:
        #: ``(reason, draining, leaving)`` — the phrase is recorded because it is
        #: half of the one commit the seam makes: a stub that silently swallowed
        #: the keyword would let a refactor drop the record publication while the
        #: frame still went out (which is exactly what the e2e visibility cell
        #: caught during the rebase).
        self.announced: list[tuple[str, bool, str]] = []
        self.events = events if events is not None else []
        self._boot_build: Any = None

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.announced.append((reason, draining, leaving))
        self.events.append(f"announce:{reason}")


class _RaisingRuntime:
    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        raise RuntimeError("the viewer's writer is gone")


# --- one work predicate ------------------------------------------------------


def test_the_work_predicate_reads_the_handle() -> None:
    assert _work_in_flight(_WorkHandle(busy=True)) is True
    assert _work_in_flight(_WorkHandle(busy=False)) is False


def test_a_handle_without_the_probe_has_no_work_in_flight() -> None:
    """Reduced handles and older implementations: unknown is not "busy"."""

    class _Bare:
        pass

    assert _work_in_flight(_Bare()) is False


def test_a_broken_probe_fails_closed() -> None:
    """Uncertainty keeps the runtime working: ``is_busy``'s own contract.

    The consequence is asserted too, because it is the point: a runtime whose
    probe raises is NOT treated as idle by the predicate the reaper and the
    signal drain share.
    """

    assert _work_in_flight(_RaisingHandle()) is True
    assert _should_exit(_RaisingHandle(), _RecordingRuntime()) is False


def test_the_reaper_and_the_drain_share_the_predicate() -> None:
    """``_should_exit`` is False exactly while the shared predicate says busy.

    A second copy of "would lose nothing" is how the two paths drifted apart in
    the first place (the reaper consulted the handle; the signal handler never
    asked), so the shared function is what both call.
    """

    runtime = _RecordingRuntime()
    assert _should_exit(_WorkHandle(busy=True), runtime) is False
    assert _should_exit(_WorkHandle(busy=False), runtime) is True


# --- one commit point --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_seam_is_the_same_one_the_build_drain_commits_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two triggers commit through ONE call, and this pins the build side.

    The reconciliation's whole claim is that a signalled runtime and a
    build-replaced one are in the same state; if ``_begin_drain`` were a second
    implementation again, the record phrase and the frame flag could drift apart
    about the same departure.
    """
    seen: dict[str, Any] = {}

    async def _spy(handle: object, runtime: object, stop: object, **kwargs: Any) -> None:
        seen.update(kwargs)
        return None

    monkeypatch.setattr(process, "_commit_to_leaving", _spy)
    handle = _WorkHandle()
    await process._begin_drain(process._BuildPoll(), handle, _RecordingRuntime(), asyncio.Event())
    assert seen["label"] == "stale-build"
    assert seen["cause"] == "runtime-retired"
    assert seen["stagger_s"] >= 0.0, "the build path still draws the successor spread"


@pytest.mark.asyncio
async def test_the_drain_commits_at_the_signal_and_waits_afterwards() -> None:
    """The commit is taken while the turn is STILL running, and that is new.

    Round 1 announced at the boundary, because the only latch available then did
    both jobs at once: it refused admissions AND wrote the cut-off cause, so
    taking it early branded the turn the drain exists to save. PR #1108 split
    them (``begin_drain`` refuses, ``begin_retire`` notes), which is what lets
    the announcement go out when the refusals start — the operator's only
    warning — instead of after the work is already over.
    """
    events: list[str] = []
    stop = asyncio.Event()
    handle = _WorkHandle(busy=True, events=events)
    runtime = _RecordingRuntime(events)

    task = asyncio.ensure_future(_drain_for_signal(handle, runtime, stop, sig_name="SIGTERM"))
    try:
        await asyncio.sleep(process.REAP_CHECK_S * 2 + 0.05)
        assert not stop.is_set(), "the drain left while the turn was still running"
        assert runtime.announced == [
            (process._SIGNAL_DRAIN_REASON, True, LEAVING_ON_SIGNAL)
        ], "the departure is announced as it is committed: draining=True and the phrase"
        assert handle.drain_latches, "and the drain is latched, so nothing new is admitted"
        assert handle.retires == [], "but the cut-off cause is NOT noted while a turn runs"
        # ORDER IS THE CORRECTNESS ARGUMENT, so it is asserted rather than
        # assumed: the frame must precede the latch, or the first refused
        # message reads as an error instead of as a handover.
        assert events == [
            f"announce:{process._SIGNAL_DRAIN_REASON}",
            "drain:runtime-shutdown",
        ]

        handle.busy = False
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()

    assert stop.is_set()
    # The retirement latch — the one that writes the cause — was reached at the
    # boundary, with the runtime idle, which is what keeps a turn that completed
    # inside the window out of the cut-off taxonomy.
    assert handle.retires == [
        ("runtime-shutdown", "SIGTERM: drained to the end of the turn in flight")
    ]
    assert len(runtime.announced) == 1, "one departure, one announcement"


@pytest.mark.asyncio
async def test_the_drain_is_bounded_and_then_disposes_as_the_signal_path_always_did(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A runtime that never goes idle still leaves — at the bound, unlatched.

    The bound is what keeps "do not lose work" from becoming "cannot be
    killed". On expiry the exit is the OLD one: no clean-exit convergence and no
    ``begin_retire`` (the dispose rung notes ``runtime-shutdown`` for the turn it
    aborts), and ``stop`` set regardless. The announcement already went out at
    the commit — a departure that had to be taken back would be a worse lie than
    a departure that took the bound — so it stays exactly one.
    """
    monkeypatch.setattr(process, "SIGNAL_DRAIN_S", 0.3)
    events: list[str] = []
    stop = asyncio.Event()
    handle = _WorkHandle(busy=True, events=events)
    runtime = _RecordingRuntime(events)

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        await asyncio.wait_for(
            _drain_for_signal(handle, runtime, stop, sig_name="SIGTERM"), timeout=5
        )

    assert stop.is_set(), "a signal must never become an unbounded wait"
    assert runtime.announced == [(process._SIGNAL_DRAIN_REASON, True, LEAVING_ON_SIGNAL)]
    assert handle.retires == [], "the bound's exit is not a retirement"
    assert any("drain bound" in record.message for record in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_the_drain_ends_at_the_first_tick_after_the_work_finishes() -> None:
    """It stops waiting when the work ends — not at the bound.

    Asserted on the elapsed time rather than on a call count: a drain that
    polled to the deadline would still produce the same announcements, and the
    latency a sweep pays is the thing that would regress silently. The signal
    path draws NO successor spread, so "the work is done" is the only thing
    between the boundary and the exit.
    """
    stop = asyncio.Event()
    handle = _WorkHandle(busy=True)
    runtime = _RecordingRuntime()

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(_drain_for_signal(handle, runtime, stop, sig_name="SIGTERM"))
    started = loop.time()

    async def _finish_the_turn() -> None:
        await asyncio.sleep(process.REAP_CHECK_S + 0.05)
        handle.busy = False

    await asyncio.gather(_finish_the_turn(), asyncio.wait_for(task, timeout=5))
    elapsed = loop.time() - started
    assert stop.is_set()
    assert elapsed < process.BUILD_STAGGER_S / 2, f"the drain waited {elapsed:.2f}s"
    assert runtime.announced, "leaving at a boundary is still a retirement"


@pytest.mark.asyncio
async def test_a_raising_probe_does_not_hang_the_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken probe fails closed AND stays bounded.

    Both halves matter: the drain must not treat an unreadable probe as "idle"
    (that is the data loss this whole path exists to prevent) and it must not
    spin forever on one either.
    """
    monkeypatch.setattr(process, "SIGNAL_DRAIN_S", 0.3)
    stop = asyncio.Event()

    await asyncio.wait_for(
        _drain_for_signal(_RaisingHandle(), _RecordingRuntime(), stop, sig_name="SIGTERM"),
        timeout=5,
    )

    assert stop.is_set()


@pytest.mark.asyncio
async def test_a_failing_announcement_still_leaves() -> None:
    """The frame is best-effort; the departure is not.

    A viewer whose writer is gone must not hold a runtime that has already been
    told to go: the sentence is lost, the exit is not.
    """
    stop = asyncio.Event()
    handle = _WorkHandle(busy=False)

    await asyncio.wait_for(
        _drain_for_signal(handle, _RaisingRuntime(), stop, sig_name="SIGTERM"), timeout=5
    )

    assert stop.is_set()
    assert handle.drain_latches, "the drain still latched after a failed announce"


@pytest.mark.asyncio
async def test_a_handle_that_refuses_the_drain_latch_still_leaves() -> None:
    """A handle that will not latch does not veto the signal.

    ``begin_drain`` refuses while this handle is disposing; the signal still has
    to end the process. It falls back to the bounded wait with nothing
    announced — a runtime that latched nothing has no handover to advertise —
    and never touches ``begin_retire``.
    """
    stop = asyncio.Event()
    handle = _WorkHandle(busy=False, drains=False)

    await asyncio.wait_for(
        _drain_for_signal(handle, _RecordingRuntime(), stop, sig_name="SIGTERM"), timeout=5
    )

    assert stop.is_set()
    assert handle.drain_latches, "the refusal was asked for"
    assert handle.retires == [], "and no retirement is latched on a path that refused"


@pytest.mark.asyncio
async def test_an_older_handle_without_the_latch_still_waits_the_work_out() -> None:
    """No latch on the handle: keep round 1's behaviour rather than losing it.

    A reduced host or a test double cannot drain — there is nothing to refuse —
    but it can still be patient, which is the property the 2026-09-14 incident
    was about. It waits the work out, bounded, and disposes afterwards.
    """
    stop = asyncio.Event()
    handle = _NoDrainHandle(busy=True)
    runtime = _RecordingRuntime()

    task = asyncio.ensure_future(_drain_for_signal(handle, runtime, stop, sig_name="SIGTERM"))
    try:
        await asyncio.sleep(process.REAP_CHECK_S * 2 + 0.05)
        assert not stop.is_set(), "the unlatched path still waits for the work"
        handle.busy = False
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()

    assert stop.is_set()
    assert runtime.announced == [], "nothing was latched, so nothing is announced"


@pytest.mark.asyncio
async def test_a_drain_already_in_force_is_not_committed_twice() -> None:
    """A build drain that got there first owns the exit; the signal only bounds it.

    Two deciders for one departure is how a runtime reaches ``_clean_exit``
    twice, so the second trigger waits on the ``stop`` the first one sets — and
    still applies its own bound, because a signal may never wait forever.
    """
    monkeypatch_stop = asyncio.Event()
    handle = _WorkHandle(busy=True)
    handle._draining = True  # a build-replaced drain is already announced
    runtime = _RecordingRuntime()

    async def _release() -> None:
        await asyncio.sleep(process.REAP_CHECK_S + 0.05)
        monkeypatch_stop.set()

    await asyncio.gather(
        _release(),
        asyncio.wait_for(
            _drain_for_signal(handle, runtime, monkeypatch_stop, sig_name="SIGTERM"), timeout=5
        ),
    )

    assert runtime.announced == [], "the departure was already announced by its owner"
    assert handle.drain_latches == [], "and already latched"
    assert handle.retires == [], "the owning path performs the exit"


@pytest.mark.asyncio
async def test_a_drain_already_in_force_is_still_bounded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The waiting half of that rule, without the release: the bound still lands."""
    monkeypatch.setattr(process, "SIGNAL_DRAIN_S", 0.3)
    stop = asyncio.Event()
    handle = _WorkHandle(busy=True)
    handle._draining = True

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        await asyncio.wait_for(
            _drain_for_signal(handle, _RecordingRuntime(), stop, sig_name="SIGTERM"), timeout=5
        )

    assert stop.is_set()
    assert any("drain bound" in record.message for record in caplog.records), caplog.text


def test_the_commit_seam_refuses_to_take_a_second_drain() -> None:
    """The guard the two triggers rely on, asserted on the seam itself."""

    async def _run() -> Any:
        handle = _WorkHandle(busy=True)
        handle._draining = True
        return await _commit_to_leaving(
            handle,
            _RecordingRuntime(),
            asyncio.Event(),
            label="shutdown-drain",
            reason="leaving",
            detail="detail",
            loaded="<unknown>",
        )

    assert asyncio.run(_run()) is None


def test_the_drain_bound_is_a_production_number() -> None:
    """The bound is minutes, not seconds, and not a knob by accident.

    It has to be long enough to cover a real turn and short enough that a
    wedged runtime is still killable; the significance here is that it must not
    silently become 0 (unbounded-free but useless) or hours (unkillable), and
    that the ladder's escalation is derived from it rather than typed beside it.
    """
    from local_operator.session.runtime import control
    from local_operator.session.runtime.types import SIGNAL_DRAIN_S

    assert process.SIGNAL_DRAIN_S == SIGNAL_DRAIN_S
    assert 30.0 <= SIGNAL_DRAIN_S <= 600.0
    assert control.SIGTERM_GRACE_S > SIGNAL_DRAIN_S, (
        "the ladder's SIGTERM→SIGKILL grace must outlast the receiver's drain, "
        "or the escalation SIGKILLs a runtime that is finishing a turn"
    )


@pytest.mark.asyncio
async def test_the_stop_event_ordering_is_untouched_for_other_waiters() -> None:
    """``stop`` is a plain event: whoever set it first wins, and it stays set.

    The reaper, the refresh stagger and ``amain`` all wait on it; the drain
    must not clear it, re-set it in a second step, or leave it unset on any
    path — including when a DIFFERENT trigger set it while the drain waited.
    """
    stop = asyncio.Event()
    handle = _WorkHandle(busy=True)
    runtime = _RecordingRuntime()

    task = asyncio.ensure_future(_drain_for_signal(handle, runtime, stop, sig_name="SIGTERM"))
    # A deliberate stop (or the reaper) lands first — the pre-existing
    # behaviour: the event is set by whoever got there first.
    stop.set()
    handle.busy = False
    await asyncio.wait_for(task, timeout=5)
    assert stop.is_set()

    waiter = asyncio.ensure_future(stop.wait())
    await asyncio.wait_for(waiter, timeout=1)
    assert isinstance(stop, asyncio.Event)


def test_an_idle_handle_never_reaches_the_drain() -> None:
    """The signal path's fast branch, pinned where it is decidable.

    ``amain``'s handler sets ``stop`` in the same synchronous step when nothing
    is in flight; the drain is only armed when the shared predicate says there
    is work. This asserts the DECISION that branch is built on, since the
    handler itself is a closure inside ``amain`` (the wiring is the e2e cell
    that measures an idle runtime leaving within seconds of a real SIGTERM).
    """
    assert _work_in_flight(_WorkHandle(busy=False)) is False
    runtime: Any = _RecordingRuntime()
    assert _should_exit(_WorkHandle(busy=False), runtime) is True
