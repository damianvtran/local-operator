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
from local_operator.session.runtime.process import (
    _commit_to_leaving,
    _drain_for_signal,
    _should_exit,
    _work_in_flight,
)
from local_operator.session.runtime.types import LEAVING_FOR_BUILD, LEAVING_ON_SIGNAL


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
    # The build trigger publishes the FLEET half of the same commit, in its own
    # words: "signalled" would be false here, and without a phrase the record
    # would show an ordinary busy row while the app painted a drain notice.
    assert seen["leaving"] == LEAVING_FOR_BUILD


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


def test_the_phrase_less_signal_frame_is_read_as_a_signal() -> None:
    """MAJOR-1's discriminator, pinned against the producer's own label.

    Every build of this branch from the work-aware SIGTERM rung through the
    commit that added the frame's ``leaving`` key announces a signal drain with
    ``draining=True``, this label, and NO phrase — so a viewer that read an
    absent phrase as "the build handover" painted a signalled runtime with the
    build sentence, both clauses false (design round 4, D9; agent review round 4,
    MAJOR-1; UX round 4, U13). The frame's ``reason`` is that discriminator, and
    it is the LABEL rather than the longer ``"leaving after SIGTERM"`` string:
    ``_commit_to_leaving`` announces ``label`` as the frame's ``reason``, and the
    longer string is the log line and the ``_Drain``'s own reason.

    Pinned against ``process._SIGNAL_DRAIN_REASON`` rather than a literal, so a
    rename at the producer fails here instead of silently sending a signalled
    runtime back to the build story.
    """
    from local_operator.session.runtime.types import leaving_phrase_for_frame

    assert leaving_phrase_for_frame(process._SIGNAL_DRAIN_REASON, "") == LEAVING_ON_SIGNAL
    assert leaving_phrase_for_frame("stale-build", "0.55.6@46a4e9b") == LEAVING_FOR_BUILD
    assert leaving_phrase_for_frame("stale-build", "") == LEAVING_FOR_BUILD
    # A frame that names neither trigger gets nothing, and the app answers that
    # with the sentence true of any drain.
    assert leaving_phrase_for_frame("", "") == ""
    assert leaving_phrase_for_frame("something a newer build invented", "") == ""


# --- when each way out is armed ---------------------------------------------
# ``amain`` has three ways to be asked to leave: SIGTERM, SIGINT and the socket
# ``stop`` op. The signals are armed by ``install_loop_signal_handlers``, the op
# by assigning ``handle.on_stop_requested``, and the bound is TWO-SIDED: both
# must sit BELOW the ``RuntimeServer`` they close over and ABOVE
# ``RuntimeServer.start()`` — the statement that publishes the record every
# sender reads to find this process. Pinned against the source, because the
# runtime observable is a race and a passing race is not evidence.

#: The PRE-FIX statement order, kept as the negative control for the check
#: below: this is what ``amain`` looked like while the window was open (the
#: handlers armed after the wait for publication, the stop hook after that).
_PRE_FIX_ARMING_ORDER = """\
async def amain() -> int:
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    await runtime.wait_until_published()

    def _on_signal(sig: object) -> None:
        stop.set()

    install_loop_signal_handlers(loop, {})
    handle.on_stop_requested = _on_socket_stop
    return 0
"""

#: The OTHER way the bound can be broken, and the one the cell used to accept:
#: the arming block hoisted ABOVE the ``RuntimeServer`` it closes over. It still
#: satisfies ``install < start``, so the lower bound alone cannot see it — and
#: under it ``_drain_for_signal(handle, runtime, ...)`` reads ``runtime`` before
#: the assignment, i.e. the signal raises instead of being answered (F-1, review
#: round 1).
_HOISTED_ARMING_ORDER = """\
async def amain() -> int:
    install_loop_signal_handlers(loop, {})
    handle.on_stop_requested = _on_socket_stop
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    return 0
"""


def _arming_lines(source: str) -> dict[str, int]:
    """Where in ``amain``'s OWN body each way out is armed, as statement line numbers.

    PARSED, NOT SUBSTRING-MATCHED — and a paragraph is not a statement. This
    file's sibling ``test_inbox`` lost that argument once already: its ordering
    assertion was written with ``source.index(...)`` and spent a while passing
    against the comment that explained the move it was supposed to police.

    THE ``RuntimeServer`` CONSTRUCTION IS REPORTED TOO, not only the call that
    names the local: the invariant the arming sits in is TWO-SIDED — below the
    runtime the handlers close over, above ``start()`` — and a reading that
    yielded only the arming lines could not express the lower bound at all
    (F-1, review round 1). One scan answers both, so the two bounds cannot
    disagree about which statement is the constructor.

    A nested scope is excluded, which matters here rather than being tidiness:
    the signal handler IS a nested ``def``, and a call inside one is not part of
    the statement order this asserts.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    body: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(tree.body[0]))
    while stack:
        node = stack.pop()
        body.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))

    runtime = ""
    construct = 0
    for node in body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "RuntimeServer"
        ):
            runtime = getattr(node.targets[0], "id", "")
            construct = node.lineno
    assert runtime and construct, "amain no longer constructs a RuntimeServer"

    lines: dict[str, int] = {"construct": construct}
    for node in body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.attr == "on_stop_requested"
                ):
                    lines.setdefault("socket_hook", node.lineno)
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if isinstance(called, ast.Name) and called.id == "install_loop_signal_handlers":
            lines.setdefault("install", node.lineno)
        if (
            isinstance(called, ast.Attribute)
            and called.attr == "start"
            and isinstance(called.value, ast.Name)
            and called.value.id == runtime
        ):
            lines.setdefault("start", node.lineno)
    missing = {"install", "start", "socket_hook"} - set(lines)
    assert not missing, f"amain no longer arms {sorted(missing)}"
    return lines


def test_the_ways_out_are_armed_before_the_record_makes_this_process_addressable() -> None:
    """THE ordering guarantee, asserted against the source that provides it.

    MEASURED FAILURE THIS PINS. With the handlers installed after the wait for
    publication, a runtime was addressable while a SIGTERM still killed it with
    the default disposition: ``tests/e2e/test_signal_drain_e2e.py``'s idle cell
    went red on CI twice, on two platforms, with exit ``-15`` and an EMPTY
    runtime-log tail apart from the interpreter's default kill (no
    ``exiting (SIGTERM`` line at all), which is the whole symptom — the record
    was never unpublished, the lease was never released, and the documented
    drain never ran, because the process simply stopped.

    WAITING ON PUBLICATION IS NOT A SUBSTITUTE, which is what made the window
    reachable: ``wait_until_published`` settles at the END of ``_serve``'s boot
    prologue, so the record — written inside ``RecordPublisher.__init__`` — is
    already readable for as long as those two boot registrations take to come
    back from the session's loop.

    THE SOCKET HOOK IS THE SAME WINDOW, IN A QUIETER SHAPE, which is why it is
    asserted here too: ``ServingSessionHandle.request_stop`` falls back to
    disposing the session in place when ``on_stop_requested`` is unset, and that
    fallback sets no stop event, so a ``stop`` op landing before the assignment
    leaves a runtime that never exits, behind a record that still reads live.

    ASSERTED STRUCTURALLY, on the order of the statements, because there is no
    runtime observable that distinguishes "armed first" from "armed fast
    enough": the e2e cell that catches this is a schedule accident — it PASSED
    on the unfixed tree on this host — so its green is not evidence the
    ordering holds. This fails if either arming moves back below ``start()``.

    AND THE BOUND IS TWO-SIDED, which is the half the first version of this cell
    was missing (F-1, agent review round 1): the arming also has to be BELOW the
    ``RuntimeServer`` the handlers close over, because ``_drain_for_signal``
    takes that runtime as an argument. Hoisting the block above the constructor
    keeps ``install < start`` true — so the lower bound alone stayed green — while
    a signal arriving with boot-prologue work in flight would raise
    ``UnboundLocalError`` inside the loop's signal callback: the signal is
    swallowed and the runtime never stops, which is the defect this ordering
    exists to remove, re-opened in a window the old assertion could not see.
    Both directions are controlled below, on snippets that are the real source
    shapes rather than paraphrases of them.
    """
    import inspect

    real = _arming_lines(inspect.getsource(process.amain))
    assert real["construct"] < real["install"], (
        "the SIGTERM/SIGINT handlers are armed ABOVE the `RuntimeServer` they close "
        "over: `_drain_for_signal(handle, runtime, ...)` then reads `runtime` before "
        "it is bound, so a signal with boot work in flight raises UnboundLocalError "
        "inside the loop's signal callback — the signal is swallowed and the runtime "
        "never stops"
    )
    assert real["install"] < real["start"], (
        "the SIGTERM/SIGINT handlers must be installed BEFORE RuntimeServer.start(): "
        "start() is what publishes the record on its thread, and the record is what "
        "makes this process addressable — a signal in that window kills the runtime "
        "with the default disposition, unpublishing nothing and releasing no lease"
    )
    assert real["construct"] < real["socket_hook"], (
        "the socket stop hook is armed ABOVE the `RuntimeServer` it is a way out OF: "
        "its own closure reads no `runtime` (it sets the stop event through `loop`), "
        "so the bound asserted here is the BLOCK's — the two arming statements are one "
        "unit and belong on the same side of the constructor, or a partial hoist "
        "leaves the signal half above it while this half looks correct"
    )
    assert real["socket_hook"] < real["start"], (
        "the socket stop hook must be armed BEFORE RuntimeServer.start(), for the same "
        "reason and in the same window: an unset on_stop_requested makes request_stop "
        "dispose in place and set no stop event, so the process would never exit"
    )

    # THE CHECK DISCRIMINATES, asserted rather than assumed: the pre-fix order
    # must read as out of order, or this cell is decoration that can never go
    # red — the failure mode AGENTS.md's "Prove the test can still fail" exists
    # for. Both arming sites are checked, so swapping either one back is caught.
    before = _arming_lines(_PRE_FIX_ARMING_ORDER)
    assert before["construct"] < before["install"]
    assert before["install"] > before["start"]
    assert before["socket_hook"] > before["start"]

    # ... and the UPPER bound has its OWN control, because it is the bound that
    # was missing (F-1). The first assertion below is the point: the hoisted
    # order satisfies `install < start`, so a control that only re-tested the
    # lower bound would prove nothing about what this cell now catches.
    hoisted = _arming_lines(_HOISTED_ARMING_ORDER)
    assert hoisted["install"] < hoisted["start"], (
        "the hoisted control must be a case `install < start` alone accepts, or it "
        "demonstrates nothing about the bound this cell was missing"
    )
    assert hoisted["construct"] > hoisted["install"]
    assert hoisted["construct"] > hoisted["socket_hook"]
