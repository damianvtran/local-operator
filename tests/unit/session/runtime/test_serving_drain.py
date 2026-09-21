"""The drain latch: leaving a build behind WITHOUT aborting what is running.

``begin_retire`` refuses while any work would be lost, and that refusal is the
defect the bound exists to bound: a session busy for hours never reaches an
idle instant, so a runtime whose install was replaced underneath it keeps
executing a tree that is gone. ``begin_drain`` (design §4 F1) drops the idle
gate and keeps every other guarantee — no new work is admitted, nothing in
flight is touched, and a message that arrives while the runtime is still
finishing is SPOOLED for the successor rather than dropped or run against the
build that is leaving.

Exercised the way ``test_cut_off_turns._LatchHost`` exercises the sibling
latch: the handle's real methods over a stub session and a stub predicate, so
the assertions land on the latch rather than on a runtime boot (the e2e stage
covers the boot).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.buildwatch import UpdateLock
from local_operator.harness.types import ImageContent
from local_operator.harness.wake import WakeSchedule
from local_operator.mobile.command_reservation import CommandReservations
from local_operator.session.errors import RuntimeRetiring
from local_operator.session.runtime.inbox import (
    INBOX_NAME,
    SOURCE_USER,
    SPOOL_RECEIPT_PROMPT,
    SPOOL_RECEIPT_WAKE,
    peek_inbox,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import LEAVING_FOR_BUILD, SIGNAL_DRAIN_CAUSE
from local_operator.session.session import Session


class FakeSession:
    """The slice of Session these latches touch."""

    def __init__(self, directory: Path, *, busy: bool = True) -> None:
        self.transcript = SimpleNamespace(directory=directory)
        self.busy = busy
        self.notes: list[tuple[str, str]] = []
        self.deliberate = 0
        self.peer_calls: list[tuple[str, str, bool]] = []
        #: ``Session.disposal_cuts_a_turn``, which is the ONLY evidence the
        #: cut-off note is now gated on. ``False`` is the default because a
        #: stub that claimed a live turn without one would pin the note's
        #: placement rather than its gate (agent review round 1, MAJOR-1).
        self.cuts_a_turn = False

    # -- the two verdict writers -------------------------------------------
    def note_cut_off(self, cause: str, detail: str = "") -> None:
        self.notes.append((cause, detail))

    def note_deliberate_stop(self) -> None:
        self.deliberate += 1

    def disposal_cuts_a_turn(self) -> bool:
        return self.cuts_a_turn

    # -- peer delivery ------------------------------------------------------
    async def receive_peer_message(
        self, text: str, *, mode: str = "mailbox", wake: bool = False, sender: Any = None
    ) -> str:
        self.peer_calls.append((text, mode, wake))
        return "delivered"


class DrainHost:
    """``ServingSessionHandle``'s drain latch over a stub session.

    ``may_refresh`` is the only thing stubbed about the predicate: a runtime
    that reports work it would lose, which is the case the whole rung exists
    for.
    """

    begin_drain = ServingSessionHandle.begin_drain
    begin_retire = ServingSessionHandle.begin_retire
    # The EXIT rung that writes what the two latches recorded, bound here for
    # the same reason they are: the note's placement is the behaviour under
    # test, and a stub that re-implemented it would pin nothing.
    _note_retirement_cut_off = ServingSessionHandle._note_retirement_cut_off
    # Bound like any other method: ``_retiring_refusal`` reads the handle's own
    # latched cause to choose its sentence, so the stub has to hand it ``self``
    # (design round 4, D10).
    _retiring_refusal = ServingSessionHandle._retiring_refusal
    _spool_for_successor = ServingSessionHandle._spool_for_successor
    receive_peer_message = ServingSessionHandle.receive_peer_message
    _note_deliberate_stop = ServingSessionHandle._note_deliberate_stop
    _check_loop_thread = lambda self: None  # noqa: E731 — the real one only asserts

    def __init__(self, session: FakeSession, *, busy: bool = True) -> None:
        self._session = session
        self._busy = busy
        self._retiring_cause = ""
        self._retiring_detail = ""
        self._draining = False
        self._exit_committed = False
        self._disposing = False
        # The UPDATE WINDOW's state, which the production ``__init__`` owns and the
        # admission paths read beside the drain latch above (``serving`` docstring,
        # ``types.UPDATING``). Empty here: every cell in this file is about a drain,
        # and the window's own cells live in ``test_update_window.py`` — but the
        # attributes have to exist, because ``prompt``/``steer``/``receive_peer_message``
        # are the REAL methods and a host that omits them is a host that raises.
        self._updating = ""
        self._update_failed = ""
        self._update_lock = UpdateLock()
        self._applied_update = ""
        self._fold = SimpleNamespace(note_peer_message=lambda *_a, **_k: None)

    def may_refresh(self) -> str:
        return "busy" if self._busy else ""

    def _notify(self) -> None:
        pass


def _host(tmp_path: Path, *, busy: bool = True) -> tuple[DrainHost, FakeSession]:
    session = FakeSession(tmp_path / "sessions" / "s1")
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return DrainHost(session, busy=busy), session


# -- the residency predicate's warm-window term (QA round 1, Q-1) ----------------
#
# Only these tests use the REAL ``may_refresh``. Everything else in this file
# stubs the predicate wholesale, because there the LATCH is what is under test;
# Q-1 lives inside the predicate, where the wedge was.


class PredicateHost:
    """The production ``may_refresh`` over stubbed terms."""

    may_refresh = ServingSessionHandle.may_refresh

    def __init__(self, *, busy: bool = False, wake_due_ms: int | None = None) -> None:
        self._busy = busy
        self._wake_due_ms = wake_due_ms

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._wake_due_ms


def test_may_refresh_names_a_wake_inside_the_warm_window() -> None:
    """The term, through the shared home, on a handle whose accessor answers."""
    now_ms = int(time.time() * 1000)
    assert (
        PredicateHost(wake_due_ms=now_ms + 1_000).may_refresh() == "wake due within the warm window"
    )
    assert PredicateHost(wake_due_ms=now_ms + 3_600_000).may_refresh() == ""
    assert PredicateHost(busy=True, wake_due_ms=now_ms).may_refresh() == "busy", "busy wins"


def test_may_refresh_fails_open_when_the_warm_window_probe_breaks(monkeypatch) -> None:
    """Q-1: a predicate that cannot be evaluated must NEVER pin the runtime.

    The blocker QA found was this exact shape: the term was reached through a
    function-local import of ``...runtime.process``, which a runtime executing
    that module as ``__main__`` answers from DISK — so at the one moment the
    predicate mattered most (the loaded tree had been replaced) it raised
    ``ImportError``. ``process._idle_for_refresh`` reads a failing predicate as
    "not idle" by design, so the drain never saw an idle instant, never reached
    ``begin_retire``/``_clean_exit``, and the session was refused forever while
    holding the lease — no successor could boot. The helper now lives in the
    stdlib-only module both sides import, and the consult is guarded besides:
    a raising probe reads as "no wake", the same answer the helper gives its
    own broken accessor.
    """
    from local_operator.session.runtime import serving as serving_mod

    def explode(_handle: Any, **_kwargs: Any) -> bool:
        raise ImportError("cannot import name 'process' from 'local_operator.session.runtime'")

    monkeypatch.setattr(serving_mod, "_wake_within_window", explode)
    assert PredicateHost().may_refresh() == "", "a broken probe must not pin the runtime"


def test_the_warm_window_term_is_not_reached_through_the_runtime_module() -> None:
    """Q-1's pin on the construct, not just the behaviour.

    The behavioural pin is the ARMED e2e stage (``test_runtime_refresh_e2e``),
    which is the only place the loaded tree can actually be gone — the repo's
    own interpreter is an editable venv, where the probe is disarmed, which is
    why the wedge shipped. This one is cheap insurance that the exact construct
    that caused it cannot come back: an import of the runtime module from
    inside the predicate. Read off the SOURCE, because a call cannot tell the
    two homes apart while the tree is intact — both return the same bool.
    """
    import inspect

    from local_operator import buildwatch
    from local_operator.session.runtime import process, serving

    assert serving._wake_within_window is buildwatch.wake_within_window
    assert process._wake_within_window is buildwatch.wake_within_window
    source = inspect.getsource(ServingSessionHandle.may_refresh)
    assert "session.runtime.process import" not in source, "the wedge's construct is back"
    assert "runtime.process" not in inspect.getsource(buildwatch.wake_within_window).replace(
        "runtime", ""
    ), "the shared home must not reach back into the runtime"


# -- the latch ------------------------------------------------------------------


def test_begin_drain_latches_while_a_turn_is_running(tmp_path: Path) -> None:
    """The one difference from ``begin_retire``: the latch does not wait."""
    host, session = _host(tmp_path, busy=True)
    assert host.begin_retire("runtime-retired") is False, "the soft rung still refuses"
    assert host._retiring_cause == ""

    assert host.begin_drain("runtime-retired", " (0.54.33@7fe8b10 → 0.54.39@dec7933)") is True
    assert host._retiring_cause == "runtime-retired"
    assert host._draining is True
    assert host._exit_committed is False, "the exit has not been taken yet"
    # The refusal is a TYPED admission category carrying the shared sentence,
    # and that sentence deliberately does NOT name the internal cause token any
    # more — the token was the complaint (design round 1, D2). The cause is
    # still on the handle, which the assertion above pins.
    refusal = host._retiring_refusal()
    assert isinstance(refusal, RuntimeRetiring)
    assert "runtime-retired" not in str(refusal)
    assert session.notes == [], "no turn is being cut off, so no cut-off may be recorded"


def test_the_refusal_describes_the_departure_the_handle_latched(tmp_path: Path) -> None:
    """D10/MAJOR-2: one gate, two departures, and the sentence follows the cause.

    ``prompt`` refuses for the whole of ANY drain, and ``begin_drain`` latches
    from the SIGTERM arm too — so this accessor served a signalled runtime the
    build sentence, "the one it loaded is gone from disk", under a notice that
    correctly said the session had been signalled to stop. The cause the handle
    latched is the discriminator, and it survives to the exit rung because
    ``process._drain_for`` re-passes ``drain.cause`` to ``begin_retire``.

    ALL THREE ARMS ARE PINNED HERE, because a fix to one is exactly how another
    goes wrong. The signal arm gets a sentence of its own. The build arm IS named
    now, and the term that makes that safe is ``_draining`` WITHOUT a committed
    exit, which is the DRAIN and nothing else: the cause it shares with ``/move``
    is latched by ``begin_retire``, which commits its exit in the same
    synchronous step (``server._retire_for``, both of its callers), so a moved or
    rotated session cannot reach the build sentence through this accessor. That
    term is what the old comment here said was missing — "the phrase, not the
    cause, tells them apart" — and with the drain's own state in hand the raiser
    CAN say which departure it is, and has to: the refusal's TAIL is chosen off
    the same token, and only a build drain owes a successor (memo §4.2 piece 3).
    The move case is pinned below, where leaving it unnamed is still right.

    AND A THIRD ARM: the reaper's ``idle-exit`` latch — a departure with no
    successor at all, which used to inherit the build sentence with nobody having
    compared a build (round 5, MINOR-1/U14/D11).
    """
    host, _session = _host(tmp_path, busy=True)

    assert host.begin_drain(SIGNAL_DRAIN_CAUSE, "SIGTERM: drained to the turn's end") is True
    signalled = host._retiring_refusal()
    assert isinstance(signalled, RuntimeRetiring)
    assert signalled.trigger == RuntimeRetiring.SIGNAL, signalled.trigger
    assert signalled.HEAD == RuntimeRetiring.HEAD_SIGNALLED, signalled.HEAD
    assert "newer build" not in str(signalled), str(signalled)
    assert "send it again" in str(signalled), "the one act left still has to be named"
    assert signalled.TAIL == RuntimeRetiring.TAIL, signalled.TAIL
    assert SIGNAL_DRAIN_CAUSE not in str(signalled), str(signalled)

    host, _session = _host(tmp_path, busy=True)
    assert host.begin_drain("runtime-retired", " (0.54.33@7fe8b10 → 0.54.39@dec7933)") is True
    build = host._retiring_refusal()
    assert isinstance(build, RuntimeRetiring)
    assert build.trigger == RuntimeRetiring.BUILD, build.trigger
    assert build.HEAD == RuntimeRetiring.HEAD, build.HEAD
    assert build.TAIL == RuntimeRetiring.TAIL_HANDOVER, build.TAIL
    # IT ASKS FOR THE RE-SEND AND NAMES THE DESTINATION, which is the pair of
    # facts this arm can establish (UX round 1, U5). What it must NOT do is
    # claim the message is on its way: this accessor serves the arms where the
    # spool FAILED (an unwritable inbox) or was impossible (an attachment), and
    # the previous wording — "a newer build is starting here to carry on" — read
    # as carriage while the user's message had been dropped (QA round 1, Q-2).
    assert "send it again" in str(build), "the next act has to be named: " + str(build)
    assert "new build is up" in str(build), str(build)
    assert "carry on" not in str(build), "this arm did not carry the message: " + str(build)
    assert "queued" not in str(build), "this arm did not queue the message: " + str(build)

    # THE FOURTH ARM: a message that IS carried. Reached where the caller cannot
    # wait for a turn this runtime will not run — a loop's ``prompt_and_wait``,
    # which correlates on an ``AgentEndEvent`` from THIS runtime and would
    # otherwise be handed a refusal that tells it to re-send a message the
    # successor already holds (``attached.prompt_and_wait``).
    carried = RuntimeRetiring(leaving=LEAVING_FOR_BUILD, queued=True)
    assert carried.trigger == RuntimeRetiring.BUILD, carried.trigger
    assert carried.TAIL == RuntimeRetiring.TAIL_QUEUED, carried.TAIL
    assert "queued" in str(carried), str(carried)
    assert "send it again" not in str(
        carried
    ), "a carried message must not be handed advice to re-send it: " + str(carried)

    # ``/move`` and the viewer-driven rotate share that cause token and latch
    # through ``begin_retire``: no build was compared and no successor is owed,
    # so the accessor must not name a build for either. This is the assertion the
    # build arm above used to carry, kept where it is still true and now actually
    # discriminating — as written it passed for the DRAIN, which is the case that
    # needed the sentence.
    host, _session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired", "moved") is True
    moved = host._retiring_refusal()
    assert isinstance(moved, RuntimeRetiring)
    assert moved.trigger == "", moved.trigger
    assert moved.HEAD == RuntimeRetiring.HEAD_UNNAMED, moved.HEAD
    assert moved.TAIL == RuntimeRetiring.TAIL, moved.TAIL

    # The reaper's quiet exit: nothing compared a build, so ``busy=False`` is
    # what lets ``begin_retire`` latch at all.
    host, _session = _host(tmp_path, busy=False)
    assert host.begin_retire("idle-exit", "idle for 20s with no viewer") is True
    idle = host._retiring_refusal()
    assert isinstance(idle, RuntimeRetiring)
    assert idle.trigger == "", idle.trigger
    assert idle.HEAD == RuntimeRetiring.HEAD_UNNAMED, idle.HEAD
    assert "newer build" not in str(idle), "nothing checked a build on the idle-exit rung: " + str(
        idle
    )


@pytest.mark.parametrize(
    "token",
    ["", "retiring", "probably-fine", RuntimeRetiring.SIGNAL, RuntimeRetiring.BUILD],
)
def test_only_the_two_enumerated_triggers_cross_the_transport(token: str) -> None:
    """The decode validates the token, because the far side rebuilds SENTENCES.

    ``admission_error`` is the only entry point that turns a peer's frame into a
    local exception, and its contract is the module docstring's: an enumerated
    category and nothing else — which is why the count is an int and not a
    string. A trigger is therefore admitted only from the closed set, and
    anything else (including absent, which is what an older runtime sends) means
    "this raiser cannot name its departure" — whose sentence names no departure
    EITHER. It used to be the build sentence, and that is the round-5 finding:
    a signal-draining runtime from this branch's pre-key builds sent no token,
    so the far side told the operator about a build that did not exist while the
    notice above it said the session had been signalled to stop (MINOR-1/U14/
    D11). The phrase arm is the next cell.
    """
    from local_operator.session.errors import admission_error

    decoded = admission_error(RuntimeRetiring.code, None, token)
    assert isinstance(decoded, RuntimeRetiring)
    expected = {
        RuntimeRetiring.SIGNAL: RuntimeRetiring.HEAD_SIGNALLED,
        RuntimeRetiring.BUILD: RuntimeRetiring.HEAD,
    }.get(token, RuntimeRetiring.HEAD_UNNAMED)
    assert decoded.HEAD == expected, (token, decoded.HEAD)


def test_an_unnamed_refusal_is_resolved_from_the_phrase_the_frame_published() -> None:
    """The cross-version arm, on the shapes those runtimes actually send (round 5).

    The raiser cannot help: ``error_trigger`` is new in this branch, so every
    build older than it — a RELEASED one and this branch's own pre-key rungs —
    crosses with ``error_code`` alone. The only witness to which departure
    refused the message is the ``retiring`` frame that crossed on the SAME
    connection moments earlier, so the far side hands its own reading of that
    frame in as evidence (``types.drain_phrase_for_frame``), and the sentence is
    built from it.

    THREE POPULATIONS, one rule. A released runtime's only draining announce is
    the stale-build handover, so it keeps the build sentence; a pre-key rung of
    this branch signal-drains, so it gets the signal sentence under its own
    signal notice; and a phrase the far side cannot place resolves nothing, which
    leaves the sentence that names no departure. The frames below are the wire
    shapes measured in rounds 4 and 5, not invented ones.
    """
    from local_operator.session.errors import admission_error
    from local_operator.session.runtime.types import (
        LEAVING_FOR_BUILD,
        LEAVING_ON_SIGNAL,
        drain_phrase_for_frame,
    )

    # ``8dd605365``: the signal drain, before the phrase key existed (round 3/5).
    pre_key_signal = {"reason": "shutdown-drain", "to": "", "draining": True}
    # A RELEASED build (0.55.4-.6): its only draining announce is the handover.
    released_build = {
        "reason": "stale-build",
        "to": "0.55.6@f4a70b9",
        "draining": True,
    }
    # A frame that named nothing at all.
    names_neither = {"reason": "", "to": "", "draining": True}

    assert drain_phrase_for_frame(pre_key_signal) == LEAVING_ON_SIGNAL
    assert drain_phrase_for_frame(released_build) == LEAVING_FOR_BUILD
    assert drain_phrase_for_frame(names_neither) == ""

    def decode(frame: dict[str, Any]) -> RuntimeRetiring:
        """The far side's own decode: category + the phrase THIS frame published."""
        decoded = admission_error(RuntimeRetiring.code, None, None, drain_phrase_for_frame(frame))
        assert isinstance(decoded, RuntimeRetiring), decoded
        return decoded

    heads = [decode(frame).HEAD for frame in (pre_key_signal, released_build, names_neither)]
    assert heads == [
        RuntimeRetiring.HEAD_SIGNALLED,
        RuntimeRetiring.HEAD,
        RuntimeRetiring.HEAD_UNNAMED,
    ], heads

    # The token wins where a raiser can send one: an inference never overrides
    # the enumeration of the side that latched the drain.
    explicit = admission_error(RuntimeRetiring.code, None, RuntimeRetiring.BUILD, LEAVING_ON_SIGNAL)
    assert isinstance(explicit, RuntimeRetiring), explicit
    assert explicit.HEAD == RuntimeRetiring.HEAD, explicit.HEAD


def test_begin_retire_records_the_cause_and_brands_no_turn(tmp_path: Path) -> None:
    """The latch RECORDS; the DISPOSAL writes, and only for a turn it cuts.

    THE ORDER IS THE FIX (2026-09-17), and the round-1 correction is that the
    second half has a GATE. ``begin_retire`` refuses while anything would be
    lost, so a retirement that names a build is proof that no turn is in
    flight: arming a note at the latch could only ever brand a run this exit did
    not cut — the operator's "Stopped with an error" for an update that caught
    nothing (agent review round 1, MAJOR-1). Both halves are pinned here, and
    the gate with them: the latch leaves the session untouched, and the disposal
    delivers what the latch recorded only when it is aborting a live turn.
    """
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired", " (a → b)") is True
    assert host._exit_committed is True
    assert host._retiring_detail == " (a → b)"
    assert session.notes == [], "a latch must not brand a run it is still waiting for"

    host._note_retirement_cut_off()
    assert session.notes == [], "a retirement proves no turn was in flight to cut"

    session.cuts_a_turn = True
    host._note_retirement_cut_off()
    assert session.notes == [("runtime-retired", " (a → b)")]


def test_an_idle_exit_owes_the_session_no_cut_off_at_all(tmp_path: Path) -> None:
    """The quiet rung: same latch, and the same rule as the build rung.

    The idle exit latches so its refusals and its log line are honest (nothing
    is admitted after it), but its whole precondition is that NOTHING is in
    flight, so there is no turn for a note to name. Arming one anyway is what
    the operator hit — a backend update that caught nothing published a durable
    ``error`` row for a run that had already ended, six times on this host
    (2026-09-17). The rung no longer needs a flag to say so: the disposal's own
    evidence decides, identically on every rung (agent review round 1,
    MAJOR-2).
    """
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("idle-exit") is True
    assert host._retiring_cause == "idle-exit", "the refusal and the log still need it"
    host._note_retirement_cut_off()
    assert session.notes == []


def test_a_disposal_with_no_latch_still_names_the_shutdown(tmp_path: Path) -> None:
    """The disposal rung's own fallback, which the move must not have eaten.

    A fatal-on-arrival SIGTERM and a host that disposes in place never latch a
    retirement, and both can catch a live turn: the disposal is the only rung
    that can say so, and it says ``runtime-shutdown``.
    """
    host, session = _host(tmp_path, busy=False)
    session.cuts_a_turn = True
    host._note_retirement_cut_off()
    assert session.notes == [("runtime-shutdown", "")]


def test_a_disposing_handle_refuses_the_latch(tmp_path: Path) -> None:
    """A second exit must not race the disposal that is already running.

    Both latches, because the guard was asymmetric until 2026-09-17: a disposal
    owns the ordering from ``_disposing`` on, and the rung that takes the exit
    IN THIS STEP must not commit to a second one. Hardening rather than the fix
    for the cut-off rows (the incident's ordering has the arming well before any
    disposal), but an asymmetry with ``begin_drain`` immediately next door is an
    invitation to write the next rung through it.
    """
    host, _ = _host(tmp_path)
    host._disposing = True
    assert host.begin_drain("runtime-retired") is False
    assert host._draining is False
    assert host.begin_retire("runtime-retired") is False
    assert host._exit_committed is False


# -- admissions during the drain ------------------------------------------------


@pytest.mark.asyncio
async def test_a_peer_wake_during_the_drain_is_spooled_for_the_successor(
    tmp_path: Path,
) -> None:
    """invariant (iv): deferred, not dropped, and never run on the dying build."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.receive_peer_message(
        "the build is moving", mode="steer", wake=True, sender={"name": "peer"}
    )
    assert receipt == SPOOL_RECEIPT_WAKE, receipt
    assert session.peer_calls == [], "a turn must not be started on a build that is leaving"

    directory = session.transcript.directory
    rows = peek_inbox(directory)
    assert len(rows) == 1
    assert rows[0].text == "the build is moving"
    assert rows[0].sender == {"name": "peer"}
    # The successor's boot drain (``process._drain_inbox_into``) reads exactly
    # this file, so the row is what the next engage delivers.
    assert (directory / INBOX_NAME).exists()


@pytest.mark.asyncio
async def test_a_committed_exit_refuses_a_wake_instead(tmp_path: Path) -> None:
    """Once the exit is being taken there is no successor window left, so the
    message keeps the vocabulary refusal the sender can act on."""
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired") is True
    with pytest.raises(RuntimeError) as caught:
        await host.receive_peer_message("too late", mode="steer", wake=True)
    assert "send it again" in str(caught.value)
    assert peek_inbox(session.transcript.directory) == []


@pytest.mark.asyncio
async def test_a_quiet_note_is_still_delivered_during_the_drain(tmp_path: Path) -> None:
    """Unchanged behaviour, and deliberately so: a record-only note opens no
    turn, so refusing it would drop something the sender was told had landed."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired") is True
    assert await host.receive_peer_message("fyi") == "delivered"
    assert session.peer_calls == [("fyi", "mailbox", False)]
    assert peek_inbox(session.transcript.directory) == []


def test_a_user_stop_during_the_drain_is_still_the_users_own(tmp_path: Path) -> None:
    """A drain lasts as long as the work does, so a ``/stop`` landing inside it
    ends the turn by the operator's own hand — reporting it as housekeeping
    would misattribute the cancel. Only the committed EXIT suppresses it."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired") is True
    host._note_deliberate_stop()
    assert session.deliberate == 1

    host._exit_committed = True
    host._note_deliberate_stop()
    assert session.deliberate == 1, "the retirement owns the verdict from here"


# -- the wake divert (the session side of invariant iv) --------------------------


class WakeHost:
    """``Session``'s wake-handover methods over a stub transcript and scheduler."""

    retire_wakes_to_inbox = Session.retire_wakes_to_inbox
    _spool_wake_to_inbox = Session._spool_wake_to_inbox
    _queue_wake_rearm = Session._queue_wake_rearm
    hand_wakes_to_successor = Session.hand_wakes_to_successor

    #: Declared for the type checker, which cannot see an attribute a borrowed
    #: method assigns on the instance (``retire_wakes_to_inbox`` sets it).
    _wake_rearms: list[Any]

    def __init__(self, directory: Path | None, *, live: tuple[Any, ...] = ()) -> None:
        self._transcript = SimpleNamespace(directory=directory)
        self._wake_deliver_hook: Any = None
        self._wake = SimpleNamespace(schedules=live)
        self.persisted: list[list[Any]] = []

    def _missed_delivery_note(self, _due: Any) -> None:
        return None

    async def _persist_wake_schedules(self, schedules: list[Any]) -> None:
        self.persisted.append(list(schedules))


def _due(text: str = "check the deploy", *, final: bool = False) -> Any:
    return SimpleNamespace(
        schedule=SimpleNamespace(id="w1", every_ms=None, message=text),
        occurrence=1,
        planned_total=None,
        final=final,
    )


def _final_due(text: str = "check the deploy") -> Any:
    """A fire that RETIRED its schedule: a real ``WakeSchedule``, so the hook can
    copy it into the one-shot the successor owes."""
    return SimpleNamespace(
        schedule=WakeSchedule(id="w1", message=text, next_due_at=1),
        occurrence=1,
        planned_total=1,
        final=True,
    )


@pytest.mark.asyncio
async def test_a_wake_fired_during_the_drain_lands_in_the_inbox(tmp_path: Path) -> None:
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    assert host._wake_deliver_hook == host._spool_wake_to_inbox

    await host._spool_wake_to_inbox(_due())
    rows = peek_inbox(tmp_path)
    assert len(rows) == 1
    assert "check the deploy" in rows[0].text
    assert rows[0].mode == "mailbox", "the same mailbox vehicle, never a new one"
    assert rows[0].wake is True, "a fired wake asked for a turn; the successor owes it one"


@pytest.mark.asyncio
async def test_a_retiring_schedule_is_re_armed_instead_of_spooled(tmp_path: Path) -> None:
    """MINOR 3, the sharp half: a fire that RETIRES its schedule leaves nothing
    able to engage the session.

    No index row survives it and no errand is raised for a schedule that has
    already fired, so a spooled note would keep the reminder and never run the
    work until a human opened the conversation. Re-armed as a one-shot due now,
    the supervisor starts a runtime for it and the successor folds it as an
    overdue occurrence.
    """
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())

    assert peek_inbox(tmp_path) == [], "the re-arm carries it; a note would double it"
    assert len(host._wake_rearms) == 1
    rearmed = host._wake_rearms[0]
    assert (rearmed.id, rearmed.message) == ("w1", "check the deploy"), "same handle, cancellable"
    assert rearmed.every_ms is None and rearmed.limit is None and rearmed.until_at is None
    assert rearmed.next_due_at > 0, "due now, so the supervisor engages on its next pass"


@pytest.mark.asyncio
async def test_a_wake_that_cannot_be_re_armed_is_still_spooled(tmp_path: Path) -> None:
    """The outcome this path must never produce: an occurrence that exists
    neither as a schedule nor as a spooled reminder.

    The fake's schedule here cannot be copied, standing in for any re-arm
    failure (a shape the harness does not model, an OOM between the two writes);
    the hook falls back to the note, loudly.
    """
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_due(final=True))

    rows = peek_inbox(tmp_path)
    assert len(rows) == 1 and rows[0].wake is True
    assert host._wake_rearms == [], "nothing to hand over, and nothing claimed"


class PersistHost:
    """The real ``Session._persist_wake_schedules`` over a recording transcript.

    The one piece of the session that review round 2's MINOR 1 turns on: the
    queue a fire leaves behind has to reach disk through the persist the
    scheduler runs right after that fire, not through an exit that may never
    come.
    """

    _persist_wake_schedules = Session._persist_wake_schedules
    retire_wakes_to_inbox = Session.retire_wakes_to_inbox
    _spool_wake_to_inbox = Session._spool_wake_to_inbox
    _queue_wake_rearm = Session._queue_wake_rearm

    def __init__(self) -> None:
        self._wake_rearms: list[Any] = []
        self._wake_fired_since_persist = True  # the delivery that just happened
        self._session_id = "s1"
        self._cwd = "/tmp"
        self.appended: list[list[Any]] = []
        self.indexed: list[list[Any]] = []
        self.supervisors = 0
        self._transcript = SimpleNamespace(append_custom=self._record_append)

    async def _record_append(self, _kind: str, payload: dict[str, Any]) -> None:
        self.appended.append(payload["schedules"])

    def _write_wake_index_entry(self, schedules: list[Any], **_kwargs: Any) -> None:
        self.indexed.append(list(schedules))

    def _ensure_wake_supervisor(self) -> None:
        self.supervisors += 1

    def _missed_delivery_note(self, _due: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_a_final_fires_re_arm_is_durable_without_the_exit() -> None:
    """MINOR 1 (review round 2): durable at FIRE time, not at the exit.

    The queue used to be written only by ``hand_wakes_to_successor``, so from
    the fire to an exit that is deliberately unbounded the occurrence existed
    only in memory: the socket ``stop`` op, a SIGTERM, a dispose or a crash in
    that window lost it, and the pump had already persisted the retire, so
    nothing retried it. The scheduler persists immediately after the delivery
    and that persist IS this method, so the queue joins whatever list it is
    handed. The exit write stays as the retry.
    """
    host = PersistHost()
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())
    assert host._wake_rearms, "the fire queued a re-arm"

    # The pump's own persist, right after that delivery: the list it hands over
    # is the post-retire one, WITHOUT the re-arm in it.
    await host._persist_wake_schedules([])

    assert len(host.appended) == 1
    # The transcript payload is the DUMPED form — that is what becomes durable.
    written = host.appended[0]
    assert [row["id"] for row in written] == ["w1"], "the re-arm rode the fire's persist"
    assert written[0]["every_ms"] is None and written[0]["limit"] is None, "and as the one-shot"
    assert [schedule.id for schedule in host.indexed[0]] == [
        "w1"
    ], "the index too, so the supervisor can engage"
    assert host.supervisors == 1, "which is the install-on-demand hook the index write feeds"


@pytest.mark.asyncio
async def test_the_durable_re_arm_supersedes_a_live_copy_of_the_same_id(tmp_path: Path) -> None:
    """One row per id on disk: the re-armed one-shot replaces the live schedule."""
    host = PersistHost()
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())
    live = WakeSchedule(id="w1", message="check the deploy", next_due_at=9, every_ms=3_600_000)
    other = WakeSchedule(id="w2", message="hourly", next_due_at=9, every_ms=3_600_000)

    await host._persist_wake_schedules([live, other])

    written = host.appended[0]
    assert [row["id"] for row in written] == ["w2", "w1"]
    assert written[1]["every_ms"] is None, "the surviving w1 is the re-arm, not the live copy"


@pytest.mark.asyncio
async def test_the_handover_writes_live_and_re_armed_wakes_once(tmp_path: Path) -> None:
    """The successor is engaged by the INDEX, so the re-arm has to reach it
    through the one writer of schedule state — with the same id appearing once,
    the live copy of a re-armed schedule dropped in favour of the one-shot."""
    live = WakeSchedule(id="w2", message="hourly", next_due_at=2, every_ms=3_600_000)
    superseded = WakeSchedule(id="w1", message="check the deploy", next_due_at=1)
    host = WakeHost(tmp_path, live=(live, superseded))
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())

    assert await host.hand_wakes_to_successor() == 1
    assert len(host.persisted) == 1, "one write, transcript-then-index"
    written = host.persisted[0]
    assert [schedule.id for schedule in written] == ["w2", "w1"], "one row per id, re-arm last"
    assert written[1].every_ms is None, "and the surviving w1 is the one-shot"
    assert await host.hand_wakes_to_successor() == 0, "handed over exactly once"


@pytest.mark.asyncio
async def test_an_unspoolable_wake_is_loud_and_does_not_raise(tmp_path: Path) -> None:
    """A drain must not die on a spool write, and a lost wake must leave a trace."""
    host = WakeHost(None)
    await host._spool_wake_to_inbox(_due())


@pytest.mark.asyncio
async def test_the_spooled_row_reaches_the_next_boot_drain(tmp_path: Path) -> None:
    """The end of the chain: what the drain spools is what the successor's boot
    drain delivers, in write order."""
    from local_operator.session.runtime.inbox import (
        InboxLine,
        append_inbox,
        drain_inbox,
    )

    assert append_inbox(
        tmp_path,
        InboxLine(text="wake fired while draining", sender={}, mode="mailbox", written_at=1.0),
    )
    lines = drain_inbox(tmp_path)
    assert [line.text for line in lines] == ["wake fired while draining"]
    assert lines[0].to_json()["mode"] == "mailbox"


# -- the owner's own prompt during the drain (memo §4.2 piece 3 / §7.2) ----------
#
# The incident's user-visible half. A drain refuses new turns for as long as its
# own work takes — measured at 1 h 40 m on the operator's host, with 21 sessions
# draining at once — and for all of it the composer accepted text that the
# runtime then handed straight back ("send it again once the session is running
# again"), which is advice to redo the operation the refusal just performed.
#
# A peer's wake already had the better answer one method over (a spool the
# successor drains at boot). These cells are that answer applied to the OWNER's
# message, and they are driven through the real ``prompt`` because the decision
# — admitted, spooled, or refused — is the method's own.


def _attachment() -> list[ImageContent]:
    """One already-bounded attachment, which is the shape that reaches ``prompt``.

    A WIRE paste (``{"data_b64": ...}``) is decoded by the imaging pipeline on
    the way in, and an entry the sniffer does not recognise is DROPPED there —
    so a wire fixture would have to carry a real encoded image to reach this
    branch at all, while what the branch decides on is the blocks that arrive.
    The composer's own paste path and every in-process caller hand over
    ``ImageContent``, which ``serving._already_bounded`` passes straight through.
    """
    return [ImageContent(data="AAAA", mime_type="image/png")]


class PromptHost(DrainHost):
    """``DrainHost`` plus the real admission path.

    Bound the way this file binds every other latch: the production method over
    stub collaborators, so the assertion lands on the ADMISSION rather than on a
    runtime boot (the e2e covers the boot).
    """

    prompt = ServingSessionHandle.prompt
    steer = ServingSessionHandle.steer
    is_busy = ServingSessionHandle.is_busy

    def __init__(self, session: PromptSession, *, busy: bool = True) -> None:
        super().__init__(session, busy=busy)
        self._prompt_commands: dict[str, Any] = {}
        self._command_reservations = CommandReservations(session)
        self._prompt_queue: deque[Any] = deque()
        self._prompt_drain_task: asyncio.Task[None] | None = None
        self._pending_futures: dict[str, Any] = {}
        self._background_tasks: set[Any] = set()
        self._mcp_grant_tasks: set[Any] = set()
        self._mcp_reload_tasks: set[Any] = set()
        self._goal_loop = None
        self._loop = asyncio.get_running_loop()
        self._fold = SimpleNamespace(
            note_peer_message=lambda *_a, **_k: None,
            note_user_message=lambda *_a, **_k: None,
        )
        self._projection = SimpleNamespace(queued_count=0)


class PromptSession(FakeSession):
    """The slice of Session ``prompt`` touches before it decides."""

    def __init__(self, directory: Path, *, busy: bool = True) -> None:
        super().__init__(directory, busy=busy)
        self.prompt_calls: list[str] = []
        self.steered: list[str] = []

    def has_admitted_command(self, command_id: str) -> bool:
        return False

    def running_subagents(self) -> int:
        return 0

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        self.prompt_calls.append(text)

    def steer(self, text: str, images: Any = None, **kwargs: Any) -> None:
        self.steered.append(text)


def _prompt_host(tmp_path: Path, *, busy: bool = True) -> tuple[PromptHost, PromptSession]:
    session = PromptSession(tmp_path / "sessions" / "s1", busy=busy)
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return PromptHost(session, busy=busy), session


@pytest.mark.asyncio
async def test_a_prompt_during_the_drain_is_spooled_for_the_successor(tmp_path: Path) -> None:
    """NOT refused: the message is carried to the build that is taking over.

    The receipt is the part that has to be unmistakable — ``prompt``'s normal
    ACK is the DURABLE APPEND, so a caller that rendered this sentence as
    "admitted" would be claiming the user's message is in a history it has not
    reached yet. It is a different receipt precisely so a viewer can say
    "queued for the next runtime" instead of "sent" (memo §4.4).
    """
    host, session = _prompt_host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.prompt("deploy the fix", command_id="p" * 8)

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    assert receipt != "prompt admitted"
    assert session.prompt_calls == [], "a turn was started on the build that is leaving"

    rows = peek_inbox(session.transcript.directory)
    assert len(rows) == 1
    assert rows[0].text == "deploy the fix"
    assert rows[0].source == SOURCE_USER, "the successor must not deliver this as a peer's"
    assert rows[0].command_id == "p" * 8, "the admission identity has to survive the handover"
    assert rows[0].wake is True, "a user prompt asks for a turn"


@pytest.mark.asyncio
async def test_a_spooled_prompt_leaves_the_drain_free_to_exit(tmp_path: Path) -> None:
    """The spool must not become in-flight work, or it would hold the drain open.

    ``process._drain_for`` leaves at the first instant ``is_busy()`` is false,
    and the whole point of spooling is that the message is NOT this runtime's
    work any more — it belongs to the successor. A spool that read as a queued
    prompt would make the drain wait for a turn this runtime is never going to
    run, which is the 1 h 40 m hold rebuilt one layer down.
    """
    host, session = _prompt_host(tmp_path, busy=False)
    assert host.begin_drain("runtime-retired", "declined 3x") is True
    assert host.is_busy() is False, "premise: an idle draining runtime may leave"

    await host.prompt("deploy the fix", command_id="p" * 8)

    assert host.is_busy() is False, "the queued message holds the drain open"
    assert host._prompt_queue == deque(), "the spool queued a turn locally"


@pytest.mark.asyncio
async def test_a_prompt_with_an_attachment_is_refused_rather_than_spooled(
    tmp_path: Path,
) -> None:
    """The one message the vehicle cannot carry, refused instead of silently shorn.

    An inbox row is text. Spooling an image-carrying prompt would answer with a
    receipt for a message that arrives without its attachment — the user's file
    lost while they were told it was queued. The refusal returns the draft, and
    the viewer's composer claim is true there because the text IS still theirs.
    """
    host, session = _prompt_host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    with pytest.raises(RuntimeRetiring) as caught:
        await host.prompt(
            "what is wrong here?",
            images=_attachment(),
            command_id="p" * 8,
        )

    assert RuntimeRetiring.REFUSED in str(caught.value), str(caught.value)
    assert peek_inbox(session.transcript.directory) == [], "the attachment was shed, not queued"


@pytest.mark.asyncio
async def test_a_committed_exit_refuses_a_prompt_instead(tmp_path: Path) -> None:
    """Once the exit is being taken there is no successor window left to write to.

    The same contract the peer wake gets one method over, and for the same
    reason: a spool written after the commit would be read by nobody, so the
    user is told the truth — this session is not taking the message — rather
    than being given a receipt for it.
    """
    host, session = _prompt_host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired") is True

    with pytest.raises(RuntimeRetiring) as caught:
        await host.prompt("too late", command_id="p" * 8)

    assert "send it again" in str(caught.value)
    assert peek_inbox(session.transcript.directory) == []


@pytest.mark.asyncio
async def test_a_build_refusal_names_the_new_build_and_asks_for_the_resend(tmp_path: Path) -> None:
    """The refusal's own half of the handover, when it is the only answer left.

    A build drain OWES a successor, so the old tail — "send it again once the
    session is running again" — sent the operator to do the one thing the
    refusal had just refused. The sentence names the new build as the place the
    message goes, and asks for the re-send, which is the one act left: this is
    the fallback path (nowhere to spool, or an exit already committed), so what
    is established is the handover and the lost message, never its carriage —
    an earlier wording promised exactly that while the spool had failed (QA
    round 1, Q-2).
    """
    host, session = _prompt_host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    with pytest.raises(RuntimeRetiring) as caught:
        await host.prompt(
            "with an attachment",
            images=_attachment(),
            command_id="p" * 8,
        )

    assert caught.value.trigger == RuntimeRetiring.BUILD
    assert "send it again once the new build is up" in str(caught.value)
    assert "carry on" not in str(caught.value), str(caught.value)
    assert "queued" not in str(caught.value), str(caught.value)
    assert not [line for line in session.notes if "queued" in line]


@pytest.mark.asyncio
async def test_an_attached_viewer_never_decides_whether_a_drain_may_exit(tmp_path: Path) -> None:
    """§4.6 pinned on the REAL predicate, as the confound UX round 1 handed over.

    UX round 1 declined to file a finding it could not pin: "if an idle viewer
    really does cause an early retirement that cuts a running tool, that is a
    second mechanism for the operator's incident". The exit path's liveness is
    ``may_refresh`` → ``is_busy``, and an attached front end is not a term in
    either direction — a viewer must not HOLD a drain open (that is the five-hour
    resident runtime ``may_refresh``'s own docstring exists to prevent) and must
    not RELEASE one either. The second half is the one that would cut in-flight
    work, so both are asserted here with a client count registered.

    ``DrainHost`` stubs ``may_refresh`` for every other cell in this file, so the
    production predicate is bound explicitly: the property under test IS the
    production one, and a stub cannot carry it.
    """
    host, session = _prompt_host(tmp_path, busy=False)
    # Bound through `Any`: this file's hosts are partial doubles of the class the
    # real predicate is declared on, which is the point — the production method
    # is what must hold, over the collaborators it actually reads.
    real: Any = ServingSessionHandle.may_refresh

    # No work, no viewer: free to act on a newer build.
    assert real(host) == "", real(host)
    # A viewer attached, still no work: STILL free. A viewer that held here is
    # the residency bug this predicate was written to end.
    setattr(host, "_registrant", SimpleNamespace(attach_clients=lambda: 1))
    assert real(host) == "", "an idle viewer must never hold a runtime resident"

    # Work in flight — a live turn, which is the tool that must not be cut —
    # decides alone, and the attached viewer changes nothing about that.
    setattr(session, "is_streaming", True)
    assert real(host) == "busy", "a live turn must keep the drain from exiting"
    setattr(host, "_registrant", SimpleNamespace(attach_clients=lambda: 0))
    assert real(host) == "busy", "the WORK decides; the viewer count may not"


@pytest.mark.asyncio
async def test_a_steer_during_the_drain_is_still_admitted(tmp_path: Path) -> None:
    """Pinned: the spool must not have widened into the steering path.

    A steer is the correction of a turn that is ALREADY running, so it is not a
    new turn and the drain's latch has no business refusing it (measured A4 on
    ``origin/main``). This cell exists so the admission rewrite above cannot
    quietly route steering through the spool and cost the user their correction
    until the successor boots.
    """
    host, session = _prompt_host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    detail = await host.steer("actually, the other way")

    assert detail == "steering queued", detail
    assert session.steered == ["actually, the other way"], "the session never saw it"
    assert peek_inbox(session.transcript.directory) == [], "a steer is not deferred"

# -- the promise a spooled row carries, and who keeps it -------------------------
#
# THE CIRCUIT, and these cells are the writer half of it. The receipt a spooling
# runtime hands back says the NEXT runtime runs the message, and this process
# cannot start one: it holds the transcript lease until it exits. Measured
# 2026-09-21: a session retired for a newer build with rows in its spool and no
# successor came for them. So the writer records the turn as owed
# (``wakes.spooled``), which is what makes the wake supervisor — the one
# always-on process whose job is "make a runtime exist" — raise one.


def _spooled_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the owed-turn store at this test's own config dir.

    ``_spool_for_successor`` reads ``paths.config_dir()``, the same seam every
    other store on this path uses, so the redirect is that one function and the
    test never touches a real store.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config_dir)
    return config_dir


@pytest.mark.asyncio
async def test_a_spooled_peer_wake_records_the_turn_the_successor_owes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.wakes.spooled import read_spooled_turn

    host, session = _prompt_host(tmp_path)
    config_dir = _spooled_store(monkeypatch, tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.receive_peer_message("run the census", wake=True)

    assert receipt == SPOOL_RECEIPT_WAKE, receipt
    record = read_spooled_turn(config_dir, session.transcript.directory.name)
    assert record is not None, "the receipt promised a turn; nothing recorded it"
    assert record["rows"] == 1


@pytest.mark.asyncio
async def test_a_spooled_quiet_note_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``wake=False`` asked for a deferral, not for a runtime to be raised.

    The receipt says so ("read when it next opens"), and raising a 283 MB process
    for a note nobody is waiting on is the trade ``deliver_peer_message`` argues
    against. This cell pins that the obligation follows the SAME line as the
    receipt, so the two can never disagree about what the sender bought.
    """
    from local_operator.wakes.spooled import read_spooled

    host, session = _prompt_host(tmp_path)
    config_dir = _spooled_store(monkeypatch, tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.receive_peer_message("no rush", wake=False)

    assert receipt != SPOOL_RECEIPT_WAKE
    assert read_spooled(config_dir) == {}, "a quiet note owes no turn"


@pytest.mark.asyncio
async def test_the_owners_own_prompt_records_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's words are the strongest form of the claim: the receipt RUNS it."""
    from local_operator.wakes.spooled import read_spooled_turn

    host, session = _prompt_host(tmp_path)
    config_dir = _spooled_store(monkeypatch, tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.prompt("deploy the fix", command_id="q" * 8)

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    record = read_spooled_turn(config_dir, session.transcript.directory.name)
    assert record is not None, "the receipt says the next runtime will run it"
    assert record["rows"] == 1
