"""The victim-side sanction gate: a terminating signal nothing staged ends nothing.

WHY THIS FILE EXISTS (2026-10-30 21:57:52, measured): twelve background exec workers
and at least five session runtimes were ended by SIGTERM inside 1.33 s while the
operator force-quit a frozen desktop app. Every receipt said ``sender: unavailable``
(macOS exposes no ``siginfo``), no stop marker had been staged for any of them, and no
``logs/stop-sweeps.jsonl`` existed — so no in-product stop was ever asked for and the
sender cannot be named. The defence therefore cannot be "the sender is different"; it
has to hold at the victim: **no terminating signal ends a runtime unless a durable
sanction covers that run** (``docs/design-surface-independence.md`` §7 at
``3ba837e6d9e``, slice 1).

WHAT IS PINNED HERE, all through the REAL routing (``process._route_signal``) with a
REAL conversation directory and markers written by the production writers:

* an unsanctioned SIGTERM is REFUSED: the stop event is not set, no drain starts, the
  turn's cause is not armed, the refusal record carries the arrival with
  ``disposition: refused``, a WARNING names it, and the run counts it;
* **the refusal is not pairing-eligible evidence and never touches the signal
  receipt**: ``runtime-signal.json`` is NOT written at all, so ``covers_run`` cannot
  pair the run with it, ``cut_off_verdict`` cannot let it outrank a later stop, and
  the rescue lane cannot reach its terminal ``unclassified/receipt-no-signals`` state;
* the same SIGTERM after ``control.note_involuntary_stop`` staged its marker drains
  exactly as before (the wake supervisor's and ``reclaim``'s stop paths must not
  break — this is why the permit is ``stop_class_of`` and not ``cut_off_verdict``);
* the same SIGTERM after a DELIBERATE marker drains exactly as before and arms no
  cut-off cause;
* an IDLE runtime refuses too — the fast branch is not a licence for an unnamed kill;
* a marker for another run, or staged after the arrival, sanctions nothing;
* refusals coalesce, so a sender looping SIGTERM cannot grow the receipt;
* a broken instrument does not make an unkillable runtime: with nothing to classify,
  the arrival behaves as today.

The e2e half (a real runtime process, a real SIGTERM, a real parked turn, and SIGKILL's
continued effect) is ``tests/e2e/test_signal_drain_e2e.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.runtime import control, process, registry, signal_receipt

SESSION_ID = "gated0000001"
STARTED_AT = 1_760_000_000.0


class _Transcript:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _Session:
    """The session surface the routing reads, with the hooks it may call."""

    def __init__(self, directory: Path, *, busy: bool) -> None:
        self.session_id = SESSION_ID
        self.busy = busy
        self._transcript = _Transcript(directory)
        self.cut_off_notes: list[tuple[str, str]] = []
        self.deliberate_notes = 0

    def is_busy(self) -> bool:
        return self.busy

    def may_refresh(self) -> str:
        return "busy" if self.busy else ""

    def disposal_cuts_a_turn(self) -> bool:
        return self.busy

    def note_cut_off(self, cause: str, detail: str = "") -> None:
        self.cut_off_notes.append((cause, detail))

    def note_deliberate_stop(self) -> None:
        self.deliberate_notes += 1


class _Handle:
    """The handle shape ``_route_signal`` reads.

    ``is_busy`` lives on the HANDLE in production (``_busy_verdict`` asks the handle,
    not the session), so a double that put it on the session only would silently
    report "idle" and pass the idle cells for the wrong reason.
    """

    _draining = False

    def __init__(self, session: _Session) -> None:
        self._session = session

    def is_busy(self) -> bool:
        return self._session.is_busy()


class _Runtime:
    class _Record:
        session_id = SESSION_ID
        # Fixed so a marker's run key must match it: a test that staged a marker for
        # another run would then fail to pair, which is the rule being kept.
        started_at = STARTED_AT

    _record = _Record()
    _boot_build = None


def _record(*, pid: int | None = None) -> Any:
    """A duck-typed record for ``control._stop_marker_payload`` (its own convention)."""
    return SimpleNamespace(
        session_id=SESSION_ID,
        pid=os.getpid() if pid is None else pid,
        started_at=STARTED_AT,
        version="0.0.0",
        source_ref="test",
    )


def _stage_deliberate(directory: Path, *, pid: int | None = None) -> None:
    """Stage a stop marker the way ``control.stop_session``'s rungs do.

    The production payload builder, not a hand-written dict: the gate reads the
    marker the ladder writes, so a test that invented its own shape could pass while
    the real one failed to pair.
    """
    registry.write_stop_marker(
        directory,
        control._stop_marker_payload(_record(pid=pid), "sigterm", command="lop stop"),
    )


def _stage_involuntary(root: Path) -> None:
    """Stage the marker ``wakes/owned_runtimes.sweep`` writes before its signal."""
    assert control.note_involuntary_stop(
        _record(), root, mechanism="wake-sweep", actor="wake supervisor"
    )


async def _drive(session: _Session, sig: signal.Signals = signal.SIGTERM):
    """Run the REAL routing once; return (state, last receipt)."""
    state = process._SignalState(stop=asyncio.Event(), trigger={})
    process._route_signal(_Handle(session), _Runtime(), sig, state=state)
    return state, (state.receipts[-1] if state.receipts else None)


async def _settle(state: process._SignalState) -> None:
    """Cancel the drain ``_route_signal`` may have started, so nothing leaks."""
    if state.draining is not None:
        state.draining.cancel()
        try:
            await state.draining
        except asyncio.CancelledError:
            pass


@pytest.fixture()
def directory(tmp_path: Path) -> Path:
    """The conversation directory a runtime's transcript owns (``sessions/<id>``)."""
    target = control.session_dir(tmp_path, SESSION_ID)
    target.mkdir(parents=True)
    return target


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_an_unsanctioned_sigterm_is_refused_and_the_turn_keeps_running(
    directory: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The measured shape: nothing staged, a busy runtime, a signal out of nowhere."""
    session = _Session(directory, busy=True)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.runtime.process"):
        state, receipt = await _drive(session)

    # IT DID NOT TAKE THE RUN AWAY — the whole point.
    assert not state.stop.is_set(), "an unsanctioned signal ended the runtime"
    assert state.draining is None, "an unsanctioned signal started a drain"
    assert state.trigger == {}, f"an unsanctioned signal recorded a trigger: {state.trigger}"
    assert session.cut_off_notes == [], "a turn nothing cut was branded as cut off"
    assert session.deliberate_notes == 0

    # IT IS OBSERVABLE: a durable row, a loud line, a tally.
    assert state.refusals == 1
    assert registry.read_signal_receipt(directory) is None, "a refusal wrote a receipt"
    on_disk = registry.read_signal_refusals(directory)
    assert on_disk is not None, "no refusal record was written for the arrival"
    refusal = on_disk["refusals"][0]
    assert refusal["disposition"] == "refused"
    assert (refusal["name"], refusal["number"]) == ("SIGTERM", 15)
    assert refusal["sanction"] == "none"
    assert refusal["count"] == 1
    assert refusal["sender"]["state"] == "unavailable"
    assert receipt is None, "the router reported a refusal as a signal receipt"
    assert "REFUSED" in caplog.text, caplog.text
    assert SESSION_ID in caplog.text, "the line must name the run it is about"
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_an_idle_runtime_refuses_too(directory: Path) -> None:
    """The fast branch is not a licence for an unnamed kill.

    An idle runtime exits in one synchronous step today, which is why the old handler
    could look harmless. Under the gate the same arrival must leave it running:
    "nobody asked" is the fact being refused, not "a turn was busy".
    """
    session = _Session(directory, busy=False)
    state, receipt = await _drive(session)
    assert not state.stop.is_set(), "an unsanctioned signal stopped an idle runtime"
    assert state.refusals == 1
    assert receipt is None
    refusals = registry.read_signal_refusals(directory)
    assert refusals is not None
    assert refusals["refusals"][0]["in_flight"] is False
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_the_refusal_is_not_pairing_eligible_evidence(directory: Path) -> None:
    """The contract's reason for a SEPARATE FILE, asserted rather than trusted.

    ``covers_run`` accepts an entry on session id + pid + ``started_at`` alone, and
    ``cut_off_verdict`` reads the LAST entry of ``signals``. A refused arrival in that
    list would therefore pair with this run forever — and outrank a genuine later stop.
    The refusal goes to its own file instead, so the receipt does not exist at all for
    this run and the stop-attribution lane falls through to other evidence exactly as
    it does today. ``rescue``'s reader is the other half: see
    ``tests/unit/session/runtime/test_rescue.py`` (R-9).
    """
    session = _Session(directory, busy=True)
    state, _ = await _drive(session)

    assert registry.read_signal_receipt(directory) is None, "a refusal wrote a receipt"
    dead = SimpleNamespace(pid=os.getpid(), started_at=STARTED_AT)
    assert signal_receipt.covers_run(None, SESSION_ID, dead) is False
    refusals = registry.read_signal_refusals(directory)
    assert refusals is not None and refusals["refusals"][0]["disposition"] == "refused"
    # The refusal is keyed to this run, so it is evidence ABOUT it — without being
    # evidence a signal was considered to have landed.
    assert refusals["pid"] == os.getpid()
    assert str(refusals["session_id"]) == SESSION_ID
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_refused_run_that_is_later_stopped_keeps_both_facts(directory: Path) -> None:
    """A refusal and the sanctioned stop that follows it are both kept, in two files.

    The stop that lands afterwards must read to every existing reader exactly as it
    would have if the refusal had never happened — same receipt, same run key, same
    ``covers_run`` answer — while the refusal stays available to a human.
    """
    session = _Session(directory, busy=True)
    refused, _ = await _drive(session)
    assert refused.refusals == 1
    assert registry.read_signal_receipt(directory) is None

    _stage_deliberate(directory)
    accepted, receipt = await _drive(session)
    assert accepted.refusals == 0
    assert accepted.draining is not None

    on_disk = registry.read_signal_receipt(directory)
    assert on_disk is not None, "the accepted stop wrote no receipt"
    assert len(on_disk["signals"]) == 1, "the accepted stop is the only signal"
    assert on_disk["signals"][0]["sanction"] == "marker"
    assert on_disk["count"] == 1, "a refused arrival inflated the receipt's signal count"
    assert "refusals" not in on_disk, "a refusal leaked into the signal receipt"
    dead = SimpleNamespace(pid=os.getpid(), started_at=STARTED_AT)
    assert signal_receipt.covers_run(on_disk, SESSION_ID, dead) is True
    refusals = registry.read_signal_refusals(directory)
    assert refusals is not None and refusals["refusals"][0]["disposition"] == "refused"
    await _settle(accepted)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_an_involuntary_marker_sanctions_the_signal_that_follows_it(
    directory: Path, tmp_path: Path
) -> None:
    """``wakes/owned_runtimes.sweep``'s stop path must still stop its runtimes.

    The sweep's marker is ``deliberate: false`` — nobody asked for the stop, the
    supervisor is going away — and it must still be OBEYED. If the gate keyed on
    ``deliberate`` instead of the stop class, this cell would go red and the wake
    supervisor's stop would silently become a no-op. The same shape covers
    ``reclaim``'s sweep, which is why the permit is not ``cut_off_verdict``.
    """
    _stage_involuntary(tmp_path)
    session = _Session(directory, busy=True)
    state, receipt = await _drive(session)

    assert state.refusals == 0
    assert state.draining is not None, "a sanctioned signal did not drain"
    entry = (receipt or {})["signals"][-1]
    assert entry["sanction"] == "marker"
    assert entry["stop_marker"]["mechanism"] == "wake-sweep"
    assert entry["action"] == "drain"
    # The cause is armed for the turn this exit will cut, exactly as before.
    assert session.cut_off_notes and session.cut_off_notes[0][0] == "runtime-killed"
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_deliberate_marker_keeps_todays_drain_byte_for_byte(directory: Path) -> None:
    """The sanctioned path: the ladder staged its marker, so the signal is obeyed."""
    _stage_deliberate(directory)
    session = _Session(directory, busy=True)
    state, receipt = await _drive(session)

    assert state.refusals == 0
    assert state.trigger == {"why": "SIGTERM"}
    assert state.draining is not None
    on_disk = registry.read_signal_receipt(directory)
    assert on_disk is not None
    entry = on_disk["signals"][-1]
    assert (entry["sanction"], entry["action"]) == ("marker", "drain")
    assert entry["stop_marker"]["deliberate"] is True
    # A run that refused nothing carries no ``refusals`` key at all: the receipt is
    # byte-identical to the v1 shape every existing reader was taught.
    assert "refusals" not in on_disk
    # A deliberate marker earns NO cut-off cause: a stop somebody asked for stays
    # ``interrupted``/``user-stop`` on every surface (wave B's rule, unchanged).
    assert session.cut_off_notes == []
    assert receipt is not None
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_sanctioned_idle_signal_still_leaves_at_once(directory: Path) -> None:
    """The other half of "byte for byte": an idle sanctioned signal sets ``stop``."""
    _stage_deliberate(directory)
    session = _Session(directory, busy=False)
    state, _ = await _drive(session)
    assert state.stop.is_set(), "a sanctioned idle signal no longer leaves"
    assert state.refusals == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_repeat_after_a_refusal_coalesces_rather_than_grows(directory: Path) -> None:
    """A sender looping SIGTERM cannot grow the receipt, and the count still says so."""
    session = _Session(directory, busy=True)
    first, _ = await _drive(session)
    second, receipt = await _drive(session)

    assert first.refusals == 1 and second.refusals == 1
    assert not second.stop.is_set() and second.draining is None
    assert receipt is None
    refusals = registry.read_signal_refusals(directory)
    assert refusals is not None
    assert len(refusals["refusals"]) == 1, "refusals were appended instead of coalesced"
    assert refusals["refusals"][0]["count"] == 2
    await _settle(first)
    await _settle(second)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_stale_marker_for_another_run_does_not_sanction_this_one(
    directory: Path,
) -> None:
    """Pairing is per run, and the gate inherits it rather than loosening it.

    A marker naming a different pid is refused for this run exactly as
    ``marker_covers_signal`` refuses it for the receipt — which is what stops an
    earlier stop attempt from narrating a later, unrelated arrival.
    """
    _stage_deliberate(directory, pid=os.getpid() + 1)
    session = _Session(directory, busy=True)
    state, receipt = await _drive(session)

    assert state.refusals == 1, "a marker for another pid sanctioned this run"
    assert not state.stop.is_set()
    assert receipt is None
    recorded = registry.read_signal_refusals(directory)
    assert recorded is not None
    assert recorded["refusals"][0]["sanction"] == "none"
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_marker_staged_after_the_signal_does_not_rewrite_the_refusal(
    directory: Path,
) -> None:
    """Ordering is the invariant: the sanction must exist BEFORE the arrival."""
    session = _Session(directory, busy=True)
    state, _ = await _drive(session)
    assert state.refusals == 1

    _stage_deliberate(directory)
    assert registry.read_stop_marker(directory) is not None
    assert registry.read_signal_receipt(directory) is None
    refusals = registry.read_signal_refusals(directory)
    assert refusals is not None
    assert refusals["refusals"][0]["disposition"] == "refused"
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_broken_instrument_does_not_make_an_unkillable_runtime(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing to classify, behave as today.

    The gate's own rule: refuse only where it can ask. ``observe`` failing is a
    programmer-error path, and refusing on a broken instrument would turn a bug into
    a runtime that no SIGTERM can end — a worse failure than the sweep.
    """
    monkeypatch.setattr(
        "local_operator.session.runtime.signal_receipt.observe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("instrument down")),
    )
    session = _Session(directory, busy=True)
    state, receipt = await _drive(session)

    assert receipt is None
    assert state.refusals == 0, "a broken instrument refused a signal"
    assert state.draining is not None, "a broken instrument stopped the drain"
    await _settle(state)


def test_the_permit_is_the_stop_class_not_the_verdict() -> None:
    """The predicate, at its edges — including the case that rules out ``cut_off_verdict``."""
    deliberate = {
        "sanction": "marker",
        "stop_marker": {"deliberate": True},
    }
    involuntary = {
        "sanction": "marker",
        "stop_marker": {"deliberate": False, "mechanism": "reclaim"},
    }
    assert signal_receipt.is_sanctioned(deliberate) is True
    assert (
        signal_receipt.is_sanctioned(involuntary) is True
    ), "an involuntary marker must be obeyed, so the permit cannot be cut_off_verdict"
    # ... which is exactly why: the involuntary marker HAS a verdict (runtime-killed).
    assert signal_receipt.cut_off_verdict(involuntary) is not None
    assert signal_receipt.is_sanctioned(None) is False
    assert signal_receipt.is_sanctioned({}) is False
    assert signal_receipt.is_sanctioned({"sanction": "none"}) is False


def test_every_default_fatal_stop_signal_takes_the_same_route() -> None:
    """The residual levers are closed, and the three exemptions are structural.

    ``SIGKILL`` cannot be handled at all; ``SIGUSR1`` is the task-stack dump whose own
    handler is what makes it non-fatal; ``SIGHUP`` is already a no-op. Everything a
    sender can otherwise END a runtime with must reach the router, because a signal
    with no handler is fatal by default and would bypass the gate entirely.
    """
    seen: list[signal.Signals] = []
    routes = process._stop_signal_routes(seen.append)

    expected = {signal.SIGTERM, signal.SIGINT}
    for name in ("SIGQUIT", "SIGUSR2", "SIGALRM"):
        member = getattr(signal, name, None)
        if member is not None:
            expected.add(member)
    assert {signal.Signals(key) for key in routes} == expected

    for callback in routes.values():
        callback()
    assert {signal.Signals(sig) for sig in seen} == expected, "a route did not reach the router"

    for name in ("SIGKILL", "SIGUSR1", "SIGHUP"):
        member = getattr(signal, name, None)
        if member is not None:
            assert int(member) not in routes, f"{name} must keep its own disposition/handler"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_sigquit_is_gated_like_a_sigterm(directory: Path) -> None:
    """The gate is about the CLASS (a terminating arrival), not one signal number."""
    session = _Session(directory, busy=True)
    state, receipt = await _drive(session, signal.SIGQUIT)
    assert state.refusals == 1
    assert not state.stop.is_set() and state.draining is None
    assert receipt is None
    recorded = registry.read_signal_refusals(directory)
    assert recorded is not None
    assert recorded["refusals"][0]["name"] == "SIGQUIT"
    await _settle(state)

    _stage_deliberate(directory)
    sanctioned, receipt = await _drive(session, signal.SIGQUIT)
    assert sanctioned.refusals == 0
    assert sanctioned.draining is not None, "a sanctioned SIGQUIT must still end the run"
    assert (receipt or {})["signals"][-1]["name"] == "SIGQUIT"
    await _settle(sanctioned)


def test_the_refusal_file_is_classified_by_both_enumerating_registries(
    tmp_path: Path,
) -> None:
    """A new name-keyed artefact in a session directory TRAPS the session if unclassified.

    ``sync.assert_complete`` refuses a deleting move whose source directory holds
    anything the copy set does not account for, so an unlisted refusals file would make
    a session that refused ONE signal permanently unmovable until someone deleted the
    evidence — the failure ``EXCLUDED_ENTRIES``' own comment records for an earlier
    artefact. ``retention._SIDECAR_NAMES`` is the silent twin: unlisted there, the file
    is counted as user CONTENT by the size budget and moves the activity clock.

    Asserted through the CONSTANT rather than a literal at either site, so a rename
    cannot leave both registries classifying a name nothing writes. The consequence is
    asserted too, not just the membership: a directory holding one must pass the guard
    that a move runs.
    """
    from local_operator.network import sync
    from local_operator.session import retention

    name = signal_receipt.SIGNAL_REFUSALS_NAME
    assert name == registry.SIGNAL_REFUSALS_NAME, "two spellings of one artefact"
    assert name in sync.COPY_SET_NAMES, "a refusal would make the session unmovable"
    assert name not in sync.EXCLUDED_ENTRIES, "a file the copy set carries cannot be excluded"
    assert name in retention._SIDECAR_NAMES, "the size budget would count evidence as content"

    directory = control.session_dir(tmp_path, SESSION_ID)
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    registry.write_signal_refusals(
        directory,
        {
            "v": 1,
            "kind": "runtime",
            "session_id": SESSION_ID,
            "pid": os.getpid(),
            "started_at": STARTED_AT,
            "refusals": [{"disposition": "refused", "name": "SIGTERM", "number": 15}],
        },
    )
    # THE GUARD A DELETING MOVE RUNS: it must accept this directory unchanged.
    sync.assert_complete(directory)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_signal_loop_cannot_grow_the_log(
    directory: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-4/Q4: the refusals FILE is one coalesced entry, so the LOG must be bounded too.

    One WARNING per arrival let a sender that retried SIGTERM grow ``runtime.log``
    without bound (100 signals -> 100 lines). The first refusal is always loud; the
    rest are counted, and the next line after the interval carries the number it
    stands in for. The tally and the durable file keep counting every arrival.
    """
    clock = [1000.0]
    monkeypatch.setattr(process.time, "monotonic", lambda: clock[0])
    caplog.set_level(logging.WARNING)
    session = _Session(directory, busy=True)
    state = process._SignalState(stop=asyncio.Event(), trigger={})
    handle = _Handle(session)

    for _ in range(50):
        process._route_signal(handle, _Runtime(), signal.SIGTERM, state=state)
        clock[0] += 0.01
    lines = [r for r in caplog.records if "REFUSED" in r.getMessage()]
    assert len(lines) == 1, f"{len(lines)} lines for one burst"
    assert state.refusals == 50, "the tally must still count every arrival"

    clock[0] += process._REFUSAL_LOG_INTERVAL_S
    process._route_signal(handle, _Runtime(), signal.SIGTERM, state=state)
    lines = [r for r in caplog.records if "REFUSED" in r.getMessage()]
    assert len(lines) == 2
    assert "49 more refused" in lines[-1].getMessage(), lines[-1].getMessage()

    recorded = registry.read_signal_refusals(directory)
    assert recorded is not None and recorded["refusals"][0]["count"] == 51
    await _settle(state)


# ---------------------------------------------------------------------------
# The sanction gate x the siginfo sender sensor (#3214)
# ---------------------------------------------------------------------------
#
# Both live in the signal router and both change what a record says. The sensor
# (``local_operator.siginfo``) names WHO sent a signal; the gate decides whether it is
# obeyed. These cells pin how they compose: the sender survives a refusal, the sensor is
# one-shot per process (first terminating signal named, later ones `no-siginfo`), a
# refusal never overwrites the receipt of a later sanctioned stop, and a
# kernel-originated signal stays unattributed in both files.


def _capture(monkeypatch: pytest.MonkeyPatch, *facts: dict[str, int] | None) -> None:
    """Make ``siginfo.take`` answer ``facts`` in order, then ``None`` (a spent sensor)."""
    from local_operator import siginfo

    queue = list(facts)
    monkeypatch.setattr(siginfo, "take", lambda _sig: queue.pop(0) if queue else None)


_KNOWN = {"signo": 15, "pid": 4321, "uid": 501, "code": 0}


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_sanctioned_sigterm_names_its_sender_and_drains_as_before(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(1) The receipt names the sender AND the drain is the one it always was."""
    _capture(monkeypatch, _KNOWN)
    _stage_deliberate(directory)
    state, receipt = await _drive(_Session(directory, busy=True))

    assert state.refusals == 0 and state.draining is not None, "a sanctioned stop must drain"
    assert receipt is not None
    sender = receipt["signals"][0]["sender"]
    assert (sender["state"], sender["pid"], sender["uid"]) == ("known", 4321, 501), sender
    assert registry.read_signal_refusals(directory) is None
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_an_unsanctioned_sigterm_is_refused_and_still_names_who_tried(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(2) The sensor's value is naming who TRIED — a refusal must not cost that fact."""
    _capture(monkeypatch, _KNOWN)
    state, receipt = await _drive(_Session(directory, busy=True))

    assert state.refusals == 1 and not state.stop.is_set() and state.draining is None
    assert receipt is None, "a refusal wrote a receipt"
    recorded = registry.read_signal_refusals(directory)
    assert recorded is not None
    sender = recorded["refusals"][0]["sender"]
    assert (sender["state"], sender["pid"], sender["uid"]) == ("known", 4321, 501), sender
    await _settle(state)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.asyncio
async def test_a_kernel_originated_signal_stays_unattributed_in_both_files(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(4) ``si_pid == 0`` is the kernel (a timer), never "process zero" — refused or not."""
    from local_operator import incidents

    kernel = {"signo": 15, "pid": 0, "uid": 0, "code": 0}
    _capture(monkeypatch, kernel)
    refused, _ = await _drive(_Session(directory, busy=True))
    recorded = registry.read_signal_refusals(directory)
    assert recorded is not None
    assert recorded["refusals"][0]["sender"] == signal_receipt.SENDER_KERNEL
    await _settle(refused)

    _capture(monkeypatch, kernel)
    _stage_deliberate(directory)
    accepted, receipt = await _drive(_Session(directory, busy=True))
    assert receipt is not None
    assert receipt["signals"][0]["sender"] == signal_receipt.SENDER_KERNEL
    for sender in (recorded["refusals"][0]["sender"], receipt["signals"][0]["sender"]):
        assert incidents.render_named_sender(sender) == "", "a kernel signal was given a name"
    assert "sender=unavailable" in signal_receipt.describe(receipt)
    await _settle(accepted)


@pytest.mark.skipif(
    os.name == "nt" or sys.platform != "darwin", reason="the sensor is armed on Darwin only"
)
@pytest.mark.asyncio
async def test_the_sensor_names_the_first_signal_only_and_a_later_stop_is_attributed_by_its_marker(
    directory: Path,
) -> None:
    """(3) The one-shot contract on a REAL kernel capture (SIGUSR2; SIGTERM is never armed here).

    ``siginfo`` is spent by the FIRST delivery and nothing re-arms it: re-installing it
    per refusal measured self-attributed receipts, hangs and SIGSEGVs under bursts. So
    the first terminating signal is named (here, a refusal); a sanctioned stop that
    follows is recorded ``no-siginfo`` and is attributed by its staged marker
    (``sanction: marker``) — and it still PROCEEDS. A refusal never overwrites a
    receipt, and the receipt never absorbs a refusal.
    """
    from local_operator import siginfo

    number = int(signal.SIGUSR2)
    previous = signal.signal(signal.SIGUSR2, lambda *_: None)
    try:
        assert siginfo.arm(number), "the sensor could not arm on this host"

        os.kill(os.getpid(), signal.SIGUSR2)
        await asyncio.sleep(0.2)
        refused, receipt = await _drive(_Session(directory, busy=True), signal.SIGUSR2)
        assert refused.refusals == 1 and receipt is None
        first = registry.read_signal_refusals(directory)
        assert first is not None
        assert first["refusals"][0]["sender"]["pid"] == os.getpid(), first
        await _settle(refused)

        _stage_deliberate(directory)
        os.kill(os.getpid(), signal.SIGUSR2)
        await asyncio.sleep(0.2)
        accepted, receipt = await _drive(_Session(directory, busy=True), signal.SIGUSR2)
        assert accepted.refusals == 0 and receipt is not None, "the sanctioned stop must proceed"
        sender = receipt["signals"][0]["sender"]
        assert sender == signal_receipt.SENDER_UNAVAILABLE, sender
        assert receipt["signals"][0]["sanction"] == "marker"

        # A REFUSAL NEVER OVERWRITES A RECEIPT, and the receipt never absorbs a refusal.
        after = registry.read_signal_refusals(directory)
        assert after is not None and after["refusals"][0]["count"] == 1
        assert after["refusals"][0]["sender"]["pid"] == os.getpid()
        assert len(receipt["signals"]) == 1
        await _settle(accepted)
    finally:
        siginfo.disarm(number)
        signal.signal(signal.SIGUSR2, previous)


@pytest.mark.skipif(
    os.name == "nt" or sys.platform != "darwin", reason="the sensor is armed on Darwin only"
)
@pytest.mark.asyncio
async def test_a_second_refused_signal_is_refused_and_recorded_unnamed(directory: Path) -> None:
    """(3) Two real refused signals: the first names the sender, the second is refused and
    recorded ``no-siginfo`` — coalesced into the one refusal entry, ``last_sender`` unnamed."""
    from local_operator import siginfo

    number = int(signal.SIGUSR2)
    previous = signal.signal(signal.SIGUSR2, lambda *_: None)
    try:
        assert siginfo.arm(number), "the sensor could not arm on this host"
        for _ in range(2):
            os.kill(os.getpid(), signal.SIGUSR2)
            await asyncio.sleep(0.2)
            state, receipt = await _drive(_Session(directory, busy=True), signal.SIGUSR2)
            assert state.refusals == 1 and receipt is None
            await _settle(state)
        recorded = registry.read_signal_refusals(directory)
        assert recorded is not None
        entry = recorded["refusals"][0]
        assert entry["count"] == 2
        assert entry["sender"]["pid"] == os.getpid()
        assert entry["last_sender"] == signal_receipt.SENDER_UNAVAILABLE, entry
    finally:
        siginfo.disarm(number)
        signal.signal(signal.SIGUSR2, previous)
