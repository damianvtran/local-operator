"""A signal that arrives mid-turn must not destroy the turn (2026-09-14).

THE INCIDENT, measured: the install went 0.54.46 → .47 → .48 and at
19:32:27.853-28.526 twenty-one live session runtimes were SIGTERM'd in one pass
(twenty in ``~/Library/Logs/local-operator/runtime.log``, one in
``~/.local-operator/logs/runtime.log`` — ``log_dir()`` resolves under
``<config>/logs`` only when ``LOCAL_OPERATOR_CONFIG_DIR`` is set, which is why
this first read as a single kill). Twenty-three daemon-owned sessions went
live→closed together with 32 turns cut off, each recorded ``kind=error,
cause=runtime-shutdown``.

The mechanism was an ASYMMETRY, not a bug in either half: the reaper consults
the work predicate and refuses to exit under a live turn (there is a passing
project test for the graceful path), while ``amain``'s signal handler set the
stop event outright. So the same event the runtime had just decided not to be
disturbed by was fatal when it arrived as a signal — and the sender of a sweep
is unnamed and cannot be taught manners, which is why the fix is receiver-side.

What these cells prove on the REAL surface — production ``process.py`` booted in
a subprocess, a real turn parked in the real ``bash`` tool via the mock
provider's ``[bash:N]`` marker, the production ``AttachedSession`` attached, and
for the rotation cells the production rotation path dialled over the real
control socket:

* a BROADCAST SIGTERM mid-turn is survived: every turn completes, nothing is
  recorded as cut off, and each runtime leaves at its own boundary instead;
* a signal with nothing in flight is still immediate (the fast branch);
* a DELIBERATE stop is still prompt mid-turn — via ``lop stop``'s own ladder and
  via ``stop_all`` — and no signal is sent for it;
* a frozen, silent runtime with a turn in flight is SKIPPED rather than
  signalled, and its work survives;
* a frozen runtime that never answers still dies — at the ladder's bound, and
  demonstrably NOT inside the receiver's drain window;
* ``lop refresh`` moves a stale idle runtime and reports a stale busy one as a
  move queued for the end of its turn, instead of killing anything.

Isolation: the ``headless_tui_env`` fixture redirects the config dir and the
root conftest redirects ``HOME``; the child environment is rebuilt from the
cut-off suite's helper, which removes EVERY ``CMUX_*`` / ``LOP_MOBILE_CHILD_*``
/ ``LOP_RUNTIME_*`` variable, because a runtime that inherited a workspace id
could address the operator's live window (#648). Nothing here signals a process
this file did not spawn, and nothing dials an ambient config root.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import control, registry
from local_operator.session.runtime.process import SIGNAL_DRAIN_S
from tests.e2e.test_cut_off_turns_e2e import (
    _attach,
    _child_env,
    _incidents,
    _park_a_turn,
    _reap,
    _seed,
    _successor_boot,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: The mock provider's ``[bash:N]`` marker caps N at 60 s (``_mock_bash_sleep``),
#: so this is the longest turn the harness can park — and it is also why the
#: drain bound has to be minutes rather than seconds: a real turn runs longer
#: than anything the mock can hold, and the bound exists to cover the tail of
#: one, not to fit this test.
PARK_S = 60

#: The reply the mock streams AFTER a tool result — the observable that says the
#: turn reached its end instead of being cut off mid-tool. It is the same string
#: ``test_cut_off_turns_e2e`` asserts the ABSENCE of when a runtime dies, so the
#: two suites read the same evidence from opposite directions.
TURN_COMPLETED = "from the mock provider"


def _spawn(config_dir: Path, session_id: str, **extra: str) -> subprocess.Popen[bytes]:
    """The cut-off suite's spawn, with per-cell environment additions.

    Reusing ``_child_env`` wholesale is deliberate: that helper is where the
    three inherited variable families are stripped and where
    ``LOP_SESSION_GRACE_S`` is raised out of the way, and a second copy here
    would be a second place to forget one of them.
    """
    env = _child_env(config_dir, session_id)
    env.update(extra)
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _runtime_log(config_dir: Path) -> str:
    """The runtimes' own log for this redirected config root.

    Every child of one test shares it, so a cell that reads it takes a byte
    OFFSET before acting and looks only at what was appended after.
    """
    path = config_dir / "logs" / "runtime.log"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _record_for(config_dir: Path, session_id: str) -> Any:
    for record, _state in registry.scan(config_dir):
        if getattr(record, "session_id", "") == session_id:
            return record
    return None


async def _wait_record(
    config_dir: Path, session_id: str, *, busy: bool = False, timeout: float = 45.0
) -> Any:
    """Wait for the published record, optionally for it to report a live turn.

    The record is the product's own statement about itself, and for the cells
    that turn on "is a turn in flight" it is the right thing to wait on: the
    ladder reads exactly this field, so waiting on something else would test a
    different question from the one the code asks.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        record = _record_for(config_dir, session_id)
        if record is not None and (record.busy or not busy):
            return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no {'busy ' if busy else ''}record for {session_id} within {timeout}s")


def _transcript(directory: Path) -> str:
    return (directory / "transcript.jsonl").read_text(encoding="utf-8", errors="replace")


def _cut_off_causes(directory: Path) -> list[str]:
    """Every cut-off cause this session's transcript carries, if any.

    Read through the product's own incident rows rather than a string scan, so
    the assertion is about what a reader of this session would be told.
    """
    causes: list[str] = []
    for entry in _incidents(directory):
        details = entry.payload.get("details") or {}
        if isinstance(details, dict) and details.get("cause"):
            causes.append(str(details["cause"]))
    return causes


class _Rig:
    """One cell's children and viewers, disposed together however it ends."""

    def __init__(self, config: Path) -> None:
        self.config = config
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self.viewers: dict[str, Any] = {}
        self.directories: dict[str, Path] = {}

    def seed_and_spawn(self, session_id: str, **extra: str) -> Path:
        self.directories[session_id] = _seed(self.config, session_id)
        self.children[session_id] = _spawn(self.config, session_id, **extra)
        return self.directories[session_id]

    async def park(self, session_id: str, seconds: int = PARK_S) -> Any:
        """Attach the production viewer and park a real turn in the real tool."""
        viewer = await _attach(self.config, session_id)
        self.viewers[session_id] = viewer
        await _park_a_turn(viewer, self.directories[session_id], seconds)
        return viewer

    async def aclose(self) -> None:
        for viewer in self.viewers.values():
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a process we may have frozen
                pass
        for child in self.children.values():
            if child.poll() is None:
                # SIGCONT first: a cell that ends mid-assertion may have left a
                # child stopped, and a stopped child ignores the reap.
                try:
                    os.kill(child.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                _reap(child, self.config)
        for record, _state in registry.scan(self.config):
            try:
                os.kill(record.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------------------
# The acceptance cell
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_broadcast_sigterm_mid_turn_lets_every_turn_finish(
    headless_tui_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE ACCEPTANCE CELL: one pass of SIGTERM over three busy runtimes.

    The sweep SHAPE is evidence rather than decoration: the incident delivered
    SIGTERM to every live runtime within ~6 ms, and a sequential ladder cannot
    produce that. Three sessions are parked in a real 60 s tool and signalled in
    one loop; all three must finish the turn they were in the middle of.

    Per session: the process exited 0, it did NOT leave in the fraction of a
    second the old handler took, the transcript carries the mock's follow-up
    reply, no cut-off incident was recorded, and the runtime's own log shows it
    DRAINED — the bound-expiry warning must not appear, which is the assertion
    that distinguishes "left at its boundary" from "left because the bound ran
    out" even on a loaded machine.
    """
    config = headless_tui_env
    session_ids = ("drainsig01", "drainsig02", "drainsig03")
    rig = _Rig(config)
    try:
        with bounded(420, "signal drain: broadcast SIGTERM mid-turn"):
            for session_id in session_ids:
                rig.seed_and_spawn(session_id)
            for session_id in session_ids:
                await rig.park(session_id)
                record = await _wait_record(config, session_id, busy=True)
                assert record.busy is True, f"{session_id}: no turn in flight"
                assert record.pid == rig.children[session_id].pid

            log_before = len(_runtime_log(config))

            # ONE pass, exactly as the sweep arrived.
            started = time.monotonic()
            for session_id in session_ids:
                os.kill(rig.children[session_id].pid, signal.SIGTERM)

            exits: dict[str, tuple[int, float]] = {}
            for session_id in session_ids:
                code = rig.children[session_id].wait(timeout=180)
                exits[session_id] = (int(code), round(time.monotonic() - started, 2))

            appended = _runtime_log(config)[log_before:]
            report = {"exit_code_and_latency_s": exits, "log": appended[-4000:]}
            # Printed rather than merely asserted: these numbers ARE the
            # evidence for the incident this file exists for, and a reviewer
            # (or a future investigator) should not have to re-run the cell to
            # read the exit latency of each of the three broadcasts.
            with capsys.disabled():
                print(f"\n=== broadcast SIGTERM mid-turn (parked {PARK_S}s) ===")
                print(json.dumps(report["exit_code_and_latency_s"], indent=2))
                for session_id in session_ids:
                    directory = rig.directories[session_id]
                    print(
                        f"{session_id}: "
                        f"completed={TURN_COMPLETED in _transcript(directory)} "
                        f"cut_offs={_cut_off_causes(directory)}"
                    )
            # 1. Every receiver took the SIGNAL path and left cleanly.
            assert all(code == 0 for code, _latency in exits.values()), report
            # 2. NOBODY was killed on arrival. The old handler was gone in
            #    ~0.65 s; a turn parked for PARK_S cannot be, and this is the
            #    single assertion the whole incident turns on.
            assert all(latency > PARK_S * 0.6 for _code, latency in exits.values()), report
            # 3. The drain is what ended the wait, not the bound: the warning
            #    fires exactly once per expiry and never on a boundary exit.
            assert "drain bound" not in appended, report
            assert appended.count("arrived with work in flight") == len(session_ids), report

            for session_id in session_ids:
                directory = rig.directories[session_id]
                text = _transcript(directory)
                # 4. The turn COMPLETED: the mock answers text after a tool
                #    result, and that reply is only ever written by a turn that
                #    reached its end.
                assert TURN_COMPLETED in text, f"{session_id}: the turn never finished"
                # 5. And nothing was classified as cut off — the durable record
                #    the incident's 32 turns carried.
                assert _cut_off_causes(directory) == [], f"{session_id}: a cut-off was recorded"

                session = await _successor_boot(directory)
                try:
                    from local_operator.session.attention import (
                        AttentionStore,
                        conversation_identity,
                    )

                    state = AttentionStore().state(conversation_identity(directory))
                    assert state.get("cause") != "runtime-shutdown", (session_id, state)
                finally:
                    await session.dispose()
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# The fast branch: nothing in flight behaves exactly as it always did
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sigterm_with_nothing_in_flight_is_still_immediate(
    headless_tui_env: Path,
) -> None:
    """No added latency for the case that was never broken.

    A signal arriving at an idle runtime must not start a drain, wait for a
    tick, or add a frame of latency: the work predicate is read, it says no,
    and the event is set in the same synchronous step. Measured on the clock
    because that is the property (a drain would still eventually exit 0).
    """
    config = headless_tui_env
    rig = _Rig(config)
    try:
        with bounded(90, "signal drain: idle SIGTERM"):
            rig.seed_and_spawn("drainsigidle")
            await _wait_record(config, "drainsigidle")
            log_before = len(_runtime_log(config))

            started = time.monotonic()
            os.kill(rig.children["drainsigidle"].pid, signal.SIGTERM)
            code = rig.children["drainsigidle"].wait(timeout=30)
            latency = time.monotonic() - started

            appended = _runtime_log(config)[log_before:]
            assert code == 0, appended[-2000:]
            assert latency < 10.0, f"an idle runtime took {latency:.2f}s to leave"
            assert "exiting (SIGTERM" in appended, appended[-2000:]
            assert "arrived with work in flight" not in appended
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# Our own senders: a deliberate stop is prompt, and says so
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_all_stops_the_idle_and_the_busy_session_promptly(
    headless_tui_env: Path,
) -> None:
    """A DELIBERATE stop still ends a busy session, immediately and by its socket.

    This is the behaviour the ladder exists for, and the change must not cost
    it: ``lop stop`` on a session mid-turn ends that session now. What the
    receiver's drain protects is a signal from OUTSIDE (an unnamed sweep), not
    the operator's own kill switch — the socket rung is the deliberate path and
    it stays prompt.

    Both sessions are in the same ``--all`` run so the receipts are compared: the
    idle one and the busy one are both ``socket``, and NO signal is sent for
    either. If the ladder had escalated, the runtime log would carry the
    signal-path exit line.
    """
    config = headless_tui_env
    rig = _Rig(config)
    try:
        with bounded(180, "signal drain: deliberate stop --all"):
            rig.seed_and_spawn("drainidle1")
            idle_directory = rig.directories["drainidle1"]
            rig.seed_and_spawn("drainbusy1")
            await _wait_record(config, "drainidle1")
            await rig.park("drainbusy1")
            await _wait_record(config, "drainbusy1", busy=True)
            log_before = len(_runtime_log(config))

            started = time.monotonic()
            outcomes = await control.stop_all(timeout_s=10.0, own_pid=None, _root=config)
            elapsed = time.monotonic() - started

            by_session = {outcome.session_id: outcome for outcome in outcomes}
            report = {"outcomes": [(o.session_id, o.method, o.line) for o in outcomes]}
            assert set(by_session) == {"drainidle1", "drainbusy1"}, report
            assert by_session["drainidle1"].method == "socket", report
            assert by_session["drainbusy1"].method == "socket", report
            # Prompt: the graceful op acks and the process leaves; nothing here
            # may wait on a drain bound.
            assert elapsed < 30.0, f"a deliberate stop took {elapsed:.1f}s: {report}"
            appended = _runtime_log(config)[log_before:]
            assert "exiting (SIGTERM" not in appended, "the deliberate path must not signal"

            # The parked turn was ended deliberately, so it must NOT read as an
            # involuntary cut-off — the one misclassification the taxonomy calls
            # worse than the bug it fixes.
            causes = _cut_off_causes(rig.directories["drainbusy1"])
            assert causes == [], causes
            assert TURN_COMPLETED not in _transcript(rig.directories["drainbusy1"])
            assert idle_directory.exists()
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# A frozen runtime: skipped while it is mid-turn, killed if it never answers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_frozen_runtime_mid_turn_is_reported_and_never_signalled(
    headless_tui_env: Path,
) -> None:
    """The skip, on the shape it exists for: alive, silent, mid-turn.

    A SIGSTOPped runtime cannot answer its socket, so rung 1 gets no ack and the
    ladder reaches its signal rungs — which is where the new work check runs. The
    record says a turn is in flight, so the ladder declines to signal and reports
    the skip instead. Asserted: the pid is untouched, the record is still
    published, and the runtime log carries no signal-path exit line for it.

    Then SIGCONT, and the one thing this cell deliberately does NOT claim: the
    frozen runtime stops anyway, because rung 1 had already ASKED it to (its stop
    frame was sitting in the socket buffer) and a deliberate stop request stays
    valid. What that half pins is the classification — the turn is ended as the
    user's own stop, never as an involuntary ``runtime-shutdown`` — because a
    skip that later fired a signal would have been recorded as a cut-off, and
    mislabelling a requested stop as an error is the mistake this taxonomy calls
    worse than the bug it fixes.
    """
    config = headless_tui_env
    rig = _Rig(config)
    try:
        with bounded(300, "signal drain: frozen runtime mid-turn"):
            rig.seed_and_spawn("drainfrozen")
            directory = rig.directories["drainfrozen"]
            await rig.park("drainfrozen")
            record = await _wait_record(config, "drainfrozen", busy=True)
            child = rig.children["drainfrozen"]
            log_before = len(_runtime_log(config))

            os.kill(child.pid, signal.SIGSTOP)
            try:
                outcome = await control.stop_session(
                    record, timeout_s=2.0, force=False, _root=config
                )
                assert outcome.method == "busy", outcome.line
                assert "turn is in flight" in outcome.line
                assert "--force" in outcome.line
                # Nothing was signalled, and nothing was resolved: the target is
                # exactly as it was. ``poll()`` rather than ``os.kill(pid, 0)``:
                # a child this test spawned stays a ZOMBIE after it exits, and a
                # zombie answers signal 0 forever — the probe would say "alive"
                # for a process that had long since gone.
                assert child.poll() is None, "a skipped target is left untouched"
                assert _record_for(config, "drainfrozen") is not None
                assert child.poll() is None
                appended = _runtime_log(config)[log_before:]
                assert "exiting (SIGTERM" not in appended, appended[-2000:]
            finally:
                os.kill(child.pid, signal.SIGCONT)

            # The request it was already given lands once it can run, and it is
            # classified as the deliberate stop it is.
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and child.poll() is None:
                await asyncio.sleep(0.25)
            assert child.poll() is not None, "the deliberate request never landed"
            assert child.returncode == 0
            causes = _cut_off_causes(directory)
            assert causes == [], causes
            appended = _runtime_log(config)[log_before:]
            assert "exiting (SIGTERM" not in appended, "no signal may be sent for a skip"
            assert "exiting (socket-stop" in appended, appended[-2000:]
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_frozen_runtime_that_never_answers_still_dies_at_the_bound(
    headless_tui_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wedged runtime must not become unkillable — and the bound is honoured.

    Two facts in one cell, because they are two halves of the same invariant:

    * At ``SIGNAL_DRAIN_S`` after the forced stop began, the process is STILL
      ALIVE. That is the observable form of the ladder's escalation being longer
      than the receiver's drain window: an escalating ladder cannot SIGKILL a
      runtime that is finishing a turn, because it does not fire inside that
      window at all.
    * It is gone by the grace's end, reported as ``sigkill``. A process that
      never answers still dies; the fix buys patience, not immortality.

    ``--force`` is required for this shape and is not a shortcut: a record whose
    heartbeat is still fresh cannot be identity-proven while its socket is
    silent (the ladder refuses, and that refusal is asserted first), so the
    operator's explicit opt-in is what admits the signal.
    """
    config = headless_tui_env
    rig = _Rig(config)
    try:
        with bounded(420, "signal drain: wedged runtime dies at the bound"):
            # An IDLE frozen runtime: a busy one is skipped by design (the cell
            # above), and what this cell is about is a target that never answers.
            rig.seed_and_spawn("drainwedged")
            record = await _wait_record(config, "drainwedged")
            child = rig.children["drainwedged"]

            os.kill(child.pid, signal.SIGSTOP)
            try:
                refused = await control.stop_session(
                    record, timeout_s=2.0, force=False, _root=config
                )
                assert refused.method == "refused", refused.line
                assert "not answering its socket" in refused.line

                started = time.monotonic()
                task = asyncio.ensure_future(
                    control.stop_session(record, timeout_s=2.0, force=True, _root=config)
                )
                # The escalation must not land inside the receiver's drain
                # window. Sampled slightly after it, so the assertion cannot be
                # satisfied by a ladder that fired a moment too early.
                await asyncio.sleep(SIGNAL_DRAIN_S + 5.0)
                assert (
                    child.poll() is None
                ), "the ladder escalated inside the receiver's drain bound"

                outcome = await asyncio.wait_for(task, timeout=120)
                elapsed = time.monotonic() - started
            finally:
                os.kill(child.pid, signal.SIGCONT)

            with capsys.disabled():
                print(
                    "\n=== wedged runtime: the ladder's bound ==="
                    f"\nSIGNAL_DRAIN_S={SIGNAL_DRAIN_S}s SIGTERM_GRACE_S={control.SIGTERM_GRACE_S}s"
                    f"\nelapsed to SIGKILL={elapsed:.1f}s method={outcome.method}"
                    f"\nlive at the drain bound: {SIGNAL_DRAIN_S}s"
                )
            assert outcome.method == "sigkill", outcome.line
            # ``poll()`` reaps and answers; it is what turns "the pid is gone"
            # into a fact rather than a zombie the OS is still holding.
            assert child.poll() is not None, "a wedged runtime must still die"
            assert elapsed >= SIGNAL_DRAIN_S, f"killed after {elapsed:.1f}s: {outcome.line}"
            assert elapsed <= control.SIGTERM_GRACE_S + 30.0, outcome.line
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# The rotation path: a new build at the next boundary, not a kill
# ---------------------------------------------------------------------------


def _install_marker(prefix: Path, ref: str, version: str) -> None:
    """One ``.lop-source`` marker, the shape ``lop-update`` writes last."""
    (prefix / ".lop-source").write_text(f"{ref} {version}\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_refresh_moves_the_idle_stale_runtime_and_queues_the_busy_one(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lop refresh``: what moved, and what is still busy — end to end.

    Both runtimes boot on OLD; the marker is flipped to NEW while they are
    running (what ``lop-update`` does last). The idle one is asked and leaves at
    once; the busy one is asked, answers ``busy``, and keeps working — its turn
    completes, and it retires by itself afterwards because the reaper's own
    refresh branch fires the moment it is idle. Nothing is signalled at any
    point, which is asserted from the runtime log (the signal path writes its
    own exit line).

    ``LOP_BUILD_STAGGER_S`` is raised so the idle runtime's OWN reaper cannot
    retire it inside the assertion window before the ask lands: this cell is
    about the command's path, and the reaper's has its own e2e file
    (``test_runtime_refresh_e2e``).
    """
    config = headless_tui_env
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    _install_marker(prefix, "46a4e9b1234567", "v0.54.46")
    env = {
        "LOP_BUILD_PREFIX": str(prefix),
        "LOP_BUILD_SETTLE_S": "0.5",
        "LOP_BUILD_STAGGER_S": "30",
    }
    rig = _Rig(config)
    try:
        with bounded(300, "signal drain: refresh reports moved and busy"):
            rig.seed_and_spawn("drainrotidle", **env)
            rig.seed_and_spawn("drainrotbusy", **env)
            await _wait_record(config, "drainrotidle")
            await rig.park("drainrotbusy")
            await _wait_record(config, "drainrotbusy", busy=True)

            log_before = len(_runtime_log(config))
            # The flip: this is what "the build on disk has moved" means.
            _install_marker(prefix, "f4a70b991234567", "v0.54.49")
            await asyncio.sleep(1.0)  # past BUILD_SETTLE_S: the move is whole

            outcomes = await control.refresh_all(timeout_s=10.0, own_pid=None, _root=config)
            by_session = {outcome.session_id: outcome for outcome in outcomes}
            report = {"outcomes": [(o.session_id, o.method, o.line) for o in outcomes]}
            assert by_session["drainrotidle"].method == "moved", report
            assert by_session["drainrotbusy"].method == "busy", report
            assert "turn in flight" in by_session["drainrotbusy"].line
            # Every method here is settled for the caller: one is leaving, the
            # other will leave by itself. Neither is the partial case.
            assert all(
                outcome.method in control.REFRESH_SETTLED_METHODS for outcome in outcomes
            ), report
            summary = control.summarize_refresh(outcomes)
            assert "1 retiring now" in summary and "1 will move when their turn ends" in summary

            # The idle one really left, at its own boundary, without a signal.
            idle = rig.children["drainrotidle"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and idle.poll() is None:
                await asyncio.sleep(0.1)
            assert idle.poll() is not None, "the idle stale runtime was asked and did not leave"
            assert idle.returncode == 0

            # The busy one kept working: its turn completed and it was never
            # signalled.
            busy_directory = rig.directories["drainrotbusy"]
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if TURN_COMPLETED in _transcript(busy_directory):
                    break
                await asyncio.sleep(0.25)
            assert TURN_COMPLETED in _transcript(
                busy_directory
            ), "a queued move must not cost the turn that queued it"
            assert _cut_off_causes(busy_directory) == []
            appended = _runtime_log(config)[log_before:]
            assert "exiting (SIGTERM" not in appended, appended[-2000:]
            assert "retiring" in appended, "the rotation announces; it does not just exit"

            # And the busy one retires on its own once it is idle — the queued
            # move lands (its own stagger is up to 30 s).
            busy = rig.children["drainrotbusy"]
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and busy.poll() is None:
                await asyncio.sleep(0.25)
            assert busy.poll() is not None, "a queued move never landed"
            assert busy.returncode == 0
    finally:
        await rig.aclose()
