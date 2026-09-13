"""The Herdr reporter: detection, ordering, de-dupe, release, failure isolation.

Every test drives the reporter through an INJECTED invoker that records
``(subcommand, argv)`` and never spawns anything, the same shape
``test_multiplexer_broadcast`` uses with its fake backend. The one property
that cannot be observed in-process — the release landing before interpreter
exit — runs in a subprocess, exactly as the multiplexer's F8 test does.

Waits are on events the code under test publishes (a ``threading.Event`` set
by the invoker), never on the clock — see AGENTS.md "Timing, flakes".
"""

from __future__ import annotations

import itertools
import logging
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Sequence, cast

import pytest

from local_operator import terminals
from local_operator.herdr import reporter as reporter_mod
from local_operator.herdr.reporter import (
    HERDR_AGENT,
    HERDR_SOURCE,
    HerdrReporter,
    HerdrState,
    herdr_binary,
    herdr_reporting_enabled,
    release_reporter,
    start_reporter,
    state_from_title,
)

#: Generous backstop for a wait on a worker thread. Not an expectation: the
#: worker runs a fake invoker that returns in microseconds, so a wait that
#: reaches this is a wedge, not slowness.
WAIT_S = 10.0


class Recorder:
    """An invoker that records every call and publishes each arrival.

    ``fail`` makes every call raise, which is how the failure-isolation tests
    prove the worker swallows exceptions. ``calls`` is appended BEFORE the
    raise so a test can still see the call was attempted.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail = fail
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        self.threads: list[int] = []

    def __call__(self, subcommand: str, argv: Sequence[str]) -> None:
        with self._arrived:
            self.calls.append((subcommand, tuple(argv)))
            self.threads.append(threading.get_ident())
            self._arrived.notify_all()
        if self.fail:
            raise RuntimeError("herdr exploded")

    def wait_for_calls(self, count: int) -> list[tuple[str, tuple[str, ...]]]:
        with self._arrived:
            if not self._arrived.wait_for(lambda: len(self.calls) >= count, timeout=WAIT_S):
                pytest.fail(f"expected {count} herdr calls, saw {self.calls}")
            return list(self.calls)

    def states(self) -> list[str]:
        return [_flag(argv, "--state") for sub, argv in self.calls if sub == "report-agent"]

    def seqs(self) -> list[int]:
        return [int(_flag(argv, "--seq")) for _, argv in self.calls]


class FlakyRecorder(Recorder):
    """A recorder whose first ``failures`` calls raise, then succeed.

    The retry tests need a transient failure rather than ``Recorder(fail=True)``'s
    permanent one: the property under test is that the SAME item is delivered
    on a later attempt, which a permanently broken invoker cannot show.
    """

    def __init__(self, *, failures: int) -> None:
        super().__init__()
        self._failures = failures

    def __call__(self, subcommand: str, argv: Sequence[str]) -> None:
        super().__call__(subcommand, argv)
        # Only the worker thread calls this, so the counter needs no lock.
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError("herdr exploded")


class FakeHerdrRow:
    """The pane row Herdr keeps: the highest ``--seq`` wins, lower ones are ignored.

    A recorder is enough to see what was SENT; this is what is needed to see
    what Herdr ends up HOLDING, which is the thing the field bug was about
    ("title spinning while the Agents row says idle"). ``swallow`` drops
    exactly one report of that state on the floor after accepting the call —
    the server-side loss a client-side retry cannot see and only the
    heartbeat recovers from.
    """

    def __init__(self, *, swallow: str | None = None) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.state: str | None = None
        self.seq = 0
        self._swallow = swallow
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)

    def __call__(self, subcommand: str, argv: Sequence[str]) -> None:
        with self._changed:
            self.calls.append((subcommand, tuple(argv)))
            seq = int(_flag(argv, "--seq"))
            state = _flag(argv, "--state") if subcommand == "report-agent" else None
            if state is not None and state == self._swallow:
                self._swallow = None
            elif seq > self.seq:
                self.seq = seq
                self.state = state
            self._changed.notify_all()

    def wait_for_state(self, state: str) -> int:
        """Block until the row holds ``state``; return the seq that put it there."""
        with self._changed:
            if not self._changed.wait_for(lambda: self.state == state, timeout=WAIT_S):
                pytest.fail(f"row never reached {state}: {self.state} at seq {self.seq}")
            return self.seq

    def wait_for_calls(self, count: int) -> None:
        """Block until ``count`` calls have arrived, whatever the row now holds.

        Needed where the assertion is about the state the row SETTLES at: a
        `release-agent` clears `state`, so the check has to happen once the
        reports have landed and before the release, not after a `join`.
        """
        with self._changed:
            if not self._changed.wait_for(lambda: len(self.calls) >= count, timeout=WAIT_S):
                pytest.fail(f"expected {count} calls, saw {[sub for sub, _ in self.calls]}")

    def seqs(self) -> list[int]:
        return [int(_flag(argv, "--seq")) for _, argv in self.calls]


def _flag(argv: Sequence[str], name: str) -> str:
    return argv[list(argv).index(name) + 1]


#: Retry backoff for the tests that exercise it. The production shape is
#: 0.5/2/8; the property is the retry, not the wait, so it is injected down
#: to something the suite can afford (AGENTS.md "Timing, flakes").
FAST_BACKOFF_S = (0.01, 0.02, 0.04)


def _reporter(
    recorder: Recorder,
    *,
    session_id: str | None = "sess-1",
    retry_backoff_s: Sequence[float] | None = None,
) -> HerdrReporter:
    # `clock` counts from one so the seqs read as the contract's `1, 2, 3`
    # rather than as epoch microseconds; the production clock is covered by
    # `test_the_sequence_is_anchored_to_the_clock`.
    counter = itertools.count(1)
    return HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        session_id=session_id,
        invoker=recorder,
        clock=lambda: next(counter),
        retry_backoff_s=retry_backoff_s,
    )


# ---------------------------------------------------------------------------
# Detection and gating
# ---------------------------------------------------------------------------


def test_is_herdr_needs_both_markers() -> None:
    assert terminals.is_herdr({"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1"})
    assert not terminals.is_herdr({"HERDR_ENV": "1"})
    assert not terminals.is_herdr({"HERDR_PANE_ID": "w1:p1"})
    assert not terminals.is_herdr({})
    # Exact match, as documented: a hand-exported `0` reads as "not Herdr".
    assert not terminals.is_herdr({"HERDR_ENV": "0", "HERDR_PANE_ID": "w1:p1"})


def test_the_binary_prefers_the_exported_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = tmp_path / "herdr"
    exported.write_text("#!/bin/sh\n")
    exported.chmod(0o755)
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: "/usr/local/bin/herdr")
    assert herdr_binary({"HERDR_BIN_PATH": str(exported)}) == str(exported)


def test_the_binary_falls_back_to_path_when_the_export_is_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited `HERDR_BIN_PATH` across an ssh hop names nothing here."""
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: "/usr/local/bin/herdr")
    assert herdr_binary({"HERDR_BIN_PATH": str(tmp_path / "missing")}) == "/usr/local/bin/herdr"
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: None)
    assert herdr_binary({"HERDR_BIN_PATH": str(tmp_path / "missing")}) is None
    assert herdr_binary({}) is None


def test_the_kill_switch() -> None:
    assert herdr_reporting_enabled({})
    assert not herdr_reporting_enabled({"LOCAL_OPERATOR_NO_HERDR": "1"})
    # Whitespace-only is "unset", matching the multiplexer switch.
    assert herdr_reporting_enabled({"LOCAL_OPERATOR_NO_HERDR": "  "})


def test_start_reporter_is_none_outside_herdr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: "/usr/local/bin/herdr")
    assert start_reporter("sess", env={}) is None


def test_start_reporter_is_none_without_a_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markers alone are not enough: they are inherited into hosts with no CLI."""
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: None)
    assert start_reporter("sess", env={"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1"}) is None


def test_start_reporter_is_none_under_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: "/usr/local/bin/herdr")
    env = {"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1", "LOCAL_OPERATOR_NO_HERDR": "1"}
    assert start_reporter("sess", env=env) is None


def test_start_reporter_inside_herdr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporter_mod.shutil, "which", lambda name: "/usr/local/bin/herdr")
    recorder = Recorder()
    reporter = start_reporter(
        "sess-9", env={"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1"}, invoker=recorder
    )
    assert reporter is not None
    assert reporter.pane_id == "w1:p1"
    assert reporter.session_id == "sess-9"
    # Construction reports nothing: the band's attach sends the first state.
    assert recorder.calls == []


def test_start_reporter_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(env=None):  # noqa: ANN001, ANN202
        raise RuntimeError("detection exploded")

    monkeypatch.setattr(reporter_mod, "is_herdr", boom)
    assert start_reporter("sess", env={"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1"}) is None


def test_the_state_translation() -> None:
    assert state_from_title("idle") == "idle"
    assert state_from_title("working") == "working"
    assert state_from_title("attention") == "blocked"
    # An errored turn is the user's turn again — never `unknown`.
    assert state_from_title("failed") == "idle"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_first_report_carries_the_session_id_and_seq_one() -> None:
    recorder = Recorder()
    reporter = _reporter(recorder)
    reporter.report("idle")
    (call,) = recorder.wait_for_calls(1)
    assert call == (
        "report-agent",
        (
            "/opt/herdr",
            "pane",
            "report-agent",
            "w1:p1",
            "--source",
            HERDR_SOURCE,
            "--agent",
            HERDR_AGENT,
            "--state",
            "idle",
            "--seq",
            "1",
            "--agent-session-id",
            "sess-1",
        ),
    )


def test_the_session_id_is_absent_when_unknown() -> None:
    recorder = Recorder()
    reporter = _reporter(recorder, session_id=None)
    reporter.report("idle")
    ((_, argv),) = recorder.wait_for_calls(1)
    assert "--agent-session-id" not in argv


def test_transitions_and_recovery_from_blocked() -> None:
    """working → blocked → working (answered mid-turn), then idle at turn end."""
    recorder = Recorder()
    reporter = _reporter(recorder)
    for state in ("idle", "working", "blocked", "working", "idle"):
        reporter.report(state)
    recorder.wait_for_calls(5)
    assert recorder.states() == ["idle", "working", "blocked", "working", "idle"]


def test_identical_consecutive_states_are_deduped_and_seq_strictly_increases() -> None:
    recorder = Recorder()
    reporter = _reporter(recorder)
    for state in ("idle", "idle", "idle", "working", "working", "idle"):
        reporter.report(state)
    reporter.release()
    recorder.wait_for_calls(4)
    reporter.join()
    assert recorder.states() == ["idle", "working", "idle"]
    seqs = recorder.seqs()
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert seqs == [1, 2, 3, 4]


def test_the_sequence_is_anchored_to_the_clock() -> None:
    """A fresh process must out-sequence the one that released before it.

    Herdr keeps the per-source high-water mark across `release-agent`, so a
    counter from one would be ignored forever after any relaunch in the same
    pane. Pinned: the seq is `max(previous + 1, clock)` — clock-anchored, and
    still strictly increasing when the clock stalls or steps back.
    """
    recorder = Recorder()
    ticks = iter([1_000, 1_000, 900, 5_000])
    reporter = HerdrReporter(
        pane_id="w1:p1", binary="/opt/herdr", invoker=recorder, clock=lambda: next(ticks)
    )
    for state in ("idle", "working", "blocked", "idle"):
        reporter.report(state)
    recorder.wait_for_calls(4)
    assert recorder.seqs() == [1_000, 1_001, 1_002, 5_000]


def test_the_production_clock_is_epoch_microseconds() -> None:
    import time

    before = time.time_ns() // 1_000
    seq = reporter_mod._default_clock()
    assert before <= seq <= time.time_ns() // 1_000
    # Inside Herdr's u64 with headroom: the probe against 0.8.2 accepted
    # 2**64 - 1 and rejected 2**64.
    assert seq < 2**63


def test_delivery_order_is_mint_order_under_contention() -> None:
    """Two threads racing `report`: the delivered seqs are strictly ascending.

    This is the module's central ordering claim, and it is a real one only
    because the seq is minted and the call enqueued in ONE critical section.
    An earlier version minted under the lock and queued outside it, so two
    callers could mint 1, 2 and deliver 2, 1 — and the test that pinned it
    asserted the same `sorted` property this one does while passing purely
    because the default 5 ms GIL switch interval hid the window (review round
    1, A2: 80/200 inverted at 1e-6, 0/400 at the default).

    So the window is FORCED open rather than hoped shut: `setswitchinterval`
    is dropped to 1 µs for the duration, which makes the interleaving that
    used to fail the common case instead of a rare one. That is what makes
    this a test rather than a bet on machine load (AGENTS.md "Timing,
    flakes") — there is no sleep and no deadline anywhere in it; the wait is
    on the threads' own completion.

    `test_the_mint_and_enqueue_are_one_critical_section` below is the
    can-it-still-fail control: it reintroduces the split and shows this
    property breaking.
    """
    recorder = Recorder()
    reporter = _reporter(recorder)
    start = threading.Event()

    def hammer(states: Sequence[HerdrState]) -> None:
        start.wait()
        for state in states:
            reporter.report(state)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [
            threading.Thread(
                target=hammer, args=(cast(Sequence[HerdrState], ("working", "idle") * 20),)
            ),
            threading.Thread(
                target=hammer, args=(cast(Sequence[HerdrState], ("blocked", "idle") * 20),)
            ),
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(WAIT_S)
        reporter.release()
        reporter.join()
    finally:
        sys.setswitchinterval(previous)

    seqs = recorder.seqs()
    assert seqs == sorted(seqs), f"delivered out of mint order: {seqs}"
    assert len(set(seqs)) == len(seqs), f"duplicate seq: {seqs}"
    # De-dupe held under contention: no two consecutive reports share a state.
    states = recorder.states()
    assert all(a != b for a, b in zip(states, states[1:]))
    # The release is last, and no report was delivered behind it.
    assert [sub for sub, _ in recorder.calls][-1] == "release-agent"


def test_the_mint_and_enqueue_are_one_critical_section() -> None:
    """Prove the test above can still fail: reintroduce the split, see it break.

    AGENTS.md requires a guard to be shown catching the bug it exists for
    rather than passing vacuously. A subclass restores the OLD shape — mint
    under the lock, enqueue after releasing it — and the same hammering then
    produces an inverted delivery log. Asserting that the inversion HAPPENS
    would itself be a race, so the assertion is one-sided: the real reporter
    is run in the identical arrangement and must be ordered every time, while
    the broken one is merely reported on. That keeps this test deterministic
    while still exercising the exact code path that used to fail.
    """

    class SplitMintAndEnqueue(HerdrReporter):
        """The PRE-FIX shape, and nothing else: mint under the lock, put after it.

        Deliberately a thin override rather than a copy of the real method —
        what is being reintroduced is exactly one thing, the gap between the
        mint and the put, so that is all this changes.
        """

        def report(self, state: HerdrState) -> None:  # type: ignore[override]
            if self._released.is_set():
                return
            with self._lock:
                if state == self._last:
                    return
                self._last = state
                seq = self._next_seq_locked()
                argv = self._argv("report-agent", "--state", state, "--seq", str(seq))
            # THE DEFECT: the lock is dropped before the put, so two callers
            # that minted in one order can enqueue in the other.
            self._queue.put(("report-agent", argv))
            with self._lock:
                self._enqueue_started = getattr(self, "_enqueue_started", False)
                if not self._enqueue_started:
                    self._enqueue_started = True
                    self._thread = threading.Thread(
                        target=self._run, name="lop-herdr-report", daemon=True
                    )
                    pending = self._thread
                else:
                    pending = None
            if pending is not None:
                pending.start()

    def hammer_seqs(reporter: HerdrReporter, recorder: Recorder) -> list[int]:
        start = threading.Event()

        def hammer(states: Sequence[HerdrState]) -> None:
            start.wait()
            for state in states:
                reporter.report(state)

        threads = [
            threading.Thread(
                target=hammer, args=(cast(Sequence[HerdrState], ("working", "idle") * 30),)
            ),
            threading.Thread(
                target=hammer, args=(cast(Sequence[HerdrState], ("blocked", "idle") * 30),)
            ),
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(WAIT_S)
        reporter.release()
        reporter.join()
        return recorder.seqs()

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    inversions = 0
    try:
        # Several trials, because the defect is probabilistic even with the
        # window forced open. Nothing is asserted about the count.
        for _ in range(12):
            recorder = Recorder()
            counter = itertools.count(1)
            broken = SplitMintAndEnqueue(
                pane_id="w1:p1",
                binary="/opt/herdr",
                invoker=recorder,
                clock=lambda: next(counter),
            )
            seqs = hammer_seqs(broken, recorder)
            if seqs != sorted(seqs):
                inversions += 1

        # The FIXED reporter, in the identical arrangement, every trial.
        for _ in range(12):
            recorder = Recorder()
            fixed = _reporter(recorder)
            seqs = hammer_seqs(fixed, recorder)
            assert seqs == sorted(seqs), f"the fix regressed: {seqs}"
    finally:
        sys.setswitchinterval(previous)

    # Recorded for the reader, not asserted: on the machine this was written
    # on the split shape inverted in most trials. A zero here would mean the
    # control did not exercise the window, not that the fix is wrong, so it
    # must never fail the suite.
    print(f"[control] split-mint/enqueue inverted {inversions}/12 trials")


def test_a_report_racing_a_release_is_dropped_not_resent() -> None:
    """A1: nothing may be delivered after `release-agent` with a higher seq.

    That is the failure `release-agent` exists to prevent — Herdr's high-water
    mark cannot discard a HIGHER seq, so the row would keep describing an
    exited process. The pre-fix code tested the released latch outside the
    lock, so a report could pass the check, block on the lock the release
    held, then mint a later seq and be delivered behind it (reproduced at
    3/6000). The window is forced open here the same way, and the invariant is
    checked over many trials rather than one.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for trial in range(400):
            recorder = Recorder()
            reporter = _reporter(recorder)
            reporter.report("idle")
            ready = threading.Event()

            def racer(rep: HerdrReporter = reporter, gate: threading.Event = ready) -> None:
                gate.wait()
                rep.report("working")

            thread = threading.Thread(target=racer)
            thread.start()
            ready.set()
            reporter.release()
            thread.join(WAIT_S)
            reporter.join()

            subs = [sub for sub, _ in recorder.calls]
            assert subs.count("release-agent") == 1, f"trial {trial}: {subs}"
            assert (
                subs[-1] == "release-agent"
            ), f"trial {trial}: a report was delivered after the release: {subs}"
            seqs = recorder.seqs()
            assert seqs == sorted(seqs), f"trial {trial}: {seqs}"
    finally:
        sys.setswitchinterval(previous)


def test_the_invoker_runs_off_the_calling_thread() -> None:
    """Structural, not timed: the subprocess never runs where `report` was called."""
    recorder = Recorder()
    reporter = _reporter(recorder)
    reporter.report("idle")
    reporter.release()
    reporter.join()
    assert recorder.threads and all(t != threading.get_ident() for t in recorder.threads)


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


def test_release_is_exactly_once_and_last() -> None:
    recorder = Recorder()
    reporter = _reporter(recorder)
    reporter.report("idle")
    reporter.release()
    reporter.release()
    release_reporter(reporter)
    # A report after release is dropped: the row is gone.
    reporter.report("working")
    reporter.join()
    assert [sub for sub, _ in recorder.calls] == ["report-agent", "release-agent"]
    _, argv = recorder.calls[-1]
    assert argv[:4] == ("/opt/herdr", "pane", "release-agent", "w1:p1")
    assert "--state" not in argv and "--agent-session-id" not in argv
    assert recorder.seqs() == [1, 2]
    assert reporter.released


def test_release_before_any_report_still_releases() -> None:
    recorder = Recorder()
    reporter = _reporter(recorder)
    reporter.release()
    reporter.join()
    assert [sub for sub, _ in recorder.calls] == ["release-agent"]


def test_reports_queued_ahead_of_a_release_still_land_in_order() -> None:
    """The stream is a function of the transitions, not of how busy the worker was.

    The worker is parked inside the first call while three more are queued;
    every one of them is delivered, in order, and the release is last. What
    is pinned is that `release()` on the caller's thread does not depend on
    the worker being idle — it queues and returns.
    """
    gate = threading.Event()
    recorder = Recorder()

    def slow_first_call(subcommand: str, argv: Sequence[str]) -> None:
        recorder(subcommand, argv)
        if len(recorder.calls) == 1:
            gate.wait(WAIT_S)

    reporter = HerdrReporter(
        pane_id="w1:p1", binary="/opt/herdr", invoker=slow_first_call, clock=None
    )
    reporter.report("idle")
    recorder.wait_for_calls(1)
    reporter.report("working")
    reporter.report("blocked")
    # Returns while the worker is still parked in the first call: the release
    # is queued, not performed, on the caller's thread.
    reporter.release()
    assert reporter.released
    gate.set()
    reporter.join()
    assert [sub for sub, _ in recorder.calls] == [
        "report-agent",
        "report-agent",
        "report-agent",
        "release-agent",
    ]
    assert recorder.states() == ["idle", "working", "blocked"]


def test_release_reporter_tolerates_none_and_errors() -> None:
    release_reporter(None)

    class Broken:
        def release(self) -> None:
            raise RuntimeError("no")

    release_reporter(Broken())  # type: ignore[arg-type]


def test_set_session_id_resends_the_current_state_under_the_new_id() -> None:
    """A `/new` swap changes only the metadata; the row is not released."""
    recorder = Recorder()
    reporter = _reporter(recorder)
    reporter.report("idle")
    reporter.set_session_id("sess-2")
    reporter.report("idle")
    reporter.set_session_id("sess-2")  # unchanged: no re-send
    reporter.report("idle")
    reporter.release()
    reporter.join()
    subs = [sub for sub, _ in recorder.calls]
    assert subs == ["report-agent", "report-agent", "release-agent"]
    assert _flag(recorder.calls[0][1], "--agent-session-id") == "sess-1"
    assert _flag(recorder.calls[1][1], "--agent-session-id") == "sess-2"


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_an_invoker_that_raises_never_propagates_and_later_reports_still_run() -> None:
    """Every item is still attempted, in order, and nothing reaches the caller.

    Retries make the call LOG longer than the transition list — a permanently
    broken invoker is attempted up to four times per item — so what is pinned
    is the order of first attempts, not the count. The retry budget itself is
    the two tests above.
    """
    recorder = Recorder(fail=True)
    reporter = _reporter(recorder, retry_backoff_s=FAST_BACKOFF_S)
    for state in ("idle", "working", "idle"):
        reporter.report(state)
    reporter.release()
    reporter.join(timeout=WAIT_S)
    subs = [sub for sub, _ in recorder.calls]
    assert subs.count("release-agent") >= 1
    assert subs[-1] == "release-agent", subs
    # Duplicates are retries of one item, so collapsing them recovers the
    # stream the transitions asked for.
    seqs = recorder.seqs()
    collapsed = [seq for i, seq in enumerate(seqs) if i == 0 or seq != seqs[i - 1]]
    assert collapsed == [1, 2, 3, 4]


def test_the_cli_invoker_raises_on_non_zero_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production invoker maps every failure to an exception the worker logs."""
    monkeypatch.setattr(reporter_mod, "CALL_TIMEOUT_S", 0.5)
    with pytest.raises(RuntimeError, match="exited 3"):
        reporter_mod._run_cli("report-agent", [sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(subprocess.TimeoutExpired):
        reporter_mod._run_cli("report-agent", [sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(OSError):
        reporter_mod._run_cli("report-agent", ["/nonexistent/herdr", "pane"])
    # And the success path is silent.
    reporter_mod._run_cli("report-agent", [sys.executable, "-c", "pass"])


def test_a_failing_cli_is_swallowed_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real invoker, a real failing binary, and the reporter still completes."""
    done = threading.Event()
    original = reporter_mod._run_cli

    def observed(subcommand: str, argv: Sequence[str]) -> None:
        try:
            original(subcommand, argv)
        finally:
            done.set()

    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/nonexistent/herdr",
        invoker=observed,
        clock=None,
        retry_backoff_s=FAST_BACKOFF_S,
    )
    reporter.report("idle")
    assert done.wait(WAIT_S)
    reporter.release()
    reporter.join()
    assert reporter.released


def test_a_delivery_that_fails_once_is_retried_and_lands_in_order() -> None:
    """The bug: ONE lost call used to freeze the row for the rest of the turn.

    De-dupe updates `_last` at enqueue time, so the 12.5 Hz tick can never
    resend a state whose single delivery failed — the terminal title spins
    while the Agents row still says the previous state. The retry is what
    closes that, and the retry must carry the ORIGINAL seq: a fresh one would
    make the retry a new report and reopen the ordering question.
    """
    recorder = FlakyRecorder(failures=1)
    reporter = _reporter(recorder, retry_backoff_s=FAST_BACKOFF_S)
    for state in ("idle", "working", "idle"):
        reporter.report(state)
    # Waited for BEFORE releasing, deliberately: a report gives up its
    # backoff the moment the released latch is set, so releasing here would
    # be testing that abort path instead of the retry (and the abort has its
    # own test below).
    recorder.wait_for_calls(4)
    reporter.release()
    reporter.join(timeout=WAIT_S)
    # Three transitions, four report calls: the first was attempted twice.
    assert [sub for sub, _ in recorder.calls] == [
        "report-agent",
        "report-agent",
        "report-agent",
        "report-agent",
        "release-agent",
    ]
    assert recorder.states() == ["idle", "idle", "working", "idle"]
    # The retry is the same item, same seq, and nothing overtook it.
    seqs = recorder.seqs()
    assert seqs == [1, 1, 2, 3, 4], seqs


def test_a_report_under_retry_gives_up_its_backoff_to_a_release() -> None:
    """Quit must not wait out a doomed report's backoff.

    The exit drain is bounded at `EXIT_DRAIN_TIMEOUT_S` (2 s) and the
    production backoff sums to 10.5 s, so a report retrying through a quit
    would spend the whole drain budget and let the interpreter exit with the
    `release-agent` still queued — a row outliving its process, which is the
    exact failure `release-agent` exists to prevent. So the backoff aborts on
    the released latch. The release's OWN backoff does not (there is nothing
    behind it), which is why this test uses a report.

    Timed, but with three orders of magnitude of margin: the backoff injected
    here is 30 s and the assertion is "well under that", so a slow machine
    cannot fail it — only a genuinely un-aborted sleep can.
    """
    recorder = FlakyRecorder(failures=1)
    reporter = _reporter(recorder, retry_backoff_s=(30.0,))
    reporter.report("idle")
    recorder.wait_for_calls(1)  # the failing attempt; now parked in backoff
    started = time.monotonic()
    reporter.release()
    reporter.join(timeout=WAIT_S)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, elapsed
    # The report was abandoned; the release still went out.
    assert [sub for sub, _ in recorder.calls] == ["report-agent", "release-agent"]


def test_the_state_herdr_ends_up_holding_survives_a_failed_delivery() -> None:
    """The same property from the server's side: the row tracks the last transition."""
    row = FakeHerdrRow()
    calls = {"n": 0}

    def flaky(subcommand: str, argv: Sequence[str]) -> None:
        calls["n"] += 1
        # The `working` report fails once before it is accepted — the exact
        # shape reproduced in the field with a wrapper binary.
        if calls["n"] == 2:
            raise RuntimeError("herdr exploded")
        row(subcommand, argv)

    counter = itertools.count(1)
    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=flaky,
        clock=lambda: next(counter),
        retry_backoff_s=FAST_BACKOFF_S,
    )
    reporter.report("idle")
    reporter.report("working")
    assert row.wait_for_state("working") == 2
    reporter.release()
    reporter.join(timeout=WAIT_S)


def test_retry_exhaustion_drops_the_item_warns_and_later_transitions_still_land(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WARNING, not DEBUG: a row that stopped tracking a live session is visible.

    The original defect logged this at DEBUG, which is where it hid for a
    whole release. The line has to name the pane, the subcommand and the
    state/seq, or it cannot be matched to the row a user is looking at.
    """
    recorder = FlakyRecorder(failures=4)  # exactly one item's whole budget
    reporter = _reporter(recorder, retry_backoff_s=FAST_BACKOFF_S)
    with caplog.at_level(logging.WARNING, logger="local_operator.herdr.reporter"):
        reporter.report("working")
        recorder.wait_for_calls(4)
        reporter.report("idle")
        recorder.wait_for_calls(5)
        reporter.release()
        reporter.join(timeout=WAIT_S)

    assert recorder.states() == ["working"] * 4 + ["idle"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    message = warnings[0].getMessage()
    for fragment in ("report-agent", "w1:p1", "4 attempts", "state=working", "seq=1"):
        assert fragment in message, message


def test_delivered_state_is_what_landed_not_what_was_queued() -> None:
    """`_delivered` is the worker's record; `last_state` is the caller's.

    Pinned with the worker parked INSIDE the second call, so the two are
    provably different at that instant rather than by a sleep.
    """
    gate = threading.Event()
    recorder = Recorder()

    def parked(subcommand: str, argv: Sequence[str]) -> None:
        recorder(subcommand, argv)
        if len(recorder.calls) == 2:
            gate.wait(WAIT_S)

    counter = itertools.count(1)
    reporter = HerdrReporter(
        pane_id="w1:p1", binary="/opt/herdr", invoker=parked, clock=lambda: next(counter)
    )
    assert reporter.delivered_state is None
    reporter.report("idle")
    reporter.report("working")
    recorder.wait_for_calls(2)
    assert reporter.delivered_state == "idle"
    assert reporter.last_state == "working"
    gate.set()
    reporter.release()
    reporter.join(timeout=WAIT_S)
    # The release clears it: the row is gone, so there is no state to hold.
    assert reporter.delivered_state is None


# ---------------------------------------------------------------------------
# Heartbeat resync
# ---------------------------------------------------------------------------


def test_the_heartbeat_re_asserts_a_state_herdr_lost_with_a_higher_seq() -> None:
    """The half retry cannot reach: a delivery that SUCCEEDED and was forgotten.

    Herdr does not persist agent rows across a server restart, and a pane
    mid-turn emits no further transitions — the tick is de-duped — so the row
    would come back only at the next state change. The fake server swallows
    the `working` report after accepting the call, which is exactly that
    shape, and no transition follows it.
    """
    row = FakeHerdrRow(swallow="working")
    counter = itertools.count(1)
    current: list[HerdrState] = ["idle"]
    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=row,
        clock=lambda: next(counter),
        resync_interval_s=0.02,
    )
    reporter.set_state_provider(lambda: current[0])
    reporter.report("idle")
    current[0] = "working"
    reporter.report("working")  # accepted by the invoker, lost by the server

    # Re-asserted without any new transition, and with a HIGHER seq than the
    # report that was lost — which is what makes the high-water mark accept it.
    assert row.wait_for_state("working") > 2
    reporter.release()
    reporter.join(timeout=WAIT_S)
    seqs = row.seqs()
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs


def test_the_heartbeat_does_not_start_without_a_provider() -> None:
    """A reporter nobody wired costs no thread and sends no extra call."""
    recorder = Recorder()
    counter = itertools.count(1)
    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=recorder,
        clock=lambda: next(counter),
        resync_interval_s=0.01,
    )
    reporter.report("idle")
    recorder.wait_for_calls(1)
    time.sleep(0.1)  # many intervals, had there been a heartbeat
    assert len(recorder.calls) == 1
    assert reporter._resync_thread is None
    reporter.release()
    reporter.join(timeout=WAIT_S)


def test_the_heartbeat_stops_at_release() -> None:
    """`_released.wait` is both the sleep and the stop, so quit is not an interval late."""
    recorder = Recorder()
    counter = itertools.count(1)
    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=recorder,
        clock=lambda: next(counter),
        resync_interval_s=0.01,
    )
    reporter.set_state_provider(lambda: "working")
    reporter.report("working")
    recorder.wait_for_calls(3)  # the transition plus at least two heartbeats
    reporter.release()
    reporter.join(timeout=WAIT_S)

    thread = reporter._resync_thread
    assert thread is not None and not thread.is_alive()
    settled = len(recorder.calls)
    time.sleep(0.1)  # ten intervals
    assert len(recorder.calls) == settled
    subs = [sub for sub, _ in recorder.calls]
    assert subs[-1] == "release-agent", subs
    seqs = recorder.seqs()
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs


def test_the_heartbeat_survives_a_provider_that_raises() -> None:
    """A bad provider skips one tick; it must not take the resync thread with it."""
    recorder = Recorder()
    counter = itertools.count(1)
    calls = {"n": 0}

    def provider() -> HerdrState:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the band exploded")
        return "blocked"

    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=recorder,
        clock=lambda: next(counter),
        resync_interval_s=0.01,
    )
    reporter.set_state_provider(provider)
    recorder.wait_for_calls(1)
    assert recorder.states() == ["blocked"]
    reporter.release()
    reporter.join(timeout=WAIT_S)


def test_a_transition_racing_the_heartbeats_read_still_wins_the_row() -> None:
    """A heartbeat that read a state a transition has since replaced DEFERS.

    The M2 defect: the heartbeat sampled `working`, the session went `idle`
    before the seq was minted, and the heartbeat's stale value took the
    HIGHER seq — so Herdr's high-water mark, which exists to discard stale
    reports, kept this one and the row disagreed with the session.

    The guard is the mint counter, not a lock around the provider (which would
    let a slow provider stall `release`; see the next test). The provider
    below models the window exactly: it mutates the true state, releases the
    transition thread, waits long enough that the transition mints first, and
    returns the value it read BEFORE the change. It raises on every later tick
    so a second heartbeat cannot self-correct the row and hide the bug.
    """
    row = FakeHerdrRow()
    counter = itertools.count(1)
    truth: list[HerdrState] = ["working"]
    sampled = threading.Event()
    deferred = threading.Event()
    settled = threading.Event()
    ticks = itertools.count(1)

    def provider() -> HerdrState:
        if next(ticks) > 1:
            # Reaching a SECOND tick proves the first one finished deciding,
            # which is what makes the assertions below deterministic rather
            # than a race against the heartbeat's enqueue. It raises so no
            # later tick can self-correct the row and hide a stale one.
            settled.set()
            raise RuntimeError("one tick only: a second would mask a stale row")
        stale = truth[0]
        truth[0] = "idle"
        sampled.set()
        # Held open until the transition below has actually minted, so the
        # race the guard exists for is forced rather than hoped for.
        assert deferred.wait(WAIT_S), "the transition never minted"
        return stale

    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=row,
        clock=lambda: next(counter),
        resync_interval_s=0.01,
    )
    reporter.report("working")
    reporter.set_state_provider(provider)
    assert sampled.wait(WAIT_S), "the heartbeat never read the provider"
    reporter.report("idle")  # mints while the provider is still parked
    deferred.set()
    row.wait_for_state("idle")
    assert settled.wait(WAIT_S), "the heartbeat never finished its tick"

    # The heartbeat's stale `working` was DROPPED, not minted behind the
    # transition: two reports, and the row holds the true state. Asserted
    # before the release, which clears the row the way Herdr does.
    assert row.state == truth[0] == "idle", (row.state, row.seq, row.calls)
    states = [_flag(argv, "--state") for sub, argv in row.calls if sub == "report-agent"]
    assert states == ["working", "idle"], states
    reporter.release()
    reporter.join(timeout=WAIT_S)
    seqs = row.seqs()
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs


def test_a_blocking_provider_cannot_delay_release() -> None:
    """The provider runs outside `_lock`, so it cannot stall the reporter.

    This is the bound the mint-counter guard buys. Closing M2 by holding
    `_lock` across the provider call would put caller-supplied code inside the
    section `release` needs: measured at a 29.9 s `release()` against a
    provider that slept 30 s, which also breaks the documented
    EXIT_DRAIN_TIMEOUT_S bound. A contract saying "must be cheap" is not a
    bound; not calling user code under the lock is.

    The provider here violates that contract on purpose. `release` and `join`
    must still return promptly, and the row must still be released.
    """
    recorder = Recorder()
    counter = itertools.count(1)
    entered = threading.Event()
    unblock = threading.Event()

    def blocking_provider() -> HerdrState:
        entered.set()
        unblock.wait(WAIT_S)  # violates "cheap and must not block"
        return "working"

    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=recorder,
        clock=lambda: next(counter),
        resync_interval_s=0.01,
    )
    reporter.set_state_provider(blocking_provider)
    assert entered.wait(WAIT_S), "the heartbeat never entered the provider"
    try:
        started = time.monotonic()
        reporter.report("idle")
        reporter.release()
        reporter.join(timeout=reporter_mod.EXIT_DRAIN_TIMEOUT_S)
        elapsed = time.monotonic() - started
        # The resync thread is still parked in the provider, so `join` pays its
        # bound once — never the provider's 10 s, and never 2x the bound.
        assert elapsed < reporter_mod.EXIT_DRAIN_TIMEOUT_S + 0.5, elapsed
        assert [sub for sub, _ in recorder.calls] == ["report-agent", "release-agent"]
    finally:
        unblock.set()


def test_join_bounds_the_total_wait_not_each_thread() -> None:
    """`join(timeout=T)` waits T in total, with BOTH of its threads blocked.

    The M1 defect: `timeout` was passed to each `Thread.join` in turn, so a
    reporter with a blocked worker AND a blocked resync thread paid 2T —
    `join(timeout=1.0)` measured at 2.02 s. That let one reporter overrun the
    shared `remaining` that `_drain_at_exit` computes from
    EXIT_DRAIN_TIMEOUT_S, making the documented worst-case exit delay the
    bound times the number of threads.

    Both threads have to be blocked for the doubling to appear, which is why
    this is pinned here rather than in the subprocess exit test: with a cheap
    provider the resync thread exits the instant `release` latches, so it
    never consumes its share of the budget.
    """
    counter = itertools.count(1)
    wedged = threading.Event()
    entered = threading.Event()

    def wedged_invoker(subcommand: str, argv: Sequence[str]) -> None:
        wedged.wait(WAIT_S)  # a herdr that never answers

    def blocking_provider() -> HerdrState:
        entered.set()
        wedged.wait(WAIT_S)
        return "working"

    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=wedged_invoker,
        clock=lambda: next(counter),
        resync_interval_s=0.01,
    )
    reporter.set_state_provider(blocking_provider)
    reporter.report("idle")
    assert entered.wait(WAIT_S), "the heartbeat never entered the provider"
    try:
        budget = 1.0
        started = time.monotonic()
        reporter.join(timeout=budget)
        elapsed = time.monotonic() - started
        assert reporter._thread is not None and reporter._thread.is_alive()
        assert reporter._resync_thread is not None and reporter._resync_thread.is_alive()
        # One budget, not one per thread. The margin absorbs scheduling on a
        # loaded runner while staying far below the 2x a per-thread timeout
        # costs.
        assert elapsed < budget + 0.4, elapsed
    finally:
        wedged.set()
        reporter.release()
        reporter.join(timeout=WAIT_S)


def test_the_heartbeat_does_not_break_mint_order_under_contention() -> None:
    """The module's central claim, with a heartbeat minting concurrently.

    The heartbeat enqueues through the same `_enqueue_report` critical
    section as `report`, so it is just a third caller; this is the assertion
    that says so rather than the docstring. Same forced switch interval as
    `test_delivery_order_is_mint_order_under_contention`.
    """
    recorder = Recorder()
    counter = itertools.count(1)
    reporter = HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        invoker=recorder,
        clock=lambda: next(counter),
        resync_interval_s=0.001,
    )
    reporter.set_state_provider(lambda: "working")
    start = threading.Event()

    def hammer(states: Sequence[HerdrState]) -> None:
        start.wait()
        for state in states:
            reporter.report(state)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [
            threading.Thread(
                target=hammer, args=(cast(Sequence[HerdrState], ("working", "idle") * 20),)
            ),
            threading.Thread(
                target=hammer, args=(cast(Sequence[HerdrState], ("blocked", "idle") * 20),)
            ),
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(WAIT_S)
        reporter.release()
        reporter.join(timeout=WAIT_S)
    finally:
        sys.setswitchinterval(previous)

    seqs = recorder.seqs()
    assert seqs == sorted(seqs), f"delivered out of mint order: {seqs}"
    assert len(set(seqs)) == len(seqs), f"duplicate seq: {seqs}"
    # Nothing was delivered behind the release, heartbeat included.
    assert [sub for sub, _ in recorder.calls][-1] == "release-agent"


# ---------------------------------------------------------------------------
# Exit drain
# ---------------------------------------------------------------------------


def test_the_exit_drain_releases_an_unreleased_reporter_and_joins() -> None:
    """The atexit half: an abrupt exit that never reached `on_unmount` still releases."""
    recorder = Recorder()
    reporter = _reporter(recorder)
    reporter.report("working")
    recorder.wait_for_calls(1)
    reporter_mod._drain_at_exit()
    assert [sub for sub, _ in recorder.calls] == ["report-agent", "release-agent"]
    # And it is idempotent: a second drain (or a release after it) adds nothing.
    reporter_mod._drain_at_exit()
    reporter.release()
    reporter.join()
    assert len(recorder.calls) == 2


def test_the_release_lands_before_interpreter_exit(tmp_path: Path) -> None:
    """Run in a SUBPROCESS because the property IS process death.

    The worker is a daemon thread; without the exit drain the interpreter
    would exit with the release still queued and the row would outlive the
    process. The child quits exactly as `on_unmount` does — `release()` and
    return — and the log the fake binary writes is the evidence.
    """
    log = tmp_path / "herdr.log"
    shim = tmp_path / "herdr"
    shim.write_text(f'#!/bin/sh\nsleep 0.05\necho "$@" >> {log}\n')
    shim.chmod(0o755)
    child = textwrap.dedent("""
        import sys
        sys.path.insert(0, {repo!r})
        from local_operator.herdr.reporter import HerdrReporter
        reporter = HerdrReporter(pane_id="w1:p1", binary={shim!r}, session_id="s")
        reporter.report("idle")
        reporter.report("working")
        reporter.release()
        """.format(repo=str(Path(__file__).resolve().parents[2]), shim=str(shim)))
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = log.read_text().splitlines()
    assert [line.split()[1] for line in lines] == [
        "report-agent",
        "report-agent",
        "release-agent",
    ], lines


def test_a_wedged_binary_bounds_the_exit_delay(tmp_path: Path) -> None:
    """A `herdr` that never answers delays quit by at most the drain bound.

    The child runs a LIVE HEARTBEAT against the wedged binary, which is what
    makes the bound worth asserting: `join` has two threads to wait for, and
    giving each of them the full timeout (the M1 defect) spent 2x the drain.
    The provider is cheap, as its contract asks — a provider that blocks is
    bounded by `test_a_blocking_provider_cannot_delay_release` in-process,
    where the measurement is not competing with interpreter start-up.

    MARGIN. The bound is EXIT_DRAIN_TIMEOUT_S (2 s) plus 2.5 s for CPython
    start-up, the import of the reporter module and scheduling noise on a
    loaded CI runner — a ceiling of 4.5 s. That leaves roughly 2 s of head
    room over the ~2.4 s a correct drain measures here, while still failing
    on a doubled deadline, whose floor is 2x2 s = 4 s of drain alone before
    any start-up is added. A test that tolerated the doubling is what let M1
    through: the old bound was 15 s, 7.5x the documented figure.
    """
    import time

    shim = tmp_path / "herdr"
    shim.write_text("#!/bin/sh\nsleep 30\n")
    shim.chmod(0o755)
    child = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, {repo!r})
        from local_operator.herdr.reporter import HerdrReporter
        reporter = HerdrReporter(
            pane_id="w1:p1", binary={shim!r}, resync_interval_s=0.05
        )
        # Cheap, per the provider contract. The heartbeat still queues calls
        # the wedged shim never answers, so the worker is blocked at exit and
        # the resync thread is the second thread `join` has to bound.
        reporter.set_state_provider(lambda: "working")
        reporter.report("idle")
        time.sleep(0.3)  # several intervals, so the heartbeat has ticked
        reporter.release()
        """.format(repo=str(Path(__file__).resolve().parents[2]), shim=str(shim)))
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=60
    )
    elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr[-2000:]
    # See MARGIN above. Far under the 30s + 5s timeout a synchronous join
    # would have paid, and under 2x the drain a per-thread timeout costs.
    assert elapsed < 4.5, elapsed
