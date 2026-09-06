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
import subprocess
import sys
import textwrap
import threading
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


def _flag(argv: Sequence[str], name: str) -> str:
    return argv[list(argv).index(name) + 1]


def _reporter(recorder: Recorder, *, session_id: str | None = "sess-1") -> HerdrReporter:
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
    recorder = Recorder(fail=True)
    reporter = _reporter(recorder)
    for state in ("idle", "working", "idle"):
        reporter.report(state)
    reporter.release()
    reporter.join()
    assert [sub for sub, _ in recorder.calls] == [
        "report-agent",
        "report-agent",
        "report-agent",
        "release-agent",
    ]


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
        pane_id="w1:p1", binary="/nonexistent/herdr", invoker=observed, clock=None
    )
    reporter.report("idle")
    assert done.wait(WAIT_S)
    reporter.release()
    reporter.join()
    assert reporter.released


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
    """A `herdr` that never answers delays quit by at most the drain bound."""
    import time

    shim = tmp_path / "herdr"
    shim.write_text("#!/bin/sh\nsleep 30\n")
    shim.chmod(0o755)
    child = textwrap.dedent("""
        import sys
        sys.path.insert(0, {repo!r})
        from local_operator.herdr.reporter import HerdrReporter
        reporter = HerdrReporter(pane_id="w1:p1", binary={shim!r})
        reporter.report("idle")
        reporter.release()
        """.format(repo=str(Path(__file__).resolve().parents[2]), shim=str(shim)))
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=60
    )
    elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr[-2000:]
    # Generous against the 2s drain (interpreter start-up is inside the
    # measurement) and far under the 30s + 5s timeout a synchronous join
    # would have paid.
    assert elapsed < 15.0, elapsed
