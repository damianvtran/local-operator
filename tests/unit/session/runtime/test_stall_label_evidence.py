"""A stall/drain label must not outlive the evidence behind it (2026-09-24).

Two labels on the live fleet stated a runtime's state long after the facts that earned
them had changed, and both nearly cost an operator the wrong action:

* ``bound held; lop stop`` stayed on two HEALTHY runtimes (pids 65820 and 83160,
  heartbeat 8-17 s, deadline sibling rewritten every beat) 5 and 14 hours after their
  held fires, because the marker is appended once per fire and nothing ever asked
  whether the runtime re-armed after it. :func:`stall_watchdog._holds` is now the one
  predicate, and it treats a fire as superseded once EVERY plane has reported since.
* a plain ``lop stop`` skipped two WEDGED runtimes (pids 42983 and 43911, heartbeat
  5.6 h and 5.9 h stale) with "it leaves by itself, nothing to do", because their record
  carried the drain latch ``leaving for the build on disk when its turn ends`` and the
  skip never asked whether that turn could still end. ``control._drain_stalled`` gates
  the skip on the same ``wedged`` verdict and held predicate the listing prints.

Each cell was run against the code it fixes (AGENTS.md, "Prove the test can still fail"),
and each docstring says which of two kinds it is: a REPRODUCTION, which fails on the
code before the fix, or a REGRESSION PIN, which passes there too because it asserts
behaviour the fix must keep (a stuck runtime stays held, a fresh drain is still skipped).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.paths import config_dir
from local_operator.session.runtime import control, registry, stall_watchdog
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S, LEAVING_FOR_BUILD
from tests.unit.session.runtime import test_control
from tests.unit.session.runtime.test_control import (
    _bare_record,
    _record_for,
    _serve,
    _StoppingHandle,
)

#: The kill-switch cells' fixture, re-bound here by name so pytest collects it for this
#: module: it keeps the ladder from signalling the test runner and makes liveness follow
#: the fake handle, which every ladder cell below depends on.
no_signals = test_control.no_signals

# ---------------------------------------------------------------------------------------
# Bug 2 — the held predicate
# ---------------------------------------------------------------------------------------

#: One bound on the real fleet's shape: fired and held at T, the steady bound 300 s.
BOUND_S = 300.0


def _held_line(fired_mono: float, *, local: str = "2026-09-24 03:23:34") -> str:
    """A held marker in the writer's format: a person's local time, the machine's stamp."""
    return (
        # LITERALS, not the module's constants, so this cell states the on-disk format
        # independently of the code under test (and runs against a head that predates
        # the constants); the writer/reader round trip below pins that the two agree.
        f"{stall_watchdog.HELD_MARKER}the bound fired at {local} "
        f"(monotonic {fired_mono:.3f}) and did NOT end this runtime. "
        "Work was in flight when the fire was observed.\n"
    )


def _arm_header(pid: int, at: float | None = None) -> str:
    """``arm``'s first line, as origin/main writes it too: ``… armed for <s>s at <epoch> …``."""
    at = time.time() - 1 if at is None else at
    return f"[stall watchdog] pid {pid} armed for {BOUND_S:g}s at {at:.0f} (boot)\n"


def _held_dump(
    logs: Path,
    pid: int,
    *,
    fired_mono: float,
    deadline_mono: float | None,
    leg: str = "serving",
    extra: str = "",
) -> Path:
    """A real held dump and (optionally) its deadline sibling, in the writers' formats.

    Every stamp is on the WRITER'S monotonic clock, which is the clock the predicate
    compares on; the epoch field is written too (a fixed, irrelevant value) so the
    sibling has the shape a reader really meets.
    """
    logs.mkdir(parents=True, exist_ok=True)
    dump = stall_watchdog.dump_path(pid, logs)
    dump.write_text(
        # The arm header every real dump opens with (the life the sibling is fenced on).
        f"{_arm_header(pid)}"
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n  File x, line 1\n"
        + _held_line(fired_mono)
        + extra,
        encoding="utf-8",
    )
    if deadline_mono is not None:
        stall_watchdog.deadline_path(pid, logs).write_text(
            f"1790000000.000 {leg} {deadline_mono:.3f}\n", encoding="utf-8"
        )
    return dump


def test_a_held_fire_the_runtime_recovered_from_is_no_longer_held(tmp_path: Path) -> None:
    """Pid 65820's shape: held at T, and every plane has reported since.

    The sibling carries ``pin()`` — the EARLIEST plane's last sign of life plus the
    bound — so a deadline a full bound past the fire says both planes beat after it.
    On origin/main ``held_fire`` answered True here and ``held_pids`` contained the pid:
    the sticky flag that painted ``bound held; lop stop`` on a healthy runtime.
    """
    logs = tmp_path / "logs"
    _held_dump(logs, 65820, fired_mono=10_000.0, deadline_mono=10_000.0 + 5 * 3600 + BOUND_S)

    assert stall_watchdog.held_fire(65820, logs) is False
    assert 65820 not in stall_watchdog.held_pids(logs)
    assert 65820 not in stall_watchdog.held_dumps(logs)
    # ...and it stays on the record as a FIRE: supersession withdraws the live state,
    # never the evidence that the bound fired.
    assert 65820 in stall_watchdog.fired_pids(logs)


def test_a_held_fire_whose_quiet_plane_never_reported_again_stays_held(tmp_path: Path) -> None:
    """The genuinely-stuck half, and the reason the sibling's MTIME is not the test.

    A runtime whose workload loop is parked keeps re-arming off its healthy serving
    plane, so its sibling is rewritten every 15 s — but its CONTENT stays pinned to the
    quiet plane's deadline, which is the fire. Same answer with no sibling at all, and
    for the progress leg (whose decided deadline nothing clears). This cell passes on
    origin/main too: it pins that the fix did not simply turn the held state off.
    """
    logs = tmp_path / "logs"
    fired = 10_000.0
    # Deadline AT the fire (the quiet plane pinned it), sibling freshly written.
    _held_dump(logs, 11, fired_mono=fired, deadline_mono=fired + 1.0)
    _held_dump(logs, 22, fired_mono=fired, deadline_mono=None)
    _held_dump(logs, 33, fired_mono=fired, deadline_mono=fired - 20_000.0, leg="progress")
    # An OLD-FORMAT sibling (``epoch leg``, no monotonic field) cannot show a recovery.
    _held_dump(logs, 44, fired_mono=fired, deadline_mono=None)
    stall_watchdog.deadline_path(44, logs).write_text("9999999999.000 serving\n", encoding="utf-8")

    for pid in (11, 22, 33, 44):
        assert stall_watchdog.held_fire(pid, logs) is True, pid
    assert stall_watchdog.held_pids(logs) == {11, 22, 33, 44}


def test_a_later_idle_fire_supersedes_an_earlier_held_one(tmp_path: Path) -> None:
    """The work the held fire named has cleared: the most recent fire found none.

    ``OBSERVED_MARKER`` is written only when the fire saw NO work in flight, so a later
    one says the stall the earlier marker described is over — the same fact the
    re-arm test establishes, from the artifact itself. Held on origin/main, which read
    any ``HELD_MARKER`` anywhere in the file.
    """
    logs = tmp_path / "logs"
    fired = 10_000.0
    _held_dump(
        logs,
        44,
        fired_mono=fired,
        deadline_mono=fired + 1.0,
        extra=f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n"
        f"{stall_watchdog.OBSERVED_MARKER}the bound fired at 2026-09-24 04:00:00 ...\n",
    )
    assert stall_watchdog.held_fire(44, logs) is False
    assert 44 not in stall_watchdog.held_pids(logs)


def test_the_last_held_fire_is_the_one_judged(tmp_path: Path) -> None:
    """A runtime that stayed stuck re-fires, and recovery is judged against the LATEST.

    First fire long ago, a second one just now, and a deadline between them: the
    runtime recovered once and then stalled again, so it IS held. Reading the first
    marker would call it recovered.
    """
    logs = tmp_path / "logs"
    first, second = 10_000.0, 17_000.0
    _held_dump(
        logs,
        55,
        fired_mono=first,
        deadline_mono=first + 3000,
        extra=f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n" + _held_line(second),
    )
    assert stall_watchdog.held_fire(55, logs) is True


def test_an_unparseable_held_marker_keeps_the_held_reading(tmp_path: Path) -> None:
    """No stamp, no proof of recovery: the quiet direction is the reading the dump states.

    Two shapes: a bare marker, and the PRE-FIX marker (local time only, no monotonic
    stamp) that every dump already on disk carries. Both keep the held reading however
    far the sibling has moved. Regression pins: held on origin/main too.
    """
    logs = tmp_path / "logs"
    for pid, marker in (
        (66, f"{stall_watchdog.HELD_MARKER}held\n"),
        (
            67,
            f"{stall_watchdog.HELD_MARKER}the bound fired at 2026-09-24 03:23:34 and did "
            "NOT end this runtime.\n",
        ),
    ):
        dump = _held_dump(logs, pid, fired_mono=10_000.0, deadline_mono=99_999.0)
        dump.write_text(f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n{marker}", encoding="utf-8")
        assert stall_watchdog.held_fire(pid, logs) is True, pid


def test_every_listing_surface_reads_the_same_held_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stall_held`` (JSON), the CLI cell and the panel word all ride ``collect``.

    ``collect_sessions`` is the single producer of ``SessionLine.stall_held``, which the
    ``/info`` panel (``HELD_STATE_WORD``), ``lop sessions``' ``HELD_CELL`` and
    ``lop sessions --json`` all render. So one recovered dump and one stuck dump through
    it pins that the listing agrees with ``held_fire`` for both. On origin/main the
    recovered row read ``stall_held=True``.
    """
    from local_operator.info.collect import collect_sessions

    logs = tmp_path / "logs"
    fired = time.time() - 3600
    started = fired - 3600
    _held_dump(logs, 101, fired_mono=10_000.0, deadline_mono=10_000.0 + 3600 + 200)
    _held_dump(logs, 202, fired_mono=10_000.0, deadline_mono=10_001.0)
    for pid in (101, 202):
        os.utime(stall_watchdog.dump_path(pid, logs), (fired, fired))

    real_scan = stall_watchdog._scan_marked
    monkeypatch.setattr(
        stall_watchdog, "_scan_marked", lambda directory, **kw: real_scan(logs, **kw)
    )
    records = [
        (
            _bare_record(
                pid=pid, session_id=f"s{pid}", started_at=started, heartbeat_at=time.time()
            ),
            "live",
        )
        for pid in (101, 202)
    ]
    info = collect_sessions(scan=lambda _root=None: records, usage=lambda _p: {})
    held = {line.pid: line.stall_held for line in info.lines}
    assert held == {101: False, 202: True}
    assert held[101] is stall_watchdog.held_fire(101, logs)
    assert held[202] is stall_watchdog.held_fire(202, logs)


def test_the_recycled_pid_fence_still_holds(tmp_path: Path) -> None:
    """``held_now`` keeps ``dump_is_current``'s fence: a predecessor's dump is not this life's."""
    logs = tmp_path / "logs"
    fired = time.time() - 3600
    dump = _held_dump(logs, 77, fired_mono=10_000.0, deadline_mono=10_001.0)
    os.utime(dump, (fired, fired))
    assert stall_watchdog.held_now(77, fired - 10, logs) is True
    assert stall_watchdog.held_now(77, fired + 10, logs) is False


class _SleepingClock:
    """``time`` as the watchdog sees it across a host sleep: wall moves, monotonic does not.

    macOS's ``time.monotonic`` is ``mach_absolute_time``, which PAUSES while the machine
    sleeps; ``time.time`` does not. ``sleep(s)`` advances only the wall clock, which is
    the whole shape agent review round 1 (M1) reproduced with ``sleep_probe.py``.
    ``strftime`` is the real one, so the marker the writer produces is the real text.
    """

    def __init__(self) -> None:
        self.mono = 50_000.0
        self.wall = 1_790_000_000.0

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def sleep(self, seconds: float) -> None:
        self.wall += seconds

    def strftime(self, fmt: str, *args: Any) -> str:
        # "Now" is THIS clock's wall time, so the marker's local stamp and the sibling's
        # epoch describe the same instant, as they do in a real runtime.
        return time.strftime(fmt, *(args or (time.localtime(self.wall),)))

    def localtime(self, *args: Any) -> Any:
        return time.localtime(*args)

    # The rest of ``time`` passes through, so a reader that still turns local time back
    # into an instant (the head this cell reproduces against) runs and FAILS on its
    # assertion rather than on a missing attribute.
    def mktime(self, *args: Any) -> float:
        return time.mktime(*args)

    def strptime(self, *args: Any) -> Any:
        return time.strptime(*args)


def _real_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid: int
) -> tuple[Any, _SleepingClock, Path]:
    """An ``_Armed`` driven through the PRODUCTION writers under a controlled clock.

    ``_record_held_fire`` writes the held marker and ``_record_deadline`` the sibling,
    exactly as a runtime does; only the C timer is stubbed (it cannot fire safely in a
    pytest worker, see ``test_runtime_stall_watchdog._FakeFaulthandler``). The fire is
    written by hand, as the C thread would, and then OBSERVED by the real code.
    """
    clock = _SleepingClock()
    monkeypatch.setattr(stall_watchdog, "time", clock)
    monkeypatch.setattr(stall_watchdog.faulthandler, "dump_traceback_later", lambda *_a, **_k: None)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    dump = stall_watchdog.dump_path(pid, logs)
    # ``arm``'s header, which every real dump opens with and the sibling is fenced on
    # (agent review round 2, m-A). Real time, because the fence compares it with the
    # sibling's real mtime.
    dump.write_text(_arm_header(pid), encoding="utf-8")
    handle = dump.open("a", encoding="utf-8")
    armed = stall_watchdog._Armed(dump, handle, BOUND_S, pid, busy=lambda: True)
    # Both planes beat at arm; then the WORKLOAD loop parks and only SERVING beats on.
    clock.mono += BOUND_S + 5
    with dump.open("a", encoding="utf-8") as fire:
        fire.write(f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n  File x, line 1\n")
    assert stall_watchdog._record_held_fire(armed) is True
    return armed, clock, dump


def _serving_beats(armed: Any, clock: _SleepingClock, seconds: float) -> None:
    """The healthy SERVING plane beating every 15 s for ``seconds`` of AWAKE time."""
    for _ in range(int(seconds // 15)):
        clock.mono += 15
        clock.wall += 15
        armed.last_beat[stall_watchdog.SERVING] = clock.mono
        stall_watchdog._record_deadline(armed)


def test_a_stuck_runtime_stays_held_across_a_host_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: a sleep must not read as a recovery.

    The previous head compared the sibling's EPOCH (derived as ``time.time() +
    (deadline - monotonic())``) with the marker's local time. The epoch jumps forward by
    every sleep while the pinned monotonic deadline does not move, so a stuck runtime
    read recovered after any sleep over 22.5 s — reproduced here through the real
    writers. REPRODUCTION: fails on 4d3323998. Pinned now: both stamps are monotonic.
    """
    armed, clock, dump = _real_writer(tmp_path, monkeypatch, 4101)
    _serving_beats(armed, clock, 60)
    assert stall_watchdog._holds(dump, dump.read_text(encoding="utf-8")) is True
    for slept in (60.0, 3600.0, 8 * 3600.0):
        clock.sleep(slept)
        _serving_beats(armed, clock, 30)
        assert (
            stall_watchdog._holds(dump, dump.read_text(encoding="utf-8")) is True
        ), f"slept {slept:g}s and a stuck runtime read as recovered"

    # ...and the same writers still report a REAL recovery: the parked plane beats.
    armed.last_beat[stall_watchdog.WORKLOAD] = clock.mono
    _serving_beats(armed, clock, 15)
    assert stall_watchdog._holds(dump, dump.read_text(encoding="utf-8")) is False
    armed.handle.close()


@pytest.mark.parametrize(
    ("writer_tz", "reader_tz", "fired_epoch"),
    [
        # The REPEATED fall-back hour: 01:30:01 EST, the second 01:30 of 2026-11-01.
        ("America/New_York", "America/New_York", 1_793_514_601.0),
        # A reader EAST of the writer: the local stamp parses three hours early there.
        ("America/Los_Angeles", "America/New_York", 1_790_230_000.0),
    ],
)
def test_the_readers_zone_and_the_dst_hour_cannot_unhold_a_stuck_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_tz: str,
    reader_tz: str,
    fired_epoch: float,
) -> None:
    """m1 / QA Q-1: the local time in the marker is for a person, never for the predicate.

    The writer stamps the marker under ``writer_tz`` at ``fired_epoch``; a stuck runtime's
    sibling stays pinned at the fire. The reader runs under ``reader_tz``. On 4d3323998 the
    first case parsed the repeated hour as EDT (an hour early) and the second by the
    zone offset, and both read the runtime as recovered. REPRODUCTION: fails there.
    """
    armed, clock, dump = _real_writer(tmp_path, monkeypatch, 4102)
    # Re-stamp the marker exactly as the writer would have under ``writer_tz``: the local
    # time is what ``time.strftime`` produces THERE, the monotonic stamp is unchanged.
    monkeypatch.setenv("TZ", writer_tz)
    time.tzset()
    try:
        local = time.strftime(stall_watchdog._FIRED_AT_FORMAT, time.localtime(fired_epoch))
        text = dump.read_text(encoding="utf-8")
        start = text.index(stall_watchdog._FIRED_AT_PHRASE) + len(stall_watchdog._FIRED_AT_PHRASE)
        text = text[:start] + local + text[start + 19 :]
        dump.write_text(text, encoding="utf-8")
        # The sibling as a stuck runtime leaves it: epoch and monotonic both AT the fire.
        dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).write_text(
            f"{fired_epoch + 1:.3f} workload {armed.after_fire_at + 1:.3f}\n", encoding="utf-8"
        )
        monkeypatch.setenv("TZ", reader_tz)
        time.tzset()
        assert stall_watchdog._holds(dump, dump.read_text(encoding="utf-8")) is True
    finally:
        monkeypatch.undo()
        time.tzset()
        armed.handle.close()


def test_the_writer_and_the_reader_agree_on_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N2: a round trip through the production writer, not a grep of its source.

    What ``_record_held_fire`` writes, ``_held_fired_mono`` reads back as the same
    monotonic instant, and ``_record_deadline``'s third field is ``pin()`` on that clock.
    REPRODUCTION: 4d3323998 wrote no monotonic stamp and no third field.
    """
    armed, clock, dump = _real_writer(tmp_path, monkeypatch, 4103)
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog._held_fired_mono(text) == pytest.approx(armed.after_fire_at, abs=1e-3)
    stall_watchdog._record_deadline(armed)
    epoch, leg, mono = dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).read_text().split()
    assert (leg, float(mono)) == (armed.pin()[0], pytest.approx(armed.pin()[1], abs=1e-3))
    assert float(epoch) == pytest.approx(clock.wall + (armed.pin()[1] - clock.mono), abs=1e-2)
    armed.handle.close()


# ---------------------------------------------------------------------------------------
# Bug 1 — the drain skip
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_wedged_draining_runtime_is_stopped_not_skipped(
    no_signals,
) -> None:
    """Pids 42983/43911: ``leaving`` set, heartbeat hours stale, `lop sessions` wedged.

    On origin/main the ladder returned ``draining`` — "it leaves by itself, nothing to
    do" — which it never would. Now it falls through to the ordinary ladder: this
    runtime's socket answers, so rung 1 stops it deliberately, no signal sent, and the
    wait line says why the drain was not believed.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(
        record,
        busy=True,
        leaving=LEAVING_FOR_BUILD,
        heartbeat_at=time.time() - 5.6 * 3600,
    )
    said: list[str] = []
    try:
        outcome = await control.stop_session(
            target, timeout_s=3.0, _root=config_dir(), _command="lop stop", on_wait=said.append
        )
        assert outcome.method == "socket", outcome.line
        assert handle.stops == [True]
        assert no_signals[0] == []
        assert said and "has not reported for 5h" in said[0], said
        assert LEAVING_FOR_BUILD in said[0], said
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_fresh_draining_runtime_is_still_skipped(
    no_signals,
) -> None:
    """The drain skip itself is unchanged for a runtime that is still reporting.

    Passes on origin/main as well: it pins that the gate narrowed the skip rather than
    removing it.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(record, busy=True, leaving=LEAVING_FOR_BUILD, heartbeat_at=time.time())
    try:
        outcome = await control.stop_session(target, timeout_s=3.0, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert "it leaves by itself, nothing to do" in outcome.line
        assert handle.stops == []
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_held_draining_runtime_is_stopped_not_skipped(
    no_signals,
) -> None:
    """Fresh heartbeat but a current, un-superseded held fire: the drain is not moving.

    The serving plane can keep the record's heartbeat fresh while the workload loop —
    the one whose turn the drain waits on — is parked, and the held dump is the
    evidence of exactly that. END TO END (agent review round 1, m3): a real held dump
    and a pinned deadline sibling in the store ``dump_evidence`` resolves, written after
    the record's start so the recycled-pid fence admits it; nothing is monkeypatched
    between ``stop_session`` and ``_holds``. REPRODUCTION: skipped on origin/main.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(
        record,
        busy=True,
        leaving=LEAVING_FOR_BUILD,
        heartbeat_at=time.time(),
        started_at=time.time() - 60,
    )
    dump = _held_dump(
        stall_watchdog.dump_path(target.pid).parent,
        target.pid,
        fired_mono=10_000.0,
        deadline_mono=10_001.0,
        leg="workload",
    )
    said: list[str] = []
    try:
        assert stall_watchdog.held_now(target.pid, target.started_at) is True
        outcome = await control.stop_session(
            target, timeout_s=3.0, _root=config_dir(), on_wait=said.append
        )
        assert outcome.method == "socket", outcome.line
        # The reason in the ``bound held`` cell's own words, not the timer's (D5).
        assert said and "its stall bound is held (it fired with work in flight)" in said[0], said
        assert "re-armed" not in said[0], said
        assert "not left to finish" in said[0], said
        # Q-2: the line names the decision, and cannot contradict a later refusal.
        assert "trying the ordinary stop" in said[0], said
        assert "stopping it rather than" not in said[0], said
    finally:
        server.close()
        dump.unlink(missing_ok=True)
        dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).unlink(missing_ok=True)


def test_the_stalled_verdict_is_the_stall_bound_not_the_listings_45s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2: ``wedged`` alone does not cut a drain; the watchdog's own bound does.

    ``classify`` calls a beat older than 45 s ``wedged``, and its own docstring records
    105.8 s and 205.8 s gaps on runtimes whose CPU was advancing. The stall bound (300 s
    by default) is the number sized above those. So the fall-through needs BOTH: the
    listing's ``wedged`` and a beat past the bound. REPRODUCTION: 4d3323998 fell through
    at 46 s. The configured bound is honoured, and a switched-off knob keeps the default.
    """
    monkeypatch.delenv(stall_watchdog.ENV_SECONDS, raising=False)
    now = time.time()

    def aged(seconds: float) -> Any:
        return _bare_record(pid=os.getpid(), heartbeat_at=now - seconds)

    for age in (HEARTBEAT_TIMEOUT_S + 1, 105.8, 205.8, stall_watchdog.DEFAULT_STALL_S - 5):
        assert registry.classify(aged(age)).state == "wedged", age
        assert control._drain_stalled(aged(age)) == "", f"a drain {age}s quiet was cut"
    assert control._drain_stalled(aged(stall_watchdog.DEFAULT_STALL_S + 5)).startswith(
        "it has not reported for"
    )

    monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "900")
    assert control._drain_stalled(aged(600)) == ""
    assert control._drain_stalled(aged(905)) != ""
    monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "off")
    assert control._drain_stalled(aged(205.8)) == ""
    assert control._drain_stalled(aged(stall_watchdog.DEFAULT_STALL_S + 5)) != ""


@pytest.mark.asyncio
async def test_a_slow_drain_inside_the_stall_bound_is_still_skipped(
    no_signals,
) -> None:
    """M2 through the ladder: a draining runtime 120 s quiet is ``draining``, not stopped.

    120 s sits inside the band the codebase documents as slow-but-working (45 s to the
    bound). REPRODUCTION: 4d3323998 stopped it by the socket rung — the U1/U2 harm.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(
        record, busy=True, leaving=LEAVING_FOR_BUILD, heartbeat_at=time.time() - 120
    )
    try:
        outcome = await control.stop_session(target, timeout_s=3.0, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert handle.stops == [], "a slow drain was cut"
        assert no_signals[0] == []
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Agent review round 2 (M-A, m-A, m-B), on the formats origin/main writes.
# ---------------------------------------------------------------------------


def _main_format_held_dump(logs: Path, pid: int, *, sibling_epoch: float) -> Path:
    """A held dump and sibling EXACTLY as a pre-PR build (origin/main) writes them.

    Spelled as literals copied from main's writer — the arm header, main's held marker
    (local time only, no monotonic stamp) and the two-field ``<epoch> <leg>`` sibling —
    so the cell reads what a runtime started before this fix really leaves on disk.
    """
    logs.mkdir(parents=True, exist_ok=True)
    dump = stall_watchdog.dump_path(pid, logs)
    dump.write_text(
        _arm_header(pid, time.time() - 6 * 3600)
        + "Timeout (0:05:00)!\n  File x, line 1\n"
        + "[stall watchdog] bound held: the bound fired at 2026-09-24 03:23:34 and did "
        "NOT end this runtime. Work was in flight when the fire was observed, so the "
        "runtime is stalled with it. Every thread's stack is above. Inspect this dump "
        "and stop the runtime explicitly if it remains stuck.\n",
        encoding="utf-8",
    )
    stall_watchdog.deadline_path(pid, logs).write_text(
        f"{sibling_epoch:.3f} serving\n", encoding="utf-8"
    )
    return dump


def test_an_old_format_held_marker_is_still_labelled_held(tmp_path: Path) -> None:
    """M-A, the LABEL half: cannot tell → ``bound held``, for a person to look at.

    A pre-PR marker carries no monotonic stamp, so nothing can show a recovery, and
    the label keeps the conservative reading. Passes on cae367569 too: a pin that the
    ladder's narrower reading did not leak into the listing.
    """
    logs = tmp_path / "logs"
    _main_format_held_dump(logs, 65820, sibling_epoch=time.time() + 290)
    assert stall_watchdog.held_fire(65820, logs) is True
    assert 65820 in stall_watchdog.held_pids(logs)


@pytest.mark.asyncio
async def test_an_old_format_held_drain_with_a_fresh_beat_is_still_skipped(
    no_signals,
) -> None:
    """M-A, the LADDER half: a pre-PR runtime that held, recovered and is draining.

    The reviewer's ``oldfmt_ladder.py`` shape, end to end through ``stop_session``: fire
    5 h ago in main's format, sibling beating at now + 290 s, heartbeat 5 s old,
    ``leaving`` set, ``started_at`` before the dump. The ladder takes the held arm only
    on complete new-format evidence, so this is judged on its heartbeat, which is
    fresh: skipped. REPRODUCTION: cae367569 returned ``socket`` — the turn was cut.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(
        record,
        busy=True,
        leaving=LEAVING_FOR_BUILD,
        heartbeat_at=time.time() - 5,
        started_at=time.time() - 7 * 3600,
    )
    dump = _main_format_held_dump(
        stall_watchdog.dump_path(target.pid).parent, target.pid, sibling_epoch=time.time() + 290
    )
    try:
        assert control._drain_stalled(target) == "", "an unproven held reading cut a drain"
        # The label this operator followed says ``bound held; lop stop``.
        assert stall_watchdog.held_now(target.pid, target.started_at) is True
        outcome = await control.stop_session(target, timeout_s=3.0, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert handle.stops == []
        # D3 / m-C / Q-6: the skip says why it contradicts that label and names only
        # ``--force``, rather than "nothing to do".
        assert (
            '"bound held" reading comes from an older build and cannot be confirmed' in outcome.line
        ), outcome.line
        assert "nothing to do" not in outcome.line, outcome.line
        assert outcome.line.endswith("(--force to stop it anyway)"), outcome.line
    finally:
        server.close()
        dump.unlink(missing_ok=True)
        dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).unlink(missing_ok=True)


def test_an_old_format_held_drain_falls_back_to_the_heartbeat(tmp_path: Path) -> None:
    """M-A: with unproven evidence the ladder still cuts a drain whose BEAT is past the bound.

    Old-format marker AND a heartbeat older than the stall bound — pid 42983's shape on
    a pre-PR build — so the heartbeat arm alone decides, and it says stalled. Passes on
    cae367569 as well (its held arm answered first); pinned so the M-A narrowing cannot
    turn into "old-format drains are never cut".
    """
    pid = os.getpid()
    logs = stall_watchdog.dump_path(pid).parent
    dump = _main_format_held_dump(logs, pid, sibling_epoch=time.time() - 5 * 3600)
    try:
        record = _bare_record(
            pid=pid,
            heartbeat_at=time.time() - 5.6 * 3600,
            started_at=time.time() - 7 * 3600,
        )
        assert control._drain_stalled(record).startswith("it has not reported for")
    finally:
        dump.unlink(missing_ok=True)
        dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).unlink(missing_ok=True)


def test_a_sibling_from_before_this_lifes_arm_is_not_a_recovery(tmp_path: Path) -> None:
    """m-A: a reboot, a recycled pid, and an old ``.deadline`` the arm failed to unlink.

    This life fired held early in its boot (small monotonic stamp); the surviving
    sibling is the PREVIOUS boot's, with a large monotonic deadline — "re-armed long
    after the fire" if believed. Its mtime predates this dump's arm header, so it is
    not this life's evidence and the reading stays held. The control: the same sibling
    written after the arm is a recovery. REPRODUCTION: cae367569 read ``held=False``.
    """
    logs = tmp_path / "logs"
    dump = _held_dump(logs, 4103, fired_mono=100.0, deadline_mono=900_000.0)
    sibling = dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX)
    before_arm = time.time() - 2 * 3600
    os.utime(sibling, (before_arm, before_arm))
    assert stall_watchdog.held_fire(4103, logs) is True, "a previous life's sibling unheld it"
    assert stall_watchdog.held_reading(dump, dump.read_text("utf-8")) == (
        stall_watchdog.HELD_UNPROVEN
    )
    os.utime(sibling, None)
    assert stall_watchdog.held_fire(4103, logs) is False


@pytest.mark.asyncio
async def test_the_drain_check_runs_off_the_event_loop(
    no_signals, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m-B: ``_drain_stalled`` may fork ``ps`` and read dumps; the TUI calls this ladder on
    its loop, so the check hops to a thread. REPRODUCTION: ran on the loop on cae367569.
    """
    import threading

    seen: list[threading.Thread] = []

    def spy(record: Any) -> str:
        seen.append(threading.current_thread())
        return ""

    monkeypatch.setattr(control, "_drain_stalled", spy)
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(record, busy=True, leaving=LEAVING_FOR_BUILD, heartbeat_at=time.time())
    try:
        await control.stop_session(target, timeout_s=3.0, _root=config_dir())
    finally:
        server.close()
    assert seen and seen[0] is not threading.main_thread(), seen


def test_held_reading_opens_each_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N-2: the sibling is opened once and the dump never, since the caller holds its text.

    ``held_pids`` runs this for every dump in a listing, and two sibling reads leave a
    window for a re-arm's replace to land between the fence and the fields. Counted on
    the three readings (proven, unproven, recovered) so no path re-reads.
    """
    logs = tmp_path / "logs"
    cases = [
        _held_dump(logs, 4201, fired_mono=10_000.0, deadline_mono=10_001.0),
        _held_dump(logs, 4202, fired_mono=10_000.0, deadline_mono=10_000.0 + 3600),
        _main_format_held_dump(logs, 4203, sibling_epoch=time.time() + 290),
    ]
    texts = {dump: dump.read_text(encoding="utf-8") for dump in cases}
    opened: list[str] = []
    real_open, real_read_text = Path.open, Path.read_text

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        opened.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    monkeypatch.setattr(Path, "read_text", counting_read_text)
    readings = []
    for dump in cases:
        opened.clear()
        readings.append(stall_watchdog.held_reading(dump, texts[dump]))
        sibling = dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).name
        assert opened.count(dump.name) == 0, (dump.name, opened)
        assert opened.count(sibling) <= 1, (dump.name, opened)
    assert readings == [
        stall_watchdog.HELD_PROVEN,
        stall_watchdog.NOT_HELD,
        stall_watchdog.HELD_UNPROVEN,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["old-format", "current-no-sibling"])
async def test_the_skip_line_names_an_older_build_only_when_the_marker_is_one(
    no_signals, marker: str
) -> None:
    """Agent review round 1 on #1541, m1: the stated cause must be the artifact's.

    Both runtimes are leaving with a fresh beat and an UNPROVEN held reading, so both
    are skipped. An old-format marker (no monotonic stamp) is from an older build and
    the line may say so. A CURRENT-build marker whose sibling is missing — reachable,
    since ``_record_held_fire`` re-arms without writing one — gets cause-neutral words.
    REPRODUCTION: a1a5de647 told the current-build runtime its reading came from an
    older build.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(
        record,
        busy=True,
        leaving=LEAVING_FOR_BUILD,
        heartbeat_at=time.time() - 5,
        started_at=time.time() - 7 * 3600,
    )
    logs = stall_watchdog.dump_path(target.pid).parent
    if marker == "old-format":
        dump = _main_format_held_dump(logs, target.pid, sibling_epoch=time.time() + 290)
    else:
        dump = _held_dump(logs, target.pid, fired_mono=10_000.0, deadline_mono=None)
    try:
        assert stall_watchdog.held_reading(dump, dump.read_text("utf-8")) == (
            stall_watchdog.HELD_UNPROVEN
        )
        outcome = await control.stop_session(target, timeout_s=3.0, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert handle.stops == []
        if marker == "old-format":
            assert (
                '"bound held" reading comes from an older build and cannot be confirmed'
                in outcome.line
            ), outcome.line
        else:
            assert "older build" not in outcome.line, outcome.line
            assert (
                '"bound held" reading cannot be confirmed from its evidence' in outcome.line
            ), outcome.line
        assert "nothing to do" not in outcome.line, outcome.line
        assert outcome.line.endswith("(--force to stop it anyway)"), outcome.line
    finally:
        server.close()
        dump.unlink(missing_ok=True)
        dump.with_suffix(stall_watchdog.DEADLINE_SUFFIX).unlink(missing_ok=True)


def test_the_row_tag_names_the_reason_the_other_surfaces_show() -> None:
    """D1: the ``/stop all`` tag uses what ``lop sessions`` and ``lop stop`` say.

    The heartbeat arm reads as HB_AGE does, the held arm as the STALLED cell does; any
    other reason keeps the held wording (the only other arm) rather than inventing one.
    """
    assert control.stall_tag("it has not reported for 5h") == "leaving, not reporting for 5h"
    assert control.stall_tag(control.STALL_HELD_REASON) == "leaving, bound held"


def test_the_stalled_probe_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ladder's never-raise contract: a probe that throws is the old skip, not a crash."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(control.registry, "classify", boom)
    record = _bare_record(pid=os.getpid())
    assert control._drain_stalled(record) == ""
