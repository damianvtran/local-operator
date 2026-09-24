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

Every cell here was run against ``origin/main``'s code first and failed there (AGENTS.md,
"Prove the test can still fail"); the failing assertion is named in each docstring.
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


def _stamp(epoch: float) -> str:
    """The moment exactly as ``_record_held_fire`` writes it: local time, to the second."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _held_dump(
    logs: Path,
    pid: int,
    *,
    fired_at: float,
    deadline: float | None,
    leg: str = "serving",
    extra: str = "",
) -> Path:
    """A real held dump and (optionally) its deadline sibling, in the writers' formats."""
    logs.mkdir(parents=True, exist_ok=True)
    dump = stall_watchdog.dump_path(pid, logs)
    dump.write_text(
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n  File x, line 1\n"
        f"{stall_watchdog.HELD_MARKER}the bound fired at {_stamp(fired_at)} and did NOT end "
        "this runtime. Work was in flight when the fire was observed.\n" + extra,
        encoding="utf-8",
    )
    if deadline is not None:
        stall_watchdog.deadline_path(pid, logs).write_text(
            f"{deadline:.3f} {leg}\n", encoding="utf-8"
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
    fired = time.time() - 5 * 3600
    _held_dump(logs, 65820, fired_at=fired, deadline=time.time() + BOUND_S - 10)

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
    fired = time.time() - 3600
    # Deadline AT the fire (the quiet plane pinned it), sibling freshly written.
    _held_dump(logs, 11, fired_at=fired, deadline=fired + 1.0)
    _held_dump(logs, 22, fired_at=fired, deadline=None)
    _held_dump(logs, 33, fired_at=fired, deadline=fired - 20_000.0, leg="progress")

    for pid in (11, 22, 33):
        assert stall_watchdog.held_fire(pid, logs) is True, pid
    assert stall_watchdog.held_pids(logs) == {11, 22, 33}


def test_a_later_idle_fire_supersedes_an_earlier_held_one(tmp_path: Path) -> None:
    """The work the held fire named has cleared: the most recent fire found none.

    ``OBSERVED_MARKER`` is written only when the fire saw NO work in flight, so a later
    one says the stall the earlier marker described is over — the same fact the
    re-arm test establishes, from the artifact itself. Held on origin/main, which read
    any ``HELD_MARKER`` anywhere in the file.
    """
    logs = tmp_path / "logs"
    fired = time.time() - 3600
    _held_dump(
        logs,
        44,
        fired_at=fired,
        deadline=fired + 1.0,
        extra=f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n"
        f"{stall_watchdog.OBSERVED_MARKER}the bound fired at {_stamp(fired + 600)} ...\n",
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
    first = time.time() - 7200
    second = time.time() - 60
    _held_dump(
        logs,
        55,
        fired_at=first,
        deadline=first + 3000,
        extra=f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n"
        f"{stall_watchdog.HELD_MARKER}the bound fired at {_stamp(second)} and did NOT end "
        "this runtime.\n",
    )
    assert stall_watchdog.held_fire(55, logs) is True


def test_an_unparseable_held_marker_keeps_the_held_reading(tmp_path: Path) -> None:
    """No stamp, no proof of recovery: the quiet direction is the reading the dump states."""
    logs = tmp_path / "logs"
    dump = _held_dump(logs, 66, fired_at=time.time() - 3600, deadline=time.time() + 300)
    dump.write_text(
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n{stall_watchdog.HELD_MARKER}held\n",
        encoding="utf-8",
    )
    assert stall_watchdog.held_fire(66, logs) is True


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
    _held_dump(logs, 101, fired_at=fired, deadline=time.time() + 200)
    _held_dump(logs, 202, fired_at=fired, deadline=fired + 1.0)
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
    dump = _held_dump(logs, 77, fired_at=fired, deadline=fired + 1.0)
    os.utime(dump, (fired, fired))
    assert stall_watchdog.held_now(77, fired - 10, logs) is True
    assert stall_watchdog.held_now(77, fired + 10, logs) is False


def test_the_predicate_reads_the_writers_own_stamp() -> None:
    """The reader parses what ``_record_held_fire`` writes, not a format of its own."""
    import inspect

    source = inspect.getsource(stall_watchdog._record_held_fire)
    assert "{HELD_MARKER}the bound fired at " in source
    assert "time.strftime('%Y-%m-%d %H:%M:%S')" in source
    assert stall_watchdog._FIRED_AT_FORMAT == "%Y-%m-%d %H:%M:%S"


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh heartbeat but a current, un-superseded held fire: the drain is not moving.

    The serving plane can keep the record's heartbeat fresh while the workload loop —
    the one whose turn the drain waits on — is parked, and the held dump is the
    evidence of exactly that. Skipped on origin/main.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(record, busy=True, leaving=LEAVING_FOR_BUILD, heartbeat_at=time.time())
    seen: list[tuple[int, float]] = []

    def held_now(pid: int, started_at: float, directory: Any = None) -> bool:
        seen.append((pid, started_at))
        return True

    monkeypatch.setattr(stall_watchdog, "held_now", held_now)
    said: list[str] = []
    try:
        outcome = await control.stop_session(
            target, timeout_s=3.0, _root=config_dir(), on_wait=said.append
        )
        assert outcome.method == "socket", outcome.line
        assert seen == [(target.pid, target.started_at)], "asked with this life's fence"
        assert said and "stall bound fired" in said[0], said
    finally:
        server.close()


def test_the_stalled_verdict_is_the_listings_own() -> None:
    """No second threshold: the ladder's ``wedged`` is ``registry.classify``'s.

    One second either side of ``HEARTBEAT_TIMEOUT_S`` against a live pid (this one).
    """
    now = time.time()
    fresh = _bare_record(pid=os.getpid(), heartbeat_at=now - HEARTBEAT_TIMEOUT_S + 1)
    stale = _bare_record(pid=os.getpid(), heartbeat_at=now - HEARTBEAT_TIMEOUT_S - 1)
    assert registry.classify(fresh).state == "live"
    assert control._drain_stalled(fresh) == ""
    assert registry.classify(stale).state == "wedged"
    assert control._drain_stalled(stale).startswith("it has not reported for")


def test_the_stalled_probe_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ladder's never-raise contract: a probe that throws is the old skip, not a crash."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(control.registry, "classify", boom)
    record = _bare_record(pid=os.getpid())
    assert control._drain_stalled(record) == ""
