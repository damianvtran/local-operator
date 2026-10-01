"""Unit tests for the machine memory pass's seat in the wake supervisor.

The seat owns the cadence and the cooldown rung; the pass itself is tested in
``tests/unit/session/runtime/test_machine_memory.py``. Everything here runs the
seat's own seams — no config store, no fleet, and the pass function is replaced
by a spy, so a seat bug cannot be hidden by a working pass or the other way
round.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

from local_operator.session.runtime import machine_memory
from local_operator.wakes import supervisor as sup


def _report(*, killed: bool = False) -> machine_memory.MemoryPassReport:
    return machine_memory.MemoryPassReport(
        state="act",
        fleet_mb=900,
        runtimes=1,
        measured=1,
        unmeasured=0,
        killed=machine_memory.memory_guard.Fragment(pid=9900001, mb=2048) if killed else None,
        reason="fleet 900 MB of 1000 MB physical",
    )


def test_first_pass_is_due_and_the_next_follows_the_interval(tmp_path: Path) -> None:
    seat = sup._MachineMemorySweep(tmp_path)
    assert seat.due()
    assert seat.seconds_until() == 0.0

    seat.next_at = time.monotonic() + 10.0
    now = seat.next_at
    assert not seat.due(now=now - 5)
    assert seat.seconds_until(now=now - 5) == pytest.approx(5.0)
    assert seat.due(now=now)


def test_kick_runs_one_pass_per_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def scenario() -> None:
        seat = sup._MachineMemorySweep(tmp_path)
        monkeypatch.setattr(seat, "sweep", lambda **kwargs: calls.append(kwargs) or _report())
        seat.kick()
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        # Not due again: the kick arms the next pass from HERE, not from the
        # completed one, so a slow pass cannot make the loop spin.
        seat.kick()
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        await seat.shutdown()

    asyncio.run(scenario())


def _fragment(pid: int, ppid: int, mb: int = 2048) -> machine_memory.memory_guard.Fragment:
    return machine_memory.memory_guard.Fragment(pid=pid, mb=mb, pids=(pid,), ppid=ppid, pgid=pid)


def _report_ending(
    fragment: machine_memory.memory_guard.Fragment,
) -> machine_memory.MemoryPassReport:
    return machine_memory.MemoryPassReport(
        state="act",
        fleet_mb=900,
        runtimes=1,
        measured=1,
        unmeasured=0,
        killed=fragment,
        reason="fleet 900 MB of 1000 MB physical",
    )


def test_the_cooldown_withholds_a_second_stop_of_the_same_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regrowing respawn the cooldown exists for: same parent, new pid."""
    held: list[bool] = []
    ended = _fragment(9900010, ppid=9900001)

    def spy(
        config_dir: Path, *, apply: bool = True, in_cooldown: object = None, **_: object
    ) -> object:
        assert callable(in_cooldown)
        held.append(in_cooldown(_fragment(9900011, ppid=9900001)))
        return _report_ending(ended)

    monkeypatch.setattr(machine_memory, "machine_memory_pass", spy)
    seat = sup._MachineMemorySweep(tmp_path)
    first = seat.sweep()
    assert held == [False] and first.killed is not None
    assert seat.last_kill_at is not None
    seat.sweep()
    assert held == [False, True]
    # The cooldown expires on the clock, not on the pass count.
    seat._ended = [(at - sup.MACHINE_MEMORY_KILL_COOLDOWN_S, r, p) for at, r, p in seat._ended]
    assert seat.in_cooldown(_fragment(9900011, ppid=9900001)) is False


def test_a_different_runaway_is_killable_on_the_next_pass(tmp_path: Path) -> None:
    """2026-09-30 02:39-02:43: a 52 GB and a 350 GB runaway were withheld for 10
    minutes by a cooldown earned by an UNRELATED kill. Different parent, different
    pid: not held."""
    seat = sup._MachineMemorySweep(tmp_path)
    seat._ended = [(time.monotonic(), 9900010, 9900001)]
    assert seat.in_cooldown(_fragment(9900010, ppid=9900001)) is True  # the same pid
    assert seat.in_cooldown(_fragment(9900011, ppid=9900001)) is True  # same parent
    assert seat.in_cooldown(_fragment(9900020, ppid=9900002)) is False  # a stranger
    # init (pid 1) is everyone's parent, so it never makes two fragments kin.
    seat._ended = [(time.monotonic(), 9900010, 1)]
    assert seat.in_cooldown(_fragment(9900030, ppid=1)) is False


def test_a_failed_pass_is_a_warning_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(config_dir: Path, *, apply: bool = True, **_: object) -> object:
        raise RuntimeError("ps exploded")

    monkeypatch.setattr(machine_memory, "machine_memory_pass", boom)
    seat = sup._MachineMemorySweep(tmp_path)
    with pytest.raises(RuntimeError):
        # The SEAT does not swallow: the exception belongs to the worker thread,
        # and `_finished` logs it. Swallowing here would hide it entirely.
        seat.sweep()


def test_the_seat_logs_the_summary_when_it_changes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class _Report:
        def __init__(self, state: str, summary: str) -> None:
            self.state = state
            self.killed = None
            self._summary = summary

        def summary(self) -> str:
            return self._summary

    async def scenario() -> None:
        seat = sup._MachineMemorySweep(tmp_path)

        async def completed(state: str, text: str) -> "asyncio.Task[object]":
            async def inner() -> object:
                return _Report(state, text)

            task = asyncio.ensure_future(inner())
            await task
            return task

        first = await completed("ok", "machine memory: ok")
        seat._finished(first)
        # Same STATE, different volatile numbers: the key must NOT move (R2-2 —
        # a text comparison logged INFO nearly every healthy pass).
        again = await completed("ok", "machine memory: ok fleet 23306 MB")
        seat._finished(again)
        warn = await completed("warn", "machine memory: warn - fleet 90% of physical")
        seat._finished(warn)

    caplog.set_level(logging.DEBUG, logger="local_operator.wakes.supervisor")
    asyncio.run(scenario())
    # First reading is a CHANGE (None -> ok) and goes out at INFO; the same
    # reading again is DEBUG; a transition back out of ok is INFO again — the
    # report is never dropped, only quieted.
    assert [record.levelno for record in caplog.records] == [
        logging.INFO,
        logging.DEBUG,
        logging.INFO,
    ]
    assert caplog.records[2].getMessage() == "machine memory: warn - fleet 90% of physical"
