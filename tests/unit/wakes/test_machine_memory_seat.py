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


Lineage = frozenset


def _report_ending(lineage: "frozenset[tuple[str, int]]") -> machine_memory.MemoryPassReport:
    return machine_memory.MemoryPassReport(
        state="act",
        fleet_mb=900,
        runtimes=1,
        measured=1,
        unmeasured=0,
        killed=machine_memory.memory_guard.Fragment(pid=9900010, mb=2048),
        killed_lineage=lineage,
        reason="fleet 900 MB of 1000 MB physical",
    )


def test_the_cooldown_withholds_a_second_stop_of_the_same_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regrowing respawn the cooldown exists for: same non-runtime parent, new pid."""
    held: list[bool] = []
    ended = frozenset({("pid", 9900010), ("ppid", 9900005)})
    respawn = frozenset({("pid", 9900011), ("ppid", 9900005)})

    def spy(
        config_dir: Path, *, apply: bool = True, in_cooldown: object = None, **_: object
    ) -> object:
        assert callable(in_cooldown)
        held.append(bool(in_cooldown(respawn)))
        return _report_ending(ended)

    monkeypatch.setattr(machine_memory, "machine_memory_pass", spy)
    seat = sup._MachineMemorySweep(tmp_path)
    first = seat.sweep()
    assert held == [False] and first.killed is not None
    seat.sweep()
    assert held == [False, True]
    # The cooldown expires on the clock, not on the pass count.
    seat._ended = [(at - sup.MACHINE_MEMORY_KILL_COOLDOWN_S, keys) for at, keys in seat._ended]
    assert seat.in_cooldown(respawn) is False


def test_a_different_runaway_is_killable_on_the_next_pass(tmp_path: Path) -> None:
    """2026-09-30 02:39-02:43: a 52 GB and a 350 GB runaway were withheld for 10
    minutes by a cooldown earned by an UNRELATED kill."""
    seat = sup._MachineMemorySweep(tmp_path)
    seat._ended = [(time.monotonic(), frozenset({("pid", 9900010), ("pgid", 9900010)}))]
    assert seat.in_cooldown(frozenset({("pid", 9900010)})) is True  # the same pid
    assert seat.in_cooldown(frozenset({("pid", 9900011), ("pgid", 9900010)})) is True  # same group
    assert seat.in_cooldown(frozenset({("pid", 9900020), ("pgid", 9900020)})) is False


def test_a_second_runaway_under_the_same_runtime_is_killable_on_the_next_pass(
    tmp_path: Path,
) -> None:
    """R2/Q2, through the REAL keying: two unrelated rigs are both direct children
    of ONE session runtime (one shares its process group, one leads its own). The
    first is ended; the second must not be held. The first revision keyed on the
    parent pid, which is the runtime's, and held it for ten minutes."""
    runtime = 9900001
    rows = {runtime: (1, runtime), 9900010: (runtime, runtime), 9900020: (runtime, 9900020)}
    memory_guard = machine_memory.memory_guard
    ended = memory_guard.Fragment(pid=9900010, mb=1, ppid=runtime, pgid=runtime)
    other = memory_guard.Fragment(pid=9900020, mb=1, ppid=runtime, pgid=9900020)
    seat = sup._MachineMemorySweep(tmp_path)
    seat._ended = [(time.monotonic(), machine_memory.lineage_keys(ended, rows, [runtime]))]
    assert seat.in_cooldown(machine_memory.lineage_keys(ended, rows, [runtime])) is True
    assert seat.in_cooldown(machine_memory.lineage_keys(other, rows, [runtime])) is False


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
