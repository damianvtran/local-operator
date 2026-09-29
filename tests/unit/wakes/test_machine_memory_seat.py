"""Unit tests for the machine memory pass's seat in the wake supervisor.

The seat owns the cadence and the cooldown rung; the pass itself is tested in
``tests/unit/session/runtime/test_machine_memory.py``. Everything here runs the
seat's own seams — no config store, no fleet, and the pass function is replaced
by a spy, so a seat bug cannot be hidden by a working pass or the other way
round.
"""

from __future__ import annotations

import asyncio
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
        killed=machine_memory.memory_guard.Fragment(pid=990001, mb=2048) if killed else None,
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


def test_the_cooldown_withholds_the_second_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def spy(config_dir: Path, *, apply: bool = True, kill_allowed: bool = True) -> object:
        seen.append(kill_allowed)
        return _report(killed=kill_allowed)

    monkeypatch.setattr(machine_memory, "machine_memory_pass", spy)
    seat = sup._MachineMemorySweep(tmp_path)
    first = seat.sweep()
    assert seen == [True]
    assert first.killed is not None
    assert seat.last_kill_at is not None
    # Immediately after: the kill rung is withheld, and the pass still runs (it
    # warns; it just cannot end anything).
    second = seat.sweep()
    assert seen == [True, False]
    assert second.killed is None
    # The cooldown expires on the clock, not on the pass count.
    seat.last_kill_at -= sup.MACHINE_MEMORY_KILL_COOLDOWN_S
    seat.sweep()
    assert seen == [True, False, True]


def test_a_failed_pass_is_a_warning_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(config_dir: Path, *, apply: bool = True, kill_allowed: bool = True) -> object:
        raise RuntimeError("ps exploded")

    monkeypatch.setattr(machine_memory, "machine_memory_pass", boom)
    seat = sup._MachineMemorySweep(tmp_path)
    with pytest.raises(RuntimeError):
        # The SEAT does not swallow: the exception belongs to the worker thread,
        # and `_finished` logs it. Swallowing here would hide it entirely.
        seat.sweep()
