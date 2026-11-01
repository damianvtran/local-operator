"""A3: authority is re-proved on a cadence, not on every poll.

One refresh runs two complete strict censuses. Doing that per 50 ms poll was ~40
``/proc`` scans a second on Linux, where ordinary fork churn then read as lost
authority and aborted healthy gates. Signals still re-prove inside ``signal()``;
this pins the *between-signals* cadence so a future edit cannot quietly restore
the per-poll sweep.
"""
from __future__ import annotations

import io
import subprocess
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401


class World:
    def __init__(self):
        self.rows = {1: (0, 1, 1, "creator"), 10: (1, 10, 10, "leader")}
        self.now = 0.0
        self.censuses = 0

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        self.censuses += 1
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        return {pid: owned.procstate.ProcessSample(False, self.rows[pid][3])
                for pid in pids if pid in self.rows}

    def identity(self, pid):
        return self.rows[pid][:3]

    def send(self, pgid, _signum):
        self.rows.pop(pgid, None)


@pytest.fixture
def wrapper(memory_wrapper, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return memory_wrapper


def test_the_census_runs_on_a_cadence_not_once_per_poll(wrapper, monkeypatch):
    world = World()

    class Clock:
        def monotonic(self):
            return world.clock()

        def sleep(self, seconds):
            world.now += seconds
    monkeypatch.setattr(wrapper, "time", Clock())

    polls = {"count": 0}

    class Process:
        pid = 10

        def poll(self):
            polls["count"] += 1
            return 0 if polls["count"] >= 40 else None

        def wait(self, **_):
            return 0

    cap = owned.OwnedGroup.capture(
        10, parent_pid=1, census=world.census, samples=world.samples,
        identity=world.identity, scheme=lambda: "fixture", clock=world.clock,
        signal_group=world.send)
    monkeypatch.setattr(wrapper, "owned_process_group",
                        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict))
    monkeypatch.setattr(wrapper, "subprocess",
                        NS(Popen=lambda *a, **k: Process(), TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(wrapper, "_memory_budget",
                        lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture"))

    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=io.StringIO()) == 0
    assert polls["count"] == 40
    # 40 polls: the per-poll sweep produced two censuses for each of them (>= 80).
    # A cadence keeps the count well under the poll count, and the exact number
    # is deliberately not pinned (capture and the final cleanup run their own).
    assert world.censuses < polls["count"], (world.censuses, polls["count"])
