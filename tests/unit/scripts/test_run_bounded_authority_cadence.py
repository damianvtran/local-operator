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
        self.sample_calls = 0

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        self.censuses += 1
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        self.sample_calls += 1
        corpse = self.sample_calls > 2
        return {
            pid: owned.procstate.ProcessSample(corpse and pid == 10, self.rows[pid][3])
            for pid in pids
            if pid in self.rows
        }

    def identity(self, pid):
        return self.rows[pid][:3]

    def send(self, pgid, _signum):
        self.rows.pop(pgid, None)


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return module


def _refresh_points(wrapper, monkeypatch, constant):
    """Poll counts at which the loop refreshed authority, for a given constant."""
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
        10,
        deadline=world.clock() + 5.0,
        parent_pid=1,
        census=world.census,
        samples=world.samples,
        identity=world.identity,
        scheme=lambda: "fixture",
        clock=world.clock,
        signal_group=world.send,
    )
    points = []
    original = cap.refresh

    def counting(**kwargs):
        points.append(polls["count"])
        return original(**kwargs)

    cap.refresh = counting
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(Popen=lambda *a, **k: Process(), TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )
    monkeypatch.setattr(wrapper, "_AUTHORITY_REFRESH_SECONDS", constant)
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=io.StringIO()) == 0
    return points, polls["count"]


def test_a_constant_longer_than_the_run_means_no_in_loop_refresh(wrapper, monkeypatch):
    """Pin the constant: the loop must not refresh once per poll.

    With a cadence longer than the whole run, every refresh that happens belongs
    to capture or final cleanup — none may occur while the loop is still polling.
    """
    points, polls = _refresh_points(wrapper, monkeypatch, 3_600.0)
    assert polls == 40
    early = [point for point in points if point < 39]
    assert early == [], early


def test_the_constant_is_what_drives_the_cadence(wrapper, monkeypatch):
    """The same run with a zero cadence refreshes immediately: the discriminator.

    This is the mutation control for the first test: if the loop ignored the
    constant, both cases would look alike and only one of the two can pass.
    """
    points, _polls = _refresh_points(wrapper, monkeypatch, 0.0)
    assert [point for point in points if point < 5], points
