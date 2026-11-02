"""A3: authority is re-proved on a cadence, not on every poll.

One refresh runs two complete strict censuses. Doing that per 50 ms poll was ~40
``/proc`` scans a second on Linux, where ordinary fork churn then read as lost
authority and aborted healthy gates. Signals still re-prove inside ``signal()``;
this pins the *between-signals* cadence so a future edit cannot quietly restore
the per-poll sweep.
"""

from __future__ import annotations

import io
import os
import subprocess
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401


class World:
    def __init__(self):
        # THE LEADER'S PARENT IS THIS PROCESS. With a creator row at pid 1, capture
        # samples a parent that does not exist, refuses "leader/parent birth
        # unavailable" at admission, and every path reports 126 before it can
        # produce its own state -- the same fixture class proved in the authority file.
        self.rows = {
            os.getpid(): (0, os.getpid(), os.getpid(), "creator"),
            10: (os.getpid(), 10, 10, "leader"),
        }
        self.now = 0.0
        self.censuses = 0
        self.sample_calls = 0
        #: WHEN the faked child exits. An event on the fake clock, never a count of
        #: how many times the sampler was asked: a call count couples the fixture to
        #: the implementation's call sequence.
        self.exit_at = 0.05

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        self.censuses += 1
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        self.sample_calls += 1
        return {
            pid: owned.procstate.ProcessSample(
                pid == 10 and self.now >= self.exit_at, self.rows[pid][3]
            )
            for pid in pids
            if pid in self.rows
        }

    def identity(self, pid):
        return self.rows[pid][:3]

    def send(self, pgid, _signum):
        self.rows.pop(pgid, None)


@pytest.fixture(autouse=True)
def _fixture_birth_scheme(monkeypatch):
    """The synthetic kernel speaks the ``fixture`` birth scheme.

    Without this the receipt and the sampled token are compared under the HOST
    scheme, the receipt is refused, and every later refresh reports "no leader
    receipt" -- published as an unproven cleanup (126) for a healthy run.
    """
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return module


def _refresh_points(wrapper, monkeypatch, constant):
    """Poll counts at which the loop refreshed authority, for a given constant."""
    world = World()

    ticks = {"n": 0}

    class Clock:
        def monotonic(self):
            return world.clock()

        def sleep(self, seconds):
            world.now += seconds
            ticks["n"] += 1  # one loop iteration, now that no poll drives it

    monkeypatch.setattr(wrapper, "time", Clock())

    class Process:
        pid = 10

        def poll(self):
            # Never consulted before cleanup: the corpse is left unreaped.
            return 0

        def wait(self, **_):
            return 0

    cap = owned.OwnedGroup.capture(
        10,
        deadline=world.clock() + 5.0,
        parent_pid=os.getpid(),
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
        points.append(ticks["n"])
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
    return points, ticks["n"]


def test_a_constant_longer_than_the_run_means_no_in_loop_refresh(wrapper, monkeypatch):
    """Pin the constant: the loop must not refresh once per iteration.

    The corpse is already pending when the loop starts, so the loop's whole life
    is a single tick. With a cadence longer than that run, no refresh may be
    attributed to tick 0 — every refresh in the list belongs to capture or to the
    final cleanup, which run outside the loop.
    """
    points, iterations = _refresh_points(wrapper, monkeypatch, 3_600.0)
    assert iterations >= 1
    assert 0 not in points, points


def test_the_constant_is_what_drives_the_cadence(wrapper, monkeypatch):
    """The discriminator: a zero cadence refreshes on the loop's FIRST tick."""
    points, _iterations = _refresh_points(wrapper, monkeypatch, 0.0)
    assert 0 in points, points
