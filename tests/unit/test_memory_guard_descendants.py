"""Escaped-descendant accounting for the per-command memory guard.

WHY THESE EXIST. Two measured accounting defects (QA session e8581ec27fa2, rig
session a3c2eb44f316): (1) a child enumerator that divided
``proc_listchildpids``' PID COUNT by ``sizeof(int)`` as if it were bytes, so 3
children read as 0; (2) ``run_bounded --memory-mb 1536`` let a rig peak at
1,852 MB because the guard metered one process GROUP and the Electron app,
daemon and runtimes had ``setsid``'d out of it. These pin the byte conversion,
that an escaped descendant is charged, that an unreadable one fails CLOSED, and
the stop path's ownership re-check. Probes are injected; the real-process
negative control lives in ``tests/unit/scripts/test_run_bounded.py``.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

import pytest

from local_operator import memory_guard as mg
from local_operator.mobile import resources

_MIB = 1024 * 1024


def _budget(ceiling_mb: int) -> mg.Budget:
    return mg.Budget(ceiling_mb, int(ceiling_mb * 0.8), 4096, 16384, 2048, "auto", "test")


def _guard(ceiling_mb, *, group, footprints, census, monkeypatch, alive=True):
    # pid_liveness/process_sample are the gone-or-zombie discriminators; pin them
    # so a synthetic pid never consults the live process table.
    monkeypatch.setattr(mg.procstate, "pid_liveness", lambda pid: alive)
    monkeypatch.setattr(mg.procstate, "process_sample", lambda pid: None)
    return mg.Guard(
        100,
        _budget(ceiling_mb),
        runner=lambda argv: (1, ""),
        footprint_probe=lambda pid: footprints.get(pid),
        group_probe=lambda pgid: list(group),
        descendant_probe=census,
    )


# --- count vs bytes ----------------------------------------------------------


class _BytesLib:
    """``proc_listpids`` as xnu's libproc.c defines it: returns BYTES written."""

    def __init__(self, pids):
        self.pids = pids

    def proc_listpids(self, kind, arg, buf, size):
        assert kind == resources._PROC_PPID_ONLY
        nbytes = len(self.pids) * ctypes.sizeof(ctypes.c_int)
        if buf is None:
            return nbytes
        for i, pid in enumerate(self.pids):
            buf[i] = pid
        return nbytes


def test_child_listing_converts_proc_listpids_bytes_to_pids_exactly_once(monkeypatch):
    """12 bytes is 3 pids. The QA rig's double division read the same 3 as 0."""
    monkeypatch.setattr(resources, "_darwin_libproc", lambda: _BytesLib([501, 502, 503]))
    assert resources._darwin_child_pids(500) == [501, 502, 503]


def test_child_listing_rejects_a_non_int_multiple_byte_count(monkeypatch):
    lib = _BytesLib([501])
    lib.proc_listpids = lambda kind, arg, buf, size: 5  # not a multiple of 4
    monkeypatch.setattr(resources, "_darwin_libproc", lambda: lib)
    assert resources._darwin_child_pids(500) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="libproc is macOS-only")
def test_libproc_wrappers_disagree_by_exactly_sizeof_int():
    """The source claim, measured: proc_listpids(PPID_ONLY) is bytes,
    proc_listchildpids is that divided by sizeof(int)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lib = resources._darwin_libproc()
        assert lib is not None
        buf = (ctypes.c_int * 64)()
        nbytes = lib.proc_listpids(resources._PROC_PPID_ONLY, os.getpid(), buf, ctypes.sizeof(buf))
        count = lib.proc_listchildpids(os.getpid(), buf, ctypes.sizeof(buf))
        assert nbytes == count * ctypes.sizeof(ctypes.c_int) and count >= 1
        assert child.pid in (resources._darwin_child_pids(os.getpid()) or [])
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(not hasattr(os, "setsid"), reason="POSIX only")
def test_the_live_census_finds_a_setsid_grandchild_the_group_listing_misses():
    code = (
        "import os,time\n"
        "if os.fork()==0:\n os.setsid(); time.sleep(30); os._exit(0)\n"
        "time.sleep(30)"
    )
    leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    grandchild = None
    try:
        deadline = time.monotonic() + 10
        tree: dict[int, int] = {}
        while time.monotonic() < deadline:
            tree = resources.direct_descendant_pids([leader.pid]) or {}
            if tree:
                break
            time.sleep(0.05)
        assert len(tree) == 1
        grandchild, parent = next(iter(tree.items()))
        assert parent == leader.pid
        assert os.getpgid(grandchild) != leader.pid  # it really left the group
        assert grandchild not in (resources.direct_group_pids(leader.pid) or [])
    finally:
        if grandchild:
            try:
                os.kill(grandchild, 9)
            except OSError:
                pass
        leader.kill()
        leader.wait()


# --- the guard ---------------------------------------------------------------


def test_a_setsid_descendant_over_budget_is_charged_and_killed(monkeypatch):
    """Group members 100+101 hold 10 MB; their setsid'd child 900 holds 400 MB."""
    footprints = {100: 5 * _MIB, 101: 5 * _MIB, 900: 400 * _MIB}
    guard = _guard(
        300,
        group=[100, 101],
        footprints=footprints,
        census=lambda roots: {101: 100, 900: 101},
        monkeypatch=monkeypatch,
    )
    sample = guard.sample_sync()
    assert sample.escaped == {900: 101}
    assert sample.bytes_used == 410 * _MIB
    assert guard.should_kill(sample) is True


def test_the_census_is_off_unless_asked_for(monkeypatch):
    """The bash tool's guard keeps its contract; only run_bounded opts in."""
    guard = mg.Guard(
        100,
        _budget(300),
        runner=lambda argv: (1, ""),
        footprint_probe={100: 5 * _MIB, 900: 400 * _MIB}.get,
        group_probe=lambda pgid: [100],
    )
    sample = guard.sample_sync()
    assert sample.escaped == {} and sample.bytes_used == 5 * _MIB
    assert guard.should_kill(sample) is False


def test_an_unreadable_live_descendant_fails_closed_after_the_streak(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
    )
    decisions = []
    sample = guard.sample_sync()
    for tick in range(mg.UNACCOUNTED_TICK_LIMIT):
        if tick:
            sample = guard.sample_sync()
        assert sample.unaccounted and "900" in sample.unaccounted
        decisions.append(guard.should_kill(sample))
    # Never read as zero: below the streak it is reported, at the streak it kills.
    assert decisions == [False] * (mg.UNACCOUNTED_TICK_LIMIT - 1) + [True]
    assert "fail closed" in mg.unaccounted_line(sample)


def test_an_unanswerable_census_fails_closed_too(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: None,
        monkeypatch=monkeypatch,
    )
    samples = [guard.sample_sync() for _ in range(mg.UNACCOUNTED_TICK_LIMIT)]
    assert samples[-1].unaccounted == "descendant census could not answer"
    assert guard.should_kill(samples[-1]) is True


def test_a_descendant_that_exited_mid_read_is_not_unaccounted(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
        alive=False,
    )
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT + 2):
        sample = guard.sample_sync()
        assert sample.unaccounted is None and guard.should_kill(sample) is False


def test_a_readable_streak_resets(monkeypatch):
    footprints: dict[int, int | None] = {100: 5 * _MIB}
    guard = _guard(
        300,
        group=[100],
        footprints=footprints,
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
    )
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT - 1):
        guard.sample_sync()
    footprints[900] = 1 * _MIB
    sample = guard.sample_sync()
    assert sample.unaccounted is None and sample.unaccounted_streak == 0
    assert guard.should_kill(sample) is False


def test_stop_escaped_signals_only_pids_whose_parent_edge_reverifies(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 1, 900: 400 * _MIB, 901: 1, 902: 1},
        census=lambda roots: {900: 100, 901: 900, 902: 100},
        monkeypatch=monkeypatch,
    )
    guard.sample_sync()
    rows = {900: (100, 900), 901: (900, 900), 902: (4242, 902)}  # 902 recycled
    monkeypatch.setattr(mg, "direct_ppid_pgid", rows.get)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert guard.stop_escaped(9) == [900, 901]
    assert sent == [(900, 9), (901, 9)]
