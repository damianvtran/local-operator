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


def _guard(ceiling_mb, *, group, footprints, census, monkeypatch, alive=True, denied=(), ps=None):
    # pid_liveness/process_sample are the gone-or-zombie discriminators; pin them
    # so a synthetic pid never consults the live process table.
    monkeypatch.setattr(mg.procstate, "pid_liveness", lambda pid: alive)
    monkeypatch.setattr(mg.procstate, "process_sample", lambda pid: None)
    # The stop path's second parentage source reads the live kernel; a synthetic
    # tree must never consult it about a real pid of the same number.
    # raising=False: the same test file must run (and fail) against a base
    # that predates the second source.
    monkeypatch.setattr(mg, "listed_ppid_pgid", lambda pid, parent: None, raising=False)
    guard = mg.Guard(
        100,
        _budget(ceiling_mb),
        runner=lambda argv: (1, ""),
        footprint_probe=lambda pid: footprints.get(pid),
        group_probe=lambda pgid: list(group),
        descendant_probe=census,
        permission_probe=lambda pid: pid in denied,
    )
    if ps is not None:
        guard.runner = lambda argv: _batched_ps(ps, argv)
    return guard


def _batched_ps(ps, argv):
    """Serve ``ps -o pid=,rss= -p a,b`` from a per-pid ``ps(pid) -> (code, rss)``."""
    rows = []
    for token in argv[-1].split(","):
        code, out = ps(int(token))
        if code == 0 and out.strip():
            rows.append(f"{token} {out.strip()}")
    return (0 if rows else 1), "\n".join(rows)


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
    assert guard.stop_escaped(9) == [901, 900]
    # Deepest first (901 is 900's child), each edge re-read before its signal.
    assert sent == [(901, 9), (900, 9)]


# --- review R1/R2/R3 (PR #3369) ---------------------------------------------


def test_a_not_permitted_descendant_is_charged_by_ps_and_never_fails_closed(monkeypatch):
    """A setuid ``top`` (EPERM) is a credential boundary, not an unknown."""
    guard = _guard(
        1536,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
        denied={900},
        ps=lambda pid: (0, "14336\n"),
    )
    sample = guard.sample_sync()
    for tick in range(mg.UNACCOUNTED_TICK_LIMIT * 3):
        if tick:
            sample = guard.sample_sync()
        assert sample.unaccounted is None and guard.should_kill(sample) is False
    assert sample.not_permitted == {900: 14 * _MIB}
    assert sample.bytes_used == 19 * _MIB
    assert guard.new_not_permitted(sample) == [900]
    assert guard.new_not_permitted(sample) == []  # disclosed ONCE
    assert "900 (ps rss 14 MiB, charged)" in mg.not_permitted_line(sample, [900])


def test_a_not_permitted_descendant_ps_cannot_size_is_disclosed_uncharged(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
        denied={900},
        ps=lambda pid: (1, ""),
    )
    sample = guard.sample_sync()
    for tick in range(mg.UNACCOUNTED_TICK_LIMIT * 2):
        if tick:
            sample = guard.sample_sync()
        assert guard.should_kill(sample) is False
    assert sample.not_permitted == {900: None}
    assert "900 (uncharged)" in mg.not_permitted_line(sample, [900])


def test_a_churn_of_short_lived_unreadable_pids_does_not_fail_closed(monkeypatch):
    """Each tick a FRESH unreadable pid (a ps/top call per second): no single pid
    is persistently unknown, so the per-pid count never reaches the limit."""
    tick = {"n": 0}

    def census(roots):
        tick["n"] += 1
        return {1000 + tick["n"]: 100}

    guard = _guard(
        300, group=[100], footprints={100: 5 * _MIB}, census=census, monkeypatch=monkeypatch
    )
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT * 3):
        sample = guard.sample_sync()
        assert sample.unaccounted is not None and sample.unaccounted_streak == 1
        assert guard.should_kill(sample) is False


def test_a_pid_that_disappears_restarts_its_count(monkeypatch):
    present = {"on": True}
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100} if present["on"] else {},
        monkeypatch=monkeypatch,
    )
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT - 1):
        guard.sample_sync()
    present["on"] = False
    guard.sample_sync()
    present["on"] = True
    sample = guard.sample_sync()
    assert sample.unaccounted_streak == 1 and guard.should_kill(sample) is False


def test_the_headline_names_escaped_memory_and_fail_closed_stops(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB, 900: 400 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
    )
    sample = guard.sample_sync()
    text = guard.over_budget_message(sample, unit="MiB")
    assert "escaped descendant(s)" in text and "400 MiB of it" in text
    assert "this command's process group reached" not in text

    closed = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {901: 100},
        monkeypatch=monkeypatch,
    )
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT):
        sample = closed.sample_sync()
    assert closed.should_kill(sample)
    text = closed.over_budget_message(sample, unit="MiB")
    assert "could not be measured" in text and "fail closed" in text
    assert "process group reached" not in text


def test_the_bash_tool_headline_is_unchanged_without_the_census() -> None:
    guard = mg.Guard(
        100,
        _budget(100),
        runner=lambda argv: (1, ""),
        footprint_probe=lambda pid: 400 * _MIB,
        group_probe=lambda g: [100],
    )
    sample = guard.sample_sync()
    assert guard.over_budget_message(sample).startswith(
        "MEMORY LIMIT EXCEEDED: this command's process group reached 0.4 GB"
    )


def test_a_verified_descendant_that_refuses_the_signal_is_reported(monkeypatch):
    guard = _guard(
        300,
        group=[100],
        footprints={100: 1, 900: 400 * _MIB, 901: 1},
        census=lambda roots: {900: 100, 901: 100},
        monkeypatch=monkeypatch,
    )
    guard.sample_sync()
    monkeypatch.setattr(mg, "direct_ppid_pgid", {900: (100, 900), 901: (100, 901)}.get)

    def kill(pid, sig):
        if pid == 901:
            raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(mg.os, "kill", kill)
    assert guard.stop_escaped(9) == [900]
    assert guard.unsignalled == [901]
    assert "901" in mg.unsignalled_line(guard.unsignalled)


@pytest.mark.skipif(sys.platform != "darwin", reason="libproc EPERM is macOS-specific")
def test_the_live_permission_probe_tells_setuid_top_from_a_gone_pid():
    top = subprocess.Popen(["/usr/bin/top", "-l", "3", "-s", "1"], stdout=subprocess.DEVNULL)
    try:
        time.sleep(0.3)
        assert resources.direct_footprint_bytes(top.pid) is None
        assert resources.footprint_denied(top.pid) is True
        assert resources.footprint_denied(os.getpid()) is False
    finally:
        top.kill()
        top.wait()
    assert resources.footprint_denied(top.pid) is False  # gone (ESRCH), not denied


# --- follow-up to PR #3369: QA Q2, review M1/M2/N1/N2 -------------------------


def _breached(monkeypatch, census, footprints=None, **kwargs):
    guard = _guard(
        300,
        group=[100],
        footprints=footprints or {100: 400 * _MIB},
        census=census,
        monkeypatch=monkeypatch,
        **kwargs,
    )
    guard.sample_sync()
    return guard


def test_a_setuid_escaped_pid_is_verified_by_the_parent_listing_and_killed(monkeypatch):
    """Q2: the fork-free self read refuses a setuid pid; the parent's child
    listing still proves the edge, so it is signalled rather than skipped."""
    guard = _breached(monkeypatch, lambda roots: {900: 100}, denied={900}, ps=lambda p: (0, "1"))
    monkeypatch.setattr(mg, "direct_ppid_pgid", lambda pid: None)
    monkeypatch.setattr(
        mg,
        "listed_ppid_pgid",
        lambda pid, parent: (parent, 900) if pid == 900 else None,
        raising=False,
    )
    sent: list[int] = []
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: sent.append(pid))
    assert guard.stop_escaped(9) == [900]
    assert sent == [900] and guard.unverified == []


def test_an_escaped_pid_no_source_can_verify_is_named_not_signalled(monkeypatch):
    guard = _breached(monkeypatch, lambda roots: {900: 100})
    monkeypatch.setattr(mg, "direct_ppid_pgid", lambda pid: None)
    sent: list[int] = []
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: sent.append(pid))
    assert guard.stop_escaped(9) == []
    assert sent == [] and guard.unverified == [900]
    assert "900" in mg.unverified_line(guard.unverified)
    assert "not signalled" in mg.unverified_line(guard.unverified)


def test_an_unverifiable_pid_that_is_gone_is_not_named(monkeypatch):
    guard = _breached(monkeypatch, lambda roots: {900: 100}, alive=False)
    monkeypatch.setattr(mg, "direct_ppid_pgid", lambda pid: None)
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: pytest.fail("signalled"))
    assert guard.stop_escaped(9) == [] and guard.unverified == []


def test_a_recycled_pid_with_a_readable_stranger_parent_stays_silent(monkeypatch):
    """A pid whose parent READS as someone else is proven not ours: skipped and
    not named, and the second source is never asked to overrule it."""
    guard = _breached(monkeypatch, lambda roots: {900: 100})
    monkeypatch.setattr(mg, "direct_ppid_pgid", lambda pid: (4242, 900))
    monkeypatch.setattr(
        mg, "listed_ppid_pgid", lambda pid, parent: pytest.fail("asked"), raising=False
    )
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: pytest.fail("signalled"))
    assert guard.stop_escaped(9) == [] and guard.unverified == []


def test_the_edge_is_rechecked_immediately_before_each_kill(monkeypatch):
    """Deepest first, each edge re-read just before its signal: a pid recycled
    between verification and signal (its parent now reads differently) is
    skipped, and killing 901 first never strands its parent's proof."""
    guard = _breached(monkeypatch, lambda roots: {900: 100, 901: 900, 902: 100})
    reads = {"n": 0}
    rows = {900: (100, 900), 901: (900, 900)}

    def read(pid):
        if pid == 902:
            reads["n"] += 1
            return (100, 902) if reads["n"] == 1 else (4242, 902)  # recycled mid-stop
        return rows.get(pid)

    monkeypatch.setattr(mg, "direct_ppid_pgid", read)
    sent: list[int] = []
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: sent.append(pid))
    assert guard.stop_escaped(9) == [901, 900]
    assert sent == [901, 900] and reads["n"] == 2 and guard.unverified == []


def test_not_permitted_pids_share_one_batched_ps_per_tick(monkeypatch):
    """M2: N denied pids cost one ``ps`` fork per read, not N."""
    calls: list[list[str]] = []
    guard = _guard(
        1536,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100, 901: 100, 902: 100},
        monkeypatch=monkeypatch,
        denied={900, 901, 902},
        ps=lambda pid: (0, "2048"),
    )
    served = guard.runner
    guard.runner = lambda argv: (calls.append(argv), served(argv))[1]
    sample = guard.sample_sync()
    assert calls == [["ps", "-o", "pid=,rss=", "-p", "900,901,902"]]
    assert sample.not_permitted == {900: 2 * _MIB, 901: 2 * _MIB, 902: 2 * _MIB}
    guard.sample_sync()
    assert len(calls) == 1  # cached inside the TTL


def test_a_not_permitted_pid_ps_never_sizes_fails_closed_after_the_read_limit(monkeypatch):
    """M1: a denied pid ``ps`` cannot size is disclosed, then after
    NOT_PERMITTED_UNSIZED_READ_LIMIT consecutive failed reads it counts as
    unaccounted memory and the run fails closed."""
    monkeypatch.setattr(mg, "_NOT_PERMITTED_RSS_TTL_S", 0.0)  # every tick reads
    monkeypatch.setattr(mg, "NOT_PERMITTED_UNSIZED_MIN_SECONDS", 0.0)  # m3 floor: own test
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
        denied={900},
        ps=lambda pid: (1, ""),
    )
    reads = mg.NOT_PERMITTED_UNSIZED_READ_LIMIT
    for _ in range(reads - 1):
        sample = guard.sample_sync()
        assert sample.unaccounted is None and guard.should_kill(sample) is False
    killed = False
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT):
        sample = guard.sample_sync()
        assert "900" in (sample.unaccounted or "")
        killed = guard.should_kill(sample)
    assert killed
    assert guard.new_not_permitted(sample) == [900]  # still disclosed once


def test_a_sized_read_resets_the_unsized_streak(monkeypatch):
    monkeypatch.setattr(mg, "_NOT_PERMITTED_RSS_TTL_S", 0.0)
    answers = iter([(1, ""), (1, ""), (0, "1024")] * 10)
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
        denied={900},
        ps=lambda pid: next(answers),
    )
    for _ in range(12):
        sample = guard.sample_sync()
        assert sample.unaccounted is None and guard.should_kill(sample) is False


def test_the_fail_closed_headline_omits_measured_when_nothing_was(monkeypatch):
    """N1: with no reading at all ``reached`` is the ceiling, not a measurement."""
    guard = _guard(
        300,
        group=[100],
        footprints={},
        census=lambda roots: {901: 100},
        monkeypatch=monkeypatch,
    )
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT):
        sample = guard.sample_sync()
    assert guard.should_kill(sample) and sample.bytes_used is None
    text = guard.over_budget_message(sample, unit="MiB")
    assert "treated as over budget (fail closed), over the 300 MiB budget" in text
    assert "measured," not in text


def test_the_escaped_headline_says_left_the_process_group(monkeypatch):
    """N2: setpgid escapes the group exactly as setsid does."""
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB, 900: 400 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
    )
    text = guard.over_budget_message(guard.sample_sync(), unit="MiB")
    assert "(left the process group; 400 MiB of it)" in text and "setsid" not in text


@pytest.mark.skipif(sys.platform != "darwin", reason="setuid top EPERM is macOS-specific")
def test_the_live_parent_listing_verifies_a_setuid_child_the_self_read_refuses():
    top = subprocess.Popen(
        ["/usr/bin/top", "-l", "0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while resources.footprint_denied(top.pid) is not True and time.monotonic() < deadline:
            time.sleep(0.05)  # until the exec into setuid top has happened
        assert resources.direct_ppid_pgid(top.pid) is None  # the refusal Q2 hit
        assert resources.listed_ppid_pgid(top.pid, os.getpid()) == (os.getpid(), top.pid)
        assert resources.listed_ppid_pgid(top.pid, 1) is None  # wrong parent
    finally:
        top.kill()
        top.wait()
    assert resources.listed_ppid_pgid(top.pid, os.getpid()) is None  # gone


# --- PR #3372 review round 1: m1/m2/m3 ----------------------------------------


def test_a_pid_now_under_a_different_tree_member_is_skipped(monkeypatch):
    """m1: the bar is equality with the RECORDED parent, not tree membership.
    901 was recorded under 900; it now reads under 902, also in our tree. That
    is not the process we recorded, so it is never signalled."""
    guard = _breached(monkeypatch, lambda roots: {900: 100, 901: 900, 902: 100})
    rows = {900: (100, 900), 901: (902, 901), 902: (100, 902)}
    monkeypatch.setattr(mg, "direct_ppid_pgid", rows.get)
    sent: list[int] = []
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: sent.append(pid))
    assert 901 not in guard.stop_escaped(9)
    assert 901 not in sent and sorted(sent) == [900, 902]


def test_a_pid_orphaned_mid_stop_is_named_not_signalled(monkeypatch):
    """m2: 901's parent 900 exits on its own between verification and 901's
    re-check, so 901 re-parents to pid 1. It was ours a tick ago: name it, but
    never widen kill authority to a pid-1 child."""
    guard = _breached(monkeypatch, lambda roots: {900: 100, 901: 900})
    reads = {"n": 0}

    def read(pid):
        if pid == 901:
            reads["n"] += 1
            return (900, 901) if reads["n"] == 1 else (1, 901)
        return (100, 900)

    monkeypatch.setattr(mg, "direct_ppid_pgid", read)
    sent: list[int] = []
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: sent.append(pid))
    assert guard.stop_escaped(9) == [900]
    assert 901 not in sent and guard.unverified == [901]


def test_an_already_orphaned_pid_is_named_at_verification(monkeypatch):
    guard = _breached(monkeypatch, lambda roots: {900: 100})
    monkeypatch.setattr(mg, "direct_ppid_pgid", lambda pid: (1, 900))
    monkeypatch.setattr(mg.os, "kill", lambda pid, sig: pytest.fail("signalled"))
    assert guard.stop_escaped(9) == [] and guard.unverified == [900]


def test_unsized_reads_inside_the_time_floor_never_fail_closed(monkeypatch):
    """m3: a burst of failed ``ps`` reads (a stalled ps on a busy host) must not
    fail closed until the streak has also lasted
    NOT_PERMITTED_UNSIZED_MIN_SECONDS. Here the clock moves 0.25 s per tick."""
    monkeypatch.setattr(mg, "_NOT_PERMITTED_RSS_TTL_S", 0.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr(mg.time, "monotonic", lambda: clock["t"])
    guard = _guard(
        300,
        group=[100],
        footprints={100: 5 * _MIB},
        census=lambda roots: {900: 100},
        monkeypatch=monkeypatch,
        denied={900},
        ps=lambda pid: (1, ""),
    )
    floor_ticks = int(mg.NOT_PERMITTED_UNSIZED_MIN_SECONDS / 0.25)
    for _ in range(floor_ticks - 1):
        sample = guard.sample_sync()
        assert sample.unaccounted is None and guard.should_kill(sample) is False
        clock["t"] += 0.25
    killed = False
    for _ in range(mg.UNACCOUNTED_TICK_LIMIT + 1):
        clock["t"] += 0.25
        sample = guard.sample_sync()
        killed = killed or guard.should_kill(sample)
    assert killed  # a sustained failure past the floor still fails closed
