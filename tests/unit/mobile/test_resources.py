"""Resource-usage probing for `lop sessions`, with an injected subprocess.

The whole point of the helper is graceful degradation: a probe that fails, a
pid that vanished, or an unparseable line must yield ``None`` for that number
and never fail the listing. These drive the pure parser + the injected runner
seam so no real ``ps``/``top`` is spawned, and they pin the direct footprint
probe too — a synthetic pid is not a reading of THIS host, so leaving that seam
live would make each of these a different test depending on which machine runs
it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from local_operator.mobile import resources
from local_operator.mobile.resources import (
    ResourceUsage,
    _parse_mem_size,
    session_resource_usage,
)


def _no_direct_probe(pid: int) -> int | None:
    """The direct footprint probe pinned to "this host cannot answer".

    Exactly what a host without ``libproc`` does, so one injection covers both
    "the mechanism is absent" and "the mechanism exists but no fixture pid is
    readable" — and the fallback under test is the same in both.
    """
    return None


def test_parses_batched_ps_rss() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            # `ps -o pid=,rss=` output: pid then rss in KiB.
            return 0, "111 2048\n222 4096\n"
        return 1, ""  # top/footprint unavailable here

    usage = session_resource_usage([111, 222], runner=runner, footprint_probe=_no_direct_probe)
    assert usage[111].rss_bytes == 2048 * 1024
    assert usage[222].rss_bytes == 4096 * 1024
    # Footprint degrades to None when the platform probe returns nothing.
    if sys.platform == "darwin":
        assert usage[111].footprint_bytes is None


def test_darwin_footprint_from_top(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, "111 1024\n"
        if argv[0] == "top":
            # `top -l1 -stats pid,mem`: a header block then PID / MEM rows.
            return 0, "PID    MEM\n111    197M\n999    5M\n"
        return 1, ""

    usage = session_resource_usage([111], runner=runner, footprint_probe=_no_direct_probe)
    assert usage[111].rss_bytes == 1024 * 1024
    assert usage[111].footprint_bytes == 197 * 1024 * 1024


def test_darwin_direct_probe_covers_the_pids_and_skips_top(monkeypatch) -> None:
    """The fast path: one direct read per pid, and NO `top` dump at all.

    The dump is the entire cost this change removes (2,344 ms against 16 ms for
    the batched `ps`), so "top was not run" is the assertion, not a timing
    bound: the runner raises if it is asked for anything but `ps`.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    asked: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        asked.append(argv[0])
        if argv[0] == "ps":
            return 0, "111 1024\n"
        raise AssertionError(f"top must not be spent when the direct read answers: {argv}")

    usage = session_resource_usage([111], runner=runner, footprint_probe=lambda pid: 206_467_072)
    assert usage[111].footprint_bytes == 206_467_072
    assert usage[111].rss_bytes == 1024 * 1024
    assert asked == ["ps"]


def test_darwin_default_probe_is_the_direct_read(monkeypatch) -> None:
    """Nothing injected: the MODULE's own probe must be the one that runs.

    The injected-probe tests above pin the dispatch; this one pins the wiring,
    which is the half a reader cannot see from them — replacing the default with
    the old ``top``-only path leaves those green and this red.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    asked: list[int] = []

    def spy(pid: int) -> int:
        asked.append(pid)
        return 206_467_072

    monkeypatch.setattr(resources, "_darwin_footprint_bytes", spy)

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, "111 1024\n"
        raise AssertionError(f"top must not be spent when the direct read answers: {argv}")

    usage = session_resource_usage([111], runner=runner)
    assert asked == [111]
    assert usage[111].footprint_bytes == 206_467_072


def test_darwin_top_covers_only_the_pids_the_direct_probe_missed(monkeypatch) -> None:
    """A partial answer: the residue pays for the dump, and only the residue.

    ``top`` reports BOTH pids here because that is what a real dump does, so the
    second assertion is the one that pins precedence: a value arriving for 111
    from the dump must not overwrite the direct read's exact number.
    """
    monkeypatch.setattr(sys, "platform", "darwin")

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, "111 1024\n222 2048\n"
        if argv[0] == "top":
            return 0, "PID    MEM\n111    999M\n222    333M\n"
        return 1, ""

    usage = session_resource_usage(
        [111, 222],
        runner=runner,
        footprint_probe=lambda pid: 206_467_072 if pid == 111 else None,
    )
    assert usage[111].footprint_bytes == 206_467_072
    assert usage[222].footprint_bytes == 333 * 1024 * 1024


def test_darwin_a_raising_direct_probe_degrades_to_top(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    def probe(pid: int) -> int | None:
        raise OSError("libproc refused this pid")

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, "111 1024\n"
        return 0, "PID    MEM\n111    197M\n"

    usage = session_resource_usage([111], runner=runner, footprint_probe=probe)
    assert usage[111].footprint_bytes == 197 * 1024 * 1024


def test_darwin_without_libproc_uses_the_top_fallback(monkeypatch) -> None:
    """The real default probe, on a host where loading libproc failed.

    ``_libproc`` is set to its False sentinel rather than the function being
    replaced, so the degradation under test is the module's own.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(resources, "_libproc", False)

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, "111 1024\n"
        return 0, "PID    MEM\n111    197M\n"

    usage = session_resource_usage([111], runner=runner)
    assert usage[111].footprint_bytes == 197 * 1024 * 1024


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is macOS-only")
def test_direct_probe_reads_this_process_footprint() -> None:
    """The real syscall, against a pid that MUST answer: this one."""
    value = resources._darwin_footprint_bytes(os.getpid())
    assert value is not None
    assert value > 1024 * 1024


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is macOS-only")
def test_direct_probe_returns_none_for_a_pid_that_is_gone() -> None:
    """A pid nothing can read: the call must report nothing, not raise.

    The child has been reaped, and pids are allocated monotonically, so this is
    not "probably gone" — the only way to get this number back is for it to have
    been recycled, which the kernel does not do on the next allocation.
    """
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert child.wait() == 0
    assert resources._darwin_footprint_bytes(child.pid) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is macOS-only")
def test_the_footprint_offset_is_the_one_the_module_reads() -> None:
    """The module hardcodes a byte offset into a kernel struct; pin the number.

    What this test can and cannot say is worth stating, because the tempting
    stronger version does not hold: ``ri_resident_size`` — the field immediately
    before the footprint — IS the same quantity ``ps`` RSS reports, but the two
    read the process at different instants, so comparing them in a test that is
    itself allocating memory fails on a moving target rather than on a wrong
    offset (measured: this pid and ``ps`` agreed exactly when stable and differed
    by 12% mid-collection). The comparison against an independent reader lives in
    ``scripts/bench_info_snapshot.py`` instead, where the pid under measurement is
    a parked fixture process and the two readers are sampled back to back.

    Here: the constant is 8 bytes past ``ri_resident_size`` (16-byte uuid plus
    seven uint64 counters, the v1/v2 layout), and the value the module returns is
    the value AT that offset. The ctypes read is duplicated from the module on
    purpose — calling ``_darwin_footprint_bytes`` to validate the offset it
    computes would assert nothing.
    """
    import ctypes

    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.restype = ctypes.c_int
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    pid = os.getpid()
    buffer = ctypes.create_string_buffer(2048)
    assert lib.proc_pid_rusage(pid, 2, ctypes.byref(buffer)) == 0

    assert resources._RUSAGE_PHYS_FOOTPRINT_OFFSET == 64 + 8
    assert resources._darwin_footprint_bytes(pid) == int.from_bytes(
        buffer.raw[
            resources._RUSAGE_PHYS_FOOTPRINT_OFFSET : resources._RUSAGE_PHYS_FOOTPRINT_OFFSET + 8
        ],
        "little",
    )


def test_degrades_to_none_when_every_probe_fails() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        return 1, ""  # ps and top both unavailable / non-zero

    usage = session_resource_usage([111, 222], runner=runner, footprint_probe=_no_direct_probe)
    # Every requested pid still gets an entry — a live session is never dropped.
    assert set(usage) == {111, 222}
    assert usage[111] == ResourceUsage(rss_bytes=None, footprint_bytes=None)


def test_runner_that_raises_is_swallowed() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        raise OSError("boom")

    # A raising runner must not propagate — the listing degrades, not crashes.
    # (session_resource_usage guards each call; a raising injected runner is
    # caught by the callers the same way a non-zero return is.)
    try:
        usage = session_resource_usage([111], runner=runner, footprint_probe=_no_direct_probe)
    except OSError:  # pragma: no cover - this is the failure we assert against
        raise AssertionError("a failing probe must not raise")
    assert usage[111].rss_bytes is None


def test_empty_pid_list_returns_empty() -> None:
    assert session_resource_usage([]) == {}


def test_parse_mem_size_units() -> None:
    assert _parse_mem_size("512B") == 512
    assert _parse_mem_size("2K") == 2048
    assert _parse_mem_size("3M") == 3 * 1024 * 1024
    assert _parse_mem_size("1G") == 1024**3
    assert _parse_mem_size("197M+") == 197 * 1024 * 1024  # top marks growth with '+'
    assert _parse_mem_size("1234") == 1234  # bare bytes
    assert _parse_mem_size("garbage") is None
