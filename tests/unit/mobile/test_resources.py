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


def _live_pid() -> int:
    """This test process's own pid: a pid that certainly exists, and is ours.

    Not a small literal. The macOS fallback is now gated on whether the pid
    still EXISTS (a gone pid earns no dump — see ``_pid_exists``), so a fixture
    pid like ``111`` reaches the dump only on the machines that happen to have
    one, which is how a unit test becomes a reading of the host it runs on.
    """
    return os.getpid()


def _reaped_pid() -> int:
    """A pid that really is gone: spawned, reaped, and not reused in-test."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert child.wait() == 0
    return child.pid


def test_darwin_footprint_from_top(monkeypatch) -> None:
    """The fallback still answers: a pid with no direct number gets top's MEM."""
    monkeypatch.setattr(sys, "platform", "darwin")
    pid = _live_pid()

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, f"{pid} 1024\n"
        if argv[0] == "top":
            # `top -l1 -stats pid,mem`: a header block then PID / MEM rows.
            return 0, f"PID    MEM\n{pid}    197M\n999    5M\n"
        return 1, ""

    usage = session_resource_usage([pid], runner=runner, footprint_probe=_no_direct_probe)
    assert usage[pid].rss_bytes == 1024 * 1024
    assert usage[pid].footprint_bytes == 197 * 1024 * 1024


def test_darwin_direct_probe_covers_the_pids_and_skips_top(monkeypatch) -> None:
    """The fast path: one direct read per pid, and NO `top` dump at all.

    The dump is the entire cost this change removes, so "top was not run" is the
    assertion rather than a timing bound: the runner raises if it is asked for
    anything but `ps`.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    pid = _live_pid()
    asked: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        asked.append(argv[0])
        if argv[0] == "ps":
            return 0, f"{pid} 1024\n"
        raise AssertionError(f"top must not be spent when the direct read answers: {argv}")

    usage = session_resource_usage([pid], runner=runner, footprint_probe=lambda _pid: 206_467_072)
    assert usage[pid].footprint_bytes == 206_467_072
    assert usage[pid].rss_bytes == 1024 * 1024
    assert asked == ["ps"]


def test_darwin_default_probe_is_the_direct_read(monkeypatch) -> None:
    """Nothing injected: the MODULE's own probe must be the one that runs.

    The injected-probe tests above pin the dispatch; this one pins the wiring,
    which is the half a reader cannot see from them — replacing the default with
    the old ``top``-only path leaves those green and this red.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    pid = _live_pid()
    asked: list[int] = []

    def spy(target: int) -> int:
        asked.append(target)
        return 206_467_072

    monkeypatch.setattr(resources, "_darwin_footprint_bytes", spy)

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, f"{pid} 1024\n"
        raise AssertionError(f"top must not be spent when the direct read answers: {argv}")

    usage = session_resource_usage([pid], runner=runner)
    assert asked == [pid]
    assert usage[pid].footprint_bytes == 206_467_072


def test_darwin_top_covers_only_the_pids_the_direct_probe_missed(monkeypatch) -> None:
    """A partial answer: the residue pays for the dump, and only the residue.

    ``top`` reports BOTH pids here because that is what a real dump does, so the
    second assertion is the one that pins precedence: a value arriving for the
    answered pid from the dump must not overwrite the direct read's exact number.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    answered = _live_pid()
    missed = os.getppid()

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, f"{answered} 1024\n{missed} 2048\n"
        if argv[0] == "top":
            return 0, f"PID    MEM\n{answered}    999M\n{missed}    333M\n"
        return 1, ""

    usage = session_resource_usage(
        [answered, missed],
        runner=runner,
        footprint_probe=lambda pid: 206_467_072 if pid == answered else None,
    )
    assert usage[answered].footprint_bytes == 206_467_072
    assert usage[missed].footprint_bytes == 333 * 1024 * 1024


def test_darwin_a_gone_pid_costs_no_dump_and_keeps_its_neighbour(monkeypatch) -> None:
    """R1: the dump's trigger is a MISSING READER, not an unknown pid.

    A reaped pid is one no reader can answer — the system-wide dump would spend
    its whole sampling interval (5 s to 17 s measured here, past the runner's own
    5 s timeout) to return nothing. It is not an exotic input either: the caller
    hands this function every record the registry classified ``live``, so a
    session dying between that scan and this read lands right here.

    The dump's fixture LISTS the gone pid on purpose: an assertion that it stays
    ``None`` proves the pid was excluded from the filter, not merely that the
    dump had no row for it.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    live = _live_pid()
    gone = _reaped_pid()
    asked: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        asked.append(argv[0])
        if argv[0] == "ps":
            return 0, f"{live} 1024\n{gone} 2048\n"
        if argv[0] == "top":
            return 0, f"PID    MEM\n{live}    197M\n{gone}    999M\n"
        return 1, ""

    usage = session_resource_usage([live, gone], runner=runner, footprint_probe=_no_direct_probe)
    assert usage[live].footprint_bytes == 197 * 1024 * 1024
    assert usage[gone].footprint_bytes is None
    # And the dump ran at all only because of the live pid — one call, for the
    # reader that could answer.
    assert asked == ["ps", "top"]


def test_darwin_a_gone_pid_alone_never_reaches_the_dump(monkeypatch) -> None:
    """One reaped pid and nothing else: no dump is spent at all.

    The sharper half of the test above, and the assertion is on the CALL, not on
    the outcome: the runner is allowed to answer `top` (so the module's own
    "any probe failure is missing data" rule cannot hide the regression) and the
    call log is what says the dump was never asked for. A fixture that merely
    refused `top` would pass under both behaviours, because the refusal is caught
    and degrades to no data exactly like a slow dump does.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    gone = _reaped_pid()
    asked: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        asked.append(argv[0])
        if argv[0] == "ps":
            return 0, f"{gone} 2048\n"
        return 0, f"PID    MEM\n{gone}    999M\n"

    usage = session_resource_usage([gone], runner=runner, footprint_probe=_no_direct_probe)
    assert usage[gone].rss_bytes == 2048 * 1024
    assert usage[gone].footprint_bytes is None
    assert asked == ["ps"], "a gone pid must not put the whole-system dump on the path"


def test_darwin_a_zero_footprint_is_not_a_reading(monkeypatch) -> None:
    """R2: the kernel's zero is the unknown sentinel here, and buys no dump.

    ``ri_phys_footprint`` is 0 for a zombie (measured), so a successful call can
    still carry no memory figure. ``None`` is what the panel renders as the
    unknown sentinel; ``0`` would print "0 MB" for a process whose memory could
    not be read, which ``info/model.py`` names as the thing not to do. A zero is
    also not a reason to spend the dump: the pid that answers this way is a
    zombie, i.e. the gone case above, and no reader can give it a figure.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    pid = _live_pid()
    asked: list[str] = []

    def runner(argv: list[str]) -> tuple[int, str]:
        asked.append(argv[0])
        if argv[0] == "ps":
            return 0, f"{pid} 1024\n"
        return 0, f"PID    MEM\n{pid}    197M\n"

    usage = session_resource_usage([pid], runner=runner, footprint_probe=lambda _pid: 0)
    assert usage[pid].rss_bytes == 1024 * 1024
    assert usage[pid].footprint_bytes is None
    assert asked == ["ps"], "a zero footprint must not spend the dump either"


def test_darwin_a_raising_direct_probe_degrades_to_top(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    pid = _live_pid()

    def probe(target: int) -> int | None:
        raise OSError("libproc refused this pid")

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, f"{pid} 1024\n"
        return 0, f"PID    MEM\n{pid}    197M\n"

    usage = session_resource_usage([pid], runner=runner, footprint_probe=probe)
    assert usage[pid].footprint_bytes == 197 * 1024 * 1024


def test_darwin_without_libproc_uses_the_top_fallback(monkeypatch) -> None:
    """The real default probe, on a host where loading libproc failed.

    ``_libproc`` is set to its False sentinel rather than the function being
    replaced, so the degradation under test is the module's own — and this is the
    case that earns the dump for EVERY pid, which is why it must keep working.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(resources, "_libproc", False)
    pid = _live_pid()

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, f"{pid} 1024\n"
        return 0, f"PID    MEM\n{pid}    197M\n"

    usage = session_resource_usage([pid], runner=runner)
    assert usage[pid].footprint_bytes == 197 * 1024 * 1024


def test_pid_exists_separates_a_gone_pid_from_a_live_one() -> None:
    """The gate R1 turns on, on real pids: gone is not the same as unreadable."""
    assert resources._pid_exists(_reaped_pid()) is False
    assert resources._pid_exists(_live_pid()) is True


@pytest.mark.skipif(
    sys.platform != "darwin" or os.geteuid() == 0,
    reason="pid 1 is another account's process unless the suite runs as root",
)
def test_an_unreadable_pid_exists_and_earns_the_dump() -> None:
    """The other side of R1's gate: EPERM is a pid the dump CAN answer.

    pid 1 is the shape of the residue — alive, not ours, unreadable by the direct
    call — and ``/usr/bin/top`` is setuid root, so the batched dump does return a
    figure for it (QA measured 24,117,248 B from the dump where the direct read
    returned nothing). The pair asserted here is what sends it to the dump
    instead of leaving it as an em dash.
    """
    assert resources._pid_exists(1) is True
    assert resources._darwin_footprint_bytes(1) is None


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
