"""Per-session resource usage for ``lop sessions``.

Reports RSS and (where available) the *true* memory footprint of each live
session's process, using only stdlib + shelling out to ``ps``/``top``/
``smaps_rollup``. ``psutil`` is deliberately NOT a dependency: the registrant
sits on the CLI startup path and every import there costs each ``lop`` launch
real milliseconds, so a heavy third-party dependency for a rarely-run listing
command is the wrong trade. The subprocess probes are cheap and portable
enough (macOS + Linux) for a laptop's single-digit session count.

Why footprint and not just RSS: on macOS the kernel compresses and swaps
memory, so RSS materially UNDER-reports what a process actually costs. ``top``
reports the phys footprint (compressed + wired + ...) — the number Activity
Monitor shows and the one that "adds up" — so ``lop sessions`` shows both: RSS
as the always-present baseline and FOOTPRINT as the honest number when we can
get it. On Linux the analog is ``smaps_rollup`` Pss (proportional set size).

**On macOS the footprint is read directly, not sampled.** ``top -l1`` dumps
the whole system's process table to answer a question about a handful of pids,
and it does not return until its first sample interval is over: measured
2,344 ms here against 16 ms for the batched ``ps``, which made this module the
dominant cost of every ``/info`` read (``collect_snapshot``: 3,133 ms median,
2,020 ms of it this call) and of ``lop sessions``. ``proc_pid_rusage`` from
libproc returns the SAME ``ri_phys_footprint`` the ``top`` MEM column prints,
for one pid, in single-digit microseconds and without a subprocess — the
kernel's own phys-footprint accounting, at the instant of the call rather than
at ``top``'s sampling instant. The ``top`` dump is kept as the fallback (it is
the only reader that can cover a pid this process is not allowed to inspect —
another account's process answers ``EPERM``) and is paid for only by the pids
the direct read missed; see :func:`session_resource_usage`.

Everything degrades to ``None`` rather than raising: a probe failing, a pid
vanishing between scan and measure, or an unparseable line must NEVER fail the
whole listing — a session with unknown memory is still a session worth showing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

#: A subprocess runner seam so tests can inject fake ``ps``/``top`` output
#: without spawning real processes. Returns ``(returncode, stdout)``; a raising
#: implementation is caught by the callers, same as a non-zero return.
SubprocessRunner = Callable[[list[str]], "tuple[int, str]"]

#: One pid's phys footprint in bytes, or ``None`` when this host cannot answer
#: for that pid. Injectable for the same reason ``SubprocessRunner`` is: a test
#: that wants to pin the ``top`` fallback must be able to say "the direct probe
#: has nothing", and a test that wants to pin the direct path must not need a
#: live libproc. Injecting a probe that returns ``None`` for every pid is also
#: exactly what a host WITHOUT libproc does, so that one injection covers both.
FootprintProbe = Callable[[int], "int | None"]

#: ``proc_pid_rusage``'s flavor for ``struct rusage_info_v2``. V2 rather than
#: the newest flavor the header offers: every field read here is already in V1,
#: it is the oldest flavor that carries ``ri_phys_footprint``, and a newer
#: flavor would fail on hosts whose libproc predates it while buying nothing.
_RUSAGE_INFO_V2 = 2

#: Byte offset of ``ri_phys_footprint`` inside ``rusage_info_v2`` — a 16-byte
#: ``ri_uuid`` followed by seven ``uint64`` counters (``ri_user_time``,
#: ``ri_system_time``, ``ri_pkg_idle_wkups``, ``ri_interrupt_wkups``,
#: ``ri_pageins``, ``ri_wired_size``, ``ri_resident_size``). Hardcoded rather
#: than spelled as a ``ctypes.Structure`` because only this one field is read
#: and the ABI is fixed by the kernel: the struct is append-only, so the offset
#: is stable across every flavor at or above V1.
_RUSAGE_PHYS_FOOTPRINT_OFFSET = 16 + 7 * 8

#: The buffer ``proc_pid_rusage`` writes into. Comfortably larger than every
#: published ``rusage_info_v*`` struct (V6 is 0x120 bytes): an undersized buffer
#: would be overflowed by the kernel, which is a memory-safety bug rather than a
#: wrong number, so this is deliberately generous and never tuned down.
_RUSAGE_BUFFER_SIZE = 2048

#: The dlopen'd libproc, or ``False`` once loading it has failed. Cached because
#: this is called per pid and ``ctypes.CDLL`` re-resolves on every call; the
#: False sentinel keeps a host without libproc from retrying per pid.
_libproc: Any = None


@dataclass
class ResourceUsage:
    """One process's memory numbers. Both optional — see module docstring."""

    rss_bytes: int | None = None
    footprint_bytes: int | None = None


def _default_runner(argv: list[str]) -> tuple[int, str]:
    """Run one probe with a short timeout, swallowing every failure mode.

    A missing binary, a timeout, or a non-zero exit all collapse to
    ``(1, "")`` so callers treat "no data" uniformly and the listing never
    crashes because a probe misbehaved on some host."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _parse_ps_rss(output: str) -> dict[int, int]:
    """Parse ``ps -o pid=,rss=`` output (``rss`` is in KiB) into bytes by pid.

    Unparseable lines are skipped, not fatal: ``ps`` output is stable but a
    stray header or blank line must not sink the whole batch."""
    result: dict[int, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            rss_kib = int(parts[1])
        except ValueError:
            continue
        result[pid] = rss_kib * 1024
    return result


def _parse_top_footprint(output: str, wanted: set[int]) -> dict[int, int]:
    """Parse ``top -l1 -stats pid,mem`` output into footprint bytes by pid.

    The MEM column carries a human size suffix (``K``/``M``/``G``/``B`` or a
    bare byte count). Only pids in ``wanted`` are kept — ``top`` without a
    ``-pid`` filter dumps every process, and we want just the sessions.
    Anything that does not parse cleanly is skipped so a format quirk on one
    row cannot break the rest."""
    result: dict[int, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid not in wanted:
            continue
        parsed = _parse_mem_size(parts[1])
        if parsed is not None:
            result[pid] = parsed
    return result


def _parse_mem_size(token: str) -> int | None:
    """Convert a ``top`` MEM token (``"42M"``, ``"1234K"``, ``"512B"``, ``"7G"``,
    or a bare number of bytes) to bytes. Returns ``None`` when it does not look
    like a size, so the caller leaves the footprint unknown."""
    token = token.strip().rstrip("+")  # top marks growing values with a trailing '+'
    if not token:
        return None
    units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    suffix = token[-1].upper()
    if suffix in units:
        number = token[:-1]
        multiplier = units[suffix]
    else:
        number = token
        multiplier = 1
    try:
        return int(float(number) * multiplier)
    except ValueError:
        return None


def _darwin_libproc() -> Any | None:
    """``libproc`` loaded through ``ctypes``, or ``None`` where it is absent.

    ``ctypes`` is imported HERE rather than at module scope for the reason this
    module's docstring gives for ``psutil``: every import on the CLI startup
    path is paid for by every ``lop`` invocation. Best-effort throughout — a
    host with no ``libproc.dylib`` (a Linux box whose ``sys.platform`` a test
    has monkeypatched, a hardened runtime that refuses the load) degrades to the
    ``top`` fallback rather than raising."""
    global _libproc
    if _libproc is not None:
        return _libproc if _libproc is not False else None
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
        lib = ctypes.CDLL(path, use_errno=True)
        lib.proc_pid_rusage.restype = ctypes.c_int
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    except Exception:  # noqa: BLE001 — no libproc is just "use the fallback"
        _libproc = False
        return None
    _libproc = lib
    return lib


def _darwin_footprint_bytes(pid: int) -> int | None:
    """``ri_phys_footprint`` for one pid via ``proc_pid_rusage``, or ``None``.

    This is the ``top`` MEM column's own number — the kernel's phys-footprint
    accounting, the figure Activity Monitor shows — read for one pid directly.
    ``None`` for every failure (no libproc, no such pid, another account's
    process answering ``EPERM``, a flavor this host does not know): the caller's
    fallback covers all of them and the listing's contract is that an unknown
    number prints as an em dash rather than failing the read."""
    lib = _darwin_libproc()
    if lib is None:
        return None
    try:
        import ctypes

        buffer = ctypes.create_string_buffer(_RUSAGE_BUFFER_SIZE)
        rc = lib.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(buffer))
        if rc != 0:
            return None
        raw = buffer.raw
        end = _RUSAGE_PHYS_FOOTPRINT_OFFSET + 8
        return int.from_bytes(raw[_RUSAGE_PHYS_FOOTPRINT_OFFSET:end], "little")
    except Exception:  # noqa: BLE001 — any probe failure is just missing data
        return None


def _linux_pss_bytes(pid: int) -> int | None:
    """Read ``/proc/<pid>/smaps_rollup`` Pss (KiB) as bytes, or ``None``.

    Pss (proportional set size) is Linux's honest footprint analog: shared
    pages are divided across sharers. Guarded by a file-exists check —
    ``smaps_rollup`` is absent on old kernels and everywhere non-Linux."""
    path = f"/proc/{pid}/smaps_rollup"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def session_resource_usage(
    pids: list[int],
    *,
    runner: SubprocessRunner | None = None,
    footprint_probe: FootprintProbe | None = None,
) -> dict[int, ResourceUsage]:
    """Measure RSS + footprint for ``pids`` in as few subprocesses as possible.

    One batched ``ps`` covers RSS for every pid at once and is the only
    subprocess on the macOS path; footprint there comes from the per-pid
    ``proc_pid_rusage`` direct read, with one batched ``top -l1`` as the
    fallback for whatever the direct read could not cover. On Linux footprint
    reads per-pid ``smaps_rollup`` (no subprocess). ``runner`` and
    ``footprint_probe`` are injectable for tests. Every pid gets an entry, even
    if both numbers come back ``None`` — the caller prints ``—`` for unknowns
    and never omits a live session from the table."""
    base = runner or _default_runner

    def run(argv: list[str]) -> tuple[int, str]:
        # A probe raising (an injected fake, or an exotic OSError the default
        # runner did not anticipate) must degrade to "no data", never sink the
        # whole listing — the graceful-degradation contract is absolute.
        try:
            return base(argv)
        except Exception:  # noqa: BLE001 — any probe failure is just missing data
            return 1, ""

    usage: dict[int, ResourceUsage] = {pid: ResourceUsage() for pid in pids}
    if not pids:
        return usage

    pid_csv = ",".join(str(pid) for pid in pids)

    # RSS: one ps for all pids. `-o pid=,rss=` suppresses headers (portable on
    # macOS and Linux); rss is KiB.
    code, out = run(["ps", "-o", "pid=,rss=", "-p", pid_csv])
    if code == 0:
        for pid, rss in _parse_ps_rss(out).items():
            if pid in usage:
                usage[pid].rss_bytes = rss

    # Footprint: platform-specific, still best-effort.
    if sys.platform == "darwin":
        # Direct read first: one `proc_pid_rusage` per pid, microseconds each,
        # no subprocess and no sampling window. The fallback below exists for
        # exactly the pids this cannot answer for — another account's process
        # (EPERM), an unresolvable libproc — and it is paid for ONLY by those
        # pids, so the common read (every listed session is this user's own
        # process) never shells `top` at all. That matters because the dump is
        # 2,344 ms against 16 ms for the batched `ps`, and it used to be the
        # whole cost of the `/info` panel.
        probe = footprint_probe or _darwin_footprint_bytes
        missed: list[int] = []
        for pid in pids:
            value: int | None
            try:
                value = probe(pid)
            except Exception:  # noqa: BLE001 — same absolute-degradation rule
                value = None
            if value is None:
                missed.append(pid)
            elif pid in usage:
                usage[pid].footprint_bytes = value

        if missed:
            # `top -l1 -stats pid,mem` for the whole system in one shot; filter
            # by our pids. Passing -pid repeatedly is slower and caps out, so we
            # take the full dump and select. The MEM column is the phys
            # footprint, the same number the direct read returned above.
            code, out = run(["top", "-l1", "-stats", "pid,mem", "-ncols", "2"])
            if code == 0:
                for pid, footprint in _parse_top_footprint(out, set(missed)).items():
                    if pid in usage:
                        usage[pid].footprint_bytes = footprint
    else:
        for pid in pids:
            pss = _linux_pss_bytes(pid)
            if pss is not None:
                usage[pid].footprint_bytes = pss

    return usage
