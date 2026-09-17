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
reports the phys footprint (compressed + wired + ...) — the number that "adds
up" across the machine's sessions — so ``lop sessions`` shows both: RSS as the
always-present baseline and FOOTPRINT as the honest number when we can get it.
On Linux the analog is ``smaps_rollup`` Pss (proportional set size).

What the direct read is checked against is the reader it REPLACES — the ``top``
MEM column, in ``bench/info-snapshot-*.json`` — and not a third one:
``/usr/bin/footprint`` reports 1.8 % less for the same pid (884,736 B against
901,312 B read together here), so it is a differently-defined reader rather
than a tie-breaker, and "the number Activity Monitor shows" is a gloss on
``top``'s column rather than a second measurement of it.

**On macOS the footprint is read directly, not sampled.** ``top -l1`` dumps the
whole system's process table to answer a question about a handful of pids, and
what it costs is scheduling rather than work: the same command returns in **8 ms**
(1,051 lines, load 585) or in 5-17 s, and 13 of 15 samples at load 470-640
straddled the 5 s subprocess timeout this module passes it. A ``TimeoutExpired``
collapses to "no data" in the runner, so under load the FOOTPRINT column went
silently empty while RSS survived — ``bench/info-snapshot-before.json`` records
the read at 5,758 ms median with 0 of 12 sessions answered. ``proc_pid_rusage``
from libproc returns the same ``ri_phys_footprint`` that column prints, for one
pid, in single-digit microseconds and without a subprocess: the kernel's own
phys-footprint accounting, read at the instant of the call rather than at
``top``'s sampling instant.

The dump is kept for the one class of pid this reader cannot answer — another
account's process, which returns ``EPERM`` (``/usr/bin/top`` is setuid root, so
its dump can read those) — plus the whole set on a host where no direct reader
exists at all (no libproc). ONLY that residue is sent to it. A pid that is GONE
is excluded deliberately: no reader can answer it, so its only effect would be
to spend the dump's whole timeout for nothing — which is the symptom this module
just stopped paying, and a session dying between the registry scan and this read
is the module's own documented normal case.

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


#: The largest pid any supported platform can allocate, used to keep
#: ``os.kill`` from raising instead of answering (see :func:`_pid_exists`). Linux
#: allows a raised ``pid_max`` up to ``2**22`` by default and ``2**31 - 1`` at the
#: kernel's ceiling, macOS tops out at 99,998; this is Linux's ceiling, so the
#: bound rejects only what cannot be a pid on either platform.
_PID_MAX = 2**31 - 1


def _pid_exists(pid: int) -> bool:
    """Whether ``pid`` names a process at all — what the fallback's cost turns on.

    ``os.kill(pid, 0)`` is the primitive ``registry.classify`` and
    ``session_lease.liveness`` already spend on this question, and signal 0
    settles it without touching the target: ``ESRCH`` means the pid is gone,
    ``EPERM`` means it exists under another account. Any other ``OSError``
    answers "exists", which is the conservative direction here — it spends the
    dump, i.e. the behaviour a pid nobody can classify would otherwise have had.

    Called only for pids the direct reader could not answer, where the two
    answers lead to opposite costs: the dump for one that exists, nothing for
    one that does not.

    Bounded above as well as below, and that is load-bearing rather than
    defensive: ``os.kill`` raises ``OverflowError`` — not an ``OSError`` — for a
    pid outside C ``int``, and that exception would escape this module's
    no-raise contract into the caller. No real pid reaches it (macOS ``PID_MAX``
    is 99,998 and Linux defaults to 4,194,304, both far under the bound), so the
    guard exists so that the shape of the check matches what it claims to cover.
    """
    if pid <= 0 or pid > _PID_MAX:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


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
    ``proc_pid_rusage`` direct read. On Linux footprint reads per-pid
    ``smaps_rollup`` (no subprocess). ``runner`` and ``footprint_probe`` are
    injectable for tests. Every pid gets an entry, even if both numbers come
    back ``None`` — the caller prints ``—`` for unknowns and never omits a live
    session from the table.

    On macOS the batched ``top`` dump is consulted ONLY where the direct reader
    has no number AND the pid still exists — another account's process, or a
    host with no libproc at all. A pid that is gone, and a pid the kernel
    answers with a footprint of zero (a zombie), get ``None`` without spending
    it: no reader can answer either, and the dump is the whole cost of this
    call. See the loop's comment and :func:`_pid_exists`.
    """
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
        # no subprocess and no sampling window.
        #
        # THE DUMP IS FOR A MISSING READER, NOT FOR EVERY UNKNOWN PID. Only two
        # cases earn it: a pid this process may not inspect (`EPERM` — `/usr/bin/
        # top` is setuid root and can read those), and a host where no direct
        # reader exists at all (libproc did not load), where every pid needs it.
        # A pid that is GONE is excluded on purpose: no reader can answer it, so
        # its only effect would be to spend the dump's sampling interval — one
        # whole-system dump, whose ceiling is the runner's own 5 s timeout — to
        # return nothing. This is not an exotic case: `collect_sessions`
        # hands this function every record `registry.scan` classified `live`, and
        # the module's own contract names "a pid vanishing between scan and
        # measure" as normal. Measured on twelve live pids plus one reaped pid,
        # through the module's own runner: 1,259.6-1,403.9 ms with this gate
        # removed (one dump, 1,334.3 ms median) against 50.6-140.2 ms with it
        # (69.2 ms median), and the dump is unbounded above — a session dying
        # while the box is loaded enough puts its 5 s timeout on the read, which
        # is what the pre-gate measurement of 5,185.2 ms for one live pid plus
        # one reaped was.
        probe = footprint_probe or _darwin_footprint_bytes
        waiters: list[int] = []
        for pid in pids:
            value: int | None
            try:
                value = probe(pid)
            except Exception:  # noqa: BLE001 — same absolute-degradation rule
                value = None
            if value is None:
                if _pid_exists(pid):
                    waiters.append(pid)
            elif value > 0 and pid in usage:
                usage[pid].footprint_bytes = value
            # value <= 0 falls through deliberately, and is neither a reading nor
            # a reason to spend the dump. `ri_phys_footprint` is 0 for a zombie
            # (measured), and 0 is never what a live process costs: reporting it
            # as the footprint would put "0 MB" in `lop sessions`'s FOOTPRINT
            # column and `0` where the wire payload's `footprint_bytes` must be
            # the unknown sentinel (``null``). The TUI's memory cell is a
            # different rule — it renders ``format_bytes(footprint_bytes or
            # rss_bytes)``, and a zombie's RSS is 0 too, so that surface shows
            # "0 MB" on this and every earlier release alike. What the guard
            # fixes is the layer that owns the sentinel. A pid that rusage answers
            # with 0 for is also exactly the gone case above, so the dump cannot
            # help it either.

        if waiters:
            # `top -l1 -stats pid,mem` for the whole system in one shot; filter
            # by our pids. Passing -pid repeatedly is slower and caps out, so we
            # take the full dump and select. The MEM column is the phys
            # footprint, the same number the direct read returned above.
            code, out = run(["top", "-l1", "-stats", "pid,mem", "-ncols", "2"])
            if code == 0:
                for pid, footprint in _parse_top_footprint(out, set(waiters)).items():
                    if pid in usage:
                        usage[pid].footprint_bytes = footprint
    else:
        for pid in pids:
            pss = _linux_pss_bytes(pid)
            if pss is not None:
                usage[pid].footprint_bytes = pss

    return usage
