"""What a runtime's boot cost the process: CPU, time ready to run, and wall.

WHY THIS EXISTS. The wake supervisor logs cold runtime starts with a 5.8 s
median and a p99 of 161 s, while an attribution of every stage on the start path
finds ~2-4 s of CPU. A record holding only wall and CPU cannot say where the
other ~150 s went: waiting for a core (a starved process on a crowded host) and
waiting on something (a lock, a socket, a timer) look the same from there.

THE QUANTITY THAT SEPARATES THEM is the kernel's own account of how long this
process was READY TO RUN — on a core or queued for one:

* ``waited_for_cpu_s = runnable_s - cpu_s``: runnable but not running, i.e. time
  lost to the scheduler. Large means starved.
* ``blocked_s = wall_s - runnable_s``: not runnable at all, i.e. waiting on I/O,
  a lock, a timer or another process. Large means blocked.

The context-switch counts ride along as raw evidence, but they DO NOT separate
those two on macOS: measured during review of #3226, XNU books waits in
``kqueue`` (asyncio's loop), ``Lock.acquire``, ``Event.wait`` and ``sleep`` as
INVOLUNTARY switches with zero voluntary, so a blocked process and a starved one
both read "0 voluntary, many involuntary". Never read the switch ratio as a
starvation signal; read ``waited_for_cpu_s`` against ``blocked_s``.

SOURCES, one call each, no subprocess, no lock, no directory walk on macOS:

* macOS — ``proc_pid_rusage(getpid(), RUSAGE_INFO_V4)``: ``ri_runnable_time``
  (running or on a run queue, process-wide including exited threads),
  ``ri_user_time + ri_system_time``, and ``ri_proc_start_abstime`` against
  ``mach_absolute_time()`` for a nanosecond-resolution process age (the
  ``sysctl`` start stamp carries whole seconds and read ~0.6 s high).
* Linux — ``/proc/self/task/*/schedstat`` (on-CPU ns, run-queue-wait ns per
  thread, summed: ``/proc/self/schedstat`` alone is the MAIN thread only), and
  the ``/proc/self/stat`` start tick against ``CLOCK_BOOTTIME`` (clock-tick
  resolution, usually 10 ms). Threads that already exited drop out of the sum,
  so ``runnable_s`` can read low there. NOT EXERCISED LOCALLY: the Linux branch
  is covered by CI's Ubuntu jobs only.
* Both — ``getrusage(RUSAGE_SELF)`` for ``cpu_s`` and the switch counts, so CPU
  and the counts are one snapshot; it excludes reaped children, which is right:
  the question is what the runtime itself spent.

Stdlib-only and import-light on purpose: it runs on the boot path and is
sampled once by the standby warm-up (``standby._warm``) so an adopted spare
does not pay ``ctypes`` resolution at its start record. Every reader degrades to
``None`` for a fact it cannot get; nothing here raises.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

#: ``RUSAGE_INFO_V4`` from ``<sys/resource.h>``; the struct below is that
#: version's layout through ``ri_runnable_time``. Only the fields this module
#: reads are named; the rest pad to the exact offsets, and the trailing reserve
#: keeps the buffer at least as large as the kernel writes (V4 is 296 bytes).
#: Offsets verified against the SDK header with a compiled ``offsetof`` probe:
#: ri_user_time=16, ri_proc_start_abstime=80, ri_runnable_time=288; pinned by
#: ``test_boot_cost``.
_RUSAGE_INFO_V4 = 4

_DARWIN: Any = None


def _darwin() -> Any:
    """``(libc, struct type, seconds-per-abstime-tick)``, resolved once, or ``False``."""
    global _DARWIN
    if _DARWIN is None:
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            u64 = ctypes.c_uint64

            class _RusageInfoV4(ctypes.Structure):
                _fields_ = [
                    ("ri_uuid", ctypes.c_uint8 * 16),
                    ("ri_user_time", u64),
                    ("ri_system_time", u64),
                    ("_pad0", u64 * 6),  # pkg_idle_wkups .. phys_footprint
                    ("ri_proc_start_abstime", u64),
                    ("_pad1", u64 * 25),  # ri_proc_exit_abstime .. ri_interval_max_phys_footprint
                    ("ri_runnable_time", u64),
                    ("_reserve", u64 * 16),
                ]

            class _Timebase(ctypes.Structure):
                _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

            timebase = _Timebase()
            libc.mach_timebase_info(ctypes.byref(timebase))
            libc.mach_absolute_time.restype = u64
            tick = timebase.numer / timebase.denom / 1e9
            _DARWIN = (libc, _RusageInfoV4, tick, ctypes)
        except Exception:  # noqa: BLE001 — a missing fact, never a failed boot
            _DARWIN = False
    return _DARWIN


def _darwin_times() -> tuple[float | None, float | None]:
    """``(runnable_s, age_s)`` from one ``proc_pid_rusage`` call."""
    resolved = _darwin()
    if not resolved:
        return None, None
    libc, struct_type, tick, ctypes = resolved
    info = struct_type()
    if libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None, None
    now = libc.mach_absolute_time()
    age = (now - info.ri_proc_start_abstime) * tick if info.ri_proc_start_abstime else None
    return info.ri_runnable_time * tick, age


def _linux_times() -> tuple[float | None, float | None]:
    """``(runnable_s, age_s)`` from schedstat and the stat start tick."""
    runnable: float | None = None
    try:
        total = 0
        for task in os.listdir("/proc/self/task"):
            with open(f"/proc/self/task/{task}/schedstat", "rb") as handle:
                on_cpu, run_delay = handle.read().split()[:2]
            total += int(on_cpu) + int(run_delay)
        runnable = total / 1e9
    except (OSError, ValueError):
        runnable = None
    age: float | None = None
    try:
        with open("/proc/self/stat", "rb") as handle:
            data = handle.read()
        # After the parenthesised comm (which may hold spaces), starttime is
        # field index 19 — the same parse ``secrets.peer._linux_stat`` uses.
        start_ticks = int(data[data.rfind(b")") + 2 :].split()[19])
        # ``getattr``: the constant exists only in Linux builds of CPython, and a
        # type checker running on macOS cannot see it.
        boottime = time.clock_gettime(getattr(time, "CLOCK_BOOTTIME"))
        age = boottime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        age = None
    return runnable, age


def sample() -> dict[str, float] | None:
    """Cumulative counters for THIS process since exec, or ``None`` if unreadable.

    Keys: ``cpu_s``, ``runnable_s`` (or ``None``), ``wall_s`` (or ``None``),
    ``voluntary_switches``, ``involuntary_switches``.
    """
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:  # noqa: BLE001
        return None
    runnable: float | None = None
    wall: float | None = None
    try:
        if sys.platform == "darwin":
            runnable, wall = _darwin_times()
        elif sys.platform.startswith("linux"):
            runnable, wall = _linux_times()
    except Exception:  # noqa: BLE001 — one missing fact, not a missing record
        runnable, wall = None, None
    return {
        "cpu_s": usage.ru_utime + usage.ru_stime,
        "runnable_s": runnable,  # type: ignore[dict-item]
        "wall_s": wall,  # type: ignore[dict-item]
        "voluntary_switches": float(usage.ru_nvcsw),
        "involuntary_switches": float(usage.ru_nivcsw),
    }


def window(end: dict[str, Any], start: dict[str, Any] | None = None) -> dict[str, Any]:
    """The readable record for the window ``start..end`` (``start=None``: since exec).

    THREAD-SECONDS, NOT A PARTITION OF WALL. ``cpu_s`` and ``runnable_s`` are
    summed over every thread, so a boot with two busy threads can show more CPU
    than wall; each part is therefore clamped at zero on its own rather than
    forced to add up. For the runtime's boot (one thread doing the work, a few
    mostly parked helpers) they add up to ``wall_s`` within a tick, and that is
    how to read them: ``waited_for_cpu_s`` is the starvation, ``blocked_s`` the
    waiting on something.
    """

    def delta(key: str) -> float | None:
        value = end.get(key)
        if value is None:
            return None
        if start is None:
            return float(value)
        base = start.get(key)
        return None if base is None else max(0.0, float(value) - float(base))

    cpu = delta("cpu_s") or 0.0
    wall = delta("wall_s")
    runnable = delta("runnable_s")
    return {
        "wall_s": None if wall is None else round(wall, 3),
        "cpu_s": round(cpu, 3),
        "waited_for_cpu_s": None if runnable is None else round(max(0.0, runnable - cpu), 3),
        "blocked_s": (
            None if runnable is None or wall is None else round(max(0.0, wall - runnable), 3)
        ),
        "voluntary_switches": int(delta("voluntary_switches") or 0),
        "involuntary_switches": int(delta("involuntary_switches") or 0),
    }
