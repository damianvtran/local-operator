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

Everything degrades to ``None`` rather than raising: a probe failing, a pid
vanishing between scan and measure, or an unparseable line must NEVER fail the
whole listing — a session with unknown memory is still a session worth showing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

#: A subprocess runner seam so tests can inject fake ``ps``/``top`` output
#: without spawning real processes. Returns ``(returncode, stdout)``; a raising
#: implementation is caught by the callers, same as a non-zero return.
SubprocessRunner = Callable[[list[str]], "tuple[int, str]"]


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
) -> dict[int, ResourceUsage]:
    """Measure RSS + footprint for ``pids`` in as few subprocesses as possible.

    One batched ``ps`` covers RSS for every pid at once; on macOS one batched
    ``top -l1`` covers footprint; on Linux footprint reads per-pid
    ``smaps_rollup`` (no subprocess). ``runner`` is injectable for tests. Every
    pid gets an entry, even if both numbers come back ``None`` — the caller
    prints ``—`` for unknowns and never omits a live session from the table."""
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
        # `top -l1 -stats pid,mem` for the whole system in one shot; filter by
        # our pids. Passing -pid repeatedly is slower and caps out, so we take
        # the full dump and select. The MEM column is the phys footprint.
        code, out = run(["top", "-l1", "-stats", "pid,mem", "-ncols", "2"])
        if code == 0:
            for pid, footprint in _parse_top_footprint(out, set(pids)).items():
                if pid in usage:
                    usage[pid].footprint_bytes = footprint
    else:
        for pid in pids:
            pss = _linux_pss_bytes(pid)
            if pss is not None:
                usage[pid].footprint_bytes = pss

    return usage
