"""Park idle sessions before ending a build: the fleet guard's first rung.

WHY. On 2026-11-12 the operator's machine ran with swap on the data volume at
18 GiB; the only lever for memory was to stop idle sessions from outside, and
every stop made their wakes and monitors dormant (a manager lost 9 prod-window
wakes; customer Slack-thread watches went silent). The machine memory pass
(:mod:`machine_memory`) meanwhile knew only one remedy: SIGKILL the largest
non-runtime fragment — usually a build or a test run someone is waiting on. An
idle session's runtime is the cheaper thing to give back: a PARK
(``control.park_session``) frees it while its wakes and monitors stay armed, and
the wake supervisor brings it back when one fires. So under pressure the guard
parks idle sessions FIRST, oldest-idle first, and the kill path runs only when
no session could be parked.

WHO IS NEVER PARKED HERE:
* a busy session — the record's ``busy``/``waiting_on``/``jobs_running`` bits
  filter the obvious ones up front, and the owner's own idle latch (the same
  ``stop_if_idle`` predicate a ``sessions op=stop`` uses: turn, approval,
  subagent, job, a wake due inside the hold window) refuses anything the record
  missed; the park never signals;
* the chief of staff (``aida.state.is_aida_session``) and any engine-owned duty
  session (``control._engine_owner``: Aida, the Librarian) — their recurring
  passes are the machine's own attention;
* a runtime already leaving, a TUI-owned session, or any runtime with a viewer
  attached (the owner's latch does not count viewers, so this filter must).

DISK PRESSURE (:func:`measure_disk_pressure`). On macOS swap files live on the
``/System/Volumes/VM`` APFS volume, which shares its container's free space with
the data volume (measured 2026-11-12 on the operator's machine: ``df`` reports
the same 8,598,620 KiB available for ``/System/Volumes/VM`` and
``/System/Volumes/Data``; ``sysctl vm.swapusage`` reported 15,360 MiB of swap
files, 13,812 MiB used). So swap growth eats exactly the free space the data
volume has left. Pressure is declared when either:
* free space on the config root's volume is below
  ``capacity_recovery.MIN_FREE_BYTES`` (2 GiB — the floor the installer's own
  capacity recovery already treats as "too little to proceed"), or
* swap shares the volume AND the swap files already outweigh the free space
  left — the next doubling of the swap set cannot fit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger("local_operator.session.runtime.park_pressure")

#: macOS's swap volume. Its free space equals the data volume's when they share
#: an APFS container (see the module docstring).
SWAP_VOLUME = "/System/Volumes/VM"

#: The most sessions one pressure pass parks. Each park is one idle-only socket
#: op plus a bounded wait (``control.DEFAULT_TIMEOUT_S``); a handful frees the
#: big wins (runtimes are hundreds of MB each) without turning one pass into a
#: minute-long walk. The next pass re-measures and parks more if still needed.
MAX_PARKS_PER_PASS = 4

#: ``vm.swapusage``'s ``total``: the bytes the swap FILES occupy on disk. Not
#: ``used``: for a DISK reading the files are what take the space, whatever
#: share of them is in use — and AGENTS.md records ``used`` as not reliably
#: decrementing on macOS, so it is a poor instantaneous signal anyway.
_SWAP_TOTAL_RE = re.compile(r"total\s*=\s*([0-9.]+)([KMGT])")
_UNIT = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


@dataclass(frozen=True)
class DiskPressure:
    free_bytes: int | None
    swap_file_bytes: int | None
    swap_shares_volume: bool
    under_pressure: bool
    reason: str


def _swap_file_bytes() -> int | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _SWAP_TOTAL_RE.search(out or "")
    if match is None:
        return None
    return int(float(match.group(1)) * _UNIT[match.group(2)])


def _free_bytes(path: str) -> int | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


def measure_disk_pressure(
    root: Path,
    *,
    swap_probe: Callable[[], int | None] = _swap_file_bytes,
    free_probe: Callable[[str], int | None] = _free_bytes,
    swap_volume: str = SWAP_VOLUME,
) -> DiskPressure:
    """Read free space and swap once; decide whether the disk is under pressure.

    Unknown never declares pressure: an unreadable volume or swap reading is
    reported, not acted on (the machine memory pass's own rule).
    """
    from local_operator.capacity_recovery import MIN_FREE_BYTES

    free = free_probe(str(root))
    swap_free = free_probe(swap_volume) if os.path.isdir(swap_volume) else None
    # Same container ⇔ same free-space figure (APFS volumes in one container
    # report the container's free space). Not a proof of identity, but the
    # property that matters — swap growth spends this space — is exactly it.
    shares = free is not None and swap_free is not None and swap_free == free
    swap = swap_probe() if shares else None
    if free is None:
        return DiskPressure(None, swap, shares, False, "free space could not be read")
    gib = free / 1024**3
    if free < MIN_FREE_BYTES:
        return DiskPressure(
            free, swap, shares, True, f"{gib:.1f} GiB free on the config volume (< 2 GiB)"
        )
    if shares and swap is not None and swap > free:
        return DiskPressure(
            free,
            swap,
            shares,
            True,
            f"swap files {swap / 1024**3:.1f} GiB on the data volume exceed its {gib:.1f} GiB free",
        )
    return DiskPressure(free, swap, shares, False, f"{gib:.1f} GiB free")


@dataclass
class ParkPassReport:
    parked: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _idle_since(root: Path, record: Any) -> float:
    """When this session last did anything: its transcript's mtime.

    Every turn appends to the transcript, so its mtime is the last-activity
    stamp without a new field on the record; a missing transcript falls back to
    the runtime's start (it never had a turn).
    """
    directory = root / "sessions" / str(record.session_id)
    newest = 0.0
    try:
        for child in directory.iterdir():
            if child.name.startswith("transcript"):
                newest = max(newest, child.stat().st_mtime)
    except OSError:
        pass
    return newest or float(getattr(record, "started_at", 0.0) or 0.0)


def park_candidates(
    root: Path,
    records: Sequence[Any],
    *,
    exclude_pids: frozenset[int] = frozenset(),
) -> tuple[list[Any], list[tuple[str, str]]]:
    """Idle, parkable records oldest-idle first, and ``(session, why)`` skips."""
    from local_operator.session.runtime.control import _engine_owner

    try:
        from local_operator.aida.state import is_aida_session
    except Exception:  # noqa: BLE001 — no Aida module: nobody is her
        is_aida_session = None  # type: ignore[assignment]
    keep: list[tuple[float, Any]] = []
    skipped: list[tuple[str, str]] = []
    for record in records:
        sid = str(record.session_id)
        if record.pid in exclude_pids:
            skipped.append((sid, "excluded"))
            continue
        if (
            getattr(record, "busy", False)
            or getattr(record, "waiting_on", 0)
            or (getattr(record, "jobs_running", 0) or 0)
        ):
            skipped.append((sid, "busy"))
            continue
        if getattr(record, "leaving", ""):
            skipped.append((sid, "leaving"))
            continue
        # SOMEONE IS LOOKING AT IT. The owner's idle latch deliberately does not
        # count an attached viewer (``ServingSessionHandle.may_refresh``), so a
        # pressure park would end a conversation under the operator's eyes — a
        # TUI owner most of all, whose session would end beneath its terminal.
        # Only runtimes nobody is attached to are the guard's to park.
        if getattr(record, "kind", "") == "tui" or not getattr(record, "detached", False):
            skipped.append((sid, "viewer attached"))
            continue
        if is_aida_session is not None and is_aida_session(root, sid):
            skipped.append((sid, "chief of staff"))
            continue
        if _engine_owner(root, sid):
            skipped.append((sid, "engine duty session"))
            continue
        keep.append((_idle_since(root, record), record))
    keep.sort(key=lambda pair: pair[0])
    return [record for _at, record in keep], skipped


def _live_records(root: Path) -> list[Any]:
    from local_operator.session.runtime import registry

    return [record for record, state in registry.scan(root, reap=False) if state == "live"]


async def park_idle_sessions(
    root: Path,
    *,
    limit: int = MAX_PARKS_PER_PASS,
    records: Sequence[Any] | None = None,
    park: Callable[..., Any] | None = None,
    reason: str = "",
) -> ParkPassReport:
    """Park up to ``limit`` idle sessions, oldest-idle first. Never raises.

    ``park`` is the test seam (defaults to ``control.park_session``). One log
    line per park and per owner refusal, naming the pressure that caused it.
    """
    from local_operator.session.runtime import control

    report = ParkPassReport()
    try:
        live = list(records) if records is not None else _live_records(root)
        candidates, report.skipped = park_candidates(
            root, live, exclude_pids=frozenset({os.getpid()})
        )
    except Exception:  # noqa: BLE001 — a pressure rung must never crash the seat
        logger.warning("pressure park: candidate scan failed", exc_info=True)
        return report
    parker = park or control.park_session
    for record in candidates:
        if len(report.parked) >= limit:
            break
        try:
            outcome = await parker(record, _root=root, _command="fleet guard park")
        except Exception as exc:  # noqa: BLE001
            report.refused.append((record.session_id, f"raised {type(exc).__name__}"))
            continue
        if getattr(outcome, "parked", False):
            report.parked.append(record.session_id)
            logger.warning(
                "pressure park: parked %s (pid %s) — %s; %s wakes and %s monitors stay armed",
                record.session_id,
                record.pid,
                reason or "fleet under pressure",
                getattr(outcome, "wakes_armed", 0),
                getattr(outcome, "monitors_armed", 0),
            )
        else:
            why = getattr(outcome, "reason", "") or getattr(outcome, "method", "")
            report.refused.append((record.session_id, str(why)))
            logger.info("pressure park: %s not parked (%s)", record.session_id, why)
    return report


def park_idle_sessions_sync(root: Path, **kwargs: Any) -> ParkPassReport:
    """:func:`park_idle_sessions` for the memory pass's worker thread."""
    return asyncio.run(park_idle_sessions(root, **kwargs))
