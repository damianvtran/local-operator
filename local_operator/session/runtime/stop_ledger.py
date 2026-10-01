"""The SWEEP LEDGER: one begin row and one end row per machine-wide stop sweep.

WHY IT EXISTS. A per-victim marker (``control._stop_marker_payload``) names the
act on ONE runtime; it cannot say "this was one sweep over fourteen targets", and
the 2026-09-30 18:14 wave could only be described by clustering timestamps. A
sweep that a cooperating party runs (``lop stop --all``, the TUI's stop-all) is
the one wave that CAN be recorded as a unit, so it is: ``begin`` BEFORE the first
marker or signal names the sweep, its actor and every target; ``end`` after names
the outcomes. Each marker the sweep stages carries the same ``sweep_id`` so a
victim joins to its sweep.

WHAT IT DELIBERATELY IS NOT. It is not a record of every wave: an UNSANCTIONED
wave (something signalled N processes and staged nothing) has no actor that ran
this code, so it yields N per-victim receipts and ZERO rows here — and a reader
must never be handed a fabricated sweep for it. An actor that dies between the
two rows leaves a ``begin`` with no ``end``, which is itself honest evidence.

HOME. ``config_dir()/logs/stop-sweeps.jsonl`` beside ``exec-jobs.jsonl`` (0700
directory, 0600 file, ``O_APPEND`` single write so concurrent sweeps cannot
interleave a row). History, not live registry state, so not under ``run/``.
Bounded: rotated once to ``.1`` past :data:`ROTATE_BYTES`, best effort, because
the sibling ledger is unbounded and a second unbounded file is not a better
neighbour.

IMPORT-LIGHT ON PURPOSE: stdlib plus ``paths`` only. ``control`` and the CLI
startup path are import-sensitive (``tests/unit/test_import_graph.py``) and
``exec_mode`` (which owns the sibling ledger) must not be imported from them.

EVERY WRITER HERE IS BEST-EFFORT AND NEVER RAISES: a stop the user asked for must
not be abandoned because its bookkeeping could not be written.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from local_operator.paths import LOG_DIRNAME, config_dir

logger = logging.getLogger(__name__)

#: File name inside ``logs/``.
SWEEPS_FILE = "stop-sweeps.jsonl"

#: Rotate to ``<name>.1`` (replacing any older one) once the file passes this.
ROTATE_BYTES = 256 * 1024

#: A sweep records at most this many targets; ``truncated`` says so. Bounded so
#: one machine with thousands of records cannot make a single ``O_APPEND`` row
#: exceed what the single-write atomicity guarantee covers.
MAX_TARGETS = 256

SCHEMA_VERSION = 1


def sweeps_path(root: Path | None = None) -> Path:
    """The ledger path under ``root`` (default :func:`config_dir`). Resolved, not created."""
    return (root if root is not None else config_dir()) / LOG_DIRNAME / SWEEPS_FILE


def new_sweep_id() -> str:
    """A short unique id (12 hex chars) stamped into the ledger and every marker."""
    return uuid.uuid4().hex[:12]


def _append(path: Path, row: dict[str, Any]) -> bool:
    """Append one JSON row, rotating first when the file is past the bound."""
    try:
        directory = path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            if path.stat().st_size > ROTATE_BYTES:
                os.replace(path, path.with_name(path.name + ".1"))
        except OSError:
            pass  # absent file, or a rotation lost to a concurrent sweep: both fine
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        fd = os.open(str(path), flags, 0o600)
        try:
            os.write(fd, (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except Exception:  # noqa: BLE001 — bookkeeping must never abort a stop
        logger.debug("stop sweep ledger write failed", exc_info=True)
        return False


def _actor_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "pid": os.getpid(),
        "argv0": os.path.basename(sys.argv[0] or "") or sys.executable,
    }
    getuid = getattr(os, "getuid", None)
    if callable(getuid):
        facts["uid"] = getuid()
    return facts


def begin_sweep(
    *,
    mechanism: str,
    command: str,
    targets: Iterable[dict[str, Any]],
    actor: str = "",
    root: Path | None = None,
) -> str:
    """Write the ``begin`` row and return the sweep id. Never raises.

    Called BEFORE the first marker or signal of the sweep: the begin row is the
    claim that this process is about to act, so it must exist even if the sweep
    dies on its first target. The id is returned even when the write failed, so
    the markers still carry it (a victim then names a sweep whose row is missing,
    which is a visible gap rather than an absent link).
    """
    sweep_id = new_sweep_id()
    try:
        listed = list(targets)
        capped = listed[:MAX_TARGETS]
        counts: dict[str, int] = {"targets": len(listed)}
        for target in listed:
            kind = str(target.get("rec_kind") or "unknown")
            counts[kind] = counts.get(kind, 0) + 1
        row: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "kind": "sweep",
            "phase": "begin",
            "sweep_id": sweep_id,
            "mechanism": mechanism,
            "command": command,
            **({"actor": actor} if actor else {}),
            **_actor_facts(),
            "at": time.time(),
            "targets": capped,
            "counts": counts,
        }
        if len(listed) > len(capped):
            row["truncated"] = True
        _append(sweeps_path(root), row)
    except Exception:  # noqa: BLE001
        logger.debug("stop sweep begin row could not be built", exc_info=True)
    return sweep_id


def end_sweep(
    sweep_id: str, outcomes: Iterable[dict[str, Any]], *, root: Path | None = None
) -> None:
    """Write the ``end`` row for ``sweep_id``. Never raises."""
    try:
        listed = list(outcomes)
        row: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "kind": "sweep",
            "phase": "end",
            "sweep_id": sweep_id,
            "at": time.time(),
            "outcomes": listed[:MAX_TARGETS],
        }
        if len(listed) > MAX_TARGETS:
            row["truncated"] = True
        _append(sweeps_path(root), row)
    except Exception:  # noqa: BLE001
        logger.debug("stop sweep end row could not be built", exc_info=True)


def read_sweeps(root: Path | None = None) -> list[dict[str, Any]]:
    """Every parseable row, oldest first (the rotated file's rows precede the live
    file's). A torn or corrupt line is skipped: a reader must never fail on the
    ledger."""
    rows: list[dict[str, Any]] = []
    path = sweeps_path(root)
    for candidate in (path.with_name(path.name + ".1"), path):
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                loaded = json.loads(line)
            except ValueError:
                continue
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows
