"""A bounded census of the session store, so deletions we did not perform show up.

WHY THIS EXISTS. ``remove_session_dir`` now writes a fail-closed receipt, so
every removal through the one audited path is attributable. The measured gap
was elsewhere: 135 delegated child directories of one parent were gone with
ZERO trace — no ``.cleanup-log.jsonl`` row and no ``session cleanup: removing``
WARNING across 1,202 runtime logs, against 528 such warnings for other
sessions. A removal that never reaches the audited remover leaves no record at
all, and nothing in the store could tell an auditor it happened.

The census closes that hole from the outside. Each pass writes what the store
holds (a bounded per-session measurement, not a copy of anything), and the NEXT
pass diffs it against the disk and the cleanup log: a directory that was there
and is now gone, with no removal row in the log range since the last census, is
reported as ``unaccounted-removal`` with the contents it last had. A pad that
materially shrank with no content row is reported the same way.

DESIGN CONSTRAINTS, all deliberate:

* **Read-only over the store.** This module never deletes, renames or writes
  inside a session directory; its only writes are the census file and
  append-only cleanup-log rows.
* **Bounded.** Per-session and GLOBAL entry budgets, plus a ``should_stop``
  callback for the maintenance pass's own stop event. The measured store
  (3,255 sessions, 298,929 entries capped at 2,000/session, 26.3 GB logical)
  walked in 9.67 s; a global budget keeps a larger store proportional instead
  of letting 82 uncapped sessions set the cost. A capped pass records
  ``complete: false``.
* **An incomplete census NEVER reports.** Partial coverage cannot prove a
  directory is missing, so a pass that stopped early (or whose previous
  baseline was partial) writes a baseline and no rows. The next complete pass
  diffs from it. This is the fail-closed direction: a missed report is
  recoverable, a false "your session vanished" is not. ``complete`` is the
  STRICT flag (every session measured in full); ``enumerated`` is the subset
  the existence diff needs (every directory visited). They differ when one
  huge pad hits the per-session cap: that is a store the census reports
  ``complete: false`` about forever, and it must keep reporting disappearances
  anyway — one clone tree must not silence the store's audit.
* **First run writes a baseline.** No previous census means no rows — a fresh
  install must not report every session it has as unaccounted.
* **Import-light.** This runs on the store-maintenance daemon thread at boot;
  it takes no dependency on the cleanup policy, the delegated sweeps or any
  model/tool machinery, and reads the cleanup log as bytes rather than
  importing its writer.
* **Runs regardless of the cleanup switch.** The census is not a retention
  feature — it is the record that retention (and anything else) left, so the
  maintenance pass runs it whether or not ``session.cleanup.enabled`` is true.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    CONTENTS_MAX_ENTRIES,
    measure_session_dir,
)
from local_operator.session.retention import SESSIONS_DIRNAME

logger = logging.getLogger(__name__)

#: The census document, inside ``sessions/``. Dotted, so a session listing
#: never mistakes it for a session id.
CENSUS_NAME = ".session-census.json"
CENSUS_SCHEMA = 1

#: Per-session and whole-pass entry caps. The per-session cap is what the
#: measured store was taken with (2,000); the global cap keeps the pass cost
#: proportional on a store whose many mid-sized pads would each stay under the
#: per-session cap — 500k entries is roughly twice the measured store's capped
#: walk, i.e. the whole measured store fits with margin while an unbounded
#: tree (a clone with a node_modules) cannot set the runtime.
CENSUS_MAX_ENTRIES_PER_SESSION = 2_000
CENSUS_MAX_TOTAL_ENTRIES = 500_000

#: A pad shrink is MATERIAL when at least this many bytes are gone, when at
#: least this fraction is gone, or when the pad went to zero from something.
#: The OR is the point: a 900 KB build output leaving a 1.5 MB pad crosses the
#: fraction, a 3 MB log leaving a 40 MB pad crosses the byte floor, and a small
#: pad emptied entirely is always worth a row. The thresholds are reporting
#: thresholds only — nothing here removes anything.
CENSUS_MATERIAL_PAD_BYTES = 1_000_000
CENSUS_MATERIAL_PAD_FRACTION = 0.5

#: The two rows this module appends. Named so the diff predicates and any
#: reader spell them once.
UNACCOUNTED_REMOVAL_POLICY = "unaccounted-removal"
UNACCOUNTED_CONTENT_POLICY = "unaccounted-content-removal"

#: Cleanup-log policies that account for a SESSION DIRECTORY disappearing. A
#: content row (``delegated_scratchpad``) or a ``hold-release`` says nothing
#: about the directory, so neither clears a vanished session; and the census's
#: own rows are explicitly excluded so a re-scan cannot use its own report as
#: the explanation.
_NON_REMOVAL_POLICIES = frozenset({"delegated_scratchpad", "hold-release"})


def census_path(sessions_dir: Path) -> Path:
    return sessions_dir / CENSUS_NAME


def read_census(sessions_dir: Path) -> dict[str, Any] | None:
    """The previous census, or ``None`` when it is absent or untrustworthy.

    Tolerant on purpose: a corrupt census is a reason to start a new baseline,
    never a reason to fail the maintenance pass. The fail-closed half lives in
    :func:`run_store_census` — an unusable baseline produces no rows, so it
    cannot manufacture reports.
    """
    try:
        raw = census_path(sessions_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_census(sessions_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically publish the census; raises ``OSError`` so the pass retries.

    Tmp beside the target, fsync, replace: a reader sees the old document or
    the new one. A write failure is deliberately NOT swallowed — the caller is
    the store-maintenance pass, whose completion stamp is only written when
    every pass succeeded, so raising makes the next launch retry rather than
    leave a stale baseline that the next diff would misread.
    """
    path = census_path(sessions_dir)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _rows_in_range(path: Path, start: int, end: int) -> list[dict[str, Any]] | None:
    """Parsed log rows in ``[start, end)``, or ``None`` when the range is unreadable.

    Reads BYTES rather than iterating the whole file: the log is append-only
    and the previous census recorded where its last read ended, so the cost of
    a pass is the rows written since, not the log's lifetime size. A range
    that cannot be read (shrunk, rotated, permission denied) returns ``None``
    and the caller rescans from zero — the honest answer is "no coverage",
    not "nothing was logged".
    """
    if end < start:
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(max(0, end - start))
    except FileNotFoundError:
        return [] if start == 0 else None
    except OSError:
        return None
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            # A torn final line (a crash mid-append) is not a row; the next
            # append starts on a fresh line, so skipping it loses nothing that
            # was ever complete.
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_policy(row: dict[str, Any]) -> str:
    policy = row.get("policy")
    return policy if isinstance(policy, str) else ""


def _accounts_for_removal(row: dict[str, Any], session_id: str) -> bool:
    if row.get("session") != session_id:
        return False
    policy = _row_policy(row)
    return (
        policy not in _NON_REMOVAL_POLICIES
        and not policy.startswith("unaccounted-")
        and policy != ""
    )


def _accounts_for_content(row: dict[str, Any], session_id: str) -> bool:
    if row.get("session") != session_id:
        return False
    policy = _row_policy(row)
    if policy == UNACCOUNTED_CONTENT_POLICY:
        return True
    # A planned content row names intent, not a removal: only a completed row
    # (or an old pre-phase row, which was written after the removals) proves
    # the pad's bytes left through the audited content pass.
    return policy == "delegated_scratchpad" and row.get("phase") != "planned"


def _is_live(config_dir: Path, session_id: str) -> bool:
    """Whether a runtime record names this session; ANY failure answers "live".

    Fail closed: the flag exists to explain a disappearance (``live: true`` in
    the row) and to SUPPRESS a content report for a session doing work. A probe
    that cannot answer must not turn "I do not know" into "nobody was using
    it" — that is the direction that produces a confident, wrong report.
    """
    try:
        from local_operator.resume import live_runtime_pid

        return live_runtime_pid(config_dir, session_id) is not None
    except Exception:  # noqa: BLE001 — unknown is "live"
        return True


def _usable_baseline(previous: dict[str, Any] | None) -> dict[str, Any] | None:
    """The previous census iff it ENUMERATED the store; else ``None``.

    An incompletely enumerated baseline is discarded rather than diffed: its
    coverage was partial, so an id it never visited would look like a
    directory that vanished between passes. Per-session content caps do not
    disqualify a baseline (their ids and last-known contents are still known,
    and the content diff skips a capped entry); ``complete`` remains the
    strict truth recorded in the payload. ``log_bytes`` must be a non-negative
    int for the incremental read to mean anything.
    """
    if not isinstance(previous, dict):
        return None
    if previous.get("schema") != CENSUS_SCHEMA:
        return None
    enumerated = previous.get("enumerated", previous.get("complete"))
    if enumerated is not True:
        return None
    if not isinstance(previous.get("sessions"), dict):
        return None
    log_bytes = previous.get("log_bytes")
    if isinstance(log_bytes, bool) or not isinstance(log_bytes, int) or log_bytes < 0:
        return None
    return previous


def _material_pad_shrink(before: dict[str, Any], after: dict[str, Any]) -> bool:
    try:
        pad_before = int(before.get("pad_bytes") or 0)
        pad_after = int(after.get("pad_bytes") or 0)
    except (TypeError, ValueError):
        return False
    if pad_before <= 0 or pad_after >= pad_before:
        return False
    lost = pad_before - pad_after
    if lost >= CENSUS_MATERIAL_PAD_BYTES:
        return True
    if pad_after == 0:
        return True
    return lost / pad_before >= CENSUS_MATERIAL_PAD_FRACTION


def _transcript_vanished(before: dict[str, Any], after: dict[str, Any]) -> bool:
    try:
        return int(before.get("transcript_bytes") or 0) > 0 and int(
            after.get("transcript_bytes") or 0
        ) == 0
    except (TypeError, ValueError):
        return False


def _walk_store(
    config_dir: Path,
    sessions_dir: Path,
    *,
    should_stop: Callable[[], bool] | None,
) -> tuple[dict[str, dict[str, Any]], bool, bool]:
    """Measure the store; ``(sessions, complete, enumerated)``.

    Per-session budget AND a global one. A session whose measurement capped is
    still recorded (a lower bound is honest) and makes the pass ``complete =
    False``; it does not affect ``enumerated``, because the directory WAS
    visited and its existence is known. Only an early stop or an exhausted
    global budget ends enumeration, and only that suppresses the diff.
    """
    sessions: dict[str, dict[str, Any]] = {}
    remaining = CENSUS_MAX_TOTAL_ENTRIES
    complete = True
    enumerated = True
    try:
        with os.scandir(sessions_dir) as scanned:
            entries = [entry for entry in scanned]
    except OSError as exc:
        logger.warning("session census: cannot scan %s: %s", sessions_dir, exc)
        return {}, False, False
    for entry in sorted(entries, key=lambda item: item.name):
        if should_stop is not None and should_stop():
            complete = False
            enumerated = False
            break
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        if remaining <= 0:
            complete = False
            enumerated = False
            break
        budget = min(CENSUS_MAX_ENTRIES_PER_SESSION, remaining, CONTENTS_MAX_ENTRIES)
        measured = measure_session_dir(Path(entry.path), max_entries=budget)
        sessions[entry.name] = measured
        remaining -= int(measured.get("entries") or 0)
        if measured.get("contents_complete") is False:
            complete = False
    return sessions, complete, enumerated


def run_store_census(
    config_dir: Path,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Take a census, report unaccounted changes since the last one, publish it.

    Returns a small summary (``count``/``reported``/``complete``/
    ``enumerated``) for logs and tests. Every side effect is outside session
    directories: cleanup-log appends and the atomic census write. See the
    module docstring for why an incomplete pass reports nothing.
    """
    from local_operator.session.cleanup import _append_cleanup_log

    config_root = Path(config_dir)
    sessions_dir = config_root / SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        return {"count": 0, "reported": 0, "complete": True}

    previous = _usable_baseline(read_census(sessions_dir))
    sessions, complete, enumerated = _walk_store(
        config_root, sessions_dir, should_stop=should_stop
    )

    log_path = sessions_dir / CLEANUP_LOG_NAME
    current_size = _log_size(log_path)
    reported = 0
    if previous is not None and enumerated:
        start = previous["log_bytes"]
        if start > current_size:
            # The log shrank (rotation, truncation): the previous offset names
            # nothing reliable, so coverage restarts at zero.
            start = 0
        rows = _rows_in_range(log_path, start, current_size)
        if rows is None:
            rows = _rows_in_range(log_path, 0, current_size) or []
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        previous_sessions = previous["sessions"]
        on_disk = set(sessions)
        for session_id in sorted(previous_sessions):
            if session_id in on_disk:
                continue
            if any(_accounts_for_removal(row, session_id) for row in rows):
                continue
            before = previous_sessions.get(session_id)
            before = before if isinstance(before, dict) else {}
            record = {
                "at": now,
                "session": session_id,
                "title": "",
                "policy": UNACCOUNTED_REMOVAL_POLICY,
                "reason": "session directory vanished without a removal record",
                "actor": "census",
                "pid": os.getpid(),
                "path": os.fspath((sessions_dir / session_id).absolute()),
                "live": _is_live(config_root, session_id),
                **_contents_fields(before),
            }
            if _append_cleanup_log(sessions_dir, record):
                reported += 1
        for session_id in sorted(on_disk):
            before = previous_sessions.get(session_id)
            before = before if isinstance(before, dict) else None
            if before is None:
                continue
            after = sessions[session_id]
            # An unreadable measurement is not evidence of a loss: skip rather
            # than report a shrink the walk could not actually see.
            if before.get("contents_complete") is False or after.get("contents_complete") is False:
                continue
            if after.get("contents_error") or before.get("contents_error"):
                continue
            shrunk = _material_pad_shrink(before, after)
            transcript_gone = _transcript_vanished(before, after)
            if not (shrunk or transcript_gone):
                continue
            if any(_accounts_for_content(row, session_id) for row in rows):
                continue
            if _is_live(config_root, session_id):
                # A live session's pad is being written by definition; its
                # bytes moving is not a loss, and the content pass could not
                # have touched it either.
                continue
            record = {
                "at": now,
                "session": session_id,
                "title": "",
                "policy": UNACCOUNTED_CONTENT_POLICY,
                "reason": "session content vanished without a removal record",
                "actor": "census",
                "pid": os.getpid(),
                "path": os.fspath((sessions_dir / session_id).absolute()),
                "live": False,
                "before": _contents_fields(before),
                "after": _contents_fields(after),
                "transcript_vanished": transcript_gone,
            }
            if _append_cleanup_log(sessions_dir, record):
                reported += 1

    payload = {
        "schema": CENSUS_SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # The size AFTER any rows this pass appended: the next pass starts
        # reading where this one finished, so its own report cannot be read
        # back as someone else's explanation.
        "log_bytes": _log_size(log_path),
        "count": len(sessions),
        "sessions": sessions,
        "complete": complete,
        # The existence-diff subset; see the module docstring on why a
        # per-session content cap must not silence the audit.
        "enumerated": enumerated,
    }
    _write_census(sessions_dir, payload)
    logger.debug(
        "session census: %d sessions, complete=%s enumerated=%s, %d unaccounted report(s)",
        len(sessions),
        complete,
        enumerated,
        reported,
    )
    return {
        "count": len(sessions),
        "reported": reported,
        "complete": complete,
        "enumerated": enumerated,
    }


def _contents_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """The census's per-session fields, projected for a cleanup-log row.

    Keeps the row's vocabulary identical to ``remove_session_dir``'s receipt
    (``transcript_bytes``/``pad_bytes``/...), so one reader handles a recorded
    removal and an unaccounted one without a second schema.
    """
    fields: dict[str, Any] = {}
    for key in (
        "transcript_bytes",
        "pad_bytes",
        "pad_entries",
        "bytes",
        "entries",
        "origin",
        "last_activity_at",
        "contents_complete",
    ):
        if key in entry:
            fields[key] = entry[key]
    return fields
