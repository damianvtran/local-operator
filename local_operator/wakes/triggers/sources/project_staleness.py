"""Project staleness: a tracked project going quiet earns a check-in.

THE RULE, and it is deliberately a MIRROR of the one authority rather than a
second opinion: a row is stale iff its status is one of the LIVE statuses —
``planning``/``active``/``qa``/``validation`` — and it has no progress text, or
no ``progress_updated_at``, or its progress is older than the configured
window. The store's own ``progress_is_stale`` (``local_operator/projects.py``)
is the authority and re-verifies at consume; this copy exists because the
supervisor must not import the harness, and the copy reads THE SAME threshold
from the published settings snapshot (``projects.stale_after_hours``), so the
two cannot disagree about a number.

WHY THE ROW IS READ AS JSON. The projects store is ``<config>/projects/<id>.json``
files written through ``ProjectRegistry``; a read-only stdlib parse of those
rows is the smallest honest thing the supervisor can do, and any row it cannot
parse is skipped (one bad file costs one candidate; the store's own readers are
the authority for repair).

THE FINGERPRINT is the ``(id, status, int(progress_updated_at))`` shape — the
same first three elements the completion-time check's latch carries (that
latch adds the refresh assertion as a fourth; this stdlib reader keeps the
three a published snapshot can see) — so "what counts as a new stale episode"
cannot mean two different things: a status move or a NEW content line moves
it, and a refresh deliberately does not (a check-in is not an update — the
store's own content clock reads the same way); milestone edits and other
metadata do not.

LIVENESS IS A FLOOR, NOT A VERDICT. The payload annotates each linked session
with the smallest honest signal a stdlib reader can get: ``live``/``wedged``
from the runtime registry when it answers, else ``cold`` (a record-less
session) or ``missing`` (no directory), plus the newest mtime among the
session's transcript and spool. "Stalled" is deliberately NOT computed here —
the decision to resume belongs to the consuming engine, which has probes this
source does not.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_operator.wakes import triggers as _triggers
from local_operator.wakes.triggers import TriggerContext, TriggerInstance

logger = logging.getLogger(__name__)

#: The registry name; stable, because records and log lines carry it.
NAME = "project_staleness"

#: The source's own kill switch, beside the rule it gates. The settings
#: anti-drift test pins the registry row's default to this constant.
DEFAULT_ENABLED = True

ENABLED_KEY = "wakes.triggers.project_staleness.enabled"

#: The seconds window when the snapshot is absent, mirroring
#: ``projects.PROJECT_PROGRESS_STALE_S`` (4 h). The supervisor cannot import
#: ``local_operator.projects`` (harness weight), so the value is restated
#: here and pinned to that constant by the source's unit tests.
_DEFAULT_STALE_S = 14400.0

#: The settings key the window is read from — the SAME key every rendered
#: reader resolves through ``projects.stale_after_s``.
STALE_AFTER_HOURS_KEY = "projects.stale_after_hours"

#: Statuses that can read stale: the store's ``PROJECT_LIVE_STATUSES``. The
#: copy is deliberate (see the module docstring) and pinned by tests.
_LIVE_STATUSES = frozenset({"planning", "active", "qa", "validation"})

#: Bounds mirroring the store's read path: stop after this many rows (one bad
#: store is a runaway writer, and a pass must stay bounded) and cap the
#: annotated session list per row.
_MAX_ROWS = 500
_MAX_SESSIONS = 64

#: The two per-session files whose newest mtime is the activity floor.
_TRANSCRIPT_NAME = "transcript.jsonl"
_INBOX_NAME = "inbox.jsonl"


class ProjectStalenessSource:
    """The v1 source: one module, no state of its own."""

    name = NAME

    def enabled(self, values: Mapping[str, Any]) -> bool:
        return _triggers._strict_bool(values.get(ENABLED_KEY), DEFAULT_ENABLED)

    def evaluate(self, ctx: TriggerContext) -> Sequence[TriggerInstance]:
        """Evaluate every readable project row; return the stale subset."""
        root = Path(ctx.config_dir)
        stale_s = stale_after_s(ctx.values)
        now_s = ctx.now_ms / 1000.0
        rows = _read_rows(root)
        if not rows:
            return []
        stale = [row for row in rows if _is_stale(row, now_s=now_s, stale_s=stale_s)]
        if not stale:
            return []
        # ONE runtime scan for the whole pass, and only once something is
        # actually stale: the scan reaps dead records (a write), so a pass with
        # nothing to report must stay a pure read.
        states = _runtime_states(root)
        return [_instance(root, row, now_s=now_s, states=states) for row in stale]


SOURCE = ProjectStalenessSource()


def stale_after_s(values: Mapping[str, Any]) -> float:
    """The configured staleness window in seconds, or the default.

    Reads ``projects.stale_after_hours`` from the published snapshot. The
    registry enforces 1–168 on writes; a value that is not a positive integer
    anyway (a hand-edited snapshot) falls back to the default rather than
    inventing an extreme window.
    """
    raw = values.get(STALE_AFTER_HOURS_KEY)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return _DEFAULT_STALE_S
    hours = float(raw)
    if hours <= 0:
        return _DEFAULT_STALE_S
    return hours * 3600.0


def _read_rows(root: Path) -> list[dict[str, Any]]:
    """Every readable project row, bounded, in file-name order."""
    out: list[dict[str, Any]] = []
    try:
        children = sorted((root / "projects").iterdir(), key=lambda path: path.name)
    except OSError:
        # Absent store (never written) and unreadable store are both "no
        # candidates": harness-only installs pay one stat.
        return out
    for child in children:
        # The same neighbourhood rules ProjectRegistry._load applies: dot
        # entries are lock/temp files, symlinks are refused, and only .json
        # files are rows.
        if child.name.startswith(".") or child.is_symlink() or not child.is_file():
            continue
        if not child.name.endswith(".json"):
            continue
        if len(out) >= _MAX_ROWS:
            logger.warning(
                "project store at %s has more than %d rows; the rest are not evaluated",
                root / "projects",
                _MAX_ROWS,
            )
            break
        try:
            data = json.loads(child.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _is_stale(row: Mapping[str, Any], *, now_s: float, stale_s: float) -> bool:
    """The staleness rule, in the store's exact shape."""
    status = row.get("status")
    status = status if isinstance(status, str) else "active"
    if status not in _LIVE_STATUSES:
        return False
    progress = row.get("progress")
    # EXACTLY the store's rule: a falsy progress ("" or absent) is stale by
    # construction; whitespace is truthy THERE and must be here too, or the
    # two rules would disagree on a one-character row.
    if not isinstance(progress, str) or not progress:
        return True
    stamp = row.get("progress_updated_at")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return True
    return (now_s - float(stamp)) > stale_s


def _instance(
    root: Path,
    row: Mapping[str, Any],
    *,
    now_s: float,
    states: Mapping[str, str],
) -> TriggerInstance:
    """One stale row → one instance (the payload the message renders from)."""
    project_id = str(row.get("id") or "")
    status = row.get("status")
    status = status if isinstance(status, str) else "active"
    stamp = row.get("progress_updated_at")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        stamp = None
    progress_age_s = int(max(0.0, now_s - float(stamp))) if stamp is not None else None
    anchor = float(stamp) if stamp is not None else _number(row.get("created_at"))
    age_s = max(0.0, now_s - anchor) if anchor is not None else 0.0
    return TriggerInstance(
        source=NAME,
        key=project_id,
        fingerprint=(project_id, status, int(stamp or 0)),
        payload={
            "display_name": _display_name(row),
            "status": status,
            "progress_age_s": progress_age_s,
            "sessions": _sessions(root, row, states=states),
        },
        age_s=age_s,
    )


def _display_name(row: Mapping[str, Any]) -> str:
    """``title`` when set, else ``name`` — the store's own precedence."""
    title = row.get("title")
    name = row.get("name")
    return str(title or name or "")


def _number(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _sessions(
    root: Path, row: Mapping[str, Any], *, states: Mapping[str, str]
) -> list[dict[str, Any]]:
    """The linked sessions, each annotated with the floor liveness signal."""
    raw = row.get("sessions")
    out: list[dict[str, Any]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return out
    for session_id in list(raw)[:_MAX_SESSIONS]:
        if not isinstance(session_id, str) or not session_id:
            continue
        directory = root / "sessions" / session_id
        if not directory.is_dir():
            out.append({"id": session_id, "last_activity_age_s": None, "live": "missing"})
            continue
        out.append(
            {
                "id": session_id,
                "last_activity_age_s": _last_activity_age_s(directory),
                "live": states.get(session_id, "cold"),
            }
        )
    return out


def _last_activity_age_s(directory: Path) -> int | None:
    """The newest mtime among the session's transcript and spool, or ``None``.

    A FLOOR on idleness, not a verdict: files can be touched by non-turn
    traffic, and a session with neither file reads ``None`` (unknown) rather
    than a fabricated age.
    """
    newest: float | None = None
    for name in (_TRANSCRIPT_NAME, _INBOX_NAME):
        try:
            stamp = (directory / name).stat().st_mtime
        except OSError:
            continue
        newest = stamp if newest is None else max(newest, stamp)
    if newest is None:
        return None
    return int(max(0.0, time.time() - newest))


def _runtime_states(root: Path) -> dict[str, str]:
    """``session id → live|wedged`` for every runtime whose record answers.

    Best-effort and function-local: the registry is harness-adjacent, so the
    sweep pays for it only when something is stale, and any failure leaves
    every session reading ``cold`` — the direction that asks the engine to
    look, never the one that claims a dead session is alive.
    """
    states: dict[str, str] = {}
    try:
        from local_operator.session.runtime.registry import scan

        for record, state in scan(root):
            session_id = getattr(record, "session_id", "")
            if not isinstance(session_id, str) or not session_id:
                continue
            # PRIORITY live > wedged > cold: two records for one session are a
            # transient, and the reading that must win is the strongest answer
            # the registry gave (a session with one answering runtime is
            # served, never "wedged" because a dead sibling was seen first).
            mapped = state if state in ("live", "wedged") else "cold"
            current = states.get(session_id)
            if mapped == "live" or (mapped == "wedged" and current != "live"):
                states[session_id] = mapped
            elif current is None:
                states[session_id] = mapped
    except Exception:  # noqa: BLE001 — liveness is an annotation, never the pass
        logger.debug("could not scan the runtime registry", exc_info=True)
    return states
