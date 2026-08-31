"""Per-session "last seen by phone" state for the mobile daemon.

The phone's session list needs an unread indicator: a session is UNSEEN when
it has user-visible activity newer than the last time the phone looked at it.
This module is the durable half of that verdict — the timestamps survive a
daemon restart, so a phone that read a session yesterday does not see it
re-marked unread today just because the daemon recycled.

Two timestamps per session, only one of them persisted:

- ``last_seen`` — written by ``POST /api/sessions/{id}/seen`` when the phone
  opens (or re-views) a session. Activity newer than it is unseen. THIS is
  the only durable fact: it is the user's own action and must outlive the
  daemon.
- ``baseline`` — the transcript mtime at the moment the daemon FIRST observed
  the session, kept in memory only. It exists so an upgrade or restart does
  not light up every session in the store: a session the phone has never
  marked seen is unseen only if it gained activity since first observation,
  not because it exists. Re-deriving it after a restart is CORRECT rather
  than a loss: the baseline of a never-seen session is simply "the activity
  clock at first observation", which the re-derivation reproduces, and
  activity that happened while the daemon was down is deliberately not
  unread (that is the no-flood-on-upgrade rule, applied per restart).

The activity clock is the transcript file's mtime, which the summaries scan
already stats for its sort key — the verdict costs no extra syscall.

Persistence is one small JSON file under ``config_dir()`` (0600, atomic
replace, bounded), written ONLY on ``mark_seen``: a user action whose effect
the phone's next list paint must already reflect.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: The store file's name, directly under ``config_dir()`` beside the other
#: owner-private state.
SEEN_STORE_NAME = "mobile-seen.json"

#: Bound on tracked sessions. The list surface shows the 100 most recent
#: durable sessions plus the live ones; 4096 is far beyond anything a phone
#: could mark seen while still bounding the file for a long-lived daemon.
#: Overflow drops the oldest stamps — a verdict for a session that old is
#: uninteresting, and the baseline rule re-derives itself on re-observation.
MAX_SEEN_ENTRIES = 4096


class SeenStore:
    """The daemon's per-session seen state, persisted across restarts."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self._baselines: dict[str, float] = {}
        self._load()

    # -- verdicts -------------------------------------------------------------

    def is_unseen(self, session_id: str, activity_mtime: float) -> bool:
        """Whether ``session_id`` has activity newer than the phone's view.

        Records the baseline on first observation — that side effect is what
        keeps an upgrade from marking the whole store unread (see the module
        docstring). ``activity_mtime`` is the transcript's mtime, which the
        caller already has from the listing scan. Memory-only; nothing here
        touches the disk.
        """
        with self._lock:
            self._baselines.setdefault(session_id, activity_mtime)
            seen_at = self._last_seen.get(session_id)
            if seen_at is not None:
                return activity_mtime > seen_at
            return activity_mtime > self._baselines[session_id]

    def mark_seen(self, session_id: str, *, now: float | None = None) -> None:
        """The phone viewed ``session_id``; persist immediately."""
        stamp = now if now is not None else time.time()
        with self._lock:
            self._last_seen[session_id] = stamp
            # A session marked seen has, by definition, been observed.
            self._baselines.setdefault(session_id, stamp)
            self._bound_locked()
            self._persist_locked()

    def last_seen(self, session_id: str) -> float | None:
        with self._lock:
            return self._last_seen.get(session_id)

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            # A torn or unreadable store degrades to "nothing seen yet"; the
            # baseline rule then re-derives itself on first observation, so
            # the worst case is a session lighting up once, not a crash.
            logger.warning("mobile seen store unreadable; starting fresh", exc_info=True)
            return
        sessions = raw.get("sessions") if isinstance(raw, dict) else None
        if not isinstance(sessions, dict):
            return
        for session_id, stamp in sessions.items():
            if isinstance(stamp, (int, float)):
                self._last_seen[str(session_id)] = float(stamp)

    def _persist_locked(self) -> None:
        """Atomic 0600 write: temp file in the same directory, then replace.

        The replace guarantees a reader sees either the old file or the new
        one, never a half-written one; the chmod happens BEFORE the replace
        so the store is never world-readable even for an instant.
        """
        payload = {"sessions": dict(sorted(self._last_seen.items()))}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, tmp_name = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=f".{SEEN_STORE_NAME}.", suffix=".tmp"
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"))
                os.chmod(tmp_name, 0o600)
                os.replace(tmp_name, self._path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError:
            # The verdicts stay correct in memory; a failed write only costs
            # them on restart, where the baseline rule re-derives safely.
            logger.warning("mobile seen store write failed", exc_info=True)

    def _bound_locked(self) -> None:
        """Drop the oldest stamps past ``MAX_SEEN_ENTRIES``."""
        excess = len(self._last_seen) - MAX_SEEN_ENTRIES
        if excess <= 0:
            return
        ranked = sorted(self._last_seen, key=self._last_seen.get)  # type: ignore[arg-type]
        for session_id in ranked[:excess]:
            self._last_seen.pop(session_id, None)
