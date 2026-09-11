"""PROTOTYPE — THROWAWAY. Read-only session data for the /resume picker variants.

See ``~/workspace/PROPOSAL-resume-picker.md``. This module is the ONLY route to
disk for the three variants: every accessor is cached per session id and safe to
call on each cursor move, and every ``open`` here is mode ``"rb"``. Nothing in
this file writes to, resumes, or mutates the session store.

Missing or unreadable transcripts return the empty value rather than raising —
that is the only error handling in this prototype.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_operator.resume import SessionRow, recent_session_rows
from local_operator.session.creation import session_created_at
from local_operator.session.search_index import build_index

#: A seek to a byte offset lands mid-line, so the tail window is always one
#: partial line longer than it looks. Mirrors ``resume.session_preview``.
TAIL_BYTES = 256_000

TRANSCRIPT_NAME = "transcript.jsonl"


class PreviewData:
    """Bounded, cached, read-only reads of the real session store."""

    def __init__(self, store: Path) -> None:
        self._store = store
        self._sessions = store / "sessions"
        self._rows: list[SessionRow] | None = None
        self._digests: dict[str, str] | None = None
        self._created: dict[str, float] = {}
        self._lines: dict[str, list[str]] = {}
        self._condensed: dict[str, list[tuple[str, str, float]]] = {}
        self._verbose: dict[str, list[tuple[str, str, float]]] = {}
        self._checkpoint: dict[str, dict[str, Any]] = {}

    # -- whole-store reads, paid once -------------------------------------

    def rows(self) -> list[SessionRow]:
        """Every resumable session, newest first. ~79 ms cold for 140 rows."""
        if self._rows is None:
            self._rows = recent_session_rows(self._store, limit=None)
        return self._rows

    def digests(self) -> dict[str, str]:
        """``{sid: searchable digest}``. ~11 ms for 140 sessions."""
        if self._digests is None:
            try:
                self._digests = build_index(self._store, [r.id for r in self.rows()])
            except Exception:
                self._digests = {}
        return self._digests

    # -- per-session reads -------------------------------------------------

    def created_at(self, sid: str) -> float:
        """Session birth time, 0.0 when unrecoverable. ``SessionRow`` never has it."""
        if sid not in self._created:
            try:
                self._created[sid] = session_created_at(self._sessions / sid)
            except Exception:
                self._created[sid] = 0.0
        return self._created[sid]

    def _tail(self, sid: str) -> list[str]:
        """The last :data:`TAIL_BYTES` of the transcript, as whole lines."""
        if sid in self._lines:
            return self._lines[sid]
        transcript = self._sessions / sid / TRANSCRIPT_NAME
        window = b""
        try:
            size = transcript.stat().st_size
            with transcript.open("rb") as handle:
                if size > TAIL_BYTES:
                    handle.seek(size - TAIL_BYTES)
                    # The seek landed at an arbitrary byte: the first line is a
                    # fragment with nothing recoverable in it.
                    _, _, window = handle.read().partition(b"\n")
                else:
                    window = handle.read()
        except OSError:
            window = b""
        self._lines[sid] = window.decode("utf-8", errors="replace").splitlines()
        return self._lines[sid]

    def _entries(self, sid: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in self._tail(sid):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
        return out

    @staticmethod
    def _text(payload: dict[str, Any]) -> str:
        # `attachment` blocks carry no `text` key and no `type` key, so a bare
        # b["text"] raises and a predicate on b["type"] matches nothing.
        return "".join(b.get("text", "[attachment]") for b in payload.get("content") or [])

    def condensed(self, sid: str) -> list[tuple[str, str, float]]:
        """Human turns only, oldest-first ``(role, text, ts)``. ~0.79 ms/session."""
        if sid in self._condensed:
            return self._condensed[sid]
        out: list[tuple[str, str, float]] = []
        for entry in self._entries(sid):
            payload = entry.get("payload") or {}
            text = self._text(payload)
            if (
                entry.get("type") == "message"
                and payload.get("kind") == "message"
                and payload.get("role") in ("user", "assistant")
                and text.strip()
            ):
                out.append((str(payload.get("role")), text, float(entry.get("ts") or 0.0)))
        self._condensed[sid] = out
        return out

    def verbose(self, sid: str) -> list[tuple[str, str, float]]:
        """Every message entry including ``role == "tool"`` and ``kind == "custom"``."""
        if sid in self._verbose:
            return self._verbose[sid]
        out: list[tuple[str, str, float]] = []
        for entry in self._entries(sid):
            payload = entry.get("payload") or {}
            text = self._text(payload)
            if entry.get("type") == "message" and text.strip():
                out.append((str(payload.get("role")), text, float(entry.get("ts") or 0.0)))
        self._verbose[sid] = out
        return out

    def checkpoint(self, sid: str) -> dict[str, Any]:
        """Newest ``frontend_state_checkpoint_v1`` state, or ``{}`` on 19% of rows.

        The discriminator is ``payload.custom_type`` — ``payload.kind`` is None
        on these entries. Callers must render ``{}`` as reserved placeholders,
        never a collapsed line.
        """
        if sid in self._checkpoint:
            return self._checkpoint[sid]
        state: dict[str, Any] = {}
        for entry in self._entries(sid):
            payload = entry.get("payload") or {}
            if (
                entry.get("type") == "custom"
                and payload.get("custom_type") == "frontend_state_checkpoint_v1"
            ):
                found = (payload.get("details") or {}).get("state")
                if isinstance(found, dict):
                    state = found
        self._checkpoint[sid] = state
        return state

    def human_ts(self, sid: str) -> float | None:
        """Timestamp of the newest human-visible turn, or ``None``."""
        rows = self.condensed(sid)
        return rows[-1][2] if rows else None

    def msg_count(self, sid: str) -> int:
        return len(self.condensed(sid))

    def grep_context(self, sid: str, query: str, width: int = 150) -> str | None:
        """An ellipsised window around an EXACT substring hit in the digest.

        ``None`` means the row matched softly/fuzzily with no literal substring,
        which the variant renders as a ``~`` mark and no context line.
        """
        digest = self.digests().get(sid)
        if not digest or not query:
            return None
        i = digest.lower().find(query.lower())
        if i < 0:
            return None
        start = max(0, i - (width - len(query)) // 2)
        end = min(len(digest), start + width)
        snippet = " ".join(digest[start:end].split())
        return f"{'…' if start > 0 else ''}{snippet}{'…' if end < len(digest) else ''}"
