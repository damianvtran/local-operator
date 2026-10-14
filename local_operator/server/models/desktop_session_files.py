"""Response models for the per-session file index.

``GET /v1/desktop/sessions/{session_id}/files`` and its ``POST …/{file_id}/materialise`` child
(``server/routes/desktop_session_files.py``; the rules live in ``session_files/service.py``).

STANCE, following ``desktop_code_requests``: ``extra="allow"`` and a default on every field, so
the backend can widen a row additively without breaking an older renderer's strict parse, and
a renderer built against a NEWER backend can ignore what it does not know. The wire is
documented in ``docs/DESKTOP_API.md`` ("Session files").

What is deliberately NOT here: file CONTENT (artifact bytes stay on the existing
``…/attachments/{digest}`` routes; a path row's bytes stay on the renderer's file bridge), the
scanner's machinery (byte offsets, cache signatures) and, on a path row, anything but the one
path the producing tool named.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FileRef(BaseModel):
    """Where the bytes are: a filesystem path, or a digest in the attachment store.

    ``type == "path"``: ``path`` is ABSOLUTE on the desktop route (``~/``/cwd-relative only
    when the service is asked for its display form). ``type == "artifact"``: ``digest`` +
    ``content_type``; NO path exists until ``…/materialise`` copies one out.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["path", "artifact"]
    path: str | None = None
    digest: str | None = None
    content_type: str | None = None


class FileSession(BaseModel):
    """Whose journal produced the file. ``child_id`` selects the child attachment route."""

    model_config = ConfigDict(extra="allow")

    id: str
    role: Literal["self", "subagent", "peer"] = "self"
    child_id: str | None = None
    label: str | None = None


class FileRow(BaseModel):
    """One file (or digest-only artifact) the conversation or one of its subagents produced."""

    model_config = ConfigDict(extra="allow")

    id: str
    ref: FileRef
    name: str
    kind: str
    #: Stat-refreshed on page rows; the size recorded at event time when the file is gone.
    size_bytes: int | None = None
    #: Seconds since the epoch. ``created_at`` is the filesystem birth time when the platform
    #: reports one, else the session's first touch; ``modified_at`` is the file's mtime, else
    #: the session's last touch. Both fall back to journal event times for a file that is gone
    #: and for artifacts, so every row has a value to sort by.
    created_at: float | None = None
    modified_at: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    origin: str
    tool: str
    tool_call_id: str | None = None
    intent: str | None = None
    first_seq: int = 0
    last_seq: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    touches: int = 1
    highlighted: bool = False
    highlight_why: str | None = None
    session: FileSession
    #: ``lstat``-style existence at response time (path rows) or store presence (artifacts).
    #: ``None`` only when it could not be determined. Point-in-time, PAGE ROWS ONLY: do not
    #: carry it past the page it arrived on.
    exists: bool | None = None


class FilesScan(BaseModel):
    """How much of the journal the answer reflects."""

    model_config = ConfigDict(extra="allow")

    rows: int = 0
    complete: bool = True
    children: int = 0
    children_skipped: int = 0
    #: ``True`` when the per-session record cap (50 000) dropped files; counted, never silent.
    truncated: bool = False
    #: Journal bytes covered / present: progress for a ``building`` answer.
    covered_bytes: int = 0
    total_bytes: int = 0


class FilesListing(BaseModel):
    """``data`` of the list route."""

    model_config = ConfigDict(extra="allow")

    #: ``ready`` (complete), ``building`` (partial rows; poll again; the scan continues in the
    #: background) or ``unsupported`` (a peer's conversation: the journal is not on this disk).
    state: Literal["ready", "building", "unsupported"]
    #: Monotone per session while the journal only grows; changes whenever the index changes.
    revision: int = 0
    scope: Literal["family", "self"] = "family"
    sort: str = "recent"
    order: Literal["asc", "desc"] = "desc"
    #: Rows matching EVERY filter, ``kind`` included: the length of the paged-through list.
    total: int = 0
    #: Per-kind census of the rows matching the other filters (q, origin, highlighted), BEFORE
    #: the ``kind`` filter, so a chip row can show every kind's count while one is selected.
    counts: dict[str, int] = Field(default_factory=dict)
    rows: list[FileRow] = Field(default_factory=list)
    next_cursor: str | None = None
    #: ``True`` when the supplied cursor belonged to an older revision and the page restarted
    #: at the top; the client should replace its rows, not append.
    cursor_reset: bool = False
    #: ``False`` when a stat-derived sort ran out of its time budget, so some rows sorted by
    #: their recorded values instead of the file's own.
    stat_complete: bool = True
    scan: FilesScan = Field(default_factory=FilesScan)


class MaterialiseReceipt(BaseModel):
    """``data`` of the materialise route."""

    model_config = ConfigDict(extra="allow")

    path: str
    name: str = ""
    size_bytes: int | None = None
    #: ``False`` when an identical copy already existed and was reused.
    created: bool = True


__all__ = [
    "FileRef",
    "FileRow",
    "FileSession",
    "FilesListing",
    "FilesScan",
    "MaterialiseReceipt",
]
