"""Response models for the Library (design §4).

Stance follows ``desktop_session_files``: ``extra="allow"`` and a default on
every field, so a backend can widen a row additively and an older renderer
still parses it. Nothing here carries a filesystem path except ``ref.path`` on
the desktop projection — the phone/mesh form is the same model with that field
``None`` and ``display_path`` populated.

The models are deliberately the WIRE shape, not internal dataclasses: a field
rename in ``library/service.py`` must fail a route test rather than silently
change what four surfaces consume.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LibraryRef(BaseModel):
    """Where the bytes are. ``path`` is absolute ONLY on the desktop route."""

    model_config = ConfigDict(extra="allow")

    type: Literal["path", "artifact"] = "path"
    path: str | None = None
    display_path: str = ""
    digest: str | None = None
    content_type: str = "application/octet-stream"


class LibraryTagBrief(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    category: str = "custom"


class LibraryOccurrence(BaseModel):
    """One conversation that saw the asset.

    ``first_seq``/``last_seq`` let a surface link back to the messages that
    produced or named it, which is the provenance the operator asked for;
    ``child_id`` names the subagent run when the occurrence came from one.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str = ""
    session_title: str = ""
    origin: str = "mentioned"
    tool: str = ""
    at: float = 0.0
    intent: str | None = None
    first_seq: int = 0
    last_seq: int = 0
    child_id: str | None = None
    label: str | None = None


class LibraryChat(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str = ""
    title: str = ""
    updated_at: float = 0.0


class LibraryOrganized(BaseModel):
    model_config = ConfigDict(extra="allow")

    at: float = 0.0
    by: str | None = None
    confidence: float | None = None


class LibraryPreview(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: Literal["ready", "unsupported"] = "unsupported"
    mime_type: str = "application/octet-stream"
    pages: int | None = None


class LibraryRow(BaseModel):
    """One catalog row (§4's JSON, field for field)."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    ref: LibraryRef = Field(default_factory=LibraryRef)
    name: str = ""
    derived_name: str = ""
    description: str = ""
    kind: str = "other"
    category: str = "other"
    origin: str = "mentioned"
    origin_group: str = "mentioned"
    origins: list[str] = Field(default_factory=list)
    tags: list[LibraryTagBrief] = Field(default_factory=list)
    primary_tag: str | None = None
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    created_at: float = 0.0
    modified_at: float = 0.0
    last_seen_at: float = 0.0
    exists: bool = False
    highlighted: bool = False
    highlight_why: str | None = None
    occurrences: list[LibraryOccurrence] = Field(default_factory=list)
    session_count: int = 0
    chat: LibraryChat | None = None
    preview: LibraryPreview = Field(default_factory=LibraryPreview)
    organized: LibraryOrganized | None = None
    locked: bool = False
    favorite: bool = False
    archived: bool = False


class LibraryScan(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessions_total: int = 0
    sessions_scanned: int = 0


class LibraryListing(BaseModel):
    """``result`` of the list route."""

    model_config = ConfigDict(extra="allow")

    state: Literal["ready", "building"] = "ready"
    revision: int = 0
    total: int = 0
    counts: dict[str, dict[str, int]] = Field(
        default_factory=lambda: {"category": {}, "origin_group": {}, "kind": {}}
    )
    tags: list[dict[str, Any]] = Field(default_factory=list)
    rows: list[LibraryRow] = Field(default_factory=list)
    next_cursor: str | None = None
    cursor_reset: bool = False
    scan: LibraryScan = Field(default_factory=LibraryScan)
    #: Present once the Librarian package lands; a surface must not assume the
    #: key is there before that slice ships.
    librarian: dict[str, Any] | None = None


class LibraryTag(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    category: str = "custom"
    created_by: str = "user"
    count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


class LibraryAssetPatch(BaseModel):
    """PATCH body. Every field optional; ``None`` means "not supplied"."""

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    favorite: bool | None = None
    locked: bool | None = None
    archived: bool | None = None


class LibraryTagsWrite(BaseModel):
    model_config = ConfigDict(extra="allow")

    tags: list[str] = Field(default_factory=list)
    primary: str | None = None


class LibraryTagCreate(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=24)
    category: str = "custom"


class LibraryTagPatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, min_length=1, max_length=24)
    category: str | None = None


class LibraryTagMerge(BaseModel):
    model_config = ConfigDict(extra="allow")

    into: str = ""


class LibraryChatWrite(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str = ""


class LibraryChatReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str = ""
    chat: LibraryChat | None = None
    cleared: bool = False


__all__ = [
    "LibraryAssetPatch",
    "LibraryChat",
    "LibraryChatReceipt",
    "LibraryChatWrite",
    "LibraryListing",
    "LibraryOccurrence",
    "LibraryOrganized",
    "LibraryPreview",
    "LibraryRef",
    "LibraryRow",
    "LibraryScan",
    "LibraryTag",
    "LibraryTagBrief",
    "LibraryTagCreate",
    "LibraryTagMerge",
    "LibraryTagPatch",
    "LibraryTagsWrite",
]
