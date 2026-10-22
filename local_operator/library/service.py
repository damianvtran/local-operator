"""Query, projection and content for the Library.

The service is the ONE place that decides what a row means:
``effective name = user > librarian > derived`` and the same for description;
``origin_group`` is the three-way projection of the scanner's raw origins; a
``display=True`` projection drops absolute paths for phone/mesh readers.

Content is resolved from an asset id ONLY. A caller never supplies a path:
the id is looked up in the derived catalog, the indexed path is re-checked
against the sensitive-path denylist, and digest rows read the attachment store
by digest. That is the whole security model (§8) and it is enforced here rather
than in each of the four surface routes.

Blocking: every function in this module may touch the filesystem or SQLite.
Callers on the event loop run them through ``asyncio.to_thread``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from local_operator.library import classify, identity
from local_operator.library import index as library_index
from local_operator.library import store as library_store
from local_operator.library.indexer import ensure_started
from local_operator.library.metadata import FIELD_SOURCES, MetadataStore, TagRefused
from local_operator.library.store import LibraryStore
from local_operator.session.attachments import AttachmentStore
from local_operator.session_files import service as files_service
from local_operator.session_files.scan import ORIGINS as _SCAN_ORIGINS

logger = logging.getLogger(__name__)

SORTS: Final = ("recent", "name", "kind", "size", "created", "modified", "organized")
ORDERS: Final = ("asc", "desc")
DEFAULT_ORDER: Final = {
    "recent": "desc",
    "name": "asc",
    "kind": "asc",
    "size": "desc",
    "created": "desc",
    "modified": "desc",
    "organized": "desc",
}
DEFAULT_LIMIT: Final = 100
MAX_LIMIT: Final = 200
MAX_QUERY_CHARS: Final = 256

TAG_MODES: Final = ("all", "any")
#: What the listing is for. ``all`` is the whole catalog; ``final`` is what the
#: operator actually kept — a highlight or a generated artifact — and excludes
#: scratch files that were merely mentioned or created in a conversation.
SCOPES: Final = ("all", "final")
SESSION_ID_SHAPE: Final = re.compile(r"^[a-f0-9]{12}$")

THUMB_MAX_WIDTH: Final = 480

#: Re-exported so a route validates against the SAME vocabulary the service
#: filters with; a second tuple in the route is how a filter it accepts starts
#: returning nothing.
CATEGORIES: Final = classify.CATEGORIES
KINDS: Final = classify.KINDS
ORIGINS: Final = _SCAN_ORIGINS


class InvalidCursor(ValueError):
    """The cursor was not minted by this query (HTTP 422)."""


class InvalidFilter(ValueError):
    """A filter value is outside its vocabulary (HTTP 422)."""


class AssetNotFound(LookupError):
    """No asset with that id is in the catalog."""


class ContentUnavailable(Exception):
    """The bytes cannot be served. ``code`` is the wire machine token."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class Query:
    q: str = ""
    categories: frozenset[str] = frozenset()
    kinds: frozenset[str] = frozenset()
    origin_groups: frozenset[str] = frozenset()
    origins: frozenset[str] = frozenset()
    tag: str | None = None
    tag_mode: str = "all"
    highlighted: bool | None = None
    organized: bool | None = None
    favorite: bool | None = None
    session: str | None = None
    #: The same filter as a comma list, for a conversation picker that offers
    #: several at once. ``session`` stays because the frozen design names it.
    session_ids: frozenset[str] = frozenset()
    #: ``all`` (default, the API contract) or ``final`` (the UI's default view).
    scope: str = "all"
    sort: str = "recent"
    order: str | None = None
    limit: int = DEFAULT_LIMIT
    cursor: str | None = None

    @property
    def effective_order(self) -> str:
        return self.order or DEFAULT_ORDER[self.sort]

    def fingerprint(self) -> str:
        """Stable identity of the query, cursor and limit excluded."""
        body = {
            "q": self.q,
            "categories": sorted(self.categories),
            "kinds": sorted(self.kinds),
            "origin_groups": sorted(self.origin_groups),
            "origins": sorted(self.origins),
            "tag": self.tag,
            "tag_mode": self.tag_mode,
            "highlighted": self.highlighted,
            "organized": self.organized,
            "favorite": self.favorite,
            "session": self.session,
            "session_ids": sorted(self.session_ids),
            "scope": self.scope,
            "sort": self.sort,
            "order": self.effective_order,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def parse_csv(raw: str | None, *, vocabulary: tuple[str, ...] | None = None) -> frozenset[str]:
    if not raw:
        return frozenset()
    tokens = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if len(tokens) > 32 or any(not re.fullmatch(r"[a-z0-9_]{1,24}", token) for token in tokens):
        raise InvalidFilter(raw[:80])
    if vocabulary is not None:
        unknown = [token for token in tokens if token not in vocabulary]
        if unknown:
            raise InvalidFilter(",".join(unknown)[:80])
    return frozenset(tokens)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _effective(metadata: dict[str, Any] | None, derived: str, field: str) -> str:
    """user > librarian > derived, exactly; an empty stored value is not a choice."""
    if metadata:
        value = metadata.get(field)
        source = metadata.get(f"{field}_source")
        # `agent` is an ordinary agent's own write (not the Librarian): it is
        # still a deliberate human-facing choice, so derived refreshes must not
        # overwrite it. Precedence stays user > librarian > agent > derived.
        if isinstance(value, str) and value and source in ("user", "librarian", "agent"):
            return value
    return derived or ""


def _preview(asset: library_index.Asset) -> dict[str, Any]:
    ready = asset.kind in (
        "image",
        "audio",
        "video",
        "pdf",
        "pptx",
        "markdown",
        "html",
        "spreadsheet",
        "text",
        "code",
    )
    return {
        "state": "ready" if ready else "unsupported",
        "mime_type": asset.content_type,
        "pages": None,
    }


def _exists(config_dir: Path, asset: library_index.Asset) -> bool:
    if asset.digest:
        return AttachmentStore(Path(config_dir) / "attachments").path_for(asset.digest).exists()
    if not asset.path:
        return False
    facts = files_service.stat_path(asset.path)
    return bool(facts.exists and facts.size is not None)


def _occurrence_row(
    occurrence: library_index.Occurrence, asset: library_index.Asset
) -> dict[str, Any]:
    # ``intent``/``tool`` are carried on the occurrence from the scan record; the
    # index currently keeps tool but not intent, so a missing value is honest.
    return {
        "session_id": occurrence.session_id,
        "session_title": occurrence.title,
        "origin": occurrence.origin,
        "tool": occurrence.tool,
        "at": occurrence.last_ts,
        "intent": occurrence.intent,
        # The message ordinals are provenance: a surface can jump back to the
        # exact turn that mentioned or produced the asset, not merely the file.
        "first_seq": occurrence.first_seq,
        "last_seq": occurrence.last_seq,
        "child_id": occurrence.child_id,
        "label": occurrence.label,
    }


def _row(
    config_dir: Path,
    asset: library_index.Asset,
    metadata: dict[str, Any] | None,
    tags: list[dict[str, Any]],
    primary: str | None,
    chat: dict[str, Any] | None,
    *,
    display: bool,
    home: str,
) -> dict[str, Any]:
    name = _effective(metadata, asset.derived_name, "name")
    description = _effective(metadata, "", "description")
    shown = (
        files_service.display_path(asset.path, "", home)
        if asset.path
        else (asset.derived_name or "")
    )
    organized = None
    if metadata and metadata.get("organized_at"):
        organized = {
            "at": metadata["organized_at"],
            "by": metadata.get("organized_by"),
            "confidence": metadata.get("confidence"),
        }
    return {
        "id": asset.id,
        "ref": {
            "type": asset.ref_type,
            # ``display=True`` is the phone/mesh projection: the absolute spelling
            # never leaves the desktop, and the display form still names the file.
            "path": None if display else asset.path,
            "display_path": shown,
            "digest": asset.digest,
            "content_type": asset.content_type,
        },
        "name": name,
        "derived_name": asset.derived_name,
        "description": description,
        "kind": asset.kind,
        "category": asset.category,
        "origin": asset.origin,
        "origin_group": asset.origin_group,
        "origins": list(asset.origins),
        "tags": [
            {"id": tag["id"], "name": tag["name"], "category": tag["category"]} for tag in tags
        ],
        "primary_tag": primary,
        "size_bytes": asset.size_bytes,
        "width": asset.width,
        "height": asset.height,
        "duration_s": asset.duration_s,
        "created_at": asset.first_ts,
        "modified_at": asset.last_ts,
        "last_seen_at": asset.last_ts,
        "exists": _exists(config_dir, asset),
        "highlighted": asset.highlighted,
        "highlight_why": None,
        "occurrences": [_occurrence_row(occurrence, asset) for occurrence in asset.occurrences],
        "session_count": len({occurrence.session_id for occurrence in asset.occurrences}),
        "chat": chat,
        "preview": _preview(asset),
        "organized": organized,
        "locked": bool(metadata.get("locked")) if metadata else False,
        "favorite": bool(metadata.get("favorite")) if metadata else False,
        "archived": bool(metadata.get("archived")) if metadata else False,
    }


def _chat_projection(
    config_dir: Path, store: MetadataStore, asset_id: str
) -> dict[str, Any] | None:
    link = store.chat(asset_id)
    if not link:
        return None
    session_id = str(link.get("session_id") or "")
    session_dir = config_dir / "sessions" / session_id
    if not SESSION_ID_SHAPE.match(session_id) or not session_dir.is_dir():
        # A deleted conversation must not leave a dead chip on the row.
        store.clear_chat(asset_id)
        return None
    return {
        "session_id": session_id,
        "title": _session_title(session_dir),
        "updated_at": float(link.get("last_used_at") or 0.0),
    }


def _session_title(session_dir: Path) -> str:
    """The stored title, never a transcript scan.

    ``resume.session_name`` falls back to scanning the journal and costs a cold
    2-6 s on the first call; a list/detail request cannot pay that per row. A
    session with no title sidecar shows its id here, while occurrence titles
    take the same stored title and fall back to the scanner's opening-line
    capture only for a session that was never titled.
    """
    try:
        from local_operator import resume

        return resume.stored_session_title(session_dir) or session_dir.name
    except Exception:  # noqa: BLE001 - a title is a label, never a request failure
        return session_dir.name


def _metadata_for(
    store: MetadataStore, assets: list[library_index.Asset]
) -> dict[str, dict[str, Any]]:
    found = store.assets(asset.id for asset in assets)
    # ``store.assets`` returns only existing rows; a row absent from SQLite is
    # simply still derived, which the projection already handles.
    return found


def _is_final(asset: library_index.Asset) -> bool:
    """The operator's kept set; defined beside the index it is written into."""
    return library_store.is_final(asset)


def _cursor(query: Query, revision: int, offset: int) -> str:
    return f"{query.fingerprint()}.{revision}.{offset}"


def _read_cursor(cursor: str, query: Query, revision: int) -> tuple[int, bool]:
    parts = cursor.split(".")
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
        raise InvalidCursor(cursor[:80])
    if parts[0] != query.fingerprint():
        raise InvalidCursor(cursor[:80])
    if int(parts[1]) != revision:
        # The catalog moved between pages. Restart at the top rather than skip
        # rows; every other list surface in this codebase uses this rule.
        return 0, True
    return int(parts[2]), False


def empty_listing(state: str = "ready") -> dict[str, Any]:
    return {
        "state": state,
        "revision": 0,
        "total": 0,
        "counts": {"category": {}, "origin_group": {}, "kind": {}},
        "tags": [],
        "rows": [],
        "next_cursor": None,
        "cursor_reset": False,
        "scan": {"sessions_total": 0, "sessions_scanned": 0},
    }


def _scoped_tags(
    tags: list[dict[str, Any]], tag_map: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Tag vocabulary of the scoped asset set, with counts within that scope.

    ``MetadataStore.list_tags`` counts every asset in the store, so a final
    listing would otherwise advertise tags only scratch assets carry.
    """
    used: dict[str, int] = {}
    for asset_tags in tag_map.values():
        for tag in asset_tags:
            used[tag["id"]] = used.get(tag["id"], 0) + 1
    return [{**tag, "count": used[tag["id"]]} for tag in tags if tag["id"] in used]


def list_assets(
    config_dir: str | Path,
    query: Query,
    *,
    display: bool = False,
    home: str | None = None,
    budget_s: float | None = library_index.DEFAULT_BUDGET_S,
) -> dict[str, Any]:
    """The whole answer for one list request. Blocking.

    ``budget_s`` is accepted and IGNORED: it used to bound a session scan on the
    request path, and there is no longer a scan here. The parameter stays so a
    caller written against the old signature keeps working; removing it would
    turn a build-on-read budget into an ImportError at the call site.
    """
    del budget_s
    config = Path(config_dir)
    home = home or str(Path.home())
    # TUI and phone callers reach the service without the server lifespan, so
    # the index has to be able to start from a read. Idempotent and cheap.
    ensure_started(config)
    store = LibraryStore(config)
    metadata = MetadataStore(config)
    # The cursor binds to BOTH halves: journals (the derived index) and the
    # metadata revision (names/descriptions/tags). A metadata edit moves a row
    # across a page boundary with no journal change at all.
    revision = store.revision() + metadata.metadata_revision()
    offset, reset = (0, False)
    if query.cursor:
        offset, reset = _read_cursor(query.cursor, query, revision)
    assets, total = store.query_page(query, offset=offset, limit=query.limit)
    page_ids = [asset.id for asset in assets]
    # Hydration is scoped to the PAGE. Materialising metadata, tags, chats and
    # occurrences for the whole catalog was what made a warm list take seconds.
    occurrences = store.occurrences(page_ids)
    for asset in assets:
        asset.occurrences = occurrences.get(asset.id, [])
    metas = store.metadata_rows(page_ids)
    tag_map = store.page_tags(page_ids)
    primary_map = store.primary_tags(page_ids)
    chats = metadata.chats_for_assets(page_ids)
    rows: list[dict[str, Any]] = []
    for asset in assets:
        chat = None
        link = chats.get(asset.id)
        if link:
            session_id = str(link.get("session_id") or "")
            session_dir = config / "sessions" / session_id
            if SESSION_ID_SHAPE.match(session_id) and session_dir.is_dir():
                chat = {
                    "session_id": session_id,
                    "title": _session_title(session_dir),
                    "updated_at": float(link.get("last_used_at") or 0.0),
                }
            else:
                metadata.clear_chat(asset.id)
        rows.append(
            _row(
                config,
                asset,
                metas.get(asset.id),
                tag_map.get(asset.id, []),
                primary_map.get(asset.id),
                chat,
                display=display,
                home=home,
            )
        )
    state, scanned, sessions_total = store.listing_state()
    return {
        "state": state,
        "revision": revision,
        "total": total,
        # Counts are wide within the request's scope on purpose: the filter rail
        # must not empty itself while the user is narrowing a search. They come
        # from ``idx_counts``, which triggers keep exact as rows move.
        "counts": store.counts(query.scope),
        "tags": metadata.list_tags() if query.scope != "final" else store.scoped_tags("final"),
        "rows": rows,
        "next_cursor": (
            _cursor(query, revision, offset + query.limit) if offset + query.limit < total else None
        ),
        "cursor_reset": reset,
        "scan": {"sessions_total": sessions_total, "sessions_scanned": scanned},
    }


def get_asset(
    config_dir: str | Path,
    asset_id: str,
    *,
    display: bool = False,
    home: str | None = None,
) -> dict[str, Any]:
    """One asset row with its tags, chat and preview. Raises :class:`AssetNotFound`."""
    if not identity.is_asset_id(asset_id):
        raise AssetNotFound(asset_id)
    config = Path(config_dir)
    home = home or str(Path.home())
    # One indexed row, its occurrences, and nothing else. The old miss rule
    # rebuilt the whole catalog here "in case the index was stale"; the index is
    # now kept fresh by a watcher, so a miss is a genuinely unknown id.
    asset = LibraryStore(config).asset(asset_id)
    if asset is None:
        raise AssetNotFound(asset_id)
    store = MetadataStore(config)
    store.ensure_asset(asset_id, asset.key, derived_name=asset.derived_name)
    metadata = store.asset(asset_id)
    tags = store.asset_tags(asset_id)
    primary = store.primary_tags([asset_id]).get(asset_id)
    chat = _chat_projection(config, store, asset_id)
    return _row(
        config,
        asset,
        metadata,
        tags,
        primary,
        chat,
        display=display,
        home=home,
    )


# ---------------------------------------------------------------------------
# Metadata writes
# ---------------------------------------------------------------------------


def _known_asset(config_dir: Path, asset_id: str) -> library_index.Asset:
    """Resolve an id against the derived catalog; writes may not invent assets.

    A syntactically valid id that names nothing is a 404, not a metadata row:
    otherwise a typo would create a permanent orphan the Librarian keeps seeing.
    """
    if not identity.is_asset_id(asset_id):
        raise AssetNotFound(asset_id)
    asset = LibraryStore(config_dir).asset(asset_id)
    if asset is None:
        raise AssetNotFound(asset_id)
    # Writes key on the SQLite row; a detail/list read creates it for most
    # assets, but a PATCH directly after indexing must not lose the edit.
    MetadataStore(config_dir).ensure_asset(asset_id, asset.key, derived_name=asset.derived_name)
    return asset


def update_asset(
    config_dir: str | Path,
    asset_id: str,
    fields: dict[str, Any],
    *,
    actor: str = "user",
    source: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Apply editable metadata. Records an edit row; returns the refreshed row.

    ``source`` defaults to ``actor``: a caller that names the writer but omits
    the provenance must not accidentally claim a USER edit and outrank a user's
    own value. The route passes both explicitly; the tool passes both.
    """
    config = Path(config_dir)
    _known_asset(config, asset_id)
    source = source or actor
    if source not in FIELD_SOURCES:
        raise TagRefused("bad_source", source)
    if actor not in ("user", "librarian", "agent", "system"):
        raise TagRefused("bad_actor", actor)
    cleaned: dict[str, Any] = {}
    for name in ("name", "description"):
        if name in fields:
            value = fields[name]
            if value is not None and not isinstance(value, str):
                raise TagRefused(f"bad_{name}", str(value)[:40])
            cleaned[name] = value
    for name in ("favorite", "locked", "archived"):
        if name in fields:
            value = fields[name]
            if not isinstance(value, bool):
                raise TagRefused(f"bad_{name}", str(value)[:40])
            cleaned[name] = value
    if "organized" in fields:
        value = fields["organized"]
        if not isinstance(value, bool):
            raise TagRefused("bad_organized", str(value)[:40])
        if value:
            cleaned["organized_at"] = time.time()
            cleaned["organized_by"] = actor
            if confidence is not None:
                cleaned["confidence"] = confidence
        else:
            cleaned["organized_at"] = None
            cleaned["organized_by"] = None
            cleaned["confidence"] = None
    if not cleaned:
        return get_asset(config, asset_id)
    if actor == "librarian" and ("name" in cleaned or "description" in cleaned):
        # Applying metadata IS the organization the daily run is measured by;
        # a row the Librarian only favorited is not "organized".
        cleaned["organized_at"] = time.time()
        cleaned["organized_by"] = actor
        if confidence is not None:
            cleaned["confidence"] = confidence
    store = MetadataStore(config)
    store.update_fields(asset_id, cleaned, actor=actor, source=source)
    return get_asset(config, asset_id)


def set_tags(
    config_dir: str | Path,
    asset_id: str,
    tag_ids: list[str],
    primary: str | None = None,
    *,
    actor: str = "user",
    source: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Replace an asset's tag set and return the refreshed row.

    ``source`` defaults to ``actor`` for the same reason as :func:`update_asset`.
    """
    config = Path(config_dir)
    asset = _known_asset(config, asset_id)
    store = MetadataStore(config)
    store.ensure_asset(asset_id, asset.key, derived_name=asset.derived_name)
    store.set_asset_tags(
        asset_id,
        tag_ids,
        primary=primary,
        actor=actor,
        source=source or actor,
        confidence=confidence,
    )
    return get_asset(config, asset_id)


def chat_link(config_dir: str | Path, asset_id: str) -> dict[str, Any] | None:
    config = Path(config_dir)
    _known_asset(config, asset_id)
    return _chat_projection(config, MetadataStore(config), asset_id)


def set_chat_link(config_dir: str | Path, asset_id: str, session_id: str) -> dict[str, Any]:
    """Record a chat link after the caller has created a REAL session.

    The route is not a session-creation path: it verifies the directory exists
    and the id has the session shape, then stores the link. That keeps one
    creation path per surface (and no second runner in the library route).
    """
    config = Path(config_dir)
    _known_asset(config, asset_id)
    if not SESSION_ID_SHAPE.match(session_id or ""):
        raise TagRefused("bad_session_id", (session_id or "")[:40])
    if not (config / "sessions" / session_id).is_dir():
        raise TagRefused("session_not_found", session_id)
    MetadataStore(config).set_chat(asset_id, session_id)
    return _chat_projection(config, MetadataStore(config), asset_id) or {}


def clear_chat_link(config_dir: str | Path, asset_id: str) -> bool:
    config = Path(config_dir)
    _known_asset(config, asset_id)
    return MetadataStore(config).clear_chat(asset_id)


# ---------------------------------------------------------------------------
# Content and thumbnails
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentRef:
    path: Path
    content_type: str
    name: str
    digest: bool = False
    modified_at: float = 0.0


def _catalog_asset(config: Path, asset_id: str) -> library_index.Asset:
    """Resolve an id for a content/thumbnail read.

    The cached/persisted aggregate answers the normal path without a scan. A
    MISS may mean an asset was indexed since the last index write, so refresh
    once (bounded by the request budget) before refusing — the old behaviour,
    without putting a scan on every hit.
    """
    if not identity.is_asset_id(asset_id):
        raise AssetNotFound(asset_id)
    asset = LibraryStore(config).asset(asset_id)
    if asset is None:
        raise AssetNotFound(asset_id)
    return asset


def resolve_content(config_dir: str | Path, asset_id: str) -> ContentRef:
    """Resolve an asset id to bytes ON THIS DEVICE. Never accepts a caller path."""
    config = Path(config_dir)
    asset = _catalog_asset(config, asset_id)
    name = asset.derived_name or asset.id
    if asset.digest:
        root = AttachmentStore(config / "attachments")
        path = root.path_for(asset.digest)
        if not path.is_file():
            raise ContentUnavailable("content_missing")
        return ContentRef(
            path=path,
            content_type=asset.content_type,
            name=name,
            digest=True,
            modified_at=path.stat().st_mtime,
        )
    if not asset.path:
        raise ContentUnavailable("content_missing")
    # Re-check the sensitive-path denylist on EVERY read. A path that passed
    # when it was indexed can become sensitive later (a config dir moved, a
    # symlink swapped); the indexed set is the only path the route may serve.
    deny = files_service.make_deny(config)
    if deny(asset.path):
        raise ContentUnavailable("content_denied")
    try:
        info = os.stat(asset.path)
    except OSError:
        raise ContentUnavailable("content_missing") from None
    if not stat.S_ISREG(info.st_mode):
        raise ContentUnavailable("content_missing")
    if os.path.realpath(asset.path) != identity.path_key(asset.path)[2:]:
        # The indexed spelling no longer resolves to the indexed file.
        raise ContentUnavailable("content_missing")
    return ContentRef(
        path=Path(asset.path),
        content_type=asset.content_type,
        name=name,
        digest=False,
        modified_at=info.st_mtime,
    )


def thumbnail(config_dir: str | Path, asset_id: str, width: int = THUMB_MAX_WIDTH) -> ContentRef:
    """Best-effort image thumbnail. ``ContentUnavailable`` for non-images.

    NEVER resizes here. A live cached WebP is served; otherwise the ORIGINAL is
    returned immediately and the indexer's thumbnail thread is asked for a
    resized copy, which the next render picks up. Resizing inline put Pillow's
    decode of a 4 MB image on the grid's critical path for every cold tile.

    SVG and GIF are served ORIGINAL: rasterising an animated GIF or a vector
    loses the thing the user wants, and both are small enough to render as-is.
    """
    config = Path(config_dir)
    width = max(1, min(int(width), THUMB_MAX_WIDTH))
    if not identity.is_asset_id(asset_id):
        raise AssetNotFound(asset_id)
    store = LibraryStore(config)
    asset = store.asset(asset_id)
    if asset is None:
        raise AssetNotFound(asset_id)
    if asset.kind != "image":
        raise ContentUnavailable("not_an_image")
    live = store.thumb_is_live(asset_id, width)
    if live is not None:
        return ContentRef(
            path=live,
            content_type="image/webp",
            name=f"{Path(asset.derived_name or asset.id).stem}.webp",
            digest=asset.digest is not None,
            modified_at=live.stat().st_mtime,
        )
    ensure_started(config).enqueue_thumbnail(asset_id)
    return resolve_content(config, asset_id)


__all__ = [
    "AssetNotFound",
    "ContentRef",
    "ContentUnavailable",
    "DEFAULT_LIMIT",
    "DEFAULT_ORDER",
    "InvalidCursor",
    "InvalidFilter",
    "MAX_LIMIT",
    "ORDERS",
    "Query",
    "SCOPES",
    "SORTS",
    "TAG_MODES",
    "THUMB_MAX_WIDTH",
    "chat_link",
    "clear_chat_link",
    "empty_listing",
    "get_asset",
    "list_assets",
    "parse_csv",
    "resolve_content",
    "set_chat_link",
    "set_tags",
    "thumbnail",
    "update_asset",
]
