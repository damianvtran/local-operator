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
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from local_operator.library import classify, identity
from local_operator.library import index as library_index
from local_operator.library.metadata import FIELD_SOURCES, MetadataStore, TagRefused
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


def _counts(assets: list[library_index.Asset]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "category": {},
        "origin_group": {},
        "kind": {},
    }
    for asset in assets:
        for bucket, value in (
            ("category", asset.category),
            ("origin_group", asset.origin_group),
            ("kind", asset.kind),
        ):
            counts[bucket][value] = counts[bucket].get(value, 0) + 1
    return counts


def _is_final(asset: library_index.Asset) -> bool:
    """The operator's kept set: a highlight, or an artifact the agent produced.

    Mentioned and merely-created scratch files are excluded from the ``final``
    scope; ``all`` still indexes and serves them.
    """
    return asset.highlighted or asset.origin_group == "generated"


def _matches(asset: library_index.Asset, row: dict[str, Any], query: Query) -> bool:
    if query.scope == "final" and not _is_final(asset):
        return False
    if query.categories and asset.category not in query.categories:
        return False
    if query.kinds and asset.kind not in query.kinds:
        return False
    if query.origin_groups and asset.origin_group not in query.origin_groups:
        return False
    if query.origins and not query.origins.intersection(asset.origins):
        return False
    if query.highlighted is not None and asset.highlighted != query.highlighted:
        return False
    if query.favorite is not None and bool(row["favorite"]) != query.favorite:
        return False
    if query.organized is not None and bool(row["organized"]) != query.organized:
        return False
    session_ids = set(query.session_ids)
    if query.session:
        session_ids.add(query.session)
    if session_ids and not any(
        occurrence.session_id in session_ids for occurrence in asset.occurrences
    ):
        return False
    if query.tag:
        tag_ids = {tag["id"] for tag in row["tags"]}
        if query.tag not in tag_ids:
            return False
    if query.q:
        haystack = " ".join(
            [
                row["name"],
                row["description"],
                row["derived_name"],
                row["ref"]["display_path"],
                " ".join(tag["name"] for tag in row["tags"]),
                " ".join(occurrence.title for occurrence in asset.occurrences if occurrence.title),
            ]
        ).lower()
        if query.q.lower() not in haystack:
            return False
    return True


def _sort_key(asset: library_index.Asset, row: dict[str, Any], sort: str) -> Any:
    """NATURAL ascending keys; ``order=desc`` is the caller's ``reverse``.

    Numeric sorts used to negate here, which made the caller invert booleans
    twice and is how "oldest first" ends up looking like "newest first". One
    direction, decided once.
    """
    if sort == "name":
        return row["name"].lower(), asset.id
    if sort == "kind":
        return asset.kind, row["name"].lower(), asset.id
    if sort == "size":
        return asset.size_bytes or 0, asset.id
    if sort == "created":
        return asset.first_ts, asset.id
    if sort == "modified":
        return asset.last_ts, asset.id
    if sort == "organized":
        organized = row["organized"]
        return (organized["at"] if organized else 0.0), asset.id
    return asset.last_ts, asset.id


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
    """The whole answer for one list request. Blocking."""
    config = Path(config_dir)
    home = home or str(Path.home())
    catalog = library_index.build_catalog(config, budget_s=budget_s, home=home)
    store = MetadataStore(config)
    all_assets = catalog.assets
    if query.scope == "final":
        # Narrow BEFORE any metadata work: a final listing must not materialise
        # rows (or count/tag) for scratch files the operator never kept. The
        # catalog still indexes everything, so scope=all and get_asset work.
        all_assets = [asset for asset in all_assets if _is_final(asset)]
    # Materialise a row per asset once, in one transaction: metadata writes
    # (tags, chat, edit history) key on ``assets.asset_id``, and a list request
    # is the first place most assets are ever seen. BEFORE the revision is read
    # — the first call creates rows and therefore bumps it, and reading first
    # would mint a cursor that is stale on the very next page.
    store.ensure_assets((asset.id, asset.key, asset.derived_name) for asset in all_assets)
    # The cursor is bound to BOTH halves: the derived catalog (journals) and
    # the metadata revision (names/descriptions/tags). A metadata edit can move
    # a row across a page boundary with no journal change, and a revision that
    # ignored it would let the next page duplicate one row and skip another.
    revision = catalog.revision + store.metadata_revision()
    metas = _metadata_for(store, all_assets)
    tag_map = store.tags_for_assets(asset.id for asset in all_assets)
    primary_map = store.primary_tags(asset.id for asset in all_assets)
    chats = store.chats_for_assets(asset.id for asset in all_assets)

    # Materialise every row's derived metadata once; filters and sorting read
    # the row, so no filter can disagree with what the page displays.
    rows: list[tuple[library_index.Asset, dict[str, Any]]] = []
    for asset in all_assets:
        metadata = metas.get(asset.id)
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
                store.clear_chat(asset.id)
        row = _row(
            config,
            asset,
            metadata,
            tag_map.get(asset.id, []),
            primary_map.get(asset.id),
            chat,
            display=display,
            home=home,
        )
        rows.append((asset, row))

    filtered = [(asset, row) for asset, row in rows if _matches(asset, row, query)]
    filtered.sort(
        key=lambda pair: _sort_key(pair[0], pair[1], query.sort),
        reverse=query.effective_order == "desc",
    )

    offset, reset = (0, False)
    if query.cursor:
        offset, reset = _read_cursor(query.cursor, query, revision)
    page = filtered[offset : offset + query.limit]
    more = offset + query.limit < len(filtered)
    return {
        "state": catalog.state,
        "revision": revision,
        "total": len(filtered),
        # Counts are wide within the request's scope on purpose: the filter rail
        # must not empty itself while the user is narrowing a search.
        "counts": _counts(all_assets),
        "tags": (
            _scoped_tags(store.list_tags(), tag_map)
            if query.scope == "final"
            else store.list_tags()
        ),
        "rows": [row for _asset, row in page],
        "next_cursor": _cursor(query, revision, offset + query.limit) if more else None,
        "cursor_reset": reset,
        "scan": {
            "sessions_total": catalog.sessions_total,
            "sessions_scanned": catalog.sessions_scanned,
        },
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
    # Read-only lookup: the cached/persisted aggregate is enough, and a get must
    # not pay a session scan (or an index rewrite) just to name one row.
    catalog = library_index.catalog_for_read(config)
    asset = _read_catalog_asset(catalog, asset_id)
    if asset is None:
        # Same miss rule as content/thumbnail: a just-indexed asset may not be
        # in the last index write yet. One bounded refresh, then refuse.
        catalog = library_index.build_catalog(config, home=home, refresh=True)
        asset = _read_catalog_asset(catalog, asset_id)
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
    catalog = library_index.build_catalog(config_dir)
    asset = next((item for item in catalog.assets if item.id == asset_id), None)
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


def _read_catalog_asset(
    catalog: library_index.Catalog, asset_id: str
) -> library_index.Asset | None:
    return next((item for item in catalog.assets if item.id == asset_id), None)


def _catalog_asset(config: Path, asset_id: str) -> library_index.Asset:
    """Resolve an id for a content/thumbnail read.

    The cached/persisted aggregate answers the normal path without a scan. A
    MISS may mean an asset was indexed since the last index write, so refresh
    once (bounded by the request budget) before refusing — the old behaviour,
    without putting a scan on every hit.
    """
    if not identity.is_asset_id(asset_id):
        raise AssetNotFound(asset_id)
    asset = _read_catalog_asset(library_index.catalog_for_read(config), asset_id)
    if asset is None:
        asset = _read_catalog_asset(library_index.build_catalog(config, refresh=True), asset_id)
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

    SVG and GIF are served ORIGINAL: rasterising an animated GIF or a vector
    loses the thing the user wants, and both are small enough to render as-is.
    """
    config = Path(config_dir)
    asset = _catalog_asset(config, asset_id)
    if asset.kind != "image":
        raise ContentUnavailable("not_an_image")
    source = resolve_content(config, asset_id)
    suffix = source.path.suffix.lower()
    if suffix in (".svg", ".gif"):
        return source
    width = max(1, min(int(width), THUMB_MAX_WIDTH))
    stamp = int(source.modified_at or 0)
    cache_dir = config / "cache" / "library" / "thumbs"
    target = cache_dir / f"{asset.id}-{width}-{stamp}.webp"
    if target.is_file():
        return ContentRef(
            path=target,
            content_type="image/webp",
            name=f"{Path(asset.derived_name or asset.id).stem}.webp",
            digest=asset.digest is not None,
            modified_at=target.stat().st_mtime,
        )
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        # Pillow is an optional extra. Serving the original is honest and
        # bounded to a real image; the grid's own sizing does the rest.
        return source
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(source.path) as image:
            image.load()
            if image.width > width:
                height = max(1, round(image.height * width / image.width))
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA")
            # Atomic: two list renders can ask for the same thumbnail at once,
            # and a half-written WebP is a broken image, not a stale one.
            temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            image.save(temp, "WEBP", quality=82)
            os.replace(temp, target)
    except Exception:  # noqa: BLE001 - a thumbnail is cosmetic; the row still works
        logger.debug("library: thumbnail failed for %s", asset_id, exc_info=True)
        return source
    return ContentRef(
        path=target,
        content_type="image/webp",
        name=f"{Path(asset.derived_name or asset.id).stem}.webp",
        digest=asset.digest is not None,
        modified_at=target.stat().st_mtime,
    )


def _waveform_shape(raw: Any) -> dict[str, Any] | None:
    """A ``{version, bars, peaks}`` payload that is safe to hand a renderer, or ``None``.

    Validated on READ (the cache is a file on disk, not a value this process holds):
    exactly ``PEAK_BARS`` finite numbers in 0..1 under the frozen version. A damaged or
    stale cache answers ``None`` and is recomputed; a damaged SIDECAR is the attachment
    store's own validator (``AttachmentStore.get_peaks``), which this mirrors.
    """
    from local_operator.audiogen.peaks import PEAK_BARS, PEAKS_VERSION

    if not isinstance(raw, dict):
        return None
    version, bars, peaks = raw.get("version"), raw.get("bars"), raw.get("peaks")
    if isinstance(bars, bool) or not isinstance(bars, int) or isinstance(version, bool):
        return None
    if version != PEAKS_VERSION or bars != PEAK_BARS or not isinstance(peaks, list):
        return None
    if len(peaks) != bars:
        return None
    values: list[float] = []
    for value in peaks:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            return None
        values.append(number)
    return {"version": version, "bars": bars, "peaks": values}


def _decode_peaks_bounded(raw: bytes) -> list[float] | None:
    """Peaks for ``raw``, decoded in a CHILD process under the recovery module's bounds.

    A native codec cannot be interrupted inside this process: a malformed clip could pin
    an HTTP worker for good. ``audiogen.recovery`` already owns exactly this problem for
    legacy attachment repair -- an input cap in the child, a deadline in the parent, and
    no fabricated silence on failure -- so its module entry point is reused rather than
    a second decoder being written here. ``None`` is the honest "no waveform".

    The decode also takes the recovery module's ONE decode slot: a Library grid of path
    audio assets asks for many waveforms at once, and without the slot each tile would
    spawn its own interpreter (measured 1.71 s wall each). A caller that cannot get the
    slot within the same deadline answers its own no-waveform 404 instead.
    """
    from local_operator.audiogen.peaks import PEAK_BARS
    from local_operator.audiogen.recovery import (
        MAX_RECOVERY_BYTES,
        RECOVERY_TIMEOUT_SECONDS,
        decode_slot,
        release_decode_slot,
    )

    if not raw or len(raw) > MAX_RECOVERY_BYTES:
        return None
    if not decode_slot():
        logger.debug("library: no decoder slot within the deadline; answering no-waveform")
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "local_operator.audiogen.recovery"],
            input=raw,
            capture_output=True,
            timeout=RECOVERY_TIMEOUT_SECONDS,
            check=True,
        )
        payload = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.debug("library: bounded decode unavailable", exc_info=True)
        return None
    finally:
        release_decode_slot()
    if not isinstance(payload, list) or len(payload) != PEAK_BARS:
        return None
    peaks: list[float] = []
    for value in payload:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        # The same 0..1/finite rule the sidecar and cache readers enforce: the child
        # clamps today, but a decoder is not the place a renderer's contract lives.
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            return None
        peaks.append(number)
    return peaks


def waveform(config_dir: str | Path, asset_id: str) -> dict[str, Any]:
    """``{version, bars, peaks}`` for one audio asset. See the module docstring's rule.

    NO RENDERER EVER DECODES AUDIO (``audiogen.peaks``): peaks are computed once and
    read afterwards. Two kinds of asset, two sources of truth:

    * digest-backed -- the bytes live in the content-addressed store, so the
      ``<digest>.peaks.json`` sidecar written at creation IS the answer; the bounded
      legacy repair behind ``waveform_payload`` covers artifacts stored before a
      decoder was installed.
    * path-backed -- an indexed user file has no sidecar, so the first request decodes
      it ONCE and caches the payload under ``cache/library/peaks/<id>-<stamp>.json``
      (stamp = the source's mtime, the thumbnail cache's own invalidation key). The
      cache file is validated on read, so a stale/half-written one is recomputed
      rather than served.

    A refusal is ``ContentUnavailable`` -- an ordinary 404 on the route. There is no
    fabricated flat waveform: a surface draws its no-waveform state, exactly as a
    missing thumbnail degrades to a placeholder.
    """
    config = Path(config_dir)
    asset = _catalog_asset(config, asset_id)
    if asset.digest:
        # A digest sidecar is AUDIO's sidecar. An image or video digest asset may hold
        # a stale/foreign peaks file, and the repair path behind ``waveform_payload``
        # would spend the one decoder slot rejecting it; refuse before either.
        if asset.kind != "audio":
            raise ContentUnavailable("content_missing")
        from local_operator.audiogen.recovery import waveform_payload

        # The store validates its own sidecar; ``_waveform_shape`` re-imposes the
        # renderer's contract here, so the route and the path-asset cache serve the
        # same checked shape.
        payload = _waveform_shape(
            waveform_payload(AttachmentStore(config / "attachments"), asset.digest)
        )
        if payload is None:
            raise ContentUnavailable("content_missing")
        return payload

    source = resolve_content(config, asset_id)
    stamp = int(source.modified_at or 0)
    cache_dir = config / "cache" / "library" / "peaks"
    target = cache_dir / f"{asset.id}-{stamp}.json"
    try:
        cached = _waveform_shape(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        cached = None
    if cached is not None:
        return cached

    from local_operator.audiogen.recovery import MAX_RECOVERY_BYTES

    try:
        if source.path.stat().st_size > MAX_RECOVERY_BYTES:
            # A partial decode of a truncated file would summarise the first seconds
            # as if they were the whole clip -- a wrong waveform is worse than none.
            raise ContentUnavailable("content_missing")
        raw = source.path.read_bytes()
    except OSError:
        raise ContentUnavailable("content_missing") from None
    peaks = _decode_peaks_bounded(raw)
    if peaks is None:
        raise ContentUnavailable("content_missing")
    from local_operator.audiogen.peaks import peaks_payload

    payload = peaks_payload(peaks)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Atomic: two grid renders can ask for the same waveform at once, and a
        # half-written JSON is a broken player, not a stale one (thumbnail's rule).
        temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, target)
    except OSError:
        # The payload is already in hand; a read-only cache is not a failed request.
        logger.debug("library: waveform cache write failed for %s", asset_id, exc_info=True)
    return payload


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
    "waveform",
]
