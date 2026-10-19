"""``/api/library*`` — the phone's read-write surface over the shared Library.

WHAT THIS IS. A transport adapter, not a second implementation. Every function
here calls :mod:`local_operator.library.service` (the ONE place a row's meaning
is decided) and projects it through ``display=True``, which drops the absolute
path spelling. That projection is a security boundary on this surface, not a
formatting nicety: the phone and the mesh must never receive a filesystem path,
and a route that forgot the flag would leak one. Keeping the flag here — the
only caller of the service on this plane — is what makes it a single decision.

IDS, NEVER PATHS. Every content route addresses ``la_`` + ``sha256(key)[:16]``.
The id is looked up in the derived catalog and the service re-checks the
resolved path against the sensitive-path denylist (``service.resolve_content``),
so a caller cannot turn this surface into an arbitrary-file read however the
query is shaped.

THREADING. Every function is blocking (SQLite, filesystem stat, image resize),
so the daemon runs them through ``asyncio.to_thread``. Nothing here creates or
mutates a session: a chat link is recorded only after the CALLER proved a real
session directory exists, which keeps one session-creation path per surface.

REFUSALS. :class:`LibraryRouteError` carries the wire status and machine token.
The daemon's five project handlers use the same split (flat JSON, 4xx for a
refusal, 503 for a store that cannot answer), so a client parses one shape.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Final

from local_operator.library import classify, identity
from local_operator.library import index as library_index
from local_operator.library import service as library_service
from local_operator.session_files.scan import ORIGINS
from local_operator.library.metadata import (
    MetadataStore,
    MetadataUnavailable,
    TagRefused,
)

#: Filter vocabularies the service owns. Imported rather than restated so a new
#: kind/category/sort cannot be accepted here and refused there (or vice versa).
SORTS: Final = library_service.SORTS
ORDERS: Final = library_service.ORDERS
TAG_MODES: Final = library_service.TAG_MODES
DEFAULT_LIMIT: Final = library_service.DEFAULT_LIMIT
MAX_LIMIT: Final = library_service.MAX_LIMIT


class LibraryRouteError(Exception):
    """A refusal the daemon maps to ``status`` and a machine ``code``."""

    def __init__(self, status: int, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.status = status
        self.code = code
        self.message = message or code


def _unavailable(error: Exception) -> LibraryRouteError:
    return LibraryRouteError(503, "library_store_busy", str(error))


def _refused(error: TagRefused) -> LibraryRouteError:
    # The store's own machine code is the wire code; inventing a second
    # vocabulary here would make the phone translate what the desktop does not.
    return LibraryRouteError(422, error.code, str(error))


def is_available() -> bool:
    """Whether the Library backend is importable on this build.

    Advertised as ``capabilities.library``. The daemon imports this module
    lazily, but a phone must not render a destination that its relay cannot
    serve; ``find_spec`` answers without importing SQLite or Pillow.
    """
    return importlib.util.find_spec("local_operator.library") is not None


def librarian_available() -> bool:
    """Whether the Librarian module exists (the capability the design names)."""
    return importlib.util.find_spec("local_operator.librarian") is not None


def _flag(raw: str | None, name: str) -> bool | None:
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise library_service.InvalidFilter(name)


def _csv(raw: str | None, vocabulary: tuple[str, ...]) -> frozenset[str]:
    return library_service.parse_csv(raw, vocabulary=vocabulary)


def _limit(raw: str | None) -> int:
    if raw is None or raw == "":
        return DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        raise library_service.InvalidFilter("limit") from None
    if value < 1 or value > MAX_LIMIT:
        raise library_service.InvalidFilter("limit")
    return value


def _query(params: dict[str, str]) -> library_service.Query:
    """Build the frozen ``Query`` from query-string values.

    ``parse_csv`` validates each vocabulary, so an unknown kind is a 422 here
    rather than a silently empty list — the phone's chip row is generated from
    the counts, and a typo in a deep link must not look like "no assets".
    """
    sort = params.get("sort") or "recent"
    if sort not in SORTS:
        raise library_service.InvalidFilter("sort")
    order = params.get("order") or None
    if order is not None and order not in ORDERS:
        raise library_service.InvalidFilter("order")
    tag_mode = params.get("tag_mode") or "all"
    if tag_mode not in TAG_MODES:
        raise library_service.InvalidFilter("tag_mode")
    session = params.get("session") or None
    if session is not None and not library_service.SESSION_ID_SHAPE.match(session):
        raise library_service.InvalidFilter("session")
    # Session ids have no frozen vocabulary (any 12-hex id can be filtered on),
    # so only the shape is validated below.
    session_ids = library_service.parse_csv(
        params.get("sessions") or params.get("session_ids")
    )
    for session_id in session_ids:
        if not library_service.SESSION_ID_SHAPE.match(session_id):
            raise library_service.InvalidFilter("sessions")
    q = params.get("q") or ""
    if len(q) > library_service.MAX_QUERY_CHARS:
        raise library_service.InvalidFilter("q")
    return library_service.Query(
        q=q,
        categories=_csv(params.get("categories") or params.get("category"), classify.CATEGORIES),
        kinds=_csv(params.get("kinds") or params.get("kind"), classify.KINDS),
        origin_groups=_csv(
            params.get("origin_groups") or params.get("origin_group"),
            library_index.ORIGIN_GROUPS,
        ),
        origins=_csv(params.get("origins") or params.get("origin"), ORIGINS),
        tag=params.get("tag") or None,
        tag_mode=tag_mode,
        highlighted=_flag(params.get("highlighted"), "highlighted"),
        organized=_flag(params.get("organized"), "organized"),
        favorite=_flag(params.get("favorite"), "favorite"),
        session=session,
        session_ids=session_ids,
        sort=sort,
        order=order,
        limit=_limit(params.get("limit")),
        cursor=params.get("cursor") or None,
    )


def list_payload(config_dir: str | Path, params: dict[str, str]) -> dict[str, Any]:
    """One list frame: rows, cursor, counts and the tag vocabulary."""
    try:
        return library_service.list_assets(
            config_dir, _query(params), display=True
        )
    except library_service.InvalidCursor as error:
        raise LibraryRouteError(422, "library_invalid_cursor", str(error)) from error
    except library_service.InvalidFilter as error:
        raise LibraryRouteError(422, "library_invalid_filter", str(error)) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def facets_payload(config_dir: str | Path) -> dict[str, Any]:
    """Counts + tags only, for a first paint that does not need a page of rows.

    Still one service call (the catalog cache makes it cheap): a second
    aggregation over the index would be a second spelling of "what is in the
    library", which is exactly what the service exists to prevent.
    """
    frame = list_payload(config_dir, {"limit": "1"})
    return {
        "state": frame["state"],
        "revision": frame["revision"],
        "total": frame["total"],
        "counts": frame["counts"],
        "tags": frame["tags"],
        "scan": frame["scan"],
    }


def detail_payload(config_dir: str | Path, asset_id: str) -> dict[str, Any]:
    try:
        return library_service.get_asset(config_dir, asset_id, display=True)
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def patch_payload(
    config_dir: str | Path, asset_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Editable metadata only. Unknown keys are ignored by the service, so a
    newer desktop client's additive field is not fatal to an older relay."""
    try:
        return library_service.update_asset(config_dir, asset_id, body)
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except TagRefused as error:
        raise _refused(error) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def tags_payload(config_dir: str | Path) -> dict[str, Any]:
    try:
        return {"tags": MetadataStore(config_dir).list_tags()}
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def create_tag_payload(config_dir: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LibraryRouteError(422, "bad_name")
    category = body.get("category") or "custom"
    if not isinstance(category, str):
        raise LibraryRouteError(422, "bad_category")
    try:
        return {"tag": MetadataStore(config_dir).create_tag(name, category)}
    except TagRefused as error:
        raise _refused(error) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def rename_tag_payload(
    config_dir: str | Path, tag_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LibraryRouteError(422, "bad_name")
    try:
        return {"tag": MetadataStore(config_dir).rename_tag(tag_id, name)}
    except TagRefused as error:
        raise _refused(error) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def merge_tag_payload(
    config_dir: str | Path, tag_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    into = body.get("into")
    if not isinstance(into, str) or not into:
        raise LibraryRouteError(422, "bad_target")
    try:
        return {"tag": MetadataStore(config_dir).merge_tags(tag_id, into)}
    except TagRefused as error:
        raise _refused(error) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def set_tags_payload(
    config_dir: str | Path, asset_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    raw = body.get("tag_ids")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise LibraryRouteError(422, "bad_tag_ids")
    primary = body.get("primary")
    if primary is not None and not isinstance(primary, str):
        raise LibraryRouteError(422, "bad_primary")
    if not identity.is_asset_id(asset_id):
        raise LibraryRouteError(404, "library_asset_not_found")
    try:
        # Ensure a well-formed but unknown id is a 404 before a write, matching
        # every other route in this family. The store write itself is the
        # authoritative one: ``service.set_tags`` forwards a ``source`` kwarg
        # the store does not accept (core WIP), so this adapter calls the store
        # directly rather than restating its rules. Reported upstream.
        library_service.get_asset(config_dir, asset_id, display=True)
        tags = MetadataStore(config_dir).set_asset_tags(asset_id, raw, primary=primary)
        return {"tags": tags}
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except TagRefused as error:
        raise _refused(error) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def chat_payload(config_dir: str | Path, asset_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Record the asset↔session link. The session must already exist."""
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise LibraryRouteError(422, "bad_session_id")
    try:
        return {"chat": library_service.set_chat_link(config_dir, asset_id, session_id)}
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except TagRefused as error:
        raise _refused(error) from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def clear_chat_payload(config_dir: str | Path, asset_id: str) -> dict[str, Any]:
    try:
        cleared = library_service.clear_chat_link(config_dir, asset_id)
        return {"ok": True, "cleared": cleared}
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except MetadataUnavailable as error:
        raise _unavailable(error) from error


def content_ref(config_dir: str | Path, asset_id: str) -> library_service.ContentRef:
    try:
        return library_service.resolve_content(config_dir, asset_id)
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except library_service.ContentUnavailable as error:
        # The catalog knows the asset but the bytes cannot be served: the phone
        # renders "missing file" for this token rather than a generic error.
        raise LibraryRouteError(404, error.code) from error


def thumb_ref(
    config_dir: str | Path, asset_id: str, width: int
) -> library_service.ContentRef:
    try:
        return library_service.thumbnail(config_dir, asset_id, width)
    except library_service.AssetNotFound as error:
        raise LibraryRouteError(404, "library_asset_not_found") from error
    except library_service.ContentUnavailable as error:
        raise LibraryRouteError(404, error.code) from error


__all__ = [
    "DEFAULT_LIMIT",
    "LibraryRouteError",
    "MAX_LIMIT",
    "ORDERS",
    "SORTS",
    "TAG_MODES",
    "chat_payload",
    "clear_chat_payload",
    "content_ref",
    "create_tag_payload",
    "detail_payload",
    "facets_payload",
    "is_available",
    "librarian_available",
    "list_payload",
    "merge_tag_payload",
    "patch_payload",
    "rename_tag_payload",
    "set_tags_payload",
    "tags_payload",
    "thumb_ref",
]
