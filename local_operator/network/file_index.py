"""``net_session_files``: a peer session's Files index, answered by the OWNER.

docs/design/remote-files-parity.md §2 (option A). The owner runs the SAME
``session_files.service.list_files`` over its own journal that its desktop route
runs, so pagination, ``building`` polling and ``cursor_reset`` keep the local
semantics; cursors and revisions are owner-minted and pass through opaquely.
Paths leave the owner only in DISPLAY form (``~/…`` or cwd-relative), never its
absolute layout.

Each page row also says whether a view member could pull it under S1's
confinement (``remote_read``) or why not (``remote_unreadable``). That is
ADVISORY: ``net_file_get`` re-checks every phase, so the listing never grants a
read. Workspace files stay unreadable (``outside_session_files``): ``net_sync``
does not hand a view member the cwd, so neither does this.

Phases:
- ``list {session_id, index: "files"|"user_attachments", query}`` → the listing.
- ``row {session_id, file_id}`` → ``{row, source}`` or a ``not_found`` refusal;
  ``source`` is what ``session_file_pull`` should ask for.

The viewer's local op ``peer_session_files`` forwards one phase over the link,
after checking the owner advertises :data:`wire.MESH_FILE_INDEX_V1`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from local_operator.network import sync, wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import PeerLink, RelayServer

logger = logging.getLogger(__name__)

#: The owner-side deadline of one ``net_session_files`` request: the files
#: service's own scan budget (1.5 s, resumable) plus a family discovery and the
#: page's stats, with headroom; ``building`` covers anything slower.
INDEX_OP_DEADLINE_S = 30.0

#: The viewer's wait for one phase through its relay; above the owner deadline.
INDEX_CLIENT_TIMEOUT_S = 45.0

INDEXES = ("files", "user_attachments")

_BACKGROUND: dict[tuple[str, str, str], threading.Thread] = {}
_BACKGROUND_GUARD = threading.Lock()


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


# ---------------------------------------------------------------------------
# The owner's half
# ---------------------------------------------------------------------------


def _session_dir(root: Path, session_id: str) -> Path:
    from local_operator.resume import is_user_session
    from local_operator.session_files.family import SESSION_ID

    directory = Path(root) / "sessions" / session_id
    if not SESSION_ID.fullmatch(session_id) or not directory.is_dir():
        raise _refuse("not_found", f"session {session_id} has no files index here")
    if not is_user_session(directory):
        raise _refuse("not_found", f"session {session_id} has no files index here")
    return directory


def _query(raw: Any) -> Any:
    from local_operator.session_files import service

    raw = raw if isinstance(raw, dict) else {}
    try:
        return service.Query(
            scope=str(raw.get("scope") or "family"),
            kinds=frozenset(str(k) for k in raw.get("kinds") or ()),
            origins=frozenset(str(o) for o in raw.get("origins") or ()),
            highlighted=(
                raw.get("highlighted") if isinstance(raw.get("highlighted"), bool) else None
            ),
            q=str(raw.get("q") or "")[: service.MAX_QUERY_CHARS],
            sort=str(raw.get("sort") or "recent"),
            order=str(raw["order"]) if raw.get("order") else None,
            limit=max(1, min(int(raw.get("limit") or service.DEFAULT_LIMIT), service.MAX_LIMIT)),
            cursor=str(raw["cursor"]) if raw.get("cursor") else None,
        )
    except (TypeError, ValueError) as error:
        raise _refuse("bad_request", f"the files query is malformed: {error}") from None


def readability(
    root: Path,
    *,
    member_id: str,
    absolute: str | None,
    digest: str | None,
    size: int | None,
    exists: bool | None,
    referenced: Callable[[str], set[str]],
) -> tuple[dict[str, Any] | None, str | None, dict[str, str] | None]:
    """``(remote_read, remote_unreadable, source)`` for one row, under S1's rules.

    The table in §2: an artifact or user attachment is readable iff the member's
    transcript references its digest; a path is readable iff it resolves inside
    the member's scratchpad and is not credential-shaped. Everything else names
    the reason. ``source`` is the ``session_file_pull`` frame for a readable row.
    """
    from local_operator.network.files import FILE_MAX_BYTES
    from local_operator.scratchpad import scratchpad_root
    from local_operator.supplements.denylist import is_sensitive_for_supplements

    if size is not None and size > FILE_MAX_BYTES:
        return None, "too_large", None
    if digest:
        if digest not in referenced(member_id):
            return None, "not_referenced", None
        return {"kind": "attachment"}, None, {"attachment": digest}
    if not absolute:
        return None, "outside_session_files", None
    if exists is False:
        return None, "missing", None
    pad = scratchpad_root(Path(root) / "sessions" / member_id)
    if pad is None:
        return None, "outside_session_files", None
    try:
        resolved = Path(os.path.realpath(absolute))
        pad_real = Path(os.path.realpath(pad))
    except (OSError, ValueError):
        return None, "missing", None
    if not resolved.is_relative_to(pad_real) or resolved == pad_real:
        return None, "outside_session_files", None
    for form in (Path(absolute), resolved):
        if is_sensitive_for_supplements(form, session_scratchpad=str(pad)):
            return None, "sensitive", None
    relative = resolved.relative_to(pad_real).as_posix()
    if any(part.startswith(".") for part in relative.split("/")):
        # The pad's own parser refuses dotfiles; say so here rather than list a
        # row as readable that every pull would refuse.
        return None, "sensitive", None
    return {"kind": "path"}, None, {"path": f"scratchpad://{relative}"}


class _Referenced:
    """Per-request memo of each member's referenced digests (the scan itself is memoised)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cache: dict[str, set[str]] = {}

    def __call__(self, member_id: str) -> set[str]:
        found = self.cache.get(member_id)
        if found is None:
            found = set(sync.referenced_attachments_in(self.root / "sessions" / member_id))
            self.cache[member_id] = found
        return found


def _decorate(root: Path, row: dict[str, Any], referenced: _Referenced, home: str) -> Any:
    """Add the remote fields to one PROJECTED (absolute) row; return its pull source."""
    from local_operator.session_files import service

    ref = row.get("ref") or {}
    member_id = str((row.get("session") or {}).get("id") or "")
    absolute = ref.get("path") if ref.get("type") == "path" else None
    remote_read, unreadable, source = readability(
        root,
        member_id=member_id,
        absolute=absolute,
        digest=ref.get("digest") if ref.get("type") == "artifact" else None,
        size=row.get("size_bytes"),
        exists=row.get("exists"),
        referenced=referenced,
    )
    if absolute:
        # DISPLAY FORM ONLY leaves the owner (desktop_session_files' mesh rule).
        directory = Path(root) / "sessions" / member_id
        cwd = service.session_cwd(root, member_id, directory)
        ref = {**ref, "path": service.display_path(absolute, cwd, home)}
        row["ref"] = ref
    row["remote_read"] = remote_read
    row["remote_unreadable"] = unreadable
    return {"session_id": member_id, **source} if source else None


def _kick(root: Path, family: Any, scope: str) -> None:
    """Finish a ``building`` scan off the request, once per (root, session, scope)."""
    from local_operator.session_files import service

    key = (str(root), family.session_id, scope)
    with _BACKGROUND_GUARD:
        running = _BACKGROUND.get(key)
        if running is not None and running.is_alive():
            return

        def run() -> None:
            try:
                service.complete_in_background(
                    root, family, scope, home=str(Path.home()), deny=service.make_deny(root)
                )
            except Exception:  # noqa: BLE001 - completion is an optimisation
                logger.debug("net_session_files background scan failed", exc_info=True)

        thread = threading.Thread(target=run, name="net-session-files-scan", daemon=True)
        _BACKGROUND[key] = thread
        thread.start()


def _fit(answer: dict[str, Any], query: Any, start: int) -> dict[str, Any]:
    """Trim tail rows until the reply fits one link record; re-mint ``next_cursor``.

    An oversize record kills the whole link (``relay.py``'s history budget note), so
    the page shrinks instead and the next page starts where this one stopped.
    """
    from local_operator.network.relay import SESSION_HISTORY_REPLY_BUDGET_BYTES
    from local_operator.session_files import service

    rows = list(answer.get("rows") or [])
    kept = len(rows)
    while kept > 0:
        candidate = {**answer, "rows": rows[:kept]}
        if len(json.dumps(candidate, separators=(",", ":"))) <= SESSION_HISTORY_REPLY_BUDGET_BYTES:
            break
        kept = max(0, kept - max(1, kept // 8))
    if kept == len(rows):
        return answer
    trimmed = {**answer, "rows": rows[:kept], "trimmed": True}
    trimmed["next_cursor"] = service.encode_cursor(int(answer["revision"]), query, start + kept)
    return trimmed


def list_index(root: Path, session_id: str, frame: dict[str, Any]) -> dict[str, Any]:
    """The ``list`` phase. Blocking: the relay calls it on its op thread."""
    from local_operator.session import compression, user_attachments
    from local_operator.session_files import family as family_mod
    from local_operator.session_files import service

    root = Path(root)
    directory = _session_dir(root, session_id)
    index = str(frame.get("index") or "files")
    if index not in INDEXES:
        raise _refuse("bad_request", f"unknown files index {index!r}")
    home = str(Path.home())
    referenced = _Referenced(root)
    if index == "user_attachments":
        try:
            listing = user_attachments.list_user_attachments(
                directory, deny=service.make_deny(root)
            )
        except compression.ArchiveError:
            return {"state": "unavailable", "items": [], "truncated": False}
        for item in listing.get("items") or []:
            digest = item.get("attachment")
            if digest and digest in referenced(session_id):
                item["remote_read"], item["remote_unreadable"] = {"kind": "attachment"}, None
            else:
                item["remote_read"], item["remote_unreadable"] = None, "not_referenced"
            if item.get("path"):
                cwd = service.session_cwd(root, session_id, directory)
                item["path"] = service.display_path(str(item["path"]), cwd, home)
            # The owner's store path names a file on ITS disk; never hand it out.
            item["stored_path"] = None
        return {"state": "ready", **listing}

    query = _query(frame.get("query"))
    if query.scope not in service.SCOPES or query.sort not in service.SORTS:
        raise _refuse("bad_request", "the files query names an unknown scope or sort")
    start = 0
    if query.cursor:
        try:
            start = service.decode_cursor(query.cursor, query)[1]
        except service.InvalidCursor:
            raise _refuse(
                "bad_cursor", "that page cursor was not issued for this request"
            ) from None
    family, discovery = family_mod.family_blocking(
        root, session_id, directory, scope=query.scope, budget_s=service.DEFAULT_BUDGET_S
    )
    try:
        answer = service.list_files(
            root,
            family,
            query,
            display=False,
            budget_s=service.DEFAULT_BUDGET_S,
            home=home,
            expected_parent_revision=(
                (discovery.revision if discovery else 0) if query.scope == "family" else None
            ),
        )
    except service.InvalidCursor:
        raise _refuse("bad_cursor", "that page cursor was not issued for this request") from None
    if answer.get("cursor_reset"):
        start = 0
    for row in answer["rows"]:
        _decorate(root, row, referenced, home)
    if answer["state"] == "building":
        _kick(root, family, query.scope)
    return _fit(answer, query, start)


def find_row(root: Path, session_id: str, file_id: str) -> dict[str, Any]:
    """The ``row`` phase: one row by id (display form) and its pull source."""
    from local_operator.session_files import family as family_mod
    from local_operator.session_files import service

    root = Path(root)
    directory = _session_dir(root, session_id)
    if not service._ID_RE.match(file_id or ""):  # noqa: SLF001 — the service's id shape
        raise _refuse("not_found", "that file is not in this conversation's list")
    home = str(Path.home())
    deny = service.make_deny(root)
    family, _ = family_mod.family_blocking(root, session_id, directory)
    # Unbudgeted, as ``materialise`` is: a row id exists because a listing scanned it.
    states = service.advance(root, family, "family", budget_s=None, home=home, deny=deny)
    entry = service.find_artifact(service.merge(states, home=home, deny=deny), file_id)
    if entry is None:
        raise _refuse("not_found", "that file is not in this conversation's list")
    from local_operator.session.attachments import store_for_transcript_dir

    row = service.project(
        entry, home=home, display=False, store_root=store_for_transcript_dir(directory).root
    )
    source = _decorate(root, row, _Referenced(root), home)
    return {"row": row, "source": source}


def make_handler(server: "RelayServer") -> Callable[["PeerLink", dict[str, Any]], dict[str, Any]]:
    """The ``net_session_files`` handler. It issues no request over the link it answers."""

    def _handle(_link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        if isinstance(frame.get("frame"), dict):
            raise _refuse("bad_request", "a files listing carries no nested frame")
        session_id = str(frame.get("session_id") or "")
        if not session_id:
            raise _refuse("bad_request", "a files listing names the session it reads")
        phase = str(frame.get("phase") or "list")
        if phase == "list":
            return list_index(server.root, session_id, frame)
        if phase == "row":
            return find_row(server.root, session_id, str(frame.get("file_id") or ""))
        raise _refuse("bad_request", f"unknown files phase {phase!r}")

    return _handle


# ---------------------------------------------------------------------------
# The viewer's half
# ---------------------------------------------------------------------------


def ask_peer(server: "RelayServer", frame: dict[str, Any]) -> dict[str, Any]:
    """Forward one ``net_session_files`` phase to the owner; refusals keep their code."""
    from local_operator.network.files import _printable

    peer = str(frame.get("peer") or "")
    session_id = str(frame.get("session_id") or "")
    if not peer or not session_id:
        raise _refuse("bad_request", "a files listing names a peer and a session")
    device_id = server._resolve_peer(peer)  # noqa: SLF001 — the family's one name resolver
    label = _printable(server._peer_label(peer))  # noqa: SLF001
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    if link is None:
        raise _refuse("peer_unreachable", f"{label} cannot be reached from this device right now")
    if wire.MESH_FILE_INDEX_V1 not in link.capabilities:
        raise _refuse(
            "predates_file_index",
            f"{label} does not list its files to other devices yet; update it with "
            f"`lop network update {label}`",
        )
    fields = {key: frame[key] for key in ("phase", "index", "query", "file_id") if key in frame}
    reply = link.request(
        {
            "op": "net_session_files",
            "req": server._next_relay_req(),  # noqa: SLF001 — the relay's req counter
            "locality": "remote",
            "session_id": session_id,
            **fields,
        },
        timeout=server.slow_request_timeout("net_session_files"),
    )
    if reply is None:
        raise _refuse("peer_unreachable", f"{label} stopped answering")
    if reply.get("op") == "error":
        message = _printable(str(reply.get("message") or "it refused the listing"))
        raise _refuse(str(reply.get("code") or "peer_refused"), f"{label} refused: {message}")
    detail = reply.get("detail")
    if reply.get("op") != "ack" or not isinstance(detail, dict):
        raise _refuse("peer_refused", f"{label} answered something this build cannot read")
    return detail


def local_handler(server: "RelayServer") -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        return ask_peer(server, frame)

    return _handle
