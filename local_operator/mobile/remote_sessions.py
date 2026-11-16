"""Owner-bound mobile access to mesh sessions.

The phone supplies only a session id. Ownership and transport facts are resolved
inside the authenticated relay; no peer address or owner control key crosses
HTTP. Reading a stopped conversation must never engage its runtime.
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_operator.resume import SessionRow


@dataclass
class RemoteSessionError(Exception):
    """A mesh failure, distinct from expired phone authentication."""

    code: str
    status: int = 503

    def payload(self) -> dict[str, Any]:
        return {"error": self.code, "code": self.code, "remote": True}


def public_error(error: Exception, *, command: bool = False) -> RemoteSessionError:
    """Keep mesh/transport failures outside the phone's HTTP-401 namespace."""
    if isinstance(error, RemoteSessionError):
        return error
    from local_operator.network.projection import ProjectionRefusal, RemoteStreamRefusal
    from local_operator.network.types import MeshRefusal

    if isinstance(error, (MeshRefusal, ProjectionRefusal, RemoteStreamRefusal)):
        code = getattr(error, "code", "")
        if code in ("unsupported", "unknown_op"):
            return RemoteSessionError("remote_unsupported", 409)
        if code in (
            "peer_unreachable",
            "relay_unavailable",
            "session_unresolved",
            "session_unreachable",
            "session.relay_unavailable",
        ):
            return RemoteSessionError("remote_unreachable")
        return RemoteSessionError("remote_forbidden", 403)
    # A lost acknowledgement is not evidence of non-admission. Retain the
    # caller's UUID and let only a deliberate retry reach owner deduplication.
    return RemoteSessionError(
        "remote_command_ambiguous" if command else "remote_unreachable",
        409 if command else 503,
    )


def resolve_owner(root: Path, session_id: str) -> SessionRow | None:
    """Resolve once, retaining the session/owner binding for the operation.

    A partial mesh read is not evidence of local ownership. In particular callers
    must not fall back to creating a local runtime after an unreachable peer.
    """
    from local_operator.session.remote_open import remote_row_and_silence

    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        raise RemoteSessionError("remote_session_missing", 404)
    row, silent = remote_row_and_silence(session_id, root)
    if row is None and silent:
        raise RemoteSessionError("remote_owner_unresolved")
    if row is not None and not row.reachable:
        raise RemoteSessionError("remote_unreachable")
    return row


READ_CAPABILITY = "mobile-session-read-v1"
#: Turn-supplement bytes on the same carrier (``wire.MOBILE_SESSION_SUPPLEMENTS_V1``).
#: Spelled as a literal like ``READ_CAPABILITY`` so this module stays import-light;
#: ``test_credentials_lifecycle_recipient`` pins both against ``wire``'s tuple.
SUPPLEMENTS_CAPABILITY = "mobile-session-supplements-v1"

#: A supplement read's address shapes. A job is ``persistence.new_job_id``'s 12 hex;
#: a digest is the attachment store's 32 hex. Both are lookup KEYS checked on both
#: peers before any lookup — never a path, never a URL.
_SUPPLEMENT_REF = {
    "file": re.compile(r"[a-f0-9]{12}"),
    "document": re.compile(r"[a-f0-9]{32}"),
    "sound": re.compile(r"[a-f0-9]{32}"),
}

#: Owner refusal tokens (``session/media.py``) -> mesh refusal codes. The codes are
#: mapped to phone statuses in :func:`_read_detail`; keep the two tables exhaustive.
_SUPPLEMENT_REFUSALS = {
    "missing": ("media_missing", "supplement unavailable"),
    "no-cwd": ("supplement_needs_running", "file previews need the session running"),
    "too-large": ("too_large", "supplement exceeds the preview limit"),
}


def validate_read(frame: dict[str, Any]) -> None:
    """Validate additive fields on BOTH sides of the authenticated carrier."""
    from local_operator.network.types import MeshRefusal

    purpose = frame.get("mobile_session_read")
    if purpose not in ("history", "projection", "media", "supplement"):
        raise MeshRefusal("protocol_error", "invalid mobile_session_read")
    allowed = {"op", "req", "peer", "locality", "session_id", "mobile_session_read", "limit"}
    if purpose == "media":
        allowed.update({"entry_id", "index", "kind", "offset", "length", "expected_media_id"})
    elif purpose == "supplement":
        allowed.update({"kind", "ref", "index", "offset", "length", "expected_media_id"})
    elif purpose == "history":
        allowed.update({"before_id", "after_id"})
    if set(frame) - allowed:
        raise MeshRefusal("protocol_error", "mixed mobile read fields")
    if purpose == "media":
        if not isinstance(frame.get("entry_id"), str) or not frame["entry_id"]:
            raise MeshRefusal("protocol_error", "invalid media entry")
        if frame.get("kind") not in ("image", "video", "file"):
            raise MeshRefusal("protocol_error", "invalid media kind")
    if purpose == "supplement":
        kind = frame.get("kind")
        if kind not in _SUPPLEMENT_REF:
            raise MeshRefusal("protocol_error", "invalid supplement kind")
        ref = frame.get("ref")
        if not isinstance(ref, str) or _SUPPLEMENT_REF[kind].fullmatch(ref) is None:
            raise MeshRefusal("protocol_error", "invalid supplement reference")
    if purpose in ("media", "supplement"):
        from local_operator.network.sync import SYNC_CHUNK_BYTES

        for key in ("index", "offset", "length"):
            if type(frame.get(key)) is not int or frame[key] < 0:
                raise MeshRefusal("protocol_error", "invalid media coordinate")
        if not 0 < frame["length"] <= SYNC_CHUNK_BYTES:
            raise MeshRefusal("protocol_error", "invalid media length")
        if purpose == "supplement" and frame["kind"] != "file" and frame["index"] != 0:
            # Digest families have exactly one blob per address; a non-zero index
            # would be a second spelling of the same read.
            raise MeshRefusal("protocol_error", "invalid supplement index")
        expected = frame.get("expected_media_id")
        if expected is not None and (
            not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None
        ):
            raise MeshRefusal("protocol_error", "invalid media identity")
    for key in ("before_id", "after_id"):
        value = frame.get(key)
        if value is not None and (not isinstance(value, str)):
            raise MeshRefusal("protocol_error", "invalid history cursor")
    if frame.get("before_id") and frame.get("after_id"):
        raise MeshRefusal("protocol_error", "exclusive history cursors")


def _owner_cwd(root: Path, session_id: str) -> str:
    """The cwd of THIS device's live runtime for ``session_id``, else ``""``.

    The owner's own discovery record — the source the relay's catalogue already
    reads (``RelayServer`` rows) — and never anything the viewer sent. ``reap=False``
    keeps a read from moving a dead record aside (reaping is the daemon's job), and a
    ``stale`` record is no runtime: an empty cwd makes a cwd-relative featured file
    ``supplement_needs_running``, exactly the local route's 409.
    """
    from local_operator.session.runtime import registry

    try:
        records = registry.scan(root, reap=False)
    except OSError:
        return ""
    for record, state in records:
        if state != "stale" and getattr(record, "session_id", None) == session_id:
            return str(getattr(record, "cwd", "") or "")
    return ""


def _serve_chunk(
    frame: dict[str, Any], session_id: str, source: Any, address: dict[str, Any]
) -> dict[str, Any]:
    """One validated byte window of ``source`` as the carrier's reply.

    Shared by the ``media`` and ``supplement`` purposes so both apply the same
    identity, short-read and reply-budget checks; ``address`` is the purpose's
    own echo (``entry_id`` or ``ref``) the viewer re-checks.
    """
    from local_operator.network.relay import (
        SESSION_HISTORY_REPLY_BUDGET_BYTES,
        session_history_reply_bytes,
    )
    from local_operator.network.types import MeshRefusal

    if frame.get("expected_media_id") not in (None, source.media_id):
        raise MeshRefusal("media_changed", "retained media changed")
    offset = frame["offset"]
    if offset > source.total:
        raise MeshRefusal("protocol_error", "media offset past end")
    try:
        data = source.read(offset, frame["length"])
    except OSError as exc:
        raise MeshRefusal("media_changed", "retained media changed") from exc
    if len(data) != min(frame["length"], source.total - offset):
        raise MeshRefusal("media_changed", "retained media changed")
    result = {
        "mobile_session_read": 1,
        "session_id": session_id,
        **address,
        "index": frame["index"],
        "kind": frame["kind"],
        "media_id": source.media_id,
        "mime": source.mime,
        "total": source.total,
        "offset": offset,
        "data_b64": base64.b64encode(data).decode("ascii"),
    }
    # Raw chunks reuse SYNC_CHUNK_BYTES, whose base64 envelope already fits
    # the mesh. Check the actual ack too: oversized metadata cannot tear
    # down a shared authenticated link.
    if session_history_reply_bytes(frame.get("req"), result) > SESSION_HISTORY_REPLY_BUDGET_BYTES:
        raise MeshRefusal("page_too_large", "media reply exceeds mesh budget")
    return result


def _supplement_source(root: Path, session_id: str, frame: dict[str, Any]) -> Any:
    """Resolve a supplement read ON THE OWNER, per chunk, by the local routes' policy.

    Re-resolved on every frame (no turn-time snapshot, no cached handle): a grant
    revoked, a row rewritten or a file changed between two chunks is judged afresh
    at the next one. Featured files run ``resolve_supplement_file`` — the owner's
    own phone route's policy, by the operator's 2026-11-15 decision — with the
    owner's OWN live cwd (:func:`_owner_cwd`), never anything the viewer sent.
    """
    from local_operator.network.types import MeshRefusal
    from local_operator.session.media import (
        resolve_supplement_file,
        supplement_digest_media,
    )

    if frame["kind"] == "file":
        source = resolve_supplement_file(
            root / "sessions" / session_id,
            frame["ref"],
            frame["index"],
            cwd=_owner_cwd(root, session_id),
        )
    else:
        source = supplement_digest_media(root, session_id, frame["ref"], frame["kind"])
    if isinstance(source, str):
        code, reason = _SUPPLEMENT_REFUSALS.get(source, _SUPPLEMENT_REFUSALS["missing"])
        raise MeshRefusal(code, reason)
    return source


def owner_read(root: Path, session_id: str, frame: dict[str, Any]) -> dict[str, Any]:
    """Serve mobile schema without waking a runtime or reading a client path."""
    from local_operator.mobile.daemon import (
        _durable_projection,
        _history_page,
        _history_response,
        _projection_frame,
    )
    from local_operator.network.relay import (
        SESSION_HISTORY_REPLY_BUDGET_BYTES,
        fit_session_history_reply,
        session_history_reply_bytes,
        validate_history_limit,
    )
    from local_operator.network.types import MeshRefusal

    validate_read(frame)
    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        raise MeshRefusal("unknown_session", "unknown user session")
    from local_operator.resume import is_user_session
    from local_operator.session.compression import has_transcript

    directory = root / "sessions" / session_id
    if not directory.is_dir() or not is_user_session(directory):
        raise MeshRefusal("unknown_session", "unknown user session")
    if not has_transcript(directory):
        # An actual owner-created user directory can be empty before its first
        # turn. Reading it is an empty OWNER frame, never a local materialize or
        # an engage. Corrupt existing transcripts do not take this branch.
        if frame["mobile_session_read"] == "projection":
            from local_operator.mobile.types import SessionProjection

            return {
                "mobile_session_read": 1,
                "session_id": session_id,
                "projection": _projection_frame(SessionProjection(session_id=session_id, pid=0)),
            }
        if frame["mobile_session_read"] == "history":
            return {
                "mobile_session_read": 1,
                "session_id": session_id,
                "entries": [],
                "has_more": False,
            }
        raise MeshRefusal("media_missing", "retained media unavailable")
    if frame["mobile_session_read"] == "media":
        from local_operator.session.media import session_media

        source = session_media(root, session_id, frame["entry_id"], frame["index"], frame["kind"])
        if source is None:
            raise MeshRefusal("media_missing", "retained media unavailable")
        return _serve_chunk(frame, session_id, source, {"entry_id": frame["entry_id"]})
    if frame["mobile_session_read"] == "supplement":
        return _serve_chunk(
            frame, session_id, _supplement_source(root, session_id, frame), {"ref": frame["ref"]}
        )
    if frame["mobile_session_read"] == "projection":
        projection = _durable_projection(session_id, root=root)
        if projection is None:
            raise MeshRefusal("unavailable", "mobile projection unavailable")
        result = {
            "mobile_session_read": 1,
            "session_id": session_id,
            "projection": _projection_frame(projection),
        }
        if (
            session_history_reply_bytes(frame.get("req"), result)
            > SESSION_HISTORY_REPLY_BUDGET_BYTES
        ):
            raise MeshRefusal("page_too_large", "projection exceeds mesh budget")
        return result
    page, more = _history_page(
        session_id,
        frame.get("before_id"),
        validate_history_limit(frame.get("limit")),
        after=frame.get("after_id"),
        root=root,
    )
    result = {"mobile_session_read": 1, "session_id": session_id, **_history_response(page, more)}
    if getattr(page, "reset", ""):
        result.update(reset=True, code="remote_cursor_missing")
    # The existing authenticated reply budget bounds transport. Its fitter
    # preserves the cursor-bearing chronological tail, exactly as journal pages.
    # Forward reads keep the OLDEST part of the requested newer window. The
    # desktop fitter keeps a tail; reversing around it preserves forward reach.
    if frame.get("after_id"):
        result["entries"].reverse()
        result = fit_session_history_reply(
            frame.get("req"), result, entry_bounder=_bound_history_entry
        )
        result["entries"].reverse()
        return result
    return fit_session_history_reply(frame.get("req"), result, entry_bounder=_bound_history_entry)


def _bound_history_entry(entry: dict[str, Any], cap_bytes: int) -> dict[str, Any]:
    from local_operator.mobile.projection import cap_projection_frame
    from local_operator.mobile.types import SessionProjection, TranscriptEntry

    # Reuse the existing mobile degradation tiers, including text_complete=false
    # for a transported prefix. A one-row projection cannot reorder/drop sibling
    # rows, and the history fitter preserves this row's cursor identity.
    frame, _degraded = cap_projection_frame(
        SessionProjection(session_id="", pid=0, transcript=[TranscriptEntry(**entry)]),
        cap_bytes=cap_bytes,
    )
    return frame["transcript"][0]


def mesh_read(root: Path, row: SessionRow, purpose: str, **fields: Any) -> dict[str, Any]:
    """Call our local authenticated courier; retain typed refusals for HTTP."""
    from local_operator.network import relay, store

    record = store.find_own_relay(root)
    if record is None:
        raise RemoteSessionError("remote_unreachable")
    reply = relay.control_request(
        record,
        "peer_session_history",
        timeout=relay.session_history_client_bound_s(),
        peer=row.owner_device,
        session_id=row.id,
        mobile_session_read=purpose,
        **fields,
    )
    return _read_detail(reply, session_id=row.id)


async def async_mesh_read(
    root: Path, row: SessionRow, purpose: str, **fields: Any
) -> dict[str, Any]:
    from local_operator.network import relay, store

    record = await asyncio.to_thread(store.find_own_relay, root)
    if record is None:
        raise RemoteSessionError("remote_unreachable")
    reply = await relay.async_control_request(
        record,
        "peer_session_history",
        timeout=relay.session_history_client_bound_s(),
        peer=row.owner_device,
        session_id=row.id,
        mobile_session_read=purpose,
        **fields,
    )
    return _read_detail(reply, session_id=row.id)


def _read_detail(reply: dict[str, Any] | None, *, session_id: str) -> dict[str, Any]:
    if reply is None:
        raise RemoteSessionError("remote_unreachable")
    if reply.get("op") == "error":
        code = str(reply.get("code", ""))
        if code in ("unsupported", "unknown_op"):
            raise RemoteSessionError("remote_unsupported", 409)
        if code == "unknown_session":
            raise RemoteSessionError("remote_session_missing", 404)
        if code == "media_missing":
            raise RemoteSessionError("remote_media_missing", 404)
        if code == "media_changed":
            raise RemoteSessionError("remote_media_changed", 409)
        # Supplement refusals the phone acts on (docs/mobile.md): "Open on <owner>"
        # when a cwd-relative file needs the owner's runtime, and the local
        # route's own 413.
        if code == "supplement_needs_running":
            raise RemoteSessionError("remote_supplement_needs_running", 409)
        if code == "too_large":
            raise RemoteSessionError("remote_too_large", 413)
        if code in ("peer_unreachable", "timeout", "unavailable", "page_too_large"):
            raise RemoteSessionError("remote_unreachable")
        # Authorization deliberately conceals its finer reason at the mesh
        # boundary. A refusal is not a phone password expiration.
        raise RemoteSessionError("remote_forbidden", 403)
    detail = reply.get("detail")
    if (
        not isinstance(detail, dict)
        or type(detail.get("mobile_session_read")) is not int
        or detail["mobile_session_read"] != 1
    ):
        raise RemoteSessionError("remote_unsupported", 409)
    if reply.get("op") != "ack" or detail.get("session_id") != session_id:
        raise RemoteSessionError("remote_owner_unresolved")
    projection = detail.get("projection")
    if isinstance(projection, dict) and projection.get("session_id") != session_id:
        raise RemoteSessionError("remote_owner_unresolved")
    return detail


class RemoteView:
    """One upstream mobile projection stream, shared by HTTP viewers only."""

    def __init__(self, root: Path, row: SessionRow) -> None:
        self.root = root
        self.row = row
        self.queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self.client: Any = None
        self.frame: dict[str, Any] | None = None
        self.lock = asyncio.Lock()
        self.readers = 0

    def publish(self, frame: dict[str, Any]) -> None:
        if not frame.get("remote"):
            self.frame = frame
        for queue in self.queues:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(frame)

    async def connect(self) -> None:
        async with self.lock:
            await self._connect()

    async def _connect(self) -> None:
        # Each newly admitted HTTP reader must cross owner authorization before
        # it can receive a cached frame. A shared live connection proves only
        # that an earlier reader was admitted, not that view is still granted.
        # Keep this inside the per-view lock and tentative-reference lifetime:
        # denial/cancellation drops only this acquire, leaving existing readers.
        result = await asyncio.to_thread(mesh_read, self.root, self.row, "projection")
        if self.client is not None and self.client.connected:
            return
        self.close()
        from local_operator.mobile.daemon import _projection_frame
        from local_operator.network.projection import PeerRow, remote_owner_for

        # Negotiate the complete owner-side mobile schema even for a LIVE
        # runtime. Otherwise an old peer could appear to open successfully but
        # fail history/media only after the user scrolls or opens an attachment.
        self.publish(result["projection"])
        owner = await asyncio.to_thread(
            remote_owner_for,
            self.row.id,
            config_dir=self.root,
            root=self.root,
            row=PeerRow(
                session_id=self.row.id,
                device_id=self.row.owner_device,
                device_name=self.row.owner_device_name,
                reachable=self.row.reachable,
            ),
        )
        record, _pid = await asyncio.to_thread(owner.locate)
        if record is None:
            # locate never engages: keep the durable owner frame already read.
            return

        def repaint(projection: Any) -> None:
            if self.client is not client:
                return
            if projection.session_id != self.row.id:
                self.close()
                self.publish(RemoteSessionError("remote_session_missing", 404).payload())
                return
            projection.pid = record.pid
            self.publish(_projection_frame(projection))

        def disconnected(_reason: str) -> None:
            if self.client is not client:
                return
            # The loaded frame remains with the phone. This is a typed event,
            # never an empty replacement projection or a login failure.
            self.publish(RemoteSessionError("remote_unreachable").payload())

        client = owner.make_client(repaint, disconnected)
        self.client = client
        try:
            await client.connect(record, self.row.id)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            client.close()


class RemoteViews:
    """Reference-count upstream readers by retained owner AND session identity.

    No idle cache/lease: the final HTTP/SSE release closes the upstream client.
    Operations still traverse mesh authorization; having a view grants no drive.
    """

    def __init__(self) -> None:
        self.views: dict[tuple[str, str], RemoteView] = {}

    async def acquire(
        self, root: Path, row: SessionRow
    ) -> tuple[RemoteView, asyncio.Queue[dict[str, Any]]]:
        key = (row.owner_device, row.id)
        view = self.views.get(key)
        if view is None:
            view = RemoteView(root, row)
            self.views[key] = view
        # Count in-flight acquires too: cancellation cannot close a viewer
        # another request is still joining. Only this session's lock spans
        # network IO, so an offline peer cannot serialize unrelated opens.
        view.readers += 1
        try:
            await view.connect()
        except BaseException:
            self._drop(view)
            raise
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        view.queues.add(queue)
        return view, queue

    def _drop(self, view: RemoteView) -> None:
        view.readers -= 1
        if not view.readers:
            key = (view.row.owner_device, view.row.id)
            if self.views.get(key) is view:
                self.views.pop(key)
            view.close()

    def release(self, view: RemoteView, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in view.queues:
            view.queues.remove(queue)
            self._drop(view)


async def admit_command(
    root: Path, row: SessionRow, op: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Use the desktop's owner admission/staging and the PHONE's stable UUID."""
    from local_operator.mobile.types import ContinuationCommand, validate_control_frame
    from local_operator.server.utils.desktop_sessions import prepare_images_for_owner
    from local_operator.session.remote_open import open_remote_viewer

    if op not in ("prompt", "steer") or body.get("files") or body.get("audio"):
        raise RemoteSessionError("remote_unsupported", 409)
    validate_control_frame({**body, "op": op, "session_id": row.id})
    command = ContinuationCommand.from_json(
        {**body, "session_id": row.id, "images": body.get("images", [])}
    )

    async def never_take_over() -> None:
        raise RemoteSessionError("remote_owner_unresolved")

    viewer = await open_remote_viewer(
        row.id,
        config_dir=root,
        row=row,
        takeover=never_take_over,
        slash_consumers=(),
    )
    if viewer is None:
        raise RemoteSessionError("remote_owner_unresolved")
    try:
        images = await asyncio.to_thread(prepare_images_for_owner, root, command.images)
        if len(images) != len(command.images):
            raise ValueError("invalid remote image")
        detail, duplicate = await viewer.admit_prompt(
            command.text,
            command_id=command.command_id,
            images=images,
            steer=op == "steer",
            input_mode=body.get("input_mode"),
            input_path=body.get("input_path"),
        )
        return {"ok": True, "detail": detail, "duplicate": duplicate}
    finally:
        await viewer.dispose()


def media_chunk(
    root: Path,
    row: SessionRow,
    *,
    entry_id: str,
    index: int,
    kind: str,
    offset: int,
    length: int,
    expected_media_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Validate an owner reply before any bytes reach an HTTP consumer."""
    fields: dict[str, Any] = {
        "entry_id": entry_id,
        "index": index,
        "kind": kind,
        "offset": offset,
        "length": length,
    }
    if expected_media_id is not None:
        fields["expected_media_id"] = expected_media_id
    result = mesh_read(root, row, "media", **fields)
    return _checked_media_result(
        row, result, address_key="entry_id", **_address(fields, "entry_id")
    )


async def async_media_chunk(
    root: Path,
    row: SessionRow,
    *,
    entry_id: str,
    index: int,
    kind: str,
    offset: int,
    length: int,
    expected_media_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    fields: dict[str, Any] = {
        "entry_id": entry_id,
        "index": index,
        "kind": kind,
        "offset": offset,
        "length": length,
    }
    if expected_media_id is not None:
        fields["expected_media_id"] = expected_media_id
    result = await async_mesh_read(root, row, "media", **fields)
    return _checked_media_result(
        row, result, address_key="entry_id", **_address(fields, "entry_id")
    )


def _supplement_fields(
    kind: str, ref: str, index: int, offset: int, length: int, expected_media_id: str | None
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "kind": kind,
        "ref": ref,
        "index": index,
        "offset": offset,
        "length": length,
    }
    if expected_media_id is not None:
        fields["expected_media_id"] = expected_media_id
    return fields


def supplement_chunk(
    root: Path,
    row: SessionRow,
    *,
    kind: str,
    ref: str,
    index: int,
    offset: int,
    length: int,
    expected_media_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """One owner-validated window of a supplement blob (``document``/``file``/``sound``).

    The supplement twin of :func:`media_chunk`: same carrier, same reply checks,
    addressed by ``ref`` (a job for ``file``, a digest otherwise) instead of an
    entry id. The relay forwards it only to an owner advertising
    :data:`SUPPLEMENTS_CAPABILITY`.
    """
    fields = _supplement_fields(kind, ref, index, offset, length, expected_media_id)
    result = mesh_read(root, row, "supplement", **fields)
    return _checked_media_result(row, result, address_key="ref", **_address(fields, "ref"))


async def async_supplement_chunk(
    root: Path,
    row: SessionRow,
    *,
    kind: str,
    ref: str,
    index: int,
    offset: int,
    length: int,
    expected_media_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    fields = _supplement_fields(kind, ref, index, offset, length, expected_media_id)
    result = await async_mesh_read(root, row, "supplement", **fields)
    return _checked_media_result(row, result, address_key="ref", **_address(fields, "ref"))


def _address(fields: dict[str, Any], key: str) -> dict[str, Any]:
    """``fields`` with the purpose's address renamed to the checker's ``address``."""
    renamed = {name: value for name, value in fields.items() if name != key}
    renamed["address"] = fields[key]
    return renamed


def _checked_media_result(
    row: SessionRow,
    result: dict[str, Any],
    *,
    address_key: str,
    address: str,
    index: int,
    kind: str,
    offset: int,
    length: int,
    expected_media_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Validate an owner's chunk reply before any byte reaches an HTTP consumer.

    ``address_key`` names the purpose's echoed address (``entry_id`` for retained
    media, ``ref`` for supplements); every other identity check is shared, so the
    two purposes cannot drift in what they accept from a peer.
    """
    import binascii

    identity = result.get("media_id")
    if (
        result.get("session_id") != row.id
        or result.get(address_key) != address
        or type(result.get("index")) is not int
        or result["index"] != index
        or result.get("kind") != kind
        or type(result.get("offset")) is not int
        or result["offset"] != offset
        or type(result.get("total")) is not int
        or result["total"] < offset
        or not isinstance(identity, str)
        or re.fullmatch(r"[a-f0-9]{64}", identity) is None
        or expected_media_id is not None
        and identity != expected_media_id
        or not isinstance(result.get("mime"), str)
        or "\r" in result["mime"]
        or "\n" in result["mime"]
        or not result["mime"].isascii()
    ):
        raise RemoteSessionError("remote_media_changed", 409)
    encoded = result.get("data_b64")
    if not isinstance(encoded, str):
        raise RemoteSessionError("remote_media_changed", 409)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RemoteSessionError("remote_media_changed", 409) from exc
    if len(data) != min(length, result["total"] - offset):
        raise RemoteSessionError("remote_media_changed", 409)
    return result, data
