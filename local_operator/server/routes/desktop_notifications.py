"""The machine-wide inbox: ``notify`` items and open asks, read and acted on.

``docs/design/notifications-inbox.md`` §5.2. Five routes over
``notifications.db`` and the ask queue's derived index:

- ``GET   /v1/desktop/notifications``        — one page of items + the waiting asks
- ``GET   /v1/desktop/notifications/count``  — badge figures, for pollers
- ``GET   /v1/desktop/notifications/{id}``   — one item with its context
- ``PATCH /v1/desktop/notifications``        — read / unread / dismiss / undismiss
- ``POST  /v1/desktop/notifications/seen``   — the in-view auto-read evidence door

**Cold, like the asks aggregate.** No route here opens a session, dials a
runtime or starts one: items outlive the runtime that raised them, so the
inbox must answer with nothing running. Store reads run in a worker thread.

**Token-bound writes (R10).** A by-id write names the ``revision`` it
rendered; a by-session or all-items write is bounded by ``through_seq`` (the
``high_water`` the client last saw). A keyed update published after the
render is therefore never cleared by it. An open ask under ``dismiss`` is
skipped with ``live_ask``: the reel never settles an ask (§2.3).

**Store failures keep their own sentences** (the ``receipts_refusal``
precedent): the classification is ``session/store_failures``', the words say
what this route was doing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pathlib
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from local_operator.i18n.keys import wire_errors_notifications as _wn
from local_operator.i18n.messages import render
from local_operator.notifications import bulk
from local_operator.notifications import items as inbox
from local_operator.notifications import reconcile
from local_operator.notifications.store import NotificationStore
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import Input, errors, reply
from local_operator.session.store_failures import (
    STORE_BUSY,
    STORE_OUT_OF_SPACE,
    StoreFailure,
    display_root,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop notifications"], dependencies=[Depends(require_desktop)])

#: An item id: a notify row (``n_`` + 16 base32) or an open-ask ref
#: (``ask:<session_id>:<ask_id>``). Shape-checked so a malformed id is a 422
#: before any store is opened.
#: ONE regex for the id shape, used by the PATCH models AND the detail path
#: (CD8). The detail route used to accept anything with the right prefix and
#: length: ``n_................`` and ``ask:..:x`` both passed, and the second
#: reached ``ask_store.read_entry(config, "..")``. Starlette refuses a ``/`` in a
#: path param (``%2F`` is a 404, measured), so it was not exploitable — but a
#: detail route validating looser than the write models is a defect regardless.
_ITEM_ID_PATTERN = r"^(n_[a-z2-7]{16}|ask:[a-f0-9]{12}:[A-Za-z0-9_-]{1,64})$"
ItemID = Annotated[str, Field(pattern=_ITEM_ID_PATTERN)]
_ITEM_ID_RE = re.compile(_ITEM_ID_PATTERN)
SessionID = Annotated[str, Field(pattern=r"^[a-f0-9]{12}$")]


def _text(message: Any) -> str:
    return render(message.code, message.params)


def notifications_refusal(failure: StoreFailure, root: pathlib.Path | None) -> str:
    """This surface's refusal sentence: it reads and marks inbox items."""
    where = display_root(root)
    if failure.code == STORE_BUSY:
        return _text(_wn.store_busy())
    if failure.code == STORE_OUT_OF_SPACE:
        return _text(_wn.store_full(where=where))
    return _text(_wn.store_unavailable(where=where))


def _config_dir(request: Request) -> pathlib.Path:
    return pathlib.Path(request.app.state.config_manager.config_dir)


def _lapsed_window(root: pathlib.Path) -> int:
    """``asks.lapsed_visible_for``, polled fresh per request ON the loop (the asks route)."""
    from local_operator.asks.visibility import lapsed_visible_ms

    return lapsed_visible_ms(root, poll_now=True)


#: A list cursor: the store's own shape (``<epoch_ms>.<item_id>``) with the two
#: wire bounds the store depends on — a length and a MAGNITUDE. Both transports
#: validate with this one annotation, so a value sqlite cannot bind is a 422 on
#: either surface and never reaches the query (QA round 2).
CursorQuery = Annotated[str | None, AfterValidator(inbox.cursor_in_range)]


class NotificationsQuery(BaseModel):
    """The list route's query, as ONE model both transports validate with.

    The relay (``mobile/daemon.py``) validates its query string through this
    same class, so the phone and the app refuse exactly the same requests
    (round-1 R7: four queries answered 200 on the relay and 422 here). Unknown
    query keys are ignored, as FastAPI ignores them on this route.
    """

    model_config = ConfigDict(extra="ignore")

    state: Literal["unread", "all", "dismissed"] = "unread"
    source: Literal["notify", "ask"] | None = None
    priority: Literal["high", "normal", "low"] | None = None
    session_id: Annotated[str | None, Field(pattern=r"^[a-f0-9]{12}$")] = None
    cursor: Annotated[CursorQuery, Field(max_length=80)] = None
    through_seq: Annotated[int | None, Field(ge=0)] = None
    limit: Annotated[int, Field(ge=1, le=inbox.LIST_LIMIT_MAX)] = inbox.LIST_LIMIT_DEFAULT


def _bulk_scope(request: Request) -> str:
    # Authorization already passed require_desktop. Persist a one-way binding,
    # never the bearer: a rotated or differently scoped credential cannot reuse
    # an old receipt, even if its token was copied to another client.
    return hashlib.sha256(request.headers["authorization"].encode()).hexdigest()


def _bulk_preview(
    root: pathlib.Path, filters: dict[str, Any], scope: str, window: int
) -> dict[str, Any]:
    asks = []
    if filters["kind"] != "updates" and filters["state"] != "dismissed":
        asks = inbox.open_asks(root, lapsed_visible_ms=window)
        if filters["session_id"]:
            asks = [row for row in asks if row.get("session_id") == filters["session_id"]]
    # Open asks are high priority and stay visible even after their read mark
    # changes. Counting the index directly avoids formatting every waiting row.
    selection = bulk.preview(
        NotificationStore(root), filters, scope=scope, skipped_ask_count=len(asks), ask_rows=asks
    )
    return {"selection": selection, **inbox.counts(root)}


@router.get("/v1/desktop/notifications", response_model=CRUDResponse[dict[str, Any]])
async def list_notifications(
    request: Request,
    state: Literal["unread", "all", "dismissed"] = "unread",
    source: Literal["notify", "ask"] | None = None,
    priority: Literal["high", "normal", "low"] | None = None,
    session_id: Annotated[str | None, Query(pattern=r"^[a-f0-9]{12}$")] = None,
    cursor: Annotated[CursorQuery, Query(max_length=80)] = None,
    through_seq: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=inbox.LIST_LIMIT_MAX)] = inbox.LIST_LIMIT_DEFAULT,
    preview: bool = False,
    include_ask_reads: bool = False,
    kind: Literal["all", "asks", "updates"] | None = None,
    bulk_priority: Literal["any", "high", "normal_up"] | None = None,
    count_session_ids: Annotated[list[SessionID] | None, Query(max_length=50)] = None,
    selection_token: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_-]{43}$")] = None,
    excluded_ids: Annotated[list[ItemID] | None, Query(max_length=500)] = None,
):
    """One page of items, newest content first; ``waiting`` on the first page only."""
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        window = _lapsed_window(root)
        if preview and selection_token is not None:
            if set(request.query_params) - {"preview", "selection_token", "excluded_ids"}:
                raise HTTPException(422, detail={"code": "selection_filter_conflict"})
            try:
                selection = await asyncio.to_thread(
                    bulk.refresh,
                    NotificationStore(root),
                    selection_token,
                    scope=_bulk_scope(request),
                    excluded_ids=excluded_ids or (),
                )
            except bulk.SelectionError as error:
                raise HTTPException(error.status, detail={"code": error.code}) from error
            return reply({"selection": selection, **await asyncio.to_thread(inbox.counts, root)})
        if (
            selection_token is not None
            or excluded_ids is not None
            or (include_ask_reads and not preview)
        ):
            raise HTTPException(422, detail={"code": "selection_preview_required"})
        if preview:
            allowed = {
                "preview",
                "state",
                "kind",
                "bulk_priority",
                "session_id",
                "count_session_ids",
                "include_ask_reads",
            }
            if set(request.query_params) - allowed:
                raise HTTPException(422, detail={"code": "selection_filter_conflict"})
            filters = {
                "state": state,
                "kind": kind or "all",
                "bulk_priority": bulk_priority or "any",
                "session_id": session_id,
                "count_session_ids": list(dict.fromkeys(count_session_ids or [])),
                "include_ask_reads": include_ask_reads,
            }
            try:
                return reply(
                    await asyncio.to_thread(
                        _bulk_preview, root, filters, _bulk_scope(request), window
                    )
                )
            except bulk.SelectionError as error:
                raise HTTPException(error.status, detail={"code": error.code}) from error
        if kind is not None or bulk_priority is not None or count_session_ids is not None:
            raise HTTPException(422, detail={"code": "selection_preview_required"})
        page = await asyncio.to_thread(
            inbox.list_page,
            root,
            state=state,
            source=source,
            priority=priority,
            session_id=session_id,
            cursor=cursor,
            limit=limit,
            lapsed_visible_ms=window,
            through_seq=through_seq,
        )
        return reply(page)


@router.get("/v1/desktop/notifications/count", response_model=CRUDResponse[dict[str, Any]])
async def count_notifications(request: Request):
    """Unread updates, pending asks and unread pending asks for a poller."""
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        return reply(await asyncio.to_thread(inbox.counts, root))


def anchor_excerpt(session_dir: pathlib.Path, anchor_id: str) -> str:
    """The notify call's own transcript row, as the operator would read it.

    The anchor is the ``tool_call_id`` of the ``notify`` call. Its result row
    (the tool's receipt) is what the transcript shows at that point, so the
    excerpt is that row's text, bounded and read from the tail of the journal
    (the call is usually recent). Absent when it cannot be found — honest
    absence, never a guessed line.
    """
    from local_operator.session.compression import read_tail_window

    # A session may have an archived journal. Use the same bounded resolver
    # as every other transcript reader; plain-name probes silently lose the
    # context precisely when the conversation becomes old enough to archive.
    raw_tail, truncated = read_tail_window(session_dir, 4 * 1024 * 1024)
    lines = raw_tail.splitlines()
    if truncated:
        lines = lines[1:]
    for raw in reversed(lines):
        if anchor_id.encode() not in raw:
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        text = _row_text(row, anchor_id)
        if text:
            return text
    return ""


def _row_text(row: Any, anchor_id: str) -> str:
    # The journal's tool-result row: ``{"type": "message", "payload": {"role":
    # "tool", "tool_call_id": ..., "content": [{"text": ...}]}}``.
    if not isinstance(row, dict) or row.get("type") != "message":
        return ""
    payload = row.get("payload")
    if not isinstance(payload, dict) or payload.get("role") != "tool":
        return ""
    if str(payload.get("tool_call_id") or "") != anchor_id:
        return ""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            # Journal blocks carry ``{"text": ...}`` with no ``type``; an image
            # block has no ``text`` and contributes nothing.
            if isinstance(block, dict) and block.get("type", "text") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


@router.get("/v1/desktop/notifications/{item_id}", response_model=CRUDResponse[dict[str, Any]])
async def notification_detail(item_id: str, request: Request):
    """One item with its history, session, anchor, projects and the session's asks."""
    if not valid_item_id(item_id):
        raise HTTPException(422, _wn.unknown_item().envelope())
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        window = _lapsed_window(root)
        answer = await asyncio.to_thread(
            inbox.detail,
            root,
            item_id,
            lapsed_visible_ms=window,
            anchor_excerpt=anchor_excerpt,
        )
    if answer is None:
        raise HTTPException(404, _wn.unknown_item().envelope())
    return reply(answer)


def valid_item_id(value: str) -> bool:
    """The PATCH models' own shape rule, shared by BOTH transports (CD8).

    Public because the relay validates detail ids through this module rather
    than hand-rolling a second predicate (R7); strict because the loose
    prefix/length check accepted ``n_................`` and ``ask:..:x`` while
    the write AND seen models refused them.
    """
    return _ITEM_ID_RE.match(value) is not None


class ItemToken(Input):
    """One item the caller RENDERED, named by id and the revision it showed."""

    id: ItemID
    revision: int = Field(ge=1)


class _NotificationTargets(Input):
    """Common legacy targets; only desktop advertises receipt-backed selection."""

    #: 1..500, the catalogue's own page bound (``attention/seen``'s reasoning):
    #: one user gesture never has to be chunked.
    ids: list[ItemToken] | None = Field(default=None, min_length=1, max_length=500)
    session_id: SessionID | None = None
    through_seq: int | None = Field(default=None, ge=0)
    all: bool = False
    observed: list[ItemToken] | None = Field(default=None, max_length=500)

    def _legacy_target(self) -> None:
        targets = sum((self.ids is not None, self.session_id is not None, bool(self.all)))
        if targets != 1:
            raise ValueError("name exactly one target: ids, session_id or all")
        if self.ids is None and (self.through_seq is None or self.observed is None):
            raise ValueError("bulk writes require through_seq and observed tokens")
        if self.ids is not None and self.observed is not None:
            raise ValueError("observed tokens belong only to bulk targets")


class NotificationsPatch(_NotificationTargets):
    """Legacy schema shared with the relay, which has no selection capability."""

    op: Literal["read", "unread", "dismiss", "undismiss"]

    @model_validator(mode="after")
    def _one_target(self) -> "NotificationsPatch":
        self._legacy_target()
        return self


#: The largest ``through_seq`` SQLite can bind. ``seq`` is an INTEGER column, so
#: a larger watermark cannot name any row; on the receipt path it would reach
#: ``bulk.apply``'s ``seq<=?`` bind and raise ``OverflowError`` (a 500). Only the
#: receipt path is bounded here: the legacy path clamps an oversized watermark to
#: ``high_water`` and the relay shares its validator, so its answer stays as it was.
_MAX_SEQ = 2**63 - 1


class DesktopNotificationsPatch(_NotificationTargets):
    """Additive desktop-only receipt contract, leaving relay validation intact.

    ``op: "preview"`` is the BODY form of the selection recount: it never
    mutates, never mints a receipt and never extends the handle's expiry. It
    exists because the GET form carries ``excluded_ids`` in the URL, and 500
    ask-shaped ids (~50 KB) exceed uvicorn/h11's 16 KiB incomplete-request-head
    limit whenever the kernel delivers the head in more than one read, so the
    request was refused with a 400 before the route ran (QA Q1 on #3103).
    Raising that limit is a server-wide hardening knob shared by every route,
    and lowering the cap would cut the 500-row page the UI needs; a body is not
    subject to that limit. PATCH is used because it is the one body-carrying
    method this path already serves, with the same token and exclusion fields.
    """

    op: Literal["read", "unread", "dismiss", "undismiss", "undo", "preview"]
    selection_token: Annotated[str | None, Field(pattern=r"^[A-Za-z0-9_-]{43}$")] = None
    operation_token: Annotated[str | None, Field(pattern=r"^[A-Za-z0-9_-]{43}$")] = None
    receipt: bool = False
    excluded_ids: list[ItemID] | None = Field(default=None, max_length=500)
    reconcile_ids: list[ItemID] | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _one_target(self) -> "DesktopNotificationsPatch":
        if self.reconcile_ids is not None and not (
            self.selection_token is not None or self.operation_token is not None or self.receipt
        ):
            raise ValueError("reconcile_ids requires a receipt-backed target")
        if self.reconcile_ids is not None and self.op == "preview":
            # The body recount returns before reconciliation is read, so the ids
            # would be silently dropped; a recount mints no receipt either.
            raise ValueError("reconcile_ids requires a receipt-backed target")
        if self.excluded_ids is not None and self.selection_token is None:
            raise ValueError("excluded_ids requires selection_token")
        if self.selection_token is not None or self.operation_token is not None:
            if any(
                (
                    self.ids is not None,
                    self.session_id is not None,
                    self.all,
                    self.observed is not None,
                    self.through_seq is not None,
                    self.receipt,
                )
            ):
                raise ValueError("receipt targets cannot be combined with legacy targets")
            if self.op == "undo":
                if self.operation_token is None or self.selection_token is not None:
                    raise ValueError("undo requires only operation_token")
            elif self.selection_token is None or self.operation_token is not None:
                raise ValueError("a mutation requires only selection_token")
            return self
        if self.op == "undo":
            raise ValueError("undo requires operation_token")
        if self.op == "preview":
            raise ValueError("preview requires selection_token")
        if self.receipt and (self.ids is None or self.through_seq is None):
            raise ValueError("receipt requires ids and through_seq")
        if self.receipt and self.through_seq is not None and self.through_seq > _MAX_SEQ:
            raise ValueError("through_seq is out of range")
        self._legacy_target()
        return self


@router.patch("/v1/desktop/notifications", response_model=CRUDResponse[dict[str, Any]])
async def patch_notifications(body: DesktopNotificationsPatch, request: Request):
    """Read / unread / dismiss / undismiss, token-bound; open asks are never dismissed."""
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        if body.selection_token is not None or body.operation_token is not None or body.receipt:
            store = NotificationStore(root)
            try:
                if body.op == "preview":
                    # The validator guarantees a selection_token and no other
                    # target. Same answer shape as the GET recount.
                    selection = await asyncio.to_thread(
                        bulk.refresh,
                        store,
                        str(body.selection_token),
                        scope=_bulk_scope(request),
                        excluded_ids=body.excluded_ids or (),
                    )
                    counts = await asyncio.to_thread(inbox.counts, root)
                    return reply({"selection": selection, **counts})
                if body.operation_token is not None:
                    result = await asyncio.to_thread(
                        bulk.undo, store, body.operation_token, scope=_bulk_scope(request)
                    )
                else:
                    result = await asyncio.to_thread(
                        bulk.apply,
                        store,
                        body.op,
                        scope=_bulk_scope(request),
                        selection_token=body.selection_token,
                        ids=[entry.model_dump() for entry in body.ids] if body.ids else None,
                        through_seq=body.through_seq,
                        excluded_ids=body.excluded_ids or (),
                    )
            except bulk.SelectionError as error:
                raise HTTPException(error.status, detail={"code": error.code}) from error
            answer = {**result, **await asyncio.to_thread(inbox.counts, root)}
            if body.reconcile_ids is not None:
                # Read only AFTER receipt auth/expiry/replay fencing. These ids
                # never enter the mutation target or its idempotency identity.
                answer["reconciliation"] = await asyncio.to_thread(
                    reconcile.read_rows, store, body.reconcile_ids
                )
            return reply(answer)
        answer = await asyncio.to_thread(
            inbox.apply_patch,
            root,
            body.op,
            ids=[entry.model_dump() for entry in body.ids] if body.ids is not None else None,
            session_id=body.session_id,
            through_seq=body.through_seq,
            all_items=body.all,
            observed=(
                [entry.model_dump() for entry in body.observed]
                if body.observed is not None
                else None
            ),
        )
        return reply(answer)


class SeenItem(Input):
    id: Annotated[str, Field(pattern=r"^n_[a-z2-7]{16}$")]
    revision: int = Field(ge=1)
    anchor_id: str = Field(min_length=1, max_length=200)
    #: When THIS item's anchor was first visible on the surface (D13). Optional
    #: because the batch field below is the older contract; when a client sends
    #: it, the server checks it against this row's own ``updated_at`` instead of
    #: letting one item's visibility vouch for another's.
    visible_since_ms: int | None = Field(None, ge=0)


class SeenEvidence(Input):
    """What the surface measured (§6.4). The server re-checks each claim it can."""

    surface: Literal["desktop", "tui"]
    session_id: SessionID
    #: A trusted input EDGE, never a focus gain (D12). The app raises its own
    #: window and alt-tab glances past it, so a focus report reads items nobody
    #: engaged with; ``focus`` is refused as a 422 rather than ignored, so a
    #: client cannot believe it reported something.
    input_kind: Literal["pointerdown", "keydown", "wheel"]
    input_at_ms: int = Field(ge=0)
    #: Fallback visibility for clients that measured one value over the batch.
    visible_since_ms: int = Field(ge=0)


class NotificationsSeen(Input):
    items: list[SeenItem] = Field(min_length=1, max_length=100)
    evidence: SeenEvidence


class NativeSeenEvidence(Input):
    """The PHONE's evidence (D11): its own edges, the desktop's re-checks.

    ``surface`` is pinned to ``native`` and the input kinds to the gestures a
    phone actually has. It is a separate model rather than a widened literal on
    ``SeenEvidence`` so a desktop client cannot label its report "native" and a
    relay client cannot label its own "desktop": each transport's door accepts
    exactly the surface it is (the relay answers any other with
    ``relay_surface_mismatch``).
    """

    surface: Literal["native"]
    session_id: SessionID
    input_kind: Literal["touch", "press", "scroll"]
    input_at_ms: int = Field(ge=0)
    visible_since_ms: int = Field(ge=0)


class NativeNotificationsSeen(Input):
    items: list[SeenItem] = Field(min_length=1, max_length=100)
    evidence: NativeSeenEvidence


@router.post("/v1/desktop/notifications/seen", response_model=CRUDResponse[dict[str, Any]])
async def seen_notifications(body: NotificationsSeen, request: Request):
    """In-view auto-read: per item, its own revision and anchor, evidence re-checked.

    The UI's foreground guard admits or refuses this before it is sent
    (``notifications.seen`` in ``FOREGROUND_RECEIPT_OPS``); the store then
    re-checks that each item belongs to ``evidence.session_id``, names its
    current revision and anchor, and that the input edge is fresh and later
    than the item's last change. Anything else is ``skipped`` with its reason.
    """
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        answer = await asyncio.to_thread(
            inbox.apply_seen,
            root,
            [entry.model_dump() for entry in body.items],
            body.evidence.model_dump(),
        )
        return reply(answer)
