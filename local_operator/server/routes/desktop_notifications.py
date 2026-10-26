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
import json
import logging
import pathlib
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import Field, model_validator

from local_operator.i18n.keys import wire_errors_notifications as _wn
from local_operator.i18n.messages import render
from local_operator.notifications import items as inbox
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


@router.get("/v1/desktop/notifications", response_model=CRUDResponse[dict[str, Any]])
async def list_notifications(
    request: Request,
    state: Literal["unread", "all", "dismissed"] = "unread",
    source: Literal["notify", "ask"] | None = None,
    priority: Literal["high", "normal", "low"] | None = None,
    session_id: Annotated[str | None, Query(pattern=r"^[a-f0-9]{12}$")] = None,
    cursor: Annotated[str | None, Query(max_length=80)] = None,
    through_seq: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=inbox.LIST_LIMIT_MAX)] = inbox.LIST_LIMIT_DEFAULT,
):
    """One page of items, newest content first; ``waiting`` on the first page only."""
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        window = _lapsed_window(root)
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


def _anchor_excerpt(session_dir: pathlib.Path, anchor_id: str) -> str:
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
    if not _valid_item_id(item_id):
        raise HTTPException(422, _wn.unknown_item().envelope())
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
        window = _lapsed_window(root)
        answer = await asyncio.to_thread(
            inbox.detail,
            root,
            item_id,
            lapsed_visible_ms=window,
            anchor_excerpt=_anchor_excerpt,
        )
    if answer is None:
        raise HTTPException(404, _wn.unknown_item().envelope())
    return reply(answer)


def _valid_item_id(value: str) -> bool:
    """The PATCH models' own shape rule, so both transports refuse the same ids."""
    return _ITEM_ID_RE.match(value) is not None


class ItemToken(Input):
    """One item the caller RENDERED, named by id and the revision it showed."""

    id: ItemID
    revision: int = Field(ge=1)


class NotificationsPatch(Input):
    """``PATCH /v1/desktop/notifications``. Exactly one target (§5.2)."""

    op: Literal["read", "unread", "dismiss", "undismiss"]
    #: 1..500, the catalogue's own page bound (``attention/seen``'s reasoning):
    #: one user gesture never has to be chunked.
    ids: list[ItemToken] | None = Field(default=None, min_length=1, max_length=500)
    session_id: SessionID | None = None
    through_seq: int | None = Field(default=None, ge=0)
    all: bool = False
    observed: list[ItemToken] | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _one_target(self) -> "NotificationsPatch":
        targets = sum((self.ids is not None, self.session_id is not None, bool(self.all)))
        if targets != 1:
            raise ValueError("name exactly one target: ids, session_id or all")
        if self.ids is None and (self.through_seq is None or self.observed is None):
            raise ValueError("bulk writes require through_seq and observed tokens")
        if self.ids is not None and self.observed is not None:
            raise ValueError("observed tokens belong only to bulk targets")
        return self


@router.patch("/v1/desktop/notifications", response_model=CRUDResponse[dict[str, Any]])
async def patch_notifications(body: NotificationsPatch, request: Request):
    """Read / unread / dismiss / undismiss, token-bound; open asks are never dismissed."""
    async with errors(request, notifications_refusal):
        root = _config_dir(request)
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
