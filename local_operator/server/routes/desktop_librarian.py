"""``/v1/desktop/librarian`` — the Librarian's control plane (Library §5).

THE CONTRACT. A GET that never creates and a POST carrying one op:
``open`` (ensure her session), ``pause``, ``resume``, ``run`` (arm a pass now)
and ``status``. The desktop Library header reads the GET to render status and
"Organize now"; the phone needs no route of its own — the pinned conversation
is the surface.

WHY THE GET ANSWERS WHEN DISABLED. A GET is how a renderer DISCOVERS the
switch, so it must answer rather than refuse: the client hides the row when
``enabled`` is false. A POST on a disabled install is a 409 with
``librarian_disabled``, because the op genuinely did not run.

WHY A HELD STORE LOCK IS A 409 WITH A CODE. ``pause``/``resume`` take the
Librarian store lock as their first act, so a peer that holds it for the whole
wait REFUSES the op with ``WakeLockBusy`` (the lock module's documented answer
for that case, retryable, normal on a seam with several attended writers). A
500 would read as a defect and a 200 would paint success for an action that
never ran, so it answers 409 plus a declared ``code``
(``librarian_store_busy``) and the same sentence the receipts render for the
identical refusal.

WHY ``run`` DOES NOT GO THROUGH THE TRAY BUDGET. It is an explicit operator
gesture ("Organize now"), not the agent asking for an extra pass; refusing it
with the daily budget would make the button lie. It still respects pause and
disable, and it is bounded by construction: one row, due now.

LIBRARY STATS are read best-effort. A rejected status call would hide the very
panel the operator opened; an unreadable catalog answers zeros with ``state``
saying so, and the GET stays a 200.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Librarian"], dependencies=[Depends(require_desktop)])

#: A ``Literal`` so an unknown op is a 422 from FastAPI's own validation rather
#: than a sentence invented here, and the generated schema publishes the
#: vocabulary to the renderer.
LibrarianOpName = Literal["open", "pause", "resume", "run", "status"]


class LibrarianOp(BaseModel):
    """The POST body. ``extra="forbid"`` so a misspelled field is a 422."""

    model_config = {"extra": "forbid"}

    op: LibrarianOpName


class LibraryStats(BaseModel):
    """The GET's library half: how much is there, and how much is untouched."""

    total: int = 0
    unorganized: int = 0
    #: The catalog's own state word (``ready`` / ``busy`` / an error shape). A
    #: renderer that only sees zeros still needs to tell "empty" from "unread".
    state: str = "ready"


class RunReceipt(BaseModel):
    """What the last accepted daily pass did, for the Library header.

    ``at`` is epoch ms (the unit every other instant on this route uses).
    ``named`` and ``tagged`` are DISTINCT ASSETS the metadata store's edit log
    attributes to ``librarian`` since that instant (a description counts as
    naming: both make the row readable). ``failed`` is carried for the
    contract but always 0 from this backend: the store records successes, and
    a failure lives in the transcript, not the catalog — a number invented
    here would be a lie a client paints.
    """

    at: int
    named: int = 0
    tagged: int = 0
    failed: int = 0


class LibrarianState(BaseModel):
    """GET's answer, and the shape every successful POST returns.

    ``LAST_RUN`` IS NESTED AND ALSO FLAT (frozen by the desktop-alignment
    note, 2026-10-18): the header binds ``last_run_at`` and ``unorganized``
    directly, while the richer receipt needs the counts. One shape is frozen:
    the nested object is authoritative, the two flat fields are additive
    mirrors of its ``at`` (as a float) and ``library.unorganized``.
    """

    enabled: bool
    session_id: str | None = None
    paused: bool
    name: str
    #: The last accepted daily pass, or null when she has not run yet.
    last_run: RunReceipt | None = None
    #: Epoch ms as a float; ``None`` until the first pass. Additive mirror of
    #: ``last_run.at`` for a header that binds the flat field.
    last_run_at: float | None = None
    library: LibraryStats = LibraryStats()
    #: Additive mirror of ``library.unorganized``.
    unorganized: int = 0
    #: Whether a LIVE session owns her rows, so this call's effect is carried
    #: out by that owner. ``run``/``resume`` answer it on a 200: a "now" pass
    #: that will arrive in the other window is not an error.
    held: bool = False


def _config_dir(request: Request):
    return request.app.state.config_manager.config_dir


def _refuse_disabled(op: str) -> HTTPException:
    return HTTPException(
        409,
        _wd.librarian_disabled(op=repr(op)).envelope(code="librarian_disabled"),
    )


def _busy_refusal(verb: str, name: str, exc: BaseException) -> HTTPException:
    from local_operator.librarian import state as librarian_state

    librarian_state.note_lock_refusal(f"the {verb} command", exc)
    return HTTPException(
        409,
        _wd.desktop_refusal(error=librarian_state.op_refusal_sentence(verb, name, exc)).envelope(
            code="librarian_store_busy"
        ),
    )


def _stats(root: Any) -> LibraryStats:
    """``{total, unorganized, state}`` — best-effort, never raises.

    ONE catalog scan: :func:`service.list_assets` materialises every row before
    filtering, so the unorganized count is derived from the same pass that
    produces ``total``.
    """
    from local_operator.library import service

    try:
        # TWO COUNTS, EACH THE FILTERED TOTAL (the page is one row): a client
        # reading `rows` would have to page the whole catalog to count the
        # untouched assets, and a limit-truncated page would silently report
        # the truncated number as the status.
        total = service.list_assets(root, service.Query(limit=1))
        unorganized = service.list_assets(root, service.Query(limit=1, organized=False))
    except Exception:  # noqa: BLE001 — a status panel must not fail its route
        logger.warning("librarian: could not read library stats", exc_info=True)
        return LibraryStats(state="unread")
    return LibraryStats(
        total=int(total.get("total") or 0),
        unorganized=int(unorganized.get("total") or 0),
        state=str(total.get("state") or "ready"),
    )


def _receipt(root: Any, last_run_at: int | None) -> RunReceipt | None:
    """The last pass's counts, derived from the store's edit log. Never raises.

    An unreadable store answers the receipt with zero counts rather than
    dropping the instant: "she ran at T" is the fact the header needs most,
    and a transient read failure must not make a pass look like it never
    happened.
    """
    if last_run_at is None:
        return None
    named = tagged = 0
    try:
        from local_operator.library.metadata import MetadataStore

        rows = MetadataStore(Path(root)).edits(limit=500)
        since = last_run_at / 1000.0
        name_ids: set[str] = set()
        tag_ids: set[str] = set()
        for row in rows:
            if row.get("actor") != "librarian" or float(row.get("at") or 0) < since:
                continue
            try:
                fields = json.loads(row.get("fields") or "[]")
            except (TypeError, ValueError):
                continue
            asset = str(row.get("asset_id") or "")
            if "name" in fields or "description" in fields:
                name_ids.add(asset)
            if "tags" in fields:
                tag_ids.add(asset)
        named, tagged = len(name_ids), len(tag_ids)
    except Exception:  # noqa: BLE001 — a receipt must not fail the read
        logger.warning("librarian: could not derive the run receipt", exc_info=True)
    return RunReceipt(at=last_run_at, named=named, tagged=tagged)


def _state(root: Any) -> LibrarianState:
    from local_operator.librarian import bootstrap
    from local_operator.librarian import proactive as engine

    status = engine.status(root)
    last_run_at = status.get("last_run_at")
    receipt = _receipt(root, last_run_at if type(last_run_at) is int else None)
    library = _stats(root)
    return LibrarianState(
        enabled=bootstrap.config_enabled(root),
        session_id=status.get("session_id"),
        paused=bool(status.get("paused")),
        name=bootstrap.display_name(root),
        last_run=receipt,
        last_run_at=float(receipt.at) if receipt is not None else None,
        library=library,
        unorganized=library.unorganized,
    )


@router.get("/v1/desktop/librarian", response_model=CRUDResponse[LibrarianState])
async def get_librarian(request: Request) -> CRUDResponse[LibrarianState]:
    """Her state. Never creates: a caller that wants her to exist POSTs ``open``."""
    root = _config_dir(request)
    state = _state(root)
    return CRUDResponse(status=200, message=f"{state.name} state retrieved.", result=state)


@router.post("/v1/desktop/librarian", response_model=CRUDResponse[LibrarianState])
async def post_librarian(body: LibrarianOp, request: Request) -> CRUDResponse[LibrarianState]:
    """Run one op. Refuses with 409 ``librarian_disabled`` on a disabled install."""
    from local_operator.librarian import bootstrap
    from local_operator.librarian import proactive as engine
    from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

    root = _config_dir(request)
    name = bootstrap.display_name(root)
    if not bootstrap.config_enabled(root):
        raise _refuse_disabled(body.op)

    if body.op == "status":
        return CRUDResponse(status=200, message=f"{name} state retrieved.", result=_state(root))

    if body.op == "open":
        session_id = await bootstrap.ensure_session(root)
        if session_id is None:
            raise HTTPException(
                500,
                _wd.librarian_bootstrap_failed().envelope(code="librarian_bootstrap_failed"),
            )
        state = _state(root)
        return CRUDResponse(status=200, message=f"{name} is ready.", result=state)

    if body.op == "run":
        # PAUSE IS A HARD GATE here: an explicit run while paused would be the
        # operator contradicting their own switch, and the receipt says which
        # one won. Disable was refused above.
        if engine.policy(root).paused:
            return CRUDResponse(
                status=200,
                message=f"{name} is paused; resume before running a pass.",
                result=_state(root),
            )
        session_id = await bootstrap.ensure_session(root)
        if session_id is None:
            raise HTTPException(
                500,
                _wd.librarian_bootstrap_failed().envelope(code="librarian_bootstrap_failed"),
            )
        outcome = await engine.run_now(root, session_id)
        state = _state(root)
        if outcome == "owner":
            # A LIVE RUNTIME HOLDS THE SESSION, so the external writer refused
            # the row. Reporting 200 here was the defect the review measured:
            # nothing was scheduled and nothing would run until an unrelated
            # persist happened to reconcile. There is no durable handoff to
            # promise — ``arm.py`` refuses precisely because the owner's stale
            # list would delete the row — so the honest answer is a typed 409.
            raise HTTPException(
                409,
                _wd.librarian_owner_active().envelope(code="librarian_owner_active"),
            )
        if outcome in ("failed", "busy"):
            raise HTTPException(
                409,
                _wd.desktop_refusal(error=engine.STATUS_SENTENCES[outcome]).envelope(
                    code="librarian_store_busy"
                ),
            )
        return CRUDResponse(status=200, message=f"{name} is running a pass now.", result=state)

    try:
        if body.op == "pause":
            current = _state(root)
            outcome = await engine.pause(root, current.session_id or "")
            state = _state(root)
            state.held = outcome.owner_blocked
            return CRUDResponse(status=200, message=f"{name} is paused.", result=state)
        current = _state(root)
        outcome = await engine.resume(root, current.session_id or "")
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        raise _busy_refusal(body.op, name, exc) from exc

    state = _state(root)
    if outcome == "owner":
        state.held = True
    return CRUDResponse(status=200, message=f"{name} is active again.", result=state)
