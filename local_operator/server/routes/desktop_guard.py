"""The desktop's Radient Guard routes (``features.guard``): the TUI ``/guard`` screen, over HTTP.

THIN SKINS, ONE ENGINE. Every handler hands a blocking call to
``server.utils.desktop_guard`` -- which calls the same ``guard.surface`` /
``guard.consent`` / ``guard.reports`` functions the TUI does -- and maps the
engine's refusal onto the plane's ``{code, text}`` envelope. The contract
(routes, enums, refusal table) is ``docs/design/guard-desktop-routes.md``; the
wire models are ``server/models/desktop_guard.py``.

THE AUTHORISATION RULE. ``require_desktop`` (the app-wide desktop bearer plus
the Origin allowlist, the same dependency ``desktop_approvals`` uses) gates the
whole router. Nothing here can GRANT: a first grant or a widening is answered
only by the typed ``guard_consent`` card under ``GET /v1/desktop/approvals``
(signed, digest-bound, verified by the engine), and a report is never
confirmed from here. The writes below are the safe directions (stop sharing,
discard a draft, mark a receipt shown) plus ``share-again``, which only lifts
a sticky "no" so the org's prompt can come back -- and is refused wherever the
TUI's own person-only gate refuses.

ADDITIVE, AND THE FEATURE KEY IS THE GATE. Every route is new, so a renderer
that predates them never calls one; ``features.guard`` is how a renderer learns
they exist. It says the ROUTES exist, not that Guard applies to this account:
that is ``GET /v1/desktop/guard``'s ``mode``.

NON-TEAM ACCOUNTS MAKE ZERO ``/v1/guard/*`` CALLS. ``GET /v1/desktop/guard`` is
local facts only; the one read that leaves the machine is
``GET /v1/desktop/guard/non-team-card``'s ``GET /v1/team-trial/eligibility``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Request

from local_operator.guard import surface
from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_guard import (
    GuardNonTeamCard,
    GuardNoticeDue,
    GuardNoticeDueRequest,
    GuardNoticeShownResult,
    GuardReportDiscardResult,
    GuardRevokeResult,
    GuardShareAgainResult,
    GuardStatus,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors, reply
from local_operator.server.utils import desktop_guard as guard

router = APIRouter(tags=["Desktop Guard"], dependencies=[Depends(require_desktop)])

#: An org id from a path. Bounded and shaped like the ids Radient issues, so a path
#: segment cannot be a multi-kilobyte string or carry a separator into a log line.
_TENANT = PathParam(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
_ID = PathParam(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


def _root(request: Request) -> Path:
    """This backend's config root, from the app's config manager (as ``desktop_approvals``).

    The Guard stores live under the config root the daemon was started against, not the
    process default: a relocated root must read THAT machine's Guard records.
    """
    return request.app.state.config_manager.config_dir


def _can_answer(request: Request) -> bool:
    """Whether this daemon was started by a person, as captured at ``lop serve`` start.

    ``app.state.guard_host_can_answer`` is set by the lifespan from the capture taken BEFORE
    ``lop serve`` dropped its agent-shell markers; reading ``os.environ`` here would always
    say yes on a served daemon (security review S-1). An app that never went through the
    lifespan falls back to the same helper, which then reads the live environment.
    """
    value = getattr(request.app.state, "guard_host_can_answer", None)
    return value if isinstance(value, bool) else surface.served_host_can_answer()


def _listed(request: Request) -> set[int]:
    """The receipt ids this daemon has listed (``notices/{id}/shown``'s allow-list)."""
    state = request.app.state
    if not isinstance(getattr(state, "guard_notices_listed", None), set):
        state.guard_notices_listed = set()
    return state.guard_notices_listed


async def _local(request: Request, work: Any) -> Any:
    """Run one blocking Guard call off the event loop and translate its refusals.

    ``revoke`` and ``share-again`` can wait on a network request to Radient; neither may
    run on the loop that is streaming a turn.
    """
    async with errors(request):
        try:
            return await asyncio.to_thread(work)
        except guard.GuardRefusal as refusal:
            raise HTTPException(
                refusal.status,
                _wd.desktop_refusal(error=refusal.message).envelope(code=refusal.code),
            ) from None


@router.get("/v1/desktop/guard", response_model=CRUDResponse[GuardStatus])
async def guard_status(request: Request):
    """The overview and the badge: states, drafts, receipts, the non-Team card. No network."""
    return reply(
        await _local(
            request,
            lambda: guard.status(
                _root(request), can_answer=_can_answer(request), listed=_listed(request)
            ),
        )
    )


@router.get("/v1/desktop/guard/non-team-card", response_model=CRUDResponse[GuardNonTeamCard])
async def guard_non_team_card(request: Request):
    """The §14' card with its trial length (``/v1/team-trial/eligibility``, never ``/v1/guard``)."""
    return reply(await _local(request, lambda: guard.non_team_card(_root(request))))


@router.post(
    "/v1/desktop/guard/orgs/{tenant}/revoke", response_model=CRUDResponse[GuardRevokeResult]
)
async def guard_revoke(request: Request, tenant: str = _TENANT):
    """ "Stop sharing" with one org: the local stop is committed before Radient is told."""
    return reply(
        await _local(
            request, lambda: guard.revoke(_root(request), tenant, can_answer=_can_answer(request))
        )
    )


@router.post(
    "/v1/desktop/guard/orgs/{tenant}/share-again",
    response_model=CRUDResponse[GuardShareAgainResult],
)
async def guard_share_again(request: Request, tenant: str = _TENANT):
    """Lift a sticky "no"; grants nothing (the prompt still needs the signed card)."""
    return reply(
        await _local(
            request,
            lambda: guard.share_again(_root(request), tenant, can_answer=_can_answer(request)),
        )
    )


@router.post(
    "/v1/desktop/guard/reports/{draft_id}/discard",
    response_model=CRUDResponse[GuardReportDiscardResult],
)
async def guard_discard_report(request: Request, draft_id: str = _ID):
    """Discard an agent-drafted report. The safe direction; there is no confirm route."""
    return reply(await _local(request, lambda: guard.discard_report(_root(request), draft_id)))


@router.post(
    "/v1/desktop/guard/notices/{notice_id}/shown",
    response_model=CRUDResponse[GuardNoticeShownResult],
)
async def guard_notice_shown(request: Request, notice_id: int):
    """Mark a grant/widen receipt shown (once, on any surface)."""
    return reply(
        await _local(
            request, lambda: guard.notice_shown(_root(request), notice_id, listed=_listed(request))
        )
    )


@router.post("/v1/desktop/guard/notice-due", response_model=CRUDResponse[GuardNoticeDue])
async def guard_notice_due(request: Request, body: GuardNoticeDueRequest):
    """The once-a-local-day C6 notice cadence for one surface. Asking consumes the day."""
    return reply(await _local(request, lambda: guard.notice_due(_root(request), body.surface)))
