"""Mesh pairing cards in the desktop's Mesh approvals (``features.mesh_pairing``).

``GET /v1/desktop/mesh/pairings`` and ``POST /v1/desktop/mesh/pairings/{side}/{invite_id}
/answer`` — the operator's request was that pairing two devices must not need terminal
commands and should happen through the app's Mesh approvals; these are the backend
half (the renderer card is the UI PR). Logic lives in
``server.utils.desktop_mesh_pairings``; this module is the HTTP skin.

WHY A NEW ROUTER. ``desktop_approvals.router``'s route set is a pinned contract
(``test_frozen_paths_remain_and_onboarding_is_additive``), so — like
``desktop_credentials`` — new paths mount on their own router beside it, under the
same ``require_desktop`` guard (401 without the bearer, 503 while the plane is closed).

THE BEARER IS NOT ENOUGH TO ANSWER. Every answer must be covered by an app attestation
(``network.pairing_attest``) consumed BEFORE any comparison; without one the answer is
403 ``pairing_needs_app`` and nothing was spent or written. That is desktop-API
protection only: the same-user terminal fallback (``lop network confirm`` / ``join
--confirm``) still exists and the relay cannot tell the two apart.

STATUS MAP: 404 ``no_pending_pairing``; 403 ``pairing_needs_app``; 400 ``bad_request``
(an inviter admit with no ``shares`` list, or shares on any other answer; nothing
spent); 409
``sas_mismatch`` (body carries ``attempts_left``; the card stays open),
``pairing_expired``, ``pairing_answered`` (a second answer; outcome-neutral copy),
``pairing_not_ready``, and any other relay refusal; 503 ``relay_unavailable``.
Messages are keyed copy (``network.pairing``); the renderer keys on ``code``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from local_operator.network.types import MeshRefusal
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors, reply

router = APIRouter(tags=["Desktop mesh pairing"], dependencies=[Depends(require_desktop)])


class PairingAnswer(BaseModel):
    """The card's answer: exactly what the app attested (``pairing_attest.content``)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["admit", "decline"]
    code: str = Field(default="", max_length=16)
    #: The inviter's reduce-only share choice, REQUIRED on an inviter admit (``[]``
    #: shares nothing) and absent or ``[]`` on every other answer — never "absent =
    #: the offered set" (S-R1-1). Optional at the model so a missing list is refused
    #: as a keyed ``bad_request`` before any attestation is spent, not as a 422.
    shares: list[str] | None = Field(default=None, max_length=256)
    #: The card row's ``nonce`` — which ceremony (and stage) this answers.
    nonce: str = Field(min_length=1, max_length=128)
    #: The attested deadline, wall-clock seconds; must not pass the ceremony's expiry.
    deadline: int


_STATUS = {
    "no_pending_pairing": 404,
    "pairing_needs_app": 403,
    "bad_request": 400,
    "relay_unavailable": 503,
}


def _copy(code: str, fallback: str) -> str:
    """The keyed sentence for a refusal code (``network.pairing``), else the relay's own."""
    from local_operator.network import pairing_copy

    return pairing_copy.refusal(code) or fallback


def _http(error: MeshRefusal) -> HTTPException:
    body: dict[str, Any] = {"code": error.code, "message": _copy(error.code, str(error))}
    left = getattr(error, "attempts_left", None)
    if left is not None:
        body["attempts_left"] = int(left)
        body["message"] = _copy_mismatch(int(left))
    return HTTPException(_STATUS.get(error.code, 409), body)


def _copy_mismatch(left: int) -> str:
    from local_operator.network import pairing_copy

    return pairing_copy.mismatch(left)


def _root(request: Request) -> Any:
    return request.app.state.config_manager.config_dir


def _service(request: Request) -> Any:
    """``lop serve``'s mesh-pairing attestation listener, or ``None`` (non-macOS/failed)."""
    return getattr(request.app.state, "mesh_pairing_attest", None)


async def _run(request: Request, work: Any) -> Any:
    async with errors(request):
        try:
            return await asyncio.to_thread(work)
        except MeshRefusal as error:
            raise _http(error) from None


@router.get("/v1/desktop/mesh/pairings", response_model=CRUDResponse[Any])
async def mesh_pairings(request: Request):
    """Open pairing cards on THIS device, both sides, oldest first (read-only)."""
    from local_operator.server.utils import desktop_mesh_pairings as util

    root, service = _root(request), _service(request)
    return reply(await _run(request, lambda: util.pairing_rows(root, service)))


@router.post(
    "/v1/desktop/mesh/pairings/{side}/{invite_id}/answer", response_model=CRUDResponse[Any]
)
async def answer_mesh_pairing(request: Request, side: str, invite_id: str, body: PairingAnswer):
    """Answer one card: ``{status, attempts_left}``; see the module's status map."""
    from local_operator.server.utils import desktop_mesh_pairings as util

    root, service = _root(request), _service(request)

    def work() -> dict[str, Any]:
        return util.answer(
            root,
            service,
            side=side,
            invite_id=invite_id,
            decision=body.decision,
            code=body.code,
            shares=body.shares,
            nonce=body.nonce,
            deadline=body.deadline,
        )

    return reply(await _run(request, work))
