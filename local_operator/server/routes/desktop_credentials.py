"""The op-generic TYPED approval routes: ``/approve-operation`` and ``/apply-operation``.

WHY THIS EXISTS. Every approval card that cannot be signed by the bodiless generic
``/approve`` -- today the credential ops, next the phone-pairing op (S4a-pair-op) -- is
signed through ONE engine, ``network.credentials.desktop_approval``, which looks the
operation up in the registry (``network.approval_ops``). These two routes are that
engine's op-generic HTTP face; ``/approve-credential`` and ``/apply-credential`` in
``server.routes.desktop_approvals`` are aliases that call the very same handlers, so the
two families cannot answer differently (pinned by the parity tests).

WHY A SEPARATE ROUTER. ``desktop_approvals.router``'s route set is a frozen contract
(``test_frozen_paths_remain_and_onboarding_is_additive`` pins it as an exact set), so the
new paths live on their own router, mounted beside it in ``server.app``. The guard is the
same: the desktop bearer (``require_desktop``) gates every route, 401 without it and 503
while the plane is closed, and refusals map through the approvals module's own
``_http_refusal`` (404 unknown, 409 decidable).

THE BODY. ``{operation, request_digest, disclosure_digest}``, strict and closed. The
server recomputes both digests from the record on disk and refuses a mismatch before the
signer runs, so the signature binds what the card SHOWED. ``can_approve_here`` stays
``False`` on every row: only a renderer that knows ``typed_approval`` posts here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_approvals import (
    OperationApprovalAcknowledgment,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_approvals import (
    _apply_operation,
    _approve_operation,
)

router = APIRouter(tags=["Desktop approvals"], dependencies=[Depends(require_desktop)])


@router.post(
    "/v1/desktop/approvals/{approval_id}/approve-operation", response_model=CRUDResponse[Any]
)
async def approve_operation(
    request: Request, approval_id: str, body: OperationApprovalAcknowledgment
):
    """Sign ONE reviewed typed card (any op in ``network.approval_ops``), then apply it.

    The frozen bodiless ``/approve`` is NOT this route and never signs a typed op: it
    carries no digests, so it cannot prove which disclosure was on screen.
    """
    return await _approve_operation(request, approval_id, body)


@router.post(
    "/v1/desktop/approvals/{approval_id}/apply-operation", response_model=CRUDResponse[Any]
)
async def apply_operation(
    request: Request, approval_id: str, body: OperationApprovalAcknowledgment
):
    """Apply an already-signed typed card (the "signed, not applied" retry)."""
    return await _apply_operation(request, approval_id, body)
