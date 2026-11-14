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
    PairAcknowledgment,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_approvals import (
    _apply_operation,
    _approve_operation,
    _local,
    _root,
    reply,
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


@router.post("/v1/desktop/approvals/{approval_id}/pair-ack", response_model=CRUDResponse[Any])
async def pair_ack(request: Request, approval_id: str, body: PairAcknowledgment):
    """Record the person's pairing decision (typed code, or confirm-mode match). No signer.

    Pairing design v2 §2: the desktop card posts here first, then ``approve-operation``
    with ``operation: "pair_phone"`` signs and applies. Splitting them keeps the
    strike-counted check off the signer's path (a wrong code never spends a Touch ID
    gesture) and gives the decision moment its own durable receipt, which is what the
    decision-scoped expiry (§2.1) is measured from. Refusals map like every other
    approval refusal: 404 unknown, 409 everything decidable (``pair_code_required``,
    ``pair_code_mismatch``, ``approval_expired``, ``pair_request_changed``).
    """
    from local_operator.operator import pair_op

    def work():
        return pair_op.desktop_ack(
            approval_id,
            request_digest=body.request_digest,
            disclosure_digest=body.disclosure_digest,
            code=body.code,
            root=_root(request),
        )

    return reply(await _local(request, work))


@router.get("/v1/desktop/devices", response_model=CRUDResponse[Any])
async def paired_phones(request: Request):
    """Settings → Phone access → Paired phones (pairing design v2 §2, §2.4).

    ``{devices: [{device_id, name, platform, paired_at, last_seen, push, state}]}`` with
    ``state`` one of ``paired | unconfirmed | removing | revoked``, derived from the
    certificate store, the revoked list, the installed anchor and the linked push rows
    -- never a stored flag, so it survives refresh and restart.
    """
    from local_operator.operator import devices_revoke

    return reply(
        await _local(request, lambda: {"devices": devices_revoke.list_phones(_root(request))})
    )


@router.delete("/v1/desktop/devices/{device_id}", response_model=CRUDResponse[Any])
async def remove_phone(request: Request, device_id: str):
    """Remove a paired phone: the unified revoke (``operator.devices_revoke``).

    Idempotent: a second call on a ``removing`` row finishes it (the OS prompt with the
    "remove a phone" purpose). The answer is the row's resulting ``state`` and which
    phases ran; a cancelled prompt is a 200 with ``state: removing``, not an error,
    because phases 1-2 already took effect and the row must say so. ``detail`` is keyed
    copy (``operator.pairing.revoke.unfinished``) while the row is not ``revoked``, else
    empty. 404 ``unknown_device`` for an id this computer never issued or listed; 422
    ``bad_device_id`` for one that is not an ASCII alphanumeric id.
    """
    from local_operator.network.types import MeshRefusal
    from local_operator.operator import devices_revoke

    def work():
        from local_operator.operator.pair_op import _copy

        try:
            # ``require_known`` (round-1 R6/Q3): this button only ever names a listed
            # row, so an id this computer never issued is a 404 BEFORE any revocation
            # record, anchor staging or admin prompt -- never a phantom ``revoked`` row.
            return devices_revoke.revoke(_root(request), device_id, require_known=True)
        except devices_revoke.UnknownDevice as exc:
            raise MeshRefusal("unknown_device", _copy("refusal_bad_device_id")) from exc
        except ValueError as exc:
            raise MeshRefusal("bad_device_id", _copy("refusal_bad_device_id")) from exc

    return reply(await _local(request, work))
