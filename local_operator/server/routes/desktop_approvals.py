"""The desktop's approval routes (``features.approvals``).

THREE ROUTES, each a thin HTTP skin over one function in
``server.utils.desktop_approvals``: the STORE does the deciding (the write-once
decision, the signature gate, the anchor provenance), and this module maps
answers onto the contract the desktop app consumes.

ADDITIVE, AND THE FEATURE KEY IS THE GATE. Every route here is new, so a
pre-approvals renderer never calls one; ``features.approvals`` is what tells a
renderer it may. The store is device-local, so a read here opens no socket and
dials no peer — the mesh relay is not in this path at all (§2.3 of the
remote-onboarding design), and neither is the operator's secret store (the
record holds a reference NAME and this path never resolves it).

THE AUTHORISATION RULE, repeated from the utils module because this is the HTTP
boundary a reader will look for it at: the desktop bearer token gates the whole
router (``require_desktop``, the app-wide dependency), and inside it the only
authority-increasing act — approve — is the SAME presence-gated signing call the
CLI's verb runs. An HTTP request cannot approve anything on its own; the OS
prompt is what decides, and a host that cannot raise one refuses with a sentence
naming the setup action.

THE REFUSAL SHAPE is the plane's ``{code, message}``: 404 for a record this
device does not hold, 409 for everything the caller can act on (a state that
refuses a decision, a missing signing surface) — never 503, because nothing on
this surface waits on a peer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from local_operator.network.types import MeshRefusal
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_approvals import (
    ApprovalDecision,
    ApprovalList,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors, reply
from local_operator.server.utils.desktop_approvals import approval_rows, approve, deny

router = APIRouter(tags=["Desktop approvals"], dependencies=[Depends(require_desktop)])


def _root(request: Request) -> Path:
    """This backend's config root — where the device-local records live.

    Read from the app's config manager rather than the process default, the same
    way every mesh read on this plane is keyed: a backend started against a
    relocated root must read THAT machine's records, not the environment's.
    """
    return request.app.state.config_manager.config_dir


def _http_refusal(error: MeshRefusal) -> Exception:
    """Map an approval refusal onto a status, keeping the sentence verbatim.

    The status split is the plane's "who can act on this": a record this device
    does not hold is 404, and every decidable refusal — running, already
    connected, already decided, no signing surface — is 409, because the
    operator (or their next action) can act on all of them.
    """
    status = 404 if error.code == "unknown_approval" else 409
    return HTTPException(status, {"code": error.code, "message": str(error)})


async def _local(request: Request, work: Any) -> Any:
    """Run one store call off the event loop and translate its refusals.

    The store can BLOCK on a cross-process file lock (the write discipline §2.3
    gives three writers), and ``approve`` waits on a human gesture on top of
    that — neither may run on the loop that is streaming a turn.
    """
    async with errors(request):
        try:
            return await asyncio.to_thread(work)
        except MeshRefusal as error:
            raise _http_refusal(error) from None


@router.get("/v1/desktop/approvals", response_model=CRUDResponse[ApprovalList])
async def approvals(request: Request):
    """The badge read: pending and recent approval records, oldest first."""
    return reply(await _local(request, lambda: {"approvals": approval_rows(_root(request))}))


@router.post(
    "/v1/desktop/approvals/{approval_id}/approve", response_model=CRUDResponse[ApprovalDecision]
)
async def approve_approval(request: Request, approval_id: str):
    """The operator's yes: the SAME signing call the CLI's ``approve`` verb runs."""
    return reply(await _local(request, lambda: approve(_root(request), approval_id)))


@router.post(
    "/v1/desktop/approvals/{approval_id}/deny", response_model=CRUDResponse[ApprovalDecision]
)
async def deny_approval(request: Request, approval_id: str):
    """The operator's no — ordinary, write-once, the safe direction (§2.4)."""
    return reply(await _local(request, lambda: deny(_root(request), approval_id)))
