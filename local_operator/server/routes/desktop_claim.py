"""``POST /v1/desktop/claim`` — the way into a daemon somebody else started.

A daemon the desktop app spawned carries ``LOCAL_OPERATOR_DESKTOP_TOKEN`` in its
environment and needs this route for nothing. A daemon a TUI, a terminal or
launchd started has no such capability, and before this route existed the whole
desktop plane answered ``503`` to the app — the app could find the daemon (see
``server/registry.py``) and still not talk to it. This is the other half of the
handshake: the app reads ``claim_key`` out of that daemon's ``0600`` record and
presents it once as a bearer.

**Deliberately NOT behind** :func:`~local_operator.server.desktop.require_desktop`.
Every other desktop route depends on it; this one cannot, because it is the door
that dependency stands in front of — a claim route gated on an unclaimed plane
is a deadlock, and the manager's design calls that out by name. It is also
deliberately NOT under ``managed_desktop_boundary``'s legacy-control gate: the
route's path is ``/v1/desktop/``-prefixed, which the boundary already treats as
sensitive (it earns the ``no-store`` header below), while the legacy-control
check matches only ``/v1/agents``, ``/v1/jobs``, ``/v1/schedules`` and the flat
singleton control paths — never this. Both facts are asserted by
``tests/unit/server/test_desktop_claim.py``: if a future change routes the claim
path through an unclaimed-plane gate, that test fails with the deadlock instead
of an app that cannot attach.

What the route exposes is the least it can: a boolean and the daemon's
``instance_id``. It never echoes the key, never says what the expected key looks
like, and the refusals (see ``desktop.accept_claim``) carry a status and a
reason sentence only.
"""

from typing import Any

from fastapi import APIRouter, Request

from local_operator.server.desktop import accept_claim
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Desktop claim"])


@router.post("/v1/desktop/claim", response_model=CRUDResponse[dict[str, Any]])
async def claim_desktop(request: Request) -> CRUDResponse[dict[str, Any]]:
    """Claim this daemon's desktop plane for the caller holding the record's key.

    The key is read from ``app.state.serve_record``, which the ``lifespan``
    published (and keeps republishing, heartbeat by heartbeat) — the record IS
    the key's publication channel, so a process that booted without announcing
    an address has none and the claim is refused with ``503`` rather than
    inventing one. Reading it from state rather than the filesystem keeps the
    route honest about whose key it compares: the one the record the daemon
    actually published carries.
    """
    record = getattr(request.app.state, "serve_record", None)
    accept_claim(request, published_key=getattr(record, "claim_key", "") or "")
    return CRUDResponse(
        status=200,
        message="Desktop plane claimed.",
        result={
            "claimed": True,
            "instance_id": getattr(request.app.state, "instance_id", ""),
        },
    )
