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

**A page cannot use this door.** A claim carrying ``Sec-Fetch-Site`` is refused
even with the correct key (see ``desktop.accept_claim``): that header is
attached by the browser's own fetch stack, page script can neither forge nor
remove it, and the intended caller — the app's main process — sends none.

What the route exposes is the least it can: a boolean and the daemon's
``instance_id``. It never echoes the key, never says what the expected key looks
like, and the refusals (see ``desktop.accept_claim``) carry a status and a
reason sentence only.

Its body is optional and carries at most one field, ``origins``: the renderer
origins the caller wants admitted for itself (a packaged app's renderer loads
from ``file://``, whose origin is the literal ``"null"`` the plane refuses, so
without a declaration there would be no way to admit a dev-server origin). Each
entry is validated by ``desktop.parse_claim_origins``.
"""

import json
import logging
from typing import Any, Sequence

from fastapi import APIRouter, HTTPException, Request

from local_operator.server.desktop import accept_claim
from local_operator.server.models.schemas import CRUDResponse

logger = logging.getLogger("local_operator.server.routes.desktop_claim")

router = APIRouter(tags=["Desktop claim"])


async def _declared_origins(request: Request) -> Sequence[object]:
    """The ``origins`` the claim body declares, or a ``400`` refusal.

    Absent body, empty body and a body without the field are all the same
    thing — nothing declared — because the field is an addition for callers that
    need it, not a requirement on the handshake. Only the SHAPE is checked here;
    what may actually be installed is ``desktop.parse_claim_origins``'s call, so
    the route cannot grow a second opinion about an origin's validity.

    A malformed body is refused rather than ignored: silently treating
    ``{"origins": "http://localhost:5173"}`` (a string where a list belongs) as
    "nothing declared" would leave a caller believing its renderer was admitted
    when it is walled off by CORS.
    """
    raw = await request.body()
    if not raw:
        return ()
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "The claim body must be a JSON object.") from None
    if body is None:  # a literal JSON `null` is an absent body
        return ()
    if not isinstance(body, dict):
        raise HTTPException(400, "The claim body must be a JSON object.")
    declared = body.get("origins", [])
    if declared is None:
        return ()
    if not isinstance(declared, list):
        raise HTTPException(400, "The claim body's 'origins' must be a list.")
    return declared


def _refresh_record(request: Request) -> None:
    """Make the published record say the plane is governed, without re-minting.

    The record is the only on-disk statement of who owns this plane, and
    ``build_record`` computed ``desktop`` before any claim could exist — so
    without this the record keeps telling a reader ``"desktop": false`` while
    handing out the ``claim_key`` that governs it, and the next consumer
    (the UI's own daemon-discovery PR) inherits that lie.

    Refresh through the PUBLISHER that owns this record — ``heartbeat(desktop=
    True)`` mutates that one object and rewrites it, so the key is preserved
    byte for byte. Rebuilding via ``build_record`` would re-mint ``claim_key``
    and turn the app's legitimate re-attach into a ``409``.

    Failures are swallowed after a warning: the latch is already set and the
    claim IS accepted, and the in-memory record is already mutated, so the
    heartbeat loop's next tick rewrites the file. Raising here would tell the
    caller its claim failed when it did not — and its retry would get a ``409``.
    """
    publisher = getattr(request.app.state, "serve_publisher", None)
    if publisher is None:
        # No published record at all means no claim was possible (no key), so
        # this is unreachable on a successful claim; a test app that fakes
        # ``serve_record`` lands here.
        return
    try:
        publisher.heartbeat(desktop=True)
    except Exception:  # noqa: BLE001 - a failed rewrite must not undo the claim
        logger.warning(
            "desktop claim accepted but the serve record could not be refreshed; "
            "the next heartbeat will rewrite it",
            exc_info=True,
        )


@router.post("/v1/desktop/claim", response_model=CRUDResponse[dict[str, Any]])
async def claim_desktop(request: Request) -> CRUDResponse[dict[str, Any]]:
    """Claim this daemon's desktop plane for the caller holding the record's key.

    The key is read from ``app.state.serve_record``, which the ``lifespan``
    published and keeps rewriting on its heartbeat timer — the record IS the
    key's publication channel, so a process that booted without announcing an
    address has none and the claim is refused with ``503`` rather than
    inventing one. Reading it from state rather than the filesystem keeps the
    route honest about whose key it compares: the one the record the daemon
    actually published carries. (The heartbeat rewrites that same record
    object; it does not re-derive its fields.)

    Acceptance is logged at WARNING because the daemon's console logging
    defaults to that level (``LOG_LEVEL``, see ``local_operator/logger.py``),
    and this line is the only diagnostic for the named rollout risk: an
    operator's local ``curl`` script starts seeing ``401`` the moment an app
    claims the plane, and the log says who claimed it and which origin it
    installed. Refusals log at INFO — they are routine (a prober, a wrong key)
    and the caller already gets the status. Neither line ever carries a key.
    """
    record = getattr(request.app.state, "serve_record", None)
    declared = await _declared_origins(request)
    instance_id = getattr(request.app.state, "instance_id", "")
    origin = request.headers.get("origin") or "<none>"
    try:
        accept_claim(
            request,
            published_key=getattr(record, "claim_key", "") or "",
            declared_origins=declared,
        )
    except HTTPException as refusal:
        logger.info(
            "desktop claim refused: instance_id=%s origin=%s declared=%d status=%d",
            instance_id,
            origin,
            len(declared),
            refusal.status_code,
        )
        raise
    logger.warning(
        "desktop claim accepted: instance_id=%s origin=%s declared=%s — the legacy "
        "control surface is now bearer-gated for every other local caller",
        instance_id,
        origin,
        ", ".join(str(item) for item in declared) if declared else "none",
    )
    _refresh_record(request)
    return CRUDResponse(
        status=200,
        message="Desktop plane claimed.",
        result={
            "claimed": True,
            "instance_id": instance_id,
        },
    )
