"""The desktop's mesh routes (``features.peers``, ``features.session_transfer``).

FOUR READS AND FOUR WRITES, and each one is a thin HTTP skin over one function in
``server.utils.desktop_mesh``: the relay does the dialling and the deciding, and this
module maps the answer onto the contract the desktop app consumes
(``local-operator-ui``'s ``desktop-session-contract.ts``, frozen as the build plan's
Addendum 2).

ADDITIVE, AND THE FEATURE KEYS ARE THE GATE. Every route here is new, so a pre-mesh
renderer never calls one; ``features.peers``/``features.session_transfer`` are what
tells a renderer it may. Nothing on an existing surface changes shape because these
exist, and on a machine in no network every read answers empty having opened no
socket (``utils.desktop_mesh.has_any_network``).

THE AUTHORISATION RULE (repeated from the utils module because this is the HTTP
boundary a reader will look for it at): the desktop bearer token gates the whole
router (``require_desktop``, the app-wide dependency), and inside it these routes
have exactly the authority an operator has at their own shell — they run the CLI's
own ops through this device's loopback, key-authenticated relay. They do not dial a
peer, do not hold a mesh key, and do not re-implement a policy: the relay's
authoriser and the owner's own guards decide, and their sentences travel out
verbatim. A renderer therefore cannot do anything from HTTP that the operator could
not do from a terminal, and no rule is enforced in JavaScript.

WHAT A ROUTE MAY EXPOSE, one rule per read: the peer catalogue publishes device ids,
labels, reachability and session COUNTS (never a peer's transcript, addresses beyond
the member row this device already holds, or any credential); the networks read
publishes this device's own member table; the session rows publish where a
conversation lives and whether its owner answered this poll. The invite route
publishes a FILE PATH and never the token — the token is a bearer credential and a
JSON answer would put it in a transcript.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from local_operator.network.types import MeshRefusal
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_mesh import (
    InviteNetwork,
    InviteReceipt,
    NetworkTopology,
    PeerList,
    RemovedMember,
    RemoveMember,
    TransferReceipt,
    TransferSession,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors, receipts, reply
from local_operator.server.utils.desktop_mesh import (
    invite,
    network_topology,
    peer_catalogue,
    remove_member,
    resolve_network,
    transfer,
    transfer_receipt,
)
from local_operator.server.utils.desktop_receipts import Unclaimed

router = APIRouter(tags=["Desktop mesh"], dependencies=[Depends(require_desktop)])

#: Refusals that did NOT leave a move behind, so a retry of the same request id must
#: be allowed to run again rather than replay a refusal forever. Anything else (a
#: move that reached a phase it cannot roll back from, or an answer this device never
#: received) is RECORDED: the renderer is told the outcome is unconfirmed and must
#: re-read the row rather than retry into a second move.
_MOVE_UNCONFIRMED_CODES = frozenset({"relay_unavailable", "deadline_exceeded", "peer_unreachable"})


def _refusal(error: MeshRefusal) -> dict[str, Any]:
    """A mesh refusal as the desktop plane's ``{code, message}`` body.

    The CODE is the machine contract and the SENTENCE is for the person, which is the
    shape every refusal on this plane already carries: the renderer keys on the code
    when it can and renders the message the machine composed instead of inventing one
    from a status.
    """
    return {"code": error.code, "message": str(error)}


async def _mesh(request: Request, work: Any) -> Any:
    """Run one blocking mesh call through the route ladder, off the event loop.

    EVERY call here can block on a loopback socket read, so they all go to a worker
    thread: the desktop's event loop is the one serving a streaming turn, and a
    listing that fans out to peers must not park it.

    ``has_any_network`` is checked inside every read for the same reason: a machine in
    no network answers without opening a socket — the zero-peer property, enforced at
    the boundary rather than trusted to each caller.
    """
    async with errors(request):
        try:
            return await asyncio.to_thread(work)
        except MeshRefusal as error:
            raise _http_refusal(error) from None


def _http_refusal(error: MeshRefusal) -> Exception:
    """Map a mesh refusal's code onto a status, keeping the sentence verbatim.

    A REFUSAL THE CALLER CAN ACT ON IS 409 and a machine that could not be asked is
    503, which is the split this plane already uses: "the peer said no" and "nothing
    answered" are different facts, and the renderer's notice says the sentence either
    way. An unknown network is 404 — the resource named does not exist here — and an
    ambiguous or malformed one is 422.
    """
    from fastapi import HTTPException

    status = {
        "unknown_network": 404,
        "ambiguous_network": 422,
        "bad_role": 422,
        "bad_ttl": 422,
    }.get(error.code, 503 if error.code == "relay_unavailable" else 409)
    return HTTPException(status, _refusal(error))


@router.get("/v1/desktop/peers", response_model=CRUDResponse[PeerList])
async def peers(request: Request):
    """The peer catalogue: one row per OTHER device, deduped across networks."""
    return reply(await _mesh(request, lambda: peer_catalogue(_root(request))))


@router.get("/v1/desktop/networks", response_model=CRUDResponse[NetworkTopology])
async def networks(request: Request):
    """Networks and their members, per network — a device in two is in both."""
    return reply(await _mesh(request, lambda: network_topology(_root(request))))


@router.post("/v1/desktop/networks/{network}/invite", response_model=CRUDResponse[InviteReceipt])
async def mint_invite(network: str, request: Request, body: InviteNetwork):
    """Mint an INVITE and answer with the path its token was written to.

    An invitation, never an "add": admission is two-sided and the joining device is
    the one that proves the code, so there is no unilateral membership write to offer
    and this route does not pretend otherwise (``mesh-ui.md`` §2.8, decision 4).

    ``device`` binds the invite so only that device may redeem it; omitting it mints
    a token any device may present. The token itself is NEVER in the answer.
    """

    def work() -> dict[str, Any]:
        # The network is resolved first so an unknown or ambiguous NAME is refused
        # with the CLI's own sentence before the relay is asked — the same order the
        # CLI's verb uses, and the reason the refusal for a typo names the networks
        # this device IS in.
        resolve_network(_root(request), network)
        detail = invite(_root(request), network, role=body.role, device=body.device or "")
        return {"token_path": str(detail.get("path") or ""), "expires_at": detail.get("expires_at")}

    return reply(await _mesh(request, work))


@router.delete(
    "/v1/desktop/networks/{network}/members/{device}",
    response_model=CRUDResponse[RemovedMember],
)
async def remove_network_member(
    network: str, device: str, request: Request, body: RemoveMember = Body(...)
):
    """Revoke a member: the tombstone, the rotation and the epoch bump.

    THE TYPED CONFIRMATION IS CHECKED HERE, against the network's own name, and that
    is a rule about consequences rather than about diligence: this is the one act in
    the mesh that changes OTHER devices' state — every peer is rekeyed and the removed
    device is locked out on its next handshake — so the request must carry the name
    the user was shown rather than a bare bool a stray retry could also send. The
    comparison is exact: the name as typed, not a case-folded near-miss.

    ANY 2xx IS THE CONTRACT (Addendum 1, item 3) and a refusal carries ``message``,
    which is what the tab shows.
    """
    root = _root(request)

    def work() -> dict[str, Any]:
        record = resolve_network(root, network)
        # EXACTLY THE NAME AS TYPED, which is what this docstring has always claimed and
        # what the UI enforces (QA round 1, Q7). ``strip()`` made the backend the LOOSER
        # of the two gates: ``{"confirm": " qa498net "}`` was accepted and removed the
        # device, while the same string left the tab's Remove button disabled. One act
        # with two answers depending on which door the request came through is the
        # disagreement this route exists to prevent — and removal is the one act that
        # changes every other device's state.
        if body.confirm != str(record.name):
            raise MeshRefusal(
                "confirmation_mismatch",
                f"removing a member of {record.name!r} needs that network's name typed "
                "exactly, so nothing was removed",
            )
        detail = remove_member(root, network, device)
        return {
            "network_id": str(detail.get("network_id") or record.network_id),
            "removed": str(detail.get("removed") or device),
            "epoch": int(detail.get("epoch") or record.epoch),
        }

    return reply(await _mesh(request, work))


@router.post(
    "/v1/desktop/sessions/{session_id}/transfer", response_model=CRUDResponse[TransferReceipt]
)
async def transfer_session(session_id: str, body: TransferSession, request: Request):
    """Move a conversation to ``to`` (a device id) or home (``"local"``).

    ONE ANSWER, NOT A STREAM. The transport reports a move as a phase transcript and
    this app's IPC is request/response, so the route waits for the move to settle and
    returns the whole transcript at once (Addendum 1, item 4). The consequence is a
    long-held request: ``wait_s`` may be up to 300 s and the move's own budget is a
    retirement, a copy and a confirmation on top of it, so this handler takes NO
    generic short bound — the only bound is the one ``mobility.request_move`` sets for
    the work it does.

    THE CLIENT'S DEADLINE IS DERIVED, NOT NEGOTIATED, and it depends on the SHAPE:
    ``mobility.move_client_bound_s(wait_s, keep, to)``, whose three answers at the
    default ``wait_s=0`` are **145 s** for an offload, **415 s** for a ``keep`` copy
    and **415 s** for a recall. Both routes' defaults are ``wait_s=0``; a client that
    gives up sooner reports its own timeout for a move this side was about to answer
    (review round 1, MAJOR 1 — the desktop gave up at ``wait_s + 15`` against a route
    answering at ``wait_s + 30``). A recall is a BUDGET rather than a promise (the copy
    is transcript-sized), so a timeout on one is "unknown", never "refused".

    A REFUSAL IS A 409 CARRYING THE MOVE'S OWN SENTENCE, never a paraphrase: the
    renderer's S7 notice shows it, and the codes a caller can branch on
    (``busy``/``unreachable``/``not_authorised``/…) are the move's own.

    THE ONE CASE THAT IS NOT "NOTHING CHANGED" is a move this device could not get an
    answer about (a relay that went away, a deadline that fired, a peer that stopped
    replying after the request arrived): the request WAS sent, so the route answers 503
    with the source's state unconfirmed and says so in the sentence, and the renderer
    re-reads the row instead of retrying into a second move.
    """
    root = _root(request)
    key = f"transfer:{session_id}:{body.request_id}" if body.request_id else ""

    async def operation() -> dict[str, Any]:
        result = await asyncio.to_thread(
            transfer,
            session_id,
            to=body.to,
            keep=bool(body.keep),
            wait_s=float(body.wait_s or 0.0),
            root=root,
        )
        if result.get("ok"):
            return transfer_receipt(result, session_id=session_id, keep=bool(body.keep), to=body.to)
        code = str(result.get("code") or "move_refused")
        document = {
            "refused": True,
            "code": code,
            "message": str(result.get("message") or "the move was refused"),
            "changed": bool(result.get("changed")),
        }
        if code in _MOVE_UNCONFIRMED_CODES:
            # RETURNED rather than raised, so the receipt journal RECORDS it: a retry
            # of this request id replays "unconfirmed" instead of starting a second
            # move for a request whose first attempt may still be running.
            return document
        # NOTHING WAS MOVED, so the id stays usable: a user who frees the session up
        # and presses again must not be answered from a refusal forever.
        raise Unclaimed(document)

    async with errors(request):
        try:
            if key:
                result = await receipts(request).run(key, body.model_dump(), operation)
            else:
                try:
                    result = await operation()
                except Unclaimed as unclaimed:
                    # No journal, so there is nothing to release; the refusal is the
                    # answer either way.
                    result = unclaimed.result
        except MeshRefusal as error:
            raise _http_refusal(error) from None
        if not result.get("refused"):
            return reply(result)
        # AN UNCONFIRMED MOVE IS A 503, never a 409 that reads as "nothing changed"
        # (Addendum 2, C): the request was sent and the outcome is unknown, which is a
        # different instruction to the user than "the move was refused".
        code = str(result.get("code") or "move_refused")
        raise HTTPException(
            503 if code in _MOVE_UNCONFIRMED_CODES else 409,
            {"code": code, "message": str(result.get("message") or "the move was refused")},
        )


def _root(request: Request) -> Path:
    """This backend's config root — the one the network records live under.

    Read from the app's config manager rather than the process default: a backend
    started against a relocated root must read THAT machine's mesh, not the
    environment's, and every mesh read on this plane is keyed the same way.
    """
    return request.app.state.config_manager.config_dir
