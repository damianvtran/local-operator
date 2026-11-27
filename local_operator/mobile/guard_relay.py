"""The phone relay's Radient Guard twin (``/api/guard*``): host side of slice L2d.

WHY THIS EXISTS. The phone shows Radient Guard's state and relays the member's answer, but
Guard is host-level: the Local Operator host enforces it and the host signs the member's
consent. The phone app was built against a mock of these routes
(``local-operator-mobile`` ``docs/guard.md`` and ``tools/mock-relay/guard.ts``); this module is
the real host side. ``docs/design/guard-relay-twin.md`` is the contract and lists every
difference from that mock.

ONE ENGINE, THIN SKINS. Nothing here decides anything. Every route calls the SAME functions the
desktop plane's ``/v1/desktop/guard`` routes call (``server/utils/desktop_guard.py``, which
composes ``guard.surface``/``guard.consent``) or the SAME typed approval engine the desktop's
``approve-operation`` calls (``network.credentials.desktop_approval``). The only things added
here are the transport (cookie-gated Starlette instead of the desktop bearer), the body shape
(bare models, the relay's ``{error, code}`` envelope) and the Guard-only door onto the
``guard_consent`` card.

WHAT THE PHONE CAN AND CANNOT DO (the invariants the tests pin, each with a mutation):

* There is **no grant route**. A first grant or a widening is the host-signed typed accept of
  the pending ``guard_consent`` card, and ``accept`` carries exactly the two digests the card
  showed (closed body, no scope field). The host recomputes both from the record on disk and
  refuses a mismatch BEFORE its signer runs (RFC §5.2.2, §5.9). The generic bodiless
  ``approve`` stays refused for the card (``guard_consent_typed_approval_required``): this
  module never calls it.
* The relay is **not an approvals proxy**. Only a record of kind ``guard_consent`` is ever
  readable or answerable here; any other id, a missing id and a malformed id all answer the
  same ``404 approval_not_found``, so the relay cannot be used to probe other cards.
* ``share-again`` is **person-gated** by the same start-time capture the desktop uses
  (``surface.served_host_can_answer``): a ``lop mobile`` daemon started inside an agent's shell
  or a console pane refuses it (and reports ``can_answer: false``). The phone's cookie is the
  member's paired channel, not proof that a person is at the keyboard of an agent-started host.
* A non-Team account makes **zero** ``/v1/guard/*`` requests: ``GET /api/guard`` is local facts
  only, and every write and the consent routes refuse ``404 guard_unavailable`` before the
  engine can build a store. ``GET /api/guard/non-team-card`` is the one read that leaves the
  machine (``/v1/team-trial/eligibility``, never ``/v1/guard``).
* No notification, push or badge is raised from here.

WHY THE ANSWER IS STAMPED ``mobile``. The host signs and sends the grant, but the person
answered on the phone. The queue's ``decided_via`` (shown as "Answered on the phone app") and
the credential audit rows both say ``mobile`` for these calls (via
``desktop_approval.answering_as``), because "a person pressed a key on this computer" and "the
paired phone relayed a yes" are different facts for an incident review.

LAYOUT. Pure functions at the top return a body or raise :class:`GuardRefusal` /
``MeshRefusal``; :func:`routes` at the bottom wraps them in Starlette handlers behind the
daemon's own ``gate`` (cookie + same-origin mutation check), so the daemon needs one line to
mount them.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

logger = logging.getLogger(__name__)

#: A path id as Radient and the approvals store issue them. Bounded and separator-free, so a
#: segment cannot be a multi-kilobyte string or carry anything into a log line. An id that does
#: not fit is answered exactly like one that does not exist (no oracle on id shape).
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

#: The typed approval engine's digest-mismatch codes. The phone treats
#: ``guard_consent_expected_mismatch`` as "the request changed, re-read it", so a digest the
#: card did not show is answered under that one name; the engine's own code rides along as
#: ``engine_code`` for anyone reading the wire.
_DIGEST_MISMATCH = frozenset({"credential_request_changed", "credential_disclosure_changed"})


def _refusal(status: int, code: str, message: str) -> Any:
    from local_operator.server.utils.desktop_guard import GuardRefusal

    return GuardRefusal(status, code, message)


def _not_found() -> Any:
    return _refusal(404, "approval_not_found", "There is no Radient Guard request with that id.")


def _consent_record(root: Path, approval_id: str) -> dict[str, Any]:
    """The folded ``guard_consent`` card, else ``404 approval_not_found``.

    The kind check is the whole reason the relay is not a general approvals proxy: a card of
    any other kind (a credential grant, a phone pairing) is indistinguishable from no card.
    """

    from local_operator.guard import consent_op
    from local_operator.network import approvals
    from local_operator.network.types import MeshRefusal

    if not _ID.fullmatch(approval_id):
        raise _not_found()
    try:
        record = approvals.load_record(approval_id, root=root)
    except (MeshRefusal, OSError, ValueError):
        raise _not_found() from None
    if record.get("kind") != consent_op.OP:
        raise _not_found()
    return record


def _facts(record: dict[str, Any]) -> dict[str, Any]:
    """The card's structured disclosure (the engine's own), or ``guard_consent_payload_invalid``."""

    from local_operator.guard import consent_op
    from local_operator.network.types import MeshRefusal

    try:
        return consent_op.disclosure(record)
    except MeshRefusal as error:
        raise _refusal(409, error.code, str(error)) from None


def _engine_refusal(error: Any) -> Any:
    """A typed-engine refusal as the relay's 409, digest mismatches under the phone's name."""

    code = str(getattr(error, "code", "") or "guard_refused")
    if code in _DIGEST_MISMATCH:
        refusal = _refusal(
            409, "guard_consent_expected_mismatch", "This is not the reviewed Guard card."
        )
        refusal.engine_code = code
        return refusal
    return _refusal(409, code, str(error))


def status(root: Path, *, can_answer: bool) -> dict[str, Any]:
    """``GET /api/guard``: ``GuardStatus``. Local facts only: zero network for every account.

    ``listed`` is a throwaway set: it is the allow-list ``notices/{id}/shown`` checks, and the
    relay serves no such route (the phone shows no receipts; the desktop and the TUI do, and
    they mark them shown), so there is nothing for the phone to acknowledge.
    """

    from local_operator.server.models.desktop_guard import GuardStatus
    from local_operator.server.utils import desktop_guard as engine

    body = engine.status(root, can_answer=can_answer, listed=set())
    return GuardStatus.model_validate(body).model_dump(mode="json")


def non_team_card(root: Path) -> dict[str, Any]:
    """``GET /api/guard/non-team-card``: ``GuardNonTeamCard`` (the §14' card)."""

    from local_operator.server.models.desktop_guard import GuardNonTeamCard
    from local_operator.server.utils import desktop_guard as engine

    return GuardNonTeamCard.model_validate(engine.non_team_card(root)).model_dump(mode="json")


def revoke(root: Path, tenant: str, *, can_answer: bool) -> dict[str, Any]:
    """``POST /api/guard/orgs/{tenant}/revoke``: "Stop sharing" (always allowed, §5.4)."""

    from local_operator.server.models.desktop_guard import GuardRevokeResult
    from local_operator.server.utils import desktop_guard as engine

    if not _ID.fullmatch(tenant):
        raise _refusal(404, "guard_org_unknown", "This computer has no Guard record for that org.")
    body = engine.revoke(root, tenant, can_answer=can_answer)
    return GuardRevokeResult.model_validate(body).model_dump(mode="json")


def share_again(root: Path, tenant: str, *, can_answer: bool) -> dict[str, Any]:
    """``POST /api/guard/orgs/{tenant}/share-again``: lifts a sticky "no"; grants nothing."""

    from local_operator.server.models.desktop_guard import GuardShareAgainResult
    from local_operator.server.utils import desktop_guard as engine

    if not _ID.fullmatch(tenant):
        raise _refusal(404, "guard_org_unknown", "This computer has no Guard record for that org.")
    body = engine.share_again(root, tenant, can_answer=can_answer)
    return GuardShareAgainResult.model_validate(body).model_dump(mode="json")


def consent_card(root: Path, approval_id: str) -> dict[str, Any]:
    """``GET /api/guard/consents/{id}``: ``GuardConsentCard``, drawn from the host's own record.

    The digests are the engine's (``desktop_approval.row_fields``), the same two values the
    desktop's ``typed_approval`` block carries; ``signable``/``refusal`` are its policy verdict
    (assurance level, card state), so a card this host would refuse to sign is never offered.
    """

    from local_operator.network.credentials import desktop_approval
    from local_operator.server.models.desktop_guard import GuardConsentCard
    from local_operator.server.utils import desktop_guard as engine

    engine._require_available(root)
    record = _consent_record(root, approval_id)
    facts = _facts(record)
    typed = desktop_approval.row_fields(record) or {}
    request_digest = typed.get("request_digest")
    disclosure_digest = typed.get("disclosure_digest")
    if not request_digest or not disclosure_digest:
        # The engine could not describe the card (the digests are absent exactly then).
        raise _refusal(
            409,
            str(typed.get("refusal") or "guard_consent_payload_invalid"),
            "This Guard card cannot be described by this host.",
        )
    signable = bool(typed.get("signable_here"))
    body = {
        "approval_id": str(record.get("approval_id") or approval_id),
        "state": str(record.get("state") or ""),
        "tenant": facts["tenant"],
        "org": engine.surface.one_line(str(facts["org_display_name"] or "")),
        "reason": str(facts["reason"]),
        "widening": facts["reason"] == "widened",
        "asked_scopes": engine._scopes(facts["asked_scopes"]),
        "request_digest": request_digest,
        "disclosure_digest": disclosure_digest,
        "signable": signable,
        "refusal": None if signable else str(typed.get("refusal") or "guard_consent_not_signable"),
    }
    return GuardConsentCard.model_validate(body).model_dump(mode="json", exclude_none=True)


def _prompt_waiting(root: Path, tenant: str, facts: dict[str, Any], approval_id: str) -> bool:
    """Whether the queue still holds THIS card's prompt unanswered (``pending``)."""

    from local_operator.guard.consent import PROMPT_PENDING, GuardConsentHost

    try:
        row = GuardConsentHost(root, transport=None, mirror=False).prompt(
            tenant, int(facts["collection_rev"])
        )
    except Exception:  # noqa: BLE001 - unreadable queue: do not retry, refuse like a stale card
        return False
    return row is not None and row.approval_id == approval_id and row.state == PROMPT_PENDING


def accept(
    root: Path, approval_id: str, *, request_digest: str, disclosure_digest: str
) -> dict[str, Any]:
    """``POST /api/guard/consents/{id}/accept``: the host signs ``guard_consent`` and sends it.

    This is the desktop's typed ``approve-operation`` with the operation fixed to
    ``guard_consent`` by THIS function, never by the caller, so the relay cannot be steered at
    another card kind. The engine (``desktop_approval.approve_and_apply``) recomputes both
    digests from the record on disk and refuses a mismatch before any signer call; the apply
    re-verifies the signature and sends the one signed grant (``consent_op.apply``).

    A card the host already signed whose grant did not reach Radient (``approved``/``failed``)
    is retried through ``apply``, the desktop's own ``apply-operation`` retry: it records no new
    APPROVAL signature and re-checks both digests; the grant itself is re-sent as a fresh
    install-key envelope (new nonce), because that is what ``consent_op.apply`` builds.

    ``can_answer`` is deliberately not consulted here (S-2): the desktop plane's
    ``approve-operation`` has no such gate, and the grant is signed by the host's own key under
    its assurance level (see ``docs/design/guard-relay-twin.md`` §2).
    """

    from local_operator.guard import consent_op
    from local_operator.network import approvals
    from local_operator.network.credentials import desktop_approval
    from local_operator.network.types import MeshRefusal
    from local_operator.server.models.desktop_guard import GuardConsentAnswer
    from local_operator.server.utils import desktop_guard as engine

    engine._require_available(root)
    record = _consent_record(root, approval_id)
    facts = _facts(record)
    tenant = str(facts["tenant"])
    state = record.get("state")
    # Retry ONLY a card that was signed and whose queued prompt is still waiting (the grant
    # did not land). A card whose grant already landed, or whose prompt moved on, is not
    # re-answered: it falls through to ``approve_and_apply``, which refuses it
    # (``approval_decision_conflict``) exactly as the desktop's approve-operation does.
    retry = state in (approvals.STATE_APPROVED, approvals.STATE_FAILED) and _prompt_waiting(
        root, tenant, facts, approval_id
    )
    try:
        with desktop_approval.answering_as("mobile"):
            if retry:
                desktop_approval.apply(
                    root,
                    approval_id,
                    op=consent_op.OP,
                    request_digest=request_digest,
                    disclosure_digest_=disclosure_digest,
                )
                applied: dict[str, Any] = {"ok": True}
            else:
                phases = desktop_approval.approve_and_apply(
                    root,
                    approval_id,
                    op=consent_op.OP,
                    request_digest=request_digest,
                    disclosure_digest_=disclosure_digest,
                )
                applied = dict(phases.get("applied") or {})
    except MeshRefusal as error:
        raise _engine_refusal(error) from None
    if not applied.get("ok"):
        # Signed but the grant did not land: the decision stands and a retry re-sends it. Say
        # why under the engine's code, never as success.
        raise _refusal(
            409,
            str(applied.get("code") or "guard_consent_not_granted"),
            str(
                applied.get("message")
                or "Radient did not record this Guard answer; nothing new is being shared."
            ),
        )
    return GuardConsentAnswer(approval_id=approval_id, state="granted", tenant=tenant).model_dump()


def decline(root: Path, approval_id: str) -> dict[str, Any]:
    """``POST /api/guard/consents/{id}/decline``: the host's ordinary ``deny`` ("Don't share").

    The safe direction, so no digests are needed (as on the desktop). THE QUEUE ARBITRATES
    (L2d security round 1, S-1): the prompt row is declined FIRST, one atomic
    ``pending -> declined`` step (``GuardConsentHost.decline``), and only a win there is
    answered ``denied``. An accept that already claimed the row (``committing``: the grant is
    being sent) or landed (``accepted``) makes that step refuse, and the answer is
    ``409 approval_decision_conflict``, never ``denied``: the earlier shape (check the prompt,
    then write the card's deny) let a grant land in the gap and acknowledge "Don't share"
    beside a live consent. Once the queue holds the decline the accept's own claim can no
    longer succeed, so the card is then denied to match; the store's per-kind deny guard
    (``consent_op.require_deny_safe``) is the second wall for any surface that skips this order.
    The fold stamps ``decided_via = mobile`` and the sticky decline NOW, so the status read
    right after this call already says what happened and where.
    """

    from local_operator.guard.consent import GuardConsentHost, GuardConsentRefused
    from local_operator.network.types import MeshRefusal
    from local_operator.server.models.desktop_guard import GuardConsentAnswer
    from local_operator.server.utils import desktop_approvals as approvals_plane
    from local_operator.server.utils import desktop_guard as engine

    engine._require_available(root)
    record = _consent_record(root, approval_id)
    facts = _facts(record)
    tenant = str(facts["tenant"])
    # Only a prompt that is still WAITING can be declined; the same refusal as an accept of a
    # spent card, so a stale button reads one way. This early read is a courtesy: the atomic
    # decision is the ``decline`` below, which re-checks inside its own transaction.
    not_waiting = _refusal(
        409,
        "approval_decision_conflict",
        "This Guard request is no longer waiting for an answer. Nothing was changed.",
    )
    if not _prompt_waiting(root, tenant, facts, approval_id):
        raise not_waiting
    try:
        host = GuardConsentHost(root, transport=None, mirror=False)
        row = host.prompt(tenant, int(facts["collection_rev"]))
        if row is None or row.approval_id != approval_id:
            raise not_waiting
        host.decline(tenant, row.collection_rev, surface="mobile")
    except GuardConsentRefused:
        # Lost the race: an accept claimed or settled the row between the read above and the
        # transaction (or another surface answered). Nothing was written; say so.
        raise not_waiting from None
    except Exception as error:  # noqa: BLE001 - e.g. an unreadable queue: the deny below stands
        if error is not_waiting:
            raise
        logger.debug("guard relay: decline fold deferred", exc_info=True)
    try:
        approvals_plane.deny(root, approval_id)
    except MeshRefusal:
        # The queue already holds the decline, which is what the member asked for. The card is
        # usually ``withdrawn`` by that very step (the queue settled it), so a refused deny
        # here is the expected shape, not a failure.
        logger.debug("guard relay: card deny after queue decline refused", exc_info=True)
    return GuardConsentAnswer(approval_id=approval_id, state="denied", tenant=tenant).model_dump()


# ---------------------------------------------------------------------------
# Starlette skins
# ---------------------------------------------------------------------------


def routes(gate: Callable[[Any], Any], *, config_dir: Callable[[], Path]) -> list[Route]:
    """The twin's routes, each behind the daemon's own ``gate``.

    ``gate`` is ``build_app``'s: the paired-device cookie check plus the exact-Origin
    mutation guard, so these routes are authenticated exactly like ``/api/asks`` and
    ``/api/notifications`` and a request without the cookie answers ``401`` before anything
    below runs. ``config_dir`` is read per request so a relocated root reads THAT machine's
    Guard records.

    ``can_answer`` is read ONCE, here, when the app is built: it is a fact about how this
    daemon was STARTED (``surface.served_host_can_answer``), the same capture the desktop
    routes use, and a request must not be able to change it.
    """

    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from local_operator.guard import surface
    from local_operator.network.types import MeshRefusal
    from local_operator.server.models.desktop_guard import GuardConsentAcceptRequest
    from local_operator.server.utils.desktop_guard import GuardRefusal

    can_answer = surface.served_host_can_answer()

    def refuse(refusal: GuardRefusal) -> Response:
        body: dict[str, Any] = {"error": refusal.message, "code": refusal.code}
        engine_code = getattr(refusal, "engine_code", None)
        if engine_code:
            body["engine_code"] = engine_code
        return JSONResponse(body, status_code=refusal.status)

    async def answer(work: Callable[[], dict[str, Any]]) -> Response:
        """Run one blocking Guard call off the event loop; a refusal is a typed 4xx."""
        try:
            return JSONResponse(await asyncio.to_thread(work))
        except GuardRefusal as refusal:
            return refuse(refusal)
        except MeshRefusal as error:
            # A store/engine refusal that escaped before mapping: still a typed 409, never a 500.
            return refuse(_engine_refusal(error))

    async def api_guard(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return await answer(lambda: status(config_dir(), can_answer=can_answer))

    async def api_guard_non_team_card(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return await answer(lambda: non_team_card(config_dir()))

    async def api_guard_revoke(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        tenant = str(request.path_params["tenant"])
        return await answer(lambda: revoke(config_dir(), tenant, can_answer=can_answer))

    async def api_guard_share_again(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        tenant = str(request.path_params["tenant"])
        return await answer(lambda: share_again(config_dir(), tenant, can_answer=can_answer))

    async def api_guard_consent(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        approval_id = str(request.path_params["approval_id"])
        return await answer(lambda: consent_card(config_dir(), approval_id))

    async def api_guard_consent_accept(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        from pydantic import ValidationError

        try:
            raw = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(
                {"error": "a JSON body is required", "code": "invalid_body"}, status_code=422
            )
        try:
            # Closed and strict: the two digests and nothing else. A scope, an operation or
            # a signature in the body is a 422 that reaches no engine call.
            body = GuardConsentAcceptRequest.model_validate(raw)
        except ValidationError:
            return JSONResponse(
                {"error": "closed body: the two digests only", "code": "invalid_body"},
                status_code=422,
            )
        approval_id = str(request.path_params["approval_id"])
        return await answer(
            lambda: accept(
                config_dir(),
                approval_id,
                request_digest=body.request_digest,
                disclosure_digest=body.disclosure_digest,
            )
        )

    async def api_guard_consent_decline(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        approval_id = str(request.path_params["approval_id"])
        return await answer(lambda: decline(config_dir(), approval_id))

    return [
        Route("/api/guard", api_guard),
        Route("/api/guard/non-team-card", api_guard_non_team_card),
        Route("/api/guard/orgs/{tenant:str}/revoke", api_guard_revoke, methods=["POST"]),
        Route("/api/guard/orgs/{tenant:str}/share-again", api_guard_share_again, methods=["POST"]),
        Route("/api/guard/consents/{approval_id:str}", api_guard_consent),
        Route(
            "/api/guard/consents/{approval_id:str}/accept",
            api_guard_consent_accept,
            methods=["POST"],
        ),
        Route(
            "/api/guard/consents/{approval_id:str}/decline",
            api_guard_consent_decline,
            methods=["POST"],
        ),
    ]
