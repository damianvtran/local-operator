"""Outbound push orchestration: paired projection -> preview -> CAS commit.

This is the ONE place the outbound pieces meet, so the order and the fences live
together rather than being re-derived by each caller (route, runner, tool):

1. Read the outbox record FIRST. A ``pending-reconciliation`` record (an earlier
   attempt whose commit outcome is UNKNOWN) is replayed VERBATIM from its stored
   intent -- same pin, same body, same expected revision -- and is never
   re-projected into a fresh create or a second paid preview. A dead-lettered
   record is skipped with a status instead.
2. Read the item's outbound receipt (A = captured local, P = accepted remote).
   With no receipt there is nothing authored to push: the first publication is a
   separate explicit action, never an implicit one.
3. Compute the projection ``C = merge(A, L, P)`` then ``M = merge(P, C, R)``,
   settle a NO-OP before any paid call when M already equals R, and refuse a
   genuine ambiguity (needs-review) before any spend.
4. Persist the commit INTENT durably (after the preview produced its pin, before
   the commit dispatch), then dispatch over the abortable async transport bound
   to the attempt budget.
5. Settle the outbox from the ACTUAL accepted remote document, fenced by the
   generation this worker started at, and hand the accepted document back so the
   caller advances the outbound receipt.

THE WIRE IS THE SERVER'S FROZEN REVISION CONTRACT (agent-server
``docs/HUB-REVISIONS.md`` + the skill family), not a private protocol: preview and
commit are the standard per-family publish routes with a ``preview_version:1``
envelope, the commit carries ``X-Radient-Preview-Token`` and the SAME ``If-Match``
as its preview, and the expected revision is the OPAQUE ETag the merge actually
derived from (``"revision-v1-<hex>"``, quotes included). A stale token is a named
412 that is never retried by swapping the header: it settles a conflict and the
next attempt re-fetches and re-derives. An EMPTY expected revision refuses
locally -- an unconditional write is exactly what CAS exists to prevent.

The transport is INJECTED so a test drives a MockTransport end-to-end and the
real wiring supplies a per-link dispatcher transport. Nothing here reaches the
network on import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from local_operator.hub_sync.budget import OperationBudget, OperationDeadlineExceeded
from local_operator.hub_sync.identity import Kind, LinkError
from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbound import ProjectionResult, project_outbound, refuses
from local_operator.hub_sync.outbox import (
    Attempt,
    Outbox,
    OutboxRecord,
    push_backoff_seconds,
)
from local_operator.hub_sync.provenance import OutboundReceipt, now_iso

logger = logging.getLogger(__name__)

#: The optimistic-concurrency header: preview and commit both carry the opaque
#: ETag token VERBATIM (quotes included) of the remote revision the merge
#: derived from. Kept as data so a route and the tool cannot drift.
ETAG_IF_MATCH = "If-Match"


class PushRefused(RuntimeError):
    """The hub (or the local precondition) refused this publication, by NAME.

    ``code`` is the server's machine-readable refusal when one was returned
    (``name_taken``, ``unsupported-precondition``, ``preview_expired``,
    ``skill_not_found``, ...) or a local named reason when the refusal happened
    before any request. Messages never carry tenant or credential bytes.
    """

    def __init__(self, code: str, message: str = "", *, outage: bool = False) -> None:
        self.code = code
        #: True when the hub produced NO DECISION for this request: a transport
        #: failure or an HTTP 5xx. Classified from the STATUS (or the absence of
        #: one), NEVER from the server's body code string -- a 5xx whose body
        #: says ``hub-unavailable`` is the same outage as one with no body at
        #: all (round-3 delta QN10). Outage refusals refund their attempt and are
        #: paced by the outbox's consecutive-outage streak; deterministic
        #: refusals (a 4xx, a malformed 200) consume attempts and dead-letter.
        self.outage = outage
        super().__init__(message or f"the hub refused the publication ({code})")


class RemoteRevisionChanged(PushRefused):
    """HTTP 412 ``remote-revision-changed``: refetch and re-project, never resubmit."""

    def __init__(self) -> None:
        super().__init__("remote-revision-changed", "remote-revision-changed")


class PushTransport(Protocol):
    """The narrow per-link server surface a push needs: preview then CAS commit.

    The two calls are the SAME publish route; the commit body is the preview's
    returned generalized document (never the caller's input, per the frozen
    contract), plus the pin and acknowledgement headers.
    """

    async def preview(
        self, *, document: Mapping[str, Any], expected_revision: str, budget: OperationBudget
    ) -> Mapping[str, Any]:
        """Return the preview result (``document``, ``pin``, ``revision``, ...) or raise."""
        ...

    async def commit(
        self,
        *,
        document: Mapping[str, Any],
        preview_token: str,
        accept_unresolved: Sequence[str],
        expected_revision: str,
        budget: OperationBudget,
    ) -> Mapping[str, Any]:
        """The conditional write; raises :class:`RemoteRevisionChanged` on HTTP 412."""
        ...


#: Per-family route stems on the frozen publish surface. A linked item ALWAYS
#: has a remote row (an accepted receipt implies one), so a push is a REPUBLISH:
#: ``PUT /{family}/{hub_id}/publish`` and its ``/preview`` twin. First
#: publication is a deliberate manual action on the client, not this path.
_FAMILY_ROUTES = {"agent": "agents", "team": "teams", "skill": "skills"}


class HttpPushTransport:
    """The real per-link preview/commit transport over the abortable dispatcher.

    ``base_url`` and ``headers`` come from the SAME per-link client the check
    path selects (``build_clients``/``ClientFor``), so a push carries the
    credential of THAT link's source/tenant and lands on THAT source, never a
    global default. ``visibility``/``tenant_id`` travel as the family's standard
    target query params (agents and teams/skills spell a lone tenant
    differently, and the client's own helpers are the one place that rule
    lives). HTTP 412 becomes :class:`RemoteRevisionChanged`; every other refusal
    becomes a named :class:`PushRefused`; a 404 message never mentions a tenant
    (an org row addressed publicly must stay indistinguishable from absent).
    """

    def __init__(
        self,
        dispatcher: Any,
        *,
        kind: str,
        hub_id: str,
        base_url: str,
        headers: Mapping[str, str],
        visibility: str,
        tenant_id: str | None,
    ) -> None:
        family = _FAMILY_ROUTES.get(kind)
        if family is None:
            raise PushRefused("unsupported-precondition", f"no publish route for kind {kind!r}")
        if not hub_id:
            raise PushRefused("unsupported-precondition", "a push needs the remote row id")
        from local_operator.clients.radient import org_target_params, team_target_params

        if kind == "agent":
            params = org_target_params(visibility, tenant_id)
        else:
            params = team_target_params(visibility, tenant_id)
        self._dispatcher = dispatcher
        self._base = str(base_url).rstrip("/")
        self._headers = dict(headers)
        self._params = params
        self._publish_path = f"/{family}/{hub_id}/publish"

    async def preview(self, *, document, expected_revision, budget):
        return await self._send(
            self._publish_path + "/preview",
            document=document,
            expected_revision=expected_revision,
            extra_headers={},
            budget=budget,
        )

    async def commit(
        self, *, document, preview_token, accept_unresolved, expected_revision, budget
    ):
        from local_operator.clients.radient import (
            PREVIEW_ACCEPT_HEADER,
            PREVIEW_TOKEN_HEADER,
        )

        headers: dict[str, str] = {}
        if preview_token:
            # A bundle commit carries no pin: the header is omitted entirely
            # rather than sent empty, which the route would read as a refusal.
            headers[PREVIEW_TOKEN_HEADER] = preview_token
        if accept_unresolved:
            headers[PREVIEW_ACCEPT_HEADER] = ",".join(str(entry) for entry in accept_unresolved)
        return await self._send(
            self._publish_path,
            document=document,
            expected_revision=expected_revision,
            extra_headers=headers,
            budget=budget,
        )

    async def _send(
        self,
        path: str,
        *,
        document: Mapping[str, Any],
        expected_revision: str,
        extra_headers: Mapping[str, str],
        budget: OperationBudget,
    ) -> Mapping[str, Any]:
        if not str(expected_revision or "").strip():
            # An unconditional write is what CAS exists to prevent; never send one.
            raise PushRefused(
                "unsupported-precondition", "no expected remote revision to condition on"
            )
        headers = dict(self._headers)
        headers.update(extra_headers)
        headers[ETAG_IF_MATCH] = expected_revision
        # WIRE FACT (verified against the server source, agent-server@bdd03ab,
        # internal/controllers/generalization_preview.go + internal/requests/skills.go):
        # PREVIEW takes the envelope {preview_version, document[, base_document]};
        # the COMMIT route reads the BARE document with a closed field set -- any
        # top-level envelope key is refused as "not a recognised field" -- and the
        # pin travels in the X-Radient-Preview-Token header, with the SAME If-Match
        # as its preview (docs/HUB-REVISIONS.md: "Commit sends only the returned
        # result.document"). The document a commit sends is the one the PREVIEW
        # RETURNED, which the caller already carries.
        envelope = path.endswith("/preview")
        json_body = (
            {"preview_version": 1, "document": dict(document)} if envelope else dict(document)
        )
        try:
            response = await self._dispatcher.send(
                "PUT",
                f"{self._base}{path}",
                budget=budget,
                json_body=json_body,
                headers=headers,
                params=self._params,
                stage="preview" if envelope else "commit",
            )
        except OperationDeadlineExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure is a named refusal
            raise PushRefused(
                "hub-error", f"the hub request failed ({type(exc).__name__})", outage=True
            ) from exc
        if response.status_code == 412:
            raise RemoteRevisionChanged()
        if response.status_code >= 400:
            # The STATUS decides the attempt class; the body code is preserved
            # for the user-facing reason only.
            raise PushRefused(
                self._refusal_code(response),
                f"the hub refused the request (HTTP {response.status_code})",
                outage=response.status_code >= 500,
            )
        return _unwrap_result(response)

    @staticmethod
    def _refusal_code(response: Any) -> str:
        """The server's machine code, else ``http-<status>``. Never a tenant hint."""

        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON refusal is still a refusal
            body = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("code"), str):
                return str(error["code"])
        return f"http-{getattr(response, 'status_code', 'error')}"


def _unwrap_result(response: Any) -> Mapping[str, Any]:
    """The ``result`` object of the hub's ``{msg, result}`` envelope, strictly."""

    try:
        body = response.json()
    except Exception as exc:  # noqa: BLE001
        raise PushRefused("unrecognised-response", f"the hub's answer was not JSON: {exc}") from exc
    if not isinstance(body, Mapping) or not isinstance(body.get("result"), Mapping):
        raise PushRefused(
            "unrecognised-response",
            f"the hub answered without its envelope (HTTP {getattr(response, 'status_code', '?')})",
        )
    return body["result"]


@dataclass(frozen=True)
class PushOutcome:
    # "no-op" | "pushed" | "refused" | "needs-review" | "conflict"
    # | "pending-reconciliation" | "skipped"
    state: str
    operation_id: str | None = None
    accepted: Mapping[str, Any] | None = None
    revision: str | None = None
    reason: str = ""


def _canonical_equal(kind: Kind, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    from local_operator.hub_sync.outbound import canonical_fields
    from local_operator.hub_sync.provenance import fingerprint_snapshot

    return fingerprint_snapshot(kind, canonical_fields(kind, a)) == fingerprint_snapshot(
        kind, canonical_fields(kind, b)
    )


def _refusal_of(exc: BaseException) -> str:
    """The named class of a local document-building refusal."""

    if isinstance(exc, LinkError):
        return exc.code
    return "publish-document-invalid"


async def push_item(
    *,
    kind: Kind,
    local_id: str,
    captured_local: Mapping[str, Any] | None,
    accepted: Mapping[str, Any] | None,
    current_local: Mapping[str, Any],
    remote_latest: Mapping[str, Any],
    expected_revision: str | None,
    transport: PushTransport,
    outbox: Outbox,
    operation_id: str,
    budget: OperationBudget,
    opts: MergeOptions = MergeOptions(),
    document_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    bundle_attested: bool = False,
) -> PushOutcome:
    """Project and conditionally publish one authored item; settle the outbox.

    ``expected_revision`` is the opaque remote ETag the local merge input was
    derived from (the check's fetched revision, or the baseline's recorded
    one). It is REQUIRED: without it the call refuses locally rather than
    sending an unconditional write. ``document_builder`` renders the wire
    document for the kind from the projected canonical fields (agent/team add
    their row-owned identity fields; a skill's projection already IS its
    manifest).
    """
    if captured_local is None or accepted is None:
        # No outbound receipt: this item was never authored from here, so there is
        # no captured ancestor to project a delta from. The first publication is a
        # deliberate, separate action -- an unattended push must not invent one.
        # Checked BEFORE any outbox write: a refusal must not even fence a record.
        return PushOutcome("refused", reason="no-outbound-receipt")

    input_fp = _projection_fingerprint(
        kind, captured_local, accepted, current_local, remote_latest, expected_revision
    )
    attempt: Attempt | None = outbox.begin_attempt(
        operation_id, kind=kind, local_id=local_id, input_fp=input_fp
    )
    if attempt is None:
        # Dead-lettered (over the attempt cap), inside its backoff window, or an
        # unreplayable uncertain record: skip with a NAMED status. The caller
        # persists the reason; nothing was sent and nothing was settled here.
        return PushOutcome(
            "skipped", operation_id=operation_id, reason=_skip_reason(outbox, operation_id)
        )
    generation = attempt.generation

    if attempt.replay is not None:
        return await _replay_verbatim(
            transport=transport,
            outbox=outbox,
            operation_id=operation_id,
            generation=generation,
            intent=attempt.replay,
            budget=budget,
        )

    try:
        projection: ProjectionResult = project_outbound(
            kind,
            captured_local=captured_local,
            accepted=accepted,
            current_local=current_local,
            remote_latest=remote_latest,
            opts=opts,
        )
    except (KeyError, LinkError) as exc:
        # A kind with no declared outbound projection cannot be published at all:
        # refuse by NAME here rather than crashing the tick (the projection's
        # KeyError is a programming-shape signal, not a transport failure).
        code = exc.code if isinstance(exc, LinkError) else "unsupported-kind"
        return _settle_deterministic(outbox, operation_id, generation, code)

    # SETTLE THE NO-OP BEFORE ANY PAID CALL: if the projected document is already
    # what the hub holds, there is nothing to accept and no preview/model is paid.
    if projection.no_op:
        outbox.settle(operation_id, generation=generation, result={"no_op": True})
        return PushOutcome("no-op", reason="M==R")

    refusal = refuses(projection)
    if refusal is not None:
        return _settle_deterministic(
            outbox, operation_id, generation, refusal, state="needs-review"
        )
    if kind == "skill" and not projection.fields:
        # Belt and braces (R15): an EMPTY skill projection must never reach the
        # wire as a document; the server's manifest decoder would refuse it, and
        # a previous path could project exactly that from an empty remote.
        return _settle_deterministic(
            outbox, operation_id, generation, "merge-refused", state="needs-review"
        )

    expected = str(expected_revision or "").strip()
    if not expected:
        return _settle_deterministic(outbox, operation_id, generation, "unsupported-precondition")

    try:
        document = dict(
            document_builder(projection.fields)
            if document_builder is not None
            else projection.fields
        )
    except Exception as exc:  # noqa: BLE001 - a local builder refusal has a named class
        code = _refusal_of(exc)
        logger.debug("hub publish document refused for %s %s: %s", kind, local_id, exc)
        return _settle_deterministic(outbox, operation_id, generation, code)

    # BUNDLE CARRIER: the manifest is committed directly. The server's bundle
    # publish path ignores preview tokens (§7: no verifyPreviewIfPresent), so the
    # paid preview/pin handshake would refuse locally on a missing token and
    # spend a generalization the bundle never uses. The attestation guard is
    # checked BEFORE anything is sent: a server that never advertised
    # ``skill_bundles:1`` would accept the manifest against no blob.
    is_bundle = kind == "skill" and "bundle" in document
    if is_bundle and not bundle_attested:
        return _settle_deterministic(
            outbox, operation_id, generation, "bundle-unsupported-server", state="needs-review"
        )

    if is_bundle:
        preview_document: Mapping[str, Any] | None = document
        token = ""
    else:
        try:
            preview = await transport.preview(
                document=document, expected_revision=expected, budget=budget
            )
        except RemoteRevisionChanged:
            return _settle_conflict(outbox, operation_id, generation)
        except PushRefused as exc:
            return _settle_refused(outbox, operation_id, generation, exc.code, outage=exc.outage)
        except OperationDeadlineExceeded:
            # A PREVIEW-stage deadline owes NOTHING: no commit was dispatched, so
            # nothing can have been written. Retry with backoff -- an intent-less
            # pending-reconciliation here used to wedge the record forever
            # (round-2 R12c). Only a COMMIT-stage deadline is uncertain.
            return _settle_deadline(outbox, operation_id, generation)

        # POSITIVE CAS VERIFICATION before the commit spends anything further: the
        # preview must name the SAME revision our merge derived from, and a preview
        # that names none cannot be verified at all.
        observed = str(preview.get("revision") or "")
        if not observed:
            return _settle_deterministic(
                outbox, operation_id, generation, "unsupported-precondition"
            )
        if observed != expected:
            return _settle_conflict(outbox, operation_id, generation)

        status = str(preview.get("status") or "")
        unresolved = list(preview.get("unresolved") or [])
        if status == "needs_ack" and unresolved:
            # An unattended push never auto-acknowledges unclassified values: the
            # publisher's own confirmation is the point of the ack, and value
            # privacy depends on it. Stop safely with the ids recorded for a
            # manual surface.
            return _settle_deterministic(
                outbox, operation_id, generation, "generalization-needs-ack", state="needs-review"
            )

        preview_document = preview.get("document")
        pin = preview.get("pin")
        token = str(pin.get("token") or "") if isinstance(pin, Mapping) else ""
        if not isinstance(preview_document, Mapping) or not token:
            return _settle_deterministic(
                outbox, operation_id, generation, "unsupported-precondition"
            )

    assert isinstance(preview_document, Mapping)
    # DURABLE INTENT BEFORE THE COMMIT LEAVES: a crash or deadline after this
    # write can replay exactly these bytes (same pin/body/expected revision).
    intent = {
        "document": dict(preview_document),
        "preview_token": token,
        "accept_unresolved": [],
        "expected_revision": expected,
        # A bundle commit carries NO pin, so replay must not demand one; the flag
        # is what distinguishes "no token needed" from "lost the token".
        "bundle": is_bundle,
    }
    if not outbox.store_intent(operation_id, generation, intent):
        # Persisting the intent must succeed BEFORE a mutating dispatch: without
        # it an uncertain outcome could never be replayed verbatim, which is the
        # double-publish this outbox exists to prevent.
        return _settle_refused(outbox, operation_id, generation, "outbox-unwritable")

    try:
        result = await transport.commit(
            document=preview_document,
            preview_token=token,
            accept_unresolved=[],
            expected_revision=expected,
            budget=budget,
        )
    except RemoteRevisionChanged:
        return _settle_conflict(outbox, operation_id, generation)
    except PushRefused as exc:
        return _settle_refused(outbox, operation_id, generation, exc.code, outage=exc.outage)
    except OperationDeadlineExceeded:
        # The commit may have landed server-side; never replay it as a fresh
        # create. Same op/pin/body on reconciliation.
        outbox.mark_pending_reconciliation(operation_id, generation=generation)
        return PushOutcome("pending-reconciliation", operation_id=operation_id, reason="deadline")

    accepted_revision = str(result.get("revision") or "") or None
    outbox.settle(
        operation_id,
        generation=generation,
        result={"revision": accepted_revision, "operation_id": result.get("operation_id")},
    )
    return PushOutcome(
        "pushed",
        operation_id=operation_id,
        accepted=result,
        revision=accepted_revision,
    )


async def _replay_verbatim(
    *,
    transport: PushTransport,
    outbox: Outbox,
    operation_id: str,
    generation: int,
    intent: Mapping[str, Any],
    budget: OperationBudget,
) -> PushOutcome:
    """Resend one UNCERTAIN commit with exactly the stored bytes.

    No new projection and no new preview: the pin is still the one the original
    attempt bought, so a reconciliation can never loop the paid generalization
    stage (the manager's "no repeated generalization loops" requirement).
    """

    # GUARD, on purpose: ``begin_attempt`` released the outbox lock before the
    # network dispatch (the lock is never held across a network call), so another
    # window can finalize or supersede the record in between. Replay ONLY while
    # the record is still the same uncertain attempt: a finalized record or a
    # superseded generation is a named skip, never a double-sent commit.
    record = outbox.read(operation_id)
    if not reconcilable(record) or not record_generation_is_fresh(outbox, operation_id, generation):
        return PushOutcome("skipped", operation_id=operation_id, reason="reconciliation-required")

    document = intent.get("document")
    token = str(intent.get("preview_token") or "")
    accepts = list(intent.get("accept_unresolved") or [])
    expected = str(intent.get("expected_revision") or "")
    is_bundle = bool(intent.get("bundle"))
    if not isinstance(document, Mapping) or not expected or (not token and not is_bundle):
        return PushOutcome(
            "skipped", operation_id=operation_id, reason="reconciliation-intent-invalid"
        )
    try:
        result = await transport.commit(
            document=document,
            preview_token=token,
            accept_unresolved=accepts,
            expected_revision=expected,
            budget=budget,
        )
    except RemoteRevisionChanged:
        return _settle_conflict(outbox, operation_id, generation)
    except PushRefused as exc:
        return _settle_refused(outbox, operation_id, generation, exc.code, outage=exc.outage)
    except OperationDeadlineExceeded:
        outbox.mark_pending_reconciliation(operation_id, generation=generation)
        return PushOutcome("pending-reconciliation", operation_id=operation_id, reason="deadline")
    accepted_revision = str(result.get("revision") or "") or None
    outbox.settle(
        operation_id,
        generation=generation,
        result={"revision": accepted_revision, "operation_id": result.get("operation_id")},
    )
    return PushOutcome(
        "pushed", operation_id=operation_id, accepted=result, revision=accepted_revision
    )


def _settle_conflict(outbox: Outbox, operation_id: str, generation: int) -> PushOutcome:
    # A 412 is a CONCURRENT REMOTE EDIT, not a client failure: it gets a backoff
    # (so a busy remote is not hammered) but the attempt is REFUNDED so conflicts
    # alone can never walk an actively edited link into the dead-letter cap
    # (round-2 R12d).
    outbox.settle(
        operation_id,
        generation=generation,
        error_class="remote-revision-changed",
        next_retry_at=_retry_at(generation, outbox, operation_id),
        refund_attempt=True,
    )
    return PushOutcome("conflict", operation_id=operation_id, reason="remote-revision-changed")


def _settle_deterministic(
    outbox: Outbox,
    operation_id: str,
    generation: int,
    code: str,
    *,
    state: str = "refused",
) -> PushOutcome:
    """A refusal retrying the SAME input cannot change (needs-review, no-op rules,
    an invalid document, a missing revision): settle without backoff, REFUND the
    attempt and wait for the INPUT fingerprint to move (round-2 R12a)."""

    outbox.settle(
        operation_id,
        generation=generation,
        error_class=code,
        refund_attempt=True,
        waiting_input=True,
    )
    return PushOutcome(state, operation_id=operation_id, reason=code)


def _settle_deadline(outbox: Outbox, operation_id: str, generation: int) -> PushOutcome:
    """A PREVIEW-stage deadline: retryable (backoff), never a reconciliation.

    Nothing was dispatched that could have written, so no commit is owed; the
    old code marked these pending-reconciliation with no stored intent, which
    ``begin_attempt`` then skipped forever (round-2 R12c).
    """

    outbox.settle(
        operation_id,
        generation=generation,
        error_class="deadline",
        next_retry_at=_retry_at(generation, outbox, operation_id),
    )
    return PushOutcome("refused", operation_id=operation_id, reason="deadline")


def _projection_fingerprint(
    kind: str,
    captured_local: Mapping[str, Any] | None,
    accepted: Mapping[str, Any] | None,
    current_local: Mapping[str, Any],
    remote_latest: Mapping[str, Any],
    expected_revision: str | None,
) -> str:
    """Hash of everything a projection decision depends on (round-2 R12).

    Retrying the SAME input cannot change a deterministic refusal, so the outbox
    waits; a differing fingerprint is a deliberate new edit and re-arms the
    attempt budget. Computed over the raw canonical mappings, sorted, so a
    re-serialized dict with identical content hashes identically.
    """

    import hashlib
    import json as _json

    payload = _json.dumps(
        {
            "kind": kind,
            "captured": captured_local,
            "accepted": accepted,
            "local": current_local,
            "remote": remote_latest,
            "expected": expected_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _settle_refused(
    outbox: Outbox,
    operation_id: str,
    generation: int,
    code: str,
    *,
    outage: bool = False,
) -> PushOutcome:
    """Settle a named refusal, refunding and pacing OUTAGES correctly.

    ``outage`` rides the refusal itself (set at the transport from the HTTP
    STATUS or a transport exception -- never inferred from ``code``, round-3
    delta QN10). Outages REFUND their attempt -- nothing about the request was
    answered, and five of them used to dead-letter a healthy link (QN8) -- so
    their pacing comes from the outbox's consecutive-outage streak instead of
    the refunded counter (R22): the window still doubles (900s, 1800s, 3600s,
    ... capped) while the cap is never approached. Deterministic refusals (a
    4xx, a malformed answer) keep consuming attempts and still dead-letter:
    those ARE answers, and re-sending one would be paid work with a known
    outcome.
    """

    if outage:
        record = outbox.read(operation_id)
        streak = (record.outage_streak if record is not None else 0) + 1
        from datetime import datetime, timedelta, timezone

        due = datetime.now(timezone.utc) + timedelta(seconds=push_backoff_seconds(streak))
        next_at = due.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        next_at = _retry_at(generation, outbox, operation_id)
    outbox.settle(
        operation_id,
        generation=generation,
        error_class=code,
        refund_attempt=outage,
        outage=outage,
        next_retry_at=next_at,
    )
    return PushOutcome("refused", operation_id=operation_id, reason=code)


def _retry_at(generation: int, outbox: Outbox, operation_id: str) -> str:
    """When the next attempt may start, in the SAME ISO spelling as ``now_iso()``.

    The spelling matters: ``begin_attempt`` compares against ``now_iso()`` as a
    string, so a different format (a Z suffix, fractional seconds, ...) would
    order incorrectly.
    """

    record = outbox.read(operation_id)
    attempts = record.attempts if record is not None else 1
    from datetime import datetime, timedelta, timezone

    due = datetime.now(timezone.utc) + timedelta(seconds=push_backoff_seconds(attempts))
    return due.strftime("%Y-%m-%dT%H:%M:%SZ")


def record_generation_is_fresh(outbox: Outbox, operation_id: str, generation: int) -> bool:
    record = outbox.read(operation_id)
    return record is not None and record.generation == generation


def _skip_reason(outbox: Outbox, operation_id: str) -> str:
    """WHY ``begin_attempt`` declined, as a named, persistable status."""

    from local_operator.hub_sync.outbox import PUSH_MAX_ATTEMPTS

    record = outbox.read(operation_id)
    if record is None:
        return "push-unavailable"
    if record.state == "pending-reconciliation":
        return "reconciliation-required"
    if record.waiting_input:
        # A deterministic refusal: the SAME input cannot change the answer, so
        # the skip names the thing that CAN end it (round-2 R12a).
        return "awaiting-input-change"
    if record.attempts >= PUSH_MAX_ATTEMPTS:
        return "push-retries-exhausted"
    return "push-backoff"


def reconcilable(record: OutboxRecord | None) -> bool:
    """Whether a pending-reconciliation record may be replayed against the server.

    Replay reuses the SAME operation id, preview pin and body -- it is never a new
    create. A record without a stored intent cannot be replayed verbatim and is
    dead-lettered by the outbox instead.
    """

    return record is not None and record.state == "pending-reconciliation"


def receipt_for(
    kind: Kind,
    accepted: Mapping[str, Any],
    current_local: Mapping[str, Any],
    *,
    revision: str | None = None,
    operation_id: str | None = None,
) -> OutboundReceipt:
    """Build the replacement outbound pair ``(captured local, accepted remote)``.

    Called ONLY after an accepted commit: A becomes the local input that was sent
    (``current_local``) and P the server-returned canonical document. ``revision``
    is the OPAQUE accepted ETag (``result['revision']``) -- the header value a
    later conditional write must send verbatim; ``content_revision`` (an integer)
    is deliberately NOT used, because it cannot be an If-Match token. ``B`` (the
    inbound integrated baseline) is never touched here.
    """
    from local_operator.hub_sync.provenance import fingerprint_snapshot

    local = dict(current_local)
    return OutboundReceipt(
        local_fields=local,
        accepted_fields=dict(accepted),
        input_fingerprint=fingerprint_snapshot(kind, local),
        remote_revision=str(revision or "") or None,
        operation_id=str(operation_id or "") or None,
    )


__all__ = [
    "Attempt",
    "ETAG_IF_MATCH",
    "HttpPushTransport",
    "PushOutcome",
    "PushRefused",
    "PushTransport",
    "RemoteRevisionChanged",
    "push_item",
    "receipt_for",
    "record_generation_is_fresh",
    "reconcilable",
]


# ``now_iso`` is re-exported for callers that stamp receipts alongside a push.
_ = now_iso
