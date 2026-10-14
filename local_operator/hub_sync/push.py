"""Outbound push orchestration: paired projection -> preview -> CAS commit.

This is the ONE place the outbound pieces meet, so the order and the fences live
together rather than being re-derived by each caller (route, runner, tool):

1. Read the item's outbound receipt (A = captured local, P = accepted remote).
   With no receipt there is nothing authored to push: the first publication is a
   separate explicit action, never an implicit one.
2. Settle a NO-OP before any paid call when the projected document already
   equals R (the hub has nothing new to accept) -- no preview, no model, no push.
3. Compute the projection ``C = merge(A, L, P)`` then ``M = merge(P, C, R)``.
   A genuine ambiguity REFUSES here (needs-review), before any spend.
4. Persist the INTENT durably in the outbox, then dispatch preview -> commit over
   the abortable async transport bound to the attempt budget. A CAS mismatch
   (HTTP 412 ``remote-revision-changed``) is NOT a blind resubmit: it refetches
   and the caller re-projects.
5. Settle the outbox from the ACTUAL accepted remote document, fenced by the
   generation this worker started at.

The transport is INJECTED so a test drives a MockTransport end-to-end and the
real server wiring supplies the configured client. Nothing here reaches the
network on import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from local_operator.hub_sync.budget import OperationBudget, OperationDeadlineExceeded
from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbound import ProjectionResult, project_outbound, refuses
from local_operator.hub_sync.outbox import Outbox
from local_operator.hub_sync.provenance import OutboundReceipt

logger = logging.getLogger(__name__)

#: The one preview/commit vocabulary the server froze. Kept as data so a route
#: and the tool cannot drift about the path or the CAS header name.
PREVIEW_PATH = "/v1/desktop/hub/publish/preview"
COMMIT_PATH = "/v1/desktop/hub/publish/commit"
#: The optimistic-concurrency header: commit verifies the preview's revision.
ETAG_IF_MATCH = "If-Match"


class PushTransport(Protocol):
    """The narrow server surface a push needs: preview then conditional commit."""

    async def preview(
        self, *, kind: str, payload: Mapping[str, Any], budget: OperationBudget
    ) -> Mapping[str, Any]:
        """Return ``{"content_revision": str, "operation_id": str, ...}`` or raise."""

    async def commit(
        self,
        *,
        kind: str,
        payload: Mapping[str, Any],
        expected_revision: str,
        operation_id: str,
        budget: OperationBudget,
    ) -> Mapping[str, Any]:
        """Conditional write; raises :class:`RemoteRevisionChanged` on HTTP 412."""


class RemoteRevisionChanged(RuntimeError):
    """HTTP 412 ``remote-revision-changed``: refetch and re-project, never resubmit."""

    code = "remote-revision-changed"


@dataclass(frozen=True)
class PushOutcome:
    state: str  # "no-op" | "pushed" | "refused" | "needs-review" | "conflict"
    operation_id: str | None = None
    accepted: Mapping[str, Any] | None = None
    revision: str | None = None
    reason: str = ""


def _canonical_equal(kind: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    from local_operator.hub_sync.outbound import canonical_fields
    from local_operator.hub_sync.provenance import fingerprint_snapshot

    return fingerprint_snapshot(kind, canonical_fields(kind, a)) == fingerprint_snapshot(
        kind, canonical_fields(kind, b)
    )


async def push_item(
    *,
    kind: str,
    local_id: str,
    captured_local: Mapping[str, Any] | None,
    accepted: Mapping[str, Any] | None,
    current_local: Mapping[str, Any],
    remote_latest: Mapping[str, Any],
    transport: PushTransport,
    outbox: Outbox,
    operation_id: str,
    budget: OperationBudget,
    mode: str = "automatic",
    opts: MergeOptions = MergeOptions(),
) -> PushOutcome:
    """Project and conditionally publish one authored item; settle the outbox."""

    if captured_local is None or accepted is None:
        # No outbound receipt: this item was never authored from here, so there is
        # no captured ancestor to project a delta from. The first publication is a
        # deliberate, separate action -- an unattended push must not invent one.
        return PushOutcome("refused", reason="no-outbound-receipt")

    projection: ProjectionResult = project_outbound(
        kind,
        captured_local=captured_local,
        accepted=accepted,
        current_local=current_local,
        remote_latest=remote_latest,
        opts=opts,
    )

    # SETTLE THE NO-OP BEFORE ANY PAID CALL: if the projected document is already
    # what the hub holds, there is nothing to accept and no preview/model is paid.
    if projection.no_op:
        return PushOutcome("no-op", reason="M==R")

    refusal = refuses(projection)
    if refusal is not None:
        return PushOutcome("needs-review", reason=refusal)

    generation = outbox.bump_generation(operation_id) if outbox.read(operation_id) else 1
    if outbox.read(operation_id) is None:
        outbox.enqueue(
            operation_id=operation_id,
            kind=kind,
            local_id=local_id,
            direction="push",
            generation=generation,
        )

    payload = dict(projection.fields)
    try:
        preview = await transport.preview(kind=kind, payload=payload, budget=budget)
    except OperationDeadlineExceeded:
        # An aborted preview is UNCERTAIN, not failed: the server may or may not
        # have recorded it. Leave the intent pending for reconciliation rather
        # than settling a definitive failure the client cannot actually prove.
        outbox.mark_pending_reconciliation(operation_id, generation=generation)
        return PushOutcome("pending-reconciliation", operation_id=operation_id, reason="deadline")
    expected = str(preview.get("content_revision") or "")
    try:
        result = await transport.commit(
            kind=kind,
            payload=payload,
            expected_revision=expected,
            operation_id=str(preview.get("operation_id") or operation_id),
            budget=budget,
        )
    except RemoteRevisionChanged:
        # Fenced settlement: the caller refetches R and re-projects; we do NOT
        # resubmit the stale merged body.
        outbox.settle(operation_id, generation=generation, error_class="remote-revision-changed")
        return PushOutcome("conflict", operation_id=operation_id, reason="remote-revision-changed")
    except OperationDeadlineExceeded:
        # The commit may have landed server-side; never replay it as a fresh
        # create. Same op/pin/body on reconciliation. A later pass with revoked
        # permission must NOT mutate.
        outbox.mark_pending_reconciliation(operation_id, generation=generation)
        return PushOutcome("pending-reconciliation", operation_id=operation_id, reason="deadline")
    accepted_revision = str(result.get("content_revision") or expected)
    outbox.settle(operation_id, generation=generation, result={"revision": accepted_revision})
    return PushOutcome(
        "pushed",
        operation_id=operation_id,
        accepted=result,
        revision=accepted_revision,
    )


def record_generation_is_fresh(outbox: Outbox, operation_id: str, generation: int) -> bool:
    record = outbox.read(operation_id)
    return record is not None and record.generation == generation


def reconcilable(record: Any) -> bool:
    """Whether a pending-reconciliation record may be replayed against the server.

    Replay reuses the SAME operation id, preview pin and body -- it is never a new
    create. A revoked/forbidden permission is a MUTATING-replay blocker: the caller
    must re-authorize before the record can be retried, never push through it.
    """
    return record is not None and record.state == "pending-reconciliation"


def receipt_for(
    kind: str, accepted: Mapping[str, Any], current_local: Mapping[str, Any]
) -> OutboundReceipt:
    """Build the replacement outbound pair ``(captured local, accepted remote)``.

    Called ONLY after an accepted commit: A becomes the local input that was sent
    (``current_local``) and P the server-returned canonical document. ``B`` (the
    inbound integrated baseline) is never touched here.
    """
    from local_operator.hub_sync.provenance import fingerprint_snapshot

    local = dict(current_local)
    return OutboundReceipt(
        local_fields=local,
        accepted_fields=dict(accepted),
        input_fingerprint=fingerprint_snapshot(kind, local),
        remote_revision=str(accepted.get("content_revision") or "") or None,
        operation_id=str(accepted.get("operation_id") or "") or None,
    )
