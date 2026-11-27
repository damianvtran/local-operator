"""The ``guard_consent`` typed approval op: the desktop's host-level Guard card.

WHY A TYPED OP (Guard RFC v0.9 §5.2.2, R2-m2). The desktop approvals plane's
generic ``POST /v1/desktop/approvals/{id}/approve`` takes no body, so it cannot
prove which disclosure was on screen. Registering ``guard_consent`` in
``network.approval_ops`` makes that route (and ``lop network approvals
approve``) refuse the card through ``require_generic_approvable``; the only
way to accept it is ``/approve-operation`` with the closed
``OperationApprovalAcknowledgment{operation, request_digest,
disclosure_digest}``. The engine recomputes both digests from the record on
disk and refuses a mismatch BEFORE the signer runs, and that body has no scope
field, so an accept is all-or-nothing by construction.

MEASURED, NOT ASSUMED: the typed engine accepts a ``machine`` where-block.
The RFC left this open. ``profile_approval.py``'s ``profile_save`` op already
files ``machine={key_id, spki_fp}`` with ``what={"anchor", op}`` and is signed
through ``/approve-operation`` (``network.credentials.desktop_approval``
``operation()`` requires exactly that envelope), so ``guard_consent`` uses the
same shape and the RFC's fallback route is not needed.

THE DIGESTS. ``request_digest`` is the approvals store's digest over the
record's immutables, which include ``what.guard_consent.prompt_digest`` -- the
digest of the queued SQLite row -- so it binds the queued prompt. The
``disclosure_digest`` is the engine's digest over :func:`disclosure`, which
carries ``rendered_terms_digest`` and the exact asked/granted scopes, so a
renderer showing stale terms is refused before signing.

WHO MAY SIGN. ``file_only_signing = True`` is an explicit operator decision,
recorded here as ``approval_ops`` requires: the CTO decision of 2026-11-27
(Guard RFC binding input) rules out hardware and biometric requirements
("no hardware requirements"), so a member whose operator key is file-backed
can still answer on the desktop. A presence key signs as well.

WHAT APPLY DOES. :func:`apply` hands the signed card to
``GuardConsentHost.accept_signed``, which re-verifies the signature
(``verify_for_run``), maps it to ``admit_increasing(signature=True)`` and
sends the ONE signed grant. A deny on this card is the member's "Don't share"
(``reconcile_mirrors`` folds it into the queue, sticky).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from local_operator.network import approval_ops
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:  # pragma: no cover
    from local_operator.guard.consent import PromptRow

OP = "guard_consent"

#: How long one desktop mirror card stays open; see ``consent.MIRROR_WINDOW_S``.
_REQUESTED_BY = {"surface": "guard-host"}


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


def _request_id(row: PromptRow) -> str:
    from local_operator.network.wire import crockford

    seed = f"{row.digest}|{row.mirror_seq}".encode()
    return "req_" + crockford(hashlib.sha256(seed).digest()[:8])


def _payload(row: PromptRow) -> dict[str, Any]:
    return {
        "version": 1,
        "tenant": row.tenant,
        "org_display_name": row.org_display_name,
        "collection_rev": row.collection_rev,
        "revocation_gen": row.revocation_gen,
        "reason": row.reason,
        "scopes": list(row.scopes),
        "asked_scopes": list(row.asked_scopes),
        "grant_scopes": list(row.grant_scopes),
        "rendered_terms_digest": row.terms_digest,
        "prompt_digest": row.digest,
    }


def file_mirror(root: Path, row: PromptRow, *, now: float) -> str | None:
    """File the card for ``row``; its approval id, or ``None`` without an operator anchor."""

    from local_operator.guard.consent import MIRROR_WINDOW_S
    from local_operator.network import approvals as A

    trio = A.local_anchor_trio(root)
    if trio is None:
        return None
    anchor = {key: trio[key] for key in ("key_id", "spki_fp", "statement_digest")}
    record = A.create_request(
        kind=A.KIND_GUARD_CONSENT,
        request_id=_request_id(row),
        requested_by=dict(_REQUESTED_BY),
        machine={"key_id": anchor["key_id"], "spki_fp": anchor["spki_fp"]},
        what={"anchor": anchor, OP: _payload(row)},
        credential_ref=None,
        created_at=now,
        expires_at=now + MIRROR_WINDOW_S,
        root=root,
    )
    return str(record["approval_id"])


def settle_mirror(root: Path, approval_id: str) -> None:
    """Withdraw a card the queue already answered (another surface, a newer rev)."""

    from local_operator.network import approvals as A

    A.withdraw(approval_id, requested_by=dict(_REQUESTED_BY), root=root)


def mirror_state(root: Path, approval_id: str) -> str:
    """The card's folded state, or ``missing``."""

    from local_operator.network import approvals as A

    if not approval_id:
        return "missing"
    try:
        return str(A.load_record(approval_id, root=root).get("state") or "missing")
    except (MeshRefusal, OSError, ValueError):
        return "missing"


def _intent(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        what = record["what"]
        if record["kind"] != OP or set(what) != {"anchor", OP}:
            raise ValueError("not a guard consent envelope")
        if record.get("credential_ref") is not None:
            raise ValueError("a guard consent carries no credential")
        payload = what[OP]
        anchor = what["anchor"]
        if record.get("machine") != {"key_id": anchor["key_id"], "spki_fp": anchor["spki_fp"]}:
            raise ValueError("guard consent subject mismatch")
        if record.get("requested_by") != _REQUESTED_BY:
            raise ValueError("guard consent filed by another surface")
        expected_keys = {
            "version",
            "tenant",
            "org_display_name",
            "collection_rev",
            "revocation_gen",
            "reason",
            "scopes",
            "asked_scopes",
            "grant_scopes",
            "rendered_terms_digest",
            "prompt_digest",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("guard consent payload is not exact")
        if payload["version"] != 1:
            raise ValueError("unknown guard consent version")
        return dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(
            "guard_consent_payload_invalid", "this Guard card is not an exact queued prompt"
        ) from exc


def disclosure(record: dict[str, Any]) -> dict[str, Any]:
    """Every fact the member is agreeing to, as data (the renderer's source)."""

    payload = _intent(record)
    return {
        "version": 1,
        "operation": OP,
        "approval_id": str(record.get("approval_id") or ""),
        "tenant": payload["tenant"],
        "org_display_name": payload["org_display_name"],
        "reason": payload["reason"],
        "collection_rev": payload["collection_rev"],
        "revocation_gen": payload["revocation_gen"],
        # What the buttons mean: "Share" grants exactly ``asked_scopes`` (the
        # offered scopes, or the added ones on a widening). All or nothing.
        "asked_scopes": payload["asked_scopes"],
        "grant_scopes": payload["grant_scopes"],
        "rendered_terms_digest": payload["rendered_terms_digest"],
        "prompt_digest": payload["prompt_digest"],
        "expires_at": record.get("expires_at"),
        "signing_principal": (record.get("what") or {}).get("anchor", {}).get("key_id"),
    }


def expected(record: Mapping[str, Any]) -> dict[str, Any]:
    from local_operator.network import approvals as A

    payload = _intent(record)
    return {
        "request_digest": A.request_digest(record),
        "tenant": payload["tenant"],
        "collection_rev": payload["collection_rev"],
        "prompt_digest": payload["prompt_digest"],
    }


def apply(approval_id: str, expected_block: dict[str, Any], root: Path) -> dict[str, Any]:
    """Send the one signed grant for a signed card; raises ``MeshRefusal`` on refusal."""

    from local_operator.guard.consent import (
        GuardConsentHost,
        GuardConsentRefused,
        default_transport,
    )
    from local_operator.network import approvals as A

    record = A._load_raw(approval_id, root)
    if expected_block != expected(record):
        raise _refuse("guard_consent_expected_mismatch", "this is not the reviewed Guard card")
    host = GuardConsentHost(root, transport=lambda: default_transport(root))
    try:
        outcome = host.accept_signed(approval_id, expected_block)
    except GuardConsentRefused as exc:
        raise _refuse(exc.code, str(exc)) from exc
    if outcome.status != "granted":
        raise _refuse(
            outcome.code or "guard_consent_not_granted",
            "Radient did not record this Guard answer; nothing new is being shared. "
            "Try again from this card.",
        )
    return {"state": "granted", "approval_id": approval_id, "tenant": expected_block["tenant"]}


class _GuardConsentOp:
    op = OP
    kind = OP
    file_only_signing = True  # CTO decision 2026-11-27; see the module docstring.
    cli_approvable = False

    def intent(self, record: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return _intent(record)
        except MeshRefusal:
            return None

    disclosure = staticmethod(disclosure)
    expected = staticmethod(expected)

    def apply(self, approval_id: str, expected: dict[str, Any], root: Path) -> dict[str, Any]:
        return apply(approval_id, expected, root)

    def generic_refusal(self, record: Mapping[str, Any], approval_id: str) -> MeshRefusal:
        return _refuse(
            "guard_consent_typed_approval_required",
            "Review this Radient Guard card in full and answer it there; it cannot be "
            "approved from a list. Nothing was shared.",
        )


approval_ops.register(_GuardConsentOp())
