"""The desktop's approval reads and answers, over the DEVICE-LOCAL store.

WHY A MODULE RATHER THAN CODE IN THE ROUTES. Three routes answer from one seam —
``network.approvals``, the SAME functions the CLI's verbs call — so the desktop
cannot grow a second policy for who may approve or what a refusal says. There is
nothing to dial: the record is DEVICE-LOCAL (remote-onboarding §2.3), carries a
secret-store REFERENCE and public fingerprints and never leaves this machine, and
the badge must answer on a machine whose relay is down — which is exactly why the
store lives in a flat directory (``<config>/network/approvals/``) rather than
behind the running relay.

THE AUTHORISATION RULE. The desktop bearer token gates the whole router; inside
it, these routes have exactly the authority the operator has at their own shell,
and no more: :func:`approve` runs the SAME presence-gated signing call the CLI's
``approve`` verb runs (the OS prompt is what decides — an HTTP request cannot
approve anything by itself), and every refusal travels out verbatim with its
machine code.

WHAT A ROW MAY EXPOSE: the operator's own record — the where-block, the scope
list, the requester, and PUBLIC key fingerprints. A credential never appears: the
record holds a reference NAME and this module never reads the secret store.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from local_operator.network import approvals as approval_store
from local_operator.network.credentials import consent, finite_share, inheritance
from local_operator.network.types import MeshRefusal


def approval_rows(root: Path | None = None) -> list[dict[str, Any]]:
    """The badge read: every record this device holds, in store order (oldest first).

    FOLDED, never materialised: a record past its window reads ``expired`` here
    without this read writing anything (the asks-store discipline) — the badge
    must not be a writer.
    """
    rows: list[dict[str, Any]] = []
    for record in approval_store.list_records(root=root):
        row = approval_store.badge_row(record)
        # OPTIONAL, KEYLESS, BACKEND-DERIVED EXTRAS for a credential permission only: its
        # plain-language summary/details come from the SIGNED record at read time (never
        # stored text) so a UI that knows the keys can state what is signed, and one that
        # does not simply ignores them. Every other kind keeps the frozen row byte for byte.
        if finite_share.has_intent(record):
            # Presence wins over parsing: malformed/mixed intent is never a
            # generic onboarding card that an older client could blindly sign.
            row["can_approve_here"] = False
            row["finite_share"] = {"version": 1, "valid": False}
            try:
                data = finite_share.checked_intent(record)
                row["finite_share"] = {
                    "version": 1,
                    "valid": True,
                    "request_digest": record["request_digest"],
                    **data,
                }
            except MeshRefusal as error:
                row["finite_share"]["refusal_code"] = error.code
        else:
            extras = inheritance.row_extras(record)
            if extras is not None:
                row.update(extras)
        rows.append(row)
    return rows


def approve(root: Path | None, approval_id: str) -> dict[str, Any]:
    """Sign and write the operator's yes; answer the frozen decision shape (§3.5).

    The signing call is the presence-gated one (Touch ID where the host offers
    it), so this call's latency includes a human's — the route dispatches it off
    the event loop for that reason as much as for the store's file lock.
    """
    import time

    from local_operator.network import enrollment

    if approval_id.startswith("apn_"):
        raise MeshRefusal(enrollment.CONFIRMATION, enrollment.REMEDY)
    record = approval_store.load_record(approval_id, root=root)
    if record.get("kind") not in approval_store.KINDS:
        raise MeshRefusal("approval_invalid", "This build cannot approve this authority kind")
    if finite_share.has_intent(record):
        consent.refuse("finite_confirmation_required")
    if inheritance.intent(record) is not None:
        # REFUSED BEFORE THE SIGNER RUNS. The desktop card cannot yet state what a credential
        # permission signs (it would read the request's operator-key anchor as an
        # "install anchor" scope and show a blind Approve), and the signature is the
        # authority-increasing act. The command-line path prints the full disclosure first.
        # Deny stays available: it is the safe direction and needs no description.
        raise MeshRefusal(
            "inherit_terminal_approval_required",
            "This credential permission can only be approved from the command line on this "
            "machine for now, where its account, recipient and duration are shown before you "
            "sign: "
            f"lop network approvals approve {approval_id}. Nothing was approved.",
        )
    decided_at = time.time()
    signature_hex = approval_store.sign_decision(
        kind=record["kind"],
        request_id=record["request_id"],
        request_digest=record["request_digest"],
        decided_at=decided_at,
    )
    decided = approval_store.approve(
        approval_id, signature_hex=signature_hex, decided_at=decided_at, root=root
    )
    return _decision(decided)


def _finite_snapshot(
    root: Path | None, approval_id: str, digest: str, *, state: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a displayed finite-only card to the existing signer/admission path."""
    from local_operator.network import enrollment
    from local_operator.network.identity import load as load_identity

    enrollment.bound_profile(root)
    record = approval_store.load_record(approval_id, root=root)
    data = finite_share.checked_intent(record)
    if record["state"] != state or time.time() >= record["expires_at"]:
        consent.refuse("finite_state_changed")
    if digest != record["request_digest"] or digest != approval_store.request_digest(record):
        consent.refuse("finite_snapshot_changed")
    identity = load_identity(root)
    if identity is None or identity.device_id != data["owner_device"]:
        consent.refuse("finite_owner_mismatch")
    return record, data


def approve_finite(root: Path | None, approval_id: str, digest: str) -> dict[str, Any]:
    """Sign the displayed finite scope through the ordinary presence-gated signer."""
    record, _data = _finite_snapshot(root, approval_id, digest, state="requested")
    decided_at = time.time()
    signature = approval_store.sign_decision(
        kind=record["kind"], request_id=record["request_id"],
        request_digest=record["request_digest"], decided_at=decided_at,
    )
    return _decision(approval_store.approve(
        approval_id, signature_hex=signature, decided_at=decided_at, root=root
    ))


def apply_finite(root: Path | None, approval_id: str, digest: str) -> dict[str, Any]:
    """Apply only the signed subject, never HTTP-authored owner/key/scope selectors."""
    _record, data = _finite_snapshot(root, approval_id, digest, state="approved")
    expected = {field: data[field] for field in (
        "owner_device", "network_id", "recipient_device", "key", "scope"
    )}
    finite_share.apply(approval_id, expected, root=root)
    # Apply owns fresh signature, policy, fingerprint, retained-fence and replay
    # checks. Do not mark the card applied merely because a signer returned yes.
    return {"approval_id": approval_id, "state": "applied"}


def deny(root: Path | None, approval_id: str) -> dict[str, Any]:
    """Write the operator's no. Ordinary, write-once, safe direction (§2.4)."""
    return _decision(approval_store.deny(approval_id, root=root))


def _decision(record: dict[str, Any]) -> dict[str, Any]:
    """The frozen decision shape: ``{approval_id, state, signature:{key_id}}``.

    A deny carries no signature (it never needs one — the safe direction), so
    ``key_id`` is ``""`` there rather than a field that vanishes.
    """
    signature = record.get("signature") or {}
    return {
        "approval_id": record.get("approval_id"),
        "state": record.get("state"),
        "signature": {"key_id": str(signature.get("key_id") or "")},
    }
