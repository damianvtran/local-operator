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

from pathlib import Path
from typing import Any

from local_operator.network import approvals as approval_store
from local_operator.network.credentials import inheritance
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

    record = approval_store.load_record(approval_id, root=root)
    if inheritance.intent(record) is not None:
        # REFUSED BEFORE THE SIGNER RUNS. The desktop card cannot yet state what a credential
        # permission signs (it would read the request's operator-key anchor as an
        # "install anchor" scope and show a blind Approve), and the signature is the
        # authority-increasing act. The terminal path prints the full disclosure first.
        # Deny stays available: it is the safe direction and needs no description.
        raise MeshRefusal(
            "inherit_terminal_approval_required",
            "This credential permission can only be approved from a terminal on this machine "
            "for now, where its account, recipient and duration are shown before you sign: "
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
