"""Deliver an explicit ``credential share`` to its new holder in one round trip.

WHY THIS MODULE EXISTS (mesh perf matrix, share-propagation finding). Before it, a
share reached the borrower only when the BORROWER pulled the owner's placement
document on its own cadence: 0-15 s of tick phase on a capable borrower, and never at
all on a default-invited (``--role drive``) one, whose cadence was gated on its own
admission-time capabilities. The owner already knows, at share time, exactly which
device just became a holder of exactly which row, and it has (or can dial) a link to
it. So the share verb asks its own relay to hand that ONE row to that ONE device now.

WHAT IT GRANTS: NOTHING NEW. The pushed row is byte-for-byte the row the borrower's
own pull would have fetched (``owner.placement_frame``'s ``want: pull`` answer), and
it is applied through the same merge (``placement.merge_from_peer``), which already
admits only rows the AUTHENTICATED sender owns, refuses anything claiming to come from
this device and drops a lower-or-equal ``doc_rev``. On top of that the receiver here
refuses a row that does not name THIS device as a holder, so the push can only ever
deliver "you may now borrow X from me" — never a third device's row, never a holder
list for somebody else. The owner remains the enforcement point for every borrow: a
placement row on the borrower is a routing hint the owner's broker re-checks on each
``grant`` against its OWN document and its OWN member row for the borrower.

WHAT IT NEVER CARRIES: credential material. The document passes
``placement._assert_no_material`` on both ends, and refresh tokens stay on the owner
(the grant op still mints only short-lived access).

MIXED VERSIONS. Sent only when both ends advertised ``credential-share-push-v1``
(``wire.CREDENTIAL_SHARE_PUSH_V1``). An older peer is not pushed to — it converges
through its own pull, exactly as before — and the share verb says so instead of
claiming delivery.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from local_operator.network.types import MeshRefusal

logger = logging.getLogger(__name__)

#: Marker on a ``net_broker kind=placement`` frame that is a share push. Distinct from
#: ``reconcile.PUSH_ORIGIN`` because the two receivers admit different row classes: the
#: consent reconciler's admits only derived native-OAuth rows, this one admits any
#: explicitly shared row that names the receiver as a holder.
SHARE_PUSH_ORIGIN = "credential_share"

#: One share names one key, so one row is the honest bound; a few spare rows keep the
#: receiver tolerant of a sender that later batches a multi-key share.
MAX_SHARE_ROWS = 8
MAX_SHARE_BYTES = 65536


def negotiated(link: Any) -> bool:
    """Both ends of THIS link advertised the share push (the ``reconcile`` rule:
    the peer's advertisement alone is not negotiation, the local set must carry it)."""
    from local_operator.network import wire

    return wire.CREDENTIAL_SHARE_PUSH_V1 in getattr(
        link, "local_capabilities", ()
    ) and wire.CREDENTIAL_SHARE_PUSH_V1 in getattr(link, "capabilities", ())


def _refuse(code: str, message: str) -> dict[str, Any]:
    return {"kind": "error", "code": code, "message": message}


# ---------------------------------------------------------------------------
# Owner side: called from the ``credential_share_push`` local op
# ---------------------------------------------------------------------------


def push_share(server: Any, network_id: str, device_id: str, key: str) -> dict[str, Any]:
    """Push the owner's row for ``key`` to ``device_id``. Returns ``{delivered, outcome}``.

    Outcomes: ``pushed`` (the holder merged it, or already had it) |
    ``offline`` (no link could be opened) | ``skipped:<why>`` (nothing to push, or a
    peer build without the feature) | ``refused:<code>`` (the holder answered no) |
    ``failed`` (the request got no answer). Every outcome but ``pushed`` leaves
    delivery to the holder's own pull, which is what the share verb tells the operator.
    """
    from local_operator.network.credentials import placement as placement_mod
    from local_operator.network.credentials import reconcile as reconcile_mod
    from local_operator.network.credentials.sync import ANNOUNCE_TIMEOUT_S

    root = Path(getattr(server, "root"))
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if not (network_id and device_id and key and self_device) or device_id == self_device:
        return {"delivered": False, "outcome": "skipped:bad_request"}
    document = placement_mod.PlacementDocument.load(network_id, root, self_device=self_device)
    entry = document.entry(key)
    if entry is None or entry.owner_device != self_device or not entry.is_holder(device_id):
        # The share verb writes the row before asking, so this is a race with a revoke
        # (or a key this device does not own): pushing anything here would be wrong.
        return {"delivered": False, "outcome": "skipped:not_a_holder"}
    ensure = getattr(server, "_ensure_link_with_reason", None)
    next_req = getattr(server, "_next_relay_req", None)
    if ensure is None or next_req is None:
        return {"delivered": False, "outcome": "skipped:no_dialer"}
    try:
        link, _why = ensure(device_id)
    except Exception:  # noqa: BLE001 — an unreachable holder is an outcome, not a crash
        logger.debug("share push: dial to %s failed", device_id, exc_info=True)
        link = None
    if link is None:
        return {"delivered": False, "outcome": "offline"}
    if not negotiated(link):
        return {"delivered": False, "outcome": "skipped:peer_lacks_share_push"}
    if str(getattr(link, "network_id", "") or "") != network_id:
        # The link authenticated in another network both devices share; the receiver
        # merges into the LINK's network, so this row would land in the wrong document.
        return {"delivered": False, "outcome": "skipped:other_network"}
    frame = {
        "op": "net_broker",
        "kind": "placement",
        "origin": SHARE_PUSH_ORIGIN,
        "network_id": network_id,
        "from_device": self_device,
        "req": next_req(),
        "document": {
            "schema": placement_mod.PLACEMENT_SCHEMA,
            "network_id": network_id,
            # Never the owner's epoch: the merge persists max(local, pushed), and a
            # push has no business moving the receiver's epoch (``reconcile`` rule).
            "epoch": 0,
            "written_by": self_device,
            "credentials": [entry.to_json()],
        },
    }
    try:
        reply = link.request(frame, timeout=ANNOUNCE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — one holder's link is that holder's problem
        logger.debug("share push: request to %s failed", device_id, exc_info=True)
        reply = None
    verdict, code = reconcile_mod._classify_reply(reply)  # noqa: SLF001 — one classifier
    if verdict == "ok":
        return {"delivered": True, "outcome": "pushed"}
    if verdict in ("peer_refused", "rejected"):
        return {"delivered": False, "outcome": f"refused:{code or 'refused'}"}
    return {"delivered": False, "outcome": "failed"}


# ---------------------------------------------------------------------------
# Holder side: routed from ``owner.relay_handler`` / ``MeshCredentialBroker.on_broker``
# ---------------------------------------------------------------------------


def member_share_push(server: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a share push. Reached with or without a broker on this device, because the
    device that most needs it (a borrower-only one) owns nothing and builds no broker.

    The relay's authoriser has already required the sender's ``broker_credential`` in
    THIS device's row for it (``types.OP_CAPABILITY['net_broker']``). Every field is
    type-checked before it is compared, and nothing is written unless all checks pass.
    """
    root = getattr(server, "root", None)
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if root is None or not self_device:
        return _refuse("unavailable", "this device has no identity; nothing applied")
    try:
        changed = _accept(Path(root), self_device, link, frame)
    except (OSError, MeshRefusal) as exc:
        return _refuse(getattr(exc, "code", "") or "busy", "the push was not applied")
    if isinstance(changed, dict):
        return changed
    # The owner just proved, by pushing, that its row for this device now admits a
    # broker op — so a refusal parked by this device's cadence (see
    # ``sync._pull_past_stale_self_view``) is stale evidence and must not delay the
    # copies this share may need.
    from local_operator.network.credentials import sync as sync_mod

    sync_mod.forget_owner_refusal(server, str(getattr(link, "device_id", "") or ""))
    return {"kind": "ack", "key": "", "changed": changed}


def _accept(
    root: Path, self_device: str, link: Any, frame: Mapping[str, Any]
) -> dict[str, Any] | list[str]:
    """The refusal detail, or the keys the merge changed."""
    from local_operator.network.credentials import placement as placement_mod
    from local_operator.network.credentials.sync import (  # noqa: SLF001
        _member_is_active,
    )

    if not negotiated(link):
        return _refuse("unsupported", "the share push was not negotiated on this link")
    sender = str(getattr(link, "device_id", "") or "")
    claimed = frame.get("from_device")
    if not sender or sender == self_device or (claimed not in (None, "") and claimed != sender):
        return _refuse("identity_mismatch", "the push's sender is not the authenticated device")
    network_id = str(getattr(link, "network_id", "") or "")
    document = frame.get("document")
    if not isinstance(document, dict) or not network_id:
        return _refuse("malformed_frame", "the push carries no placement document")
    if (
        document.get("network_id") != network_id
        or document.get("schema") != placement_mod.PLACEMENT_SCHEMA
    ):
        return _refuse("identity_mismatch", "the push names another network")
    epoch = document.get("epoch")
    if type(epoch) is not int or epoch != 0:
        return _refuse("malformed_frame", "the push carries an epoch this feature never sends")
    rows = document.get("credentials")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_SHARE_ROWS:
        return _refuse("malformed_frame", "the push carries no rows or too many")
    try:
        if len(json.dumps(document, separators=(",", ":")).encode()) > MAX_SHARE_BYTES:
            raise ValueError("oversize")
        placement_mod._assert_no_material(document, where="pushed share")  # noqa: SLF001
    except (ValueError, TypeError, RecursionError):
        return _refuse("malformed_frame", "the push is oversized or carries material fields")
    for row in rows:
        holders = row.get("holders") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(holders, list):
            return _refuse("malformed_frame", "a pushed row is malformed")
        if row.get("owner_device") != sender:
            return _refuse("not_owner", "a pushed row is not one its sender owns")
        # THE NARROWING THAT MAKES THIS A SHARE DELIVERY, not a document sync: the row
        # must name THIS device as a holder. A push can therefore only tell a device
        # what it may borrow, never rewrite what it believes about anyone else.
        if not any(isinstance(h, dict) and h.get("device") == self_device for h in holders):
            return _refuse("not_a_holder", "a pushed row does not name this device as a holder")
    if not _member_is_active(root, network_id, sender):
        return _refuse("not_owner", "the sender is not an active member of this network")
    return placement_mod.merge_from_peer(
        network_id, document, from_device=sender, self_device=self_device, root=root
    )
