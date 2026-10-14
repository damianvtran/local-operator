"""Explicit signed recovery of ONE revoked finite share, never future consent.

The normal approval envelope is deliberately reused. Its operation-specific what
must be dispatched before onboarding/signing; neither an old onboarding approval
nor the bare share verb authorizes removing an owner-local tombstone.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import time
from pathlib import Path
from typing import Any

from local_operator.network import approvals, identity, onboard_approvals, store
from local_operator.network.credentials import consent, placement
from local_operator.network.types import MeshRefusal

OP = "credential_finite_share"
STEP = "credential_finite_share"


def intent(record: dict[str, Any]) -> dict[str, Any] | None:
    what = record.get("what")
    return what.get(OP) if isinstance(what, dict) and isinstance(what.get(OP), dict) else None


def summary(record: dict[str, Any]) -> str | None:
    data = intent(record)
    if data is None:
        return None
    return (
        f"Restore FINITE sharing of {data.get('key', '')!r} with "
        f"{data.get('recipient_device', '')} ({data.get('scope', '')}); "
        "clear only the listed withdrawals. No automatic future-sign-in inheritance "
        "enabled by this operation; no install, join or trust."
    )


def _reject_future_overlap(rec: dict[str, Any], key: str) -> None:
    if not rec.get("enabled"):
        return
    from local_operator.network.credentials.types import credential_key_for_mcp

    # Derive exact signed selector IDs/URLs using the existing trusted MCP key
    # constructor. Broad selectors cannot prove exclusion here; do NOT guess a
    # native class/account from a key's spelling or disable other permissions.
    selected: set[str] = set()
    for kind in ("providers", "mcp"):
        subset = rec["inherit"][kind]
        if not subset["future_logins"]:
            continue
        if "selector" in subset:
            raise MeshRefusal(
                "finite_future_scope_conflict",
                "This broad future-sign-in permission cannot prove this key excluded. "
                "Exact selector-exclusion recovery is not supported yet; no permission changed.",
            )
        if kind == "providers":
            selected.update(subset["ids"])
        else:
            selected.update(credential_key_for_mcp(url) for url in subset["urls"])
    if key in selected:
        raise MeshRefusal(
            "finite_future_scope_conflict",
            "Clearing this withdrawal would also reauthorize an existing future-sign-in selection. "
            "Exact selector-exclusion recovery is not supported yet; no permission changed.",
        )


def _audit(event: str, data: dict[str, Any], approval_id: str, root: Path | None) -> None:
    from local_operator.network.audit import AuditEvent, AuditLog

    log = AuditLog(root=root)
    try:
        log.record(
            AuditEvent(
                event=event,
                actor=data["owner_device"],
                subject=data["recipient_device"],
                detail={
                    "credential_key": data["key"],
                    "act": data["owner_device"],
                    "sub": data["recipient_device"],
                    "scope": data["scope"],
                    "approval_id": approval_id,
                },
            )
        )
    finally:
        log.close()


def _subject(
    network: str, owner: str, recipient: str, key: str, root: Path | None
) -> dict[str, Any]:
    for item in (network, owner, recipient):
        store.safe_component(item)
    own = identity.load(root)
    record = store.load(network, root)
    member = record.member(recipient)
    if (
        own is None
        or own.device_id != owner
        or record.self_device_id != owner
        or record.trust != "active"
        or member is None
        or not member.active
        or member.device_id != recipient
    ):
        consent.refuse("finite_subject_mismatch")
    from local_operator.network.relay import set_member_capabilities

    # Preview the existing capability policy on a COPY before authority writes.
    # A final normal CLI step cannot discover a refused admin source only after
    # a finite holder was restored and its tombstone removed.
    set_member_capabilities(copy.deepcopy(record), device_id=recipient, grant=["broker_credential"])
    public = base64.urlsafe_b64decode(member.public_key + "=" * (-len(member.public_key) % 4))
    if len(public) != 32:
        consent.refuse("scope_unknown")
    doc = placement.PlacementDocument.load(network, root, self_device=owner)
    entry = doc.entry(key)
    if entry is None or entry.owner_device != owner:
        consent.refuse("not_owner")
    return {
        "owner_device": owner,
        "network_id": network,
        "recipient_device": recipient,
        "public_key_fingerprint": "sha256:" + hashlib.sha256(public).hexdigest(),
        "key": key,
    }


def request(
    network: str, owner: str, recipient: str, key: str, scope: str, root: Path | None = None
) -> dict[str, Any]:
    subject = _subject(network, owner, recipient, key, root)
    if scope not in {"device", "session"} or (key == "github" and scope != "device"):
        consent.refuse("finite_scope_mismatch")
    authority = consent.ConsentDocument(network, owner, root)
    with authority.locked():
        row = authority.load()
        rec = row["recipients"].get(recipient, {})
        _reject_future_overlap(rec, key)
        denies = [
            copy.deepcopy(d)
            for d in rec.get("denies", [])
            if d["selector"]["credential_key"] == key
        ]
        if not denies:
            consent.refuse("finite_supersede_unneeded")
        subject.update(scope=scope, denies=denies, policy_revision=row["revision"])
    anchor = approvals.local_anchor_trio(root)
    if anchor is None:
        consent.refuse("approval_anchor_unavailable")
    now = time.time()
    # Current member metadata only: NO invented SSH host/target or new approval
    # kind. This envelope authorizes the exclusive finite operation, not setup.
    result = approvals.create_request(
        kind=approvals.KIND_DEVICE_ONBOARD,
        request_id=approvals.new_request_id(),
        requested_by={"device_id": owner, "session_id": "", "surface": "cli"},
        device={"device_id": recipient, "fingerprint": subject["public_key_fingerprint"]},
        what={
            OP: subject,
            "anchor": {k: anchor[k] for k in ("key_id", "spki_fp", "statement_digest")},
        },
        credential_ref=None,
        created_at=now,
        expires_at=now + 3600,
        root=root,
    )
    _audit("credential.finite_requested", subject, result["approval_id"], root)
    return result


def apply(approval_id: str, expected: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    source = approvals.load_record(approval_id, root=root)
    if source.get("state") == approvals.STATE_CONNECTED:
        raise MeshRefusal(
            "approval_already_connected",
            "This finite key/scope approval already finished; "
            "replay cannot clear a later withdrawal. "
            "Request a new exact finite-share approval; no install or connection ran.",
        )
    data = intent(source)
    if data is None or set(source.get("what", {})) != {OP, "anchor"}:
        consent.refuse("finite_intent_missing")
    consent._exact(
        data,
        {
            "owner_device",
            "network_id",
            "recipient_device",
            "public_key_fingerprint",
            "key",
            "scope",
            "denies",
            "policy_revision",
        },
        {
            "owner_device",
            "network_id",
            "recipient_device",
            "public_key_fingerprint",
            "key",
            "scope",
            "denies",
            "policy_revision",
        },
    )
    for field in ("owner_device", "network_id", "recipient_device", "key", "scope"):
        if data[field] != expected.get(field):
            consent.refuse("finite_subject_mismatch")
    current = _subject(
        data["network_id"], data["owner_device"], data["recipient_device"], data["key"], root
    )
    if current["public_key_fingerprint"] != data["public_key_fingerprint"]:
        consent.refuse("finite_subject_mismatch")
    run = onboard_approvals.begin_run(approval_id, root=root, operation=OP)
    authority = consent.ConsentDocument(data["network_id"], data["owner_device"], root)
    projection_path = placement.placement_path(data["network_id"], root)
    # Writers below are existing factored LOCKED entrypoints. No await/network,
    # SSH/onboarding/copy work or reentrant record flock occurs in this transaction.
    with (
        approvals._record_lock(approval_id, root),
        authority.locked(),
        placement._document_lock(projection_path),
    ):
        admitted = consent._admitted(approval_id, run.run_id, root)
        fresh = _subject(
            data["network_id"], data["owner_device"], data["recipient_device"], data["key"], root
        )
        if fresh != current:
            consent.refuse("finite_subject_mismatch")
        row = authority.load()
        if row["revision"] != data["policy_revision"]:
            consent.refuse("consent_conflict")
        rec = row["recipients"].get(data["recipient_device"])
        _reject_future_overlap(rec or {}, data["key"])
        actual = (
            [d for d in rec.get("denies", []) if d["selector"]["credential_key"] == data["key"]]
            if rec
            else []
        )
        if not actual or actual != data["denies"]:
            consent.refuse("finite_denial_mismatch")
        old_row = copy.deepcopy(row)
        projection = placement.PlacementDocument.load(
            data["network_id"], root, self_device=data["owner_device"]
        )
        old_projection = store._read_json(projection_path)
        try:
            # Keep deny effective while the finite projection/execution proof is
            # committed. Authority removal is LAST; no unverified receipt can
            # leave an accessible stale holder. Other keys' tombstones survive.
            entry = projection.grant(
                data["key"], data["recipient_device"], scope=data["scope"], by=data["owner_device"]
            )
            projection.save()
            checked = placement.PlacementDocument.load(data["network_id"], root).entry(data["key"])
            if checked is None or not checked.is_holder(data["recipient_device"]):
                consent.refuse("finite_write_unconfirmed")
            consent._admitted(approval_id, run.run_id, root)
            approvals._append_receipt_locked(
                approval_id,
                run_id=run.run_id,
                step=STEP,
                ok=True,
                detail="EXECUTED: exact finite-share permission; no future inheritance",
                digest=admitted["request_digest"],
                root=root,
            )
            approvals._mark_connected_locked(
                approval_id,
                run_id=run.run_id,
                step=STEP,
                detail=(
                    "Finite projection confirmed; apply must confirm withdrawal removal; "
                    "no automatic future-sign-in inheritance enabled"
                ),
                root=root,
            )
            if time.time() >= float(admitted["expires_at"]):
                consent.refuse("approval_expired")
            if (
                _subject(
                    data["network_id"],
                    data["owner_device"],
                    data["recipient_device"],
                    data["key"],
                    root,
                )
                != current
            ):
                consent.refuse("finite_subject_mismatch")
            rec["denies"] = [d for d in rec["denies"] if d not in actual]
            authority.save(row, data["policy_revision"])
            result = entry.to_json()
        except (MeshRefusal, OSError, ValueError):
            # Original valid tombstones are conservative even if restoring the
            # projection fails; do not report an incomplete prepared operation
            # as success or drop denial to fit bounds. CAS stays forward-only.
            try:
                now_row = authority.load()
                old_row["revision"] = now_row["revision"]
                authority.save(old_row, now_row["revision"])
            except (MeshRefusal, OSError, ValueError):
                # If the policy filesystem cannot accept compensation after its
                # last publish, persist a WITHHELD projection instead. Never
                # restore a formerly eligible holder with unconfirmed denies.
                withheld = placement.PlacementDocument.load(data["network_id"], root)
                withheld.revoke(data["key"], data["recipient_device"], by=data["owner_device"])
                withheld.save()
                checked = placement.PlacementDocument.load(data["network_id"], root).entry(
                    data["key"]
                )
                if checked is not None and checked.is_holder(data["recipient_device"]):
                    consent.refuse("finite_compensation_unconfirmed")
                consent.refuse("finite_projection_withheld")
            try:
                # Rollback is restoration, not a newly observed projection:
                # preserve its prior timestamp/revision and full document under
                # the existing private atomic writer while its flock is held.
                store._write_private_json(projection_path, old_projection)
                if store._read_json(projection_path) != old_projection:
                    consent.refuse("finite_projection_withheld")
            except OSError:
                # Persisted original deny still withholds any partial holder.
                consent.refuse("finite_projection_withheld")
            consent.refuse("finite_recovery_failed")
    _audit("credential.finite_applied", data, approval_id, root)
    return result
