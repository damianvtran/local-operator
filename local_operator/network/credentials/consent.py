"""Owner-local inheritance authority; never a peer-merged placement document.

A finite holder and broker capability are NOT future-login consent. This first
slice records explicit executed permission and durable reductions only: it never
projects holders, exports material or advertises a lifecycle protocol. Old serving
processes ignore this file; live activation therefore requires deny-aware owner
broker/copy reader adoption, not merely a new CLI or runtime pointer.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn

from local_operator.network import store
from local_operator.network.credentials import placement
from local_operator.network.types import MeshRefusal

SCHEMA = 1
MAX_REVISION = 2**53 - 2**20 - 1
NETWORK = re.compile(r"n_[0-9a-f]{24}\Z")
DEVICE = re.compile(r"d_[0-9a-f]{32}\Z")
REFERENCE = re.compile(r"[A-Za-z0-9_.:/-]{1,256}\Z")
PROVIDER = re.compile(r"[a-z][a-z0-9_-]{0,199}\Z")
STEP = "credential_inheritance"


def refuse(code: str) -> NoReturn:
    # Never include rejected source/frame contents or material in the refusal.
    raise MeshRefusal(code, "credential permission refused: " + code)


def _exact(value: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        refuse("consent_invalid")
    return value


def _revision(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_REVISION:
        refuse("consent_revision_invalid")
    return value


def _ref(value: Any) -> str:
    if not isinstance(value, str) or REFERENCE.fullmatch(value) is None:
        refuse("scope_unknown")
    return value


def _selector(value: Any) -> dict[str, Any]:
    row = _exact(value, {"credential_key", "account_ref"}, {"credential_key"})
    key = row["credential_key"]
    if not isinstance(key, str) or not 1 <= len(key.encode()) <= 200:
        refuse("consent_invalid")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in key):
        refuse("consent_invalid")
    # Existing finite secret copies may be reduced, never positively inherited.
    if not (PROVIDER.fullmatch(key) or key.startswith(("mcp:", "secret:"))):
        refuse("consent_invalid")
    if "account_ref" in row:
        _ref(row["account_ref"])
    return row


def _inherit(value: Any) -> dict[str, Any]:
    row = _exact(
        value,
        {"providers", "mcp", "origin", "scope_ref"},
        {"providers", "mcp", "origin", "scope_ref"},
    )
    if row["origin"] != "owner-native-login":
        refuse("consent_origin_refused")
    _ref(row["scope_ref"])
    for kind, broad, finite in (
        ("providers", "eligible-owner-native", "ids"),
        ("mcp", "approved-owner-servers", "urls"),
    ):
        subset = _exact(row[kind], {"selector", finite, "future_logins"}, {"future_logins"})
        if type(subset["future_logins"]) is not bool:
            refuse("consent_invalid")
        if ("selector" in subset) == (finite in subset):
            refuse("consent_invalid")
        if "selector" in subset:
            if subset["selector"] != broad:
                refuse("consent_invalid")
        else:
            values = subset[finite]
            if (
                not isinstance(values, list)
                or len(values) > 8
                or any(not isinstance(item, str) for item in values)
                or len(set(values)) != len(values)
            ):
                refuse("consent_invalid")
            for item in values:
                if not isinstance(item, str):
                    refuse("consent_invalid")
                if kind == "providers" and PROVIDER.fullmatch(item) is None:
                    refuse("consent_invalid")
                if kind == "mcp":
                    from urllib.parse import urlsplit

                    from local_operator.network.credentials.types import (
                        credential_key_for_mcp,
                    )

                    url = urlsplit(item)
                    if (
                        url.scheme not in {"http", "https"}
                        or not url.hostname
                        or url.username
                        or url.password
                        or url.fragment
                        or len(credential_key_for_mcp(item).encode()) > 200
                    ):
                        refuse("consent_invalid")
    return row


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            refuse("consent_invalid")
        result[key] = value
    return result


class ConsentDocument:
    """Bounded, owner-only metadata. Readers never create directories/files.

    Lock order is consent -> placement. Never call this from a held placement
    mutation, and never hold the lock across an upstream refresh/network await.
    A restriction-only recipient has no fabricated positive approval or scope.
    """

    def __init__(self, network_id: str, owner_device: str, root: Path | None = None):
        if NETWORK.fullmatch(network_id) is None or DEVICE.fullmatch(owner_device) is None:
            refuse("consent_identity_invalid")
        self.network_id = network_id
        self.owner_device = owner_device
        self.root = root
        self.path = placement.placement_path(network_id, root).with_name("consent.json")
        if self.path.parent.is_symlink() or self.path.is_symlink():
            refuse("consent_path_invalid")

    @contextmanager
    def locked(self) -> Iterator[None]:
        placement.credentials_dir_for(self.network_id, self.root)
        with store._write_lock(self.path):
            with placement._cross_process_lock(self.path.with_name(".consent.lock")):
                yield

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema": SCHEMA,
                "network_id": self.network_id,
                "owner_device": self.owner_device,
                "revision": 1,
                "recipients": {},
            }
        try:
            raw = self.path.read_bytes()
            if len(raw) > 262144:
                refuse("consent_invalid")
            data = json.loads(
                raw, object_pairs_hook=_pairs, parse_constant=lambda _: refuse("consent_invalid")
            )
        except (ValueError, OSError):
            refuse("consent_invalid")
        row = _exact(
            data,
            {"schema", "network_id", "owner_device", "revision", "recipients"},
            {"schema", "network_id", "owner_device", "revision", "recipients"},
        )
        if type(row["schema"]) is not int or row["schema"] != SCHEMA:
            refuse("consent_unsupported")
        if row["network_id"] != self.network_id or row["owner_device"] != self.owner_device:
            refuse("consent_identity_invalid")
        _revision(row["revision"])
        recipients = row["recipients"]
        if not isinstance(recipients, dict) or len(recipients) > 256:
            refuse("consent_invalid")
        for device, recipient in recipients.items():
            if DEVICE.fullmatch(device) is None:
                refuse("consent_identity_invalid")
            rec = _exact(
                recipient,
                {"public_key_fingerprint", "enabled", "approval", "inherit", "denies"},
                {"public_key_fingerprint", "enabled", "denies"},
            )
            if type(rec["enabled"]) is not bool or not isinstance(
                rec["public_key_fingerprint"], str
            ):
                refuse("consent_invalid")
            if rec["enabled"] or "approval" in rec or "inherit" in rec:
                approval = _exact(
                    rec.get("approval"),
                    {"kind", "ref", "approved_at", "approved_by"},
                    {"kind", "ref", "approved_at", "approved_by"},
                )
                if approval["kind"] != "operator-request":
                    refuse("consent_invalid")
                _ref(approval["ref"])
                _ref(approval["approved_by"])
                if type(approval["approved_at"]) not in (int, float) or not math.isfinite(
                    approval["approved_at"]
                ):
                    refuse("consent_invalid")
                _inherit(rec.get("inherit"))
            if not isinstance(rec["denies"], list) or len(rec["denies"]) > 256:
                refuse("consent_invalid")
            for denial in rec["denies"]:
                deny = _exact(
                    denial,
                    {"selector", "revision", "reason", "approval_ref"},
                    {"selector", "revision", "reason", "approval_ref"},
                )
                _selector(deny["selector"])
                _revision(deny["revision"])
                _ref(deny["reason"])
                _ref(deny["approval_ref"])
        return row

    def save(self, row: dict[str, Any], expected: int) -> None:
        current = self.load()
        if current["revision"] != expected:
            refuse("consent_conflict")
        row["revision"] = _revision(expected + 1)
        store._write_private_json(self.path, row)
        # The shared writer fsyncs the staged file; authority also needs the
        # renamed directory entry durable before placement can be reduced.
        import os

        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # Read back through the same strict reader; a write is not evidence of
        # successful authority publication until the actual persisted shape holds.
        if self.load() != row:
            refuse("consent_write_unconfirmed")


def check_borrow(
    network_id: str,
    owner: str,
    recipient: str,
    key: str,
    native_context: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> None:
    """Negative overlay only. Positive future policy cannot mint/rehome finite grants.

    None context is the pre-resolve provider-wide check; an empty context at the
    final boundary is UNKNOWN account identity, not provider-confirmed N/A.
    """
    path = placement.placement_path(network_id, root).with_name("consent.json")
    if not path.exists():
        return
    doc = ConsentDocument(network_id, owner, root)
    rec = doc.load()["recipients"].get(recipient)
    if rec is None:
        return
    for denial in rec["denies"]:
        selector = denial["selector"]
        if selector["credential_key"] != key:
            continue
        account = selector.get("account_ref")
        if account is None:
            refuse("not_a_holder")
        if native_context is None:
            continue
        actual = native_context.get("account_id")
        if not isinstance(actual, str) or not actual:
            refuse("scope_unknown")
        if actual == account:
            refuse("not_a_holder")


@contextmanager
def serving_guard(
    network_id: str,
    owner: str,
    recipient: str,
    key: str,
    context: Mapping[str, Any],
    root: Path | None = None,
) -> Iterator[None]:
    """Serialize final deny check/mint or local copy extraction, never refresh I/O."""
    path = placement.placement_path(network_id, root).with_name("consent.json")
    if not path.exists():
        # The lock is still needed: first revoke may race an initially absent file.
        # Invalid legacy fixture IDs cannot name a new consent document.
        if NETWORK.fullmatch(network_id) is None or DEVICE.fullmatch(owner) is None:
            yield
            return
    doc = ConsentDocument(network_id, owner, root)
    with doc.locked():
        check_borrow(network_id, owner, recipient, key, context, root)
        yield


def restrict_and_revoke(
    network_id: str,
    owner: str,
    recipient: str,
    selector: dict[str, Any],
    source_ref: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Durable owner reduction FIRST; stale placement cannot undo this tombstone."""
    _selector(selector)
    _ref(source_ref)
    if DEVICE.fullmatch(recipient) is None:
        refuse("consent_identity_invalid")
    doc = ConsentDocument(network_id, owner, root)
    with doc.locked():
        # Ownership must be decided before any authority write. This read does
        # not acquire placement in reverse; the later mutation keeps the single
        # consent -> placement order and verifies ownership again.
        present = placement.PlacementDocument.load(network_id, root, self_device=owner).entry(
            selector["credential_key"]
        )
        if present is not None and present.owner_device != owner:
            refuse("not_owner")
        row = doc.load()
        rec = row["recipients"].setdefault(
            recipient, {"public_key_fingerprint": "", "enabled": False, "denies": []}
        )
        duplicate = any(
            d["selector"] == selector and d["approval_ref"] == source_ref for d in rec["denies"]
        )
        if not duplicate:
            rec["denies"].append(
                {
                    "selector": copy.deepcopy(selector),
                    "revision": row["revision"] + 1,
                    "reason": "withdrawn",
                    "approval_ref": source_ref,
                }
            )
            doc.save(row, row["revision"])
        with placement.mutate(network_id, self_device=owner, root=root) as projection:
            entry = projection.entry(selector["credential_key"])
            if entry is not None and entry.owner_device != owner:
                refuse("not_owner")
            if entry is not None:
                entry = projection.revoke(selector["credential_key"], recipient, by=owner)
            return {
                "kind": "restricted",
                "revision": row["revision"],
                "entry": entry.to_json() if entry is not None else None,
            }


def _admitted(approval_id: str, run_id: str, root: Path | None) -> dict[str, Any]:
    from local_operator.network import onboard_approvals

    view = onboard_approvals.require_step_allowed(approval_id, run_id=run_id, root=root)
    record = view.record
    run = record.get("run")
    if view.state != "connecting" or not isinstance(run, dict) or run.get("run_id") != run_id:
        refuse("consent_run_stale")
    return record


def resolve_approved_scope(
    record: Mapping[str, Any], subject: Mapping[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """Reference ONLY immutable signed what + executed receipt, not identity inference.

    The operator key proves the approving principal, not provider-person/tenant
    ownership. Explicit account/org constraints match owner-selected native fields;
    required person context is unknown unless separately supplied/proven. There is
    no universal person/tenant resolver hidden behind this opaque scope reference.
    """
    from local_operator.network import approvals, identity

    what = record.get("what")
    raw = what.get("credential_inheritance") if isinstance(what, dict) else None
    data = _exact(
        raw,
        {
            "action",
            "owner_device",
            "network_id",
            "recipient_device",
            "public_key_fingerprint",
            "inherit",
            "scope",
            "supersedes",
        },
        {
            "action",
            "owner_device",
            "network_id",
            "recipient_device",
            "public_key_fingerprint",
            "inherit",
            "scope",
        },
    )
    if data["action"] not in {"grant", "supersede"}:
        refuse("consent_action_refused")
    for field in ("owner_device", "network_id", "recipient_device", "public_key_fingerprint"):
        if data[field] != subject.get(field):
            refuse("consent_subject_mismatch")
    owner = identity.load(root)
    if owner is None or owner.device_id != data["owner_device"]:
        refuse("consent_subject_mismatch")
    network = store.load(data["network_id"], root)
    if network.trust != "active" or network.self_device_id != data["owner_device"]:
        refuse("consent_subject_mismatch")
    member = network.member(data["recipient_device"])
    if (
        member is None
        or member.lifecycle != "active"
        or member.removed_at
        or member.device_id != data["recipient_device"]
    ):
        refuse("consent_recipient_inactive")
    try:
        public = base64.urlsafe_b64decode(member.public_key + "=" * (-len(member.public_key) % 4))
    except (ValueError, TypeError):
        refuse("consent_subject_mismatch")
    fingerprint = "sha256:" + hashlib.sha256(public).hexdigest()
    if len(public) != 32 or fingerprint != data["public_key_fingerprint"]:
        refuse("consent_subject_mismatch")
    anchor = approvals.local_anchor_trio(root)
    scope = _exact(
        data["scope"],
        {"principal_ref", "account_id", "org_id", "person_ref", "not_applicable"},
        {"principal_ref"},
    )
    if anchor is None or scope["principal_ref"] != anchor["key_id"]:
        refuse("scope_unknown")
    for field in ("account_id", "org_id", "person_ref"):
        if field in scope:
            _ref(scope[field])
    na = scope.get("not_applicable", [])
    if (
        not isinstance(na, list)
        or any(x not in {"account_id", "org_id", "person_ref"} for x in na)
        or any(x in scope for x in na)
    ):
        refuse("scope_unknown")
    # A signed not-applicable expectation is not itself provider confirmation.
    # Actual native context must confirm it when a future inherited serve exists.
    # Internal owner adapter supplies native-selected context. It is not a peer
    # account_ref, a device label, an email or a guessed principal. Until genuine
    # context exists, required constraints cannot be enabled even as metadata.
    context = subject.get("native_context")
    if not isinstance(context, Mapping):
        context = {}
    if (
        context.get("local_only") is True
        or context.get("origin", "owner-native-login") != "owner-native-login"
    ):
        refuse("consent_origin_refused")
    for field in ("account_id", "org_id", "person_ref"):
        if field in scope:
            if not isinstance(context.get(field), str) or not context[field]:
                refuse("scope_unknown")
            if context[field] != scope[field]:
                refuse("scope_mismatch")
    confirmed_na = context.get("provider_confirmed_not_applicable", [])
    if any(field not in confirmed_na for field in na):
        refuse("scope_unknown")
    inherit = copy.deepcopy(data["inherit"])
    inherit["scope_ref"] = "approval:" + str(record["approval_id"]) + ":credential-scope"
    _inherit(inherit)
    return {"source": data, "scope": scope, "inherit": inherit, "principal": anchor["key_id"]}


def provision_from_run(
    approval_id: str, run_id: str, expected_subject: Mapping[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """Prepare disabled -> verified EXECUTED receipt -> re-admit -> CAS enable.

    No positive authority is visible before execution evidence. Interruption leaves
    a truthful disabled prepared record; replay never drops a sticky deny. This is
    metadata-only and creates no holder, capability or raw/static material copy.
    """
    from local_operator.network import approvals, onboard_approvals

    record = _admitted(approval_id, run_id, root)
    proof = resolve_approved_scope(record, expected_subject, root)
    data = proof["source"]
    doc = ConsentDocument(data["network_id"], data["owner_device"], root)
    recipient = data["recipient_device"]
    source = {
        "kind": "operator-request",
        "ref": approval_id,
        "approved_at": record["decided_at"],
        "approved_by": proof["principal"],
    }
    with doc.locked():
        row = doc.load()
        existing = row["recipients"].get(recipient, {})
        # Same request/run is idempotent; another signed snapshot cannot silently
        # replace restrictions or reinterpret old finite pairing shares.
        if (
            existing.get("approval") == source
            and existing.get("inherit") == proof["inherit"]
            and existing.get("enabled") is True
        ):
            executed = [
                r
                for r in record.get("receipts", [])
                if r.get("step") == STEP
                and r.get("run_id") == run_id
                and r.get("ok") is True
                and r.get("digest") == record["request_digest"]
            ]
            if not executed:
                refuse("consent_execution_missing")
            return {
                "kind": "permission_recorded",
                "revision": row["revision"],
                "activation": "static_storage_activation_pending",
            }
        denies = copy.deepcopy(existing.get("denies", []))
        if data["action"] == "supersede":
            supersedes = data.get("supersedes")
            if not isinstance(supersedes, list) or not supersedes:
                refuse("consent_action_refused")
            # Only fresh signed exact withdrawal references can be superseded;
            # a replay of the source that originally granted cannot clear one.
            for item in supersedes:
                _exact(
                    item,
                    {"selector", "revision", "approval_ref"},
                    {"selector", "revision", "approval_ref"},
                )
                _selector(item["selector"])
                _revision(item["revision"])
                _ref(item["approval_ref"])
                matching = [d for d in denies if all(d[k] == item[k] for k in item)]
                if not matching or item["approval_ref"] == approval_id:
                    refuse("consent_action_refused")
                selected = item["selector"]
                key = selected["credential_key"]
                if key.startswith("mcp:"):
                    subset = proof["inherit"]["mcp"]
                    allowed = subset.get("selector") == "approved-owner-servers" or key[
                        4:
                    ] in subset.get("urls", [])
                else:
                    subset = proof["inherit"]["providers"]
                    allowed = PROVIDER.fullmatch(key) is not None and (
                        subset.get("selector") == "eligible-owner-native"
                        or key in subset.get("ids", [])
                    )
                if not allowed or (
                    selected.get("account_ref") is not None
                    and selected["account_ref"] != proof["scope"].get("account_id")
                ):
                    refuse("scope_mismatch")
        elif data.get("supersedes"):
            refuse("consent_action_refused")
        row["recipients"][recipient] = {
            "public_key_fingerprint": data["public_key_fingerprint"],
            "enabled": False,
            "approval": source,
            "inherit": proof["inherit"],
            "denies": denies,
        }
        _admitted(approval_id, run_id, root)
        doc.save(row, row["revision"])
        prepared = row["revision"]
    try:
        _admitted(approval_id, run_id, root)
        onboard_approvals.append_receipt(
            approval_id,
            run_id,
            {
                "step": STEP,
                "ok": True,
                "detail": "EXECUTED: scoped permission prepared; no placement/copy",
                "digest": record["request_digest"],
            },
            root=root,
        )
        latest = _admitted(approval_id, run_id, root)
        if not any(
            r.get("step") == STEP
            and r.get("run_id") == run_id
            and r.get("ok") is True
            and r.get("digest") == record["request_digest"]
            for r in latest.get("receipts", [])
        ):
            refuse("consent_execution_missing")
        # Source cancellation and final enable share the approval record lock.
        # Order is approval -> consent -> placement (no placement in this step).
        # Append ran before this block, so the existing non-reentrant record flock
        # is never reacquired by a writer. Only metadata/public verification runs.
        with approvals._record_lock(approval_id, root):
            with doc.locked():
                row = doc.load()
                if row["revision"] != prepared:
                    refuse("consent_conflict")
                _admitted(approval_id, run_id, root)
                rec = row["recipients"][recipient]
                if data["action"] == "supersede":
                    rec["denies"] = [
                        d
                        for d in rec["denies"]
                        if not any(
                            all(d[k] == item[k] for k in item) for item in data["supersedes"]
                        )
                    ]
                resolve_approved_scope(_admitted(approval_id, run_id, root), expected_subject, root)
                rec["enabled"] = True
                doc.save(row, prepared)
                _admitted(approval_id, run_id, root)
    except (MeshRefusal, OSError):
        with doc.locked():
            row = doc.load()
            rec = row["recipients"].get(recipient)
            if rec is not None and rec.get("approval") == source and rec.get("enabled") is True:
                rec["enabled"] = False
                # Compensate a failed supersession too: the denial survives until
                # a successful fresh action completes, not just its receipt append.
                rec["denies"] = list(
                    {json.dumps(d, sort_keys=True): d for d in rec["denies"] + denies}.values()
                )
                doc.save(row, row["revision"])
        refuse("consent_prepared_pending")
    if doc.load()["recipients"][recipient]["enabled"] is not True:
        refuse("consent_write_unconfirmed")
    return {
        "kind": "permission_recorded",
        "revision": row["revision"],
        "activation": "static_storage_activation_pending",
        "serving_adoption": "deny_aware_owner_generation_required",
    }


def verified_permission(
    network_id: str,
    owner: str,
    recipient: str,
    key: str,
    native_context: Mapping[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    """Read durable executed intent, NOT the onboarding execution-window lease.

    This is authority metadata only for the later reviewed reconciler. It creates
    no holder/capability/grant/copy. Initial run completion/window expiry does not
    revoke indefinite future permission; explicit current denial still wins.
    """
    from local_operator.network import approvals

    doc = ConsentDocument(network_id, owner, root)
    row = doc.load()
    rec = row["recipients"].get(recipient)
    if rec is None or rec["enabled"] is not True or "approval" not in rec:
        refuse("consent_not_enabled")
    check_borrow(network_id, owner, recipient, key, native_context, root)
    source_id = rec["approval"]["ref"]
    # This existing helper verifies immutable intent/signature/digest and denial,
    # but (intentionally here) does not enforce a new execution window. No begin
    # run, require_step_allowed, execution replay or invented permission TTL.
    record = approvals.verify_for_run(source_id, root=root)
    if not any(
        r.get("step") == STEP
        and r.get("ok") is True
        and r.get("digest") == record["request_digest"]
        and str(r.get("detail", "")).startswith("EXECUTED: scoped permission prepared")
        for r in record.get("receipts", [])
    ):
        refuse("consent_execution_missing")
    subject = {
        "owner_device": owner,
        "network_id": network_id,
        "recipient_device": recipient,
        "public_key_fingerprint": rec["public_key_fingerprint"],
        "native_context": native_context,
    }
    proof = resolve_approved_scope(record, subject, root)
    if (
        proof["inherit"] != rec["inherit"]
        or rec["approval"]["approved_by"] != proof["principal"]
        or rec["approval"]["approved_at"] != record["decided_at"]
    ):
        refuse("consent_source_mismatch")
    if key.startswith("mcp:"):
        subset = rec["inherit"]["mcp"]
        selected = subset.get("selector") == "approved-owner-servers" or key[4:] in subset.get(
            "urls", []
        )
    else:
        subset = rec["inherit"]["providers"]
        selected = PROVIDER.fullmatch(key) is not None and (
            subset.get("selector") == "eligible-owner-native" or key in subset.get("ids", [])
        )
    if not selected or subset["future_logins"] is not True:
        refuse("consent_subset_refused")
    # Caller must recheck this revision before any later positive projection.
    # PR1 deliberately does not implement such projection or static activation.
    return {
        "kind": "permission_only",
        "revision": row["revision"],
        "scope": proof["scope"],
        "activation": "static_storage_activation_pending",
        "serving_adoption": "deny_aware_owner_generation_required",
    }
