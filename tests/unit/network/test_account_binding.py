"""P1 adapter contracts: SQLite/signatures real; account/presence/link synthetic.

The imported rig never opens a relay socket. These tests prove purpose-limited
binding and fail-closed inputs, not native attestation or P3 byte-release safety.
The separate process fixture owns real authenticated two-relay listener evidence.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import pickle
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from local_operator.network import account_binding as B
from local_operator.network import file_device_approval as F
from local_operator.network import store
from tests.unit.network import test_file_device_approval as approval_tests
from tests.unit.network.test_file_device_approval import activate, request

# Reuse the exact existing fixture rather than a second consent/auth harness.
rig = approval_tests.rig


def _read(rig, **changes):
    fields = {
        "owner_key": F.MeshKey(device_id=rig.own.device_id, public_key=rig.own.public_key),
        "network_id": rig.network.network_id,
        "network_epoch": rig.network.epoch,
        "link_id": rig.link.link_id,
        "transcript_hash": rig.link.result.transcript_hash,
        "request_id": "req_synthetic_read",
        "session_id": "synthetic-owned-session",
        "source_row_id": "synthetic-transcript-row",
        "provenance_id": "synthetic-provenance-id",
        "proof_digest": "sha256:" + "a" * 64,
        "file_version": "synthetic-sealed-version-not-established-here",
        "phase": "fetch",
        "offset": 0,
        "length": 17,
        "transfer_nonce": "a" * 32,
    }
    return B.ReadBinding(**{**fields, **changes})


def _active(rig):
    record = request(rig)
    activate(rig, record)
    return B.OwnerFileEvidenceReference(association=F._association(F._intent(record)))


def _verify(rig, reference, **changes):
    args = {
        "incoming_link": rig.link,
        "auth": rig.auth,
        "owner_account": B.ExactNativeAccountSelection(credential_id=rig.row.id),
        "evidence": reference,
        "required_subject": B.CanonicalSubject(
            issuer=F.DEFAULT_RADIENT_API_BASE_URL, opaque_id="account-a"
        ),
        "required_purpose": F.PURPOSE,
        "read_binding": _read(rig),
    }
    return asyncio.run(B.verify_live(**{**args, **changes}))


def test_existing_consent_yields_frozen_exact_binding_without_another_prompt(rig):
    evidence = _active(rig)
    signs, accounts = rig.calls["sign"], len(rig.calls["accounts"])
    verified = _verify(rig, evidence)
    assert type(verified) is B.VerifiedAccountDeviceBinding
    assert rig.calls["sign"] == signs
    assert len(rig.calls["accounts"]) == accounts + 1
    assert verified.subject == B.CanonicalSubject(
        issuer=F.DEFAULT_RADIENT_API_BASE_URL, opaque_id="account-a"
    )
    assert verified.requester_key.public_key == rig.link.result.peer_public_key
    assert verified.owner_key.public_key == rig.own.public_key
    assert verified.native_account.login_incarnation == rig.auth.native_login_incarnation(
        rig.row.id
    )
    assert verified.association == evidence.association
    assert verified.purpose == F.PURPOSE and verified.assurance == "installed_presence"
    assert verified.journal_revision >= verified.generation == 1
    assert verified.deny_generation == 0
    assert verified.read_binding == _read(rig)
    verified.require_read(rig.link, _read(rig))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(verified, "generation", 2)
    with pytest.raises(ValidationError):
        verified.requester_key.public_key = rig.own.public_key
    for operation in (
        lambda: bool(verified),
        lambda: pickle.dumps(verified),
        B.VerifiedAccountDeviceBinding,
    ):
        with pytest.raises(TypeError):
            operation()
    with pytest.raises(TypeError):
        dataclasses.replace(verified, generation=2)


def test_legacy_projection_and_internal_snapshot_share_one_verifier(rig, monkeypatch):
    _active(rig)
    snapshot = asyncio.run(F._verified_context(rig.link, rig.auth, rig.row.id))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snapshot, "journal_revision", 0)
    with pytest.raises(ValidationError):
        snapshot.intent.account.account_id = "account-b"
    observed = []

    async def checked(link, auth, credential_id):
        observed.append((link, auth, credential_id))
        return snapshot

    monkeypatch.setattr(F, "_verified_context", checked)
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id)) == {
        "association": snapshot.association,
        "authority_epoch": snapshot.authority_epoch,
        "generation": snapshot.generation,
        "request_digest": snapshot.request_digest,
    }
    assert observed == [(rig.link, rig.auth, rig.row.id)]


READ_CHANGES = [
    ("owner_key", "peer"),
    ("network_id", "different-network"),
    ("network_epoch", 2),
    ("link_id", "different-link"),
    ("transcript_hash", "b" * 64),
    ("request_id", "req_different"),
    ("session_id", "different-session"),
    ("source_row_id", "different-row"),
    ("provenance_id", "different-proof"),
    ("proof_digest", "sha256:" + "b" * 64),
    ("file_version", "different-version"),
    ("phase", "verify"),
    ("offset", 1),
    ("length", 18),
    ("transfer_nonce", "b" * 32),
]


@pytest.mark.parametrize("field,value", READ_CHANGES)
def test_receipt_cannot_cross_any_read_binding_field(rig, field, value):
    verified = _verify(rig, _active(rig))
    if field == "owner_key":
        value = F.MeshKey(device_id=rig.peer.device_id, public_key=rig.peer.public_key)
    with pytest.raises(B.BindingRefusal, match="account_binding_read_mismatch"):
        verified.require_read(rig.link, _read(rig, **{field: value}))
    with pytest.raises(B.BindingRefusal):
        verified.require_read(copy.copy(rig.link), _read(rig))


@pytest.mark.parametrize("field,value", READ_CHANGES[:5])
def test_verifier_refuses_read_identity_not_matching_actual_listener(rig, field, value):
    evidence = _active(rig)
    if field == "owner_key":
        value = F.MeshKey(device_id=rig.peer.device_id, public_key=rig.peer.public_key)
    with pytest.raises(B.BindingRefusal, match="account_binding_read_mismatch"):
        _verify(rig, evidence, read_binding=_read(rig, **{field: value}))


@pytest.mark.parametrize(
    "field,value", [("issuer", "https://other.invalid"), ("opaque_id", "account-b")]
)
def test_required_subject_is_only_a_constraint_not_cached_authority(rig, field, value):
    evidence = _active(rig)
    subject = {"issuer": F.DEFAULT_RADIENT_API_BASE_URL, "opaque_id": "account-a", field: value}
    with pytest.raises(B.BindingRefusal, match="account_binding_subject_mismatch"):
        _verify(rig, evidence, required_subject=B.CanonicalSubject(**subject))


@pytest.mark.parametrize("failure", ["mismatch", "offline"])
def test_every_read_requires_fresh_canonical_account(rig, failure):
    evidence = _active(rig)
    previous = _verify(rig, evidence)
    if failure == "mismatch":
        rig.observed["account"] = "account-b"
    else:
        rig.observed["error"] = httpx.ConnectError("synthetic-offline")
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence, required_subject=previous.subject)


@pytest.mark.parametrize(
    "change",
    [
        "epoch",
        "full-key",
        "owner-key",
        "requester-admission",
        "owner-admission",
        "removed",
        "suspect",
        "no-view",
        "untrusted",
    ],
)
def test_current_mesh_context_and_admission_are_not_cached(rig, change):
    evidence = _active(rig)
    _verify(rig, evidence)
    if change == "epoch":
        rig.network.epoch += 1
    elif change == "full-key":
        # Keep the device ID/label; the actual FULL authenticated key must match.
        rig.network.members[1].public_key = rig.own.public_key
    elif change == "owner-key":
        rig.server.identity = SimpleNamespace(public_key=rig.peer.public_key)
    elif change.endswith("admission"):
        rig.network.members[0 if change.startswith("owner") else 1].added_at += 1
    elif change == "removed":
        rig.network.removed_ids.append(rig.peer.device_id)
    elif change == "suspect":
        rig.network.members[1].suspect = True
    elif change == "no-view":
        rig.network.members[1].capabilities = []
    else:
        rig.network.trust = "untrusted"
    store.save(rig.network, rig.root)
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)


@pytest.mark.parametrize(
    "change", ["disabled", "incarnation", "missing", "account", "hub", "mesh-origin"]
)
def test_exact_native_account_context_is_rechecked(rig, change):
    evidence = _active(rig)
    _verify(rig, evidence)
    if change == "missing":
        rig.auth._conn.execute("DELETE FROM auth_credentials WHERE id = ?", (rig.row.id,))
    elif change in {"disabled", "incarnation"}:
        column = "disabled_cause" if change == "disabled" else "login_incarnation"
        rig.auth._conn.execute(
            f"UPDATE auth_credentials SET {column} = 'changed' WHERE id = ?", (rig.row.id,)
        )
    else:
        data = dict(rig.auth.get_credential(rig.row.id).data)
        data[
            {"account": "account_id", "hub": "api_endpoint", "mesh-origin": "mesh_origin"}[change]
        ] = "different"
        rig.auth._conn.execute(
            "UPDATE auth_credentials SET data = ? WHERE id = ?", (json.dumps(data), rig.row.id)
        )
    rig.auth._conn.commit()
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)


def test_revocation_dominates_restored_positive_slot(rig):
    evidence = _active(rig)
    _verify(rig, evidence)
    positive = F._path(rig.root).read_bytes()
    revocation = request(rig, action="revoke")
    activate(rig, revocation)
    F._path(rig.root).write_bytes(positive)
    with pytest.raises(B.BindingRefusal, match="file_approval_negative_fence"):
        _verify(rig, evidence)


@pytest.mark.parametrize(
    "change", ["pending", "missing-journal", "stale-journal", "anchor", "signature"]
)
def test_negative_witness_and_signed_evidence_fail_closed(rig, change):
    evidence = _active(rig)
    if change == "stale-journal":
        # The existing invariant is a journal/slot generation match, not a
        # global anti-rollback counter. Restore a genuine old witness against a
        # newer signed slot, rather than editing an otherwise coherent revision.
        old = F._journal_path(rig.root).read_bytes()
        activate(rig, request(rig))
        F._journal_path(rig.root).write_bytes(old)
    elif change == "pending":
        journal = json.loads(F._journal_path(rig.root).read_bytes())
        journal["fences"][evidence.association]["phase"] = "pending"
        F._journal_path(rig.root).write_text(json.dumps(journal))
    elif change == "missing-journal":
        F._journal_path(rig.root).unlink()
    elif change == "anchor":
        rig.trio["statement_digest"] = "sha256:" + "f" * 64
    else:
        document = json.loads(F._path(rig.root).read_bytes())
        document["slots"][evidence.association]["record"]["signature"]["sig"] = "00"
        F._path(rig.root).write_text(json.dumps(document))
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)


def test_unbound_view_only_member_has_no_default_grant(rig):
    evidence = B.OwnerFileEvidenceReference(association="0" * 64)
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)


def test_prepared_or_signed_but_unapplied_evidence_cannot_issue_binding(rig):
    record = request(rig)
    evidence = B.OwnerFileEvidenceReference(association=F._association(F._intent(record)))
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)
    approval_tests.approve(rig, record)
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)


def test_a_listener_shaped_object_cannot_supply_owner_authority(rig):
    evidence = _active(rig)
    forged = SimpleNamespace(**rig.link.__dict__)
    with pytest.raises(B.BindingRefusal, match="file_approval_listener_required"):
        _verify(rig, evidence, incoming_link=forged)


@pytest.mark.parametrize(
    "field,value",
    [
        ("required_purpose", "credentials"),
        ("required_purpose", True),
        ("required_subject", {"issuer": F.DEFAULT_RADIENT_API_BASE_URL, "opaque_id": "account-a"}),
        ("owner_account", {"credential_id": 1}),
        ("evidence", True),
        ("read_binding", {}),
    ],
)
def test_dict_bool_and_wrong_domain_inputs_are_typed_refusals(rig, field, value):
    with pytest.raises(B.BindingRefusal, match="account_binding_invalid_context"):
        _verify(rig, _active(rig), **{field: value})


def test_wrong_association_selector_cannot_supply_evidence(rig):
    _active(rig)
    with pytest.raises(B.BindingRefusal, match="account_binding_evidence_mismatch"):
        _verify(rig, B.OwnerFileEvidenceReference(association="0" * 64))


def test_validation_bypass_on_typed_model_is_not_trusted(rig):
    evidence = _active(rig)
    malformed = _read(rig).model_copy(update={"length": True})
    with pytest.raises(B.BindingRefusal, match="account_binding_invalid_context"):
        _verify(rig, evidence, read_binding=malformed)


@pytest.mark.parametrize(
    "field,value", [("offset", False), ("length", 17.0), ("network_epoch", True)]
)
def test_receipt_rejects_unvalidated_models_even_when_python_values_compare_equal(
    rig, field, value
):
    verified = _verify(rig, _active(rig))
    malformed = _read(rig).model_copy(update={field: value})
    assert malformed == verified.read_binding  # Numeric equality is not strict validation.
    with pytest.raises(B.BindingRefusal):
        verified.require_read(rig.link, malformed)


@pytest.mark.parametrize(
    "field,value", [("length", True), ("offset", -1), ("phase", "unknown"), ("network_epoch", True)]
)
def test_read_values_are_closed_strict_and_frozen(rig, field, value):
    with pytest.raises(ValidationError):
        _read(rig, **{field: value})
    with pytest.raises(ValidationError):
        _read(rig, extra_security_field="unsupported")


@pytest.mark.parametrize("change", ["link", "transcript", "relay-instance"])
def test_receipt_rejects_changed_listener_incarnation(rig, change):
    verified = _verify(rig, _active(rig))
    exact = _read(rig)
    if change == "link":
        rig.link.link_id = "different-link"
    elif change == "transcript":
        rig.link.result.transcript_hash = "b" * 64
    else:
        rig.server.instance_id = "different-relay-instance"
    with pytest.raises(B.BindingRefusal):
        verified.require_read(rig.link, exact)


@pytest.mark.parametrize("change", ["incarnation", "anchor", "admission", "transcript"])
def test_changes_during_fresh_canonical_io_cannot_issue_a_receipt(rig, monkeypatch, change):
    evidence = _active(rig)
    original = F.verify_canonical_account

    async def canonical_then_change(access):
        account = await original(access)
        if change == "incarnation":
            rig.auth._conn.execute(
                "UPDATE auth_credentials SET login_incarnation = 'new-login' WHERE id = ?",
                (rig.row.id,),
            )
            rig.auth._conn.commit()
        elif change == "anchor":
            rig.trio["statement_digest"] = "sha256:" + "f" * 64
        elif change == "admission":
            rig.network.members[1].added_by = "different-admitter"
            store.save(rig.network, rig.root)
        else:
            rig.link.result.transcript_hash = "b" * 64
        return account

    monkeypatch.setattr(F, "verify_canonical_account", canonical_then_change)
    with pytest.raises(B.BindingRefusal):
        _verify(rig, evidence)


def test_snapshot_is_constructed_inside_existing_lock_without_exporting_maps(rig, monkeypatch):
    from contextlib import contextmanager

    evidence = _active(rig)
    original_lock, original_snapshot = F._locked, F._VerifiedFileContext
    held = []

    @contextmanager
    def locked(*args, **kwargs):
        with original_lock(*args, **kwargs):
            held.append(True)
            try:
                yield
            finally:
                held.pop()

    def construct(**kwargs):
        assert held == [True]
        assert "record" not in kwargs and "journal" not in kwargs
        return original_snapshot(**kwargs)

    monkeypatch.setattr(F, "_locked", locked)
    monkeypatch.setattr(F, "_VerifiedFileContext", construct)
    verified = _verify(rig, evidence)
    assert not held
    assert not any(
        isinstance(getattr(verified, f.name), dict) for f in dataclasses.fields(verified)
    )


def test_adapter_has_no_registration_or_content_egress_surface():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(B.__file__).read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {"register", "send", "sendall", "serve_file", "_read_span", "_hash_fd"}
        for node in ast.walk(tree)
    )
    assert not hasattr(B.VerifiedAccountDeviceBinding, "commit_response_to_link")
