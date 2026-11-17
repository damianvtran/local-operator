"""Dormant P1 contracts, not physical presence/native-network attestation.

Real SQLite native-login completion, approval serialization, ES256 verification,
current-slot fsync/readback and CAS are exercised. HTTP, installed presence and
incoming handshake admission are SYNTHETIC: no OS gesture, live login or peer
socket is claimed. Independent real-path QA owns the isolated typed HTTP flow.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from local_operator.network import approval_ops
from local_operator.network import approvals as A
from local_operator.network import file_device_approval as F
from local_operator.network import identity, relay, store, wire
from local_operator.network.relay import PeerLink
from local_operator.network.types import (
    HandshakeResult,
    MemberRecord,
    MeshRefusal,
    NetworkRecord,
)
from local_operator.operator import trust
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.oauth import radient
from tests.unit.providers.test_radient_account_retention import _verified_login_result

N = "file-approval-test-network"


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_operator import operator
    from local_operator.operator import sign

    root = tmp_path / "owner"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(AuthStore, "_rearm_parked_tunnel_login", lambda *a, **kw: None)
    auth = AuthStore(root / "auth.db", config_dir=root)
    row = auth.complete_radient_login(asyncio.run(_verified_login_result("account-a")))
    own = identity.load_or_mint(root)
    peer = identity.load_or_mint(tmp_path / "peer")
    network = NetworkRecord(
        network_id=N,
        name="synthetic",
        self_device_id=own.device_id,
        members=[
            MemberRecord(
                device_id=own.device_id, public_key=own.public_key, capabilities=["view", "admin"]
            ),
            MemberRecord(
                device_id=peer.device_id, public_key=peer.public_key, capabilities=["view"]
            ),
        ],
    )
    store.save(network, root)
    keys = wire.link_keys(b"s" * 32, b"t" * 32)
    result = HandshakeResult(
        role="listener",
        peer_device_id=peer.device_id,
        peer_instance_id="synthetic",
        peer_public_key=peer.public_key,
        network_id=N,
        epoch=network.epoch,
        phase="member",
        link_id=keys.link_id.hex(),
        keys=keys,
        sas="synthetic",
        transcript_hash=hashlib.sha256(b"synthetic-handshake-not-physical-proof").hexdigest(),
    )
    # Admission itself is not under test. Use the REAL class/predicates without
    # constructing sockets/reader threads or silently calling this a handshake.
    server = SimpleNamespace(
        root=root,
        identity=own,
        instance_id="synthetic-relay-instance",
        links={},
        _links_lock=threading.RLock(),
    )
    link = object.__new__(PeerLink)
    # This unit-only server is an explicit shape double, not a relay process.
    monkeypatch.setattr(link, "server", server, raising=False)
    link.result = result
    link.link_id, link.device_id = result.link_id, result.peer_device_id
    link.network_id, link.epoch = result.network_id, result.epoch
    link.phase, link.frames_in, link.frames_out = result.phase, 0, 0
    link._closed, link._flushing = threading.Event(), threading.Event()
    server.links[link.link_id] = link

    private = ec.generate_private_key(ec.SECP256R1())
    spki = private.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    trio = {
        "key_id": "synthetic-presence-key",
        "spki_fp": "sha256:" + hashlib.sha256(spki).hexdigest(),
        "statement_digest": "sha256:"
        + hashlib.sha256(b"synthetic-installed-statement").hexdigest(),
    }
    loaded = SimpleNamespace(
        usable=True, root_owned=True, anchor=SimpleNamespace(backend=trust.SECURE_ENCLAVE)
    )
    level = {"value": operator.LEVEL_OPERATOR_PRESENCE}
    monkeypatch.setattr(operator, "authority_level_load", lambda: (level["value"], loaded))
    monkeypatch.setattr(trust, "anchor_trio", lambda anchor: dict(trio))
    monkeypatch.setattr(A, "local_anchor_trio", lambda root=None: {**trio, "spki": spki})
    monkeypatch.setattr(sign, "resolve_backend_name", lambda root: loaded.anchor.backend)
    calls = {"sign": 0, "accounts": [], "selected_rows": []}

    class SyntheticPresenceSigner:
        handle = SimpleNamespace(key_id=trio["key_id"])

        def sign(self, message, *, timeout=None):
            calls["sign"] += 1
            return private.sign(message, ec.ECDSA(hashes.SHA256()))

        def close(self):
            pass

    monkeypatch.setattr(sign, "load_signer", lambda **kw: SyntheticPresenceSigner())
    observed = {"account": "account-a", "error": None}

    async def canonical(access_token, **kwargs):
        current_row = auth.get_credential(row.id)
        assert current_row is not None
        assert access_token == current_row.data["access"]
        calls["accounts"].append(observed["account"])
        if observed["error"]:
            raise observed["error"]
        return observed["account"]

    original_fresh = auth.ensure_oauth_fresh_or_raise

    async def exact_row(credential_id, **kwargs):
        calls["selected_rows"].append(credential_id)
        return await original_fresh(credential_id, **kwargs)

    def own_relay(selected_root=None):
        if selected_root is None or Path(selected_root).resolve() != root:
            return None
        return SimpleNamespace(instance_id=server.instance_id, device_id=own.device_id)

    def local_status(record, op, **kwargs):
        assert op == "net_status"
        rows = [
            {
                "link_id": value.link_id,
                "device_id": value.device_id,
                "network_id": value.network_id,
                "epoch": value.epoch,
                "phase": value.phase,
                "role": value.result.role,
                "alive": value.alive,
                "closing": value.closing,
            }
            for value in server.links.values()
            if isinstance(value, PeerLink)
        ]
        return {
            "op": "ack",
            "req": "synthetic-local-control",
            "detail": {
                "instance_id": server.instance_id,
                "device_id": own.device_id,
                "links": rows,
            },
        }

    monkeypatch.setattr(F, "_local_relay_record", own_relay)
    monkeypatch.setattr(relay, "control_request", local_status)
    monkeypatch.setattr(F, "verify_canonical_account", canonical)
    monkeypatch.setattr(auth, "ensure_oauth_fresh_or_raise", exact_row)
    F._CHECKED.set(None)
    r = SimpleNamespace(
        root=root,
        own=own,
        peer=peer,
        network=network,
        auth=auth,
        row=row,
        link=link,
        server=server,
        trio=trio,
        spki=spki,
        loaded=loaded,
        level=level,
        calls=calls,
        observed=observed,
    )
    try:
        yield r
    finally:
        F._CHECKED.set(None)
        auth.close()


def request(rig, **kwargs):
    return asyncio.run(F.file_request(rig.link, rig.auth, rig.row.id, **kwargs))


def approve(rig, record):
    decided = time.time()
    signature = A.sign_decision(
        kind=record["kind"],
        request_id=record["request_id"],
        request_digest=record["request_digest"],
        decided_at=decided,
        config_root=rig.root,
    )
    return A._record_decision(
        record["approval_id"],
        decision="approve",
        decided_at=decided,
        signature_hex=signature,
        root=rig.root,
        expected_digest=record["request_digest"],
    )


def activate(rig, record):
    approved = approve(rig, record)
    return F.apply(approved["approval_id"], F.expected(approved), rig.root)


def test_typed_provider_is_registered_but_dormant_and_generic_signers_refuse(rig):
    record = request(rig)
    operation = approval_ops.get(F.OP)
    assert operation is not None
    assert operation.kind == A.KIND_FILE_DEVICE_APPROVAL
    assert operation.file_only_signing is False
    assert operation.cli_approvable is False
    assert not A.list_records(root=rig.root), "dormant cards must not add a UI interaction"
    for surface in (approval_ops.SURFACE_CLI, approval_ops.SURFACE_DESKTOP):
        with pytest.raises(MeshRefusal):
            approval_ops.require_generic_approvable(record, record["approval_id"], surface=surface)
    assert rig.calls["sign"] == 0


def test_valid_local_context_activates_only_metadata_and_reconciles_idempotently(rig):
    record = request(rig)
    assert activate(rig, record) == {"status": "active", "generation": 1}
    context = asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    assert context["generation"] == 1
    before = F._path(rig.root).read_bytes()
    assert F.apply(record["approval_id"], F.expected(record), rig.root)["generation"] == 1
    assert F._path(rig.root).read_bytes() == before
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == A.STATE_CONNECTED
    assert set(rig.calls["selected_rows"]) == {rig.row.id}
    state = json.loads(before)
    assert state["slots"] and len(state["slots"]) == 1
    assert "test-only" not in before.decode()  # no native bearer material persisted
    assert stat.S_IMODE(F._path(rig.root).stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", "other"),
        ("action", "revoke"),
        ("owner.public_key", "A" * 43),
        ("requester.public_key", "B" * 43),
        ("owner.algorithm", "other"),
        ("account.hub", "https://other.invalid/v1"),
        ("account.account_id", "account-b"),
        ("account.credential_id", 999),
        ("account.credential_created_at", 1),
        ("account.login_incarnation", "another-login"),
        ("network_id", "another-network"),
        ("network_epoch", 2),
        ("requester_added_at", 1.0),
        ("anchor.statement_digest", "different"),
        ("authority_epoch", "0" * 32),
        ("expected_generation", 5),
        ("generation", 9),
        ("nonce", "0" * 32),
        ("link_id", "another-link"),
        ("transcript_hash", "0" * 64),
        ("created_at", 1.0),
        ("expires_at", 2.0),
    ],
)
def test_every_security_fact_is_inside_the_commitment_and_tampering_cannot_apply(rig, field, value):
    record = approve(rig, request(rig))
    bad = copy.deepcopy(record)
    target = bad["what"][F.OP]
    path = field.split(".")
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert A.request_digest(bad) != record["request_digest"]
    # Raw isolated disk corruption deliberately bypasses the production writer;
    # do NOT weaken its guard just to construct a negative fixture.
    store._write_private_json(A.record_path(record["approval_id"], rig.root), bad)
    with pytest.raises(MeshRefusal):
        F.apply(record["approval_id"], F.expected(record), rig.root)
    assert not json.loads(F._path(rig.root).read_bytes())["slots"]


def test_file_decision_has_its_own_domain_and_old_domains_are_unchanged(rig):
    from local_operator.network import enrollment

    request_id, digest = "req_synthetic", "sha256:" + "0" * 64

    def payload(kind: str):
        return A.signed_payload(
            kind=kind,
            request_id=request_id,
            request_digest=digest,
            decision="approve",
            decided_at=1.0,
        )

    assert payload(A.KIND_FILE_DEVICE_APPROVAL).startswith(F.DECISION_DOMAIN)
    old = payload(A.KIND_DEVICE_ONBOARD)
    expected = A.APPROVAL_DOMAIN_LITERAL + b"".join(
        A._lp(x)
        for x in (
            A.KIND_DEVICE_ONBOARD,
            request_id,
            digest,
            "approve",
            A.decided_at_text(1.0),
        )
    )
    assert old == expected
    assert payload(enrollment.KIND).startswith(enrollment.DECISION_DOMAIN)
    record = approve(rig, request(rig))
    forged = {**record, "kind": A.KIND_DEVICE_ONBOARD}
    trio = A.local_anchor_trio(rig.root)
    assert trio is not None
    with pytest.raises(MeshRefusal):
        A._verify_decision_signature(
            forged,
            decision="approve",
            decided_at=record["decided_at"],
            signature_hex=record["signature"]["sig"],
            spki=trio["spki"],
        )


@pytest.mark.parametrize(
    "change",
    [
        "dialer",
        "empty-key",
        "wrong-key",
        "alias",
        "removed",
        "suspect",
        "epoch",
        "no-view",
        "closed",
        "replaced",
    ],
)
def test_authenticated_listener_and_current_exact_membership_are_required(rig, change):
    if change == "dialer":
        rig.link.result.role = "dialer"
    elif change == "empty-key":
        rig.link.result.peer_public_key = ""
    elif change == "wrong-key":
        rig.link.result.peer_public_key = rig.own.public_key
    elif change == "alias":
        rig.network.members[1].device_id = "different-id"
        rig.network.members[1].previous_ids = [rig.peer.device_id]
    elif change == "removed":
        rig.network.removed_ids.append(rig.peer.device_id)
    elif change == "suspect":
        rig.network.members[1].suspect = True
    elif change == "epoch":
        rig.network.epoch += 1
    elif change == "no-view":
        rig.network.members[1].capabilities = []
    elif change == "closed":
        rig.link._closed.set()
    elif change == "replaced":
        rig.server.links[rig.link.link_id] = object()
    store.save(rig.network, rig.root)
    with pytest.raises(MeshRefusal):
        request(rig)
    assert rig.calls["sign"] == 0
    assert not F._path(rig.root).exists()


@pytest.mark.parametrize("change", ["file-only", "staged", "backend", "trio"])
def test_installed_presence_is_required_at_entry_sign_write_and_apply(rig, change, monkeypatch):
    record = request(rig)
    approved = approve(rig, record)
    from local_operator import operator

    if change == "file-only":
        rig.level["value"] = operator.LEVEL_OPERATOR_FILE_ONLY
    elif change == "staged":
        rig.loaded.root_owned = False
    elif change == "backend":
        rig.loaded.anchor.backend = "file-only"
    else:
        # Installed authority and the profile-local read disagree. Updating BOTH
        # to a genuine new installed statement would correctly permit filing a
        # fresh card; that is reapproval, not a refusal of all future requests.
        monkeypatch.setattr(
            A,
            "local_anchor_trio",
            lambda root=None: {
                **rig.trio,
                "statement_digest": "sha256:" + "f" * 64,
                "spki": rig.spki,
            },
        )
    count = rig.calls["sign"]
    for operation in (
        lambda: request(rig),
        lambda: A.sign_decision(
            kind=record["kind"],
            request_id=record["request_id"],
            request_digest=record["request_digest"],
            decided_at=time.time(),
            config_root=rig.root,
        ),
        lambda: A._write_record(approved, rig.root),
        lambda: F.apply(record["approval_id"], F.expected(record), rig.root),
    ):
        with pytest.raises(MeshRefusal):
            operation()
    assert rig.calls["sign"] == count


def test_prompt_context_change_is_refused_before_signature_is_returned(rig, monkeypatch):
    from local_operator.operator import sign

    real = sign.sign_message
    record = request(rig)

    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        rig.link._closed.set()
        return result

    monkeypatch.setattr(sign, "sign_message", changed)
    with pytest.raises(MeshRefusal):
        approve(rig, record)
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == A.STATE_REQUESTED


def test_generic_filing_and_low_level_unsigned_positive_write_cannot_mint_authority(rig):
    record = request(rig)
    with pytest.raises(MeshRefusal):
        A.create_request(
            kind=record["kind"],
            request_id=record["request_id"],
            requested_by=record["requested_by"],
            what=record["what"],
            machine=record["machine"],
            credential_ref=None,
            created_at=record["created_at"],
            expires_at=record["expires_at"],
            root=rig.root,
        )
    with pytest.raises(MeshRefusal):
        A._write_record({**record, "state": A.STATE_APPROVED}, rig.root)
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == A.STATE_REQUESTED
    assert rig.calls["sign"] == 0


@pytest.mark.parametrize(
    "change", ["account", "hub", "borrowed", "disabled", "incarnation", "missing"]
)
def test_native_row_context_change_invalidates_activation(rig, change):
    record = approve(rig, request(rig))
    if change == "missing":
        rig.auth._conn.execute("DELETE FROM auth_credentials WHERE id = ?", (rig.row.id,))
    elif change == "disabled":
        rig.auth._conn.execute(
            "UPDATE auth_credentials SET disabled_cause = 'synthetic' WHERE id = ?", (rig.row.id,)
        )
    elif change == "incarnation":
        rig.auth._conn.execute(
            "UPDATE auth_credentials SET login_incarnation = 'new-login' WHERE id = ?",
            (rig.row.id,),
        )
    else:
        data = dict(rig.auth.get_credential(rig.row.id).data)
        if change == "account":
            data["account_id"] = "account-b"
        elif change == "hub":
            data["api_endpoint"] = "https://other.invalid/v1"
        else:
            data["mesh_origin"] = {"owner_device": rig.peer.device_id}
        rig.auth._conn.execute(
            "UPDATE auth_credentials SET data = ? WHERE id = ?", (json.dumps(data), rig.row.id)
        )
    rig.auth._conn.commit()
    with pytest.raises(MeshRefusal):
        F.apply(record["approval_id"], F.expected(record), rig.root)
    assert not json.loads(F._path(rig.root).read_bytes())["slots"]


@pytest.mark.parametrize("failure", ["mismatch", "offline"])
def test_canonical_account_recheck_never_falls_back_or_accepts_cached_account(rig, failure):
    record = approve(rig, request(rig))
    if failure == "mismatch":
        rig.observed["account"] = "account-b"
    else:
        rig.observed["error"] = httpx.ConnectError("synthetic-offline")
    with pytest.raises(MeshRefusal):
        F.apply(record["approval_id"], F.expected(record), rig.root)
    assert not json.loads(F._path(rig.root).read_bytes())["slots"]


@pytest.mark.parametrize(
    "change", ["epoch", "requester-key", "owner-key", "admission", "removed", "suspect", "no-view"]
)
def test_changed_mesh_context_denies_an_existing_active_slot(rig, change):
    activate(rig, request(rig))
    if change == "epoch":
        rig.network.epoch += 1
    elif change == "requester-key":
        rig.network.members[1].public_key = rig.own.public_key
    elif change == "owner-key":
        rig.server.identity = SimpleNamespace(public_key=rig.peer.public_key)
    elif change == "admission":
        rig.network.members[1].added_at += 1
    elif change == "removed":
        rig.network.removed_ids.append(rig.peer.device_id)
    elif change == "suspect":
        rig.network.members[1].suspect = True
    else:
        rig.network.members[1].capabilities = []
    store.save(rig.network, rig.root)
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))


def test_routine_refresh_same_context_does_not_change_generation(rig, monkeypatch):
    record = request(rig)
    activate(rig, record)
    row = rig.auth.get_credential(rig.row.id)
    data = {**row.data, "access": "refreshed-test-only", "access_token": "refreshed-test-only"}
    incarnation = rig.auth.native_login_incarnation(row.id)

    async def refresh(existing):
        return {**existing, **data}

    # Interactive upsert intentionally starts a new login incarnation. Exercise
    # the actual exact-row refresh path instead, without making a network call.
    monkeypatch.setattr(rig.auth, "_refresh_fn", lambda provider: refresh)
    asyncio.run(rig.auth.ensure_oauth_fresh_or_raise(row.id, force=True))
    assert rig.auth.native_login_incarnation(row.id) == incarnation
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, row.id))["generation"] == 1


def test_cas_revoke_and_replay_do_not_reactivate_old_generation(rig):
    first, competing = request(rig), request(rig)
    # Both decisions precede mutation: after g1 commits the stronger early
    # source guard correctly refuses even SIGNING a competing expected-g0 card.
    approve(rig, competing)
    activate(rig, first)
    with pytest.raises(MeshRefusal):
        F.apply(competing["approval_id"], F.expected(competing), rig.root)
    revocation = request(rig, action="revoke")
    approve(rig, revocation)
    assert F.revoke(revocation["approval_id"], F.expected(revocation), rig.root) == {
        "status": "revoked",
        "generation": 2,
    }
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    with pytest.raises(MeshRefusal):
        F.apply(first["approval_id"], F.expected(first), rig.root)
    assert (
        next(iter(json.loads(F._path(rig.root).read_bytes())["slots"].values()))["status"]
        == "revoked"
    )


def test_request_expiry_refuses_activation_but_does_not_expire_valid_current_slot(rig, monkeypatch):
    first = request(rig)
    activate(rig, first)
    pending = approve(rig, request(rig))
    monkeypatch.setattr(F, "time", SimpleNamespace(time=lambda: pending["expires_at"] + 1))
    with pytest.raises(MeshRefusal):
        F.apply(pending["approval_id"], F.expected(pending), rig.root)
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))["generation"] == 1


def test_history_pruning_and_lost_pending_provenance_never_rebuild_missing_state(rig):
    record = request(rig)
    activate(rig, record)
    A.sweep(root=rig.root, now=time.time() + 40 * 86400)
    assert not A.record_path(record["approval_id"], rig.root).exists()
    F._CHECKED.set(None)  # A new desktop worker has no in-memory authority.
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))["generation"] == 1
    F._path(rig.root).unlink()  # only this test's own newly-created state
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    assert not F._path(rig.root).exists()


@pytest.mark.parametrize("corruption", ["json", "unknown", "signature", "duplicate", "symlink"])
def test_corrupt_or_redirected_state_never_becomes_authority(rig, corruption):
    record = request(rig)
    activate(rig, record)
    path = F._path(rig.root)
    if corruption == "json":
        path.write_text("{")
    elif corruption == "duplicate":
        path.write_text('{"schema":1,"schema":1}')
    elif corruption == "symlink":
        replacement = rig.root / "only-this-test-state.json"
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
    else:
        data = json.loads(path.read_bytes())
        slot = next(iter(data["slots"].values()))
        if corruption == "unknown":
            slot["unknown"] = True
        else:
            slot["record"]["signature"]["sig"] = "00"
        path.write_text(json.dumps(data))
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))


def test_restored_active_snapshot_cannot_undo_completed_revocation(rig):
    original = request(rig)
    activate(rig, original)
    before_revoke = F._path(rig.root).read_bytes()
    revocation = request(rig, action="revoke")
    activate(rig, revocation)
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    # A previously valid signed generation is not a new local approval. A
    # current-slot design must state and enforce its rollback boundary, rather
    # than imply that generation/CAS alone protects a rewritable private file.
    F._path(rig.root).write_bytes(before_revoke)
    witness = F._load_journal(rig.root)
    assert next(iter(witness.fences.values())).deny_generation == 2
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))


def test_existing_typed_backend_uses_the_registered_file_operation(rig):
    from local_operator.network.credentials import desktop_approval

    record = request(rig)
    operation = approval_ops.get(F.OP)
    assert operation is not None
    facts = operation.disclosure(record)
    args = {
        "op": F.OP,
        "request_digest": record["request_digest"],
        "disclosure_digest_": desktop_approval.disclosure_digest(facts),
    }
    result = desktop_approval.approve(rig.root, record["approval_id"], **args)
    assert result["state"] == A.STATE_APPROVED
    assert result["operation"] == F.OP
    desktop_approval.apply(rig.root, record["approval_id"], **args)
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == A.STATE_CONNECTED
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))["generation"] == 1
    assert not A.list_records(root=rig.root), "no default Mesh UI exposure"


@pytest.mark.parametrize(
    "target,corruption",
    [
        ("journal", "missing"),
        ("marker", "missing"),
        ("journal", "json"),
        ("marker", "json"),
        ("journal", "symlink"),
        ("marker", "symlink"),
        ("journal", "mode"),
        ("marker", "mode"),
        ("journal", "schema-bool"),
        ("marker", "schema-bool"),
        ("journal", "epoch"),
        ("marker", "epoch"),
    ],
)
def test_initialized_witness_unknown_or_redirected_always_denies(rig, target, corruption):
    activate(rig, request(rig))
    path = F._journal_path(rig.root) if target == "journal" else F._sentinel_path(rig.root)
    if corruption == "missing":
        path.unlink()
    elif corruption == "json":
        path.write_text("{")
    elif corruption == "mode":
        path.chmod(0o644)
    elif corruption == "symlink":
        copy = rig.root / (target + "-only-this-test.json")
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
    else:
        data = json.loads(path.read_bytes())
        if corruption == "schema-bool":
            data["schema"] = True
        else:
            data["authority_epoch"] = "f" * 32
        path.write_text(json.dumps(data))
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    with pytest.raises(MeshRefusal):
        request(rig)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", None),
        ("role", "dialer"),
        ("alive", False),
        ("alive", 1),
        ("closing", True),
        ("closing", None),
        ("epoch", True),
        ("epoch", 2),
        ("link_id", "different"),
        ("phase", "pair"),
    ],
)
def test_pending_worker_denies_old_or_malformed_local_status(rig, monkeypatch, field, value):
    record = request(rig)
    original = relay.control_request

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        assert isinstance(result, dict)
        row = result["detail"]["links"][0]
        if value is None:
            row.pop(field, None)
        else:
            row[field] = value
        return result

    monkeypatch.setattr(relay, "control_request", changed)
    with pytest.raises(MeshRefusal):
        approve(rig, record)
    assert rig.calls["sign"] == 0
    assert F._load_journal(rig.root).fences == {}


def test_new_consumer_refuses_non_ack_even_with_positive_status_detail(rig, monkeypatch):
    record = request(rig)
    original = relay.control_request

    def non_ack(*args, **kwargs):
        reply = original(*args, **kwargs)
        assert isinstance(reply, dict)
        reply.pop("op")
        reply["ok"] = True
        return reply

    monkeypatch.setattr(relay, "control_request", non_ack)
    with pytest.raises(MeshRefusal):
        approve(rig, record)
    assert rig.calls["sign"] == 0


@pytest.mark.parametrize("revision", [True, 0, F._MAX_REVISION + 1])
def test_journal_cas_rejects_invalid_revision_without_mutation(rig, revision):
    request(rig)
    before = F._journal_path(rig.root).read_bytes()
    journal = F._load_journal(rig.root)
    with pytest.raises(MeshRefusal):
        F._publish_journal(rig.root, journal, expected_revision=revision)
    assert F._journal_path(rig.root).read_bytes() == before


def test_network_and_native_prompt_work_never_runs_under_mutation_locks(rig, monkeypatch):
    from contextlib import contextmanager

    from local_operator.operator import sign

    depth = {"domain": 0, "record": 0}

    def unlocked():
        assert depth == {"domain": 0, "record": 0}

    original_domain, original_record = F._locked, A._record_lock

    @contextmanager
    def domain(*args, **kwargs):
        with original_domain(*args, **kwargs):
            depth["domain"] += 1
            try:
                yield
            finally:
                depth["domain"] -= 1

    @contextmanager
    def record(*args, **kwargs):
        with original_record(*args, **kwargs):
            depth["record"] += 1
            try:
                yield
            finally:
                depth["record"] -= 1

    monkeypatch.setattr(F, "_locked", domain)
    monkeypatch.setattr(A, "_record_lock", record)
    signer = sign.load_signer(config_root=rig.root)
    assert signer is not None
    monkeypatch.setattr(sign, "load_signer", lambda **kwargs: signer)
    original_http, original_control, original_sign = (
        F.verify_canonical_account,
        relay.control_request,
        signer.sign,
    )

    async def http(*args, **kwargs):
        unlocked()
        return await original_http(*args, **kwargs)

    def control(*args, **kwargs):
        unlocked()
        return original_control(*args, **kwargs)

    def checked_sign(*args, **kwargs):
        unlocked()
        return original_sign(*args, **kwargs)

    monkeypatch.setattr(F, "verify_canonical_account", http)
    monkeypatch.setattr(relay, "control_request", control)
    monkeypatch.setattr(signer, "sign", checked_sign)
    activate(rig, request(rig))
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))["generation"] == 1


def test_filing_new_candidate_does_not_suspend_existing_association(rig):
    activate(rig, request(rig))
    request(rig)
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))["generation"] == 1


def test_selective_journal_rewind_denies_newer_valid_active_slot(rig):
    activate(rig, request(rig))
    old = F._journal_path(rig.root).read_bytes()
    activate(rig, request(rig))
    assert asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))["generation"] == 2
    F._journal_path(rig.root).write_bytes(old)
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))


def test_projection_failure_retains_revoke_watermark_and_exact_retry(rig, monkeypatch):
    activate(rig, request(rig))
    rev = approve(rig, request(rig, action="revoke"))
    original = store._write_private_json_durable

    def fail_projection(path, payload):
        if path == F._path(rig.root):
            raise OSError("synthetic-projection-failure")
        return original(path, payload)

    monkeypatch.setattr(store, "_write_private_json_durable", fail_projection)
    with pytest.raises(MeshRefusal):
        F.apply(rev["approval_id"], F.expected(rev), rig.root)
    fence = next(iter(F._load_journal(rig.root).fences.values()))
    assert (fence.phase, fence.deny_generation) == ("pending", 2)
    assert next(iter(F._load(rig.root).slots.values())).generation == 1
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    monkeypatch.setattr(store, "_write_private_json_durable", original)
    assert F.apply(rev["approval_id"], F.expected(rev), rig.root) == {
        "status": "revoked",
        "generation": 2,
    }


def test_indeterminate_finish_compensates_to_pending_at_higher_revision(rig, monkeypatch):
    first = approve(rig, request(rig))
    original = store._write_private_json_durable

    def fail_after_complete(path, payload):
        result = original(path, payload)
        if path == F._journal_path(rig.root) and any(
            row["phase"] == "complete" for row in payload["fences"].values()
        ):
            raise OSError("synthetic-completion-ack-failure")
        return result

    monkeypatch.setattr(store, "_write_private_json_durable", fail_after_complete)
    with pytest.raises(MeshRefusal):
        F.apply(first["approval_id"], F.expected(first), rig.root)
    journal = F._load_journal(rig.root)
    assert next(iter(journal.fences.values())).phase == "pending"
    assert journal.revision >= 5  # prepared, pending, failed finish, compensation
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    monkeypatch.setattr(store, "_write_private_json_durable", original)
    assert F.apply(first["approval_id"], F.expected(first), rig.root)["generation"] == 1


def test_status_extension_preserves_old_rows_and_unknown_new_fields(rig):
    server = rig.server
    server._membership_reports = {}
    server.peer_record = lambda: SimpleNamespace(listen={})
    server.network_summary = lambda row, report=None: {}
    server._control_port = 0
    server.audit = SimpleNamespace(
        degraded=False, degraded_reason="", published_through=0, recorded_through=0
    )
    server.output_log, server.started_at = "synthetic", time.time()
    old = {
        "link_id": rig.link.link_id,
        "device_id": rig.link.device_id,
        "network_id": rig.link.network_id,
        "epoch": rig.link.epoch,
        "phase": rig.link.phase,
        "frames_in": 0,
        "frames_out": 0,
    }
    result = relay.RelayServer.status(server)["links"][0]
    assert list(result)[-3:] == ["role", "alive", "closing"]
    stripped = {k: v for k, v in result.items() if k not in {"role", "alive", "closing"}}
    assert json.dumps(stripped, separators=(",", ":")) == json.dumps(old, separators=(",", ":"))
    legacy = SimpleNamespace(**old)
    server.links = {legacy.link_id: legacy}
    row = relay.RelayServer.status(server)["links"][0]
    assert {k: row[k] for k in old} == old
    assert [row[k] for k in ("role", "alive", "closing")] == [None, None, None]


def test_parent_sync_failure_is_indeterminate_and_same_signed_generation_reconciles(
    rig, monkeypatch
):
    record = approve(rig, request(rig))
    original = store._sync_private_directory

    def fail_after_rename(directory):
        path = F._path(rig.root)
        if path.exists() and json.loads(path.read_bytes())["slots"]:
            raise OSError("synthetic-parent-fsync-failure")
        return original(directory)

    monkeypatch.setattr(store, "_sync_private_directory", fail_after_rename)
    with pytest.raises(MeshRefusal, match="file_approval_write_unconfirmed"):
        F.apply(record["approval_id"], F.expected(record), rig.root)
    # The rename DID happen. Never claim a failed durability acknowledgement
    # means no mutation, and never issue a new approval/generation to retry.
    assert (
        next(iter(json.loads(F._path(rig.root).read_bytes())["slots"].values()))["generation"] == 1
    )
    with pytest.raises(MeshRefusal):
        asyncio.run(F.authorized_context(rig.link, rig.auth, rig.row.id))
    monkeypatch.setattr(store, "_sync_private_directory", original)
    assert F.apply(record["approval_id"], F.expected(record), rig.root) == {
        "status": "active",
        "generation": 1,
    }


def test_strict_writer_syncs_new_parent_chain_and_legacy_writer_does_not_opt_in(
    tmp_path, monkeypatch
):
    seen = []
    original = store._sync_private_directory
    monkeypatch.setattr(
        store, "_sync_private_directory", lambda p: (seen.append(p), original(p))[-1]
    )
    path = tmp_path / "new" / "nested" / "state.json"
    store._write_private_json_durable(path, {"schema": 1})
    assert {tmp_path, path.parent, path.parent.parent} <= set(seen)
    assert store._read_private_json_durable(path) == {"schema": 1}
    monkeypatch.setattr(
        store,
        "_sync_private_directory",
        lambda p: (_ for _ in ()).throw(AssertionError("legacy opted into strict policy")),
    )
    store._write_private_json(tmp_path / "legacy.json", {"legacy": True})
    assert json.loads((tmp_path / "legacy.json").read_text()) == {"legacy": True}


@pytest.mark.parametrize(
    "status,body",
    [
        (302, {}),
        (401, {}),
        (200, {}),
        (200, {"result": {"identity": {"account_id": "a"}, "account": {"id": "b"}}}),
        (200, {"result": {"identity": {"account_id": " a "}, "account": {"id": " a "}}}),
    ],
)
def test_factored_canonical_verifier_rejects_redirects_and_mismatched_accounts(status, body):
    calls = []

    async def run():
        async def handler(request):
            calls.append(request)
            assert str(request.url) == radient.DEFAULT_RADIENT_API_BASE_URL + "/me"
            return httpx.Response(
                status, json=body, headers={"location": "https://other.invalid/me"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ) as client:
            with pytest.raises(radient.LoginError):
                await radient.verify_canonical_account("synthetic-test-access", http_client=client)

    asyncio.run(run())
    assert len(calls) == 1
