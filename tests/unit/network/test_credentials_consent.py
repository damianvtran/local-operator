"""Real source permission/deny boundaries with isolated software signatures.

No native signer, provider endpoint, live policy or new storage writer is exercised.
The file-only operator fixture is the existing approvals store fixture; broker
requests run the actual owner loop and AuthStore over synthetic SQLite material.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network import identity, onboard_approvals, store
from local_operator.network.credentials import consent, owner, placement, sync
from local_operator.network.types import MemberRecord, MeshRefusal, NetworkRecord
from local_operator.providers import registry
from local_operator.providers.auth_store import AuthStore
from tests.unit.network.test_approvals_store import _make_key, _sign_decision

N = "n_" + "1" * 24
P = "permissiontest"


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from local_operator.operator import trust

    root = tmp_path / "isolated-owner"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(
        trust,
        "load_anchor",
        lambda uid=None: trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="synthetic file-only operator",
            exists=False,
        ),
    )
    key = _make_key(root)
    own = identity.load_or_mint(root, name="synthetic-owner")
    other_root = tmp_path / "isolated-recipient"
    peer = identity.load_or_mint(other_root, name="synthetic-recipient")
    # Identity's public JSON is the same membership binding the real relay uses.
    public = peer.to_json()["public_key"]
    fp = (
        "sha256:"
        + hashlib.sha256(base64.urlsafe_b64decode(public + "=" * (-len(public) % 4))).hexdigest()
    )
    net = NetworkRecord(
        N,
        "synthetic",
        self_device_id=own.device_id,
        created_by=own.device_id,
        members=[
            MemberRecord(
                peer.device_id, public_key=public, role="admin", capabilities=["broker_credential"]
            )
        ],
    )
    store.save(net, root)
    definition = registry.ProviderDefinition(
        id=P,
        name="Permission test",
        store_credentials_as=P,
        get_api_key=lambda row: str(row.get("access") or ""),
    )
    monkeypatch.setitem(registry._BY_ID, P, definition)
    auth = AuthStore(db_path=root / "auth.db")
    auth.upsert_credential(
        P,
        {
            "type": "oauth",
            "email": "synthetic@example.invalid",
            "access": "synthetic-private-marker",
            "refresh": "synthetic-refresh-marker",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": "account-one",
            "org_id": "org-one",
        },
    )
    doc = placement.PlacementDocument(N, root=root, written_by=own.device_id)
    doc.declare(
        P,
        owner_device=own.device_id,
        owner_device_name="synthetic-owner",
        provider=P,
        kind="oauth",
        by=own.device_id,
    )
    doc.grant(P, peer.device_id, by=own.device_id)
    doc.save()
    broker = owner.MeshCredentialBroker(
        root=root,
        self_device=own.device_id,
        self_device_name="synthetic-owner",
        network_id=N,
        auth_store=auth,
    )
    subject = {
        "owner_device": own.device_id,
        "network_id": N,
        "recipient_device": peer.device_id,
        "public_key_fingerprint": fp,
        "native_context": {"account_id": "account-one", "org_id": "org-one"},
    }
    try:
        yield SimpleNamespace(
            root=root, own=own, peer=peer, broker=broker, auth=auth, subject=subject, anchor=key
        )
    finally:
        broker.close()
        auth.close()


def request(rig: Any, *, signed: bool = True, change: Any = None) -> tuple[str, str]:
    data = {
        **{k: v for k, v in rig.subject.items() if k != "native_context"},
        "action": "grant",
        "inherit": {
            "providers": {"ids": [P], "future_logins": True},
            "mcp": {"urls": [], "future_logins": True},
            "origin": "owner-native-login",
        },
        "scope": {
            "principal_ref": rig.anchor.key_id,
            "account_id": "account-one",
            "org_id": "org-one",
        },
    }
    if change:
        change(data)
    trio = A.local_anchor_trio(rig.root)
    assert trio is not None
    record = A.create_request(
        kind=A.KIND_DEVICE_ONBOARD,
        request_id=A.new_request_id(),
        requested_by={"session_id": "synthetic", "device_id": rig.own.device_id, "surface": "cli"},
        device={
            "device_id": rig.peer.device_id,
            "name": "synthetic",
            "host": "192.0.2.1",
            "user": "synthetic",
            "transport": "ssh",
            "host_key_fp": "SHA256:synthetic",
        },
        what={
            "install": False,
            "connect": False,
            "credential_inheritance": data,
            "anchor": {k: v for k, v in trio.items() if k != "spki"},
        },
        credential_ref=None,
        created_at=time.time() - 1,
        expires_at=time.time() + 3600,
        root=rig.root,
    )
    aid = record["approval_id"]
    if not signed:
        return aid, "run-not-admitted"
    moment = time.time()
    signature = _sign_decision(rig.root, record, "approve", moment)
    A.approve(aid, signature_hex=signature, decided_at=moment, root=rig.root)
    run = onboard_approvals.begin_run(aid, root=rig.root)
    return aid, run.run_id


def grant(rig: Any) -> dict[str, Any]:
    link = SimpleNamespace(device_id=rig.peer.device_id, network_id=N, epoch=1)
    return rig.broker.on_broker(
        link,
        {
            "op": "net_broker",
            "kind": "grant",
            "req": 1,
            "key": P,
            "provider": P,
            "from_device": rig.peer.device_id,
            "for_session": "synthetic-session",
        },
    )["detail"]


def test_signed_executed_policy_has_no_projection_or_copy(rig: Any) -> None:
    aid, run = request(rig)
    before = placement.placement_path(N, rig.root).read_bytes()
    reply = consent.provision_from_run(aid, run, rig.subject, rig.root)
    assert reply["kind"] == "permission_recorded"
    assert reply["activation"] == "static_storage_activation_pending"
    policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    row = policy["recipients"][rig.peer.device_id]
    assert row["enabled"] is True
    assert row["approval"]["ref"] == aid
    assert placement.placement_path(N, rig.root).read_bytes() == before
    receipts = A.load_record(aid, root=rig.root)["receipts"]
    assert any(r["step"] == consent.STEP and r["ok"] for r in receipts)


def test_requested_source_is_not_future_consent(rig: Any) -> None:
    aid, run = request(rig, signed=False)
    with pytest.raises(MeshRefusal):
        consent.provision_from_run(aid, run, rig.subject, rig.root)
    assert not consent.ConsentDocument(N, rig.own.device_id, rig.root).path.exists()


@pytest.mark.parametrize(
    "field", ["owner_device", "network_id", "recipient_device", "public_key_fingerprint"]
)
def test_signed_wrong_subject_cannot_write_policy(rig: Any, field: str) -> None:
    aid, run = request(rig, change=lambda data: data.__setitem__(field, "wrong"))
    with pytest.raises(MeshRefusal):
        consent.provision_from_run(aid, run, rig.subject, rig.root)
    assert not consent.ConsentDocument(N, rig.own.device_id, rig.root).path.exists()


def test_unknown_required_scope_and_copied_origin_refuse_before_write(rig: Any) -> None:
    aid, run = request(rig)
    subject = copy.deepcopy(rig.subject)
    subject["native_context"] = {}
    with pytest.raises(MeshRefusal, match="scope_unknown"):
        consent.provision_from_run(aid, run, subject, rig.root)
    subject["native_context"] = {**rig.subject["native_context"], "origin": "borrowed"}
    with pytest.raises(MeshRefusal, match="consent_origin_refused"):
        consent.provision_from_run(aid, run, subject, rig.root)
    assert not consent.ConsentDocument(N, rig.own.device_id, rig.root).path.exists()


def test_source_denied_before_executed_receipt_leaves_prepared_disabled(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    aid, run = request(rig)

    def deny(*args: Any, **kwargs: Any) -> None:
        A.deny(aid, root=rig.root)
        raise MeshRefusal("approval_denied", "synthetic cancellation")

    monkeypatch.setattr(onboard_approvals, "append_receipt", deny)
    with pytest.raises(MeshRefusal, match="consent_prepared_pending"):
        consent.provision_from_run(aid, run, rig.subject, rig.root)
    row = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
        rig.peer.device_id
    ]
    assert row["enabled"] is False


def test_legacy_grant_then_durable_revoke_survives_projection_crash(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert grant(rig)["kind"] == "grant"
    before = placement.placement_path(N, rig.root).read_bytes()
    original = placement.PlacementDocument.save

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic second-write failure")

    monkeypatch.setattr(placement.PlacementDocument, "save", fail)
    with pytest.raises(OSError):
        consent.restrict_and_revoke(
            N,
            rig.own.device_id,
            rig.peer.device_id,
            {"credential_key": P},
            "synthetic-withdrawal",
            rig.root,
        )
    assert placement.placement_path(N, rig.root).read_bytes() == before
    calls = []
    original_resolve = rig.broker._resolve_once

    async def resolve(**kwargs: Any) -> Any:
        calls.append(True)
        return await original_resolve(**kwargs)

    monkeypatch.setattr(rig.broker, "_resolve_once", resolve)
    result = grant(rig)
    assert result["kind"] == "error" and result["code"] == "not_a_holder"
    assert calls == []
    assert "access_token" not in result
    monkeypatch.setattr(placement.PlacementDocument, "save", original)
    # Reload/re-login/direct stale projection cannot erase the authoritative deny.
    doc = placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)
    doc.grant(P, rig.peer.device_id, by=rig.own.device_id)
    doc.save()
    assert grant(rig)["kind"] == "error"


def test_account_deny_uses_owner_selected_identity_zero_mint(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep a stale holder to discriminate policy from projection-only refusal.
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P, "account_ref": "account-one"},
        "synthetic-account-withdrawal",
        rig.root,
    )
    doc = placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)
    doc.grant(P, rig.peer.device_id, by=rig.own.device_id)
    doc.save()
    minted = []
    old = rig.broker._mint_grant

    def mint(**kwargs: Any) -> Any:
        minted.append(True)
        return old(**kwargs)

    monkeypatch.setattr(rig.broker, "_mint_grant", mint)
    reply = grant(rig)
    assert "kind" in reply, sorted(reply)
    assert reply["kind"] == "error" and reply["code"] == "not_a_holder"
    assert minted == [] and "access_token" not in reply


def test_owner_copy_explicit_deny_precedes_value_extraction(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)
    entry = doc.entry(P)
    assert entry is not None
    entry.kind = "api-key-static"
    doc.save()
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "synthetic-copy-withdrawal",
        rig.root,
    )
    doc = placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)
    doc.grant(P, rig.peer.device_id, by=rig.own.device_id)
    doc.save()
    extracted = []
    monkeypatch.setattr(sync, "read_copy_value", lambda *a, **k: extracted.append(True))
    link = SimpleNamespace(device_id=rig.peer.device_id, network_id=N, epoch=1)
    reply = rig.broker.on_broker(
        link,
        {
            "op": "net_broker",
            "kind": "copy",
            "req": 1,
            "key": P,
            "from_device": rig.peer.device_id,
            "act": rig.peer.device_id,
            "gen": 0,
        },
    )["detail"]
    assert "kind" in reply, sorted(reply)
    assert reply["kind"] == "error" and reply["code"] == "not_a_holder"
    assert extracted == [] and "value" not in reply


@pytest.mark.parametrize(
    "state", ["denied", "expired", "connected", "bad_signature", "tampered_digest", "stale_run"]
)
def test_non_runnable_or_forged_source_cannot_publish_policy(
    rig: Any, state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    aid, run = request(rig)
    record = A.load_record(aid, root=rig.root)
    if state == "bad_signature":
        record["signature"]["sig"] = "00" * 64
    elif state == "tampered_digest":
        record["what"]["credential_inheritance"]["inherit"]["providers"]["ids"] = ["different"]
    elif state == "stale_run":
        run = "run_wrong"
    elif state == "expired":
        monkeypatch.setattr(onboard_approvals.time, "time", lambda: record["expires_at"] + 1)
    else:
        record["state"] = state
    if state not in {"stale_run", "expired"}:
        A._write_record(record, rig.root)
    with pytest.raises(MeshRefusal):
        consent.provision_from_run(aid, run, rig.subject, rig.root)
    assert not consent.ConsentDocument(N, rig.own.device_id, rig.root).path.exists()


def test_withdrawal_while_native_resolution_waits_wins_before_mint(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()
    minted = []
    original = rig.auth.get_oauth_access

    async def paused(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        entered.set()
        assert await asyncio.to_thread(release.wait, 3)
        return result

    monkeypatch.setattr(rig.auth, "get_oauth_access", paused)
    oldmint = rig.broker._mint_grant

    def mint(**kwargs: Any) -> Any:
        minted.append(True)
        return oldmint(**kwargs)

    monkeypatch.setattr(rig.broker, "_mint_grant", mint)
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(grant, rig)
        try:
            assert entered.wait(3)
            consent.restrict_and_revoke(
                N,
                rig.own.device_id,
                rig.peer.device_id,
                {"credential_key": P},
                "synthetic-inflight-withdrawal",
                rig.root,
            )
        finally:
            release.set()
        reply = job.result(timeout=4)
    assert reply["kind"] == "error" and reply["code"] == "not_a_holder"
    assert minted == []


def test_original_approval_replay_cannot_clear_sticky_denial(rig: Any) -> None:
    aid, run = request(rig)
    consent.provision_from_run(aid, run, rig.subject, rig.root)
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "synthetic-sticky-revoke",
        rig.root,
    )
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    consent.provision_from_run(aid, run, rig.subject, rig.root)
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).load() == before
    assert grant(rig)["kind"] == "error"


def test_malformed_or_duplicate_authority_never_becomes_positive(rig: Any) -> None:
    document = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    with document.locked():
        baseline = json.dumps(document.load())
        document.path.write_text(baseline[:-1] + ',"revision":999}')
    before = document.path.read_bytes()
    with pytest.raises(MeshRefusal):
        document.load()
    assert document.path.read_bytes() == before


def test_fresh_signed_exact_supersession_only_clears_matching_deny(rig: Any) -> None:
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "synthetic-superseded",
        rig.root,
    )
    denial = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
        rig.peer.device_id
    ]["denies"][0]

    def action(data: Any) -> None:
        data["action"] = "supersede"
        data["supersedes"] = [
            {k: copy.deepcopy(denial[k]) for k in ["selector", "revision", "approval_ref"]}
        ]

    aid, run = request(rig, change=action)
    consent.provision_from_run(aid, run, rig.subject, rig.root)
    assert (
        consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
            rig.peer.device_id
        ]["denies"]
        == []
    )
    # Superseding permission still does not recreate any holder automatically.
    entry = placement.PlacementDocument.load(N, rig.root).entry(P)
    assert entry is not None and not entry.is_holder(rig.peer.device_id)


def test_executed_permission_outlives_run_and_window_without_replaying_or_granting(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    aid, run = request(rig)
    consent.provision_from_run(aid, run, rig.subject, rig.root)
    A.mark_connected(aid, run_id=run, step=consent.STEP, root=rig.root)
    source = A.load_record(aid, root=rig.root)
    before = placement.placement_path(N, rig.root).read_bytes()
    # Advancing past normal source-window/terminal-prune age is not a new
    # permission expiry. The signed intent + executed receipt must still exist.
    future = source["expires_at"] + 40 * 86400
    monkeypatch.setattr(A.time, "time", lambda: future)
    A.sweep(root=rig.root, now=future)
    assert A.record_path(aid, rig.root).exists()
    proof = consent.verified_permission(
        N, rig.own.device_id, rig.peer.device_id, P, rig.subject["native_context"], rig.root
    )
    assert proof["kind"] == "permission_only"
    assert placement.placement_path(N, rig.root).read_bytes() == before
    with pytest.raises(MeshRefusal):
        consent.provision_from_run(aid, run, rig.subject, rig.root)
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "synthetic-after-window-revoke",
        rig.root,
    )
    with pytest.raises(MeshRefusal, match="not_a_holder"):
        consent.verified_permission(
            N, rig.own.device_id, rig.peer.device_id, P, rig.subject["native_context"], rig.root
        )


def test_stale_revision_cannot_drop_another_writer_deny(rig: Any) -> None:
    doc = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    old = doc.load()
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "synthetic-cas-deny",
        rig.root,
    )
    with doc.locked(), pytest.raises(MeshRefusal, match="consent_conflict"):
        doc.save(old, old["revision"])
    assert doc.load()["recipients"][rig.peer.device_id]["denies"]


def test_257th_denial_refusal_preserves_readable_authority(rig: Any) -> None:
    document = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    row = document.load()
    row["recipients"][rig.peer.device_id] = {
        "public_key_fingerprint": "",
        "enabled": False,
        "denies": [
            {
                "selector": {"credential_key": P},
                "revision": 1,
                "reason": "withdrawn",
                "approval_ref": "capacity-" + str(i),
            }
            for i in range(256)
        ],
    }
    with document.locked():
        document.save(row, row["revision"])
    before = document.path.read_bytes()
    snapshot = document.load()
    with pytest.raises(MeshRefusal, match="consent_invalid"):
        consent.restrict_and_revoke(
            N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "new-257", rig.root
        )
    assert document.path.read_bytes() == before
    assert document.load() == snapshot
    assert len(document.load()["recipients"][rig.peer.device_id]["denies"]) == 256


@pytest.mark.parametrize("bad", ["recipients", "serialized_bytes", "context", "depth"])
def test_rejected_candidate_preserves_bytes_revision_snapshot_and_tombstones(
    rig: Any, bad: str
) -> None:
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "preserved-denial",
        rig.root,
    )
    document = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    before = document.path.read_bytes()
    persisted = document.load()
    candidate = copy.deepcopy(persisted)
    if bad == "recipients":
        candidate["recipients"] = {
            "d_" + format(i, "032x"): {"public_key_fingerprint": "", "enabled": False, "denies": []}
            for i in range(257)
        }
    elif bad == "serialized_bytes":
        candidate["recipients"][rig.peer.device_id]["public_key_fingerprint"] = "λ" * 140000
    elif bad == "context":
        candidate["network_id"] = "n_" + "f" * 24
    else:
        candidate["recipients"][rig.peer.device_id]["denies"][0]["selector"]["credential_key"] = [
            [[[P]]]
        ]
    untouched = copy.deepcopy(candidate)
    names = sorted(p.name for p in document.path.parent.iterdir())
    with document.locked(), pytest.raises(MeshRefusal):
        document.save(candidate, candidate["revision"])
    assert candidate == untouched
    assert document.path.read_bytes() == before
    assert document.load() == persisted
    assert sorted(p.name for p in document.path.parent.iterdir()) == names


@pytest.mark.parametrize("source", ["app", "token", "gh"])
def test_github_account_ambiguity_refuses_before_any_source_extract_mint_register(
    rig: Any, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    from local_operator.network.credentials import github as github

    monkeypatch.setattr(github, "_resolve_program", lambda *a, **kw: None)
    doc = placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)
    doc.declare(
        github.GITHUB_KEY,
        owner_device=rig.own.device_id,
        owner_device_name="synthetic",
        provider=github.GITHUB_KEY,
        kind=github.GITHUB_KIND,
        by=rig.own.device_id,
    )
    doc.grant(github.GITHUB_KEY, rig.peer.device_id, scope="device", by=rig.own.device_id)
    doc.save()
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": github.GITHUB_KEY, "account_ref": "unknown-forge-account"},
        "forge-account-denial",
        rig.root,
    )
    doc = placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)
    doc.grant(github.GITHUB_KEY, rig.peer.device_id, scope="device", by=rig.own.device_id)
    doc.save()
    calls = []

    def touch(name: str, value: Any) -> Any:
        calls.append(name)
        return value

    class Lender:
        def mint(self, *args: Any) -> Any:
            return touch(
                "mint",
                SimpleNamespace(
                    token="synthetic-unused", expires_at_ms=int((time.time() + 600) * 1000)
                ),
            )

        def register(self, **kwargs: Any) -> None:
            touch("register", None)

    mode = {"app": github.SOURCE_APP, "token": github.SOURCE_TOKEN, "gh": github.SOURCE_GH}[source]
    monkeypatch.setattr(github, "resolve_source", lambda *a: touch("source", mode))
    monkeypatch.setattr(github, "read_app_key", lambda *a: touch("app-key", SimpleNamespace()))
    monkeypatch.setattr(
        github, "read_token_secret", lambda *a: touch("PAT-extract", "synthetic-unused")
    )
    monkeypatch.setattr(github, "read_gh_token", lambda: touch("gh-extract", "synthetic-unused"))
    monkeypatch.setattr(github, "repositories_for", lambda *a: ["synthetic/repo"])
    monkeypatch.setattr(rig.broker, "_lender", lambda: Lender())
    link = SimpleNamespace(device_id=rig.peer.device_id, network_id=N, epoch=1)
    # A caller cannot turn a denied/unknown owner account into a different one.
    frame = dict(
        op="net_broker",
        kind="grant",
        req=1,
        key=github.GITHUB_KEY,
        provider=github.GITHUB_KEY,
        from_device=rig.peer.device_id,
        for_session="same-synthetic-session",
        account_ref="guessed-other-account",
    )
    reply = rig.broker.on_broker(link, frame)["detail"]
    assert reply["kind"] == "error" and reply["code"] == "scope_unknown"
    assert calls == []
    assert "access_token" not in reply
