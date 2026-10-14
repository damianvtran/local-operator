"""Public source CLI finite recovery; software operator fixture, no hardware signer."""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network import store
from local_operator.network.credentials import consent, finite_share, placement
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_approvals_store import _sign
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig
from tests.unit.network.test_credentials_real_link import _lop_network

# Reuse the existing software-authority fixture, not a new signer stack.
rig = consent_rig


def _signer(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def software(**kw: Any) -> str:
        return _sign(
            rig.root,
            A.signed_payload(
                kind=kw["kind"],
                request_id=kw["request_id"],
                request_digest=kw["request_digest"],
                decision="approve",
                decided_at=kw["decided_at"],
            ),
        )

    monkeypatch.setattr(A, "sign_decision", software)
    # A real local admin member, not merely a displayed self_role label, is
    # required by the existing public finite capability-authority gate.
    from local_operator.network.types import MemberRecord

    network = store.load(N, rig.root)
    network.members.append(
        MemberRecord(
            rig.own.device_id, public_key=rig.own.public_key, role="admin", capabilities=["admin"]
        )
    )
    store.save(network, rig.root)


def test_public_request_approve_apply_restores_only_exact_finite_share(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "finite-one", rig.root
    )
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": "unrelated"},
        "finite-other",
        rig.root,
    )
    base = [
        "credential",
        "share",
        P,
        "--with",
        rig.peer.device_id,
        "--network",
        N,
        "--scope",
        "session",
    ]
    assert _lop_network(*base) == 1
    assert "request-approval" in capsys.readouterr().err
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    assert _lop_network(*base, "--request-approval", "--json") == 0
    output = capsys.readouterr()
    payload = json.loads(output.out)
    aid = payload["approval_id"]
    assert payload["state"] == "requested" and "FINITE" in payload["summary"]
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).load() == before
    assert _lop_network("approvals", "run", aid, "--json") == 1
    assert "finite-share" in capsys.readouterr().err
    assert _lop_network("approvals", "approve", aid, "--json") == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["state"] == "approved"
    assert "No automatic future-sign-in inheritance enabled" in output.err
    assert _lop_network(*base, "--approval", aid, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    row = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
        rig.peer.device_id
    ]
    assert row["enabled"] is False and "inherit" not in row
    assert [d["selector"]["credential_key"] for d in row["denies"]] == ["unrelated"]
    entry = placement.PlacementDocument.load(N, rig.root).entry(P)
    assert entry is not None
    holder = entry.holder(rig.peer.device_id)
    assert holder is not None and holder.scope == "session"
    source = A.load_record(aid, root=rig.root)
    assert source["state"] == "connected"
    assert any(r["step"] == finite_share.STEP and r["ok"] for r in source["receipts"])
    unchanged = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    assert _lop_network(*base, "--approval", aid) == 1
    capsys.readouterr()
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == unchanged


@pytest.mark.parametrize(
    "bad",
    [
        "wrong-peer",
        "key",
        "scope",
        "revision",
        "bad-signature",
        "digest",
        "fingerprint",
        "expired",
        "cancel",
    ],
)
def test_invalid_public_source_cannot_clear_or_grant(
    rig: Any, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "negative-one", rig.root
    )
    source = finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)
    aid = source["approval_id"]
    import time

    moment = time.time()
    signature = A.sign_decision(
        kind=source["kind"],
        request_id=source["request_id"],
        request_digest=source["request_digest"],
        decided_at=moment,
    )
    A.approve(aid, signature_hex=signature, decided_at=moment, root=rig.root)
    expected = dict(
        owner_device=rig.own.device_id,
        network_id=N,
        recipient_device=rig.peer.device_id,
        key=P,
        scope="session",
    )
    if bad == "wrong-peer":
        expected["recipient_device"] = rig.own.device_id
    if bad == "key":
        expected["key"] = "different-owned-key"
    if bad == "scope":
        expected["scope"] = "device"
    if bad == "fingerprint":
        network = store.load(N, rig.root)
        peer = network.member(rig.peer.device_id)
        assert peer is not None
        peer.public_key = rig.own.public_key
        store.save(network, rig.root)
    if bad == "digest":
        record = A.load_record(aid, root=rig.root)
        record["what"][finite_share.OP]["scope"] = "device"
        A._write_record(record, rig.root)
    if bad == "revision":
        consent.restrict_and_revoke(
            N,
            rig.own.device_id,
            rig.peer.device_id,
            {"credential_key": "extra"},
            "new-rev",
            rig.root,
        )
    if bad == "bad-signature":
        record = A.load_record(aid, root=rig.root)
        record["signature"]["sig"] = "00" * 64
        A._write_record(record, rig.root)
    if bad == "expired":
        monkeypatch.setattr(A.time, "time", lambda: source["expires_at"] + 1)
    if bad == "cancel":
        A.deny(aid, root=rig.root)
    path = consent.ConsentDocument(N, rig.own.device_id, rig.root).path
    before = path.read_bytes()
    projection = placement.placement_path(N, rig.root).read_bytes()
    with pytest.raises(MeshRefusal):
        finite_share.apply(aid, expected, rig.root)
    assert (
        path.read_bytes() == before
        and placement.placement_path(N, rig.root).read_bytes() == projection
    )


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "a/b", "a\\b", "C:thing", "a\x00b", "a\nb", "x" * 257]
)
def test_storage_components_reject_path_syntax_without_normalization(bad: str) -> None:
    with pytest.raises(MeshRefusal):
        store.safe_component(bad)


def test_safe_opaque_ids_retain_exact_bytes() -> None:
    for value in ["n_copy_receipts", "d_legacy-owner", "opaque ID", "é"]:
        assert store.safe_component(value) == value


@pytest.mark.parametrize("phase", ["projection", "receipt", "terminal", "authority", "late-expiry"])
def test_partial_operation_restores_deny_and_original_projection(
    rig: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "fault-deny", rig.root
    )
    source = finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)
    import time

    moment = time.time()
    sig = A.sign_decision(
        kind=source["kind"],
        request_id=source["request_id"],
        request_digest=source["request_digest"],
        decided_at=moment,
    )
    A.approve(source["approval_id"], signature_hex=sig, decided_at=moment, root=rig.root)
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    old = placement.placement_path(N, rig.root).read_bytes()
    calls = 0
    if phase == "projection":
        obj, attr = placement.PlacementDocument, "save"
    elif phase == "receipt":
        obj, attr = A, "_append_receipt_locked"
    elif phase in {"terminal", "late-expiry"}:
        obj, attr = A, "_mark_connected_locked"
    else:
        obj, attr = consent.ConsentDocument, "save"
    original = getattr(obj, attr)

    def fault(*args: Any, **kw: Any) -> Any:
        nonlocal calls
        result = original(*args, **kw)
        calls += 1
        if calls == 1:
            if phase == "late-expiry":
                monkeypatch.setattr(finite_share.time, "time", lambda: source["expires_at"] + 1)
            else:
                raise OSError("synthetic post-publication failure")
        return result

    monkeypatch.setattr(obj, attr, fault)
    with pytest.raises(MeshRefusal):
        finite_share.apply(
            source["approval_id"],
            dict(
                owner_device=rig.own.device_id,
                network_id=N,
                recipient_device=rig.peer.device_id,
                key=P,
                scope="session",
            ),
            rig.root,
        )
    after = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    assert after["recipients"] == before["recipients"]
    assert placement.placement_path(N, rig.root).read_bytes() == old
    with pytest.raises(MeshRefusal):
        consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)


def test_finite_cannot_revive_existing_future_permission(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.network.test_credentials_consent import request as future_request

    _signer(rig, monkeypatch)
    aid, run = future_request(rig)
    consent.provision_from_run(aid, run, rig.subject, rig.root)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "future-deny", rig.root
    )
    path = consent.ConsentDocument(N, rig.own.device_id, rig.root).path
    before = path.read_bytes()
    with pytest.raises(MeshRefusal) as error:
        finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)
    assert error.value.code == "finite_future_scope_conflict"
    assert path.read_bytes() == before
    rec = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
        rig.peer.device_id
    ]
    # Existing approved finite IDs/URLs prove exclusions without asserting that
    # these spellings are native classes, or guessing a secret/account identity.
    from local_operator.network.credentials.types import credential_key_for_mcp

    for outside in (
        "github",
        "secret:existing-finite",
        credential_key_for_mcp("https://other.invalid"),
    ):
        finite_share._reject_future_overlap(rec, outside)


def test_failed_deny_compensation_persists_withheld_holder(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "persistent-fault",
        rig.root,
    )
    source = finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)
    import time

    now = time.time()
    A.approve(
        source["approval_id"],
        signature_hex=A.sign_decision(
            kind=source["kind"],
            request_id=source["request_id"],
            request_digest=source["request_digest"],
            decided_at=now,
        ),
        decided_at=now,
        root=rig.root,
    )
    original = consent.ConsentDocument.save

    def fail_after_publish(self: Any, *args: Any, **kw: Any) -> None:
        if any(
            d["selector"]["credential_key"] == P
            for d in args[0]["recipients"][rig.peer.device_id]["denies"]
        ):
            raise OSError("synthetic persistent policy-only write failure")
        original(self, *args, **kw)
        raise OSError("synthetic failure after final CAS publish")

    monkeypatch.setattr(consent.ConsentDocument, "save", fail_after_publish)
    with pytest.raises(MeshRefusal) as error:
        finite_share.apply(
            source["approval_id"],
            dict(
                owner_device=rig.own.device_id,
                network_id=N,
                recipient_device=rig.peer.device_id,
                key=P,
                scope="session",
            ),
            rig.root,
        )
    assert error.value.code == "finite_projection_withheld"
    entry = placement.PlacementDocument.load(N, rig.root).entry(P)
    assert entry is not None and not entry.is_holder(rig.peer.device_id)
