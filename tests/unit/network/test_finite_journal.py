"""Source fence durability/public recovery, file-only software normal signer."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network.credentials import (
    consent,
)
from local_operator.network.credentials import finite_journal as J
from local_operator.network.credentials import finite_share as F
from local_operator.network.credentials import owner, placement
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_credentials_consent import N, P, grant
from tests.unit.network.test_credentials_consent import rig as consent_rig
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_finite_share import _signer

rig = consent_rig


def _approved(r: Any, mp: pytest.MonkeyPatch) -> dict[str, Any]:
    _signer(r, mp)
    consent.restrict_and_revoke(
        N, r.own.device_id, r.peer.device_id, {"credential_key": P}, "journal-deny", r.root
    )
    rec = F.request(N, r.own.device_id, r.peer.device_id, P, "session", r.root)
    now = time.time()
    A.approve(
        rec["approval_id"],
        signature_hex=A.sign_decision(
            kind=rec["kind"],
            request_id=rec["request_id"],
            request_digest=rec["request_digest"],
            decided_at=now,
        ),
        decided_at=now,
        root=r.root,
    )
    return rec


def _apply(r: Any, rec: dict[str, Any]) -> dict[str, Any]:
    return F.apply(
        rec["approval_id"],
        dict(
            owner_device=r.own.device_id,
            network_id=N,
            recipient_device=r.peer.device_id,
            key=P,
            scope="session",
        ),
        r.root,
    )


def test_lost_deny_cannot_be_regranted_after_reload_and_fresh_public_recovery(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = _approved(rig, monkeypatch)
    original = consent.ConsentDocument.save

    def fault(self: Any, *args: Any, **kw: Any) -> None:
        if any(
            d["selector"]["credential_key"] == P
            for d in args[0]["recipients"][rig.peer.device_id]["denies"]
        ):
            raise OSError("same independent QA policy-only restore failure")
        original(self, *args, **kw)
        raise OSError("same independent QA post-CAS publication failure")

    monkeypatch.setattr(consent.ConsentDocument, "save", fault)
    with pytest.raises(MeshRefusal):
        _apply(rig, rec)
    projection = placement.PlacementDocument.load(N, rig.root).entry(P)
    assert projection is not None and not projection.is_holder(rig.peer.device_id)
    assert (
        consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
            rig.peer.device_id
        ]["denies"]
        == []
    )
    journal = J.Journal(N, rig.own.device_id, rig.root)
    guard = journal.pending(rig.peer.device_id, P)
    assert guard is not None
    # The actual source public CLI and a fresh Owner/same session are blocked
    # while the exact policy restoration fault is STILL ACTIVE, not bypassed.
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
    assert _lop_network(*base, "--json") == 1
    assert json.loads(capsys.readouterr().out)["code"] == "finite_recovery_pending"
    fresh = owner.MeshCredentialBroker(
        root=rig.root,
        self_device=rig.own.device_id,
        self_device_name="reload",
        network_id=N,
        auth_store=rig.auth,
    )
    previous = rig.broker
    rig.broker = fresh
    try:
        reply = grant(rig)
        assert reply["kind"] == "error" and "access_token" not in reply
        # Bulk/onboard production shares this guard even while placement locked:
        with placement.mutate(N, root=rig.root, self_device=rig.own.device_id) as document:
            granted, skipped = placement.grant_key_set(
                document,
                device=rig.peer.device_id,
                keys=[P],
                config=rig.root,
                self_device=rig.own.device_id,
            )
        assert granted == [] and skipped[0]["reason"] == "finite_recovery_pending"
        monkeypatch.setattr(consent.ConsentDocument, "save", original)
        assert _lop_network(*base, "--approval", rec["approval_id"], "--json") == 1
        capsys.readouterr()
        assert _lop_network(*base, "--request-approval", "--json") == 0
        aid = json.loads(capsys.readouterr().out)["approval_id"]
        signed = A.load_record(aid, root=rig.root)["what"][F.OP]
        assert signed["guard"] == {"id": guard["id"], "revision": guard["revision"]}
        assert signed["denies"] == guard["denies"]
        assert _lop_network("approvals", "approve", aid, "--json") == 0
        disclosure = capsys.readouterr()
        assert "journal-deny" in disclosure.err and guard["id"] in disclosure.err
        assert _lop_network(*base, "--approval", aid, "--json") == 0
        capsys.readouterr()
        assert journal.pending(rig.peer.device_id, P) is None
        assert grant(rig)["kind"] == "grant"
    finally:
        rig.broker = previous
        fresh.close()


@pytest.mark.parametrize(
    "phase",
    [
        "initial",
        "pending-readback",
        "projection",
        "receipt",
        "terminal",
        "capability",
        "completion",
    ],
)
def test_crash_boundary_keeps_fence_or_aborts_before_mutation(
    rig: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    rec = _approved(rig, monkeypatch)
    old_policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    old_projection = placement.placement_path(N, rig.root).read_bytes()
    if phase == "initial":
        monkeypatch.setattr(
            J.Journal,
            "_publish",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("initial journal write unavailable")),
        )
    elif phase == "pending-readback":
        original = J.Journal.save

        def pending_fault(self: Any, *a: Any, **kw: Any) -> Any:
            original(self, *a, **kw)
            raise OSError("pending published then no acknowledgement")

        monkeypatch.setattr(J.Journal, "save", pending_fault)
    else:
        obj, attr = {
            "projection": (placement.PlacementDocument, "save"),
            "receipt": (A, "_append_receipt_locked"),
            "terminal": (A, "_mark_connected_locked"),
            "capability": (F, "_complete_capability"),
            "completion": (J.Journal, "_finish"),
        }[phase]

        def crash(*a: Any, **kw: Any) -> None:
            raise SystemExit("synthetic hard boundary; no compensation ran")

        monkeypatch.setattr(obj, attr, crash)
    with pytest.raises((MeshRefusal, OSError, SystemExit)):
        _apply(rig, rec)
    if phase == "initial":
        assert (
            consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == old_policy
        )
        assert placement.placement_path(N, rig.root).read_bytes() == old_projection
    else:
        assert J.Journal(N, rig.own.device_id, rig.root).pending(rig.peer.device_id, P) is not None
        with pytest.raises(MeshRefusal):
            consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)


@pytest.mark.parametrize(
    "corruption", ["missing-journal", "missing-marker", "corrupt", "duplicate", "symlink"]
)
def test_initialized_unknown_store_never_becomes_empty_legacy(
    rig: Any, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    rec = _approved(rig, monkeypatch)
    monkeypatch.setattr(
        placement.PlacementDocument,
        "save",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("leave pending")),
    )
    with pytest.raises(SystemExit):
        _apply(rig, rec)
    journal = J.Journal(N, rig.own.device_id, rig.root)
    if corruption == "missing-journal":
        journal.path.unlink()
    elif corruption == "missing-marker":
        journal.sentinel.unlink()
    elif corruption == "corrupt":
        journal.path.write_text("{bad")
    elif corruption == "duplicate":
        journal.path.write_text('{"schema":1,"schema":1}')
    else:
        original = journal.path.with_name("saved-test-only-journal.json")
        journal.path.rename(original)
        journal.path.symlink_to(original)
    with pytest.raises(MeshRefusal):
        consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)


def test_complete_disclosure_is_before_signer_and_stdout_stays_one_object(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _signer(rig, monkeypatch)
    for i in range(3):
        try:
            consent.restrict_and_revoke(
                N,
                rig.own.device_id,
                rig.peer.device_id,
                {"credential_key": P, "account_ref": "account-" + str(i)},
                "disclosed-" + str(i),
                rig.root,
            )
        except MeshRefusal as error:
            # Restriction FIRST is durable even when no remaining holder exists;
            # keep these real negative rows, not a manufactured approval payload.
            assert error.code == "not_a_holder"
    rec = F.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)
    expected = F.details(rec)
    software = A.sign_decision
    observed = []

    def before_signer(**kw: Any) -> str:
        stream = capsys.readouterr()
        observed.append(stream.err)
        assert stream.out == "" and all(line in stream.err for line in expected)
        return software(**kw)

    monkeypatch.setattr(A, "sign_decision", before_signer)
    assert _lop_network("approvals", "approve", rec["approval_id"], "--json") == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["state"] == "approved"
    assert len(observed) == 1


@pytest.mark.parametrize("bad", ["expired", "denied", "scope", "fingerprint", "revision"])
def test_finite_refusal_keeps_codes_and_exact_recovery_not_onboarding(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    rec = _approved(rig, monkeypatch)
    if bad == "expired":
        monkeypatch.setattr(F.time, "time", lambda: rec["expires_at"] + 1)
    if bad == "denied":
        A.deny(rec["approval_id"], root=rig.root)
    if bad == "revision":
        consent.restrict_and_revoke(
            N,
            rig.own.device_id,
            rig.peer.device_id,
            {"credential_key": "other"},
            "new-withdrawal",
            rig.root,
        )
    if bad == "fingerprint":
        from local_operator.network import store

        network = store.load(N, rig.root)
        peer = network.member(rig.peer.device_id)
        assert peer is not None
        peer.public_key = rig.own.public_key
        store.save(network, rig.root)
    scope = "device" if bad == "scope" else "session"
    assert (
        _lop_network(
            "credential",
            "share",
            P,
            "--with",
            rig.peer.device_id,
            "--network",
            N,
            "--scope",
            scope,
            "--approval",
            rec["approval_id"],
            "--json",
        )
        == 1
    )
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert (
        "--request-approval" in payload["message"]
        and "approvals approve" in payload["message"]
        and "--approval" in payload["message"]
    )
    assert "new onboarding" not in payload["message"] and "setup" not in payload["message"]


@pytest.mark.parametrize(
    "bad",
    [
        "records",
        "denies",
        "bytes",
        "duplicate-tuple",
        "status",
        "revision",
        "foreign-owner",
        "extra-positive-field",
    ],
)
def test_owner_journal_bounds_and_schema_fail_closed_without_eviction(
    rig: Any, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    rec = _approved(rig, monkeypatch)

    def crash(*a: Any, **k: Any) -> None:
        raise SystemExit("pending persisted")

    monkeypatch.setattr(placement.PlacementDocument, "save", crash)
    with pytest.raises(SystemExit):
        _apply(rig, rec)
    journal = J.Journal(N, rig.own.device_id, rig.root)
    row = journal.load()
    record = row["records"][0]
    if bad == "records":
        row["records"] = [
            {
                **copy.deepcopy(record),
                "id": "tx-" + str(i),
                "key": "key-" + str(i),
                "denies": [
                    {
                        **copy.deepcopy(record["denies"][0]),
                        "selector": {"credential_key": "key-" + str(i)},
                    }
                ],
            }
            for i in range(65)
        ]
    elif bad == "denies":
        row["records"] = [
            {
                **copy.deepcopy(record),
                "id": "tx-" + str(i),
                "key": "key-" + str(i),
                "denies": [
                    {
                        "selector": {"credential_key": "key-" + str(i)},
                        "revision": 1,
                        "approval_ref": "deny-" + str(j),
                        "reason": "withdrawn",
                    }
                    for j in range(65)
                ],
            }
            for i in range(4)
        ]
    elif bad == "duplicate-tuple":
        record["denies"].append(copy.deepcopy(record["denies"][0]))
    elif bad == "status":
        record["status"] = "enabled"
    elif bad == "revision":
        record["revision"] = True
    elif bad == "foreign-owner":
        row["owner_device"] = rig.peer.device_id
    elif bad == "extra-positive-field":
        record["enabled"] = True
    raw = " " * (J.LIMIT + 1) if bad == "bytes" else json.dumps(row)
    journal.path.write_text(raw)
    before = journal.path.read_bytes()
    with pytest.raises(MeshRefusal):
        consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)
    assert journal.path.read_bytes() == before


def test_fence_precedes_absent_consent_and_unrelated_key_survives(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _approved(rig, monkeypatch)

    def crash(*a: Any, **k: Any) -> None:
        raise SystemExit("pending persisted")

    monkeypatch.setattr(placement.PlacementDocument, "save", crash)
    with pytest.raises(SystemExit):
        _apply(rig, rec)
    consent.ConsentDocument(N, rig.own.device_id, rig.root).path.unlink()
    with pytest.raises(MeshRefusal) as error:
        consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)
    assert error.value.code == "finite_recovery_pending"
    consent.check_borrow(
        N, rig.own.device_id, rig.peer.device_id, "unrelated-owned-key", {}, rig.root
    )


def test_uncertain_final_journal_publish_retains_fence(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _approved(rig, monkeypatch)
    original = J.Journal.save

    def fault(self: Any, row: Any, expected: int) -> Any:
        result = original(self, row, expected)
        if any(record["status"] == "complete" for record in row["records"]):
            raise OSError("synthetic unconfirmed acknowledgement after final journal publish")
        return result

    monkeypatch.setattr(J.Journal, "save", fault)
    with pytest.raises(MeshRefusal):
        _apply(rig, rec)
    assert J.Journal(N, rig.own.device_id, rig.root).pending(rig.peer.device_id, P) is not None
    with pytest.raises(MeshRefusal):
        consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)


def test_legacy_owner_without_fence_files_is_unchanged(rig: Any) -> None:
    journal = J.Journal(N, rig.own.device_id, rig.root)
    assert not journal.path.exists() and not journal.sentinel.exists()
    # A normal borrow on a never-fenced owner must neither refuse nor create files.
    consent.check_borrow(N, rig.own.device_id, rig.peer.device_id, P, {}, rig.root)
    assert not journal.path.exists() and not journal.sentinel.exists()
    assert journal.peek(rig.peer.device_id, P) is None
