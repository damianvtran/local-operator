"""Public source CLI finite recovery; software operator fixture, no hardware signer."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import approvals as A
from local_operator.network import store
from local_operator.network.credentials import (
    consent,
    finite_share,
    inheritance,
    placement,
)
from local_operator.network.types import MeshRefusal
from local_operator.server.routes import desktop_approvals as desktop_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.network.test_approvals_store import _device_request, _sign
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
    assert payload["state"] == "requested" and "Restore sharing of" in payload["summary"]
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).load() == before
    assert _lop_network("approvals", "run", aid, "--json") == 1
    assert "finite-share" in capsys.readouterr().err
    assert _lop_network("approvals", "approve", aid, "--json") == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["state"] == "approved"
    assert "No automatic future sign-ins" in " ".join(output.err.split())
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


# ---------------------------------------------------------------------------
# Desktop containment: typed credential requests never reach the generic signer
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _desktop_client(rig: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """The real bearer-gated API skin over this fixture's isolated store."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "finite-desktop-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(rig.root)
    app.state.desktop_sessions = DesktopSessions(rig.root)
    app.include_router(desktop_routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer finite-desktop-token"},
    ) as client:
        yield client


def _finite_surface(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, int]]:
    """A real finite request and counters around the two forbidden side effects.

    The signer remains the real file-only implementation. If a guard regresses,
    the counter proves it ran; the signature is not fabricated and verification
    remains available as an independent control. The details counter separately
    distinguishes immediate typed-intent containment from CLI validation.
    """
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "desktop-finite", rig.root
    )
    valid = finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)
    assert finite_share.details(valid)
    software = A.sign_decision
    validate = finite_share.details
    counters = {"signer": 0, "details": 0}

    def counted_signer(**kw: Any) -> str:
        counters["signer"] += 1
        return software(**kw)

    def counted_details(record: dict[str, Any]) -> list[str]:
        counters["details"] += 1
        return validate(record)

    monkeypatch.setattr(A, "sign_decision", counted_signer)
    monkeypatch.setattr(finite_share, "details", counted_details)
    return valid, counters


def _record_with_what(source: dict[str, Any], what: dict[str, Any], root: Any) -> dict[str, Any]:
    """Reuse the reproduction's valid generic envelope and anchor provenance."""
    return A.create_request(
        kind=source["kind"],
        request_id=A.new_request_id(),
        requested_by=copy.deepcopy(source["requested_by"]),
        device=copy.deepcopy(source["device"]),
        what=what,
        credential_ref=None,
        created_at=time.time(),
        expires_at=time.time() + 3600,
        root=root,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    ["valid", "empty-withdrawals", "malformed-operation", "mixed-unknown", "mixed-install"],
)
async def test_desktop_finite_refusal_precedes_signer_and_validator(
    rig: Any, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    valid, counters = _finite_surface(rig, monkeypatch)
    what = copy.deepcopy(valid["what"])
    if shape == "empty-withdrawals":
        what[finite_share.OP]["denies"] = []
    elif shape == "malformed-operation":
        # intent() returns None for this shape. Key-presence containment must
        # refuse it anyway rather than treating it as ordinary onboarding.
        what[finite_share.OP] = ["not", "a", "typed", "payload"]
    elif shape == "mixed-unknown":
        # R1: a VALID finite intent beside an unknown credential operation. The CLI
        # validator reads only the finite half, so its remedy would sign the rest blind.
        what["credential_unknown_typed_operation"] = {"scope": "unknown"}
    elif shape == "mixed-install":
        what["install"] = True
    record = valid if shape == "valid" else _record_with_what(valid, what, rig.root)
    policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()

    async with _desktop_client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")
        assert response.status_code == 409, response.text
        detail = response.json()["detail"]
        # ``empty-withdrawals`` joined the unsafe set in #3087 r1 (R1-1): the row count is
        # part of the shared ``exact_shape``, so ``/approve`` now gives the CLI's own answer
        # instead of directing the operator to a CLI whose disclosure then refuses it.
        unsafe = shape in {
            "empty-withdrawals",
            "malformed-operation",
            "mixed-unknown",
            "mixed-install",
        }
        expected_code = (
            "finite_disclosure_invalid" if unsafe else "finite_terminal_approval_required"
        )
        assert detail["code"] == expected_code
        if unsafe:
            # Never direct a shape the finite validator cannot fully disclose to the CLI.
            assert "lop network approvals approve" not in detail["text"]
        else:
            assert "complete withdrawal disclosure" in detail["text"]
            assert f"lop network approvals approve {record['approval_id']}" in detail["text"]
        if shape == "valid":
            denied = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/deny")
            assert denied.status_code == 200, denied.text
            assert denied.json()["result"]["state"] == "denied"

    expected_state = "denied" if shape == "valid" else "requested"
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == expected_state
    assert counters == {"signer": 0, "details": 0}
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == policy


@pytest.mark.asyncio
async def test_desktop_unknown_credential_intent_fails_closed_without_cli_remedy(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid, counters = _finite_surface(rig, monkeypatch)
    what = {
        "anchor": copy.deepcopy(valid["what"]["anchor"]),
        "credential_unknown_typed_operation": {"scope": "unknown"},
    }
    record = _record_with_what(valid, what, rig.root)

    async with _desktop_client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "credential_approval_unsupported"
    assert "lop network approvals approve" not in detail["text"]
    assert counters == {"signer": 0, "details": 0}
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == "requested"


@pytest.mark.asyncio
async def test_desktop_inheritance_refusal_is_unchanged(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid, counters = _finite_surface(rig, monkeypatch)
    what = {
        "anchor": copy.deepcopy(valid["what"]["anchor"]),
        inheritance.OP: copy.deepcopy(valid["what"][finite_share.OP]),
    }
    record = _record_with_what(valid, what, rig.root)

    async with _desktop_client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "inherit_terminal_approval_required"
    assert counters == {"signer": 0, "details": 0}
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == "requested"


@pytest.mark.asyncio
async def test_desktop_ordinary_onboarding_still_signs_and_verifies(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid, counters = _finite_surface(rig, monkeypatch)
    fields = _device_request(A.new_request_id())
    fields["what"]["anchor"] = copy.deepcopy(valid["what"]["anchor"])
    ordinary = A.create_request(**fields, root=rig.root)

    async with _desktop_client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{ordinary['approval_id']}/approve")

    assert response.status_code == 200, response.text
    assert response.json()["result"]["state"] == "approved"
    assert counters == {"signer": 1, "details": 0}
    assert A.load_record(ordinary["approval_id"], root=rig.root)["state"] == "approved"
    A.verify_for_run(ordinary["approval_id"], root=rig.root)


def test_cli_finite_approval_still_validates_then_signs_without_apply(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    valid, counters = _finite_surface(rig, monkeypatch)
    policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    events: list[str] = []
    software = A.sign_decision
    validate = finite_share.details

    def ordered_signer(**kw: Any) -> str:
        events.append("signer")
        return software(**kw)

    def ordered_details(record: dict[str, Any]) -> list[str]:
        events.append("details")
        return validate(record)

    monkeypatch.setattr(A, "sign_decision", ordered_signer)
    monkeypatch.setattr(finite_share, "details", ordered_details)

    assert _lop_network("approvals", "approve", valid["approval_id"], "--json") == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["state"] == "approved"
    assert "Withdrawal 1:" in output.err
    # The disclosure is rendered ONCE, before the one signature (UX #3002 r1, U5: the
    # post-decision block used to repeat it). That ordering is the discriminating part.
    assert events == ["details", "signer"]
    assert counters == {"signer": 1, "details": 1}
    assert A.load_record(valid["approval_id"], root=rig.root)["state"] == "approved"
    A.verify_for_run(valid["approval_id"], root=rig.root)
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == policy
