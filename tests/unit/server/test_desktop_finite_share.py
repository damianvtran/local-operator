"""Finite desktop disclosure and admission over disposable software authority.

The HTTP client stays in-process and the existing finite-policy rig owns every
credential and signature. No live card, daemon or OS signing prompt is touched.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import approvals as A
from local_operator.network import onboard_approvals
from local_operator.network.types import MeshRefusal
from local_operator.network.credentials import consent, finite_share, placement
from local_operator.server.routes import desktop_approvals
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.network.test_approvals_store import _device_request
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig
from tests.unit.network.test_finite_share import _signer

rig = consent_rig
TOKEN = "synthetic-finite-desktop-token"
_REAL_SIGN_DECISION = A.sign_decision


@pytest.fixture(autouse=True)
def guard_native_signing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_operator.operator import keychain, sign
    from local_operator.operator.macos import keyagent

    monkeypatch.setenv("HOME", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("native_operator_signing_forbidden")

    choose = keychain.choose_backend

    def disposable_backend(preference, *, config_root):
        # The existing rig still performs real P-256 signing and verification.
        # A missed route mock must not probe an OS key or leave this fixture root.
        assert preference == keychain.FILE_ONLY
        assert Path(config_root).resolve().is_relative_to(tmp_path.resolve())
        return choose(preference, config_root=config_root)

    monkeypatch.setattr(keychain, "choose_backend", disposable_backend)
    monkeypatch.setattr(sign, "choose_backend", disposable_backend)
    monkeypatch.setattr(sign, "resolve_backend_name", forbidden)
    # These are the actual native helper launch/preflight seams, not Popen-only
    # substitutes. Guarding both prevents a missed upper alias from prompting.
    monkeypatch.setattr(keyagent.KeyagentClient, "_spawn", forbidden)
    monkeypatch.setattr(keyagent, "_run", forbidden)


@pytest_asyncio.fixture
async def finite_api(rig: Any, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(rig.root.parent))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    _signer(rig, monkeypatch)
    app = FastAPI()
    app.state.config_manager = ConfigManager(rig.root)
    app.state.desktop_sessions = DesktopSessions(rig.root)
    app.include_router(desktop_approvals.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


def request(rig: Any) -> dict[str, Any]:
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P},
        "desktop-finite-withdrawal", rig.root,
    )
    return finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "device", rig.root)


def acknowledgment(record: dict[str, Any]) -> dict[str, Any]:
    return {"authority_version": 1, "scope": "finite_share", "request_digest": record["request_digest"]}


@pytest.mark.asyncio
async def test_finite_http_approve_is_not_apply_and_replay_cannot_regrant(finite_api, rig):
    source = request(rig)
    path = f"/v1/desktop/approvals/{source['approval_id']}"
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    listed = await finite_api.get("/v1/desktop/approvals")
    row = next(row for row in listed.json()["result"]["approvals"] if row["approval_id"] == source["approval_id"])
    assert row["can_approve_here"] is False
    assert row["finite_share"]["valid"] is True
    assert row["finite_share"]["public_key_fingerprint"] == source["what"][finite_share.OP]["public_key_fingerprint"]
    legacy = await finite_api.post(path + "/approve")
    assert legacy.status_code == 409
    signed = await finite_api.post(path + "/approve-finite-share", json=acknowledgment(source))
    assert signed.status_code == 200, signed.text
    assert signed.json()["result"]["state"] == "approved"
    # A desktop signature remains finite-only authority, never permission for
    # the generic remote install/join runner to consume this same envelope.
    with pytest.raises(MeshRefusal) as refused:
        onboard_approvals.begin_run(source["approval_id"], root=rig.root)
    assert refused.value.code == "finite_apply_required"
    assert A.load_record(source["approval_id"], root=rig.root)["state"] == "approved"
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == before
    applied = await finite_api.post(path + "/apply-finite-share", json=acknowledgment(source))
    assert applied.status_code == 200, applied.text
    assert applied.json()["result"]["state"] == "applied"
    policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    recipient = policy["recipients"][rig.peer.device_id]
    # Only inheritance SETTINGS stay unchanged. The restored explicit key
    # holder can still receive later same-key sign-ins or replacement material.
    assert recipient["enabled"] is False and "inherit" not in recipient
    holder = placement.PlacementDocument.load(N, rig.root).entry(P).holder(rig.peer.device_id)
    assert holder is not None and holder.scope == "device"
    after = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    replay = await finite_api.post(path + "/apply-finite-share", json=acknowledgment(source))
    assert replay.status_code == 409
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == after


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["absent-auth", "wrong-digest", "extra-selector", "wrong-scope", "wrong-version"])
async def test_finite_http_refuses_before_signer(finite_api, rig, monkeypatch, bad):
    source = request(rig)
    body = acknowledgment(source)
    headers = {}
    if bad == "absent-auth":
        headers["Authorization"] = ""
    elif bad == "wrong-digest":
        body["request_digest"] = "sha256:" + "0" * 64
    elif bad == "extra-selector":
        body["recipient_device"] = "another-device"
    elif bad == "wrong-scope":
        body["scope"] = "future_logins"
    else:
        body["authority_version"] = 2
    calls = []
    monkeypatch.setattr(A, "sign_decision", lambda **kwargs: calls.append(kwargs))
    response = await finite_api.post(
        f"/v1/desktop/approvals/{source['approval_id']}/approve-finite-share",
        json=body, headers=headers,
    )
    assert response.status_code in {401, 403, 409, 422}
    assert calls == []
    assert A.load_record(source["approval_id"], root=rig.root)["state"] == "requested"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["malformed", "mixed-inherit", "mixed-install", "fingerprint"])
async def test_invalid_finite_presence_never_falls_back_to_generic_signing(finite_api, rig, monkeypatch, bad):
    source = request(rig)
    what = copy.deepcopy(source["what"])
    if bad == "malformed":
        what[finite_share.OP] = None
    elif bad == "mixed-inherit":
        what["credential_inheritance"] = {}
    elif bad == "mixed-install":
        what["install"] = True
    else:
        what[finite_share.OP]["public_key_fingerprint"] = "invalid"
    record = A.create_request(**_device_request(A.new_request_id(), what=what), root=rig.root)
    calls = []
    monkeypatch.setattr(A, "sign_decision", lambda **kwargs: calls.append(kwargs))
    for endpoint in ("approve", "approve-finite-share"):
        response = await finite_api.post(
            f"/v1/desktop/approvals/{record['approval_id']}/{endpoint}",
            json=acknowledgment(record) if endpoint != "approve" else None,
        )
        assert response.status_code == 409
    assert calls == []
    rows = (await finite_api.get("/v1/desktop/approvals")).json()["result"]["approvals"]
    row = next(row for row in rows if row["approval_id"] == record["approval_id"])
    assert row["finite_share"]["valid"] is False and row["can_approve_here"] is False


@pytest.mark.asyncio
async def test_finite_apply_keeps_existing_policy_revision_guard(finite_api, rig):
    source = request(rig)
    path = f"/v1/desktop/approvals/{source['approval_id']}"
    signed = await finite_api.post(path + "/approve-finite-share", json=acknowledgment(source))
    assert signed.status_code == 200
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P},
        "newer-desktop-withdrawal", rig.root,
    )
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    response = await finite_api.post(path + "/apply-finite-share", json=acknowledgment(source))
    assert response.status_code == 409
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == before


@pytest.mark.asyncio
async def test_finite_http_authentication_also_gates_application(finite_api, rig):
    source = request(rig)
    path = f"/v1/desktop/approvals/{source['approval_id']}"
    assert (await finite_api.post(path + "/approve-finite-share", json=acknowledgment(source))).status_code == 200
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    response = await finite_api.post(
        path + "/apply-finite-share", json=acknowledgment(source), headers={"Authorization": ""}
    )
    assert response.status_code in {401, 403}
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == before


def test_real_signer_escape_is_blocked_before_native_access(rig):
    from local_operator.operator.macos import keyagent

    with pytest.raises(AssertionError, match="native_operator_signing_forbidden"):
        _REAL_SIGN_DECISION(
            kind="device_onboard", request_id="synthetic-escape",
            request_digest="sha256:" + "a" * 64, decided_at=1.0,
        )
    with pytest.raises(AssertionError, match="native_operator_signing_forbidden"):
        object.__new__(keyagent.KeyagentClient)._spawn([], timeout=1)
    with pytest.raises(AssertionError, match="native_operator_signing_forbidden"):
        keyagent._run([])


@pytest.mark.asyncio
async def test_finite_http_rejects_invalid_signature_with_real_verifier(finite_api, rig, monkeypatch):
    source = request(rig)
    before = consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes()
    monkeypatch.setattr(A, "sign_decision", lambda **kwargs: "00" * 64)
    response = await finite_api.post(
        f"/v1/desktop/approvals/{source['approval_id']}/approve-finite-share",
        json=acknowledgment(source),
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "approval_signature_invalid"
    assert A.load_record(source["approval_id"], root=rig.root)["state"] == "requested"
    assert consent.ConsentDocument(N, rig.own.device_id, rig.root).path.read_bytes() == before


@pytest.mark.asyncio
async def test_expired_finite_snapshot_never_invokes_signer(finite_api, rig, monkeypatch):
    source = request(rig)
    import time

    monkeypatch.setattr(time, "time", lambda: source["expires_at"] + 1)
    calls = []
    monkeypatch.setattr(A, "sign_decision", lambda **kwargs: calls.append(kwargs))
    response = await finite_api.post(
        f"/v1/desktop/approvals/{source['approval_id']}/approve-finite-share",
        json=acknowledgment(source),
    )
    assert response.status_code == 409
    assert calls == []
