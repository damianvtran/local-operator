"""Pairing design v2 §8 E2E steps 2-3: a Python P-256 phone against the REAL relay and
desktop HTTP apps, over one isolated config root.

WHAT IS REAL: the relay's Starlette app (``mobile.daemon.build_app``) behind its own
password gate; the desktop FastAPI routers (``desktop_approvals`` +
``desktop_credentials``) behind the desktop bearer; the approval store, the typed engine,
``admit_device``, certificate issuance and the relay's device-certificate check; an
RS256 edge assertion verified against a pinned JWKS in ``tunnels/config.json`` (the
gateway forward is pinned separately in ``test_gateway_pair_assertion``).

WHAT IS STUBBED, AND WHY (the spec allows a stub gesture runner): the OS admin prompt
(``elevate.run_elevated``) and the root ownership of the installed anchor -- no CI host
can answer a GUI prompt, and the anchor's path is redirected into the test root. The
stub writes the staged bytes exactly as the privileged script would, or answers
"cancelled". A REAL sudo/pkexec/osascript prompt needs a person and is out of CI's
reach: that row is real-path evidence on the PR, not this file.

Runs on Ubuntu CI with no network, no keychain and no sudo.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI
from starlette.testclient import TestClient

from local_operator.config import ConfigManager
from local_operator.mobile import pair_requests, push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval
from local_operator.operator import devices, pair_op
from local_operator.operator.elevate import ElevationResult
from local_operator.server import desktop
from local_operator.server.routes import desktop_approvals as approval_routes
from local_operator.server.routes import desktop_credentials as operation_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.network.test_approvals_store import _make_key, _sign
from tests.unit.network.test_approvals_store import root as approvals_root  # noqa: F401

HOST = "phone.example.test"
OWNER = "acct_owner_e2e"
TOKEN = "e2e-desktop-bearer"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ---------------------------------------------------------------------------
# The phone: a P-256 key, the SAS it derives, and its HTTP calls
# ---------------------------------------------------------------------------


class Phone:
    """A phone in the PWA's/app's shape: its own key, its own SAS derivation."""

    def __init__(self, relay: TestClient, *, edge: "Edge | None" = None) -> None:
        self.relay = relay
        self.edge = edge
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.point = self.key.public_key().public_bytes(
            Encoding.X962, PublicFormat.UncompressedPoint
        )
        self.spki = _b64(self.point)
        self.device_id = devices.new_device_id(self.point)
        self.np = os.urandom(32)
        self.request_id = "req_" + uuid.uuid4().hex
        self.claim: dict[str, Any] = {}

    def rekey(self) -> None:
        """U15: discard the revoked key and start over with a fresh one."""
        self.__init__(self.relay, edge=self.edge)  # type: ignore[misc]

    def availability(self) -> dict[str, Any]:
        response = self.relay.get("/api/pair/availability")
        assert response.status_code == 200, response.text
        return response.json()

    def file(self, *, forged: str | None = None) -> Any:
        commit = pair_op.sas_commit(self.np)
        claimed = self.relay.post("/api/pair/request", json={"spki": self.spki, "commit": commit})
        if claimed.status_code != 200:
            return claimed
        self.claim = claimed.json()
        raw = json.dumps(
            {
                "request_id": self.request_id,
                "spki": self.spki,
                "commit": commit,
                "np": _b64(self.np),
                "name": "Ben\u2019s iPhone",
                "platform": "ios",
            }
        ).encode()
        headers = {"content-type": "application/json"}
        if forged is not None:
            headers[pair_requests.ASSERTION_HEADER] = forged
        elif self.edge is not None:
            headers[pair_requests.ASSERTION_HEADER] = self.edge.assert_for(raw)
        return self.relay.post("/api/pair/request", content=raw, headers=headers)

    def sas(self, approval_id: str) -> str:
        okm = HKDF(algorithm=hashes.SHA256(), length=8, salt=None, info=b"lop-pair-sas-v1").derive(
            self.np
            + _unb64(self.claim["nd"])
            + self.point
            + _unb64(self.claim["computer_id_pub"])
            + approval_id.encode()
        )
        return f"{int.from_bytes(okm, 'big') % 1_000_000:06d}"

    def status(self, request_id: str | None = None) -> dict[str, Any]:
        response = self.relay.get(f"/api/pair/request/{request_id or self.request_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def withdraw(self, *, signer: Any = None, request_id: str | None = None) -> Any:
        rid = request_id or self.request_id
        key = signer or self.key
        sig = key.sign(pair_requests.WITHDRAW_LABEL + rid.encode(), ec.ECDSA(hashes.SHA256()))
        return self.relay.post(f"/api/pair/request/{rid}/withdraw", json={"sig": sig.hex()})

    def certificate(self) -> dict[str, Any]:
        return self.relay.get(f"/api/pair/{self.device_id}").json()


class Edge:
    """The Radient edge: an RS256 key whose JWKS is pinned in ``tunnels/config.json``."""

    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        jwk["kid"] = "edge-e2e"
        self.access = {
            "issuer": "https://radient.example.test",
            "owner_account_id": OWNER,
            "tunnel_id": "tun_e2e",
            "version": 3,
            "jwks": {"keys": [jwk]},
        }

    def assert_for(self, body: bytes, **over: Any) -> str:
        now = int(time.time())
        claims = {
            "iss": self.access["issuer"],
            "aud": HOST,
            "sub": OWNER,
            "tunnel_id": "tun_e2e",
            "version": 3,
            "harness_id": "local-operator",
            "method": "POST",
            "target": "/api/pair/request",
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "iat": now,
            "exp": now + 30,
            "jti": uuid.uuid4().hex,
            **over,
        }
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "edge-e2e"})


# ---------------------------------------------------------------------------
# The rig
# ---------------------------------------------------------------------------


class Rig(SimpleNamespace):
    root: Path
    relay: TestClient
    desktop: TestClient
    edge: Edge
    prompts: list[Any]
    gesture: dict[str, Any]


@pytest.fixture
def rig(approvals_root: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:  # noqa: F811
    from local_operator import operator as operator_pkg
    from local_operator.operator import handlers
    from local_operator.operator import trust as trust_module
    from local_operator.tunnels import config as tunnel_config

    root = approvals_root
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    # ``operator`` re-exports ``load_anchor`` by value, so the store fixture's "no
    # installed anchor" pin on ``trust`` does not reach ``operator_authority_unusable``
    # (the availability route's predicate). Delegate it, so it follows whatever ``trust``
    # answers now -- pinned absent here, the installed rig after ``_install_authority``.
    monkeypatch.setattr(
        operator_pkg, "load_anchor", lambda *a, **k: trust_module.load_anchor(*a, **k)
    )
    pair_requests._clear_for_tests()
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()
    _make_key(root)
    edge = Edge()
    tunnel_config.private_write(
        tunnel_config.directory() / "config.json", json.dumps({"origin_auth": edge.access})
    )
    relay = TestClient(
        build_app(MobileDaemon(port=0, password="pw123")),
        base_url=f"http://{HOST}",
        follow_redirects=False,
    )
    assert relay.post("/login", data={"password": "pw123"}).status_code == 303

    monkeypatch.setattr(desktop, "_SPAWN_BEARER", None)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(approval_routes.router)
    app.include_router(operation_routes.router)
    desk = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})

    # The decision signature: the REAL software key, signing the real payload (no Touch
    # ID on CI). Counted so a refused path can be shown to have spent no signature.
    signatures = {"n": 0}

    def sign_decision(**kw: Any) -> str:
        signatures["n"] += 1
        return _sign(
            root,
            A.signed_payload(
                kind=kw["kind"],
                request_id=kw["request_id"],
                request_digest=kw["request_digest"],
                decision=kw.get("decision", "approve"),
                decided_at=kw["decided_at"],
            ),
        )

    monkeypatch.setattr(A, "sign_decision", sign_decision)
    monkeypatch.setattr(
        desktop_approval, "assurance", lambda: (desktop_approval.ASSURANCE_FILE_ONLY, "file")
    )
    rig = Rig(root=root, relay=relay, desktop=desk, edge=edge, prompts=[], signatures=signatures)
    rig.gesture = {"answer": "ok", "during": None}
    rig.trust = trust_module
    rig.handlers = handlers
    rig.operator_pkg = operator_pkg
    rig.monkeypatch = monkeypatch
    return rig


def _install_authority(rig: Rig) -> None:
    """C3's end state, through the test seam: the staged anchor is the installed one.

    The stub gesture runner (``elevate.run_elevated``) writes the staged bytes to the
    redirected anchor path on "ok", or answers cancelled -- the two outcomes a person
    can give an OS prompt.
    """
    trust_module = rig.trust
    monkeypatch = rig.monkeypatch
    raw = trust_module.staging_path(rig.root).read_bytes()
    anchor_root = rig.root / "anchor-root"
    anchor_root.mkdir(exist_ok=True)
    monkeypatch.setattr(trust_module, "_ANCHOR_ROOT_OVERRIDE", anchor_root)
    target = trust_module.anchor_path()
    target.write_bytes(raw)
    real_open = trust_module._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        assert opened is not None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    monkeypatch.setattr(trust_module, "_open_no_follow", as_root)
    monkeypatch.setattr(trust_module, "load_anchor", rig.handlers.load_anchor)
    monkeypatch.setattr(trust_module, "install_refusal", lambda: None)
    monkeypatch.setattr("local_operator.operator.elevate.prompt_refusal", lambda: None)

    def elevate(command: Any, **kw: Any) -> Any:
        rig.prompts.append((command, kw.get("purpose", "confirm")))
        if rig.gesture["during"] is not None:
            rig.gesture["during"]()
        if rig.gesture["answer"] == "cancel":
            return ElevationResult(1, cancelled=True)
        target.write_bytes(trust_module.staging_path(rig.root).read_bytes())
        return ElevationResult(0)

    monkeypatch.setattr("local_operator.operator.elevate.run_elevated", elevate)


def _row(rig: Rig, approval_id: str) -> dict[str, Any]:
    rows = rig.desktop.get("/v1/desktop/approvals").json()["result"]["approvals"]
    return next(row for row in rows if row["approval_id"] == approval_id)


def _ack(rig: Rig, approval_id: str, code: str | None = None) -> Any:
    pairing = _row(rig, approval_id)["pairing"]
    body = {
        "request_digest": pairing["request_digest"],
        "disclosure_digest": pairing["disclosure_digest"],
    }
    if code is not None:
        body["code"] = code
    return rig.desktop.post(f"/v1/desktop/approvals/{approval_id}/pair-ack", json=body)


def _approve(rig: Rig, approval_id: str) -> Any:
    pairing = _row(rig, approval_id)["pairing"]
    return rig.desktop.post(
        f"/v1/desktop/approvals/{approval_id}/approve-operation",
        json={
            "operation": "pair_phone",
            "request_digest": pairing["request_digest"],
            "disclosure_digest": pairing["disclosure_digest"],
        },
    )


def _pair(rig: Rig, phone: Phone) -> str:
    filed = phone.file()
    assert filed.status_code == 201, filed.text
    approval_id = filed.json()["approval_id"]
    if pair_op.mode(filed.json()) == pair_op.MODE_CONFIRM:
        assert _ack(rig, approval_id).status_code == 200
    else:
        assert _ack(rig, approval_id, phone.sas(approval_id)).status_code == 200
    approved = _approve(rig, approval_id)
    assert approved.status_code == 200, approved.text
    return approval_id


# ---------------------------------------------------------------------------
# Step 2 + the attested row
# ---------------------------------------------------------------------------


def test_availability_is_false_until_operator_authority_is_installed(rig: Rig) -> None:
    phone = Phone(rig.relay, edge=rig.edge)
    assert phone.availability() == {"available": False, "reason": "no_operator_authority"}
    _install_authority(rig)
    assert phone.availability() == {"available": True, "reason": "ok"}


def test_attested_confirm_issues_and_admits_and_the_phone_reads_paired(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay, edge=rig.edge)
    filed = phone.file()
    assert filed.status_code == 201, filed.text
    record = filed.json()
    approval_id = record["approval_id"]
    assert record["what"]["pair_phone"]["via"] == {
        "route": "radient",
        "account": OWNER,
        "attested": "edge-v1",
    }
    # The feed's probe names this request within one tick (one snapshot read).
    assert A.approval_snapshot(rig.root)[3] == approval_id
    assert phone.status() == {
        "state": "requested",
        "reason": "waiting",
        "mode": "confirm",
        "device_id": phone.device_id,
        "expires_at": record["expires_at"],
    }
    pairing = _row(rig, approval_id)["pairing"]
    # CONFIRM: both screens show the same six digits.
    assert pairing["mode"] == "confirm" and pairing["sas"] == phone.sas(approval_id)
    assert pairing["phone_name"] == "Ben\u2019s iPhone" and pairing["progress"] == "waiting"
    assert _ack(rig, approval_id).status_code == 200
    assert phone.status()["reason"] == "signing"
    approved = _approve(rig, approval_id)
    assert approved.status_code == 200, approved.text
    assert [purpose for _, purpose in rig.prompts] == ["confirm"]
    assert phone.status()["reason"] == "paired"
    cert = phone.certificate()
    assert cert["paired"] is True and devices.is_admitted(cert["certificate"])
    assert A.approval_snapshot(rig.root)[3] is None


def test_custom_route_needs_the_code_and_three_wrong_codes_close_it(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay)  # no edge: a custom tunnel / password cookie
    filed = phone.file()
    approval_id = filed.json()["approval_id"]
    assert filed.json()["what"]["pair_phone"]["via"] == {"route": "custom"}
    pairing = _row(rig, approval_id)["pairing"]
    assert pairing["mode"] == "code" and pairing["sas"] is None  # never rendered on desktop
    refused = _ack(rig, approval_id)
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "pair_code_required"
    # A bodiless approve cannot sign it either.
    assert _approve(rig, approval_id).status_code == 409
    assert rig.signatures["n"] == 0
    wrong = "000000" if phone.sas(approval_id) != "000000" else "111111"
    for _ in range(3):
        assert _ack(rig, approval_id, wrong).status_code == 409
    assert phone.status()["reason"] == "strikes_exhausted"
    assert rig.signatures["n"] == 0


def test_a_forged_assertion_files_as_typed(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay)
    attacker = Edge()  # a well-formed RS256 token under a key the computer never pinned
    body_less = attacker.assert_for(b"{}")
    filed = phone.file(forged=body_less)
    assert filed.status_code == 201
    assert filed.json()["what"]["pair_phone"]["via"] == {"route": "custom"}
    assert phone.status()["mode"] == "code"


def test_a_replayed_assertion_for_another_body_is_not_attested(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay)
    filed = phone.file(forged=rig.edge.assert_for(b'{"other":"body"}'))
    assert filed.json()["what"]["pair_phone"]["via"] == {"route": "custom"}


# ---------------------------------------------------------------------------
# One live request, withdraw, rate limit
# ---------------------------------------------------------------------------


def test_second_key_busy_same_key_resumes_withdraw_and_refile(rig: Rig) -> None:
    _install_authority(rig)
    first = Phone(rig.relay, edge=rig.edge)
    assert first.file().status_code == 201
    other = Phone(rig.relay, edge=rig.edge)
    busy = other.file()
    assert busy.status_code == 409
    assert busy.json()["code"] == "pair_request_pending"
    assert busy.json()["same_key"] is False and "request_id" not in busy.json()
    # The SAME phone retrying with a new request id learns it is its own request.
    again = Phone(rig.relay, edge=rig.edge)
    again.key, again.point, again.spki = first.key, first.point, first.spki
    mine = again.file()
    assert mine.status_code == 409
    assert mine.json() | {"error": ""} == {
        "error": "",
        "code": "pair_request_pending",
        "same_key": True,
        "request_id": first.request_id,
    }
    # Wrong key: 403 and nothing written.
    forbidden = first.withdraw(signer=other.key)
    assert forbidden.status_code == 403
    assert forbidden.json()["code"] == "pair_withdraw_forbidden"
    assert first.status()["state"] == "requested"
    withdrawn = first.withdraw()
    assert withdrawn.status_code == 200 and withdrawn.json() == {"state": "withdrawn"}
    assert first.withdraw().json() == {"state": "withdrawn"}  # idempotent
    assert first.status()["reason"] == "withdrawn"
    assert other.file().status_code == 201  # re-file succeeds once the slot is free


def test_withdraw_during_the_admin_prompt_is_refused(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay, edge=rig.edge)
    approval_id = phone.file().json()["approval_id"]
    assert _ack(rig, approval_id).status_code == 200
    seen: dict[str, Any] = {}

    def during() -> None:
        seen["status"] = phone.status()["reason"]
        seen["withdraw"] = phone.withdraw()

    rig.gesture["during"] = during
    assert _approve(rig, approval_id).status_code == 200
    assert seen["status"] == "admin_prompt_open"
    assert seen["withdraw"].status_code == 409
    assert seen["withdraw"].json()["code"] == "approval_in_progress"
    assert phone.status()["reason"] == "paired"


def test_the_sixth_filing_in_the_window_is_rate_limited(rig: Rig) -> None:
    _install_authority(rig)
    for _ in range(pair_requests.FILING_LIMIT):
        phone = Phone(rig.relay)
        assert phone.file().status_code == 201
        assert phone.withdraw().status_code == 200
    sixth = Phone(rig.relay).file()
    assert sixth.status_code == 429
    assert sixth.json()["code"] == "pair_filing_rate_limited"
    assert 0 < sixth.json()["retry_after_s"] <= pair_requests.FILING_WINDOW_S


# ---------------------------------------------------------------------------
# The decision-scoped window (§2.1)
# ---------------------------------------------------------------------------


def test_the_window_passing_during_the_admin_prompt_still_pairs(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay, edge=rig.edge)
    record = phone.file().json()
    approval_id = record["approval_id"]
    assert _ack(rig, approval_id).status_code == 200
    late = float(record["expires_at"]) + 5
    real = time.time

    def during() -> None:
        # The person answers the OS prompt after the 300 s window closed; any writer's
        # sweep runs meanwhile.
        monkeypatch.setattr(time, "time", lambda: late)
        A.sweep(root=rig.root, now=late)

    rig.gesture["during"] = during
    approved = _approve(rig, approval_id)
    monkeypatch.setattr(time, "time", real)
    assert approved.status_code == 200, approved.text
    assert A.load_record(approval_id, root=rig.root)["state"] == A.STATE_CONNECTED


def test_a_decision_one_second_past_the_signing_bound_refuses(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.operator.macos.keyagent import SIGN_TIMEOUT_SECONDS

    _install_authority(rig)
    phone = Phone(rig.relay, edge=rig.edge)
    record = phone.file().json()
    approval_id = record["approval_id"]
    # The person decides one second before the window closes...
    late_ack = float(record["expires_at"]) - 1
    monkeypatch.setattr(time, "time", lambda: late_ack)
    assert _ack(rig, approval_id).status_code == 200
    acked = pair_op.decision_at(A.load_record(approval_id, root=rig.root))
    assert acked == late_ack
    within = acked + SIGN_TIMEOUT_SECONDS
    past = within + 1
    assert within > float(record["expires_at"])  # the gap is past the window
    # ...the signature inside the signer's bound is accepted (pinned in the cell above);
    # one second past it the window applies again.
    monkeypatch.setattr(time, "time", lambda: past)
    refused = _approve(rig, approval_id)
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "approval_expired"
    assert rig.signatures["n"] == 0


def test_a_stale_run_past_the_window_settles_failed(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay, edge=rig.edge)
    record = phone.file().json()
    approval_id = record["approval_id"]
    assert _ack(rig, approval_id).status_code == 200
    rig.gesture["answer"] = "cancel"
    assert _approve(rig, approval_id).status_code == 200  # signed; apply refused
    # Put the record in the shape a crashed runner leaves: connecting, lease pid dead.
    raw = A._load_raw(approval_id, rig.root)
    raw["state"] = A.STATE_CONNECTING
    raw["run"] = {"run_id": "run_dead", "pid": 2**22 + 12345, "started_at": time.time()}
    A._write_record(raw, rig.root)
    A.sweep(root=rig.root, now=float(record["expires_at"]) + 1)
    after = A._load_raw(approval_id, rig.root)
    assert after["state"] == A.STATE_FAILED
    assert after["receipts"][-1]["step"] == "stale"


# ---------------------------------------------------------------------------
# Remove: removing -> revoked, relay refusal, push tombstone, and the U15 re-key
# ---------------------------------------------------------------------------


def test_remove_cancelled_is_removing_then_finishes_and_the_phone_rekeys(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay, edge=rig.edge)
    _pair(rig, phone)
    cert = phone.certificate()["certificate"]
    # The push link: registered with the device certificate as proof.
    registered = rig.relay.post(
        "/api/push/register",
        json={
            "platform": "ios",
            "token": "apns-token",
            "environment": "sandbox",
            "app_version": "1.0",
            "install_id": str(uuid.uuid4()),
            "operator_device_id": phone.device_id,
            "operator_cert": cert,
        },
    )
    assert registered.status_code == 200, registered.text
    listed = rig.desktop.get("/v1/desktop/devices").json()["result"]["devices"]
    assert [(row["device_id"], row["state"], row["push"]) for row in listed] == [
        (phone.device_id, "paired", True)
    ]

    rig.gesture["answer"] = "cancel"
    first = rig.desktop.delete(f"/v1/desktop/devices/{phone.device_id}")
    assert first.status_code == 200, first.text
    assert first.json()["result"]["state"] == "removing"
    assert rig.prompts[-1][1] == "remove"
    # Survives a re-read (derived, not stored).
    again = rig.desktop.get("/v1/desktop/devices").json()["result"]["devices"]
    assert [(row["state"], row["push"]) for row in again] == [("removing", False)]
    # Phases 1-2 already took effect: the relay refuses the phone's certificate...
    refused = rig.relay.post(
        "/api/sessions/s_any/command", json={"op": "approve", "operator_cert": cert}
    )
    assert refused.status_code == 403 and refused.json()["code"] == "operator_device_revoked"
    # ...and its push row is tombstoned.
    rows = push_devices.list_devices(rig.root)["devices"]
    assert [row["state"] for row in rows] == ["revoked"]

    rig.gesture["answer"] = "ok"
    second = rig.desktop.delete(f"/v1/desktop/devices/{phone.device_id}")
    assert second.json()["result"]["state"] == "revoked"
    assert rig.desktop.get("/v1/desktop/devices").json()["result"]["devices"][0]["state"] == (
        "revoked"
    )

    # U15: the revoked key stays revoked forever...
    old_id = phone.device_id
    phone.np, phone.request_id = os.urandom(32), "req_" + uuid.uuid4().hex
    refiled = phone.file()
    assert refiled.status_code == 422 and refiled.json()["code"] == "pair_device_revoked"
    # ...so the phone re-keys and a NEW approval issues a different device id.
    phone.rekey()
    _pair(rig, phone)
    assert phone.device_id != old_id
    assert phone.status()["reason"] == "paired"
    assert devices.is_admitted(phone.certificate()["certificate"])


def test_push_link_without_a_valid_certificate_is_refused(rig: Rig) -> None:
    _install_authority(rig)
    phone = Phone(rig.relay)
    refused = rig.relay.post(
        "/api/push/register",
        json={
            "platform": "ios",
            "token": "apns-token",
            "environment": "sandbox",
            "app_version": "1.0",
            "install_id": str(uuid.uuid4()),
            "operator_device_id": phone.device_id,
            "operator_cert": "not-a-certificate",
        },
    )
    assert refused.status_code in (400, 422), refused.text
    assert push_devices.list_devices(rig.root)["devices"] == []


def test_desktop_routes_require_the_bearer(rig: Rig) -> None:
    bare = TestClient(rig.desktop.app)
    assert bare.get("/v1/desktop/devices").status_code == 401
    assert bare.post("/v1/desktop/approvals/ap_x/pair-ack", json={}).status_code in (401, 422)
