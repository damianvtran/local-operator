"""The phone-side typed pairing filer, driven through the real relay HTTP app."""

from __future__ import annotations

import ast
import base64
import os
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from starlette.testclient import TestClient

from local_operator.mobile import pair_requests
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval
from local_operator.network.types import MeshRefusal
from local_operator.operator import pair_op
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_approvals_store import root as approvals_root  # noqa: F401


@pytest.fixture
def root(approvals_root: Path) -> Path:  # noqa: F811 — imported fixture
    return approvals_root


@pytest.fixture(autouse=True)
def _fresh_filer() -> Any:
    pair_requests._clear_for_tests()
    yield
    pair_requests._clear_for_tests()


def _point() -> bytes:
    return ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _radient_account(root: Path) -> None:
    from local_operator.tunnels import config

    config.private_write(
        config.directory() / "config.json",
        '{"origin_auth": {"owner_account_id": "acct_from_validated_tunnel"}}',
    )


def _client(root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _make_key(root)
    _radient_account(root)
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code == 303
    return client


def _file(client: TestClient, *, request_id: str = "req_TESTPHONE01") -> dict[str, Any]:
    spki, nonce = _b64(_point()), os.urandom(32)
    np = _b64(nonce)
    claim = client.post(
        "/api/pair/request", json={"spki": spki, "commit": pair_op.sas_commit(nonce)}
    )
    assert claim.status_code == 200, claim.text
    return client.post(
        "/api/pair/request",
        json={
            "request_id": request_id,
            "spki": spki,
            "commit": pair_op.sas_commit(nonce),
            "np": np,
            "name": "Test phone",
            "platform": "ios",
        },
    )


def test_http_filing_produces_exact_apply_shape_and_honest_attribution(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    response = _file(client)
    assert response.status_code == 201, response.text
    record = A.load_record(response.json()["approval_id"], root=root)
    assert pair_op.exact_shape(record)
    assert A._immutable_request(record)["what"] == record["what"]
    assert record["device"]["device_id"] == record["requested_by"]["device_id"]
    assert record["requested_by"] == {
        "surface": "radient-relay",
        "device_id": record["device"]["device_id"],
        "session_id": "",
    }
    assert record["what"]["pair_phone"]["via"] == {
        "route": "radient",
        "account": "acct_from_validated_tunnel",
    }
    # The person-facing disclosure and apply's exact expectation both accept what
    # the HTTP route filed; no hand-maintained sibling shape is involved.
    assert pair_op.disclosure(record)["key_fingerprint"] == record["device"]["key_fp"]
    assert pair_op.expected(record)["device_id"] == record["device"]["device_id"]
    assert A.list_records(root=root)[0]["state"] == "requested"


def test_filing_rejects_client_identity_fields_and_mismatched_fingerprint(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    spki = _b64(_point())
    forged = client.post(
        "/api/pair/request",
        json={
            "spki": spki,
            "commit": pair_op.sas_commit(b"x" * 32),
            "account": "attacker",
            "session_id": "desktop-session",
            "device_id": "chosen-device",
            "key_fp": "chosen-fingerprint",
        },
    )
    assert forged.status_code == 422
    assert A.list_records(root=root) == []
    # The actual fingerprint check is exercised through create_request, not just
    # omitted from the wire schema: a mismatched key identity cannot become a card.
    body = {
        "kind": A.KIND_OPERATOR_DEVICE_PAIR,
        "request_id": "req_MISMATCHEDFP",
        "requested_by": {"surface": "radient-relay", "device_id": "", "session_id": ""},
        "device": {
            "device_id": "",
            "spki": spki,
            "key_fp": "wrong",
            "name": "p",
            "platform": "ios",
        },
        "what": {},
        "credential_ref": None,
    }
    assert not pair_op.exact_shape(body)
    with pytest.raises(MeshRefusal) as caught:
        pair_op.require_filing_shape(body, root)
    assert caught.value.code == "pair_payload_invalid"


def test_route_has_no_approval_endpoint_and_typed_engine_still_requires_code(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    response = _file(client)
    approval_id = response.json()["approval_id"]
    for path in (
        f"/api/pair/{approval_id}/approve",
        f"/api/pair/{approval_id}/apply",
        "/api/pair/approve",
    ):
        assert client.post(path, json={"code": "000000"}).status_code in (404, 405)
    assert A.load_record(approval_id, root=root)["state"] == "requested"
    # Even a caller outside HTTP cannot skip the sole typed-code decision gate.
    with pytest.raises(MeshRefusal) as caught:
        pair_op.require_acked(A.load_record(approval_id, root=root))
    assert caught.value.code == "pair_code_required"


def test_relay_filer_imports_no_decision_signer() -> None:
    module = Path(pair_requests.__file__)
    tree = ast.parse(module.read_text())
    forbidden = {"sign_decision", "load_signer", "approve", "approve_and_apply", "apply"}
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert not names & forbidden
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "desktop_approval" not in imported
    assert A.approve.__module__ not in pair_requests.__name__
    assert desktop_approval.approve_and_apply.__module__ != pair_requests.__name__


def test_second_live_filing_refused_and_idempotent_reconnect_is_free(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    first = _file(client)
    assert first.status_code == 201
    assert _file(client, request_id="req_TESTPHONE02").status_code == 409
    # The claim cache can disappear at reconnect; the durable request is the retry
    # idempotency key. Reconstruct the same reveal and body to prove that.
    record = A.load_record(first.json()["approval_id"], root=root)
    pair_requests._clear_for_tests()
    retry = client.post(
        "/api/pair/request",
        json={
            "request_id": "req_TESTPHONE01",
            "spki": record["device"]["spki"],
            "commit": record["what"]["pair_phone"]["sas"]["commit"],
            "np": record["what"]["pair_phone"]["sas"]["np"],
            "name": "Test phone",
            "platform": "ios",
        },
    )
    assert retry.status_code == 201, retry.text
    assert retry.json()["approval_id"] == record["approval_id"]
    assert len(A.list_records(root=root)) == 1


def test_five_new_records_per_ten_minutes_then_rate_limited(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    for number in range(pair_requests.FILING_LIMIT):
        assert _file(client, request_id=f"req_RATEPHONE{number}").status_code == 201
        record = A.list_records(root=root)[-1]
        A.deny(record["approval_id"], root=root)
    limited = _file(client, request_id="req_RATEPHONE6")
    assert limited.status_code == 429, limited.text
    assert limited.json()["code"] == "pair_filing_rate_limited"


def test_three_wrong_codes_burn_durable_request_and_fourth_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    approval_id = _file(client).json()["approval_id"]
    pair_requests._clear_for_tests()  # a reconnect cannot clear persisted strikes
    for expected in ("pair_code_mismatch",) * 3:
        with pytest.raises(MeshRefusal) as caught:
            pair_op.check_ack(approval_id, code="000000", root=root)
        assert caught.value.code == expected
    burned = A.load_record(approval_id, root=root)
    assert burned["state"] == "denied"
    with pytest.raises(MeshRefusal) as caught:
        pair_op.check_ack(approval_id, code="000000", root=root)
    assert caught.value.code == "approval_decision_conflict"


def test_custom_route_has_no_client_asserted_account(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(root, monkeypatch)
    config_path = root / "tunnel" / "config.json"
    config_path.unlink()
    record = _file(client).json()
    assert record["requested_by"]["surface"] == "phone"
    assert record["what"]["pair_phone"]["via"] == {"route": "custom"}
