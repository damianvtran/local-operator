"""The ``guard_consent`` card on the desktop approvals plane (Guard RFC v0.9 §5.2.2).

Driven over the REAL desktop routers (``server.routes.desktop_approvals`` and
``desktop_credentials``, mounted as ``server.app`` mounts them) with a REAL
file-only operator key staged exactly as ``lop operator init`` stages it, and
the real ``RadientClient`` against a local fixture server. ``assurance`` is
pinned to file-only (the CTO's no-hardware decision is what lets that level
sign this op); no OS signer, Touch ID or Keychain is reached.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.guard import consent as C
from local_operator.guard import install_key
from local_operator.network import approval_ops
from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval
from local_operator.network.types import MeshRefusal
from local_operator.server import desktop
from local_operator.server.routes import desktop_approvals as approval_routes
from local_operator.server.routes import desktop_credentials as operation_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.guard.test_guard_consent import (
    TENANT,
    CountingKeyStore,
    _client,
    _offer,
)
from tests.unit.guard.test_guard_consent import (
    fixture_server as _consent_fixture_server,
)
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_approvals_store import root as approvals_root

root = approvals_root
# Re-exported under the conventional name (pytest discovers module attributes).
fixture_server = _consent_fixture_server
TOKEN = "t" * 43
FILE_ONLY = ("file_only", "operator-file-only")
ZERO = "sha256:" + "0" * 64


@pytest.fixture(autouse=True)
def _staged_key(root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any):  # noqa: F811
    _make_key(root)
    monkeypatch.setenv(install_key.BACKEND_ENV, "file")
    monkeypatch.setattr(desktop_approval, "assurance", lambda: FILE_ONLY)
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()
    # The typed op's apply resolves the signed-in person; here that is the fixture.
    monkeypatch.setattr(C, "default_transport", lambda _c: _client(fixture_server))
    yield
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()


@asynccontextmanager
async def _http(root: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", None)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(approval_routes.router)
    app.include_router(operation_routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


def _counted(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    real = A.sign_decision
    counters = {"signer": 0}

    def counted(**kw: Any) -> str:
        counters["signer"] += 1
        return real(**kw)

    monkeypatch.setattr(A, "sign_decision", counted)
    return counters


def _queued(root: Path, fx: Any, **over: Any) -> tuple[C.GuardConsentHost, C.PromptRow]:
    host = C.GuardConsentHost(root, transport=_client(fx), key_store=CountingKeyStore())
    outcome = host.offer(C.ConsentOffer.from_wire(_offer(**over)))
    assert outcome.status == "queued" and outcome.row is not None
    assert outcome.row.approval_id, "an anchored host files the desktop mirror"
    return host, outcome.row


async def _row(client: AsyncClient, approval_id: str) -> dict[str, Any]:
    response = await client.get("/v1/desktop/approvals")
    assert response.status_code == 200, response.text
    rows = {r["approval_id"]: r for r in response.json()["result"]["approvals"]}
    return rows[approval_id]


def test_mirror_is_a_machine_block_typed_record(
    root: Path, fixture_server: Any
) -> None:  # noqa: F811
    _host, row = _queued(root, fixture_server)
    record = A.load_record(row.approval_id, root=root)
    assert record["kind"] == "guard_consent"
    assert set(record) >= {"machine"} and "device" not in record
    assert set(record["what"]) == {"anchor", "guard_consent"}
    assert record["what"]["guard_consent"]["prompt_digest"] == row.digest
    assert approval_ops.get("guard_consent") is not None


@pytest.mark.asyncio
async def test_generic_bodiless_approve_is_refused_before_the_signer(
    root: Path,
    fixture_server: Any,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    _host, row = _queued(root, fixture_server)
    counters = _counted(monkeypatch)
    async with _http(root, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{row.approval_id}/approve")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "guard_consent_typed_approval_required"
    assert counters["signer"] == 0
    assert fixture_server.guard_requests() == []
    assert A.load_record(row.approval_id, root=root)["state"] == A.STATE_REQUESTED
    # The CLI's generic verb runs the same check (``cli_approvable`` is False).
    with pytest.raises(MeshRefusal):
        approval_ops.require_generic_approvable(
            A.load_record(row.approval_id, root=root),
            row.approval_id,
            surface=approval_ops.SURFACE_CLI,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", ["disclosure_digest", "request_digest"])
async def test_stale_digest_is_refused_before_signing(
    root: Path,
    fixture_server: Any,
    monkeypatch: pytest.MonkeyPatch,
    stale: str,  # noqa: F811
) -> None:
    _host, row = _queued(root, fixture_server)
    counters = _counted(monkeypatch)
    async with _http(root, monkeypatch) as client:
        typed = (await _row(client, row.approval_id))["typed_approval"]
        assert typed["signable_here"] is True
        body = {
            "operation": "guard_consent",
            "request_digest": typed["request_digest"],
            "disclosure_digest": typed["disclosure_digest"],
            stale: ZERO,
        }
        response = await client.post(
            f"/v1/desktop/approvals/{row.approval_id}/approve-operation", json=body
        )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] in {
        "credential_disclosure_changed",
        "credential_request_changed",
    }
    assert counters["signer"] == 0
    assert fixture_server.guard_requests() == []


@pytest.mark.asyncio
async def test_a_scope_field_cannot_ride_the_acknowledgment(
    root: Path,
    fixture_server: Any,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    """All-or-nothing (R2-m2): the closed body has no scope field to narrow or widen."""
    _host, row = _queued(root, fixture_server)
    counters = _counted(monkeypatch)
    async with _http(root, monkeypatch) as client:
        typed = (await _row(client, row.approval_id))["typed_approval"]
        response = await client.post(
            f"/v1/desktop/approvals/{row.approval_id}/approve-operation",
            json={
                "operation": "guard_consent",
                "request_digest": typed["request_digest"],
                "disclosure_digest": typed["disclosure_digest"],
                "scopes": ["decisions"],
            },
        )
    assert response.status_code == 422, response.text
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []


@pytest.mark.asyncio
async def test_typed_accept_signs_once_and_sends_exactly_one_signed_grant(
    root: Path,
    fixture_server: Any,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    host, row = _queued(root, fixture_server)
    counters = _counted(monkeypatch)
    async with _http(root, monkeypatch) as client:
        typed = (await _row(client, row.approval_id))["typed_approval"]
        disclosure = typed["disclosure"]
        assert disclosure["asked_scopes"] == ["decisions", "task_time"]
        assert disclosure["rendered_terms_digest"] == row.terms_digest
        response = await client.post(
            f"/v1/desktop/approvals/{row.approval_id}/approve-operation",
            json={
                "operation": "guard_consent",
                "request_digest": typed["request_digest"],
                "disclosure_digest": typed["disclosure_digest"],
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()["result"]
    assert body["applied"]["ok"] is True, body
    assert counters["signer"] == 1
    grants = fixture_server.guard_requests()
    assert [(r["method"], r["path"]) for r in grants] == [("POST", "/v1/guard/consents")]
    assert grants[0]["body"]["payload"]["scopes"] == ["decisions", "task_time"]
    assert "enroll" in grants[0]["body"]["payload"]
    assert C.GuardConsentHost(root).consent(TENANT)["state"] == "live"
    assert C.GuardConsentHost(root).pending() == []
    # The install key went to the 0600 file store (BACKEND_ENV=file), never Keychain.
    assert install_key.key_file(root).is_file()


@pytest.mark.asyncio
async def test_desktop_deny_is_a_sticky_decline_and_sends_nothing(
    root: Path,
    fixture_server: Any,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    host, row = _queued(root, fixture_server)
    async with _http(root, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{row.approval_id}/deny")
    assert response.status_code == 200, response.text
    host.reconcile_mirrors()
    assert host.sticky(TENANT) == "declined"
    assert host.pending() == []
    assert fixture_server.guard_requests() == []


def test_answer_on_another_surface_withdraws_the_desktop_card(
    root: Path,
    fixture_server: Any,  # noqa: F811
) -> None:
    host, row = _queued(root, fixture_server)
    host.decline(TENANT, 1, surface="tui")
    assert A.load_record(row.approval_id, root=root)["state"] == A.STATE_WITHDRAWN
    assert json.dumps(fixture_server.guard_requests()) == "[]"


def test_newer_revision_supersedes_and_withdraws_the_older_card(
    root: Path,
    fixture_server: Any,  # noqa: F811
) -> None:
    host, first = _queued(root, fixture_server)
    second = host.offer(C.ConsentOffer.from_wire(_offer(collection_rev=2)))
    assert second.status == "queued" and second.row is not None
    assert A.load_record(first.approval_id, root=root)["state"] == A.STATE_WITHDRAWN
    assert [p.collection_rev for p in host.pending()] == [2]
