"""The desktop browser reads owner evidence through the real authenticated API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from local_operator.browser_bridge.resources import RESOURCE_NAME, BrowserResource
from local_operator.server import desktop
from local_operator.server.app import app
from local_operator.server.utils import browser_owner_status
from local_operator.session.retention import claim_session, release_session
from local_operator.session_lease import LEASE_NAME, acquire_session_lease

ROUTE = "/v1/desktop/browser/owner-status"
BASE = "http://127.0.0.1:1111"
BEARER = "synthetic-browser-owner-bearer"
ORIGIN = "http://desktop.fixture.example"
AUTH = {"Authorization": f"Bearer {BEARER}"}
PARENT = "bb0a00000001"
CHILD = "bb0a00000002"
MISSING = "bb0a00000003"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", None)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    monkeypatch.setenv(desktop.TOKEN_ENV, BEARER)
    monkeypatch.setenv(desktop.ORIGINS_ENV, ORIGIN)
    monkeypatch.setattr(app.state, "config_manager", SimpleNamespace(config_dir=tmp_path), raising=False)
    # No lifespan: the HTTP middleware/auth/route are real, but the daemon's
    # background supervisors and deferred boot services are not under test.
    return TestClient(app, base_url=BASE)


def _row(session_id, verdict="unknown", retained=None, provenance="none", generation=None):
    return dict(
        session_id=session_id,
        verdict=verdict,
        turn="unknown",
        retained=retained,
        provenance=provenance,
        generation=hashlib.sha256(generation.encode()).hexdigest() if generation else None,
    )


def _post(client, *ids):
    response = client.post(ROUTE, json={"session_ids": list(ids)}, headers=AUTH)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert set(response.json()["result"]) == {"owners"}
    return response.json()["result"]["owners"]


async def _finish(
    directory: Path, status: str = "completed", *, transport_error: bool = False
) -> BrowserResource:
    resource = BrowserResource(directory, directory.name, host="ui")
    resource.initialize()
    resource.remember("ui:1:private-nonce", host="ui")
    resource.ensure_recovered = AsyncMock()  # type: ignore[method-assign]
    resource.client = lambda: SimpleNamespace(  # type: ignore[method-assign]
        call=AsyncMock(side_effect=OSError("temporary host error"))
        if transport_error else AsyncMock(return_value={"state": "closed"})
    )
    result = await resource.finish(resource.execution_generation, status)
    assert result.state == ("pending" if transport_error else "closed")
    return resource


@pytest.mark.asyncio
async def test_live_parent_completed_child_then_same_generation_resume(client, tmp_path):
    parent, child = (tmp_path / "sessions" / sid for sid in (PARENT, CHILD))
    lease = acquire_session_lease(parent)
    try:
        claim_session(child)
        resource = await _finish(child)
        # The in-process run uses the same process as its parent, but its own
        # browser finalization is insufficient until its live claim is released.
        assert _post(client, CHILD) == [_row(CHILD, "live", None, "session_claim")]
        release_session(child)
        receipt = child / RESOURCE_NAME
        original = receipt.read_bytes(), receipt.stat().st_mtime_ns
        assert _post(client, CHILD, PARENT, MISSING) == [
            _row(CHILD, "ended", False, "browser_terminal", resource.execution_generation),
            _row(PARENT, "live", None, "execution_lease", lease.generation),
            _row(MISSING),
        ]
        assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == original
        assert not (child.parent / MISSING).exists()

        claim_session(child)
        assert _post(client, CHILD) == [_row(CHILD, "live", None, "session_claim")]
        resumed = BrowserResource(child, CHILD)
        resumed.initialize()
        assert resumed.execution_generation == resource.execution_generation
        assert _post(client, CHILD) == [
            _row(CHILD, "live", False, "session_claim", resumed.execution_generation)
        ]
        release_session(child)
        # The new execution retired the receipt, even though its digest didn't change.
        assert _post(client, CHILD) == [_row(CHILD)]
    finally:
        lease.release()


@pytest.mark.asyncio
async def test_new_lease_does_not_reuse_stale_browser_receipt(client, tmp_path):
    directory = tmp_path / "sessions" / PARENT
    old = acquire_session_lease(directory)
    try:
        await _finish(directory, "disposed")
    finally:
        old.release()
    original = _post(client, PARENT)
    assert original == [_row(PARENT, "ended", False, "browser_terminal", old.generation)]
    new = acquire_session_lease(directory)
    try:
        assert _post(client, PARENT) == [_row(PARENT, "live", None, "execution_lease", new.generation)]
    finally:
        new.release()


@pytest.mark.asyncio
async def test_retention_and_pending_host_error_never_prove_ended(client, tmp_path):
    directory = tmp_path / "sessions" / CHILD
    pending = await _finish(directory, transport_error=True)
    assert _post(client, CHILD) == [_row(CHILD)]
    pending.record.update(retention="private retention reason", state="retained")
    pending._save()
    assert _post(client, CHILD) == [_row(CHILD, retained=True)]


@pytest.mark.parametrize(
    "changes",
    [
        {"ownership": "unavailable"},
        {"host": "bridge"},
        {"session_id": PARENT},
        {"state": "cleanup_pending"},
        {"terminal": ""},
        {"resumed_scope": True},
        {"unresolved_surface_id": "ui:1:private-nonce"},
        {"surface_id": []},
    ],
)
@pytest.mark.asyncio
async def test_partial_or_ambiguous_evidence_is_unknown(client, tmp_path, changes):
    resource = await _finish(tmp_path / "sessions" / CHILD)
    resource.record.update(changes)
    resource._save()
    assert _post(client, CHILD) == [_row(CHILD)]


@pytest.mark.asyncio
async def test_death_or_malformed_claim_is_not_terminal(client, tmp_path):
    directory = tmp_path / "sessions" / CHILD
    resource = await _finish(directory, transport_error=True)
    resource.record.pop("terminal")
    resource._save()
    dead = acquire_session_lease(directory, pid=2**30)
    try:
        assert _post(client, CHILD) == [_row(CHILD)]
        (directory / LEASE_NAME).write_text("[]", encoding="utf-8")
        assert _post(client, CHILD, MISSING) == [_row(CHILD), _row(MISSING)]
    finally:
        # Restore the claim for its generation-fenced release.
        (directory / LEASE_NAME).write_text(json.dumps({"generation": dead.generation, "pid": dead.pid}))
        dead.release()


def test_changed_witness_invalidates_the_sample(client, tmp_path, monkeypatch):
    directory = tmp_path / "sessions" / PARENT
    lease = acquire_session_lease(directory)
    real = browser_owner_status.lease_holder

    def replaced_claim(path):
        observed = real(path)
        (path / LEASE_NAME).write_text("{}", encoding="utf-8")
        return observed

    monkeypatch.setattr(browser_owner_status, "lease_holder", replaced_claim)
    assert _post(client, PARENT) == [_row(PARENT)]


@pytest.mark.parametrize(
    ("headers", "status"),
    [({}, 401), ({"Authorization": "Bearer wrong"}, 401),
     ({**AUTH, "Origin": "https://untrusted.example"}, 403),
     ({**AUTH, "Sec-Fetch-Site": "cross-site"}, 403)],
)
def test_boundary_refuses_unauthorized_requests(client, headers, status):
    response = client.post(ROUTE, headers=headers, json={"session_ids": [PARENT]})
    assert response.status_code == status
    assert "result" not in response.json()


def test_desktop_posture_absent(client, monkeypatch):
    monkeypatch.delenv(desktop.TOKEN_ENV)
    assert client.post(ROUTE, headers=AUTH, json={"session_ids": [PARENT]}).status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {}, {"session_ids": []}, {"session_ids": PARENT}, {"session_ids": [7]},
        {"session_ids": ["../escape"]}, {"session_ids": ["line\n"]},
        {"session_ids": [PARENT, PARENT]}, {"session_ids": ["x" * 65]},
        {"session_ids": [f"owner-{i}" for i in range(65)]},
        {"session_ids": [PARENT], "verdict": "ended", "token": "must-not-echo"},
    ],
)
def test_malformed_requests_are_redacted(client, body):
    response = client.post(ROUTE, headers=AUTH, json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "The request has invalid fields."}


def test_maximum_batch_preserves_order(client):
    ids = [f"owner-{i}" for i in range(64)]
    assert _post(client, *ids) == [_row(sid) for sid in ids]
