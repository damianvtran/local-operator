"""`/v1/desktop/models?live=true` for a stored sign-in whose grant is dead.

Reproduction. The operator's Vertex OAuth refresh was refused for good; the
store recorded the dead-grant tombstone and from then on handed out no listing
token. Discovery answered ``unauthenticated`` and fell back to the bundled
Vertex rows, and the desktop picker published every one of them with
``connected: true`` -- stale models presented as a working account, with no
prompt to sign in again. The fix keys on the persisted tombstone
(``ProviderController.live_catalogue``): those rows go out unconnected and
``provider_status`` carries ``reauth_required``, the same map the phone relay's
``/api/models`` publishes.

The credential is SYNTHETIC and lives in this test's isolated HOME/config; the
listing transport is stubbed so no request leaves the process.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.model import discovery
from local_operator.providers.auth_store import AuthStore
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-reauth-token"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("GOOGLE_CLOUD_PROJECT", "VERTEX_PROJECT", "GOOGLE_VERTEX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # No listing ever reaches a network: a failed fetch is all any provider gets.
    monkeypatch.setattr(discovery, "fetch_models", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


def _vertex_sign_in(*, dead: bool) -> None:
    with contextlib.closing(AuthStore()) as store:
        row = store.upsert_credential(
            "google-vertex",
            {
                "type": "oauth",
                "access": "synthetic-vertex-access",
                "refresh": "synthetic-vertex-refresh",
                "expires": int((time.time() + 3600) * 1000),
                "project_id": "synthetic-project",
            },
        )
        if dead:
            assert store._mark_grant_dead(row.id, rejected_refresh_token="synthetic-vertex-refresh")


async def _vertex(client: AsyncClient) -> tuple[list[dict], dict]:
    response = await client.get("/v1/desktop/models", params={"live": "true"})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert "synthetic" not in response.text
    return [row for row in result["models"] if row["provider"] == "google-vertex"], result


async def test_a_dead_grant_is_not_connected_and_reports_reauth(catalogue) -> None:
    _vertex_sign_in(dead=True)
    rows, result = await _vertex(catalogue)
    assert rows
    assert not any(row["connected"] for row in rows)
    assert result["provider_status"] == {
        "google-vertex": {"status": "reauth_required", "name": "Google Vertex AI (Cloud sign-in)"}
    }
    # Not a listing failure: the `errors` banner keeps its own meaning.
    assert "google-vertex" not in result["errors"]


async def test_a_live_grant_is_unchanged(catalogue) -> None:
    _vertex_sign_in(dead=False)
    rows, result = await _vertex(catalogue)
    assert rows and all(row["connected"] for row in rows)
    assert result["provider_status"] == {}


async def test_the_initial_frame_reports_no_provider_status(catalogue) -> None:
    """Only a live read can know a grant is dead; the paint-on-open frame says {}."""
    _vertex_sign_in(dead=True)
    response = await catalogue.get("/v1/desktop/models")
    assert response.status_code == 200, response.text
    assert response.json()["result"]["provider_status"] == {}


async def test_no_token_without_a_tombstone_is_not_a_reauth(
    catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient refresh failure also yields no listing token. Without the
    persisted tombstone that is NOT "sign in again": the rows keep their
    pre-existing status and no notice is published."""

    async def no_access(self, provider, **_kwargs):
        return None

    monkeypatch.setattr(AuthStore, "get_catalogue_access", no_access)
    monkeypatch.setattr(AuthStore, "get_api_key", no_access)
    _vertex_sign_in(dead=False)
    rows, result = await _vertex(catalogue)
    assert rows and all(row["connected"] for row in rows)
    assert result["provider_status"] == {}
