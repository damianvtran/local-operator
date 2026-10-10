"""``GET /v1/capabilities`` and the desktop document stub for turn supplements (lane C0).

Lives under ``tests/unit/server`` because the ``test_app_client`` fixture does. The owner
record, the attach gate and the relay stubs are in ``tests/unit/supplements/test_surfaces.py``.
"""

from __future__ import annotations

import pytest

DIGEST = "04c2d29b140086cc637ad19ef34012c5"


@pytest.mark.asyncio
async def test_v1_capabilities_advertises_supplements(test_app_client) -> None:
    response = await test_app_client.get("/v1/capabilities")
    assert response.status_code == 200
    features = response.json()["result"]["features"]
    # `>= 1` is the comparison clients make (`desktopFeatureState`); 0 reads as absent
    assert features.get("supplements", 0) >= 1


def _desktop_client(tmp_path, monkeypatch):
    """The real app behind the desktop bearer boundary (the shape the search-route tests use).

    ``/v1/desktop/`` answers 503 until the desktop token is configured, so a bare
    ``test_app_client`` would test the boundary and never reach the stub.
    """
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from local_operator.server.app import app
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    return TestClient(app), {"Authorization": "Bearer token"}


def test_the_desktop_document_route_answers_404(tmp_path, monkeypatch) -> None:
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/abc/supplements/{DIGEST}/document", headers=headers
        )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "supplement_unavailable"


def test_the_desktop_document_route_keeps_the_digest_gate(tmp_path, monkeypatch) -> None:
    """A non-digest path never reaches the handler: 422, the traversal gate."""
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            "/v1/desktop/sessions/abc/supplements/not-a-digest/document", headers=headers
        )
    assert response.status_code == 422


def test_the_desktop_document_route_is_behind_the_bearer(tmp_path, monkeypatch) -> None:
    client, _headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(f"/v1/desktop/sessions/abc/supplements/{DIGEST}/document")
    assert response.status_code in (401, 403)
