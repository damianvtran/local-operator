"""Regression proofs for same-site sibling origins and malformed JSON bodies."""

import json
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, build_app


def test_sibling_origin_cannot_start_owner_session(tmp_path) -> None:
    daemon = MobileDaemon(port=0, password="test-password", dial_registrants=False)
    daemon.spawn_session = AsyncMock(return_value=4242)
    with TestClient(
        build_app(daemon), base_url="https://owner-lop.radienthq.com", follow_redirects=False
    ) as client:
        assert client.post("/login", data={"password": "test-password"}).status_code == 303
        response = client.post(
            "/api/sessions/start",
            content=json.dumps({"cwd": str(tmp_path)}),
            headers={"origin": "https://other-lop.radienthq.com", "content-type": "text/plain"},
        )
        assert response.status_code == 403
        daemon.spawn_session.assert_not_called()
        response = client.post(
            "/api/sessions/start",
            json={"cwd": str(tmp_path)},
            headers={"origin": "https://owner-lop.radienthq.com"},
        )
        assert response.status_code == 200
        daemon.spawn_session.assert_awaited_once()


@pytest.mark.parametrize("route", ["start", "resume"])
@pytest.mark.parametrize("body", [[], None, "text", 12])
def test_session_mutations_require_object(route, body) -> None:
    with TestClient(build_app(MobileDaemon(password="pw")), follow_redirects=False) as client:
        client.post("/login", data={"password": "pw"})
        response = client.post(f"/api/sessions/{route}", content=json.dumps(body))
        assert response.status_code == 400
        assert response.json() == {"error": "request body must be an object"}
