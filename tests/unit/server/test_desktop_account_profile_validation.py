"""Malformed captured owners refuse at the route, before any OAuth/client work."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.routes import desktop_account_profile


@pytest.mark.parametrize("account_id", ["qa_human_a", "../acc_fixture", "acc_", "not-an-account"])
def test_invalid_account_is_controlled_422_before_client(tmp_path, monkeypatch, account_id):
    calls = []

    async def unexpected_client(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Invalid account reached the profile client")

    monkeypatch.setattr(desktop_account_profile, "_reply", unexpected_client)
    app = FastAPI()
    app.include_router(desktop_account_profile.router)
    app.dependency_overrides[require_desktop] = lambda: None
    app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(config_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get(
            "/v1/desktop/account/profile", params={"expected_account_id": account_id}
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    assert calls == []
