"""Actual settings HTTP contract, persisted canonical/legacy resolution."""

from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from local_operator.config import ConfigManager
from local_operator.providers.account_policy import effective_account_selection_policy
from local_operator.providers.failover import RetrySettings
from local_operator.server.app import desktop_validation_error, managed_desktop_boundary
from local_operator.server.routes import settings

pytestmark = pytest.mark.asyncio
KEY = "retry.accountSelectionPolicy"
TOKEN = "synthetic-account-policy-desktop-token"


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.include_router(settings.router)
    app.middleware("http")(managed_desktop_boundary)
    app.exception_handler(RequestValidationError)(desktop_validation_error)
    app.state.config_manager = ConfigManager(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app.state.config_manager


async def rows(client):
    response = await client.get("/v1/settings")
    assert response.status_code == 200
    return {row["key"]: row for row in response.json()["result"]["settings"]}


async def test_canonical_save_read_restart_reset_and_legacy(api):
    client, manager = api
    assert (await rows(client))[KEY]["value"] == "reset-soonest"
    for policy in ("fair", "headroom", "reset-soonest"):
        response = await client.patch(f"/v1/settings/{KEY}", json={"value": policy})
        assert response.status_code == 200, response.text
        assert (await rows(client))[KEY]["value"] == policy
        fresh = ConfigManager(manager.config_dir)
        assert effective_account_selection_policy(fresh.get_config().values) == policy
        assert (
            RetrySettings.from_settings(fresh.get_config().values).account_selection_policy
            == policy
        )
    # A LEGACY configuration: the canonical key must be absent for the alias to
    # resolve. The write now merges into the file (PR #2819), so the removal is
    # DECLARED rather than implied by replacing the whole `retry` subtree.
    manager.set_config_value(
        "retry",
        {"usageAwareAccountPick": True},
        drop={
            ("retry", "accountSelectionPolicy"): "seeding a legacy config",
            ("retry", "account_selection_policy"): "seeding a legacy config",
        },
    )
    assert (await rows(client))[KEY]["value"] == "headroom"
    assert (
        effective_account_selection_policy(ConfigManager(manager.config_dir).get_config().values)
        == "headroom"
    )
    manager.set_config_value("retry", {"usageAwareAccountPick": False})
    assert (await rows(client))[KEY]["value"] == "fair"
    manager.set_config_value(
        "retry", {"usageAwareAccountPick": False, "accountSelectionPolicy": "headroom"}
    )
    assert (await rows(client))[KEY]["value"] == "headroom"
    response = await client.post(f"/v1/settings/{KEY}/reset")
    assert response.status_code == 200
    assert (await rows(client))[KEY]["value"] == "reset-soonest"
    fresh = ConfigManager(manager.config_dir)
    assert effective_account_selection_policy(fresh.get_config().values) == "reset-soonest"


async def test_permission_rejection_preserves_effective_value_and_recovers(api):
    client, manager = api
    assert (
        await client.patch(f"/v1/settings/{KEY}", json={"value": "headroom"})
    ).status_code == 200
    original = manager.config_file.read_bytes()
    manager.config_dir.chmod(0o500)
    try:
        for response in (
            await client.patch(f"/v1/settings/{KEY}", json={"value": "fair"}),
            await client.post(f"/v1/settings/{KEY}/reset"),
        ):
            assert response.status_code == 503, response.text
            assert response.json()["detail"]["code"] == "setting_write_failed"
            assert manager.config_file.read_bytes() == original
            assert effective_account_selection_policy(manager.get_config().values) == "headroom"
            assert (await rows(client))[KEY]["value"] == "headroom"
    finally:
        manager.config_dir.chmod(0o700)
    response = await client.patch(f"/v1/settings/{KEY}", json={"value": "fair"})
    assert response.status_code == 200
    assert (await rows(client))[KEY]["value"] == "fair"


async def test_invalid_and_unauthorized_save_leave_persisted_config_unchanged(api):
    client, manager = api
    await client.patch(f"/v1/settings/{KEY}", json={"value": "fair"})
    files = {path: path.read_bytes() for path in Path(manager.config_dir).glob("*.yml")}
    assert files
    for invalid in ("bogus", True, 1, None, [], {}):
        response = await client.patch(f"/v1/settings/{KEY}", json={"value": invalid})
        assert response.status_code == 422, response.text
        assert all(path.read_bytes() == before for path, before in files.items())
    response = await client.patch(
        f"/v1/settings/{KEY}",
        json={"value": "headroom"},
        headers={"Authorization": f"Bearer {TOKEN}-wrong"},
    )
    assert response.status_code == 401
    assert all(path.read_bytes() == before for path, before in files.items())
    assert (await rows(client))[KEY]["value"] == "fair"


@pytest.mark.parametrize("language", ["fr", "zh-CN", "en", "unknown"])
async def test_write_refusals_follow_injected_backend_config(api, monkeypatch, language):
    """PATCH and reset share localized copy, never the raw storage exception."""
    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME
    from local_operator.i18n import messages
    from local_operator.i18n.resolve import resolve_language

    monkeypatch.setenv("LOP_LANG", "en")
    client, manager = api
    manager.set_config_value("language", language)
    locale = resolve_language(language)
    path = manager.config_dir / CONFIG_FILE_NAME
    before = path.read_bytes()
    denied = await client.patch(
        f"/v1/settings/{KEY}",
        json={"value": "fair"},
        headers={"Authorization": "Bearer synthetic-wrong-token"},
    )
    assert denied.status_code == 401
    assert path.read_bytes() == before

    def refuse(*args, **kwargs):
        raise OSError("synthetic-private-path-must-not-leak")

    monkeypatch.setattr(settings_io, "write_setting", refuse)
    monkeypatch.setattr(settings_io, "reset_setting", refuse)
    for response in (
        await client.patch(f"/v1/settings/{KEY}", json={"value": "fair"}),
        await client.post(f"/v1/settings/{KEY}/reset"),
    ):
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "setting_write_failed"
        assert detail["params"] == {}
        assert detail["text"] == messages.render(
            "wire.settings.account_policy.error.write_failed", {}, locale=locale
        )
        assert path.read_bytes() == before
        assert "synthetic-private-path" not in response.text
