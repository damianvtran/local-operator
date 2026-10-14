"""Muse usage reads existing receipts, never an inference or key-mint request."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from local_operator.providers.usage import (
    META_MUSE_ALLOWANCE_NOTE,
    META_MUSE_NO_OBSERVATION_NOTE,
    META_MUSE_PASSIVE_NOTE,
    UsageReport,
    fetch_usage,
    meta_muse_usage_report,
    usage_supported,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.routes import desktop_catalogues, desktop_quota
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.utils.desktop_auth import DesktopAuth


def receipt() -> dict[str, Any]:
    return {
        "observedAtMs": int(time.time() * 1000) - 1000,
        "source": "responses",
        "window": {"usedPercent": 31, "windowDurationMins": 300, "resetsAtMs": 1_800_000_000_000},
        "weekly": {"usedPercent": 105, "windowDurationMins": 10080},
    }


def test_receipt_keeps_observation_age_and_account_windows():
    raw = receipt()
    report = meta_muse_usage_report({"subscription_observation": raw})
    assert report.fetched_at == raw["observedAtMs"]
    assert [limit.label for limit in report.limits] == ["5 hour", "7 day"]
    assert [limit.amount.fraction() for limit in report.limits] == [0.31, 1.05]
    assert all(limit.shared for limit in report.limits)
    assert report.limits[0].resets_at_ms == 1_800_000_000_000
    assert report.limits[1].effective_status() == "exhausted"
    assert "Last response allowance" in (report.notes or "")
    assert "refresh does not query Muse" in (report.notes or "")
    assert raw["observedAtMs"] == report.fetched_at


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"observedAtMs": True},
        {"observedAtMs": float("nan")},
        {"observedAtMs": 10**400},
        {"observedAtMs": 10**13},
        {"window": {"usedPercent": 30}},
        {"observedAtMs": 1000, "window": {"usedPercent": -1}},
        {"observedAtMs": 1000, "window": {"usedPercent": True}},
        {"observedAtMs": 1000, "window": {"usedPercent": 10**400}},
    ],
)
def test_missing_or_damaged_receipt_is_an_unknown_provider_row(raw):
    report = meta_muse_usage_report({"subscription_observation": raw})
    assert report.provider == "meta-muse"
    assert report.limits == []
    assert report.fetched_at == 0
    from local_operator.providers.quota_notice import report_is_fresh

    assert not report_is_fresh(report, int(time.time() * 1000))
    assert "No allowance observation" in (report.notes or "")


@pytest.mark.asyncio
async def test_dispatch_never_spends_a_request_and_requires_subscription_auth():
    def forbidden(request):
        raise AssertionError(f"Muse usage must not call {request.url}")

    token_fixture = "fixture"
    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        assert usage_supported("meta-muse")
        for raw in (None, receipt()):
            report = await fetch_usage(
                client,
                "meta-muse",
                access_token=token_fixture,
                oauth_creds={"subscription_observation": raw},
            )
            assert report is not None and report.provider == "meta-muse"
            assert bool(report.limits) == bool(raw)
        assert await fetch_usage(client, "meta-muse", api_key="unrelated-payg") is None
        assert await fetch_usage(client, "meta-muse") is None


@pytest.mark.asyncio
async def test_controller_shared_cache_and_desktop_payload_include_muse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Real SQLite auth and usage stores make this an enumeration/cache contract
    # test, rather than merely proving that a hand-picked fetcher returns data.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    auth = DesktopAuth(store, tmp_path)
    controller = ProviderController(store, tmp_path)
    network: list[str] = []
    original_client = httpx.AsyncClient

    def handle(request):
        network.append(str(request.url))
        assert request.url.host == "api.deepseek.com"
        return httpx.Response(
            200,
            json={
                "is_available": True,
                "balance_infos": [{"currency": "USD", "total_balance": "7.50"}],
            },
        )

    def fixture_client(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handle))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fixture_client)
    try:
        for account, observation in (("1234", receipt()), ("5678", None)):
            row = store.upsert_credential(
                "meta-muse",
                {
                    "type": "oauth",
                    "access": f"LLM|{account}|synthetic_key",
                    "account_id": account,
                    "email": f"{account}@example.test",
                    "subscription_observation": observation,
                },
            )
            if observation:
                from local_operator.providers.auth_store import OAuthAccess

                # Ordinary login receipts are deliberately not live usage.
                # Simulate the serving account's later passive SSE observation.
                observation["observedAtMs"] = int(time.time() * 1000)
                assert store.record_muse_usage(
                    OAuthAccess(access_token="fixture", credential_id=row.id, raw=row.data),
                    observation,
                )
        store.upsert_credential("deepseek", {"type": "api_key", "key": "synthetic"})
        assert controller.usage_reportable_providers() == ["deepseek", "meta-muse"]
        reports = await controller.fetch_usage(force_refresh=True)
        assert [report.provider for report in reports] == ["deepseek", "meta-muse", "meta-muse"]
        muse = [report for report in reports if report.provider == "meta-muse"]
        assert [report.identity for report in muse] == ["1234@example.test", "5678@example.test"]
        assert [bool(report.limits) for report in muse] == [True, False]
        assert all(report.consecutive_failures == 0 for report in muse)
        assert [report.provider for report in controller.cached_usage_reports()] == [
            "deepseek",
            "meta-muse",
            "meta-muse",
        ]
        app = FastAPI()
        app.state.config_manager = ConfigManager(tmp_path)
        app.include_router(desktop_catalogues.router)
        app.include_router(desktop_quota.router)
        app.dependency_overrides[get_desktop_auth] = lambda: auth
        app.dependency_overrides[require_desktop] = lambda: None
        async with original_client(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            for query in ("", "?live=true&refresh=true"):
                response = await client.get("/v1/desktop/usage" + query)
                assert response.status_code == 200, response.text
                rows = response.json()["result"]["reports"]
                assert [row["provider"] for row in rows] == ["deepseek", "meta-muse", "meta-muse"]
                assert [row["state"] for row in rows] == ["available", "available", "unavailable"]
                assert [row["notes"] for row in rows] == [
                    None,
                    META_MUSE_PASSIVE_NOTE,
                    META_MUSE_NO_OBSERVATION_NOTE,
                ]
                assert rows[2]["fetched_at"] == 0
                assert rows[2]["age_ms"] is None
            # An account with no receipt is not a live quota confirmation.
            # Remove only the synthetic observed account to discriminate that
            # source decision from the sibling's real observation timestamp.
            observed_account = store.list_credentials("meta-muse")[0]
            store.delete_credential(observed_account.id)
            response = await client.get("/v1/desktop/quota-notice?provider=meta-muse&refresh=true")
            assert response.status_code == 200, response.text
            assert response.json()["result"]["source"] == "cached"
            assert response.json()["result"]["age_ms"] is None
        assert network == ["https://api.deepseek.com/user/balance"] * 2
    finally:
        controller.close()
        await auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,note",
    [
        ("meta-muse", "upstream private error text"),
        ("meta-muse", ["malformed cached note"]),
        ("deepseek", META_MUSE_ALLOWANCE_NOTE),
    ],
)
async def test_public_usage_does_not_expose_arbitrary_or_misattributed_notes(provider, note):
    # Retaining fixed Muse provenance must not open the general error-copy
    # boundary, even if another provider happens to use the identical text.
    report = UsageReport(provider=provider, notes=note)
    controller = SimpleNamespace(cached_usage_reports=lambda _: [report], close=lambda: None)
    auth = SimpleNamespace(controller=lambda: controller)
    app = FastAPI()
    app.include_router(desktop_catalogues.router)
    app.dependency_overrides[get_desktop_auth] = lambda: auth
    app.dependency_overrides[require_desktop] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        response = await client.get("/v1/desktop/usage")
        assert response.status_code == 200
        assert response.json()["result"]["reports"][0]["notes"] is None
