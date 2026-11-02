"""Phone usage is an authenticated cached display, never a provider probe."""

import asyncio
from unittest.mock import Mock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from local_operator.mobile import daemon as mobile
from local_operator.providers.controller import (
    ProviderController,
    ReadOnlyUsageSnapshot,
)
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport


def test_phone_usage_gate_precedes_cached_data(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mobile,
        "_cached_phone_usage",
        lambda: calls.append(True)
        or {
            "reports": [],
            "source": "cached",
            "may_be_stale": False,
            "cached_at": None,
        },
    )
    app = mobile.build_app(mobile.MobileDaemon(password="fixture-password", dial_registrants=False))
    with TestClient(app, follow_redirects=False) as client:
        assert client.get("/api/usage").status_code == 401
        assert not calls
        client.post("/login", data={"password": "fixture-password"})
        response = client.get("/api/usage")
        assert response.status_code == 200
        assert response.json() == {
            "reports": [],
            "source": "cached",
            "may_be_stale": False,
            "cached_at": None,
        }
        assert calls == [True]


def test_phone_projection_preserves_blocked_amount_without_identity(monkeypatch):
    report = UsageReport(
        provider="openai",
        limits=[
            UsageLimit(
                id="wallet",
                label="private-account@example.invalid",
                paid_fallback=True,
                available=False,
                amount=UsageAmount(remaining=61899.95877, unit="credits"),
                detail="private detail",
                unlimited=False,
            )
        ],
        identity="private-account@example.invalid",
    )
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = ReadOnlyUsageSnapshot(
        reports=[report],
    )
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ) as factory:
        rows = mobile._cached_phone_usage()["reports"]
    factory.assert_called_once_with()
    controller.cached_persisted_usage_reports_read_only.assert_called_once_with(None)
    controller.auth_store.close.assert_called_once_with()
    assert "private-account" not in str(rows)
    wallet = rows[0]["limits"][0]
    assert wallet["amount"]["remaining"] == 61899.95877
    assert wallet["amount"]["limit"] is None
    assert wallet["available"] is False
    assert wallet["payable"] is False


def test_phone_cache_read_failure_is_unavailable_not_empty(monkeypatch):
    def fail():
        raise RuntimeError("private-account backend failure")

    monkeypatch.setattr(mobile, "_cached_phone_usage", fail)
    app = mobile.build_app(mobile.MobileDaemon(password="fixture-password", dial_registrants=False))
    with TestClient(app, follow_redirects=False) as client:
        client.post("/login", data={"password": "fixture-password"})
        response = client.get("/api/usage")
    assert response.status_code == 503
    assert response.json() == {"error": "Cached usage unavailable"}


def test_phone_projection_retains_backend_fraction_first_and_unlimited():
    limits = [
        UsageLimit(
            id="over",
            label="paid",
            paid_fallback=True,
            available=True,
            amount=UsageAmount(remaining=10, used_fraction=1, unit="credits"),
        ),
        UsageLimit(
            id="uncapped",
            label="paid",
            paid_fallback=True,
            available=True,
            unlimited=True,
            amount=UsageAmount(unit="credits"),
        ),
    ]
    report = UsageReport(provider="openai", limits=limits)
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = ReadOnlyUsageSnapshot(
        reports=[report],
    )
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ):
        rows = mobile._cached_phone_usage()["reports"]
    assert rows[0]["limits"][0]["payable"] is False
    assert rows[0]["limits"][1]["payable"] is True
    assert rows[0]["limits"][1]["amount"]["limit"] is None


def test_phone_window_metadata_is_exact_allowlist_not_vendor_copy():
    report = UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="a",
                label="secret label",
                window="5 hour",
                shared=True,
                amount=UsageAmount(used_fraction=0.2),
            ),
            UsageLimit(
                id="b",
                label="secret label",
                window="7 day",
                tier="opus",
                amount=UsageAmount(used_fraction=1),
                resets_at_ms=1700000000000,
            ),
            UsageLimit(
                id="c",
                label="secret label",
                window="private@example.invalid",
                tier="secret family",
                amount=UsageAmount(),
            ),
        ],
    )
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = ReadOnlyUsageSnapshot(
        reports=[report],
    )
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ):
        rows = mobile._cached_phone_usage()["reports"]
    windows = rows[0]["limits"]
    assert windows[0]["window"] == "5h" and windows[0]["shared"] is True
    assert windows[1]["window"] == "7d" and windows[1]["family"] == "opus"
    assert windows[1]["resets_at_ms"] == 1700000000000
    assert windows[2]["window"] == "unknown" and windows[2]["family"] == "unknown"
    assert "secret" not in str(rows) and "private@" not in str(rows)


def test_phone_unavailable_authority_is_not_empty_and_closes_owned_handle():
    import pytest

    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = ReadOnlyUsageSnapshot(
        unavailable=True,
    )
    with (
        patch(
            "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
            return_value=controller,
        ),
        pytest.raises(RuntimeError, match="cached usage unavailable"),
    ):
        mobile._cached_phone_usage()
    controller.auth_store.close.assert_called_once_with()


def test_actual_absent_store_is_empty_without_writes(tmp_path):
    """Exercise the real read-only authority, not a list-shaped seam double."""
    auth = tmp_path / "absent-auth.db"
    usage = tmp_path / "absent-usage.db"
    controller = ProviderController.read_only_for_persisted_usage(auth, usage)
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ):
        app = mobile.build_app(
            mobile.MobileDaemon(password="fixture-password", dial_registrants=False)
        )
        with TestClient(app, follow_redirects=False) as client:
            client.post("/login", data={"password": "fixture-password"})
            response = client.get("/api/usage")
    assert response.status_code == 200
    assert response.json() == {
        "reports": [],
        "source": "cached",
        "may_be_stale": False,
        "cached_at": None,
    }
    assert not auth.exists() and not usage.exists()


def test_actual_unreadable_store_is_unavailable_not_empty(tmp_path):
    auth = tmp_path / "corrupt-auth.db"
    auth.write_bytes(b"not a sqlite database")
    before = auth.stat().st_mtime_ns
    controller = ProviderController.read_only_for_persisted_usage(auth, tmp_path / "usage.db")
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ):
        app = mobile.build_app(
            mobile.MobileDaemon(password="fixture-password", dial_registrants=False)
        )
        with TestClient(app, follow_redirects=False) as client:
            client.post("/login", data={"password": "fixture-password"})
            response = client.get("/api/usage")
    assert response.status_code == 503
    assert response.json() == {"error": "Cached usage unavailable"}
    assert auth.stat().st_mtime_ns == before
    assert not (tmp_path / "usage.db").exists()


def test_snapshot_freshness_survives_phone_projection():
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = ReadOnlyUsageSnapshot(
        reports=[],
        may_be_stale=True,
        cached_at=1700000000000,
    )
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ):
        value = mobile._cached_phone_usage()
    assert value["may_be_stale"] is True
    assert value["cached_at"] == 1700000000000


# -- POST /api/usage/refresh ---------------------------------------------------

def _refresh_body(**extra):
    body = {
        "reports": [],
        "source": "live",
        "may_be_stale": False,
        "cached_at": None,
        "refreshed_at": None,
    }
    body.update(extra)
    return body


def _cached_body():
    return {"reports": [], "source": "cached", "may_be_stale": False, "cached_at": 7}


def _authed_client(app):
    client = TestClient(app, follow_redirects=False)
    client.post("/login", data={"password": "fixture-password"})
    return client


def _app():
    return mobile.build_app(mobile.MobileDaemon(password="fixture-password", dial_registrants=False))


def test_refresh_requires_pairing_and_same_origin(monkeypatch):
    calls = []

    async def refresh(force):
        calls.append(force)
        return _refresh_body()

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    with TestClient(_app(), follow_redirects=False) as client:
        assert client.post("/api/usage/refresh", json={"force": True}).status_code == 401
        client.post("/login", data={"password": "fixture-password"})
        crossed = client.post(
            "/api/usage/refresh",
            json={"force": True},
            headers={"origin": "https://someone-else.example.test"},
        )
        assert crossed.status_code == 403
        assert calls == []
        ok = client.post("/api/usage/refresh", json={"force": False})
    assert ok.status_code == 200
    assert ok.json()["refresh"] == {"status": "ok", "forced": False, "retry_after_ms": None}
    assert calls == [False]


def test_forced_refresh_is_floored_and_reports_retry_after(monkeypatch):
    calls = []

    async def refresh(force):
        calls.append(force)
        return _refresh_body()

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    with _authed_client(_app()) as client:
        first = client.post("/api/usage/refresh", json={"force": True}).json()["refresh"]
        second = client.post("/api/usage/refresh", json={"force": True}).json()["refresh"]
    assert first == {"status": "ok", "forced": True, "retry_after_ms": None}
    assert second["status"] == "throttled" and second["forced"] is False
    assert 0 < second["retry_after_ms"] <= 15_000
    # The throttled request still refreshes, just without skipping the cache.
    assert calls == [True, False]


def test_concurrent_refreshes_share_one_fetch(monkeypatch):
    """Two phones (or a double tap) joining one in-flight refresh fetch once."""
    calls = []
    release = asyncio.Event()

    async def refresh(force):
        calls.append(force)
        await release.wait()
        return _refresh_body()

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    app = _app()
    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://phone") as client:
            await client.post("/login", data={"password": "fixture-password"})
            first = asyncio.create_task(client.post("/api/usage/refresh", json={"force": True}))
            second = asyncio.create_task(client.post("/api/usage/refresh", json={"force": True}))
            await asyncio.sleep(0.05)
            release.set()
            return await first, await second

    first, second = asyncio.run(scenario())
    assert calls == [True]
    assert first.status_code == second.status_code == 200


def test_slow_refresh_times_out_to_cached_data_without_cancelling(monkeypatch):
    """The bound stops WAITING; the shielded task still finishes and writes."""
    finished = []

    async def refresh(force):
        await asyncio.sleep(0.3)
        finished.append(True)
        return _refresh_body()

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    monkeypatch.setattr(mobile, "_cached_phone_usage", _cached_body)
    monkeypatch.setattr(mobile, "PHONE_REFRESH_BOUND_S", 0.05)
    app = _app()
    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://phone") as client:
            await client.post("/login", data={"password": "fixture-password"})
            response = await client.post("/api/usage/refresh", json={"force": False})
            assert not finished, "the bound must answer before the refresh lands"
            await asyncio.sleep(0.5)
            return response

    response = asyncio.run(scenario())
    body = response.json()
    assert body["refresh"]["status"] == "timeout"
    assert body["source"] == "cached" and body["cached_at"] == 7
    assert finished == [True], "the timed-out refresh was cancelled"


def test_failed_refresh_answers_cached_projection(monkeypatch):
    async def refresh(force):
        raise RuntimeError("vendor private-account detail")

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    monkeypatch.setattr(mobile, "_cached_phone_usage", _cached_body)
    with _authed_client(_app()) as client:
        response = client.post("/api/usage/refresh", json={"force": False})
    assert response.status_code == 200
    body = response.json()
    assert body["refresh"]["status"] == "failed"
    assert "private-account" not in response.text
    assert body["cached_at"] == 7


def test_projection_derives_tone_fraction_state_and_account_positions():
    reports = [
        UsageReport(
            provider="anthropic",
            limits=[
                UsageLimit("5h", "5 hour", UsageAmount(used_fraction=0.9, unit="percent"), "5h"),
                UsageLimit("7d", "7 day", UsageAmount(used_fraction=1.04, unit="percent"), "7d"),
                UsageLimit("x", "odd", UsageAmount(unit="unknown")),
                UsageLimit(
                    "v", "vendor", UsageAmount(used_fraction=0.1), status="exhausted"
                ),
            ],
            consecutive_failures=1,
        ),
        UsageReport(provider="anthropic", credential_invalid=True),
        UsageReport(provider="deepseek", usage_unavailable=True),
    ]
    rows = mobile._phone_usage_projection(reports, [(1, 2), (0, 2), (0, 1)])
    first = rows[0]
    assert first["provider_name"] == "Anthropic (Claude Pro/Max)"
    assert (first["account_index"], first["account_count"]) == (1, 2)
    assert first["state"] == "partial" and first["credential_invalid"] is False
    tones = [(lim["tone"], lim["remaining_fraction"]) for lim in first["limits"]]
    assert tones[0][0] == "warning" and tones[0][1] == pytest.approx(0.1)
    assert tones[1] == ("exhausted", 0.0)  # over-quota clamps, never negative
    assert tones[2] == ("unknown", None)
    assert tones[3][0] == "exhausted"  # vendor status outranks the threshold
    assert rows[1]["state"] == "reauth_required" and rows[1]["credential_invalid"] is True
    assert rows[2]["state"] == "unavailable"
    # Without positions, numbering falls back to order within each provider.
    fallback = mobile._phone_usage_projection(reports)
    assert [(r["account_index"], r["account_count"]) for r in fallback] == [(0, 2), (1, 2), (0, 1)]


def test_get_keeps_original_keys_and_adds_only(monkeypatch):
    """Older phone builds read only the first-release keys; none may change."""
    report = UsageReport(
        provider="openai",
        limits=[UsageLimit("c", "Credits", UsageAmount(remaining=1, limit=2, unit="credits"))],
    )
    rows = mobile._phone_usage_projection([report])
    original = {"provider", "limits", "unavailable", "fetched_at"}
    assert original <= set(rows[0])
    original_limit = {
        "amount", "paid_fallback", "available", "unlimited", "supersedes_tiers",
        "family_cap", "shared", "window", "family", "resets_at_ms", "payable", "exhausted",
    }
    assert original_limit <= set(rows[0]["limits"][0])
