"""Phone usage is an authenticated cached display, never a provider probe."""

import asyncio
import logging
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
    daemon = mobile.MobileDaemon(password="fixture-password", dial_registrants=False)
    return mobile.build_app(daemon)


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
    assert ok.json()["refresh"] == {
        "status": "ok",
        "forced": False,
        "joined": False,
        "retry_after_ms": None,
    }
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
    assert first == {"status": "ok", "forced": True, "joined": False, "retry_after_ms": None}
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
    # The joiner got the forced refresh it asked for: ok, never "throttled"
    # over numbers that just landed (QA round 1, Q2).
    assert first.json()["refresh"] == {
        "status": "ok",
        "forced": True,
        "joined": False,
        "retry_after_ms": None,
    }
    assert second.json()["refresh"] == {
        "status": "ok",
        "forced": True,
        "joined": True,
        "retry_after_ms": None,
    }


def test_forced_request_joining_unforced_refresh_says_it_was_not_forced(monkeypatch):
    """A forced tap landing on the open-time unforced refresh rides it, and the
    answer says so (review round 1, R3): forced false, joined true, ok."""
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
            opener = asyncio.create_task(client.post("/api/usage/refresh", json={"force": False}))
            await asyncio.sleep(0.02)
            tap = asyncio.create_task(client.post("/api/usage/refresh", json={"force": True}))
            await asyncio.sleep(0.02)
            release.set()
            return await opener, await tap

    _, tap = asyncio.run(scenario())
    assert calls == [False]
    assert tap.json()["refresh"] == {
        "status": "ok",
        "forced": False,
        "joined": True,
        "retry_after_ms": None,
    }


def test_forced_refresh_that_fetched_nothing_new_is_failed(monkeypatch):
    """Every vendor refused: the controller keeps last-good rows, so the task
    succeeds with source "cached". A FORCED tap must not read as "Updated"
    (QA round 1, Q1); an unforced one served from a fresh cache stays ok."""

    async def refresh(force):
        return _refresh_body(source="cached", refreshed_at=None)

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    with _authed_client(_app()) as client:
        unforced = client.post("/api/usage/refresh", json={"force": False}).json()["refresh"]
        forced = client.post("/api/usage/refresh", json={"force": True}).json()["refresh"]
    assert unforced["status"] == "ok"
    assert forced["status"] == "failed" and forced["forced"] is True


def test_background_refresh_failure_is_logged_once_without_vendor_text(monkeypatch, caplog):
    """After a timeout nobody awaits the shielded task; its exception must be
    retrieved and logged with no vendor message (review round 1, R1)."""

    async def refresh(force):
        await asyncio.sleep(0.15)
        raise RuntimeError("vendor exploded private-a@example.invalid")

    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    monkeypatch.setattr(mobile, "_cached_phone_usage", _cached_body)
    monkeypatch.setattr(mobile, "PHONE_REFRESH_BOUND_S", 0.02)
    app = _app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://phone") as client:
            await client.post("/login", data={"password": "fixture-password"})
            response = await client.post("/api/usage/refresh", json={"force": False})
            await asyncio.sleep(0.3)
            return response

    with caplog.at_level(logging.WARNING):
        response = asyncio.run(scenario())
    assert response.json()["refresh"]["status"] == "timeout"
    records = [r for r in caplog.records if "phone usage refresh failed" in r.getMessage()]
    assert len(records) == 1
    assert "RuntimeError" in records[0].getMessage()
    assert "private-a" not in caplog.text and "never retrieved" not in caplog.text


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
                UsageLimit("v", "vendor", UsageAmount(used_fraction=0.1), status="exhausted"),
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
        "amount",
        "paid_fallback",
        "available",
        "unlimited",
        "supersedes_tiers",
        "family_cap",
        "shared",
        "window",
        "family",
        "resets_at_ms",
        "payable",
        "exhausted",
    }
    assert original_limit <= set(rows[0]["limits"][0])


def test_refresh_updated_time_counts_only_reports_this_refresh_fetched(monkeypatch):
    """A reauth account's last-good row keeps yesterday's fetched_at; it must
    not become the status line's "Updated" time (review round 1, R2). The
    footer's ``as_of`` stays the oldest report shown."""
    now_ms = 1_800_000_000_000
    yesterday = now_ms - 86_400_000
    reports = [
        UsageReport(provider="anthropic", fetched_at=now_ms + 5),
        UsageReport(provider="anthropic", fetched_at=yesterday, credential_invalid=True),
    ]

    class FakeController:
        auth_store = Mock()

        async def refresh_persisted_usage(self, force):
            return reports

        def order_persisted_usage_reports(self, got):
            return [(report, index, len(got)) for index, report in enumerate(got)]

        def close(self):
            pass

    monkeypatch.setattr(
        ProviderController, "for_persisted_usage_refresh", staticmethod(FakeController)
    )
    monkeypatch.setattr(mobile.time, "time", lambda: now_ms / 1000)
    body = asyncio.run(mobile._refresh_phone_usage(True))
    assert body["source"] == "live"
    assert body["refreshed_at"] == now_ms + 5
    assert body["as_of"] == yesterday

    reports[0] = UsageReport(provider="anthropic", fetched_at=yesterday - 1)
    stale = asyncio.run(mobile._refresh_phone_usage(True))
    assert stale["source"] == "cached" and stale["refreshed_at"] is None
    assert stale["as_of"] == yesterday - 1
