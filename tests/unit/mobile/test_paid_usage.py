"""Phone usage is an authenticated cached display, never a provider probe."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from local_operator.mobile import daemon as mobile
from local_operator.providers.controller import (
    ProviderController,
    ReadOnlyUsageSnapshot,
)
from local_operator.providers.usage import (
    ResetConsumeResult,
    UsageAmount,
    UsageLimit,
    UsageReport,
)


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


# -- reset credits: the projection field ---------------------------------------


def test_projection_carries_resets_available_with_the_exact_int_rule():
    """A count, never a name -- and only a plain non-negative int.

    ``bool`` is an ``int`` in Python, so ``True`` would otherwise reach a button
    that SPENDS something as "1 credit". Anything untrustworthy is ``None``,
    which every surface reads as "no reset to offer".
    """
    reports = [
        UsageReport(provider="openai", limits=[], fetched_at=1, resets_available=2),
        UsageReport(provider="openai", limits=[], fetched_at=1, resets_available=0),
        UsageReport(provider="openai", limits=[], fetched_at=1),
        UsageReport(provider="openai", limits=[], fetched_at=1, resets_available=True),
        UsageReport(provider="openai", limits=[], fetched_at=1, resets_available=-1),
    ]
    rows = mobile._phone_usage_projection(reports, [(0, 5)] * 5)
    assert [row["resets_available"] for row in rows] == [2, 0, None, None, None]
    # Additive: the original keys are untouched for older phone builds.
    assert rows[0]["provider"] == "openai" and "limits" in rows[0]


# -- POST /api/usage/reset -----------------------------------------------------


def _reset_body(**extra):
    body = {
        "provider": "openai",
        "account_index": 1,
        "account_count": 2,
        "redeem_request_id": "redeem-1",
    }
    body.update(extra)
    return body


def _patch_reset(
    monkeypatch: pytest.MonkeyPatch,
    identity: str | None = "second@example.test",
    outcome: str = "reset",
    windows_reset: int = 1,
):
    """Patch the two helpers the route delegates to and record what it asked."""
    calls = {"identity": [], "consume": [], "refresh": []}

    def resolve(provider, account_index, account_count):
        calls["identity"].append((provider, account_index, account_count))
        return identity

    async def consume(provider, who, redeem):
        calls["consume"].append((provider, who, redeem))
        return ResetConsumeResult(outcome=outcome, windows_reset=windows_reset)

    async def refresh(force):
        calls["refresh"].append(force)
        return _refresh_body(reports=[{"provider": "openai", "limits": []}])

    monkeypatch.setattr(mobile, "_phone_reset_identity", resolve)
    monkeypatch.setattr(mobile, "_consume_phone_usage_reset", consume)
    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    return calls


def _cached_usage():
    return {
        "reports": [{"provider": "openai", "limits": []}],
        "source": "cached",
        "may_be_stale": False,
        "cached_at": 7,
    }


def test_reset_requires_pairing_and_same_origin(monkeypatch):
    calls = _patch_reset(monkeypatch)
    with TestClient(_app(), follow_redirects=False) as client:
        assert client.post("/api/usage/reset", json=_reset_body()).status_code == 401
        client.post("/login", data={"password": "fixture-password"})
        crossed = client.post(
            "/api/usage/reset",
            json=_reset_body(),
            headers={"origin": "https://someone-else.example.test"},
        )
        assert crossed.status_code == 403
    assert calls["identity"] == []
    assert calls["consume"] == []


def test_reset_spends_the_indexed_account_and_answers_a_fresh_envelope(monkeypatch):
    calls = _patch_reset(monkeypatch, identity="second@example.test")
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body())
    assert answer.status_code == 200
    assert answer.json()["reset"] == {"outcome": "reset", "windows_reset": 1}
    assert answer.json()["reports"] == [{"provider": "openai", "limits": []}]
    # The phone named an INDEX; the daemon resolved it and spent on that login.
    assert calls["identity"] == [("openai", 1, 2)]
    assert calls["consume"] == [("openai", "second@example.test", "redeem-1")]
    # A spent credit dropped the shared cache row, so the answer re-fetched.
    assert calls["refresh"] == [True]


def test_reset_refuses_when_the_account_list_moved(monkeypatch):
    """An unresolvable index asks the phone to refresh; it never resets a sibling."""
    calls = _patch_reset(monkeypatch, identity=None)
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body())
    assert answer.status_code == 409
    assert answer.json()["code"] == "accounts_changed"
    assert calls["consume"] == []
    assert calls["refresh"] == []


def test_reset_refuses_a_provider_without_resets(monkeypatch):
    calls = _patch_reset(monkeypatch)
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body(provider="anthropic"))
    assert answer.status_code == 422
    assert answer.json()["code"] == "not_supported"
    assert calls["identity"] == []


def test_reset_refuses_a_body_that_is_not_an_exact_index(monkeypatch):
    """`true`/`1.0`/`"1"` are wire bugs: coercing one onto an account spends a
    credit the operator did not confirm."""
    calls = _patch_reset(monkeypatch)
    with _authed_client(_app()) as client:
        for bad in (True, 1.0, "1", -1):
            answer = client.post("/api/usage/reset", json=_reset_body(account_index=bad))
            assert answer.status_code == 400, bad
        assert client.post("/api/usage/reset", json=_reset_body(account_count=0)).status_code == 400
        assert (
            client.post("/api/usage/reset", json=_reset_body(redeem_request_id="")).status_code
            == 400
        )
        assert client.post("/api/usage/reset", content=b"not json").status_code == 400
    assert calls["identity"] == []


def test_reset_is_single_flight_for_the_same_account(monkeypatch):
    """A double tap on one card loses at the claim; a sibling index does not."""
    release = asyncio.Event()
    seen = []

    def resolve(provider, account_index, account_count):
        return f"account-{account_index}"

    async def consume(provider, identity, redeem):
        seen.append(identity)
        await release.wait()
        return ResetConsumeResult(outcome="reset", windows_reset=1)

    async def refresh(force):
        return _refresh_body()

    monkeypatch.setattr(mobile, "_phone_reset_identity", resolve)
    monkeypatch.setattr(mobile, "_consume_phone_usage_reset", consume)
    monkeypatch.setattr(mobile, "_refresh_phone_usage", refresh)
    app = _app()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://phone") as client:
            await client.post("/login", data={"password": "fixture-password"})
            first = asyncio.create_task(client.post("/api/usage/reset", json=_reset_body()))
            await asyncio.sleep(0.05)
            second = await client.post(
                "/api/usage/reset", json=_reset_body(redeem_request_id="redeem-2")
            )
            release.set()
            done = await first
            sibling = await client.post(
                "/api/usage/reset",
                json=_reset_body(account_index=0, redeem_request_id="redeem-3"),
            )
            return done, second, sibling

    done, second, sibling = asyncio.run(scenario())
    assert done.status_code == 200
    assert second.status_code == 409 and second.json()["code"] == "in_flight"
    assert sibling.status_code == 200
    # One spend per attempt, in arrival order, on the accounts they named.
    assert seen == ["account-1", "account-0"]


@pytest.mark.parametrize(
    "outcome,windows,refreshes",
    [
        ("reset", 2, True),
        ("already_redeemed", 0, True),
        ("nothing_to_reset", 0, False),
        ("no_credit", 0, False),
        ("failed", 0, False),
    ],
)
def test_reset_answers_every_outcome_without_claiming_more_than_it_saw(
    monkeypatch, outcome, windows, refreshes
):
    calls = _patch_reset(monkeypatch, outcome=outcome, windows_reset=windows)
    monkeypatch.setattr(mobile, "_cached_phone_usage", _cached_usage)
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body())
    assert answer.status_code == 200
    assert answer.json()["reset"] == {"outcome": outcome, "windows_reset": windows}
    assert answer.json()["reports"] == [{"provider": "openai", "limits": []}]
    assert bool(calls["refresh"]) is refreshes
    if not refreshes:
        # No spend: the cached numbers are still true and no vendor call is made.
        assert answer.json()["source"] == "cached"


def test_an_unnameable_outcome_reads_as_failed(monkeypatch):
    """A wire change is an UNKNOWN spend, never an implied success."""
    calls = _patch_reset(monkeypatch, outcome="reset_pending", windows_reset=0)
    monkeypatch.setattr(mobile, "_cached_phone_usage", _cached_usage)
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body())
    assert answer.json()["reset"] == {"outcome": "failed", "windows_reset": 0}
    assert calls["refresh"] == []


def test_reset_answers_without_an_envelope_when_the_cache_cannot_be_read(monkeypatch):
    """The spend is the fact; the numbers are what could not be re-read."""
    _patch_reset(monkeypatch, outcome="no_credit", windows_reset=0)

    def unavailable():
        raise RuntimeError("cached usage unavailable")

    monkeypatch.setattr(mobile, "_cached_phone_usage", unavailable)
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body())
    assert answer.status_code == 200
    assert answer.json() == {"reset": {"outcome": "no_credit", "windows_reset": 0}}


def test_reset_answers_the_reset_alone_when_the_post_spend_refresh_fails(monkeypatch):
    """A timeout after a spend still says what happened; it invents no numbers."""
    _patch_reset(monkeypatch, outcome="reset", windows_reset=1)

    async def broken(force):
        raise RuntimeError("vendor refused")

    def unavailable():
        raise RuntimeError("cached usage unavailable")

    monkeypatch.setattr(mobile, "_refresh_phone_usage", broken)
    monkeypatch.setattr(mobile, "_cached_phone_usage", unavailable)
    with _authed_client(_app()) as client:
        answer = client.post("/api/usage/reset", json=_reset_body())
    assert answer.status_code == 200
    assert answer.json() == {"reset": {"outcome": "reset", "windows_reset": 1}}


# -- index -> login resolution --------------------------------------------------


class _FakeResetStore:
    def __init__(self, rows):
        self._rows = rows

    def list_credentials(self, provider=None):
        return list(self._rows)

    def close(self):
        pass


class _FakeResetController:
    def __init__(self, snapshot, rows):
        self._snapshot = snapshot
        self.auth_store = _FakeResetStore(rows)
        self.closed = False

    def cached_persisted_usage_reports_read_only(self, provider=None):
        return self._snapshot

    def close(self):
        self.closed = True


def _fake_row(row_id, email=None, kind="oauth"):
    return SimpleNamespace(
        id=row_id, credential_type=kind, identity_key=f"oauth:{row_id}", data={"email": email}
    )


def _snapshot(count, unavailable=False, with_positions=True):
    reports = [UsageReport(provider="openai", limits=[], fetched_at=1) for _ in range(count)]
    positions = [(index, count) for index in range(count)] if with_positions else []
    return ReadOnlyUsageSnapshot(
        reports=reports, unavailable=unavailable, account_positions=positions
    )


def _serve_reset_controller(monkeypatch, controller):
    monkeypatch.setattr(
        ProviderController,
        "read_only_for_persisted_usage",
        staticmethod(lambda *args, **kwargs: controller),
    )


def test_reset_identity_resolves_the_index_in_login_row_order(monkeypatch):
    """Row id order is the ranking key the projection numbers by -- not the
    alphabetical order of the labels, which would renumber on a rename."""
    controller = _FakeResetController(
        _snapshot(2),
        [_fake_row(7, "zeta@example.test"), _fake_row(3, "alpha@example.test")],
    )
    _serve_reset_controller(monkeypatch, controller)
    assert mobile._phone_reset_identity("openai", 0, 2) == "alpha@example.test"
    assert mobile._phone_reset_identity("openai", 1, 2) == "zeta@example.test"
    assert controller.closed is True


def test_reset_identity_refuses_when_the_displayed_count_moved(monkeypatch):
    controller = _FakeResetController(_snapshot(1), [_fake_row(1, "only@example.test")])
    _serve_reset_controller(monkeypatch, controller)
    assert mobile._phone_reset_identity("openai", 0, 2) is None


def test_reset_identity_refuses_when_the_stored_logins_disagree(monkeypatch):
    controller = _FakeResetController(
        _snapshot(2), [_fake_row(1, "one@example.test"), _fake_row(2, "two@example.test")]
    )
    _serve_reset_controller(monkeypatch, controller)
    assert mobile._phone_reset_identity("openai", 0, 3) is None


def test_reset_identity_refuses_an_index_past_the_displayed_accounts(monkeypatch):
    controller = _FakeResetController(
        _snapshot(2), [_fake_row(1, "one@example.test"), _fake_row(2, "two@example.test")]
    )
    _serve_reset_controller(monkeypatch, controller)
    assert mobile._phone_reset_identity("openai", 2, 2) is None


def test_reset_identity_refuses_an_unreadable_snapshot(monkeypatch):
    controller = _FakeResetController(_snapshot(0, unavailable=True), [_fake_row(1, "x@y.z")])
    _serve_reset_controller(monkeypatch, controller)
    assert mobile._phone_reset_identity("openai", 0, 1) is None


def test_reset_identity_refuses_an_api_key_row_and_an_unlabelled_one(monkeypatch):
    key_only = _FakeResetController(_snapshot(1), [_fake_row(1, None, kind="api_key")])
    _serve_reset_controller(monkeypatch, key_only)
    assert mobile._phone_reset_identity("openai", 0, 1) is None
    unlabelled = _FakeResetController(_snapshot(1), [_fake_row(1, None)])
    _serve_reset_controller(monkeypatch, unlabelled)
    assert mobile._phone_reset_identity("openai", 0, 1) is None
