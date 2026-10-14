"""Phone usage is an authenticated cached display, never a provider probe."""
from unittest.mock import Mock, patch

from starlette.testclient import TestClient

from local_operator.mobile import daemon as mobile
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport


def test_phone_usage_gate_precedes_cached_data(monkeypatch):
    calls = []
    monkeypatch.setattr(mobile, "_cached_phone_usage", lambda: calls.append(True) or [])
    app = mobile.build_app(mobile.MobileDaemon(password="fixture-password", dial_registrants=False))
    with TestClient(app, follow_redirects=False) as client:
        assert client.get("/api/usage").status_code == 401
        assert not calls
        client.post("/login", data={"password": "fixture-password"})
        response = client.get("/api/usage")
        assert response.status_code == 200
        assert response.json() == {"reports": [], "source": "cached"}
        assert calls == [True]


def test_phone_projection_preserves_blocked_amount_without_identity(monkeypatch):
    report = UsageReport(provider="openai", limits=[UsageLimit(
        id="wallet", label="private-account@example.invalid", paid_fallback=True,
        available=False, amount=UsageAmount(remaining=61899.95877, unit="credits"),
        detail="private detail", unlimited=False,
    )], identity="private-account@example.invalid")
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = [report]
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ) as factory:
        rows = mobile._cached_phone_usage()
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
        UsageLimit(id="over", label="paid", paid_fallback=True, available=True,
                   amount=UsageAmount(remaining=10, used_fraction=1, unit="credits")),
        UsageLimit(id="uncapped", label="paid", paid_fallback=True, available=True,
                   unlimited=True, amount=UsageAmount(unit="credits")),
    ]
    report = UsageReport(provider="openai", limits=limits)
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = [report]
    with patch(
        "local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
        return_value=controller,
    ):
        rows = mobile._cached_phone_usage()
    assert rows[0]["limits"][0]["payable"] is False
    assert rows[0]["limits"][1]["payable"] is True
    assert rows[0]["limits"][1]["amount"]["limit"] is None


def test_phone_window_metadata_is_exact_allowlist_not_vendor_copy():
    report = UsageReport(provider="anthropic", limits=[
        UsageLimit(id="a", label="secret label", window="5 hour", shared=True,
                   amount=UsageAmount(used_fraction=0.2)),
        UsageLimit(id="b", label="secret label", window="7 day", tier="opus",
                   amount=UsageAmount(used_fraction=1), resets_at_ms=1700000000000),
        UsageLimit(id="c", label="secret label", window="private@example.invalid", tier="secret family",
                   amount=UsageAmount()),
    ])
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = [report]
    with patch("local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
               return_value=controller):
        rows = mobile._cached_phone_usage()
    windows = rows[0]["limits"]
    assert windows[0]["window"] == "5h" and windows[0]["shared"] is True
    assert windows[1]["window"] == "7d" and windows[1]["family"] == "opus"
    assert windows[1]["resets_at_ms"] == 1700000000000
    assert windows[2]["window"] == "unknown" and windows[2]["family"] == "unknown"
    assert "secret" not in str(rows) and "private@" not in str(rows)


def test_phone_unavailable_authority_is_not_empty_and_closes_owned_handle():
    import pytest
    controller = Mock()
    controller.cached_persisted_usage_reports_read_only.return_value = None
    with patch("local_operator.providers.controller.ProviderController.read_only_for_persisted_usage",
               return_value=controller), pytest.raises(RuntimeError, match="cached usage unavailable"):
        mobile._cached_phone_usage()
    controller.auth_store.close.assert_called_once_with()
