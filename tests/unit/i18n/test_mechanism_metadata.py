"""Recognition, negotiation and the shipped filter across the active set.

Every supported tag must normalise to its canonical locale, and the shipped
filter must admit exactly the locales the ledger passed. Chinese was the
long-held example here; its disposition is released (2026-10-24), so the held
case these tests still exercise is a locale the SUPPORTED set does not carry
at all (zh-Hant/unknown), not zh-CN.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from local_operator.i18n import locale_direction, resolve
from local_operator.i18n.format import format_number, plural_category
from local_operator.server.routes import capabilities, i18n


@pytest.mark.parametrize(
    "tag, canonical, direction, expected",
    [
        ("zh_CN.UTF-8", "zh-CN", "ltr", "zh-CN"),
        ("zh-Hans", "zh-CN", "ltr", "zh-CN"),
        ("zh-cn", "zh-CN", "ltr", "zh-CN"),
        ("ur_PK.UTF-8", "ur", "rtl", "ur"),
        ("ur_IN", "ur", "rtl", "ur"),
        ("zh-Hant", None, "ltr", "en"),
        ("unknown", None, "ltr", "en"),
        (None, None, "ltr", "en"),
    ],
)
def test_recognition_direction_and_shipped_filter(tag, canonical, direction, expected) -> None:
    assert resolve.normalise_locale(tag) == canonical
    assert locale_direction(tag) == direction
    assert resolve.resolve_language(tag, environ={}, system="Linux") == expected
    # Both audited waves are in the generated set; the filter, not recognition,
    # is what decides eligibility.
    assert "zh-CN" in resolve.shipped_locales()
    assert "ur" in resolve.shipped_locales()
    options = {value for value, _ in resolve.language_options()}
    assert "zh-CN" in options
    assert "ur" in options


@pytest.mark.parametrize("count", [0, 1, 2, 11, 101, 1.5])
def test_existing_wave_plural_rules(count) -> None:
    assert plural_category("zh-CN", count) == "other"
    assert plural_category("ur", count) == ("one" if count == 1 else "other")


@pytest.mark.parametrize("tag", ["ur_PK", "ur_IN", "zh_CN", "zh-Hans"])
def test_existing_formats_use_rfc_latin_digits(tag) -> None:
    # ur-IN's regional Intl default differs; v1 intentionally normalizes to ur
    # and uses Latin digits, not a new unreviewed regional numeral override.
    locale = resolve.normalise_locale(tag)
    assert locale is not None
    assert format_number(1234567.5, locale) == "1,234,567.5"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tag, expected, direction",
    [("ur_PK", "ur", "rtl"), ("zh-Hans", "zh-CN", "ltr"), ("fr", "fr", "ltr")],
)
async def test_mounted_capabilities_metadata_and_catalogue_gate(
    monkeypatch, tag, expected, direction
) -> None:
    monkeypatch.setattr(resolve, "config_value", lambda: tag)
    app = FastAPI()
    app.include_router(capabilities.router)
    app.include_router(i18n.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://isolated"
    ) as client:
        response = await client.get("/v1/capabilities")
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["language"] == expected
        assert result["language_direction"] == direction
        for blocked_path in ("zh-CN/wire.settings.account_policy", "ur/wire.exec"):
            blocked = await client.get(
                f"/v1/i18n/catalogues/{blocked_path}", headers={"If-None-Match": "*"}
            )
            # Neither locale carries that namespace: account_policy is an
            # approved English-only child, and Urdu does not carry wire.exec.
            # The raw route serves committed namespaces only, not synthetic
            # fallback files.
            assert blocked.status_code == 404
        for shipped in ("ur", "zh-CN", "en"):
            control = await client.get(
                f"/v1/i18n/catalogues/{shipped}/wire.errors", headers={"If-None-Match": "*"}
            )
            assert control.status_code == 304
        # The relay.asks slice's own namespace is served from the locale that
        # carries it: a shipped locale's committed catalogue revalidates, and
        # the namespace it never carried stays a plain inventory 404.
        served = await client.get(
            "/v1/i18n/catalogues/ur/relay.asks", headers={"If-None-Match": "*"}
        )
        assert served.status_code == 304
        absent = await client.get(
            "/v1/i18n/catalogues/ur/relay.approvals", headers={"If-None-Match": "*"}
        )
        assert absent.status_code == 404


@pytest.mark.asyncio
async def test_capabilities_resolves_once_for_consistent_direction(monkeypatch) -> None:
    # The production Urdu path above returns ur/rtl and Chinese returns
    # zh-CN/ltr. This synthetic sequence isolates the double-read race: capabilities
    # must derive language and direction from one resolution snapshot.
    calls = []

    def changing_language():
        calls.append(None)
        return "ur" if len(calls) == 1 else "en"

    monkeypatch.setattr(capabilities, "resolve_language", changing_language)
    response = await capabilities.capabilities()
    result = response.result
    assert result is not None
    assert result["language"] == "ur"
    assert result["language_direction"] == "rtl"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "tag, canonical, expected",
    [
        ("zh-cn", "zh-CN", "zh-CN"),
        ("zh-Hans", "zh-CN", "zh-CN"),
        ("zh_CN.UTF-8", "zh-CN", "zh-CN"),
        ("ur_PK", "ur", "ur"),
        ("ur_IN", "ur", "ur"),
    ],
)
def test_os_alias_resolves_through_shipped_filter(tag, canonical, expected) -> None:
    environ = {"LANG": tag}
    assert resolve.os_language(system="Linux", environ=environ) == canonical
    assert resolve.resolve_language("auto", system="Linux", environ=environ) == expected
