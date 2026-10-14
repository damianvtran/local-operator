"""Recognition and negotiation do not activate the held Chinese/Urdu waves."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from local_operator.i18n import locale_direction, resolve
from local_operator.i18n.format import format_number, plural_category
from local_operator.server.routes import capabilities, i18n


@pytest.mark.parametrize(
    "tag, canonical, direction",
    [
        ("zh_CN.UTF-8", "zh-CN", "ltr"),
        ("zh-Hans", "zh-CN", "ltr"),
        ("zh-cn", "zh-CN", "ltr"),
        ("ur_PK.UTF-8", "ur", "rtl"),
        ("ur_IN", "ur", "rtl"),
        ("zh-Hant", None, "ltr"),
        ("unknown", None, "ltr"),
        (None, None, "ltr"),
    ],
)
def test_recognition_direction_and_closed_activation(tag, canonical, direction) -> None:
    assert resolve.normalise_locale(tag) == canonical
    assert locale_direction(tag) == direction
    assert resolve.resolve_language(tag, environ={}, system="Linux") == "en"
    assert "zh-CN" not in resolve.shipped_locales()
    assert "ur" not in resolve.shipped_locales()
    assert {value for value, _ in resolve.language_options()}.isdisjoint({"zh-CN", "ur"})


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
@pytest.mark.parametrize("tag, expected", [("ur_PK", "en"), ("zh-Hans", "en"), ("fr", "fr")])
async def test_mounted_capabilities_metadata_and_catalogue_gate(monkeypatch, tag, expected) -> None:
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
        assert result["language_direction"] == "ltr"
        for held in ("zh-CN", "ur"):
            blocked = await client.get(
                f"/v1/i18n/catalogues/{held}/wire.exec", headers={"If-None-Match": "*"}
            )
            assert blocked.status_code == 404
        control = await client.get(
            "/v1/i18n/catalogues/en/wire.exec", headers={"If-None-Match": "*"}
        )
        assert control.status_code == 304


@pytest.mark.asyncio
async def test_capabilities_resolves_once_for_consistent_direction(monkeypatch) -> None:
    # Synthetic future eligibility is not an activation change: the production
    # held-locale path above must still return en/ltr. This exposes a double-read
    # race and proves that metadata can represent RTL without changing a picker.
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


@pytest.mark.parametrize("tag", ["zh-cn", "zh-Hans", "zh_CN.UTF-8", "ur_PK", "ur_IN"])
def test_os_alias_cannot_activate_an_unshipped_locale(tag) -> None:
    environ = {"LANG": tag}
    assert resolve.os_language(system="Linux", environ=environ) in {"zh-CN", "ur"}
    assert resolve.resolve_language("auto", system="Linux", environ=environ) == "en"
