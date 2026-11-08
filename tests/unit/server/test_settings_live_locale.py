"""A language change moves the RUNNING server's copy, not just the next boot's.

The server activates its locale once at startup. Before this fix nothing moved
it afterwards, so ``settings_io``'s registry (label/help/description/choices,
frozen at import under the boot locale and re-rendered only by a locale move's
``register_refresh`` hooks) and every ``tr(...)`` wire resolver
(``routes/agents.py``'s ``_t``) kept answering in the BOOT language after
``PATCH /v1/settings/language``. These tests boot the real routers in one
locale, change the language through the real PATCH / the real feed doorbell,
and read the same endpoints back.

The active locale is process-global by design (``tests/conftest.py``
``restore_active_locale`` restores it), so these tests set it through the public
setter and never assign the scalar.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from local_operator.config import ConfigManager
from local_operator.i18n import text as i18n_text
from local_operator.server.routes import agents as agents_routes
from local_operator.server.routes import settings as settings_routes

# Distinct, shipped, non-English locales plus English: every hop changes the copy.
_LABELS = {"fr": "Langue", "en": "Language", "es": "Idioma"}


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "test-token")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_LANG", raising=False)
    manager = ConfigManager(tmp_path)

    async def _credentials(*_args: Any, **_kwargs: Any) -> tuple[SecretStr, str]:
        # The hub credential is irrelevant to the copy under test; the route
        # reaches its local ``hub_agent_id`` validation (an ``_t`` sentence)
        # before any network call.
        return SecretStr("k"), "http://hub.invalid"

    monkeypatch.setattr(agents_routes, "_radient_route_credentials", _credentials)
    app = FastAPI()
    app.include_router(settings_routes.router)
    app.include_router(agents_routes.router)
    app.dependency_overrides[settings_routes.get_config_manager] = lambda: manager
    app.dependency_overrides[agents_routes.get_agent_registry] = lambda: object()
    app.dependency_overrides[agents_routes.get_env_config] = lambda: object()
    app.dependency_overrides[agents_routes.get_config_manager] = lambda: manager
    app.dependency_overrides[agents_routes.get_provider_auth_store] = lambda: object()
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://ui.test",
        headers={"Authorization": "Bearer test-token"},
    )


async def _language_row(ui: httpx.AsyncClient) -> dict[str, Any]:
    listing = await ui.get("/v1/settings")
    assert listing.status_code == 200, listing.text
    return next(row for row in listing.json()["result"]["settings"] if row["key"] == "language")


async def _bool_choice_description(ui: httpx.AsyncClient) -> str:
    # BOOL rows carry per-choice descriptions (the language row's ``auto`` has none).
    listing = await ui.get("/v1/settings")
    row = next(r for r in listing.json()["result"]["settings"] if r["key"] == "compaction.enabled")
    assert row["choices"][0]["description"], row
    return row["choices"][0]["description"]


async def _publish_refusal(ui: httpx.AsyncClient) -> str:
    # ``hub_agent_id`` empty: the ``_t(listing_id_required)`` arm, reached before any hub call.
    response = await ui.put("/v1/agents/anything/publish", json={"hub_agent_id": ""})
    assert response.status_code == 422, response.text
    return response.json()["detail"]["text"]


@pytest.mark.asyncio
async def test_patch_language_moves_registry_and_wire_copy_in_the_running_server(
    tmp_path, monkeypatch
):
    ui = _client(tmp_path, monkeypatch)
    # Boot the process in French, as ``activate_resolved_locale`` would.
    i18n_text.set_active_locale("fr")
    seen: dict[str, tuple[str, str, str, str]] = {}
    async with ui:
        before = await _language_row(ui)
        assert before["label"] == _LABELS["fr"]
        for locale in ("en", "es", "fr"):
            patched = await ui.patch("/v1/settings/language", json={"value": locale})
            assert patched.status_code == 200, patched.text
            # The PATCH's own response is already in the new language ...
            assert patched.json()["result"]["label"] == _LABELS[locale]
            # ... and so is the next GET (label, help, choice descriptions) ...
            row = await _language_row(ui)
            assert row["label"] == _LABELS[locale]
            seen[locale] = (
                row["label"],
                row["help"],
                await _bool_choice_description(ui),
                await _publish_refusal(ui),
            )
    # ... and a wire resolver on another router (``routes/agents.py``).
    assert len({copy[1] for copy in seen.values()}) == 3, seen
    assert len({copy[2] for copy in seen.values()}) == 3, seen
    assert len({copy[3] for copy in seen.values()}) == 3, seen
    assert i18n_text.active_locale() == "fr"


@pytest.mark.asyncio
async def test_reset_language_follows_the_resolved_language(tmp_path, monkeypatch):
    ui = _client(tmp_path, monkeypatch)
    async with ui:
        assert (await ui.patch("/v1/settings/language", json={"value": "es"})).status_code == 200
        assert (await _language_row(ui))["label"] == _LABELS["es"]
        # Reset goes back to ``auto``; with no LOP_LANG/system locale it is English.
        monkeypatch.setenv("LC_ALL", "C")
        reset = await ui.post("/v1/settings/language/reset")
        assert reset.status_code == 200, reset.text
        assert reset.json()["result"]["label"] == _LABELS["en"]
        assert (await _language_row(ui))["label"] == _LABELS["en"]


@pytest.mark.asyncio
async def test_unrelated_setting_write_does_not_move_the_locale(tmp_path, monkeypatch):
    ui = _client(tmp_path, monkeypatch)
    i18n_text.set_active_locale("fr")
    async with ui:
        row = await _language_row(ui)
        assert row["label"] == _LABELS["fr"]
        other = next(
            r
            for r in (await ui.get("/v1/settings")).json()["result"]["settings"]
            if r["kind"] == "bool"
        )
        saved = await ui.patch(f"/v1/settings/{other['key']}", json={"value": other["value"]})
        assert saved.status_code == 200, saved.text
    assert i18n_text.active_locale() == "fr"


def test_feed_doorbell_follows_a_language_written_by_another_process(tmp_path, monkeypatch):
    """A write from the CLI/TUI never touches this process's PATCH route; the feed's
    ``config.yml`` doorbell is the server's view of it and must move the locale."""
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    (tmp_path / "config.yml").write_text("values:\n  language: fr\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_LANG", raising=False)
    i18n_text.set_active_locale("fr")
    from local_operator import settings_io

    assert settings_io.BY_KEY["language"].label == _LABELS["fr"]

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _queued(subscription)

    (tmp_path / "config.yml").write_text("values:\n  language: es\n", encoding="utf-8")
    _tick(feed)

    assert i18n_text.active_locale() == "es"
    assert settings_io.BY_KEY["language"].label == _LABELS["es"]
    asyncio.run(feed.close())
