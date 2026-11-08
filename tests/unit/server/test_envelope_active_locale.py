"""Route envelopes render ``detail.text`` in the process's ACTIVE locale.

The desktop UI paints ``detail.text`` for a server-originated refusal, and the
active locale (kept in step with the ``language`` setting by
``follow_language``) is the only language the server has. These tests drive
REAL routes with the process locale set to ``fr`` and pin the contract:

* ``text`` is exactly the ``fr`` catalogue string;
* ``code`` and ``params`` are byte-identical to the English response (clients
  switch on them, never on ``text``);
* with the default ``en`` locale the whole body is unchanged;
* transcript-persisted notice text stays English under ``fr``.

Not covered by design: per-request language for remote clients (the server has
one process locale) and the ``settings_io`` prose carried in
``wire.errors.settings.invalid_value``'s ``{error}`` param, which is English
by origin.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.i18n import catalogues, text
from local_operator.i18n.messages import envelope, msg
from local_operator.server.app import managed_desktop_boundary
from local_operator.server.routes import settings
from local_operator.server.routes.desktop_lifecycle import Aside
from tests.unit.server.test_desktop_aside_route import (
    REQUEST_ID,
    SESSION_ID,
    TOKEN,
    _FakeRemote,
    _install,
    restore_app_state,  # noqa: F401  (fixture re-export, used by name)
)

INLINE_CREDENTIALS = "wire.errors.settings.inline_credentials"
ASIDE_LIMIT = "wire.errors.desktop.aside_limit_reached"


def _catalogue(locale: str, namespace: str, code: str) -> str:
    return catalogues.load_catalogue(locale, namespace)[code]


@pytest_asyncio.fixture
async def settings_client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    # The managed-desktop boundary is the real front door of /v1/settings:
    # without its token the route answers 503 before reaching the refusal.
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(settings.router)
    app.middleware("http")(managed_desktop_boundary)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


async def _refused_settings_save(client: AsyncClient) -> dict[str, Any]:
    response = await client.patch(
        "/v1/settings/web_search.searxng_endpoint",
        json={"value": "https://user:private@example.org"},
    )
    assert response.status_code == 422, response.text
    return response.json()["detail"]


@pytest.mark.asyncio
async def test_settings_refusal_text_follows_the_active_locale(settings_client) -> None:
    english = await _refused_settings_save(settings_client)
    assert english["text"] == _catalogue("en", "wire.errors.settings", INLINE_CREDENTIALS)

    text.set_active_locale("fr")
    french = await _refused_settings_save(settings_client)

    assert french["text"] == _catalogue("fr", "wire.errors.settings", INLINE_CREDENTIALS)
    assert french["text"] != english["text"]
    # The machine contract does not move with the language.
    assert french["code"] == english["code"] == INLINE_CREDENTIALS
    assert french["params"] == english["params"] == {}


@pytest.mark.asyncio
async def test_aside_limit_refusal_text_follows_the_active_locale(
    tmp_path, monkeypatch, restore_app_state: None  # noqa: F811
) -> None:
    app, _bridge = _install(monkeypatch, tmp_path, _FakeRemote())
    # 64 live panels is the route's cap; the 65th ask is the refusal under test.
    app.state.desktop_asides = {f"existing-{n}": Aside(SESSION_ID, [], time.monotonic()) for n in range(64)}

    async def refused() -> dict[str, Any]:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            response = await client.post(
                f"/v1/desktop/sessions/{SESSION_ID}/asides",
                json={"request_id": REQUEST_ID, "text": "why?"},
            )
        assert response.status_code == 409, response.text
        return response.json()["detail"]

    english = await refused()
    assert english["text"] == _catalogue("en", "wire.errors.desktop", ASIDE_LIMIT)

    text.set_active_locale("fr")
    french = await refused()

    assert french["text"] == _catalogue("fr", "wire.errors.desktop", ASIDE_LIMIT)
    assert french["text"] != english["text"]
    assert french["code"] == english["code"] == ASIDE_LIMIT
    assert french["params"] == english["params"] == {}


def test_envelope_default_is_byte_identical_in_english_and_follows_a_move() -> None:
    code = INLINE_CREDENTIALS
    en_text = _catalogue("en", "wire.errors.settings", code)
    # Control: an unmoved process renders exactly what an explicit pin renders.
    assert envelope(code) == envelope(code, locale="en")
    assert envelope(code) == {"code": code, "params": {}, "text": en_text}
    assert msg(code).envelope() == msg(code).envelope(locale="en")

    text.set_active_locale("fr")
    assert envelope(code)["text"] == _catalogue("fr", "wire.errors.settings", code)
    # The explicit pin is the way to stay English under fr.
    assert envelope(code, locale="en")["text"] == en_text
    assert msg(code).envelope(locale="en")["text"] == en_text


def test_session_notices_and_fragments_stay_english_under_fr() -> None:
    """A NoticeEvent is persisted and replayed: its ``text`` must not follow the process language.

    The probe message is one whose ``fr`` text really differs from ``en``, so
    the assertion can fail if the pin is removed.
    """
    from local_operator.harness import loop as loop_module
    from local_operator.session import session as session_module

    probe = msg(ASIDE_LIMIT)
    english = _catalogue("en", "wire.errors.desktop", ASIDE_LIMIT)
    assert _catalogue("fr", "wire.errors.desktop", ASIDE_LIMIT) != english

    text.set_active_locale("fr")
    for notice_factory in (session_module._notice, loop_module._notice):
        notice = notice_factory(probe)
        assert notice.text == english
        assert (notice.code, notice.params) == (ASIDE_LIMIT, {})
    assert session_module._text(probe) == english
