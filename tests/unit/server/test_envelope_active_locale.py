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
from tests.unit.server.test_desktop_aside_route import (  # noqa: F401
    REQUEST_ID,
    SESSION_ID,
    TOKEN,
    _FakeRemote,
    _install,
    restore_app_state,
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
    now = time.monotonic()
    app.state.desktop_asides = {f"existing-{n}": Aside(SESSION_ID, [], now) for n in range(64)}

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


def test_model_switch_timeout_suffix_stays_english_under_fr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only consumer interpolates the suffix into an English slash result.

    ``card_timed_out_badge`` really differs between ``en`` and ``fr``, so the
    assertion fails if the ``locale="en"`` pin at the raise site is removed.
    """
    import asyncio

    from local_operator.i18n.keys.relay_asks import card_timed_out_badge
    from local_operator.session.runtime import serving

    english = card_timed_out_badge().envelope(locale="en")["text"]
    assert card_timed_out_badge().envelope(locale="fr")["text"] != english
    # Expire immediately; the window must convert the empty TimeoutError.
    monkeypatch.setattr(serving, "_MODEL_SWITCH_RESOLVE_S", 0.0)

    async def run() -> None:
        async with serving._model_resolution_window(time.monotonic() + 1):
            await asyncio.sleep(1)

    text.set_active_locale("fr")
    with pytest.raises(TimeoutError) as raised:
        asyncio.run(run())
    assert str(raised.value) == english


def _unsupported_projection(config_dir: Any) -> Any:
    """The REAL LinkError raised by the managed-skill storage on this refusal."""
    from local_operator.hub_sync.identity import LinkError
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage

    storage = ManagedSkillStorage(config_dir)
    storage._projection_supported = lambda: False  # type: ignore[method-assign]
    with pytest.raises(LinkError) as raised:
        storage._require_projection_support()
    return raised.value


def test_link_error_paints_localized_but_model_and_persisted_consumers_get_english(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three catalogued hub_sync refusals serve two audiences from one error.

    ``str(error)`` is what the desktop/CLI/TUI paint (active locale); the
    org_pull tool result, the skill tool result and the persisted hub-state
    ``last_error`` re-render from ``code``/``params`` pinned to ``en``.
    Reproduces the review finding: before the pin, ``PullOutcome.detail`` carried
    the French sentence into a model-read tool result.
    """
    from local_operator.hub_org import pull
    from local_operator.hub_sync.identity import refusal_text_en

    key = "wire.skill_adoption.projection_unsupported"
    english = _catalogue("en", "wire.skill_adoption", key)
    french = _catalogue("fr", "wire.skill_adoption", key)
    assert french != english

    text.set_active_locale("fr")
    error = _unsupported_projection(tmp_path)
    assert error.code == "projection-unsupported"
    assert str(error) == french  # the human-facing consumers keep the locale
    assert refusal_text_en(error) == english

    # org_pull path: a per-item failure becomes PullOutcome.detail.
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(pull, "_pull_skill", boom)
    item = pull.CatalogueItem("skill", "hub-1", "demo", "", "", "absent", None)
    outcome, _ = pull._one(None, item, on_conflict="skip", subscribe=True)  # type: ignore[arg-type]
    assert outcome.outcome == "failed"
    assert outcome.detail == f"projection-unsupported: {english[:300]}"
    assert french[:40] not in outcome.detail


def test_adoption_refusals_carry_their_message_and_other_errors_pass_through() -> None:
    from local_operator.hub_sync.identity import LinkError, refusal_text_en
    from local_operator.hub_sync.skill_adoption_service import _refusal

    for code, key in (
        ("alias-occupied", "wire.skill_adoption.preview_required"),
        ("adoption-preview-changed", "wire.skill_adoption.preview_changed"),
        ("adoption-needs-choice", "wire.skill_adoption.needs_choice"),
    ):
        text.set_active_locale("fr")
        error = _refusal(code, "unused")
        assert str(error) == _catalogue("fr", "wire.skill_adoption", key)
        assert refusal_text_en(error) == _catalogue("en", "wire.skill_adoption", key)
    # A hand-written English LinkError and a foreign exception are unchanged.
    assert refusal_text_en(_refusal("something-else", "plain english")) == "plain english"
    assert refusal_text_en(LinkError("x", "plain")) == "plain"
    assert refusal_text_en(ValueError("boom")) == "boom"


def test_skill_check_detail_stays_english_when_storage_refuses_under_fr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA Q1: the check's persisted ``detail`` must not carry the French sentence.

    ``check_hub_skills`` wraps the storage's catalogued ``projection-unsupported``
    refusal (Windows-style: no secure projection, but a projection file already
    exists for the alias) into ``ItemCheck.detail``, which ``status.json``
    persists as ``last_error`` and the model reads. Wrapping ``str(error)`` put
    the active-locale (fr) text there; ``refusal_text_en`` keeps it English.
    """
    from local_operator.hub_sync import check as chk
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from tests.unit.hub_sync.test_skill_apply import _profile

    key = "wire.skill_adoption.projection_unsupported"
    english = _catalogue("en", "wire.skill_adoption", key)
    french = _catalogue("fr", "wire.skill_adoption", key)
    assert french != english

    storage, hub, _ctx, _identity, document = _profile(tmp_path)
    # A projection file the (simulated) unsupported platform cannot read safely:
    # ``_projection_absent`` is False, so ``snapshot`` raises the catalogued error.
    storage._write_projection(document.name, {})
    monkeypatch.setattr(ManagedSkillStorage, "_projection_supported", lambda self: False)

    text.set_active_locale("fr")
    (row,) = chk.check_hub_skills(
        config_dir=storage.config_dir, client_for_tenant=lambda _t: hub.client()
    )
    assert row.reason == "local-concurrent-edit"
    assert row.detail == f"could not read the local skill: {english}"
    assert french not in (row.detail or "")


def test_local_skill_fields_wrap_keeps_storage_refusal_english_under_fr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish/link path re-wraps the same storage refusal; it stays English.

    ``_local_skill_fields`` turns an unreadable local copy into a new LinkError
    whose text is persisted (link status / pending receipts) and read by the
    model, so it must not embed the active-locale ``str(error)``.
    """
    from local_operator.hub_sync import provenance as prov
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from tests.unit.hub_sync.test_skill_apply import _profile

    key = "wire.skill_adoption.projection_unsupported"
    english = _catalogue("en", "wire.skill_adoption", key)

    storage, _hub, ctx, identity, document = _profile(tmp_path)
    storage._write_projection(document.name, {})
    monkeypatch.setattr(ManagedSkillStorage, "_projection_supported", lambda self: False)
    record = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert record is not None

    text.set_active_locale("fr")
    with pytest.raises(LinkError) as raised:
        svc._local_skill_fields(ctx, record)
    assert raised.value.code == "local-concurrent-edit"
    assert str(raised.value) == f"could not read the local skill: {english}"
