"""The presentation boundary must never turn translated bytes into auth state."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from local_operator.config import ConfigManager
from local_operator.i18n.catalogues import load_catalogue
from local_operator.i18n.text import active_locale, set_active_locale
from local_operator.tui.app import OperatorApp
from local_operator.tui.provider_state import ProviderStateRow, provider_state_label
from local_operator.tui.widgets.settings_view import SettingsView
from tests.unit.tui.test_app_pilot import FakeProviderController, FakeSession, _factory

STATES = ("logged_in", "env_key", "needs_login", "local_ready", "local_unconfigured")
LOCALES = ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi")


@pytest.fixture(autouse=True)
def restore_locale():
    before = active_locale()
    yield
    set_active_locale(before)


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("code", STATES)
def test_resolver_uses_active_catalogue(locale, code):
    set_active_locale(locale)
    expected = load_catalogue(locale, "tui.provider_state")[f"tui.provider_state.{code}"]
    assert provider_state_label(code) == expected


def test_same_row_live_locale_and_unknown_code():
    row = ProviderStateRow("needs_login", "logged_in")
    for locale, expected in (("en", "logged in"), ("fr", "connecté"), ("en", "logged in")):
        set_active_locale(locale)
        assert provider_state_label(row.state_code) == expected
    with pytest.raises(KeyError):
        provider_state_label("future_unknown")
    with pytest.raises(FrozenInstanceError):
        row.state_code = "needs_login"


@pytest.mark.parametrize("locale", ("en", "fr"))
@pytest.mark.parametrize("code", STATES)
def test_model_access_semantics_do_not_depend_on_label(locale, code, monkeypatch):
    set_active_locale(locale)
    row = SimpleNamespace(definition=SimpleNamespace(id="needs_login"), state=code)
    owner = SimpleNamespace(_providers=SimpleNamespace(view_rows=lambda: [row]))
    # A success label deliberately collides with the old English predicate.
    monkeypatch.setattr("local_operator.tui.app.provider_state_label", lambda code: "needs login")
    suffix, warning = OperatorApp._model_access_note(owner, "needs_login")
    if code == "needs_login":
        assert (suffix, warning) == ("", "needs_login needs login — /login needs_login")
    else:
        assert (suffix, warning) == (" · needs_login needs login", None)


@pytest.mark.parametrize("locale", ("en", "fr"))
def test_missing_unreadable_and_opaque_details(locale):
    set_active_locale(locale)
    owner = SimpleNamespace(_providers=SimpleNamespace(view_rows=lambda: []))
    assert OperatorApp._model_access_note(owner, "opaque") == (
        "", "opaque needs login — /login opaque"
    )
    owner._providers.view_rows = lambda: None
    assert OperatorApp._model_access_note(owner, "opaque") == (
        "", "cannot check opaque credentials: the credential store is unreadable"
    )
    assert OperatorApp._row_state_detail(None) == ""
    for detail in ("needs_login", "logged in", "connecté"):
        row = SimpleNamespace(state="logged_in", state_detail=detail)
        assert OperatorApp._row_state_detail(row) == detail
    assert OperatorApp._row_state_detail(
        SimpleNamespace(state="logged_in")
    ) == provider_state_label("logged_in")
    assert OperatorApp._row_state_detail(
        SimpleNamespace(state="env_key", state_detail="")
    ) == provider_state_label("env_key")


def test_settings_explicit_api_preserves_legacy_collisions(tmp_path, monkeypatch):
    view = SettingsView(ConfigManager(tmp_path))
    # This row-admission unit has no Textual host; mounted paint is tested below.
    monkeypatch.setattr(view, "_repaint", lambda: None)
    view.load(providers=[("needs_login", "logged_in")])
    assert view._providers == [("needs_login", "logged_in")]
    for coded in ([], [ProviderStateRow("needs_login", "logged_in")]):
        with pytest.raises(ValueError):
            view.load(providers=[("needs_login", "logged_in")], provider_states=coded)
    view.load(provider_states=[])
    assert view._providers == []


@pytest.mark.asyncio
async def test_mounted_coded_rows_repaint_reload_reopen_without_store_reads(monkeypatch):
    controller = FakeProviderController()
    reads = 0
    original = controller.view_rows

    def counted():
        nonlocal reads
        reads += 1
        return original()

    monkeypatch.setattr(controller, "view_rows", counted)
    app = OperatorApp(
        lambda: _factory(FakeSession()),
        provider_controller=controller,
        warm_session_imports=False,
    )
    async with app.run_test(size=(150, 45)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        assert view._providers == [ProviderStateRow("openrouter", "logged_in")]
        snapshot = list(view._providers)
        before = reads
        for locale in ("fr", "en", "fr"):
            app._apply_language(locale)
            await pilot.pause()
            assert provider_state_label("logged_in") in view.rendered_pane()
            assert view._providers == snapshot
        view.reload_from_disk()
        await pilot.pause()
        monkeypatch.setattr(view, "_mid_gesture", lambda: True)
        assert view.reload_from_disk() is False
        assert view._reload_pending
        app._apply_language("en")
        await pilot.pause()
        assert "logged in" in view.rendered_pane()
        monkeypatch.setattr(view, "_mid_gesture", lambda: False)
        view._repaint()
        app._apply_language("fr")
        await pilot.pause()
        assert view._providers == snapshot
        assert reads == before
        # Explicit legacy and coded paths coexist without content-based guessing.
        view.load(providers=[("needs_login", "logged_in")])
        assert "logged_in" in view.rendered_pane()
        view.load(provider_states=[ProviderStateRow("needs_login", "logged_in")])
        assert "connecté" in view.rendered_pane()
        assert "needs_login" in view.rendered_pane()
        assert reads == before
        app._close_settings_view()
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        reopened = app.query_one(SettingsView)
        assert "connecté" in reopened.rendered_pane()
        assert reopened._providers == snapshot


@pytest.mark.parametrize("locale", ("en", "fr"))
@pytest.mark.parametrize("code", STATES)
def test_real_model_suffix_and_detail_labels(locale, code):
    set_active_locale(locale)
    row = SimpleNamespace(definition=SimpleNamespace(id="opaque-id"), state=code)
    owner = SimpleNamespace(_providers=SimpleNamespace(view_rows=lambda: [row]))
    suffix, warning = OperatorApp._model_access_note(owner, "opaque-id")
    assert OperatorApp._row_state_detail(row) == provider_state_label(code)
    if code == "needs_login":
        assert suffix == ""
        assert warning == "opaque-id needs login — /login opaque-id"
    else:
        assert suffix == f" · opaque-id {provider_state_label(code)}"
        assert warning is None


@pytest.mark.parametrize("view", (None, {}))
def test_settings_missing_or_unreadable_view_keeps_empty_selection(view):
    owner = SimpleNamespace(
        _providers=SimpleNamespace(login_providers=lambda: [SimpleNamespace(id="opaque")]),
        _provider_view=lambda: view,
    )
    assert OperatorApp._settings_provider_rows(owner) == []


def test_settings_zero_stored_count_is_not_widened():
    row = SimpleNamespace(state="env_key", stored_credentials=0)
    owner = SimpleNamespace(
        _providers=SimpleNamespace(login_providers=lambda: [SimpleNamespace(id="opaque")]),
        _provider_view=lambda: {"opaque": row},
    )
    assert OperatorApp._settings_provider_rows(owner) == []
