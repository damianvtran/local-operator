"""Keep display localization separate from machine selectors and legacy adapters."""

from types import SimpleNamespace

import pytest

from local_operator.i18n.keys import tui_model_persistence as keys
from local_operator.i18n.text import active_locale, set_active_locale
from local_operator.tui.app import OperatorApp
from local_operator.slash_commands import PERSIST_HINT


def _notice(*, setup: str | None = None, missing: bool = False, label: str = "") -> str:
    # This adapter supplies only the state the production method reads. Real
    # picker opening, styled wrapping and pinned-tail dedup remain pilot tests.
    state = SimpleNamespace(
        _session=SimpleNamespace(model_label=label),
        _setup_state=setup,
        _model_missing_for=missing,
    )
    return OperatorApp._persist_hint_notice(state)


@pytest.fixture(autouse=True)
def restore_locale():
    previous = active_locale()
    yield
    set_active_locale(previous)


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
@pytest.mark.parametrize(
    ("setup", "missing", "routes"),
    [(None, False, ("persist", "saved", "settings")),
     ("missing_model", True, ("persist", "settings")),
     ("no_hosting", False, ("settings",)),
     ("unknown_hosting", False, ("settings",))],
)
def test_notice_resolves_each_semantic_branch_at_current_locale(locale, setup, missing, routes):
    set_active_locale(locale)
    expected_routes = " · ".join(
        getattr(keys, route)().envelope(locale=locale)["text"] for route in routes
    )
    # Selector syntax and labels are opaque input, even when they look like
    # markup. Changing either here would silently break copy/paste workflows.
    label = "vendor/[model]/v1:exact"
    expected = keys.with_model(label=label, routes=expected_routes).envelope(locale=locale)["text"]
    assert _notice(setup=setup, missing=missing, label=label) == expected
    assert _notice(setup=setup, missing=missing) == expected_routes
    assert label in expected
    for route in routes:
        assert {"persist": "/model default", "saved": "/model saved", "settings": "/settings"}[route] in expected


def test_notice_follows_live_locale_without_rewriting_previous_text():
    set_active_locale("en")
    english = _notice(label="opaque/model")
    assert english == (
        f"model: opaque/model — {PERSIST_HINT} · "
        "switch to the boot default with /model saved · /settings edits the boot default too"
    )
    set_active_locale("fr")
    french = _notice(label="opaque/model")
    assert french != english
    assert "saves this for new sessions" not in french
    assert "edits the boot default too" not in french
    assert keys.persist().envelope(locale="en")["text"] == PERSIST_HINT
    set_active_locale("en")
    assert _notice(label="opaque/model") == english
