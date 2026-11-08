"""A LIVE language switch reaches the composer's fallback banner text.

``tui.notify._reload_copy`` rebinds ``BACKGROUND_FALLBACK_TITLE``,
``BODY_COMPLETE`` and ``CONTEXT_COMPLETE`` when ``set_active_locale`` moves the
process; ``BODIES``/``CONTEXTS`` are updated in place. ``compose`` and the push
payload used to import those scalars BY VALUE, so after ``/language fr`` a
nameless completion still came out as
``A session finished | Terminé | Tâche terminée`` — English title beside French
status and body.

Both modules are imported (in English) at the top of this file and the switch
happens inside the test, which is exactly the live-switch ordering; a test that
switched before importing would pass against a by-value import. Expectations
come from the catalogue through the generated key functions, not literals.
"""

from __future__ import annotations

import pytest

from local_operator.i18n.keys import tui_chrome_banners as _x
from local_operator.i18n.resolve import SUPPORTED_LOCALES
from local_operator.i18n.text import set_active_locale, tr
from local_operator.mobile import push_payload
from local_operator.notifications import compose, gate_body

_t = tr("tui.chrome.banners")

#: A kind in neither ``BODIES`` nor ``CONTEXTS``: forces the ``BODY_COMPLETE`` /
#: ``CONTEXT_COMPLETE`` fallback paths, which are the by-value reads.
UNKNOWN_KIND = "not-a-kind"


@pytest.fixture
def names_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The privacy flag on, so a nameless completion reaches the fallback title."""
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)


def _english() -> dict[str, str]:
    set_active_locale("en")
    return {
        "title": _t(_x.fallback_title()),
        "body": _t(_x.body_complete()),
        "status": _t(_x.context_complete()),
    }


@pytest.mark.parametrize("locale", [loc for loc in SUPPORTED_LOCALES if loc != "en"])
def test_a_live_switch_reaches_every_fallback_the_composer_and_push_read(
    names_on: None, locale: str
) -> None:
    english = _english()
    set_active_locale(locale)  # AFTER compose / push_payload were imported in English
    want = {
        "title": _t(_x.fallback_title()),
        "body": _t(_x.body_complete()),
        "status": _t(_x.context_complete()),
    }

    nameless = compose("complete", session_dir=None)
    assert nameless.title == want["title"]
    assert nameless.status == want["status"]
    assert nameless.body == want["body"]

    # The unknown-kind paths read BODY_COMPLETE / CONTEXT_COMPLETE directly.
    unknown = compose(UNKNOWN_KIND, session_dir=None)  # type: ignore[arg-type]
    assert unknown.status == want["status"]
    assert unknown.body == want["body"]
    assert gate_body(UNKNOWN_KIND, "", "") == want["body"]
    alert = push_payload.completion_alert(UNKNOWN_KIND, 1)
    assert alert[push_payload.ALERT_BODY_FIELD].startswith(want["body"])

    # Non-vacuity: where the locale genuinely differs from English, the text
    # that came out is not the English that was bound at import.
    for field, text in (
        ("title", nameless.title),
        ("status", nameless.status),
        ("body", nameless.body),
    ):
        if want[field] != english[field]:
            assert text != english[field], (locale, field)


def test_the_catalogue_really_differs_in_french_so_the_switch_test_is_not_vacuous() -> None:
    english = _english()
    set_active_locale("fr")
    assert _t(_x.fallback_title()) != english["title"]
    assert _t(_x.body_complete()) != english["body"]
    assert _t(_x.context_complete()) != english["status"]


def test_switching_back_restores_english(names_on: None) -> None:
    english = _english()
    set_active_locale("fr")
    set_active_locale("en")
    banner = compose("complete", session_dir=None)
    assert (banner.title, banner.status, banner.body) == (
        english["title"],
        english["status"],
        english["body"],
    )
