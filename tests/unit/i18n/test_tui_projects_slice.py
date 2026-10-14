"""The Projects send card's extraction-time English and locale boundaries.

The golden is transcribed from the pre-slice source at 9fbd5c89. This slice owns
only projects_send.py: other Projects views remain advisory. Rendered-frame
before/after evidence is a separate allocation, not claimed by these checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve, runtime, text
from local_operator.i18n.keys import tui_projects as copy
from local_operator.tui.widgets import projects_send as send

FIXTURE = Path(__file__).parent / "fixtures" / "tui_projects_en_golden.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _en(key: str, **params: object) -> str:
    return runtime.render_message(GOLDEN[f"tui.projects.send.{key}"], params, "en")


def test_catalogue_matches_the_extraction_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.projects") == GOLDEN
    assert len(GOLDEN) == 18
    context = json.loads(
        (catalogues.catalogue_root() / "en" / "tui.projects.context.json").read_text()
    )
    assert set(context) == set(GOLDEN)
    assert all(all(value.get(k) for k in ("note", "role", "tone")) for value in context.values())


def test_real_send_receipts_and_constants_match_golden() -> None:
    target = send.SendTarget("session", "abcdefghijkl-more", "work", "live", True)
    handle = "◆ abcdefghijkl"
    assert send.compose_band(target) == _en("compose", handle=handle)
    assert send.pending_line(target) == _en("pending", handle=handle)
    assert send.sent_line(target, "delivered") == _en("sent", handle=handle, state_word="delivered")
    assert send.refusal_notice("opaque reason") == _en("refusal", reason="opaque reason")
    assert send.error_notice("opaque reason") == _en("error", reason="opaque reason")
    assert send.NO_TARGET_FOOTER == _en("empty")
    assert send.NO_MATCH_FOOTER == _en("no_match")
    assert send.SEND_FAULT_NOTICE == _en("fault")
    assert send.amber_notice("wake unconfirmed") == _en("wake_unconfirmed")
    assert send.amber_notice("delivery unconfirmed") == _en("delivery_unconfirmed")
    assert send.amber_notice("unknown opaque state") == "unknown opaque state"
    assert target.row_text == "work  · session abcdefghijkl  · [live]"
    assert send.SendTarget("session", "abc", "abc", "missing").row_text == "abc  · [missing]"
    for count in (0, 1, 2, 21, 1000):
        assert send._t(copy.send_more(count=count, label=str(count))) == f"+{count} more"


def test_real_card_chrome_matches_golden() -> None:
    card = send.SendTargetCard([])
    widgets = list(card.compose())
    assert str(widgets[0].render()) == _en("title")
    assert widgets[2].placeholder == _en("filter")
    assert str(widgets[-1].render()) == _en("legend")
    assert [
        b.description for bindings in card._bindings.key_to_bindings.values() for b in bindings
    ] == [_en("binding_move"), _en("binding_move"), _en("binding_choose"), _en("binding_close")]


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    for locale, values in (("en", GOLDEN), ("fr", {k: "FR " + v for k, v in GOLDEN.items()})):
        folder = tmp_path / locale
        folder.mkdir()
        (folder / "tui.projects.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [send._reload_copy])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr", "ru"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    # Restore materialised constants and Textual declarations before restoring
    # the real catalogue root, so fixture translations never escape this test.
    text.set_active_locale("en")
    send._reload_copy()
    text._sources.cache_clear()


def test_backend_locale_refreshes_constants_bindings_and_dynamic_receipts(corpus) -> None:
    card = send.SendTargetCard([])
    target = send.SendTarget("session", "abc", "work")
    text.set_active_locale("fr")
    assert send.NO_TARGET_FOOTER == "FR " + _en("empty")
    assert send.NO_MATCH_FOOTER == "FR " + _en("no_match")
    assert send.SEND_FAULT_NOTICE == "FR " + _en("fault")
    assert send.amber_notice("wake unconfirmed") == "FR " + _en("wake_unconfirmed")
    assert send.pending_line(target) == "FR " + _en("pending", handle="◆ abc")
    assert card._bindings.key_to_bindings["enter"][0].description == "FR Choose"
    assert send.SendTargetCard([])._bindings.key_to_bindings["enter"][0].description == "FR Choose"
    assert list(card.compose())[2].placeholder == "FR " + _en("filter")
    text.set_active_locale("en")
    assert card._bindings.key_to_bindings["enter"][0].description == "Choose"


def test_missing_namespace_falls_back_with_english_plural_rules(corpus) -> None:
    # The production English branches deliberately paint identical bytes. A
    # fixture-only distinct body makes the wrong source-locale choice observable:
    # Russian 21 selects one, whereas English 21 must select other.
    source = "{count, plural, one {EN ONE {label}} other {EN OTHER {label}}}"
    values = dict(GOLDEN, **{"tui.projects.send.more": source})
    path = catalogues.catalogue_root() / "en" / "tui.projects.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    text._sources.cache_clear()
    params = {"count": 21, "label": "21"}
    assert runtime.render_message(source, params, "ru") == "EN ONE 21"
    text.set_active_locale("ru")
    assert text.active_locale() == "ru"
    assert send._t(copy.send_more(count=21, label="21")) == "EN OTHER 21"
    assert send.NO_TARGET_FOOTER == _en("empty")


@pytest.mark.asyncio
async def test_app_fault_consumer_resolves_after_backend_locale_move(corpus, monkeypatch) -> None:
    """Exercise the real worker's fault branch, not just a refreshed module global."""
    from types import SimpleNamespace
    from typing import cast

    from local_operator.mobile import peer_send
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.projects_view import ProjectsView

    async def identity(_pid):
        return {"session_id": "synthetic-sender"}

    def fail_resolution(**_kwargs):
        raise RuntimeError("synthetic local fault")

    monkeypatch.setattr(peer_send, "peer_sender_identity_async", identity)
    monkeypatch.setattr(peer_send, "validate_peer_body", lambda _body: None)
    monkeypatch.setattr(peer_send, "resolve_peer_target", fail_resolution)
    settled = []
    host = cast(
        OperatorApp, SimpleNamespace(_quick_send_settled=lambda *args: settled.append(args))
    )
    view = cast(ProjectsView, object())
    target = send.SendTarget("session", "synthetic-recipient", "work")
    # Import the application BEFORE moving locale: a by-value notice import
    # would retain English here despite the source module's refreshed constant.
    text.set_active_locale("fr")
    await OperatorApp._quick_send_worker(host, view, target, "synthetic draft")
    assert settled == [(view, target, "FR " + _en("fault"), "synthetic draft")]
    text.set_active_locale("en")
    await OperatorApp._quick_send_worker(host, view, target, "synthetic draft")
    assert settled[-1] == (view, target, _en("fault"), "synthetic draft")


def test_disk_only_locale_fails_explicit_eligibility(corpus, monkeypatch) -> None:
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en",))
    # Disk presence is not publication permission, even when all keys exist.
    assert catalogues.load_catalogue("fr", "tui.projects")
    assert resolve.eligible_locale("fr") == "en"
    text.set_active_locale("fr")
    assert text.active_locale() == "en"
    assert send.NO_TARGET_FOOTER == _en("empty")
    assert send._t(copy.send_title()) == _en("title")
