"""Projects send/view extraction-time English and locale boundaries.

Send copy was transcribed at 9fbd5c89; summary/view copy at e24f5d675. Only
projects_send.py and projects_view.py are activated, not their render/detail/
form neighbours. These checks prove text bytes, not rendered-frame design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.widgets import Input

from local_operator.i18n import catalogues, resolve, runtime, text
from local_operator.i18n.keys import tui_projects as copy
from local_operator.tui.widgets import projects_send as send
from local_operator.tui.widgets import projects_view as view

FIXTURE = Path(__file__).parent / "fixtures" / "tui_projects_en_golden.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _en(key: str, **params: object) -> str:
    return runtime.render_message(GOLDEN[f"tui.projects.send.{key}"], params, "en")


def test_catalogue_matches_the_extraction_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.projects") == GOLDEN
    assert len(GOLDEN) == 58
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
    # ``compose`` yields the title and rule as ``Static`` before the filter, so the
    # filter is the third widget. Assert its concrete type first: pyright types the
    # list as ``Widget``, and ``placeholder`` lives on ``Input`` only, so the type
    # check needs the narrowing that the runtime already guarantees here.
    filter_widget = widgets[2]
    assert isinstance(filter_widget, Input)
    assert filter_widget.placeholder == _en("filter")
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
    monkeypatch.setattr(text, "_refreshers", [send._reload_copy, view._reload_copy])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr", "ru"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    # Restore materialised constants and Textual declarations before restoring
    # the real catalogue root, so fixture translations never escape this test.
    text.set_active_locale("en")
    send._reload_copy()
    view._reload_copy()
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
    fr_filter = list(card.compose())[2]
    assert isinstance(fr_filter, Input)
    assert fr_filter.placeholder == "FR " + _en("filter")
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


def test_view_keys_counts_and_binding_grammar_match_source() -> None:
    assert view.VIEWS == ("list", "board", "timeline")
    assert view._VIEW_LABELS == {token: token for token in view.VIEWS}
    for count in (0, 1, 2, 21, 1000):
        # No hand-rolled plural existed: keep the original ungrouped count,
        # including 1 tracked, rather than introducing new English grammar.
        assert view._t(copy.tracked(count=str(count))) == f" · {count} tracked"
        assert view._t(copy.this_session(count=str(count))) == f" · this session ({count})"
    for binding in view._project_bindings():
        assert binding.description == GOLDEN[f"tui.projects.binding_{binding.action}"]
        assert binding.show is False


@pytest.mark.asyncio
async def test_real_view_chrome_bytes_and_live_locale(corpus, monkeypatch) -> None:
    import time

    from textual.app import App, ComposeResult

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield view.ProjectsView()

    monkeypatch.setattr(
        time, "localtime", lambda _stamp: time.struct_time((2026, 10, 14, 14, 5, 0, 2, 287, -1))
    )
    app = Host()
    async with app.run_test(size=(200, 40)) as pilot:
        page = app.query_one(view.ProjectsView)
        page._views = [{}, {}, {}]
        page._view = "timeline"
        page._tier = "month"
        page._updated_at = 1
        page._associated = frozenset({"synthetic-project"})
        page._paint_chrome()
        assert str(page._title.render()) == (
            "projects · timeline · this session (1) · 3 tracked · zoom: month · updated 14:05"
        )
        for token in view.VIEWS:
            page._view = token
            page._paint_chrome()
            assert str(page._title.render()).startswith(f"projects · {token} · this session (1)")
        page._view = "timeline"
        page._cursor = 2
        text.set_active_locale("fr")
        await pilot.pause()
        assert str(page._title.render()) == (
            "FR projectsFR  · FR timelineFR  · this session (1)FR  · 3 tracked"
            "FR  · zoom: monthFR  · updated 14:05"
        )
        assert (page._view, page._cursor, page._tier) == ("timeline", 2, "month")
        assert page._bindings.key_to_bindings["1"][0].description == "FR List view"
        assert view.ProjectsView()._bindings.key_to_bindings["1"][0].description == "FR List view"
        page._mode = "form"
        page._paint_form_chrome()
        assert str(page._title.render()) == "FR projectsFR  · new project"
        page._mode = "detail"
        page._paint_detail_chrome()
        assert (
            str(page._title.render()) == "FR projects · FR (unnamed) [● active]FR  · updated 14:05"
        )
        text.set_active_locale("en")
        assert page._bindings.key_to_bindings["1"][0].description == "List view"
        assert str(page._title.render()) == "projects · (unnamed) [● active] · updated 14:05"
        page._mode = "form"
        page._paint_form_chrome()
        assert str(page._title.render()) == "projects · new project"


@pytest.mark.parametrize(
    "planted",
    [
        'Static("planted untranslated project title")',
        'Input(placeholder="planted untranslated project hint")',
        'title.append("planted untranslated project clause")',
    ],
)
def test_view_activation_rejects_real_source_literals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, i18n_check, corpus, planted: str
) -> None:
    """Run the real checker against a shadow of this file, never mutate live source."""
    repo = Path(__file__).resolve().parents[3]
    relative = "local_operator/tui/widgets/projects_view.py"
    source = (repo / relative).read_text(encoding="utf-8")
    shadow = tmp_path / "shadow"
    target = shadow / relative
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    state = shadow / "i18n"
    state.mkdir()
    baseline = json.loads((repo / "i18n/baseline.json").read_text(encoding="utf-8"))
    assert relative in baseline["enforced"]
    assert baseline["files"].get(relative, 0) == 0
    (state / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (state / "allowlist.toml").write_text("entries = []\n", encoding="utf-8")
    monkeypatch.setattr(i18n_check, "REPO", shadow)
    monkeypatch.setattr(i18n_check, "BASELINE", state / "baseline.json")
    monkeypatch.setattr(i18n_check, "ALLOWLIST", state / "allowlist.toml")
    assert i18n_check.main([]) == 0
    target.write_text(source + "\n" + planted + "\n", encoding="utf-8")
    assert i18n_check.main([]) != 0
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0
