"""Production TUI bidi paint tests; these are not terminal shaping evidence."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.i18n import catalogues, messages, resolve, text
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import set_active_locale
from local_operator.tui.i18n_status import (
    localized_display,
    message_display,
    technical_display,
)

RLI, LRI, PDI = "\u2067", "\u2066", "\u2069"


@pytest.fixture()
def urdu_corpus(tmp_path, monkeypatch):
    root = tmp_path / "catalogues"
    entries = {
        "tui.status": {
            "en": {"tui.status.demo": "Run {command} for {name}; again {command}"},
            "ur": {"tui.status.demo": "کام {command} کے لیے {name}؛ دوبارہ {command}"},
        },
        "tui.welcome": {
            "en": {"tui.welcome.status.update": "Update to {version}"},
            "ur": {"tui.welcome.status.update": "ورژن {version} پر اپ ڈیٹ کریں — /update"},
        },
        "tui.transcript": {
            "en": {
                "tui.transcript.catchup.missed": "{count, plural, one {# missed} other {# missed}}",
                "tui.transcript.catchup.headline": "Catch up: {label}",
                "tui.transcript.queued.newest": "Queued — press up to recall",
                "tui.transcript.queued.older": "Queued",
            },
            "ur": {
                "tui.transcript.catchup.missed": (
                    "{count, plural, one {# missing} other {# missed}}"
                ),
                "tui.transcript.catchup.headline": "اپ ڈیٹ: {label}",
                "tui.transcript.queued.newest": "قطار میں — واپس لینے کے لیے esc دبائیں",
                "tui.transcript.queued.older": "قطار میں",
            },
        },
    }
    for namespace, locales in entries.items():
        for locale, catalogue in locales.items():
            (root / locale).mkdir(parents=True, exist_ok=True)
            (root / locale / f"{namespace}.json").write_text(
                json.dumps(catalogue), encoding="utf-8"
            )
    # The app imports many refreshers. Restrict this partial fixture to the one
    # module whose import-time rows are under test so unrelated constants never
    # re-materialize as message codes from this deliberately tiny corpus.
    from local_operator.tui.widgets import transcript

    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ur"))
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [transcript._reload_copy])
    messages._namespace_for.cache_clear()
    real_root = catalogues.catalogue_root()
    set_active_locale("ur")
    yield
    # Restore against the REAL root before the locale moves back: teardown can
    # otherwise re-materialise every scoped constant from a deleted tmp tree and
    # leave message CODES for the next test sharing this worker.
    patched = catalogues._CATALOGUES
    catalogues._CATALOGUES = real_root
    try:
        set_active_locale("en")
    finally:
        catalogues._CATALOGUES = patched
    messages._namespace_for.cache_clear()


def test_parameter_spans_cover_repeated_nested_placeholders_and_raw_stays_clean(urdu_corpus):
    message = Msg("tui.status.demo", {"command": "--x={id}", "name": "احمد"})
    raw = messages.render(message.code, message.params, locale="ur")
    painted = message_display(message, technical_params=("command",)).plain

    assert painted == RLI + raw.replace("--x={id}", LRI + "--x={id}" + PDI) + PDI
    assert painted.count(LRI) == 2
    assert messages.render(message.code, message.params, locale="ur") == raw
    assert not any(control in raw for control in (RLI, LRI, PDI))


def test_actual_welcome_and_nested_transcript_paths_do_not_double_isolate(urdu_corpus):
    from local_operator.i18n.keys import tui_transcript, tui_welcome
    from local_operator.tui.widgets import transcript, welcome

    version = welcome._t(tui_welcome.status_update(version="0.88.1"))
    assert LRI + "0.88.1" + PDI in version
    assert LRI + "/update" + PDI in version
    rows = welcome._status_rows(welcome.WelcomeInfo(update_available="0.88.1"), 80)
    update_lines = [line.plain for _, line in rows if "0.88.1" in line.plain]
    assert update_lines
    for line in update_lines:
        assert LRI + "0.88.1" + PDI in line
        assert LRI + "/update" + PDI in line
        _assert_balanced(line)
    hints = welcome._hint_lines(80)
    assert any(LRI + "/help" in line.plain for line in hints)
    status = welcome._status_rows(welcome.WelcomeInfo(cwd="/tmp/project"), 80)
    assert any(LRI + "/tmp/project" + PDI in line.plain for _, line in status)

    label = transcript._raw_t(tui_transcript.catchup_missed(count=2))
    headline = transcript._t(tui_transcript.catchup_headline(label=label))
    assert not any(control in label for control in (RLI, LRI, PDI))
    assert headline.count(RLI) == 1
    assert headline.count(PDI) == 1
    assert LRI not in headline


def test_standalone_technical_lines_and_settings_paint(urdu_corpus):
    for value in ("/session", "/tmp/main.py", "abc123", "diff --git a/a b/b\n--- a/a"):
        painted = technical_display(value).plain
        assert painted.count(LRI) == len(value.split("\n"))
        assert painted.replace(LRI, "").replace(PDI, "") == value

    raw = "ترتیبات"
    assert localized_display(raw).plain == RLI + raw + PDI
    with pytest.raises(ValueError, match="control sequences"):
        localized_display(RLI + raw + PDI)


def test_palette_commands_are_ltr_but_search_text_stays_raw(urdu_corpus):
    from local_operator.tui.command_palette import (
        KIND_COMMAND,
        PaletteEntry,
        _label_display,
    )

    entry = PaletteEntry(kind=KIND_COMMAND, label="/help", slash="/help")
    assert _label_display(entry).plain == LRI + "/help" + PDI
    assert entry.label == "/help"
    assert entry.slash == "/help"


def test_ur_to_en_recovery_removes_import_time_and_new_paint_controls(urdu_corpus):
    from local_operator.tui.widgets import transcript

    assert transcript.QUEUED_ROW_TEXT.startswith("قطار")
    assert not any(control in transcript.QUEUED_ROW_TEXT for control in (RLI, LRI, PDI))
    assert message_display(Msg("tui.status.demo", {"command": "/run", "name": "A"})).plain != (
        "Run /run for A; again /run"
    )
    set_active_locale("en")
    assert transcript.QUEUED_ROW_TEXT == "Queued — press up to recall"
    assert transcript.QUEUED_ROW_TEXT_OLDER == "Queued"
    assert (
        message_display(
            Msg("tui.status.demo", {"command": "/run", "name": "A"}), technical_params=("command",)
        ).plain
        == "Run /run for A; again /run"
    )
    assert technical_display("/run").plain == "/run"
    assert (
        message_display(Msg("tui.status.demo", {"command": "/run", "name": RLI})).plain
        == f"Run /run for {RLI}; again /run"
    )
    assert technical_display(RLI).plain == RLI


@pytest.fixture()
def urdu_app_corpus(tmp_path, monkeypatch):
    """Complete English fallback plus small Urdu overlays for real-app tests."""
    real = catalogues.catalogue_root()
    root = tmp_path / "app-catalogues"
    shutil.copytree(real / "en", root / "en")
    (root / "ur").mkdir()
    overlays = {
        "tui.transcript": {
            "tui.transcript.queued.newest": "قطار میں — واپس لینے کے لیے esc دبائیں",
            "tui.transcript.queued.older": "قطار میں",
            "tui.transcript.catchup.missed": "{count, plural, one {# کم} other {# کم}}",
            "tui.transcript.catchup.headline": "اپ ڈیٹ: {label}",
        },
        "tui.welcome": {
            "tui.welcome.status.update": "ورژن {version} پر اپ ڈیٹ کریں — /update",
        },
        "tui.small": {
            "tui.small.settings_view.bool_on": "فعال",
            "tui.small.settings_view.bool_off": "غیرفعال",
        },
    }
    for namespace, values in overlays.items():
        (root / "ur" / f"{namespace}.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ur"))
    monkeypatch.setenv("LOP_LANG", "en")
    messages._namespace_for.cache_clear()
    text._sources.cache_clear()
    set_active_locale("en")
    yield
    # Same root-first restore as the partial fixture: this tree is a copy, but
    # the tmp path still dies before the autouse locale reset in some orders.
    patched = catalogues._CATALOGUES
    catalogues._CATALOGUES = real
    try:
        set_active_locale("en")
    finally:
        catalogues._CATALOGUES = patched
    messages._namespace_for.cache_clear()
    text._sources.cache_clear()


def _assert_balanced(value: str) -> None:
    for line in value.split("\n"):
        depth = 0
        for char in line:
            if char in (RLI, LRI):
                depth += 1
            elif char == PDI:
                depth -= 1
                assert depth >= 0, repr(line)
        assert depth == 0, repr(line)


async def _boot_app(pilot, app) -> None:
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("OperatorApp did not boot")


def _select_setting(view, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return
    raise AssertionError(f"missing setting {key}")


def test_queued_marker_wraps_raw_then_isolates_each_twenty_cell_line(urdu_corpus):
    from local_operator.tui.widgets.transcript import UserBlock

    rows = UserBlock("", queued=True, fold_width=20)._rows(20)
    painted = "\n".join(rows)
    _assert_balanced(painted)
    assert LRI + "esc" + PDI in painted
    assert len(rows) > 2, "the marker must actually wrap at the degraded width"


@pytest.mark.asyncio
async def test_real_wakeblock_copy_excludes_projection_controls(urdu_app_corpus) -> None:
    from textual.selection import Selection

    from local_operator.tui.widgets.transcript import TranscriptView, WakeBlock
    from tests.unit.tui.conftest import StyledTranscriptApp

    set_active_locale("ur")
    app = StyledTranscriptApp()
    async with app.run_test(size=(42, 18)) as pilot:
        view = app.query_one(TranscriptView)
        block = WakeBlock(
            "(alarm) resumed\n\n- w1 (due 09:00): missed.\n- w2 (due 10:00): missed.",
            catchup=True,
        )
        view.append_block(block)
        await pilot.pause()
        rendered = block.renderable
        assert isinstance(rendered, Text)
        _assert_balanced(rendered.plain)
        copied = block.get_selection(Selection(None, None))
        assert copied is not None
        assert copied[0]
        assert not any(control in copied[0] for control in (RLI, LRI, PDI))


@pytest.mark.asyncio
async def test_real_undeclared_bool_and_registry_truncation_stay_raw(
    urdu_app_corpus, tmp_path: Path, monkeypatch
) -> None:
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets import settings_view as settings_mod
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    config_dir = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await _boot_app(pilot, app)
        app._run_slash_command("/language ur")
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select_setting(view, "memory.enabled")
        await pilot.press("enter")
        await pilot.pause()
        assert view.expanded_key == "memory.enabled"
        assert all(
            not any(c in choice.label for c in (RLI, LRI, PDI))
            for choice in settings_mod._BOOL_CHOICES
        )
        choice_index = next(index for index, row in enumerate(view._rows) if row.kind == "choice")
        _assert_balanced(view._row_text(view._rows[choice_index], choice_index, 90).plain)
        # The REAL raster (post-`_paint_list` truncation), not just row assembly.
        for line in view.render_lines_for_test():
            _assert_balanced(line)

        # Registry fallback remains English even under active Urdu. Fit a long
        # source label through the real row painter and verify no sliced controls.
        setting_index, setting_row = next(
            (index, row)
            for index, row in enumerate(view._rows)
            if row.kind == "setting" and row.setting is not None
        )
        original = setting_row.setting
        assert original is not None
        setting_row.setting = replace(original, label="English registry label " * 8)
        try:
            painted = view._row_text(setting_row, setting_index, 90).plain
        finally:
            setting_row.setting = original
        assert "…" in painted
        assert not any(control in painted for control in (RLI, LRI, PDI))


@pytest.mark.parametrize("width", [4, 9, 17, 31])
def test_final_line_fit_rebalances_isolates_cut_by_ellipsis(width: int) -> None:
    from rich.text import Text

    from local_operator.tui.i18n_status import balance_display_line

    line = Text(RLI + "قطار میں — واپس لینے کے لیے esc دبائیں" + PDI)
    line.truncate(width, overflow="ellipsis")
    balance_display_line(line)
    _assert_balanced(line.plain)
    assert cell_len(line.plain) <= width, line.plain


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [100, 36])
async def test_real_language_picker_remounts_queued_copy_without_resize(
    urdu_app_corpus, tmp_path: Path, monkeypatch, width: int
) -> None:
    from local_operator.config import ConfigManager
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets import transcript
    from local_operator.tui.widgets.settings_view import SettingsView
    from local_operator.tui.widgets.transcript import UserBlock
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    config_dir = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, 30)) as pilot:
        await _boot_app(pilot, app)
        app._run_slash_command("/language ur")
        await pilot.pause()
        block = UserBlock("queued prompt", queued=True)
        app._append_block(block)
        await pilot.pause()
        before_width = block._built_width
        rendered = block.renderable
        assert isinstance(rendered, Text)
        assert "قطار" in rendered.plain
        _assert_balanced(rendered.plain)

        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select_setting(view, "language")
        await pilot.press("enter")
        await pilot.pause()
        assert view.expanded_key == "language"
        # Fixture options are auto, en, ur: the current Urdu row moves up to en.
        await pilot.press("up")
        await pilot.press("enter")
        await pilot.pause()

        assert text.active_locale() == "en"
        assert ConfigManager(config_dir).get_config().values["language"] == "en"
        assert transcript.QUEUED_ROW_TEXT.startswith("· queued")
        assert block._built_width == before_width, "the test itself must not resize"
        rendered = block.renderable
        assert isinstance(rendered, Text)
        assert "· queued for the next runtime" in rendered.plain
        assert not any(control in rendered.plain for control in (RLI, LRI, PDI))


def test_external_tool_and_live_wake_summaries_pass_through_unprojected(urdu_corpus):
    """No provenance means not catalogue copy: model/tool prose stays raw."""
    from local_operator.tui.widgets.tool_card import ToolCard
    from local_operator.tui.widgets.transcript import WakeBlock

    live = (
        "(alarm) Scheduled wake w3 (3/8, every 1h) — "
        'cancel with wake({op:"cancel",id:"w3"}) once its goal is met.\n\ncheck it'
    )
    for block in (WakeBlock(live), ToolCard("t1", "read", {"path": "a.py"})):
        painted = block._build_row(80).plain
        assert painted
        assert not any(control in painted for control in (RLI, LRI, PDI)), painted


def test_urdu_language_choice_documents_degraded_terminal(monkeypatch):
    from local_operator import settings_io

    monkeypatch.setattr(resolve, "language_options", lambda: (("ur", "ur"),))
    choice = settings_io._language_choices()[0]
    assert choice.value == "ur"
    assert "best effort" in choice.description
    assert "bidi" in choice.description
    assert "shaping" in choice.description
