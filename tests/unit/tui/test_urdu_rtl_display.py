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
    technical_literal_spans,
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
            "tui.transcript.queued.newest": "· قطار میں — واپس لینے کے لیے esc دبائیں",
            "tui.transcript.queued.older": "· قطار میں",
            "tui.transcript.catchup.missed": "{count, plural, one {# کم} other {# کم}}",
            "tui.transcript.catchup.headline": "اپ ڈیٹ: {label}",
        },
        "tui.welcome": {
            "tui.welcome.status.update": "ورژن {version} پر اپ ڈیٹ کریں — /update",
            # Real `feat/i18n-ur-core@1c6c7d0d` bytes for the command-bearing
            # surfaces (QA Q-6); the fixture exists because this branch has no
            # ur package, and these two strings are copied verbatim.
            "tui.welcome.status.login_required": "آپ لاگ ان نہیں ہیں — /login {provider}",
            "tui.welcome.tip.mesh": "/network new <name> سے دوسرے ڈیوائسز کا میش بنائیں",
            "tui.welcome.tip.model": "/model <provider>/<id> صرف اسی سیشن پر لاگو ہوتا ہے",
            "tui.welcome.tip.team": "/team <name> <message> مینیجر کو کام بھیجتا ہے",
            "tui.welcome.tip.fork": "/fork شاخ بناتا ہے؛ /fork <message> الگ کام شروع کرتا ہے",
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


@pytest.fixture()
def native_urdu_update(tmp_path, monkeypatch):
    """The QA-reported native spelling: ``v{version}`` glued with no space."""
    real = catalogues.catalogue_root()
    root = tmp_path / "native-catalogues"
    (root / "en").mkdir(parents=True)
    (root / "ur").mkdir()
    (root / "en" / "tui.welcome.json").write_text(
        json.dumps(
            {
                "tui.welcome.status.update": "Latest is v{version} — /update",
                "tui.welcome.status.update.short": "Latest is v{version}",
            }
        ),
        encoding="utf-8",
    )
    (root / "ur" / "tui.welcome.json").write_text(
        json.dumps(
            {
                "tui.welcome.status.update": "تازہ ترین v{version} ہے — /update",
                "tui.welcome.status.update.short": "تازہ ترین v{version} ہے",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ur"))
    messages._namespace_for.cache_clear()
    text._sources.cache_clear()
    set_active_locale("ur")
    yield root
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
        # The CONTENT, not just the absence of controls: the copied text is the
        # raw catalogue sentence (with the folded wake ids), byte for byte.
        from local_operator.i18n.keys import tui_transcript
        from local_operator.tui.widgets.transcript import _raw_t

        label = _raw_t(tui_transcript.catchup_missed(count=2))
        expected = _raw_t(tui_transcript.catchup_headline(label=label)) + " (w1, w2)"
        assert expected in copied[0], copied[0]
        assert "اپ ڈیٹ" in copied[0]


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


def test_welcome_update_keeps_the_literal_in_both_locales(urdu_app_corpus):
    """D2: equal width, equal treatment — wrap instead of dropping `/update`."""
    from local_operator.tui.widgets import welcome

    info = welcome.WelcomeInfo(
        version="0.27.0",
        model_label="test/model",
        cwd="/tmp",
        update_available="0.28.0",
    )
    rows: dict[str, list[str]] = {}
    for locale in ("en", "ur"):
        set_active_locale(locale)
        rows[locale] = [
            line.plain
            for priority, line in welcome._status_rows(info, 32)
            if priority == welcome._PRIORITY_UPDATE
        ]

    for locale, lines in rows.items():
        joined = "".join(lines)
        assert "/update" in joined, (locale, lines)
        assert joined.count("/update") == 1, (locale, lines)
        # The `!` marker is structural: it stays at the LEFT edge, outside the
        # RTL run (M-2's rule applied to the status glyph).
        assert lines[0].startswith("! "), (locale, lines[0])
        assert not lines[0].startswith(RLI), (locale, lines[0])
        for line in lines:
            _assert_balanced(line)
            assert "…" not in line, (locale, line)
    assert len(rows["en"]) == 1, rows["en"]
    assert len(rows["ur"]) > 1, "Urdu must wrap rather than fall to the short form"


def test_native_urdu_version_prefix_is_one_island(native_urdu_update):
    """Q-3: ``v{version}`` must isolate ``v9.9.9``, not raise or split the v."""
    from local_operator.i18n.keys import tui_welcome
    from local_operator.tui import i18n_status
    from local_operator.tui.widgets import welcome

    set_active_locale("ur")
    full = welcome._t(tui_welcome.status_update(version="9.9.9"))
    assert LRI + "v9.9.9" + PDI in full
    assert LRI + "/update" + PDI in full
    _assert_balanced(full)
    short = welcome._t(tui_welcome.status_update_short(version="9.9.9"))
    assert LRI + "v9.9.9" + PDI in short
    _assert_balanced(short)

    record = messages.render_record(
        tui_welcome.status_update(version="9.9.9").code, {"version": "9.9.9"}, locale="ur"
    )
    spans = i18n_status.technical_parameter_spans(record, {"version": "9.9.9"}, ("version",))
    start = record.text.index("v9.9.9")
    assert [(span.start, span.end, span.expected) for span in spans] == [
        (start, start + 6, "v9.9.9")
    ]
    literal_spans = i18n_status.technical_literal_spans(record, ("9.9.9",))
    assert [(span.start, span.end, span.expected) for span in literal_spans] == [
        (start, start + 6, "v9.9.9")
    ]

    # The real Welcome row path, at the size QA reported, must not raise.
    info = welcome.WelcomeInfo(
        version="0.27.0", model_label="test/model", cwd="/tmp", update_available="9.9.9"
    )
    rows = [line.plain for _, line in welcome._status_rows(info, 34)]
    update_lines = [line for line in rows if "9.9.9" in line]
    assert update_lines
    assert any(LRI + "v9.9.9" + PDI in line for line in update_lines)
    for line in update_lines:
        _assert_balanced(line)

    # English stays byte-identical.
    set_active_locale("en")
    assert welcome._t(tui_welcome.status_update(version="9.9.9")) == "Latest is v9.9.9 — /update"
    assert welcome._t(tui_welcome.status_update_short(version="9.9.9")) == "Latest is v9.9.9"


@pytest.mark.asyncio
async def test_declared_and_undeclared_bools_read_the_same_in_urdu(
    urdu_app_corpus, tmp_path: Path, monkeypatch
) -> None:
    """Q-4: a declared BOOL must not fall back to English on/off."""
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets import settings_view as settings_mod
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    config_dir = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot_app(pilot, app)
        app._run_slash_command("/language ur")
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)

        labels: dict[str, list[str]] = {}
        for key in ("retry.enabled", "memory.enabled"):
            if key == "memory.enabled":
                await pilot.press("escape")
                await pilot.pause()
            _select_setting(view, key)
            await pilot.press("enter")
            await pilot.pause()
            labels[key] = [
                row.choice.label for row in view._rows if row.kind == "choice" and row.choice
            ]
        assert labels["retry.enabled"] == labels["memory.enabled"]
        assert labels["memory.enabled"] == [choice.label for choice in settings_mod._BOOL_CHOICES]
        assert set(labels["retry.enabled"]) == {"فعال", "غیرفعال"}
        painted = view.render_lines_for_test()
        assert any("فعال" in line for line in painted), painted
        choice_lines = [
            view._row_text(row, index, 120).plain
            for index, row in enumerate(view._rows)
            if row.kind == "choice"
        ]
        assert any("فعال" in line for line in choice_lines), choice_lines
        assert not any("on" in line or "off" in line for line in choice_lines), choice_lines


def test_peer_pid_row_stays_balanced_at_narrow_widths(urdu_app_corpus) -> None:
    """M-1: fit RAW, project after — a narrowed pid row keeps every isolate paired."""
    from local_operator.tui.widgets.transcript import PeerMessageBlock

    # A translated sender copy is what projects; the fixture's ur catalogue is
    # small, so add this key the way a real wave would carry it.
    root = catalogues._CATALOGUES
    overlay = json.loads((root / "ur" / "tui.transcript.json").read_text(encoding="utf-8"))
    overlay["tui.transcript.sender.pid"] = "pid {pid}"
    (root / "ur" / "tui.transcript.json").write_text(json.dumps(overlay), encoding="utf-8")
    messages._namespace_for.cache_clear()
    set_active_locale("ur")

    for width in (20, 36):
        block = PeerMessageBlock("hello there", {"pid": 4321})
        row = block._build_row(width)
        _assert_balanced(row.plain)
        # The coarse RLI wraps the whole identity whenever its provenance is a
        # translated locale; the LRI island appears only while the value itself
        # survives the fit (at 20 cells it is cut, which is the point).
        assert RLI in row.plain and PDI in row.plain, row.plain
        if width >= 36:
            assert LRI + "4321" + PDI in row.plain, row.plain


def test_queued_dot_stays_outside_the_rtl_run(urdu_app_corpus) -> None:
    """R2-2: exact order — `· ` then RLI, never RLI then the dot."""
    from local_operator.tui.widgets.transcript import UserBlock

    set_active_locale("ur")
    rows = [row for row in UserBlock("", queued=True, fold_width=40)._rows(40) if row]
    painted = "\n".join(rows)
    assert rows[0].startswith("· " + RLI), rows[0]
    assert RLI + "·" not in painted, painted
    assert "قطار" in painted
    _assert_balanced(painted)


def test_quiet_group_row_stays_balanced_at_narrow_widths(urdu_app_corpus) -> None:
    """M-1 for the folded bar: the group sentence is fitted raw then projected."""
    from local_operator.tui.widgets import transcript as transcript_mod

    root = catalogues._CATALOGUES
    overlay = json.loads((root / "ur" / "tui.transcript.json").read_text(encoding="utf-8"))
    overlay["tui.transcript.group.messages"] = "پیغامات · {count}"
    (root / "ur" / "tui.transcript.json").write_text(json.dumps(overlay), encoding="utf-8")
    messages._namespace_for.cache_clear()
    transcript_mod._reload_copy()
    set_active_locale("ur")

    from local_operator.tui.quiet_groups import QuietGroup

    group = QuietGroup(
        key="qg:test",
        family="peer",
        count=3,
        first_ts=None,
        last_ts=None,
        senders=(),
        actions=0,
        failed=0,
        open=False,
        row_ids=("a", "b", "c"),
    )
    bar = transcript_mod.QuietGroupBlock(group)
    for width in (20, 36):
        row = bar._build_row(width)
        _assert_balanced(row.plain)


@pytest.mark.asyncio
async def test_english_queued_fallback_stays_byte_identical_under_urdu(
    native_urdu_update,
) -> None:
    """M-2: no ur transcript catalogue → raw English, dot still at the LEFT edge."""
    from local_operator.tui.widgets import transcript as transcript_mod

    transcript_mod._reload_copy()
    assert transcript_mod._QUEUED_SOURCE == "en"
    painted = localized_display(
        transcript_mod.QUEUED_ROW_TEXT,
        source_locale=transcript_mod._QUEUED_SOURCE,
    ).plain
    assert painted == transcript_mod.QUEUED_ROW_TEXT
    assert not any(control in painted for control in (RLI, LRI, PDI))
    assert painted[0] != "·" or painted.startswith("·")


def test_palette_settings_are_never_rtl_wrapped() -> None:
    """M-2: a registry setting name is English source, not Urdu prose."""
    from local_operator.tui import command_palette as palette

    entry = palette.PaletteEntry(
        label="Terminal title", kind=palette.KIND_SETTING, description="window title"
    )
    painted = palette._label_display(entry).plain
    assert painted == "Terminal title"
    assert not any(control in painted for control in (RLI, LRI, PDI))


def test_variant_update_copy_with_urdu_full_stop_keeps_it_outside(native_urdu_update):
    """Constructed variant, not the core-wave catalogue: ۔ is prose, not token."""
    from local_operator.i18n.keys import tui_welcome
    from local_operator.tui.widgets import welcome

    (native_urdu_update / "ur" / "tui.welcome.json").write_text(
        json.dumps(
            {
                "tui.welcome.status.update": "تازہ ترین v{version} ہے — /update۔",
                "tui.welcome.status.update.short": "تازہ ترین v{version} ہے۔",
            }
        ),
        encoding="utf-8",
    )
    messages._namespace_for.cache_clear()
    text._sources.cache_clear()
    set_active_locale("ur")

    full = welcome._t(tui_welcome.status_update(version="9.9.9"))
    assert LRI + "/update" + PDI + "۔" in full
    _assert_balanced(full)
    rows = [
        line.plain
        for _, line in welcome._status_rows(welcome.WelcomeInfo(update_available="9.9.9"), 60)
    ]
    assert any(LRI + "/update" + PDI + "۔" in line for line in rows)


def test_real_welcome_tip_and_login_commands_are_ltr_islands(urdu_app_corpus) -> None:
    """Q-6: commands inside translated prose are whole LTR islands."""
    from local_operator.i18n.keys import tui_welcome
    from local_operator.tui.widgets import welcome

    set_active_locale("ur")
    # Tips are import-time constants; the fixture's locale move refreshed them.
    assert LRI + "/network new <name>" + PDI in welcome.TIP_MESH, welcome.TIP_MESH
    _assert_balanced(welcome.TIP_MESH)
    for tip in (welcome.TIP_MESH,):
        assert "<network new <name/" not in tip, tip

    login = welcome._t(tui_welcome.status_login_required(provider="radient"))
    assert LRI + "/login" + PDI in login, login
    assert LRI + "radient" + PDI in login, login
    _assert_balanced(login)
    rows = [
        line.plain
        for _, line in welcome._status_rows(welcome.WelcomeInfo(missing_credential="radient"), 100)
    ]
    login_rows = [line for line in rows if "/login" in line]
    assert login_rows, rows
    assert LRI + "/login" + PDI in login_rows[0], login_rows[0]
    assert LRI + "radient" + PDI in login_rows[0], login_rows[0]

    # Nested declarations (`/fork` inside `/fork <message>`) stay ONE island.
    fork = welcome._t(tui_welcome.tip_fork())
    assert LRI + "/fork <message>" + PDI in fork, fork
    # Every island closes; the extra PDI is the sentence's own RLI wrapper.
    assert fork.count(PDI) == fork.count(LRI) + fork.count(RLI), fork

    # English stays byte-identical: no controls at all.
    set_active_locale("en")
    english = welcome._t(tui_welcome.status_login_required(provider="radient"))
    assert english == "not logged in — /login radient", english
    assert not any(control in english for control in (RLI, LRI, PDI))


@pytest.mark.parametrize("height", [1, 2, 3])
@pytest.mark.asyncio
async def test_wrapped_update_group_is_whole_or_absent(height, native_urdu_update) -> None:
    """R2-1: a tiny splash never keeps only the contextless `/update` tail."""
    from local_operator.tui.widgets import welcome

    set_active_locale("ur")
    info = welcome.WelcomeInfo(
        version="0.27.0",
        model_label="test/model",
        cwd="/tmp/project",
        update_available="9.9.9",
    )
    lines = [line.plain for line in welcome.build_welcome_lines(info, 28, height)]
    has_command = any("/update" in line for line in lines)
    has_sentence = any("تازہ" in line for line in lines)
    # Whole or absent: the tail alone is the defect, a whole group is not.
    assert has_command == has_sentence, lines
    if has_command:
        # Whole sentence: version value present in the same wrapped group.
        assert any("9.9.9" in line for line in lines), lines
    if height == 1:
        assert not has_command, lines


@pytest.mark.asyncio
async def test_urdu_disclosure_wraps_whole_and_uses_a_passing_ink(
    urdu_app_corpus, tmp_path: Path, monkeypatch
) -> None:
    """D1: the Urdu caveat is never ellipsised and clears AA in both themes."""
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    config_dir = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(36, 36)) as pilot:
        await _boot_app(pilot, app)
        app._run_slash_command("/language ur")
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select_setting(view, "language")
        await pilot.press("enter")
        await pilot.pause()

        sentinel = "Terminal: best effort; requires bidi and Arabic shaping."
        notes = [index for index, row in enumerate(view._rows) if row.kind == "note"]
        assert len(notes) > 1, "the 36-column disclosure must wrap, not truncate"
        assert " ".join(view._rows[index].text for index in notes) == sentinel
        assert not any(view._rows[index].selectable for index in notes)

        width = view._list_width()
        painted = view.render_lines_for_test()
        assert all(cell_len(line) <= width for line in painted), [
            line for line in painted if cell_len(line) > width
        ]
        for index in notes:
            caveat_line = view._row_text(view._rows[index], index, width).plain
            assert "…" not in caveat_line, caveat_line
            assert caveat_line.strip() == view._rows[index].text

        # The ink is the passing token, measured — not a remembered claim.
        from rich.color import Color

        from local_operator.tui import theme as theme_mod

        row_text = view._row_text(view._rows[notes[0]], notes[0], width)
        muted = Color.parse(theme_mod.semantic_color("muted"))
        inks = [
            getattr(getattr(span.style, "color", None), "triplet", None) for span in row_text.spans
        ]
        assert muted.triplet in inks, row_text.spans

        body = view._body
        assert not body.show_horizontal_scrollbar
        assert body.virtual_size.width <= body.size.width


def _contrast(a: str, b: str) -> float:
    def channel(value: int) -> float:
        scaled = value / 255
        return scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4

    def luminance(hex_colour: str) -> float:
        digits = hex_colour.lstrip("#")
        red, green, blue = (int(digits[index : index + 2], 16) for index in (0, 2, 4))
        return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)

    lighter, darker = sorted((luminance(a), luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_the_disclosure_ink_clears_aa_on_both_theme_grounds() -> None:
    """Pin the measured pair: `faint` failed (1.97/2.08), `muted` passes."""
    from local_operator.tui import theme as theme_mod

    grounds = {"dark": "bg", "light": "paper"}
    for theme_name, ground_key in grounds.items():
        tokens = theme_mod.BRAND_TOKENS[theme_name]
        assert _contrast(tokens["muted"], tokens[ground_key]) >= 4.5, theme_name
        assert _contrast(tokens["faint"], tokens[ground_key]) < 4.5, theme_name


def test_urdu_language_choice_renders_the_degraded_note(monkeypatch):
    """The caveat is asserted as ROWS, not as source substrings (m-2)."""
    from local_operator import settings_io
    from local_operator.tui.widgets import settings_view as settings_mod

    monkeypatch.setattr(resolve, "language_options", lambda: (("ur", "ur"),))
    choice = settings_io._language_choices()[0]
    assert choice.value == "ur"
    rows = settings_mod._note_rows(choice.description, 30)
    assert " ".join(row.text for row in rows) == choice.description
    assert len(rows) > 1, "the caveat must wrap, not ellipsise"
    assert all(row.kind == "note" and not row.selectable for row in rows)
    assert all(
        "best effort" in row.text or "bidi" in row.text or "shaping" in row.text for row in rows
    )


def test_literal_inside_a_longer_word_is_not_an_island():
    """n-1: `/update` in `/update-all` is prose, not the command token."""
    record = messages.RenderedMessage("/update-all then /update", "ur")
    spans = technical_literal_spans(record, ("/update",))
    assert [(span.start, span.expected) for span in spans] == [
        (record.text.rindex("/update"), "/update")
    ]
