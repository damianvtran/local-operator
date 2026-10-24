"""Production TUI bidi paint tests; these are not terminal shaping evidence."""

from __future__ import annotations

import json

import pytest

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
            "ur": {"tui.welcome.status.update": "ورژن {version} پر اپ ڈیٹ کریں"},
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
                "tui.transcript.queued.newest": "قطار میں — یاد کرنے کے لیے اوپر دبائیں",
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
    set_active_locale("ur")
    yield
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

    assert transcript.QUEUED_ROW_TEXT.startswith(RLI)
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


def test_urdu_language_choice_documents_degraded_terminal(monkeypatch):
    from local_operator import settings_io

    monkeypatch.setattr(resolve, "language_options", lambda: (("ur", "ur"),))
    choice = settings_io._language_choices()[0]
    assert choice.value == "ur"
    assert "best effort" in choice.description
    assert "bidi" in choice.description
    assert "shaping" in choice.description
