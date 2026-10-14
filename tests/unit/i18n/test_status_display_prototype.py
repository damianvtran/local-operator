"""Source-only status boundary proof, not terminal bidi/shaping validation."""

from __future__ import annotations

import json
from datetime import date

import pytest

from local_operator.i18n import catalogues, messages, resolve, runtime
from local_operator.i18n.display import TechnicalSpan
from local_operator.tui.i18n_status import status_display

RLI, LRI, PDI = "\u2067", "\u2066", "\u2069"


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "catalogues"
    en = {
        "tui.status.demo": "Run {command} for {name}",
        "tui.status.fallback": "{count, plural, one {# job} other {# jobs}}",
        "tui.status.typed": "{count, number} {when, date, short}",
        "tui.status.passthrough": "{error}",
    }
    ur = {
        "tui.status.demo": "کام {command} کے لیے {name}",
        "tui.status.typed": "تعداد {count, number} تاریخ {when, date, short}",
        "tui.status.passthrough": "{error}",
    }
    for locale, entries in [("en", en), ("ur", ur)]:
        (root / locale).mkdir(parents=True)
        (root / locale / "tui.status.json").write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ur"))
    return ur


def token(text, value):
    start = text.index(value)
    return TechnicalSpan(start, start + len(value), value)


def test_actual_source_fallback_unknown_and_raw_paths(corpus):
    message = messages.Msg("tui.status.demo", {"command": "--force", "name": "احمد"})
    record = messages.render_record(message.code, message.params, locale="ur")
    assert record.source_locale == "ur"
    assert record.text == runtime.render_message(corpus[message.code], message.params, "ur")
    assert messages.render(message.code, message.params, locale="ur") == record.text
    assert message.envelope(locale="ur")["text"] == record.text
    assert not any(c in record.text for c in (RLI, LRI, PDI))
    painted = status_display(message, locale="ur", technical=(token(record.text, "--force"),))
    assert painted.plain == RLI + record.text.replace("--force", LRI + "--force" + PDI) + PDI
    fallback = messages.Msg("tui.status.fallback", {"count": 21})
    selected = messages.render_record(fallback.code, fallback.params, locale="ur")
    assert selected.source_locale == "en"
    assert selected.text == "21 jobs"
    assert status_display(fallback, locale="ur").plain == selected.text
    unknown = messages.Msg("tui.status.unknown")
    assert messages.render_record(unknown.code, {}, locale="ur").source_locale is None
    assert status_display(unknown, locale="ur").plain == unknown.code


def test_typed_icu_is_formatted_before_display(corpus):
    params = {"count": 1234.5, "when": date(2025, 7, 19)}
    record = messages.render_record("tui.status.typed", params, locale="ur")
    assert record.text == runtime.render_message(corpus["tui.status.typed"], params, "ur")
    assert "1,234.5" in record.text
    assert status_display(messages.Msg("tui.status.typed", params), locale="ur").plain == (
        RLI + record.text + PDI
    )
    assert isinstance(params["when"], date)
    assert params["count"] == 1234.5


@pytest.mark.parametrize("command", ["--force", "/session", "'id-7'", '"id-7"', "{names}"])
def test_complete_technical_tokens_and_multiline(corpus, command):
    message = messages.Msg("tui.status.demo", {"command": command, "name": "احمد\nعلی"})
    raw = messages.render(message.code, message.params, locale="ur")
    painted = status_display(message, locale="ur", technical=(token(raw, command),)).plain
    lines = raw.replace(command, LRI + command + PDI).split("\n")
    assert painted == "\n".join(RLI + line + PDI for line in lines)
    for line in painted.split("\n"):
        assert line.count(RLI) + line.count(LRI) == line.count(PDI)
    assert painted.replace(RLI, "").replace(LRI, "").replace(PDI, "") == raw


def test_passthrough_prose_has_no_catalogue_direction(corpus):
    record = messages.render_record(
        "tui.status.passthrough", {"error": "External error"}, locale="ur"
    )
    assert record.source_locale is None
    assert (
        status_display(
            messages.Msg("tui.status.passthrough", {"error": "External error"}), locale="ur"
        ).plain
        == "External error"
    )


def test_hint_validation_and_control_rejection(corpus):
    message = messages.Msg("tui.status.demo", {"command": "--force", "name": "احمد"})
    raw = messages.render(message.code, message.params, locale="ur")
    whole = token(raw, "--force")
    invalid = [
        TechnicalSpan(whole.start + 2, whole.end, "force"),
        TechnicalSpan(whole.start, whole.end, "--wrong"),
        TechnicalSpan(-1, 2, "abc"),
        TechnicalSpan(True, 2, "ab"),
    ]
    for hint in invalid:
        with pytest.raises(ValueError):
            status_display(message, locale="ur", technical=(hint,))
    with pytest.raises(ValueError):
        status_display(message, locale="ur", technical=(whole, whole))
    with pytest.raises(ValueError):
        status_display(
            messages.Msg(message.code, {"command": "--force" + PDI, "name": "احمد"}), locale="ur"
        )
    # The new projection cannot be smuggled into arbitrary wire/receipt surfaces.
    with pytest.raises(ValueError):
        status_display(messages.Msg("wire.exec.receipt_suffix"), locale="ur")


def test_unshipped_urdu_stays_english(corpus, monkeypatch):
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en",))
    message = messages.Msg("tui.status.demo", {"command": "--force", "name": "Ahmed"})
    record = messages.render_record(message.code, message.params, locale="ur")
    assert record.source_locale == "en"
    assert record.text == "Run --force for Ahmed"
    assert status_display(message, locale="ur").plain == record.text


@pytest.mark.parametrize(
    "source, expected",
    [("{error}", True), ("'{error}'", False), ("Error: {error}", False), ("{n, number}", False)],
)
def test_passthrough_provenance_uses_existing_icu_parser(source, expected):
    assert runtime.is_passthrough_message(source) is expected


@pytest.mark.parametrize("value", ["--force.extra", "'id-7", "{names", "--force\n/session"])
def test_partial_or_unbalanced_technical_hints_rejected(corpus, value):
    message = messages.Msg("tui.status.demo", {"command": value, "name": "احمد"})
    raw = messages.render(message.code, message.params, locale="ur")
    hint = token(raw, "--force" if value == "--force.extra" else value)
    with pytest.raises(ValueError):
        status_display(message, locale="ur", technical=(hint,))


@pytest.mark.parametrize(
    "whole, partial",
    [("--force", "force"), ("/session", "session"), ("'id-7'", "id-7"), ("{names}", "names")],
)
def test_partial_islands_cannot_drop_token_prefix_or_quotes(corpus, whole, partial):
    message = messages.Msg("tui.status.demo", {"command": whole, "name": "احمد"})
    raw = messages.render(message.code, message.params, locale="ur")
    with pytest.raises(ValueError):
        status_display(message, locale="ur", technical=(token(raw, partial),))


def test_controls_never_enter_raw_paths_and_cannot_be_hint_insertions(corpus):
    message = messages.Msg("tui.status.demo", {"command": "--force", "name": "احمد"})
    raw = messages.render(message.code, message.params, locale="ur")
    whole = token(raw, "--force")
    injected = TechnicalSpan(whole.start, whole.end, LRI + whole.expected + PDI)
    with pytest.raises(ValueError):
        status_display(message, locale="ur", technical=(injected,))
    external = messages.Msg(message.code, {"command": "--force" + PDI, "name": "احمد"})
    external_raw = messages.render(external.code, external.params, locale="ur")
    assert PDI in external_raw
    assert external.envelope(locale="ur")["text"] == external_raw
    with pytest.raises(ValueError):
        status_display(external, locale="ur")
    assert messages.render_record(message.code, {}, locale="ur").source_locale is None
