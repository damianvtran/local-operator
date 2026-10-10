"""The message envelope: additive `{code, params, text}` with a safe fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, messages


@pytest.fixture()
def fixture_catalogues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "wire.errors.json").write_text(
        json.dumps(
            {
                "wire.errors.model_unavailable": "Model {model} is unavailable.",
                "wire.errors.files": "{n, plural, one {# file} other {# files}}",
                "wire.errors.count": "{n, number} items",
                "wire.errors.when": "on {d, date}",
                "wire.errors.broken": "{unclosed",
            }
        ),
        encoding="utf-8",
    )
    (root / "fr").mkdir()
    (root / "fr" / "wire.errors.json").write_text(
        json.dumps({"wire.errors.model_unavailable": "Le modèle {model} est indisponible."}),
        encoding="utf-8",
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)


def test_envelope_renders_text_from_the_catalogue(fixture_catalogues: None) -> None:
    got = messages.envelope("wire.errors.model_unavailable", {"model": "gpt-x"})
    assert got == {
        "code": "wire.errors.model_unavailable",
        "params": {"model": "gpt-x"},
        "text": "Model gpt-x is unavailable.",
    }


def test_envelope_uses_the_requested_locale_then_falls_back_to_en(
    fixture_catalogues: None,
) -> None:
    got = messages.envelope("wire.errors.model_unavailable", {"model": "m"}, locale="fr")
    assert got["text"] == "Le modèle m est indisponible."
    # A locale with no file at all falls back to en, not to an error.
    got = messages.envelope("wire.errors.model_unavailable", {"model": "m"}, locale="es")
    assert got["text"] == "Model m is unavailable."


def test_unknown_code_degrades_to_the_code_itself(fixture_catalogues: None) -> None:
    got = messages.envelope("wire.errors.not_extracted_yet", {"n": 1})
    assert got["code"] == "wire.errors.not_extracted_yet"
    assert got["params"] == {"n": 1}
    assert got["text"] == "wire.errors.not_extracted_yet"


def test_dotless_code_degrades_without_touching_the_disk(fixture_catalogues: None) -> None:
    got = messages.envelope("plain")
    assert got["text"] == "plain"


def test_malformed_message_degrades_instead_of_raising(fixture_catalogues: None) -> None:
    # The additive contract wins at run time: a catalogue that does not render
    # still yields an envelope (the parity check is where this goes red).
    got = messages.envelope("wire.errors.broken", {})
    assert got["text"] == "wire.errors.broken"


def test_non_numeric_plural_binding_degrades_instead_of_raising(
    fixture_catalogues: None,
) -> None:
    # round-1 M1: the envelope's never-raise contract must hold for the first
    # non-numeric binding a wire slice passes, not just for missing codes.
    for bad in ("abc", True):
        got = messages.envelope("wire.errors.files", {"n": bad})
        assert got["code"] == "wire.errors.files"
        assert got["params"] == {"n": bad}
        assert got["text"] == "wire.errors.files"
    assert messages.envelope("wire.errors.files", {"n": 3})["text"] == "3 files"


def test_non_numeric_number_and_date_bindings_degrade(fixture_catalogues: None) -> None:
    # round-2 M2 + Q2-1: the never-raise contract must hold for the number and
    # date branches too, not only plurals.
    for code, params in (
        ("wire.errors.count", {"n": "abc"}),
        ("wire.errors.count", {"n": True}),
        ("wire.errors.when", {"d": "abc"}),
        ("wire.errors.when", {"d": True}),
    ):
        got = messages.envelope(code, params)
        assert got["text"] == code
        assert got["params"] == params
    assert messages.envelope("wire.errors.count", {"n": 1234})["text"] == "1,234 items"


def test_msg_carries_code_and_params(fixture_catalogues: None) -> None:
    message = messages.Msg("wire.errors.model_unavailable", {"model": "x"})
    assert message.envelope() == {
        "code": "wire.errors.model_unavailable",
        "params": {"model": "x"},
        "text": "Model x is unavailable.",
    }
    assert messages.msg("wire.errors.model_unavailable", model="x") == message
