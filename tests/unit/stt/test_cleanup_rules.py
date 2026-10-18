"""Client rules v1: safety fixtures, boundaries, cues and idempotency."""

from __future__ import annotations

import pytest

from local_operator.stt.cleanup import (
    CLIENT_RULES_PROVENANCE,
    apply_client_rules,
    choose_cleanup_language,
    normalize_language_hint,
)


def _clean(text: str, language: str | None = "en"):
    return apply_client_rules(text, language=language)


def test_english_fillers_are_removed_with_sentence_case() -> None:
    result = _clean("Uh, so I, uh, think we should ship.")

    assert result.text == "So I think we should ship."
    assert result.raw_text == "Uh, so I, uh, think we should ship."
    assert result.cleanup == CLIENT_RULES_PROVENANCE
    assert [(span.kind, span.token) for span in result.removed] == [
        ("filler", "Uh"),
        ("filler", "uh"),
    ]


def test_french_filler_is_removed() -> None:
    assert _clean("Euh, je ne sais pas.", "fr").text == "Je ne sais pas."


@pytest.mark.parametrize(
    "text",
    [
        # `um` is a real Portuguese numeral/article; it is deliberately absent
        # from the English set even when the take is labelled English.
        "Um, the album is in the sum.",
        "The album is in the sum. Hum a tune.",
        # Hyphenated compounds: `-` is a word-continuation character.
        "uh-huh, that works",
        "um-hum, that works",
        # A combining mark directly after the token is part of a word.
        "uh\u0301",
        # ZWJ/ZWNJ joins stay word-internal.
        "uh\u200d",
        "\u200cuh",
        # All-caps abbreviations.
        "UM and UH are universities.",
        # Quoted/code spans are not transcript noise.
        'He said "uh" and left.',
        "`uh = 1` is code",
        "```\nuh = 1\n```",
        # Arabic diacritics are never split from their letters.
        "أُمّي في البيت",
        "أمي في البيت",
    ],
)
def test_negative_controls_are_byte_identical(text: str) -> None:
    result = _clean(text)

    assert result.text == text
    assert result.raw_text == text
    assert result.removed == ()


def test_rtl_controls_survive_a_standalone_token_removal() -> None:
    text = "\u200fuh\u200f"

    result = _clean(text)

    assert result.text == "\u200f\u200f"
    assert [span.token for span in result.removed] == ["uh"]


def test_cue_lexicon_removes_only_known_cues() -> None:
    assert _clean("I agree. [lip smack] Let's go.", None).text == "I agree. Let's go."
    assert _clean("[laughter] That was funny.", None).text == "That was funny."
    assert _clean("The value is [inaudible].", None).text == "The value is [inaudible]."
    assert _clean("Keep [REDACTED] and [sic].", None).text == "Keep [REDACTED] and [sic]."
    # A cue-shaped label inside code or quotes is content, not an audio event.
    assert _clean("He wrote `[laughter]` in code.", None).text == "He wrote `[laughter]` in code."
    assert _clean('He said "[laughter]" aloud.', None).text == 'He said "[laughter]" aloud.'


def test_unknown_language_removes_cues_but_not_tokens() -> None:
    text = "[lip smack] Uh, so I think."

    result = _clean(text, None)

    assert result.text == "Uh, so I think."
    assert [span.kind for span in result.removed] == ["cue"]


def test_clean_is_idempotent() -> None:
    once = _clean("Uh, so I, uh, think we should ship.")

    twice = _clean(once.text)

    assert twice.text == once.text
    assert twice.removed == ()


def test_language_hints_normalise_to_the_iso_allowlist() -> None:
    assert normalize_language_hint("en-US") == "en"
    assert normalize_language_hint("EN_us") == "en"
    assert normalize_language_hint("eng") == "en"
    assert normalize_language_hint("zh-Hans") == "zh"
    assert normalize_language_hint("por") == "pt"
    # `fil` is a real ISO-639-3 code but not ISO-639-1; Radient rejects it.
    assert normalize_language_hint("fil-PH") is None
    assert normalize_language_hint("xx") is None
    assert normalize_language_hint("") is None
    assert normalize_language_hint(None) is None


def test_a_caller_hint_outranks_detection_and_invalid_hints_do_not_guess() -> None:
    # Valid caller hint wins.
    assert choose_cleanup_language("en-US", "por", 0.99) == "en"
    # An unmapped caller hint is unknown; do not fall back to the provider.
    assert choose_cleanup_language("fil-PH", "eng", 0.99) is None
    # No caller hint: only a confident provider label is used.
    assert choose_cleanup_language(None, "eng", 0.79) is None
    assert choose_cleanup_language(None, "eng", 0.8) == "en"
    assert choose_cleanup_language("", "eng", None) is None


def test_removed_spans_are_in_raw_coordinates() -> None:
    text = "I, uh, think we should ship."

    result = _clean(text)

    assert result.text == "I think we should ship."
    assert len(result.removed) == 1
    span = result.removed[0]
    assert text[span.start : span.end] == span.token == "uh"
