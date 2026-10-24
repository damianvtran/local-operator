from types import SimpleNamespace

import pytest

from local_operator.stt.radient_cleanup import (
    normalize_radient_cleanup,
    radient_cleanup_options,
    stt_cleanup_prompt,
)


def _result(**overrides):
    values = {
        "text": "Um, えーと、hello",
        "raw_text": None,
        "raw_unavailable": "",
        "cleanup": "none",
        "language": "en",
        "language_code": "eng",
        "language_probability": 0.99,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("mode", "remove", "events"),
    [
        ("auto", True, False),
        ("provider", True, False),
        ("client", False, False),
        ("off", None, None),
    ],
)
def test_radient_cleanup_options(mode, remove, events):
    options = radient_cleanup_options(mode, None, "en-US")
    assert options["remove_disfluencies"] is remove
    assert options["include_audio_events"] is events
    if mode in ("auto", "provider"):
        assert "omit filler" in options["prompt"].lower()
        assert len(options["prompt"]) <= 1000
    else:
        assert options["prompt"] is None


def test_prompt_hardening_never_overrides_or_truncates_caller_prompt():
    prompt = "Caller context " + "x" * 1100
    assert stt_cleanup_prompt("auto", prompt, "en") == prompt
    assert stt_cleanup_prompt("client", None, "en") is None
    assert stt_cleanup_prompt("off", None, "en") is None
    japanese_prompt = stt_cleanup_prompt("auto", None, "ja")
    generic_prompt = stt_cleanup_prompt("provider", None, None)
    assert japanese_prompt is not None
    assert generic_prompt is not None
    assert "Japanese" in japanese_prompt
    assert "audio's language" in generic_prompt


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en", ("English", "um", "uh", "er", "erm", "hmm")),
        ("de", ("German", "äh", "ähm", "disgust")),
        ("fr", ("French", "euh", "ben", "assent")),
        ("it", ("Italian", "ehm")),
        ("es", ("Spanish", "eh", "este", "demonstrative")),
        ("pt", ("Portuguese", "Um", "hã", "article")),
        ("ja", ("Japanese", "えーと", "あの", "demonstrative")),
        ("zh", ("Chinese", "呃", "嗯", "assent")),
        ("hi", ("Hindi", "उम्म", "noun")),
        ("ru", ("Russian", "э", "ну", "urging")),
        ("he", ("Hebrew", "אממ")),
    ],
)
def test_prompt_has_bounded_language_specific_hesitation_guidance(language, expected):
    prompt = stt_cleanup_prompt("auto", None, language)

    assert prompt is not None
    assert len(prompt) <= 1000
    for text in expected:
        assert text in prompt


def test_auto_layers_deterministic_rules_after_acknowledged_native_cleanup():
    normalized = normalize_radient_cleanup(
        _result(
            text="えーと、hello",
            cleanup="elevenlabs.no_verbatim",
            raw_unavailable="provider_no_verbatim",
            language="ja",
            language_code="jpn",
        ),
        "auto",
    )
    assert normalized.text == "Hello"
    assert normalized.cleanup == "elevenlabs.no_verbatim.client.rules.v2"
    assert normalized.raw_text is None
    assert normalized.raw_unavailable == "provider_no_verbatim"
    assert normalized.removed[0].token == "えーと"


def test_provider_requires_acknowledgement_and_skips_deterministic_rules():
    unacknowledged = normalize_radient_cleanup(_result(text="Um, hello"), "provider")
    assert unacknowledged.text == "Um, hello"
    assert unacknowledged.cleanup == "none"
    assert unacknowledged.raw_text == "Um, hello"

    acknowledged = normalize_radient_cleanup(
        _result(
            text="えーと、hello",
            cleanup="elevenlabs.no_verbatim",
            raw_unavailable="provider_no_verbatim",
            language="ja",
        ),
        "provider",
    )
    assert acknowledged.text == "えーと、hello"
    assert acknowledged.cleanup == "elevenlabs.no_verbatim"
    assert acknowledged.raw_text is None


def test_legacy_missing_raw_metadata_means_raw_available():
    normalized = normalize_radient_cleanup(_result(text="Um, hello"), "client")
    assert normalized.text == "Hello"
    assert normalized.raw_text == "Um, hello"
    assert normalized.raw_unavailable == ""
    assert normalized.cleanup == "client.rules.v2"
    assert (normalized.removed[0].start, normalized.removed[0].end) == (0, 2)


def test_off_preserves_provider_text_and_raw_metadata():
    normalized = normalize_radient_cleanup(
        _result(
            text="Um, hello",
            cleanup="elevenlabs.no_verbatim",
            raw_unavailable="provider_no_verbatim",
        ),
        "off",
    )
    assert normalized.text == "Um, hello"
    assert normalized.raw_text is None
    assert normalized.raw_unavailable == "provider_no_verbatim"
    assert normalized.cleanup == "none"
