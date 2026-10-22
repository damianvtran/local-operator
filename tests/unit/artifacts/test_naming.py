"""The ONE naming rule for generated media: the request names the asset.

The negative controls matter most here: ``suggested`` is model-supplied text
that an agent can be steered into producing, and the name is joined onto a
real directory at the destination, so a suggestion is a hostile input.
"""

from __future__ import annotations

import pytest

from local_operator.artifacts.naming import media_filename


def test_prompt_names_the_artifact_without_a_leading_article() -> None:
    name = media_filename("A red fox in snow, painterly", "image/png", kind="image")
    assert name == "red-fox-in-snow-painterly.png"


def test_agent_supplied_stem_wins_and_loses_its_extension() -> None:
    name = media_filename(
        "anything at all",
        "audio/mpeg",
        suggested="Bird Chirping Sound Effect.mp3",
        kind="audio",
    )
    assert name == "bird-chirping-sound-effect.mp3"


@pytest.mark.parametrize(
    ("hostile", "expected"),
    [
        ("../../etc/passwd.png", "passwd.png"),
        ("..\\..\\evil.png", "evil.png"),
        ("/tmp/../x/y/z.webp", "z.png"),
        ("na\x00me.png", "na-me.png"),
        ("...", "quiet-hum.png"),
        ("///", "quiet-hum.png"),
    ],
)
def test_traversal_shaped_suggestions_become_a_label(hostile: str, expected: str) -> None:
    name = media_filename("a quiet hum", "image/png", suggested=hostile, kind="image")
    assert name == expected
    assert "/" not in name and "\\" not in name and "\x00" not in name


def test_a_dotdot_suggestion_falls_through_to_the_prompt() -> None:
    assert media_filename("swirling smoke", "image/png", suggested="..", kind="image") == (
        "swirling-smoke.png"
    )
    assert media_filename("swirling smoke", "image/png", suggested=".", kind="image") == (
        "swirling-smoke.png"
    )
    assert media_filename(None, "image/png", suggested="..", kind="image") == "image.png"


def test_the_extension_follows_the_bytes_not_the_suggestion() -> None:
    assert media_filename("x", "image/png", suggested="clip.mov", kind="image") == "clip.png"
    assert media_filename("x", "video/quicktime", suggested="clip.png", kind="video") == "clip.mov"


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("image/jpeg", ".jpg"),
        ("image/webp", ".webp"),
        ("audio/mpeg", ".mp3"),
        ("audio/mp3", ".mp3"),
        ("audio/wav", ".wav"),
        ("audio/x-wav", ".wav"),
        ("video/webm", ".webm"),
        ("video/quicktime", ".mov"),
        ("video/mp4; codecs=avc1", ".mp4"),
    ],
)
def test_the_preferred_extension_map(content_type: str, expected: str) -> None:
    assert media_filename("x", content_type, kind="media") == f"x{expected}"


def test_an_unknown_content_type_falls_back_to_the_kind() -> None:
    assert media_filename("x", "image/x-unknown", kind="image") == "x.png"
    assert media_filename("x", "audio/x-unknown", kind="audio") == "x.mp3"
    assert media_filename("x", None, kind="video") == "x.mp4"
    assert media_filename("x", None, kind="media") == "x.bin"
    # ``mimetypes`` itself knows ``application/octet-stream`` -> ``.bin``, which
    # is a truer suffix than any kind convention for bytes nobody labelled.
    assert media_filename("x", "application/octet-stream", kind="image") == "x.bin"


def test_the_stem_is_capped_at_a_word_boundary() -> None:
    prompt = "a photorealistic red fox running through deep powdery snow at golden hour in the alps"
    name = media_filename(prompt, "image/png", kind="image")
    assert name == "photorealistic-red-fox-running-through-deep-powdery-snow-at.png"
    assert len(name.removesuffix(".png")) <= 60


def test_a_single_overlong_word_is_hard_cut() -> None:
    assert media_filename("x" * 100, "image/png", kind="image") == f"{'x' * 60}.png"


@pytest.mark.parametrize("prompt", ["", "   ", "!!!", "..."])
def test_a_useless_prompt_falls_back_to_the_kind(prompt: str) -> None:
    assert media_filename(prompt, "audio/wav", kind="audio") == "audio.wav"
    assert media_filename(prompt, "audio/wav") == "media.wav"


def test_a_counter_numbers_a_multi_asset_call_only() -> None:
    assert media_filename("x", "image/webp", kind="image") == "x.webp"
    assert media_filename("x", "image/webp", kind="image", index=1) == "x.webp"
    assert media_filename("x", "image/webp", kind="image", index=2) == "x-2.webp"
    assert media_filename("x", "image/webp", kind="image", index=3) == "x-3.webp"


def test_unicode_words_survive() -> None:
    assert media_filename("Café über 日本", "image/png", kind="image") == "café-über-日本.png"


def test_an_article_is_only_dropped_at_a_word_boundary() -> None:
    # "Anvil"/"Theory" must keep their first letters: the article needs a
    # separator after it to be an article.
    assert media_filename("Anvil in a meadow", "image/png", kind="image") == "anvil-in-a-meadow.png"
    assert media_filename("Theory of light", "image/png", kind="image") == "theory-of-light.png"
