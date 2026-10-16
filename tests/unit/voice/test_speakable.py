"""What is read aloud (design V8): agent text, cleaned and capped."""

from __future__ import annotations

import pytest

from local_operator.voice import speakable as sp


def test_code_urls_and_markdown_are_not_read_aloud() -> None:
    text = (
        "# Result\n"
        "**Done.** See https://example.com/a?b=c and [the docs](https://x.y/z).\n"
        "```python\nprint('secret')\n```\n"
        "- first item\n- `second` item\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    spoken = sp.speakable(text).text
    assert "print" not in spoken
    assert "http" not in spoken
    assert "**" not in spoken and "`" not in spoken and "#" not in spoken and "|" not in spoken
    assert sp.CODE_PLACEHOLDER in spoken
    assert "the docs" in spoken and "link" in spoken
    assert spoken.startswith("Result Done.")


def test_an_unterminated_fence_still_hides_the_code() -> None:
    assert "rm -rf" not in sp.speakable("Run this:\n```sh\nrm -rf build").text


def test_short_text_is_not_truncated() -> None:
    result = sp.speakable("Two files changed.", max_chars=800)
    assert result == sp.Speakable("Two files changed.", truncated=False)


def test_the_cap_cuts_on_a_sentence_and_never_exceeds_the_limit() -> None:
    text = " ".join(f"Sentence number {index} is here." for index in range(100))
    result = sp.speakable(text, max_chars=200)
    assert result.truncated
    assert len(result.text) <= 200
    assert result.text.endswith(sp.TRUNCATION_TAIL)
    body = result.text[: -len(sp.TRUNCATION_TAIL)].rstrip()
    assert body.endswith("is here.")


def test_the_cap_falls_back_to_a_word_boundary() -> None:
    text = "word " * 400
    result = sp.speakable(text, max_chars=150)
    assert result.truncated and len(result.text) <= 150
    assert "wor " not in result.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, sp.DEFAULT_MAX_SPOKEN_CHARS),
        ("garbage", sp.DEFAULT_MAX_SPOKEN_CHARS),
        (True, sp.DEFAULT_MAX_SPOKEN_CHARS),
        (5, sp.MIN_MAX_SPOKEN_CHARS),
        (10**9, sp.MAX_MAX_SPOKEN_CHARS),
        ("300", 300),
    ],
)
def test_a_hand_edited_cap_never_breaks_voice(value: object, expected: int) -> None:
    assert sp.normalise_max_chars(value) == expected
