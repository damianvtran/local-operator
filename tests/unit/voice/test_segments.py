"""The segmenter (session-voice-groq D3): a property table over hand-built inputs."""

from __future__ import annotations

import random
import re

import pytest

from local_operator.voice import speakable
from local_operator.voice.segments import (
    VOICE_SEGMENT_MAX_CHARS,
    segment,
    speak_segments,
)

CAP = VOICE_SEGMENT_MAX_CHARS
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|\Z)")

LONG_SENTENCE = (
    "This one sentence keeps going, with a clause here; another clause there: and a dash — "
    "then more and more words until it is well past the two hundred character limit of a "
    "single request, which forces a cut inside the sentence."
)

CASES = {
    "short": "Hello there.",
    "two_sentences": "First one. Second one is a bit longer than the first.",
    "many_short": " ".join(f"Sentence number {i}." for i in range(40)),
    "long_sentence": LONG_SENTENCE + " After that, a tail.",
    "no_punctuation": " ".join(f"word{i}" for i in range(120)),
    "long_word": "before " + "x" * 450 + " after.",
    "url": "See https://example.com/" + "a" * 230 + "/end for more. Then stop.",
    "code": "Run `pip install --upgrade some-package-name` then `python -m pytest -q`. " * 6,
    "decimals": "Pi is 3.14159 and e is 2.71828. Both are irrational. See example.com today!",
    "newlines": "Line one.\n\nLine two has more.\nLine three?  Yes!   Done.",
    "leading_space": "   Leading spaces. Then text.",
    "exactly_cap": "a" * CAP,
    "exactly_cap_words": ("abcd " * 40).rstrip(),
    "cap_plus_one": "a" * (CAP + 1),
    "closing_quote": 'He said "stop." Then left. (Really!) Yes.',
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_segments_join_to_the_input_exactly_and_fit(name: str) -> None:
    text = CASES[name]
    parts = segment(text)
    assert "".join(parts) == text
    assert all(0 < len(part) <= CAP for part in parts), [len(p) for p in parts]


@pytest.mark.parametrize(
    "name", [n for n in sorted(CASES) if n not in {"long_word", "url", "cap_plus_one"}]
)
def test_no_word_is_split(name: str) -> None:
    text = CASES[name]
    words = text.split()
    parts = segment(text)
    assert [w for part in parts for w in part.split()] == words


def test_an_unbroken_run_over_the_cap_is_the_only_forced_cut() -> None:
    parts = segment(CASES["long_word"])
    assert [len(p) for p in parts if len(p) == CAP] != []
    assert "".join(parts) == CASES["long_word"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_first_segment_is_one_sentence_alone(name: str) -> None:
    text = CASES[name]
    first = segment(text)[0]
    body = first.strip()
    ends = list(_SENTENCE_END.finditer(body))
    # At most one sentence end, and only at the very end of the segment.
    assert len(ends) <= 1
    if ends:
        assert ends[0].end() == len(body)


def test_the_first_segment_does_not_pack_the_next_sentence() -> None:
    assert segment("One. Two. Three.")[0] == "One. "
    assert segment("Hello there. " + "word " * 30)[0] == "Hello there. "


def test_a_first_sentence_over_the_cap_is_cut_but_still_stands_alone() -> None:
    parts = segment(LONG_SENTENCE + " Next.")
    assert parts[0] != LONG_SENTENCE and len(parts[0]) <= CAP
    assert parts[-1] == "Next."
    assert LONG_SENTENCE.startswith(parts[0])


def test_later_sentences_pack_up_to_the_cap() -> None:
    parts = segment("One. " + " ".join(f"S{i}." for i in range(80)))
    assert len(parts) < 80
    assert len(parts[1]) > CAP - 10


def test_a_long_sentence_is_cut_at_a_clause_before_a_space() -> None:
    parts = segment(LONG_SENTENCE)
    assert parts[0].rstrip()[-1] in ",;:—"


def test_empty_and_blank_input() -> None:
    assert segment("") == []
    assert segment("   ") == ["   "]
    assert speak_segments("", max_chars=800).segments == ()


def test_exactly_the_cap_is_one_segment_and_one_more_is_two() -> None:
    assert segment("a" * CAP) == ["a" * CAP]
    assert len(segment("a" * (CAP + 1))) == 2
    assert segment(CASES["exactly_cap_words"]) == [CASES["exactly_cap_words"]]


def test_random_text_keeps_every_property() -> None:
    rng = random.Random(7)
    alphabet = ["a", "bb", "ccc", "d.", "e!", "f?", "g,", "h;", "i:", "j—", "k", "\n", "  "]
    for _ in range(300):
        pieces = (rng.choice(alphabet) * rng.randint(1, 3) for _ in range(rng.randint(1, 120)))
        text = " ".join(pieces)
        parts = segment(text)
        assert "".join(parts) == text
        assert all(0 < len(p) <= CAP for p in parts)
        assert [w for p in parts for w in p.split()] == text.split()


def test_the_cap_applies_before_segmenting() -> None:
    raw = " ".join(f"This is sentence number {i} of a long reply." for i in range(100))
    reply = speak_segments(raw, max_chars=300)
    assert reply.truncated is True
    assert len(reply.text) <= 300
    assert "".join(reply.segments) == reply.text
    # The closing sentence is its own segment, so the phrase cache can serve it.
    assert reply.segments[-1] == speakable.TRUNCATION_TAIL
    assert sum(len(s) for s in reply.segments) <= 300


def test_an_uncut_reply_is_cleaned_not_truncated() -> None:
    reply = speak_segments("Here is code. ```\nls\n``` Done.", max_chars=800)
    assert reply.truncated is False
    assert "ls" not in reply.text.split()
    assert "".join(reply.segments) == reply.text


@pytest.mark.parametrize(
    "cluster",
    [
        "\U0001f1fa\U0001f1f8",  # flag: two regional indicators
        "\U0001f468\u200d\U0001f469\u200d\U0001f467",  # ZWJ family
        "e\u0301\u0323",  # letter with stacked combining marks
        "\U0001f44d\U0001f3fd",  # emoji with skin-tone modifier
    ],
)
@pytest.mark.parametrize("offset", range(0, 4))
def test_a_cut_never_lands_inside_a_grapheme_cluster(cluster: str, offset: int) -> None:
    from local_operator.voice.segments import VOICE_SEGMENT_MAX_CHARS as cap

    # An unbroken run, so the cut is forced near the cap; slide the cluster across it.
    text = "a" * (cap - offset - 1) + cluster + "b" * 50
    parts = segment(text)
    assert "".join(parts) == text
    assert all(len(part) <= cap for part in parts)
    # Whole in one segment means no seam fell inside it.
    assert any(cluster in part for part in parts)
