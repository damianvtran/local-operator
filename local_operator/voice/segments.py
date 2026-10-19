"""Cut a spoken reply into segments a 200-character TTS rung can speak (D3).

Groq's Orpheus refuses more than 200 characters per request, and one cap for
every rung is simpler than asking which rung will serve (the server does not
know until it tries). The server cuts, never the client: a client that cut would
have to post free text back, breaking V8.

Properties, pinned by ``tests/unit/voice/test_segments.py``:

* every segment is at most :data:`VOICE_SEGMENT_MAX_CHARS` characters;
* the segments joined with ``""`` equal the input EXACTLY (whitespace stays on
  the segment it followed, so nothing is dropped or invented);
* the FIRST segment is one sentence alone (cut when it alone exceeds the cap),
  so the first audio starts as soon as a short request can finish;
* no word is split, with one unavoidable exception: a single unbroken run of
  non-space characters longer than the cap (a URL, a minified blob) has no
  boundary to cut at, and the cap is a provider refusal, so it is cut at the cap.

Pure and dependency-free, like :mod:`local_operator.voice.speakable`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_operator.voice import speakable as voice_speakable

#: One constant for every rung. Equals ``tts.clients.GROQ_TTS_MAX_CHARS``; this
#: module stays free of client imports, and the route test pins the two equal.
VOICE_SEGMENT_MAX_CHARS = 200

#: A clause cut earlier than this throws most of the budget away; a space cut is
#: better there.
_MIN_CLAUSE_CUT = 50
_CLAUSE_MARKS = ",;:—"

#: A sentence ends at ``.!?`` (plus closing quotes or brackets) before whitespace
#: or the end, so ``example.com`` and ``3.5`` stay whole. The trailing whitespace
#: belongs to the sentence it followed.
_SENTENCE = re.compile(r"[^\s].*?(?:[.!?]+[\"')\]]*(?=\s|\Z)|\Z)\s*|\s+", re.DOTALL)


@dataclass(frozen=True)
class SpokenReply:
    """The cleaned, capped text and the segments cut from it."""

    text: str
    truncated: bool
    segments: tuple[str, ...]


def _cut_point(text: str, limit: int) -> int:
    """Where to cut ``text`` (longer than ``limit``) so the head fits ``limit``.

    Clause punctuation first, then a space, then (an unbroken run) the limit.
    A cut sits next to whitespace, which stays with the head when it fits.
    """
    candidates = [
        index
        for index in range(limit, 0, -1)
        if (text[index - 1].isspace() or text[index].isspace()) and text[:index].strip()
    ]
    for index in candidates:
        head = text[:index].rstrip()
        if len(head) >= _MIN_CLAUSE_CUT and head[-1] in _CLAUSE_MARKS:
            return index
    return candidates[0] if candidates else limit


def _split_to_fit(piece: str, limit: int) -> list[str]:
    parts: list[str] = []
    while len(piece) > limit:
        cut = _cut_point(piece, limit)
        parts.append(piece[:cut])
        piece = piece[cut:]
    if piece:
        parts.append(piece)
    return parts


def segment(text: str, *, max_chars: int = VOICE_SEGMENT_MAX_CHARS) -> list[str]:
    """Cut ``text`` into segments of at most ``max_chars``; see the module docstring."""
    if not text:
        return []
    sentences = [match.group(0) for match in _SENTENCE.finditer(text)]
    out: list[str] = []
    # The first segment is the first sentence ALONE: a short request is the
    # quickest way to a first sound, and the rest can be packed behind it.
    if len(sentences) > 1 and sentences[0].isspace():
        sentences[:2] = [sentences[0] + sentences[1]]
    first, rest = sentences[0], sentences[1:]
    pieces = _split_to_fit(first, max_chars)
    out.extend(pieces[:-1])
    # A long first sentence's last piece is still "the first sentence", so it
    # also stands alone rather than packing with the next sentence.
    out.append(pieces[-1])
    buffer = ""
    for sentence in rest:
        for piece in _split_to_fit(sentence, max_chars):
            if buffer and len(buffer) + len(piece) > max_chars:
                out.append(buffer)
                buffer = ""
            buffer += piece
    if buffer:
        out.append(buffer)
    return out


def speak_segments(raw: str, *, max_chars: int) -> SpokenReply:
    """Clean, cap, then segment: the one call the voice routes make.

    ``max_chars`` (``voice.max_spoken_chars``) caps the WHOLE text BEFORE
    segmenting, so a long reply never costs more requests than the cap allows.
    The truncation tail is a fixed phrase that the phrase cache serves, so it is
    its own final segment and never packs into a paid one.
    """
    spoken = voice_speakable.speakable(raw, max_chars=max_chars)
    text = spoken.text
    tail = voice_speakable.TRUNCATION_TAIL
    if spoken.truncated and text.endswith(tail):
        segments = [*segment(text[: -len(tail)]), tail]
    else:
        segments = segment(text)
    return SpokenReply(text=text, truncated=spoken.truncated, segments=tuple(segments))
