"""Turn agent-written text into text worth reading aloud (V8).

The SERVER decides what is spoken: the speech route takes a source reference
(a message id, a gate), resolves the text itself and passes it through here. A
renderer never posts free text, so nothing outside the agent's own words — or a
host-authored gate card — can ever be voiced.

What this does, in order:

1. Fenced code blocks become a short spoken placeholder. Reading a diff aloud is
   noise, and the code is on screen anyway.
2. Markdown punctuation that a synthesiser would read literally (``**``, ``#``
   headings, list bullets, table pipes, inline-code backticks) is dropped, and
   ``[text](url)`` links keep their text.
3. Bare URLs are replaced by the word "link". A URL read aloud is a string of
   letters nobody can act on.
4. Whitespace is folded.
5. The result is capped at ``max_spoken_chars`` on a sentence boundary when one
   exists inside the cap, else a word boundary, and a short closing sentence
   says the rest is on screen.

Pure and dependency-free, so it is cheap to pin in tests and safe to call on
the request path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The registry default for ``voice.max_spoken_chars`` (``settings_io.py``
#: carries the same literal; ``tests/unit/test_settings_io.py`` pins the two).
#: About a minute of speech at an ordinary pace: long enough for a real
#: answer, short enough that a long report does not hold the speaker hostage.
DEFAULT_MAX_SPOKEN_CHARS = 800

#: The registry's bounds. Below the floor the cap would cut most answers to a
#: fragment; above the ceiling one answer could outlast any provider's timeout.
MIN_MAX_SPOKEN_CHARS = 100
MAX_MAX_SPOKEN_CHARS = 4000

CODE_PLACEHOLDER = "(code block omitted)"
LINK_WORD = "link"
TRUNCATION_TAIL = "The rest is on screen."

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()\s]+)\)")
_URL = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


@dataclass(frozen=True)
class Speakable:
    """The text to synthesise, and whether the cap cut it."""

    text: str
    truncated: bool


def normalise_max_chars(value: object) -> int:
    """The cap as the route will apply it: an int inside the registry's bounds.

    A hand-edited ``config.yml`` can hold anything. A value that is not a
    number reads as the default; one outside the bounds is clamped. Voice never
    fails because of this setting.
    """
    if isinstance(value, bool):
        return DEFAULT_MAX_SPOKEN_CHARS
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return DEFAULT_MAX_SPOKEN_CHARS
    return max(MIN_MAX_SPOKEN_CHARS, min(MAX_MAX_SPOKEN_CHARS, number))


def clean(text: str) -> str:
    """Steps 1-4 of the module docstring: what a listener should hear."""
    text = _FENCE.sub(f" {CODE_PLACEHOLDER} ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(LINK_WORD, text)
    text = _TABLE_RULE.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _QUOTE.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    # Emphasis is unwrapped twice so a nested `**_x_**` loses both layers.
    text = _EMPHASIS.sub(r"\2", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = text.replace("|", " ")
    return _WHITESPACE.sub(" ", text).strip()


def cap(text: str, max_chars: int) -> Speakable:
    """Step 5: cut ``text`` to ``max_chars`` at the best boundary inside it.

    The closing sentence counts against the cap, so the spoken total never
    exceeds it: ``X-Voice-Spoken-Chars`` is a promise the renderer can trust.
    """
    if len(text) <= max_chars:
        return Speakable(text=text, truncated=False)
    budget = max(0, max_chars - len(TRUNCATION_TAIL) - 1)
    window = text[:budget]
    ends = [match.end() for match in _SENTENCE_END.finditer(window)]
    # A sentence boundary in the back half wins; one near the start would
    # throw most of the budget away, so a word boundary is better there.
    if ends and ends[-1] >= budget // 2:
        cut = window[: ends[-1]]
    else:
        space = window.rfind(" ")
        cut = window[:space] if space > 0 else window
    cut = cut.rstrip(" ,;:-")
    if cut and cut[-1] not in ".!?":
        cut += "."
    spoken = f"{cut} {TRUNCATION_TAIL}".strip()
    return Speakable(text=spoken[:max_chars], truncated=True)


def speakable(text: str, *, max_chars: int = DEFAULT_MAX_SPOKEN_CHARS) -> Speakable:
    """Clean then cap: the one call the speech route makes."""
    return cap(clean(text), normalise_max_chars(max_chars))
