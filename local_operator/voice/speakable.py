"""Turn agent-written text into text worth reading aloud (V8).

The SERVER decides what is spoken: ``/v1/desktop/voice/text`` takes a source reference
(a message id, a gate), resolves the text itself and passes it through here. A
renderer never posts free text, so nothing outside the agent's own words — or a
pending approval card as the server reads it — can ever be voiced.

What this does, in order:

1. Fenced code blocks become a short spoken placeholder. Reading a diff aloud is
   noise, and the code is on screen anyway.
2. Markdown punctuation that a synthesiser would read literally (``**``, ``#``
   headings, list bullets, table pipes, inline-code backticks) is dropped, and
   ``[text](url)`` links keep their text.
3. Bare URLs are replaced by the word "link". A URL read aloud is a string of
   letters nobody can act on.
4. ``[[…]]`` embedded commands (a control token the agent writes for the
   harness, never prose) are removed outright: there is nothing in one for a
   listener, and reading one aloud would voice the machinery.
5. Whitespace is folded.
6. The result is capped at ``max_spoken_chars`` on a sentence boundary when one
   exists inside the cap, else a word boundary, and a short closing sentence
   says the rest is on screen. The closing sentence is spoken, so it rides
   INSIDE the cap rather than after it (see :func:`cap`).

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
#: Spoken, not shown: the listener has no way to see that a cut happened, so the
#: closing sentence has to say it in the first person. It counts against the cap
#: like any other spoken word.
TRUNCATION_TAIL = "That's where I'll stop — the rest is on screen."

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
#: Non-greedy and single-line-bounded so one stray ``[[`` cannot swallow the rest
#: of a reply; an unclosed opener stays as text rather than eating the answer.
_EMBEDDED_COMMAND = re.compile(r"\[\[[^\n]*?\]\](?:\([^()\s]*\))?")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()\s]+)\)")
_URL = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_WHITESPACE = re.compile(r"\s+")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
#: A paragraph's first token: ``Spoken:`` or ``**Spoken:**`` (the model bolds it
#: now and then). Paragraph start only, so the word mid-sentence is not a marker.
_SPOKEN_LABEL = re.compile(
    r"\s*(?:\*\*\s*Spoken\s*:\s*\*\*|\*\*\s*Spoken\s*\*\*\s*:|Spoken\s*:)", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


@dataclass(frozen=True)
class Speakable:
    """The text to speak, and whether the cap cut it."""

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
    """Steps 1-5 of the module docstring: what a listener should hear."""
    text = _FENCE.sub(f" {CODE_PLACEHOLDER} ", text)
    # Before the link pass, and taking a trailing ``(target)`` with it: a
    # ``[[x]](y)`` would otherwise be read as a link whose text is ``[x``.
    text = _EMBEDDED_COMMAND.sub(" ", text)
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
    """Step 6: cut ``text`` to ``max_chars`` at the best boundary inside it.

    The closing sentence counts against the cap, so the spoken total never
    exceeds it: ``/voice/text``'s ``spoken_chars`` is a promise the caller can trust.
    That arithmetic is why :data:`TRUNCATION_TAIL` may not grow past
    ``MIN_MAX_SPOKEN_CHARS - 1``: at the floor the tail would otherwise consume
    the whole budget and the reply would be nothing but a goodbye.
    """
    if len(text) <= max_chars:
        return Speakable(text=text, truncated=False)
    # Room for the answer, with the tail RESERVED out of the budget and joined
    # after. The assembled string is never sliced: a no-whitespace window takes
    # the whole budget, the full stop added to end it costs one more character,
    # and a final ``[:max_chars]`` therefore used to remove the tail's own full
    # stop — the listener's only sign that the reply stopped.
    room = max_chars - len(TRUNCATION_TAIL) - 1
    if room <= 0:
        # Unreachable through the route (the registry floor is 100), but ``cap``
        # is pure and callable with anything: there is no room for an answer, so
        # the announcement stands alone, still inside the promise.
        return Speakable(text=TRUNCATION_TAIL[: max(0, max_chars)], truncated=True)
    window = text[:room]
    ends = [match.end() for match in _SENTENCE_END.finditer(window)]
    # A sentence boundary in the back half wins; one near the start would
    # throw most of the budget away, so a word boundary is better there.
    if ends and ends[-1] >= room // 2:
        cut = window[: ends[-1]]
    else:
        space = window.rfind(" ")
        cut = window[:space] if space > 0 else window
    cut = cut.rstrip(" ,;:-")
    if cut and cut[-1] not in ".!?":
        cut += "."
    # The tail outranks the answer: the cut loses its last character rather than
    # the tail losing its full stop (in practice only the no-whitespace case,
    # whose added period is exactly the one character over).
    if len(cut) + 1 + len(TRUNCATION_TAIL) > max_chars:
        cut = cut[:-1].rstrip(" ,;:-")
    spoken = f"{cut} {TRUNCATION_TAIL}".strip()
    return Speakable(text=spoken, truncated=True)


def select_spoken(text: str) -> str:
    """The LAST ``Spoken:`` paragraph with its label dropped, else ``text`` whole.

    The ``<voice>`` system section (``prompts_api.VOICE_BLOCK``) asks the model to
    end each reply with that paragraph. Fences are masked first so a ``Spoken:``
    inside a code block is neither a marker nor split by a blank line in the code.
    An empty summary falls back to the whole text, the pre-summary behaviour.
    """
    masked = _FENCE.sub(f" {CODE_PLACEHOLDER} ", text)
    for paragraph in reversed(_BLANK_LINE.split(masked)):
        label = _SPOKEN_LABEL.match(paragraph)
        if label is None:
            continue
        body = paragraph[label.end() :]
        # An empty LAST marker does not reach back to an earlier summary: the
        # model's final word is the one that was meant to be heard.
        return body if clean(body) else text
    return text


def speakable(text: str, *, max_chars: int = DEFAULT_MAX_SPOKEN_CHARS) -> Speakable:
    """Select the summary, clean, then cap: the one call ``/voice/text`` makes."""
    return cap(clean(select_spoken(text)), normalise_max_chars(max_chars))
