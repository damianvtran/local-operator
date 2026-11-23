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
from collections.abc import Callable
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

#: The English stand-ins, and the DEFAULTS of ``clean``/``speakable``. The routes
#: pass the active locale's words (``wire.voice.spoken.*``) so this module stays
#: pure and a Spanish reply does not say "link" through the TTS.
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
# Display cleaning (``display_text``): the same markup removal, but lines survive.
_BULLET_LINE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s*")
_QUOTE_LINE = re.compile(r"^\s*>\s?")
_INLINE_SPACE = re.compile(r"[ \t]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
#: A URL or inline-code span is parked behind this sentinel (private-use code
#: points) while emphasis is unwrapped, so an underscore or asterisk inside an
#: address or an identifier is never eaten. ``display_text`` strips both code
#: points from its input first, so a reply can never forge a placeholder.
_URL_HOLD = re.compile("\ue000(\\d+)\ue001")
#: ``clean`` only knows backtick fences; the display path also hides ``~~~`` ones.
#: A tilde fence must open its own line (a lone ``~~~`` inside prose is not one).
_DISPLAY_FENCE = re.compile(r"```.*?(?:```|\Z)|(?:(?<=\n)|\A)[ \t]{0,3}~~~.*?(?:~~~|\Z)", re.DOTALL)
#: GFM accepts one dash per delimiter cell. ``-`` or ``|-`` alone is ordinary text
#: (an empty bullet, a dash), so a short delimiter row only counts directly under a
#: header line that carries a pipe (``_is_table_rule``).
_DISPLAY_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
#: Display emphasis, one alternative per marker, tried in this order at each position.
#: Unlike ``clean`` (whose output nobody reads), a screen shows identifiers, so an
#: underscore or asterisk INSIDE a word is not emphasis: ``my_var_name``, ``a*b*c``,
#: ``foo_bar.py``. ``_``/``__`` need a non-word character (or the line edge) outside
#: both ends. ``*``/``**`` are only refused when an end sits between two ASCII word
#: characters, so ``**bold**`` still works beside CJK text, which has no spaces.
#: ``__`` also needs whitespace inside: a bare ``__init__`` / ``__name__`` is a Python
#: dunder far more often than one bold word, and the agent writes bold as ``**``.
_W = "A-Za-z0-9_"
#: Longest emphasised span: bounds the backtracking per opener (see _MAX_EMPHASIS_LINE).
_SPAN = 300
_DISPLAY_EMPHASIS = re.compile(
    r"(?<!\w)__(?=\S)(?P<d>\S(?:(?!__).)*?\s(?:(?!__).)*?)(?<=\S)__(?!\w)"
    rf"|(?<!\w)_(?=[^\s_])(?P<u>.{{1,{_SPAN}}}?)(?<=\S)_(?!\w)"
    rf"|(?:(?<![{_W}])|(?=\*+(?![{_W}*])))(?P<m>\*\*|\*)(?=\S)(?P<s>.{{1,{_SPAN}}}?)(?<=\S)"
    rf"(?:(?<![{_W}])(?P=m)|(?P=m)(?![{_W}]))"
    rf"|~~(?=\S)(?P<t>.{{1,{_SPAN}}}?)(?<=\S)~~"
)
#: ``_DISPLAY_EMPHASIS`` backtracks on unclosed markers (``_a _a _a ...``), so a line
#: this long is left as written rather than risk quadratic time on a request path.
_MAX_EMPHASIS_LINE = 2000
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


def clean(text: str, *, code_omitted: str = CODE_PLACEHOLDER, link: str = LINK_WORD) -> str:
    """Steps 1-5 of the module docstring: what a listener should hear.

    ``code_omitted`` and ``link`` are the words spoken for a skipped code block
    and for a web address, in the caller's language. They are inserted through a
    function so a backslash or ``\\1`` in a translation is text, never a group
    reference.
    """
    text = _FENCE.sub(lambda _match: f" {code_omitted} ", text)
    # Before the link pass, and taking a trailing ``(target)`` with it: a
    # ``[[x]](y)`` would otherwise be read as a link whose text is ``[x``.
    text = _EMBEDDED_COMMAND.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(lambda _match: link, text)
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


def _is_table_rule(line: str, above: str) -> bool:
    stripped = line.strip()
    if _TABLE_RULE.fullmatch(stripped) and "-" in stripped:
        return True
    header = above.strip()
    return (
        "|" in header
        and bool(header.strip("| \t"))
        and not _DISPLAY_TABLE_RULE.fullmatch(header)
        and bool(_DISPLAY_TABLE_RULE.fullmatch(stripped))
    )


def _unwrap_emphasis(match: re.Match[str]) -> str:
    return next(group for name, group in match.groupdict().items() if name != "m" and group)


def _inline(text: str, hold: Callable[[str], str]) -> str:
    """Park code spans and URLs, then unwrap emphasis (twice: ``**_x_**`` has two layers).

    Code is parked first so a URL's greedy match cannot swallow the closing backtick.
    A URL gives its trailing emphasis marks back (``**https://a.com**``), so they pair
    up with the opener instead of printing as asterisks.
    """
    text = _INLINE_CODE.sub(lambda match: hold(match.group(1)), text)

    def _url(match: re.Match[str]) -> str:
        url = match.group(0)
        core = url.rstrip("*_~")
        return hold(core) + url[len(core) :]

    text = _URL.sub(_url, text)
    if len(text) <= _MAX_EMPHASIS_LINE:
        text = _DISPLAY_EMPHASIS.sub(_unwrap_emphasis, text)
        text = _DISPLAY_EMPHASIS.sub(_unwrap_emphasis, text)
    return text


def display_text(text: str, *, code_omitted: str = CODE_PLACEHOLDER) -> str:
    """The reply cleaned for a SCREEN: ``clean``'s markup removal, lines kept.

    ``clean`` folds every run of whitespace and drops pipes because a speaker has
    no lines; shown flat, a markdown table reads as run-on words. This keeps the
    line structure instead: single newlines stay, three or more become one blank
    line, a bullet keeps its marker, a table row is one line with its cells joined
    by `` · `` (the ``|---|`` rule row goes), and a web address stays as written
    (it is on screen anyway, unlike in ``clean``). Fenced code is still
    ``code_omitted`` (``~~~`` fences too), links keep their text, and emphasis,
    inline-code backticks, heading and quote markers are removed. Unlike ``clean``,
    what is shown must stay readable: code-span contents and URLs are untouched, and
    an ``_``/``*`` inside a word (``my_var_name``) is not emphasis. Pure and
    uncapped: the audio cap never applies.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # The placeholder takes its own line, without adding a blank one when the
    # fence already started or ended a line.
    text = _DISPLAY_FENCE.sub(
        lambda match: (
            ("" if match.start() == 0 or text[match.start() - 1] == "\n" else "\n")
            + code_omitted
            + ("" if text[match.end() : match.end() + 1] == "\n" else "\n")
        ),
        text,
    )
    # After the fence pass, so the placeholder is covered too: no input may carry
    # the code points ``hold`` parks text behind (it would restore the wrong span).
    text = text.replace("\ue000", "").replace("\ue001", "")
    text = _EMBEDDED_COMMAND.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    held: list[str] = []

    def hold(span: str) -> str:
        # Resolved now, so a span parked around another (code holding an escaped
        # pipe) restores in one pass at the end.
        held.append(_URL_HOLD.sub(lambda match: held[int(match.group(1))], span))
        return f"\ue000{len(held) - 1}\ue001"

    lines = text.split("\n")
    rules = {
        index
        for index, line in enumerate(lines)
        if _is_table_rule(line, lines[index - 1] if index else "")
    }
    # A table is the run of pipe lines touching a rule row (header above, body
    # below); a line that merely STARTS with a pipe is a row on its own.
    rows: set[int] = set()
    for index in rules:
        for step in (-1, 1):
            at = index + step
            while 0 <= at < len(lines) and "|" in lines[at] and at not in rules:
                rows.add(at)
                at += step
    out: list[str] = []
    for index, line in enumerate(lines):
        if index in rules:
            continue
        if index in rows or line.lstrip().startswith("|"):
            # An escaped pipe is a literal one, not a cell break (GFM).
            line = line.replace("\\|", hold("|"))
            cells = [
                _INLINE_SPACE.sub(" ", _inline(cell, hold)).strip() for cell in line.split("|")
            ]
            out.append(" · ".join(cell for cell in cells if cell))
            continue
        line = _QUOTE_LINE.sub("", _HEADING_LINE.sub("", line))
        bullet = _BULLET_LINE.match(line)
        if bullet:
            indent, marker, rest = bullet.groups()
            body = _INLINE_SPACE.sub(" ", _inline(rest, hold)).strip()
            out.append(f"{indent.expandtabs(4)}{marker} {body}")
            continue
        out.append(_INLINE_SPACE.sub(" ", _inline(line, hold)).strip())
    joined = _BLANK_RUNS.sub("\n\n", "\n".join(out)).strip()
    return _URL_HOLD.sub(lambda match: held[int(match.group(1))], joined)


def cap(text: str, max_chars: int, *, tail: str = TRUNCATION_TAIL) -> Speakable:
    """Step 6: cut ``text`` to ``max_chars`` at the best boundary inside it.

    The closing sentence counts against the cap, so the spoken total never
    exceeds it: ``/voice/text``'s ``spoken_chars`` is a promise the caller can trust.
    That arithmetic is why the tail may not grow past
    ``MIN_MAX_SPOKEN_CHARS - 1``: at the floor the tail would otherwise consume
    the whole budget and the reply would be nothing but a goodbye.

    ``tail`` defaults to :data:`TRUNCATION_TAIL`; the desktop route passes the
    active locale's ``wire.voice.phrase.truncated`` so a translated listener
    hears the cut in their own language. This module stays pure (no i18n import).
    """
    if len(text) <= max_chars:
        return Speakable(text=text, truncated=False)
    # Room for the answer, with the tail RESERVED out of the budget and joined
    # after. The assembled string is never sliced: a no-whitespace window takes
    # the whole budget, the full stop added to end it costs one more character,
    # and a final ``[:max_chars]`` therefore used to remove the tail's own full
    # stop — the listener's only sign that the reply stopped.
    room = max_chars - len(tail) - 1
    if room <= 0:
        # Unreachable through the route (the registry floor is 100), but ``cap``
        # is pure and callable with anything: there is no room for an answer, so
        # the announcement stands alone, still inside the promise.
        return Speakable(text=tail[: max(0, max_chars)], truncated=True)
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
    if len(cut) + 1 + len(tail) > max_chars:
        cut = cut[:-1].rstrip(" ,;:-")
    spoken = f"{cut} {tail}".strip()
    return Speakable(text=spoken, truncated=True)


def speakable(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_SPOKEN_CHARS,
    tail: str = TRUNCATION_TAIL,
    code_omitted: str = CODE_PLACEHOLDER,
    link: str = LINK_WORD,
) -> Speakable:
    """Clean then cap: the one call ``/voice/text`` makes."""
    return cap(
        clean(text, code_omitted=code_omitted, link=link), normalise_max_chars(max_chars), tail=tail
    )
