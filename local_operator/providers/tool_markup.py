"""Strip leaked DeepSeek tool-call markup from streamed assistant text.

WHY THIS EXISTS. DeepSeek models write tool calls in a native markup
(``<｜DSML｜invoke name="x">`` / ``<｜DSML｜parameter name="i" string="true">…``,
the bars being fullwidth U+FF5C). An OpenAI-compatible gateway in front of the
model (seen live on ``radient`` / ``auto``) parses that markup into structured
``tool_calls`` -- correctly -- but lets fragments of it through on the
``delta.content`` channel as well. Observed fragments, always at the start of a
line and always trailing the real prose::

    ｜DSML｜ parameter name="channel_id" string="true">C0123456789
    parameter name="i" string="true">Logged quietly
    ="i" string="true">Logged quietly
    name="keywords" string="false">["a", "b"]
    i" string="true">Logged quietly

Left alone, the fragment is persisted as assistant prose, painted in the
transcript, and replayed to the model on the next request -- where the model
reads its own leak as precedent and starts writing more of it.

WHY IN THE STREAM, NOT AT ASSEMBLY. Cleaning only the assembled text would leave
the live render showing markup the transcript later does not contain, and the
provider's replay text (whose fingerprint must equal the persisted text) would
have to be cleaned separately again. Filtering the deltas themselves keeps all
three copies -- live render, persisted ``assistant.content``, native replay text
-- the same string by construction.

THE RULES, each one load-bearing:

- Only a LINE START can open markup (after optional indentation), and never
  inside a fenced code block. Prose that quotes the syntax mid-sentence, or an
  answer explaining it in a fence, is never touched. Fences follow CommonMark:
  a line opens one only if it is a run of three or more backticks or tildes
  plus an info string (a backtick run's info string holds no backtick, so a line
  that starts with an inline code span opens nothing), and only a bare run of
  the same character, at least as long, closes it.
- A line led by the fullwidth ``｜DSML｜`` token is unambiguous: it is markup
  whether or not the turn carried structured calls.
- A bare ``name="…" string="true|false">value`` line -- its head cut down to
  any degree, as far as a bare identifier tail like ``i"`` or nothing at all --
  is stripped ONLY when the turn carried structured tool calls -- that is the
  evidence the gateway parsed the call and the text is its residue. Without
  calls the line is handed back verbatim.
- With structured calls, everything from the first markup line to the end of the
  turn is residue (a parameter value may itself span lines), together with the
  blank lines that separated it from the prose. A turn that was nothing but
  markup therefore emits no text at all: a tool-only turn, not a blank bubble.
  Non-markup lines dropped this way are logged at debug level, by size only.
- A ``\r\n`` line break is one break: it goes with the gap whole, so the prose
  kept before stripped markup never ends in a stray ``\r``.

WHAT HOLDING COSTS. Text that might be the head of a markup line is held back
until it can be classified, and once markup has started the rest of the turn is
held until the stream ends (only then is it known whether structured calls
arrived). Ordinary prose streams exactly as before; the visible delays are
trailing blank lines (released with the next prose), a line that starts like
``name="`` (released at its newline), the first word of each line until the
character after it shows it is not a parameter-name tail, and the tail after a
markup line in a turn that turns out to have no calls (released at the end of
the stream).

WHAT HOLDING LOSES. Held text is released only by ``feed`` or ``finish``. When
the provider fails mid-stream the client calls ``finish(has_tool_calls=False)``
before raising -- no call from a failed turn is acted on, so held text is not
the residue of one -- and the partial prose, its last unfinished word included,
still reaches the operator (DSML-led lines are still dropped). When the
CONSUMER closes the stream or the task is cancelled, there is nobody left to
receive text, so what was held at that moment is never sent: trailing blank
lines, the undecided start of a line (its first word, at least) and -- once a
markup line has been seen -- the whole rest of the turn, including any genuine
prose after it. There is no evidence for or against structured calls before the
stream ends, so this cannot be decided earlier.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_DSML_TOKEN = "\uff5cDSML\uff5c"  # ｜DSML｜

#: A line opened by the DSML token, allowing the ``<``, ``</`` and ``<|`` lead-ins
#: the native format and its mangled variants put in front of it.
_DSML_LEAD = re.compile(r"[ \t]*[<|/]{0,3}" + _DSML_TOKEN)

#: The characters of a parameter name. ASCII only on purpose: a ``\w`` run would
#: also cover CJK prose, which has no spaces and would then be held at every line
#: start until its first punctuation mark.
_NAME_CHARS = r"[A-Za-z0-9_.-]"

#: A parameter-tag fragment. Every observed leak keeps the ``string="true|false">``
#: attribute -- the part no prose writes at a line start -- while the head before
#: it is truncated to varying degrees: ``parameter name="i"``, ``name="i"``,
#: ``="i"``, only the tail of the name and its closing quote (``i"``,
#: ``keywords"``), or nothing at all. So the head is optional and the attribute
#: is required.
_LEAD_IN = r"[<|/]{0,3}"
_NAME_HEAD = r'(?:parameter[ \t]+)?(?:name)?="[^"\n]*"'
_TAIL_HEAD = _NAME_CHARS + r'*"?'
_ATTR = r'string="(?:true|false)">'
_BARE_PARAM = re.compile(
    r"[ \t]*(?:" + _LEAD_IN + _DSML_TOKEN + r"[ \t]*)?"
    r"(?:(?:" + _NAME_HEAD + "|" + _TAIL_HEAD + r")[ \t]+)?" + _ATTR
)


def _prefixes(literal: str) -> str:
    """A pattern matching every prefix of ``literal``, the empty one included."""
    pattern = ""
    for char in reversed(literal):
        pattern = f"(?:{re.escape(char)}{pattern})?"
    return pattern


#: Exactly the line starts that can still grow into a ``_DSML_LEAD`` or
#: ``_BARE_PARAM`` match, built from the same pieces so the two cannot drift.
#: Holding precisely these -- no more, no fewer -- is what makes the verdict
#: independent of how the stream is chunked. It includes a line that is still
#: only a name tail (``i``, ``keywords"``, ``i" st``), so the first word of every
#: line is held until the character after it rules that out: a few characters of
#: live-render delay.
_UNDECIDED = re.compile(
    # Inside the DSML lead-in: once the token is whole the line is markup.
    r"(?:" + _LEAD_IN + _prefixes(_DSML_TOKEN)
    # Inside a name head, or a bare name tail (which covers ``parameter``).
    + r"|(?:parameter[ \t]+)?(?:" + _prefixes("name") + r'|(?:name)?=(?:"[^"\n]*"?)?)'
    + "|" + _TAIL_HEAD
    # Inside the attribute, after a head or with none.
    + r"|(?:(?:" + _NAME_HEAD + "|" + _TAIL_HEAD + r")[ \t]+)?"
    + r"(?:" + _prefixes('string="')
    + r'|string="(?:' + _prefixes('true">') + "|" + _prefixes('false">') + "))"
    + r")\Z"
)  # fmt: skip

#: A line still undecided after this many characters is prose: a real parameter
#: name is a short identifier, so the ``string=`` attribute arrives well before.
#: Only this much of a line is ever classified, in one piece or in many.
_MAX_UNDECIDED = 256

_SPACE_RUN = re.compile(r"\s*")

#: A fence run at the start of a line's content. Indentation is stripped before
#: this is tried and is not bounded to three spaces as CommonMark has it: a fence
#: inside a list item is indented further, and missing it would read the code
#: inside as prose.
_FENCE_RUN = re.compile(r"`{3,}|~{3,}")

#: How much of a prose line is kept to test it for a fence. Only the head
#: matters, and keeping the whole line would make a long line quadratic; past
#: the probe only "did a backtick / any text follow" is tracked.
_FENCE_PROBE = 64


def _classify(line: str, *, complete: bool) -> str:
    """``markup``, ``prose`` or ``undecided`` for a line starting at non-space."""
    if _DSML_LEAD.match(line) or _BARE_PARAM.match(line):
        return "markup"
    if complete or len(line) >= _MAX_UNDECIDED:
        return "prose"
    return "undecided" if _UNDECIDED.match(line) else "prose"


def _next_fence(
    fence: tuple[str, int] | None, probe: str, *, late_backtick: bool, late_text: bool
) -> tuple[str, int] | None:
    """The fence state after a line, per CommonMark's opening and closing rules.

    ``probe`` is the line's first characters; ``late_*`` say whether a backtick,
    or any non-space text, came after the probe. A backtick fence's info string
    may hold no backtick (so a line that merely starts with an inline code span
    opens nothing), and only a bare run of the same character at least as long
    as the opener closes it (so a three-backtick line inside a four-backtick
    fence is content).
    """
    run_match = _FENCE_RUN.match(probe)
    if run_match is None:
        return fence
    run, info = run_match.group(), probe[run_match.end() :]
    if fence is None:
        if run[0] == "`" and ("`" in info or late_backtick):
            return None
        return (run[0], len(run))
    char, length = fence
    if run[0] == char and len(run) >= length and not info.strip() and not late_text:
        return None
    return fence


class ToolMarkupFilter:
    """Incremental filter: ``feed`` each text delta, then ``finish`` once.

    The concatenation of everything returned equals the input exactly whenever
    nothing was stripped, so a turn without leaked markup is byte-identical.
    Every delta is scanned once (an undecided line start is re-read, but never
    past ``_MAX_UNDECIDED`` characters), so the cost is linear in the stream.
    """

    def __init__(self) -> None:
        #: Where the stream is: between lines (``gap``), at the start of a line not
        #: yet classified (``head``), or inside a prose line (``prose``).
        self._at = "gap"
        #: Held between lines: whitespace through the last newline, then the
        #: indentation after it, then the undecided start of the line.
        self._gap: list[str] = []
        self._indent: list[str] = []
        self._head: list[str] = []
        self._head_len = 0
        #: A ``\r`` that ended the last delta inside a prose line. It is held in
        #: case it is half of ``\r\n``: the line break goes with the gap, so a
        #: markup line after it must not leave a stray ``\r`` on the prose.
        self._cr = False
        self._probe = ""
        self._late_backtick = False
        self._late_text = False
        self._fence: tuple[str, int] | None = None
        #: Once markup has started: the blank lines before it, and everything since.
        self._tail_gap = ""
        self._tail: list[str] | None = None

    def feed(self, text: str) -> str:
        if self._tail is not None:
            self._tail.append(text)
            return ""
        out: list[str] = []
        pos = 0
        while pos < len(text) and self._tail is None:
            if self._at == "prose":
                pos = self._feed_prose(text, pos, out)
            elif self._at == "head":
                pos = self._feed_head(text, pos, out)
            else:
                pos = self._feed_gap(text, pos)
        return "".join(out)

    def finish(self, *, has_tool_calls: bool) -> str:
        """Release whatever is still held, stripped per the module rules."""
        if self._tail is None:
            held = ("\r" if self._cr else "") + "".join(self._gap + self._indent + self._head)
            self._cr, self._gap, self._indent, self._head = False, [], [], []
            return held
        gap, tail = self._tail_gap, "".join(self._tail)
        self._tail_gap, self._tail = "", None
        if has_tool_calls:
            _log_dropped_prose(tail)
            return ""
        kept = _drop_dsml_lines(tail)
        if not kept.strip():
            return ""
        return gap + kept

    def _feed_gap(self, text: str, pos: int) -> int:
        end = _SPACE_RUN.match(text, pos).end()  # type: ignore[union-attr]  # ``*`` always matches
        run = text[pos:end]
        newline = run.rfind("\n")
        if newline < 0:
            if run:
                self._indent.append(run)
        else:
            # Indentation belongs to the line, the blank lines before it to the gap.
            self._gap += self._indent
            self._gap.append(run[: newline + 1])
            self._indent = [run[newline + 1 :]] if newline + 1 < len(run) else []
        if end < len(text):
            self._at = "head"
        return end

    def _feed_head(self, text: str, pos: int, out: list[str]) -> int:
        newline = text.find("\n", pos)
        piece = text[pos:] if newline < 0 else text[pos:newline]
        room = _MAX_UNDECIDED - self._head_len
        line = "".join(self._head) + piece[:room]
        if self._fence is not None:
            verdict = "prose"
        else:
            verdict = _classify(line, complete=newline >= 0 and len(piece) <= room)
        if verdict == "undecided":
            self._head.append(piece)
            self._head_len += len(piece)
            return len(text)
        if verdict == "markup":
            self._tail_gap = "".join(self._gap)
            self._tail = ["".join(self._indent), "".join(self._head), text[pos:]]
            self._gap, self._indent, self._head, self._head_len = [], [], [], 0
            return len(text)
        head = "".join(self._head)
        if head.endswith("\r"):
            # Possibly half of a ``\r\n``: held as in a prose line, so that the
            # line break goes with the gap whole.
            head, self._cr = head[:-1], True
        out.append("".join(self._gap) + "".join(self._indent) + head)
        self._gap, self._indent, self._head, self._head_len = [], [], [], 0
        self._at = "prose"
        self._note_prose(head)
        return pos

    def _feed_prose(self, text: str, pos: int, out: list[str]) -> int:
        if self._cr:
            self._cr = False
            if text[pos] == "\n":
                self._end_prose_line("\r\n")
                return pos + 1
            out.append("\r")
            self._note_prose("\r")
        newline = text.find("\n", pos)
        if newline < 0:
            piece = text[pos:]
            if piece.endswith("\r"):
                self._cr, piece = True, piece[:-1]
            out.append(piece)
            self._note_prose(piece)
            return len(text)
        # The line break stays held: if the next line is markup it is part of the
        # gap that goes with it.
        end = newline - 1 if newline > pos and text[newline - 1] == "\r" else newline
        out.append(text[pos:end])
        self._note_prose(text[pos:end])
        self._end_prose_line(text[end : newline + 1])
        return newline + 1

    def _note_prose(self, piece: str) -> None:
        if len(self._probe) < _FENCE_PROBE:
            take = _FENCE_PROBE - len(self._probe)
            self._probe, piece = self._probe + piece[:take], piece[take:]
        if piece:
            self._late_backtick = self._late_backtick or "`" in piece
            self._late_text = self._late_text or not piece.isspace()

    def _end_prose_line(self, line_break: str) -> None:
        self._fence = _next_fence(
            self._fence,
            self._probe,
            late_backtick=self._late_backtick,
            late_text=self._late_text,
        )
        self._probe, self._late_backtick, self._late_text = "", False, False
        self._gap.append(line_break)
        self._at = "gap"


def _lines(text: str) -> list[str]:
    """Split on ``\\n`` only, keeping it, as the stream does (not ``splitlines``,
    which also breaks on form feeds and U+2028 and so would disagree with it)."""
    return re.findall(r"[^\n]*\n|[^\n]+", text)


def _fence_after_line(fence: tuple[str, int] | None, line: str) -> tuple[str, int] | None:
    content = line.rstrip("\r\n").lstrip()
    probe, late = content[:_FENCE_PROBE], content[_FENCE_PROBE:]
    return _next_fence(
        fence,
        probe,
        late_backtick="`" in late,
        late_text=bool(late) and not late.isspace(),
    )


def _drop_dsml_lines(text: str) -> str:
    """Remove DSML-led lines outside fences; every other line is kept verbatim."""
    kept: list[str] = []
    fence: tuple[str, int] | None = None
    for line in _lines(text):
        if fence is None and _DSML_LEAD.match(line):
            continue
        fence = _fence_after_line(fence, line)
        kept.append(line)
    return "".join(kept)


def _log_dropped_prose(tail: str) -> None:
    """Record, by size only, text dropped with the markup that was not markup.

    The end-of-turn rule drops everything after the first markup line, which in
    the observed data is only markup and its multi-line values. If a model ever
    writes prose after it, this line is what makes that findable. Content is
    never logged: it is the user's conversation.
    """
    other = [
        line
        for line in _lines(tail)
        if line.strip() and not (_DSML_LEAD.match(line) or _BARE_PARAM.match(line))
    ]
    if other:
        logger.debug(
            "tool markup filter: dropped %d non-markup line(s), %d chars, after a markup "
            "line in a turn with tool calls",
            len(other),
            sum(len(line) for line in other),
        )


def strip_tool_markup(text: str, *, has_tool_calls: bool) -> str:
    """One-shot form of :class:`ToolMarkupFilter` for an already-assembled text.

    The client only streams, so production code drives the filter directly. This
    stays public because it is the whole-text reference the stream is held to
    (every chunking must give the same result) and the entry point for
    re-checking recorded transcripts offline.
    """
    stream = ToolMarkupFilter()
    return stream.feed(text) + stream.finish(has_tool_calls=has_tool_calls)
