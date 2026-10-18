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
  answer explaining it in a fence, is never touched.
- A line led by the fullwidth ``｜DSML｜`` token is unambiguous: it is markup
  whether or not the turn carried structured calls.
- A bare ``name="…" string="true|false">value`` line is stripped ONLY when the
  turn carried structured tool calls -- that is the evidence the gateway parsed
  the call and the text is its residue. Without calls the line is handed back
  verbatim.
- With structured calls, everything from the first markup line to the end of the
  turn is residue (a parameter value may itself span lines), together with the
  blank lines that separated it from the prose. A turn that was nothing but
  markup therefore emits no text at all: a tool-only turn, not a blank bubble.

WHAT HOLDING COSTS. Text that might be the head of a markup line is held back
until it can be classified, and once markup has started the rest of the turn is
held until the stream ends (only then is it known whether structured calls
arrived). Ordinary prose streams exactly as before; the visible delays are
trailing blank lines (released with the next prose), a line that starts like
``name="`` (released at its newline), and the tail after a markup line in a turn
that turns out to have no calls (released at the end of the stream).
"""

from __future__ import annotations

import re

_DSML_TOKEN = "\uff5cDSML\uff5c"  # ｜DSML｜

#: A line opened by the DSML token, allowing the ``<``, ``</`` and ``<|`` lead-ins
#: the native format and its mangled variants put in front of it.
_DSML_LEAD = re.compile(r"[ \t]*[<|/]{0,3}" + _DSML_TOKEN)

#: A parameter-tag fragment. Every observed leak keeps the ``string="true|false">``
#: attribute -- the part no prose writes at a line start -- while the head before
#: it is truncated to varying degrees (``parameter name=``, ``name=``, or just
#: ``=``), so the head is optional and the attribute is required.
_BARE_PARAM = re.compile(
    r"[ \t]*(?:[<|/]{0,3}" + _DSML_TOKEN + r"[ \t]*)?"
    r'(?:parameter[ \t]+)?(?:name)?="[^"\n]*"[ \t]+string="(?:true|false)">'
)

_BARE_HEADS = ('parameter name="', 'name="', '="')

#: A line still undecided after this many characters is prose: a real parameter
#: name is a short identifier, so the ``string=`` attribute arrives well before.
_MAX_UNDECIDED = 256

_FENCE = re.compile(r"[ \t]*(?:```|~~~)")

#: How much of a prose line is kept to test it for a fence opener. Only the head
#: matters, and keeping the whole line would make a long line quadratic.
_FENCE_PROBE = 64


def _classify(line: str, *, complete: bool) -> str:
    """``markup``, ``prose`` or ``undecided`` for a line starting at non-space."""
    if _DSML_LEAD.match(line) or _BARE_PARAM.match(line):
        return "markup"
    if complete or len(line) >= _MAX_UNDECIDED:
        return "prose"
    token_tail = line.lstrip("<|/")
    if token_tail and _DSML_TOKEN.startswith(token_tail):
        return "undecided"
    if line.startswith(("<", "|")) and not token_tail:
        return "undecided"
    if any(head.startswith(line) or line.startswith(head) for head in _BARE_HEADS):
        return "undecided"
    return "prose"


class ToolMarkupFilter:
    """Incremental filter: ``feed`` each text delta, then ``finish`` once.

    The concatenation of everything returned equals the input exactly whenever
    nothing was stripped, so a turn without leaked markup is byte-identical.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._line_is_prose = False
        self._line_text = ""
        self._in_fence = False
        #: Once markup has started: the blank lines before it, and everything since.
        self._tail_gap: str | None = None
        self._tail: list[str] = []

    def feed(self, text: str) -> str:
        if self._tail_gap is not None:
            self._tail.append(text)
            return ""
        self._buf += text
        out: list[str] = []
        while self._buf:
            if self._line_is_prose:
                newline = self._buf.find("\n")
                if newline < 0:
                    out.append(self._buf)
                    if len(self._line_text) < _FENCE_PROBE:
                        self._line_text += self._buf[:_FENCE_PROBE]
                    self._buf = ""
                    break
                # The newline stays held: if the next line is markup it is part
                # of the gap that goes with it.
                out.append(self._buf[:newline])
                self._end_prose_line(self._line_text + self._buf[: min(newline, _FENCE_PROBE)])
                self._buf = self._buf[newline:]
                continue
            gap_end = len(self._buf) - len(self._buf.lstrip())
            if gap_end == len(self._buf):
                break  # whitespace only: hold until something follows it
            # Indentation belongs to the line, the blank lines before it to the gap.
            line_start = self._buf.rfind("\n", 0, gap_end) + 1
            rest = self._buf[gap_end:]
            newline = rest.find("\n")
            line = rest if newline < 0 else rest[:newline]
            verdict = "prose" if self._in_fence else _classify(line, complete=newline >= 0)
            if verdict == "undecided":
                break
            if verdict == "markup":
                self._tail_gap = self._buf[:line_start]
                self._tail = [self._buf[line_start:]]
                self._buf = ""
                break
            out.append(self._buf[:gap_end])
            self._buf = rest
            self._line_is_prose = True
            self._line_text = ""
        return "".join(out)

    def finish(self, *, has_tool_calls: bool) -> str:
        """Release whatever is still held, stripped per the module rules."""
        if self._tail_gap is None:
            held, self._buf = self._buf, ""
            return held
        gap, tail = self._tail_gap, "".join(self._tail)
        self._tail_gap, self._tail = None, []
        if has_tool_calls:
            return ""
        kept = _drop_dsml_lines(tail)
        if not kept.strip():
            return ""
        return gap + kept

    def _end_prose_line(self, line: str) -> None:
        self._line_is_prose = False
        self._line_text = ""
        if _FENCE.match(line):
            self._in_fence = not self._in_fence


def _drop_dsml_lines(text: str) -> str:
    """Remove DSML-led lines outside fences; every other line is kept verbatim."""
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if not in_fence and _DSML_LEAD.match(line):
            continue
        if _FENCE.match(line):
            in_fence = not in_fence
        kept.append(line)
    return "".join(kept)


def strip_tool_markup(text: str, *, has_tool_calls: bool) -> str:
    """One-shot form of :class:`ToolMarkupFilter` for an already-assembled text."""
    stream = ToolMarkupFilter()
    return stream.feed(text) + stream.finish(has_tool_calls=has_tool_calls)
