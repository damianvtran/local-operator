"""Deterministic normalization of tool-result text before it enters context.

Why this exists
---------------
Tool results are what a long or short agent session actually pays for. On a
measured paired run (12 Pergamon enrichment records, 2026-11-23) tool results
were ~95% of all prompt-cache writes, and inside fetched page text the
markdown *link targets* alone were 24.7% of the characters and image markup
another 4.0%. Those bytes carry almost no evidence: navigation bullets,
relative wiki links, ``#cite_note`` anchors, tracking pixels and icon images.
Every token of them is written to the cache once and then re-read on every
later turn, so the cost grows with session length. An offline replay of the
real run through this transform re-priced the candidate arm from 0.939x to
~0.82x Claude Code with every cited fact (45/45 and 58/58) still present in
the normalized text of the page it was cited from.

What it does (``normalize_markdown``)
-------------------------------------
* ``![alt](src)`` becomes ``alt``; an image with empty alt is dropped.
* A list item that is exactly one link (a nav bullet) is dropped.
* Wikipedia ``[[24]](#cite_note-24)`` markers are dropped. A bare ``[24]`` is
  never touched: it collides with code (``list[0]``) and footnote syntax.
* Link hover titles (``[a](url "title")``) are dropped.
* Relative, fragment and protocol-relative (``//host``) targets, and
  ``javascript:`` pseudo-links, become their anchor text.
* Absolute ``http(s)`` targets survive only on *prose* lines (>= 80 visible
  characters) with non-empty anchor text, in ``prose-links`` mode; in
  ``strip-links`` mode every target becomes its anchor text. Prose links are
  how a navigation-style long task finds its next page, so the default keeps
  them (the replay measured one link-only discovered fetch in 52).
* Other schemes (``mailto:``, ``tel:``) are left alone: their target is often
  the datum itself (a contact address).
* Interior whitespace runs collapse to one space, trailing whitespace goes,
  and blank-line runs collapse to a single blank line.
* Fenced code blocks, indented code blocks and inline code spans are copied
  verbatim.

``normalize_tool_text`` routes JSON payloads (the common MCP shape, e.g.
``{"url": ..., "content": ...}``) through a structural walk instead: only
string leaves of >= 200 characters that contain whitespace are normalized,
keys, key order, numbers and short values (ids, URLs, offsets) are untouched,
and the value is re-emitted compactly. Citation-bound workflows compare the
exact ``url`` string they fetched, so a URL field must never change.

Constraints
-----------
* **Cache byte-stability.** The transform is applied ONCE, when the
  ``ToolResult`` is created, and the normalized bytes are what is persisted.
  Replay and resume therefore re-send identical bytes and no cached prefix is
  invalidated. It must be pure, deterministic (no clock, locale, hash order or
  environment) and idempotent (``f(f(x)) == f(x)``) so a result that passes
  through twice -- a resumed spill, a re-imported transcript -- cannot drift.
* **No evidence removed.** Nothing here selects passages, deduplicates or
  truncates: those levers measurably dropped cited support. The raw server
  payload is kept in ``details["server_result"]`` (never sent to a model).
* **Errors are not normalized.** An error body is diagnostics, often small,
  and its exact text matters more than its size.

How to disable
--------------
``/settings`` -> ``mcp.result_normalization`` (or ``lop config edit``;
``values.mcp.result_normalization`` in ``config.yml``): ``prose-links``
(default), ``strip-links``, or ``off``. ``off`` is byte-identical to the
behaviour before this module. Read per result, so an edit lands on the next
MCP call.
"""

from __future__ import annotations

import bisect
import json
import re
from typing import Any, Final, Literal

NormalizationMode = Literal["prose-links", "strip-links", "off"]

#: The registry default and the value an unreadable/unknown setting falls back to.
RESULT_NORMALIZATION_DEFAULT: Final[NormalizationMode] = "prose-links"
RESULT_NORMALIZATION_MODES: Final[tuple[NormalizationMode, ...]] = (
    "prose-links",
    "strip-links",
    "off",
)
#: ``config.yml`` path of the ``mcp.result_normalization`` setting. Read by
#: ``mcp.manager.read_result_normalization_mode`` (this module does no I/O).
RESULT_NORMALIZATION_PATH: Final = ("mcp", "result_normalization")

#: A line with at least this many visible characters (links rendered as their
#: anchor text) is prose; its absolute links are kept in ``prose-links`` mode.
#: 80 is the threshold the replay was measured at: nav bars, breadcrumbs,
#: headings and table cells fall below it, running sentences above it.
PROSE_LINE_MIN_CHARS: Final = 80

#: JSON string leaves shorter than this are identifiers, URLs, titles or
#: snippets too small to carry markup worth removing; leaving them alone keeps
#: every short field byte-equal.
JSON_LEAF_MIN_CHARS: Final = 200

# A link destination: no whitespace or parens, except one balanced level of
# parens (Wikipedia's ``Foo_(bar)`` titles). The two alternatives start with
# disjoint characters, so the repetition cannot backtrack catastrophically.
_DEST = r"(?:[^()\s]|\([^()\s]*\))*"
_TITLE = r"""(?:\s+(?:"[^"\n]*"|'[^'\n]*'))?"""
# Anchor text is bounded so a long line full of unmatched ``[`` cannot turn
# the scan quadratic; a real anchor is far shorter than this.
_ANCHOR = r"([^\]\n]{0,1000})"
_IMAGE = re.compile(r"!\[" + _ANCHOR + r"\]\(\s*" + _DEST + _TITLE + r"\s*\)")
_LINK = re.compile(r"\[" + _ANCHOR + r"\]\(\s*(" + _DEST + ")" + _TITLE + r"\s*\)")
_CITE = re.compile(r"\[\[[^\]\s]{1,12}\]\]\(#cite_note[^)\s]*\)")
_LIST_MARKER = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])(?:\s+|$)")
_ONE_LINK_ITEM = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])\s+" + _LINK.pattern + r"\s*$")
_MARKER_ONLY = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])\s*$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")
_SPACE_RUN = re.compile(r"(?<=\S)[ \t]{2,}")
_ABSOLUTE = re.compile(r"^https?://", re.IGNORECASE)
_DROP_TARGET = re.compile(r"^(?:/|#|\.|\?|javascript:|[^:/?#]*(?:[/?#]|$))", re.IGNORECASE)
_URLISH_KEY = re.compile(r"(?:^|_)(?:url|uri|href|link)s?$", re.IGNORECASE)


def _indent_width(line: str) -> int:
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4 - width % 4
        else:
            break
    return width


def _is_blank(line: str) -> bool:
    return not line.strip()


def _split_code(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """Partition lines into ``(is_code, lines)`` runs.

    Fences follow CommonMark: a closing fence uses the same character and is at
    least as long; an unclosed fence runs to the end. An indented code block
    starts at a line indented >= 4 columns after a blank line (or the start),
    whose own text is not a list marker and whose preceding non-blank line is
    not a list item -- otherwise it is a list continuation, which markdown
    renders as text. Being conservative only costs savings, never content.
    """
    runs: list[tuple[bool, list[str]]] = []

    def emit(is_code: bool, line: str) -> None:
        if runs and runs[-1][0] == is_code:
            runs[-1][1].append(line)
        else:
            runs.append((is_code, [line]))

    i = 0
    last_text: str | None = None  # last non-blank, non-code line
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_OPEN.match(line)
        if fence:
            marker = fence.group(1)
            closing = re.compile(
                r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$"
            )
            emit(True, line)
            i += 1
            while i < len(lines):
                emit(True, lines[i])
                i += 1
                if closing.match(lines[i - 1]):
                    break
            last_text = None
            continue
        prev_blank = i == 0 or _is_blank(lines[i - 1])
        if (
            prev_blank
            and _indent_width(line) >= 4
            and not _is_blank(line)
            and not _LIST_MARKER.match(line)
            and not (
                last_text is not None
                and (_LIST_MARKER.match(last_text) or _indent_width(last_text) >= 2)
            )
        ):
            # Consume the block: indented lines and interior blank lines.
            j = i
            while j < len(lines) and (_indent_width(lines[j]) >= 4 or _is_blank(lines[j])):
                j += 1
            while j > i and _is_blank(lines[j - 1]):
                j -= 1  # trailing blanks belong to the text that follows
            for k in range(i, j):
                emit(True, lines[k])
            i = j
            last_text = None
            continue
        emit(False, line)
        if not _is_blank(line):
            last_text = line
        i += 1
    return runs


def _inline_code_spans(line: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each inline code span, CommonMark-style.

    An opening backtick run closes at the next run of exactly the same length;
    a run with no closer is literal text. Pairing precomputed runs keeps this
    linear -- a backreference regex rescans to the end of the line for every
    unmatched backtick, which is quadratic on minified script text.
    """
    runs = [(m.start(), m.end()) for m in _BACKTICKS.finditer(line)]
    by_length: dict[int, list[int]] = {}
    for index, (start, end) in enumerate(runs):
        by_length.setdefault(end - start, []).append(index)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(runs):
        start, end = runs[i]
        same = by_length[end - start]
        position = bisect.bisect_right(same, i)
        if position == len(same):
            i += 1  # no closer: a literal backtick run
            continue
        closer = same[position]
        spans.append((start, runs[closer][1]))
        i = closer + 1
    return spans


def _outside_inline_code(line: str, transform: Any) -> str:
    """Apply ``transform`` to the parts of ``line`` outside inline code spans."""
    if "`" not in line:
        return transform(line)
    out: list[str] = []
    pos = 0
    for start, end in _inline_code_spans(line):
        out.append(transform(line[pos:start]))
        out.append(line[start:end])
        pos = end
    out.append(transform(line[pos:]))
    return "".join(out)


def _collapse(line: str) -> str:
    return _SPACE_RUN.sub(" ", line).rstrip()


def _normalize_line(line: str, keep_prose_links: bool) -> str | None:
    """Normalize one text line; ``None`` drops it."""
    line = _collapse(line)
    if not line:
        return line
    line = _outside_inline_code(
        line, lambda seg: _CITE.sub("", _IMAGE.sub(lambda m: m.group(1).strip(), seg))
    )
    if _ONE_LINK_ITEM.match(line):
        return None
    visible = _collapse(
        _outside_inline_code(line, lambda seg: _LINK.sub(lambda m: m.group(1), seg))
    )
    prose = keep_prose_links and len(visible.strip()) >= PROSE_LINE_MIN_CHARS

    def relink(match: re.Match[str]) -> str:
        anchor, target = match.group(1), match.group(2)
        if _ABSOLUTE.match(target):
            if prose and anchor.strip():
                return f"[{anchor}]({target})"
            return anchor
        if target.startswith("//") or _DROP_TARGET.match(target):
            return anchor
        # mailto:, tel: and other schemes: the target may be the datum.
        return f"[{anchor}]({target})"

    line = _collapse(_outside_inline_code(line, lambda seg: _LINK.sub(relink, seg)))
    if _MARKER_ONLY.match(line):
        return None  # a bullet whose only content was an image or a cite marker
    return line


def normalize_markdown(text: str, *, keep_prose_links: bool = True) -> str:
    """Remove link/image/citation scaffolding from markdown-ish ``text``.

    Pure, deterministic and idempotent; see the module docstring for the rules.
    """
    if not text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    for is_code, run in _split_code(lines):
        if is_code:
            out.extend(run)
            continue
        for line in run:
            normalized = _normalize_line(line, keep_prose_links)
            if normalized is None:
                continue
            if not normalized and out and not out[-1]:
                continue  # one blank line between blocks is enough
            out.append(normalized)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


class _DuplicateKey(Exception):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Re-emitting would silently drop one of the values; leave the
            # payload exactly as the server sent it instead.
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _walk(value: Any, keep_prose_links: bool, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _walk(v, keep_prose_links, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, keep_prose_links, key) for v in value]
    if (
        isinstance(value, str)
        and len(value) >= JSON_LEAF_MIN_CHARS
        and any(c.isspace() for c in value)
        and not (key is not None and _URLISH_KEY.search(key))
    ):
        return normalize_markdown(value, keep_prose_links=keep_prose_links)
    return value


def normalize_tool_text(text: str, mode: str = RESULT_NORMALIZATION_DEFAULT) -> str:
    """Normalize one tool result's flattened text under ``mode``.

    ``off`` returns ``text`` unchanged (the identity, byte for byte). A JSON
    object or array is walked structurally and re-emitted compactly; anything
    else -- including JSON that does not parse, scalars and objects with
    duplicate keys -- takes the markdown path or, for duplicate keys, is
    returned unchanged.
    """
    if mode == "off" or not text:
        return text
    if mode not in RESULT_NORMALIZATION_MODES:
        raise ValueError(f"unknown result normalization mode: {mode!r}")
    keep_prose_links = mode == "prose-links"
    head = text.lstrip()[:1]
    if head in ("{", "["):
        try:
            parsed = json.loads(text, object_pairs_hook=_pairs_without_duplicates)
        except _DuplicateKey:
            return text
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return json.dumps(
                _walk(parsed, keep_prose_links), ensure_ascii=False, separators=(",", ":")
            )
    return normalize_markdown(text, keep_prose_links=keep_prose_links)
