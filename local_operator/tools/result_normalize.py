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
0.843x Claude Code (0.807x in ``strip-links``) with no cited fact lost: every
fact the raw page states outside a URL (44 of 45, and 58 of 58 in the second
arm) is still in the normalized text of the page it was cited from.

What it does (``normalize_markdown``)
-------------------------------------
Only text that looks like web/markdown content is touched: it must carry an
absolute or protocol-relative link, an image or a Wikipedia cite marker
(``_MARKDOWN_SIGNAL``). Anything else -- a source file, a regex, a log, ``ps``
output -- is returned byte for byte, because ``handlers[name](arg)`` is code,
not a link, and its line numbers and spacing are evidence.

* ``![alt](src)`` becomes ``alt``; an image with empty alt is dropped.
* A list item that is exactly one *on-site* link (relative or fragment target,
  anchor of at most ``NAV_ANCHOR_MAX_CHARS``) is site navigation and is
  dropped -- unless such items are most of the text's non-blank lines, in
  which case they are the content (a docs index) and stay as text. A one-link
  item with an absolute target (a search result, a reading list) is never
  dropped; it follows the link rules below like any other line, which keep
  its URL in ``prose-links`` mode.
* Wikipedia ``[[24]](#cite_note-24)`` markers are dropped. A bare ``[24]`` is
  never touched: it collides with code (``list[0]``) and footnote syntax.
* Link hover titles (``[a](url "title")``) are dropped.
* A link is only ``[text](target)`` whose ``[`` is not glued to an identifier,
  ``]`` or ``)`` and whose target is URL-shaped. Relative, fragment and
  protocol-relative targets (leading ``/``, ``#``, ``?``, ``./``, ``../``, or a
  scheme-less string containing ``/``) and ``javascript:`` pseudo-links become
  their anchor text.
* Absolute ``http(s)`` targets with non-empty anchor text survive, in
  ``prose-links`` mode, on *prose* lines (>= 80 visible characters) and on any
  line whose only link is that absolute one (a search result, a README
  reference); in ``strip-links`` mode every such target becomes its anchor
  text. These links are how a navigation-style long task finds its next page
  and what a citation-bound task cites, so the default keeps them (the replay
  measured one link-only discovered fetch in 52).
* Other schemes (``mailto:``, ``tel:``) and non-URL targets are left exactly
  as written: the target may be the datum itself (a contact address).
* A line no rule changed is copied byte for byte (spacing inside a line is
  never collapsed). A changed line loses trailing spaces, and a changed line
  left with nothing but whitespace or a bare list marker is dropped. Only runs
  of 3+ blank lines shrink (to 2), and a trailing newline is kept.
* Fenced code blocks, indented code blocks and inline code spans are copied
  verbatim.

``normalize_tool_text`` routes JSON payloads (the common MCP shape, e.g.
``{"url": ..., "content": ...}``) through a structural walk instead: only
string leaves of >= 200 characters that contain whitespace are normalized;
keys, key order, numbers and short values (ids, URLs, offsets) are untouched.
The value is re-emitted compactly only when a leaf actually changed --
otherwise the server's bytes are kept -- and never when a number would change
value (``1e400``, 20 significant digits, ``NaN``). Lone surrogates (a string
cut mid-emoji) stay ``\\uXXXX`` escapes so the result still encodes as UTF-8.
Citation-bound workflows compare the exact ``url`` string they fetched, so a
URL field must never change.

Constraints
-----------
* **Cache byte-stability.** The transform is applied ONCE, when the
  ``ToolResult`` is created, and the normalized bytes are what is persisted.
  Replay and resume therefore re-send identical bytes and no cached prefix is
  invalidated. It must be pure, deterministic (no clock, locale, hash order or
  environment) and idempotent (``f(f(x)) == f(x)``) so a result that passes
  through twice -- a resumed spill, a re-imported transcript -- cannot drift.
  Idempotence is by construction: the transform repeats until it is a fixed
  point (``MAX_PASSES``), and a text that never settles is sent unchanged.
* **Never worse than off.** Any output that cannot be encoded as UTF-8 falls
  back to the exact input.
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
import decimal
import json
import math
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

#: A one-link list item pointing on-site (relative/fragment target) whose
#: anchor is at most this long is navigation chrome ("Home", "Talk", "Edit
#: links") and is dropped. Longer anchors are content titles and are kept as
#: text. See ``_nav_item`` for the measured distribution behind the number.
NAV_ANCHOR_MAX_CHARS: Final = 40

#: ``normalize_tool_text`` repeats the transform until it stops changing the
#: text (that is what makes ``f(f(x)) == f(x)`` hold); real inputs settle on
#: the second pass. A text still changing after this many passes is returned
#: exactly as the server sent it -- also a fixed point, so idempotence holds.
MAX_PASSES: Final = 4

# A link destination: no whitespace or parens, except one balanced level of
# parens (Wikipedia's ``Foo_(bar)`` titles). The two alternatives start with
# disjoint characters, so the repetition cannot backtrack catastrophically.
_DEST = r"(?:[^()\s]|\([^()\s]*\))*"
_TITLE = r"""(?:\s+(?:"[^"\n]*"|'[^'\n]*'))?"""
# Anchor text is bounded so a long line full of unmatched ``[`` cannot turn
# the scan quadratic; a real anchor is far shorter than this. It may not
# contain ``[`` either, so ``[[a](b)](c)`` matches only the inner link and
# the outer one is reached on the next pass (see ``normalize_tool_text``).
_ANCHOR = r"([^\[\]\n]{0,1000})"
# A ``[`` glued to an identifier, ``]`` or ``)`` is an index or a call --
# ``handlers[name](arg)``, ``callbacks[0](fn)``, ``f(x)[i](y)`` -- never a
# link. Costs a converter's rare ``word[link](/x)`` with no space; that line
# keeps its markup, which only costs savings.
_NOT_GLUED = r"(?<![\w\])])"
_IMAGE = re.compile(r"!\[" + _ANCHOR + r"\]\(\s*" + _DEST + _TITLE + r"\s*\)")
_LINK = re.compile(_NOT_GLUED + r"\[" + _ANCHOR + r"\]\(\s*(" + _DEST + ")" + _TITLE + r"\s*\)")
_CITE = re.compile(r"\[\[[^\]\s]{1,12}\]\]\(#cite_note[^)\s]*\)")
_LIST_MARKER = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])(?:\s+|$)")
_ONE_LINK_ITEM = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])\s+" + _LINK.pattern + r"\s*$")
_MARKER_ONLY = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])\s*$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")
_ABSOLUTE = re.compile(r"^https?://", re.IGNORECASE)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_SITE_PREFIX = re.compile(r"^(?:/|#|\?|\./|\.\./|javascript:)", re.IGNORECASE)
_URLISH_KEY = re.compile(r"(?:^|_)(?:url|uri|href|link)s?$", re.IGNORECASE)
# What makes a text web/markdown content rather than code, a log or a table:
# an absolute or protocol-relative link, an image, or a Wikipedia cite marker.
# Fetched pages always carry at least one; a source file, a regex or ``ps``
# output essentially never does, and without one the text is sent untouched.
_MARKDOWN_SIGNAL = re.compile(
    r"\]\(\s*(?:https?:)?//|!\[[^\[\]\n]{0,1000}\]\(|\[\[[^\]\s]{1,12}\]\]\(#cite_note",
    re.IGNORECASE,
)


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


def _is_site_target(target: str) -> bool:
    """A relative, fragment, protocol-relative or ``javascript:`` target.

    Only URL-shaped strings qualify: a leading ``/``, ``#``, ``?``, ``./`` or
    ``../``, or a scheme-less string containing ``/`` (``docs/a.md``). A bare
    word or expression (``arg``, ``*args``, ``\\d+``, ``foo|bar``) is not a
    target, so code that merely looks like ``[x](y)`` is never rewritten.
    """
    if _SITE_PREFIX.match(target):
        return True
    return "/" in target and not _SCHEME.match(target)


def _strip_media(line: str) -> str:
    """Images become their alt text and cite markers go (outside inline code)."""
    return _outside_inline_code(
        line, lambda seg: _CITE.sub("", _IMAGE.sub(lambda m: m.group(1).strip(), seg))
    )


def _nav_item(line: str) -> bool:
    """A list item that is exactly one short on-site link: site navigation.

    Absolute targets never qualify -- a list of absolute links is a search
    result list, a reading list or a README's references, i.e. the content.
    The 40-character anchor bound comes from the replay corpus (106 fetched
    pages of the 2026-11-23 paired run, PR #4514): its 606 on-site one-link
    items had anchors of median 14, p90 28, p95 35, p99 50 and max 77
    characters; 40 drops 96.7% of them, and keeping the longer tail as text
    cost 344 of ~137k normalized cl100k tokens.
    """
    match = _ONE_LINK_ITEM.match(line)
    return (
        match is not None
        and _is_site_target(match.group(2))
        and len(match.group(1).strip()) <= NAV_ANCHOR_MAX_CHARS
    )


def _normalize_line(original: str, line: str, keep_prose_links: bool, drop_nav: bool) -> str | None:
    """Normalize one text line (``line`` = ``original`` after ``_strip_media``).

    ``None`` drops the line. A line no rule changed is returned byte-for-byte:
    spacing inside a line is never collapsed (column layouts and code survive).
    """
    if drop_nav and _nav_item(line):
        return None

    targets: list[str] = []

    def as_text(match: re.Match[str]) -> str:
        target = match.group(2)
        if _ABSOLUTE.match(target) or _is_site_target(target):
            targets.append(target)
            return match.group(1)
        return match.group(0)

    visible = _outside_inline_code(line, lambda seg: _LINK.sub(as_text, seg))
    # A short line whose ONLY link is absolute -- a search result, a reading
    # list entry, a README reference -- is often the one place the model learns
    # which URL to fetch or cite, so ``prose-links`` keeps it like prose. A nav
    # bar or breadcrumb carries several links per line (or on-site targets,
    # handled above), so it still loses its targets.
    single_absolute = len(targets) == 1 and _ABSOLUTE.match(targets[0]) is not None
    prose = keep_prose_links and (
        single_absolute or len(" ".join(visible.split())) >= PROSE_LINE_MIN_CHARS
    )

    def relink(match: re.Match[str]) -> str:
        anchor, target = match.group(1), match.group(2)
        if _ABSOLUTE.match(target):
            if prose and anchor.strip():
                return f"[{anchor}]({target})"
            return anchor
        if _is_site_target(target):
            return anchor
        # mailto:, tel: and other schemes (the target may be the datum), and
        # anything not URL-shaped (code), stay exactly as written.
        return match.group(0)

    line = _outside_inline_code(line, lambda seg: _LINK.sub(relink, seg))
    if line == original:
        return line
    line = line.rstrip(" \t")
    if _is_blank(line) or _MARKER_ONLY.match(line):
        return None  # the line was only scaffolding (an icon, a cite, an empty bullet)
    return line


def normalize_markdown(text: str, *, keep_prose_links: bool = True) -> str:
    """Remove link/image/citation scaffolding from markdown/web ``text``.

    Text without a markdown signal (``_MARKDOWN_SIGNAL``) is returned
    unchanged. Pure and deterministic; ``normalize_tool_text`` adds the
    fixed-point loop that makes it idempotent. See the module docstring.
    """
    if not text or not _MARKDOWN_SIGNAL.search(text):
        return text
    # A trailing newline is a terminator, not a blank line: keep it exactly.
    body, terminator = (text[:-1], "\n") if text.endswith("\n") else (text, "")
    runs = [
        (is_code, run, run if is_code else [_strip_media(line) for line in run])
        for is_code, run in _split_code(body.split("\n"))
    ]
    # On-site nav bullets are chrome around content; when they ARE most of the
    # result (a docs index, a sitemap tool) they are the content and stay.
    nonblank = sum(1 for _, run, _ in runs for line in run if not _is_blank(line))
    nav = sum(1 for is_code, _, staged in runs if not is_code for line in staged if _nav_item(line))
    drop_nav = nav * 2 <= nonblank

    out: list[str] = []
    dropped = False  # a line was removed since the last one kept
    for is_code, run, staged in runs:
        if is_code:
            out.extend(run)
            dropped = False
            continue
        for original, line in zip(run, staged):
            normalized = _normalize_line(original, line, keep_prose_links, drop_nav)
            if normalized is None:
                dropped = True
                continue
            if _is_blank(normalized) and out and _is_blank(out[-1]):
                # Removing lines must not open a gap; elsewhere only runs of 3+
                # blank lines shrink (to 2), so line layout stays recognisable.
                if dropped or (len(out) > 1 and _is_blank(out[-2])):
                    continue
            out.append(normalized)
            dropped = False
    return "\n".join(out) + terminator


class _DuplicateKey(Exception):
    pass


class _LossyNumber(Exception):
    """A JSON number that would not re-emit with the same value."""


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Re-emitting would silently drop one of the values; leave the
            # payload exactly as the server sent it instead.
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _exact_float(token: str) -> float:
    # ``1e400`` would re-emit as ``Infinity`` (invalid JSON) and
    # ``0.1234567890123456789`` would lose digits a Decimal-backed server
    # meant. Spelling may change (``1.50`` -> ``1.5``); the value may not.
    value = float(token)
    if not math.isfinite(value) or decimal.Decimal(repr(value)) != decimal.Decimal(token):
        raise _LossyNumber(token)
    return value


def _reject_constant(token: str) -> float:
    raise _LossyNumber(token)  # NaN / Infinity are not JSON; never re-emit them


def _escape_surrogate(match: re.Match[str]) -> str:
    return "\\u%04x" % ord(match.group())


# A Python ``str`` from ``json.loads`` joins valid surrogate pairs, so any
# surrogate left is lone (a server that cut a string mid-emoji). Written raw
# it cannot be UTF-8 encoded and every later provider request fails; as a
# ``\uXXXX`` escape it is valid JSON with the same parsed value.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _walk(value: Any, keep_prose_links: bool, changed: list[bool], key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _walk(v, keep_prose_links, changed, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, keep_prose_links, changed, key) for v in value]
    if (
        isinstance(value, str)
        and len(value) >= JSON_LEAF_MIN_CHARS
        and any(c.isspace() for c in value)
        and not (key is not None and _URLISH_KEY.search(key))
    ):
        normalized = normalize_markdown(value, keep_prose_links=keep_prose_links)
        if normalized != value:
            changed.append(True)
        return normalized
    return value


def _normalize_once(text: str, keep_prose_links: bool) -> str:
    head = text.lstrip()[:1]
    if head in ("{", "["):
        try:
            parsed = json.loads(
                text,
                object_pairs_hook=_pairs_without_duplicates,
                parse_float=_exact_float,
                parse_constant=_reject_constant,
            )
        except (_DuplicateKey, _LossyNumber):
            return text
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            changed: list[bool] = []
            walked = _walk(parsed, keep_prose_links, changed)
            if not changed:
                return text  # nothing to remove: keep the server's exact bytes
            out = json.dumps(walked, ensure_ascii=False, separators=(",", ":"))
            return _LONE_SURROGATE.sub(_escape_surrogate, out)
    return normalize_markdown(text, keep_prose_links=keep_prose_links)


def normalize_tool_text(text: str, mode: str = RESULT_NORMALIZATION_DEFAULT) -> str:
    """Normalize one tool result's flattened text under ``mode``.

    ``off`` returns ``text`` unchanged (the identity, byte for byte). A JSON
    object or array is walked structurally and re-emitted compactly, but only
    when a string leaf actually changed; JSON with duplicate keys or a number
    that would not re-emit exactly is returned unchanged; anything else takes
    the markdown path. The transform is repeated to a fixed point (at most
    ``MAX_PASSES``), and an output that would not encode as UTF-8 falls back
    to ``text``.
    """
    if mode == "off" or not text:
        return text
    if mode not in RESULT_NORMALIZATION_MODES:
        raise ValueError(f"unknown result normalization mode: {mode!r}")
    keep_prose_links = mode == "prose-links"
    current = text
    for _ in range(MAX_PASSES):
        following = _normalize_once(current, keep_prose_links)
        if following == current:
            break
        current = following
    else:
        return text
    if current is not text:
        try:
            current.encode("utf-8")
        except UnicodeEncodeError:
            return text
    return current
