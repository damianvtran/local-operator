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
0.864x Claude Code (0.806x in ``strip-links``) with no cited fact lost: every
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
* A list item that is exactly one *on-site* link (root-path ``/x``, fragment
  ``#x`` or ``javascript:`` target, anchor of at most
  ``NAV_ANCHOR_MAX_CHARS``) is site navigation and is dropped -- unless such
  items are most of the text's non-blank lines, in which case they are the
  content (a docs index) and stay as text. A one-link item with an absolute
  or file-relative target (a search result, a reading list, a README's
  ``docs/install.md``) is never dropped; it follows the link rules below like
  any other line.
* Wikipedia ``[[24]](#cite_note-24)`` markers are dropped. A bare ``[24]`` is
  never touched: it collides with code (``list[0]``) and footnote syntax.
* Link hover titles (``[a](url "title")``) are dropped.
* A link is only ``[text](target)`` whose ``[`` is not glued to an identifier
  or ``]`` (nor to a ``)`` that did not close a link) and whose target is
  URL-shaped. On-site targets (leading ``/``, ``#``, ``?``) and
  ``javascript:`` pseudo-links become their anchor text.
* File-relative targets (``./a.md``, ``../b.md``, or a scheme-less string
  containing ``/`` such as ``docs/c.md``) name the file holding the content,
  so the path always survives: ``prose-links`` keeps ``[text](path)``,
  ``strip-links`` writes ``text (path)``.
* Absolute ``http(s)`` and protocol-relative ``//`` targets with non-empty
  anchor text survive, in
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
  truncates page text: those levers measurably dropped cited support. The
  three narrow exceptions are each replayed against every cited fact (41/41
  kept on run 2): search-hit snippets are clipped (below) and dropped for a
  URL the session already showed (``elide_seen_snippets``) -- discovery text,
  never a source -- and a fetched page loses its References / See also /
  External links / Notes / Further reading sections
  (``elide_boilerplate_sections``). The raw server payload is kept in
  ``details["server_result"]`` (never sent to a model).
* **Errors are not normalized.** An error body is diagnostics, often small,
  and its exact text matters more than its size.

Search-hit snippet clipping (``clip_search_hits``)
--------------------------------------------------
A *search* tool's hit list is the one place this module shortens text, and it
is the only exception to "nothing truncates" above. Why: on the same paired
run every search hit carried a provider snippet of 1,098 characters on
average (median 1,161, max 1,510, 213 hits) and a ``published_date`` that was
null on 213 of 213. Snippets exist to choose what to fetch, not to cite -- the
built-in ``web_search`` tool has clipped them to 320 characters for exactly
that reason -- yet the MCP path passed them whole, and each one is re-read on
every later turn. Replayed through the real module, clipping hit snippets to
400 characters re-priced the candidate arm from 0.864x to 0.707x Claude Code
(``prose-links``; 0.806x to 0.648x in ``strip-links``) with every
search-discovered cited URL still naming its subject in its title, clipped
snippet or url (39/39 and 43/43 across the two arms). 320 lost one of 43, so
the MCP default is 400. Fetched pages are never clipped, so every cited fact
the page supports is still there.

Constraints: it applies only to a tool whose bare MCP name has ``search``
as a whole word (``is_search_tool``), only to a JSON list whose every item is a hit
object (a ``title`` key plus a URL-shaped key, see ``_is_hit``), and inside a
hit only to snippet-like string fields (``SNIPPET_KEYS``) and empty fields.
URL-keyed and ``title`` values are never touched (citation-bound workflows
compare the exact ``url`` they fetched). It runs inside the fixed-point loop,
so it is idempotent: a clipped field is at most ``max_chars`` long and is
never clipped again. ``0`` disables it, and ``off`` normalization disables it
too (``off`` stays the byte-identical escape hatch).

How to disable
--------------
``/settings`` -> ``mcp.result_normalization`` (or ``lop config edit``;
``values.mcp.result_normalization`` in ``config.yml``): ``prose-links``
(default), ``strip-links``, or ``off``. ``off`` is byte-identical to the
behaviour before this module. Read per result, so an edit lands on the next
MCP call. ``mcp.search_snippet_chars`` (default 400; ``0`` = keep snippets
whole) sets the hit-snippet clip on its own. ``mcp.result_text_rendering``
(default on) switches the lossless JSON -> text rendering of hit lists and
fetched pages (``render_result_text``) on its own.

Result shaping v2 (seen-URL elision, text rendering, section elision and the
glued/cut link fix) re-priced POC run 2 from 0.658x to 0.626x Claude Code
(``strip-links``, the run's mode) with 41/41 cited facts, 233/233 ``url``
fields byte-equal and 39/39 cited search discoveries still naming their
subject (structural-economy replay, PR description).
"""

from __future__ import annotations

import bisect
import decimal
import json
import math
import re
from collections.abc import Callable
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

#: Default for ``mcp.search_snippet_chars``: the longest a snippet-like field
#: of a search hit may stay (``0`` = never clip). 400, not the built-in
#: ``web_search`` tool's 320: on the replayed run 320 hid the subject name of
#: one of 43 search-discovered cited URLs and 400 hid none (module docstring).
SEARCH_SNIPPET_CHARS_DEFAULT: Final = 400
#: ``config.yml`` path of ``mcp.search_snippet_chars``; read by
#: ``mcp.manager.read_search_snippet_chars``.
SEARCH_SNIPPET_CHARS_PATH: Final = ("mcp", "search_snippet_chars")
#: The registry's upper bound. A cap above any measured snippet (max 1,510
#: chars on the replayed run) is effectively "off"; the bound only stops a
#: typo from becoming a multi-megabyte integer in the config.
SEARCH_SNIPPET_CHARS_MAX: Final = 100_000

#: Fields of a search hit that hold discovery text: provider snippets
#: (Tavily ``snippet``/``content``, Exa ``text``/``summary``, Brave
#: ``description``). Only these are clipped; every other field (ids, dates,
#: scores, ``title``, any URL-shaped key) is copied as is.
SNIPPET_KEYS: Final = frozenset({"snippet", "content", "description", "text", "summary"})
#: The words of a tool name: an uppercase run not followed by lowercase
#: (``HTTP`` in ``HTTPSearch``), a capitalised or lowercase word, or a digit
#: run. Anything else (``_``, ``-``, ``.``, spaces) separates words, so
#: ``searchWeb``, ``tavily-search`` and ``brave_web_search`` each yield a
#: ``search`` word while ``research`` and ``elasticsearch`` stay one word.
_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def clip_snippet(text: str, limit: int) -> str:
    """Collapse whitespace and cut ``text`` to at most ``limit`` characters.

    The one snippet clipper: the built-in ``web_search`` tool and the MCP
    search-hit clip both use it, so a snippet looks the same whichever path
    found it. A cut text ends with ``…`` and is exactly ``limit`` characters
    or fewer, so clipping it again changes nothing.
    """
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def is_search_tool(tool_name: str) -> bool:
    """Whether an MCP tool's BARE name (as the server lists it) marks a search.

    The bare name, never the harness label ``mcp__<server>_<tool>``: a server
    called ``elasticsearch`` or ``research`` must not turn its fetch or
    document-read tool into a "search" whose content gets clipped.

    ``search`` must be a whole word of the name (split on ``_``, ``-``,
    ``.``, camelCase and digits), not a substring: ``perplexity_research``,
    ``deep_researcher_check`` or ``elasticsearch_get_document`` return
    documents or reports whose body IS the answer, and clipping their text
    fields as if they were discovery snippets would drop cited content
    (agent review R1 / QA Q1 on PR #4525).
    """
    return any(word.lower() == "search" for word in _NAME_WORD.findall(tool_name))


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
#: the second pass. The count INCLUDES the final pass that confirms nothing
#: changed, so at most ``MAX_PASSES - 1`` passes may change the text. A text
#: still changing after that is returned exactly as the server sent it --
#: also a fixed point, so idempotence holds.
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
# A ``[`` right after ``)`` is a call result being indexed (``f(x)[i](y)``)
# UNLESS that ``)`` closed a link the same scan just matched: converters emit
# adjacent ``<a>`` tags with no space (``[a](/x)[b](/y)``). A lookbehind
# cannot tell the two apart, so ``_sub_links`` makes that call.
#
# EXCEPT when the destination is absolute (``http(s)://`` or ``//``): code
# indexing never takes an absolute URL as its argument, while infobox cells
# glue a label to its link (``Preceded by[Carolyn Parrish](https://...)``,
# ``p[Canada](https://...)``) -- 3 of the 8 ``](http`` leaks left in the
# run-2 fetches (structural-economy §6). The lookahead re-reads the anchor
# with the same bound, so it adds no unbounded scan.
_GLUED_ABSOLUTE = r"(?=\[[^\[\]\n]{0,1000}\]\(\s*(?i:https?:)?//)"
_NOT_GLUED = r"(?:(?<![\w\]])|" + _GLUED_ABSOLUTE + ")"
_IMAGE = re.compile(r"!\[" + _ANCHOR + r"\]\(\s*" + _DEST + _TITLE + r"\s*\)")
_LINK = re.compile(_NOT_GLUED + r"\[" + _ANCHOR + r"\]\(\s*(" + _DEST + ")" + _TITLE + r"\s*\)")
_CITE = re.compile(r"\[\[[^\]\s]{1,12}\]\]\(#cite_note[^)\s]*\)")
_LIST_MARKER = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])(?:\s+|$)")
_ONE_LINK_ITEM = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])\s+" + _LINK.pattern + r"\s*$")
_MARKER_ONLY = re.compile(r"^\s*(?:[*+-]|\d{1,9}[.)])\s*$")
# A link or image whose target was cut by a server's chunk boundary: an
# absolute target with no closing ``)`` running to the end of the text (an
# unclosed ``(`` inside it is the ``Foo_(bar`` Wikipedia title being cut).
# 5 of the 8 ``](http`` leaks on run 2 were these (structural-economy §6);
# one of them was cut inside the scheme itself (``[Pierre Poilievre](http``),
# so a bare prefix of ``https://`` counts too. Any other short target
# (``handlers[name](arg``) is code or unknowable and keeps its bytes.
_TRUNCATED_LINK = re.compile(
    r"!?\[([^\[\]\n]{0,1000})\]\(\s*"
    r"(?:(?i:https?:)?//(?:[^()\s]|\([^()\s]*\))*(?:\([^()\s]*)?"
    r"|(?i:h(?:t(?:t(?:ps?(?::/?)?)?)?)?))\Z"
)
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")
# Absolute: ``http(s)://`` or protocol-relative ``//host/...`` (another host
# as surely as an ``https://`` link, so it is treated like one).
_ABSOLUTE = re.compile(r"^(?:https?:)?//", re.IGNORECASE)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
# On-site: resolves against the fetched page's own origin -- a root path, a
# fragment, a query or a ``javascript:`` pseudo-link. These are the targets
# of site chrome (menus, TOCs, edit links) and carry no evidence on their own.
_ON_SITE = re.compile(r"^(?:/(?!/)|#|\?|javascript:)", re.IGNORECASE)
# Nav-droppable: exactly the on-site classes the replay corpus's 395 dropped
# one-link items used (199 root ``/...``, 189 ``#...``, 7 ``javascript:``;
# QA round 2, PR #4514). ``?`` never occurred, so it is not dropped.
_NAV_TARGET = re.compile(r"^(?:/(?!/)|#|javascript:)", re.IGNORECASE)
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


def _is_file_relative(target: str) -> bool:
    """A path relative to the DOCUMENT: ``./a.md``, ``../b.md``, ``docs/c.md``.

    Unlike an on-site target, the path itself is evidence: in a README or a
    docs index it is the only record of which file holds the content (QA Q4,
    PR #4514: ``- [Install guide](docs/install.md)`` must not vanish). Only
    URL-shaped strings qualify -- a leading ``./``/``../`` or a scheme-less
    string containing ``/``. A bare word or expression (``arg``, ``*args``,
    ``\\d+``, ``foo|bar``) is not a target, so code that merely looks like
    ``[x](y)`` is never rewritten.
    """
    if target.startswith(("./", "../")):
        return True
    return (
        "/" in target
        and not _SCHEME.match(target)
        and not _ON_SITE.match(target)
        and not _ABSOLUTE.match(target)
    )


def _is_link_target(target: str) -> bool:
    """URL-shaped: what makes ``[x](y)`` a link rather than code."""
    return bool(
        _ABSOLUTE.match(target)
        or _ON_SITE.match(target)
        or _SCHEME.match(target)
        or _is_file_relative(target)
    )


def _sub_links(seg: str, repl: Callable[[re.Match[str]], str]) -> str:
    """``_LINK.sub(repl, seg)``, plus adjacent links (``[a](/x)[b](/y)``).

    A match right after ``)`` is taken only when that ``)`` ended a link this
    scan already accepted; otherwise it is code (``f(x)[i](y)``) and is left
    exactly as written. See ``_NOT_GLUED``.
    """
    last_end = -1

    def guarded(match: re.Match[str]) -> str:
        nonlocal last_end
        start = match.start()
        if (
            start
            and seg[start - 1] == ")"
            and start != last_end
            # ``(1957)[Hong Kong](https://...)``: an absolute target is a
            # link even after a ``)``, for the reason given at ``_NOT_GLUED``.
            and not _ABSOLUTE.match(match.group(2))
        ):
            return match.group(0)
        if _is_link_target(match.group(2)):
            last_end = match.end()
        return repl(match)

    return _LINK.sub(guarded, seg)


def _strip_media(line: str) -> str:
    """Images become their alt text and cite markers go (outside inline code)."""
    return _outside_inline_code(
        line, lambda seg: _CITE.sub("", _IMAGE.sub(lambda m: m.group(1).strip(), seg))
    )


def _strip_truncated_tail(line: str) -> str:
    """Cut a final ``[anchor](http...`` with no ``)`` to its anchor.

    Called only on the LAST line of a text block, and only outside code: the
    target was cut by the server's chunk boundary, so it is a broken URL the
    model can neither fetch nor cite, while the anchor is content. A
    completed link, a non-absolute target (``handlers[name](arg``) or a tail
    inside an inline code span keeps its bytes.
    """
    if "](" not in line:
        return line
    match = _TRUNCATED_LINK.search(line)
    if match is None:
        return line
    if any(start <= match.start() < end for start, end in _inline_code_spans(line)):
        return line
    return line[: match.start()] + match.group(1)


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
        and _NAV_TARGET.match(match.group(2)) is not None
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
        if _ABSOLUTE.match(target) or _ON_SITE.match(target) or _is_file_relative(target):
            targets.append(target)
            return match.group(1)
        return match.group(0)

    visible = _outside_inline_code(line, lambda seg: _sub_links(seg, as_text))
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
        if _ON_SITE.match(target):
            return anchor
        if _is_file_relative(target):
            # The path names the file holding the content, so it survives in
            # both modes; ``strip-links`` only drops the markup around it.
            if keep_prose_links:
                return f"[{anchor}]({target})"
            text = anchor.strip()
            return f"{anchor} ({target})" if text and text != target else target
        # mailto:, tel: and other schemes (the target may be the datum), and
        # anything not URL-shaped (code), stay exactly as written.
        return match.group(0)

    line = _outside_inline_code(line, lambda seg: _sub_links(seg, relink))
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
    if not text:
        return text
    if not _MARKDOWN_SIGNAL.search(text):
        # A text whose only link is one cut inside its scheme at the very end
        # (``[Pierre Poilievre](http``) carries no ``//`` signal, yet that
        # tail is still a broken link; only the last line is scanned.
        last = text.rstrip("\n").rsplit("\n", 1)[-1]
        if _strip_truncated_tail(last) == last:
            return text
    # A trailing newline is a terminator, not a blank line: keep it exactly.
    body, terminator = (text[:-1], "\n") if text.endswith("\n") else (text, "")
    runs = [
        (is_code, run, run if is_code else [_strip_media(line) for line in run])
        for is_code, run in _split_code(body.split("\n"))
    ]
    if runs and not runs[-1][0]:
        # The text's final line (not code) may end in a link the server's
        # chunk boundary cut; see ``_strip_truncated_tail``.
        last_staged = runs[-1][2]
        last_staged[-1] = _strip_truncated_tail(last_staged[-1])
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
            if _is_blank(normalized) and not out and dropped:
                # Removed leading chrome must not leave the text opening on a
                # blank line; a blank line the server sent first still stays.
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


def _is_hit(item: Any) -> bool:
    if not isinstance(item, dict) or "title" not in item:
        return False
    return any(isinstance(k, str) and _URLISH_KEY.search(k) for k in item)


def _is_hit_list(value: list[Any]) -> bool:
    # EVERY item must be a hit and at least one must carry discovery text: a
    # mixed list, or a bare list of {title, url} links, is something other
    # than a provider's ranked results and stays as the server sent it.
    return (
        bool(value)
        and all(_is_hit(item) for item in value)
        and any(isinstance(item.get(k), str) for item in value for k in SNIPPET_KEYS)
    )


def _protected(key: str) -> bool:
    return key == "title" or bool(_URLISH_KEY.search(key))


def _clip_hit(item: dict[str, Any], max_chars: int, changed: list[bool]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if _protected(key):
            out[key] = value
            continue
        if value is None or value == "" or value == [] or value == {}:
            # Empty fields carry nothing yet cost tokens on every later turn
            # (``published_date: null`` on 213/213 replayed hits).
            changed.append(True)
            continue
        if key in SNIPPET_KEYS and isinstance(value, str) and len(value) > max_chars:
            # Only an OVER-long field is rewritten, so a short snippet keeps
            # its exact bytes (``clip_snippet`` would also fold its spacing).
            value = clip_snippet(value, max_chars)
            changed.append(True)
        out[key] = value
    return out


def clip_search_hits(value: Any, max_chars: int, changed: list[bool]) -> Any:
    """Clip snippet fields and drop empty fields of every hit list in ``value``.

    A hit list may sit anywhere in the payload: top level, a ``results``
    array (Tavily, Exa) or deeper (Brave's ``web.results``). The caller gates
    on the tool being a search (``is_search_tool``); see the module docstring
    for the measurement and the constraints. Appends to ``changed`` when it
    rewrites anything, the contract ``_walk`` uses.
    """
    if isinstance(value, dict):
        return {k: clip_search_hits(v, max_chars, changed) for k, v in value.items()}
    if isinstance(value, list):
        if _is_hit_list(value):
            return [_clip_hit(item, max_chars, changed) for item in value]
        return [clip_search_hits(v, max_chars, changed) for v in value]
    return value


# --- seen-URL snippet elision --------------------------------------------


def seen_url_key(url: str) -> str:
    """The key under which a URL counts as "already shown" in a session.

    Deliberately trivial -- the fragment and trailing slashes go, nothing
    else -- because two URLs that differ in scheme, host case, ``www.`` or a
    query are not provably the same page, and a false match would hide a
    snippet the model never saw. Used only to compare; the emitted ``url``
    field is never rewritten.
    """
    return url.split("#", 1)[0].rstrip("/")


def _hit_url(item: dict[str, Any]) -> str | None:
    """A hit's URL: its ``url`` field, else its first URL-shaped string field."""
    url = item.get("url")
    if isinstance(url, str):
        return url
    for key, value in item.items():
        if isinstance(key, str) and _URLISH_KEY.search(key) and isinstance(value, str):
            return value
    return None


def elide_seen_snippets(value: Any, seen: frozenset[str], changed: list[bool]) -> Any:
    """Drop the snippet fields of every hit whose URL the session already showed.

    Why: on POC run 2, 32 of 184 search hits repeated a URL the model had
    already been shown (24 by a sibling search in the same turn, the rest by
    an earlier search or a fetch), and each repeat re-sent its snippet into
    the cache, to be re-read on every later turn. The repeat keeps its title
    and URL -- so ranking, discovery and the citation key are unchanged --
    and loses only the discovery text the model already has (a snippet is a
    lead, never a source). Replay: -0.016x Claude Code.

    ``seen`` holds ``seen_url_key`` keys; the caller (the MCP bridge's
    per-session ``SeenUrlLedger``) passes a snapshot, so this stays pure and
    a second pass over its own output with the same snapshot changes
    nothing. Only hit lists (``_is_hit_list``) are touched, under the same
    search-tool gate as ``clip_search_hits``.
    """
    if not seen:
        return value
    if isinstance(value, dict):
        return {k: elide_seen_snippets(v, seen, changed) for k, v in value.items()}
    if isinstance(value, list):
        if not _is_hit_list(value):
            return [elide_seen_snippets(v, seen, changed) for v in value]
        out: list[Any] = []
        for item in value:
            url = _hit_url(item)
            if url is not None and seen_url_key(url) in seen and SNIPPET_KEYS & item.keys():
                item = {k: v for k, v in item.items() if k not in SNIPPET_KEYS}
                changed.append(True)
            out.append(item)
        return out
    return value


def _strict_json(text: str) -> Any:
    """``text`` parsed the way ``_normalize_once`` parses it, or ``None``."""
    if text.lstrip()[:1] not in ("{", "["):
        return None
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_float=_exact_float,
            parse_constant=_reject_constant,
        )
    except (_DuplicateKey, _LossyNumber, ValueError, RecursionError):
        return None


def _is_page_payload(value: Any) -> bool:
    """A single fetched web page: ``{"url": <absolute http(s)>, "content": str, ...}``."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("content"), str)
        and isinstance(value.get("url"), str)
        and _ABSOLUTE.match(value["url"]) is not None
    )


def shown_urls(text: str, *, tool_name: str = "") -> list[str]:
    """The URLs a result shows the model, in order: hit URLs and a page's URL.

    Read from the server's text (before normalization). Hit lists count only
    from a search tool (``is_search_tool``), matching where elision applies;
    a fetched page's ``url`` counts from any tool. The bridge records only the
    ones that survive into the text the model actually receives (a spill can
    cut a tail off).
    """
    parsed = _strict_json(text)
    if parsed is None:
        return []
    urls: list[str] = []
    if _is_page_payload(parsed):
        urls.append(parsed["url"])
    if is_search_tool(tool_name):

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                if _is_hit_list(value):
                    urls.extend(u for item in value if (u := _hit_url(item)) is not None)
                else:
                    for child in value:
                        visit(child)

        visit(parsed)
    return urls


# --- boilerplate section elision ------------------------------------------

#: Section titles that hold a page's apparatus, not its facts. Exactly the
#: structural-economy §5 list (plus the common combined Wikipedia heading).
#: On run 2 no cited fact (0/41) sat in one of these sections. Broader lists
#: ("Related", "Share", newsletter blocks) were not measured, so they are not
#: here. Nav-chrome LINES are never dropped: that rule lost a cited fact
#: ("Minister of Foreign Affairs", rec-005) and was rejected.
BOILERPLATE_SECTION_TITLES: Final = frozenset(
    {"references", "see also", "external links", "notes", "further reading", "notes and references"}
)
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")


def _heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING.match(line)
    if match is None:
        return None
    title = (match.group(2) or "").rstrip("#").strip().rstrip(":").strip()
    return len(match.group(1)), " ".join(title.split()).lower()


def elide_boilerplate_sections(text: str) -> str:
    """Drop References / See also / External links / Notes / Further reading.

    Each such ATX section is removed from its heading up to (not including)
    the next heading of the same or a higher level, so a subsection's own
    siblings and the following top-level section survive. Headings inside
    code (fenced or indented, ``_split_code``) are not headings. Applied only
    to the ``content`` of a fetched page payload (``_is_page_payload``): a
    README, an issue body or a docs tool's "## Notes" is the author's text,
    not a web page's apparatus. Measured worth: -0.002 to -0.004x Claude Code
    on run 2 (the forge server's 12,000-char chunks rarely reach a page's
    tail); kept because it is evidence-safe and free in the same pass.
    """
    if "#" not in text:
        return text
    body, terminator = (text[:-1], "\n") if text.endswith("\n") else (text, "")
    lines = body.split("\n")
    headings: list[tuple[int, str] | None] = []
    for is_code, run in _split_code(lines):
        headings.extend(None if is_code else _heading(line) for line in run)
    kept: list[str] = []
    i = 0
    dropped_tail = False
    while i < len(lines):
        heading = headings[i]
        if heading is not None and heading[1] in BOILERPLATE_SECTION_TITLES:
            level = heading[0]
            i += 1
            while i < len(lines) and not ((h := headings[i]) is not None and h[0] <= level):
                i += 1
            dropped_tail = i == len(lines)
            continue
        kept.append(lines[i])
        i += 1
    if len(kept) == len(lines):
        return text
    if dropped_tail:
        # A tail section leaves the blank lines that separated it.
        while kept and _is_blank(kept[-1]):
            kept.pop()
    return "\n".join(kept) + terminator


# --- JSON -> text rendering -------------------------------------------------
#
# Why: a JSON result spends tokens on keys, quotes and escapes -- the run-2
# fetched pages alone carried 3,827 escaped ``\n`` -- and every one is
# written to the cache and re-read each later turn. Rendering the two
# dominant shapes as text saved 6.0% (fetch) / 3.2% (search) of their tokens
# on the replay (-0.017x Claude Code) with every field kept.
#
# Grammar (the inverse is mechanical; ``tests/unit/test_tool_result_normalize.py``
# carries a reference decoder that round-trips every case):
#
# * SCALAR ``enc(v)``: a string is written raw when it is non-empty, has no
#   leading/trailing whitespace, no ``\n``/``\r``, no lone surrogate, and does
#   NOT parse as JSON (so ``"0"``, ``"true"``, ``"null"``, ``"[1]"`` are not
#   raw). Everything else -- other strings, numbers, booleans, null, nested
#   lists and objects -- is compact single-line JSON. Decode: ``json.loads``
#   it; if that fails it is the raw string.
# * SLOT ``slot(v)``: ``enc(v)``, but a raw string that looks like a field
#   line (``key: ...``) is JSON-encoded instead, so slots and fields cannot
#   be confused.
# * Search hit list (search tools only; ``_is_rendered_hit_list``: every item
#   a hit, with or without a snippet left, and every hit carrying ``title``
#   and ``url``):
#       <key>: enc(value)        each top-level field, in order
#       <hit-key>:               the ONE top-level hit list, at its position
#       N. enc(title)            hit N (1-based)
#          slot(url)             3-space indent, always present
#          slot(snippet)         only when the hit has a ``snippet`` key
#          <key>: enc(value)     every other hit field, in order
#   A top-level hit list (no wrapping object) is just the ``N.`` blocks.
#   Decode: a column-0 ``key: v`` is a field, a bare ``key:`` opens the hit
#   list; ``N. `` opens a hit; its first indented line is the url, a second
#   one that is not ``key: v`` is the snippet, the rest are fields.
# * Fetched page (``{"url": str, "content": str, ...}``, any tool):
#       <key>: enc(value)        every field except ``content``, in order
#       (blank line)
#       <content, raw>           to the end of the text
#   Decode: split at the first blank line (header values are single-line and
#   never empty).
#
# Losslessness is JSON-value equality: object key ORDER may differ (a hit's
# ``title``/``url``/``snippet`` come first, a page's ``content`` last), which
# RFC 8259 objects do not carry as data. Keys must look like identifiers
# (``_FIELD_KEY``); any other shape -- nested hit lists (Brave's
# ``web.results``), two hit lists, non-search lists such as GitHub issues,
# arbitrary JSON -- stays JSON.
#
# Constraints: ``url`` values appear byte-equal (a URL is always raw unless
# it would be ambiguous). A rendering that the markdown pass would itself
# rewrite (a short title still holding image markup) is not used, so
# ``normalize_tool_text`` stays idempotent by construction. Gated by
# ``mcp.result_text_rendering`` (A/B switch) and never under ``off``.

#: ``config.yml`` path and registry default of ``mcp.result_text_rendering``;
#: read by ``mcp.manager.read_result_text_rendering``.
RESULT_TEXT_RENDERING_PATH: Final = ("mcp", "result_text_rendering")
RESULT_TEXT_RENDERING_DEFAULT: Final = True

_FIELD_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_FIELD_LINE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*:(?: |\Z)")
_HIT_INDENT: Final = "   "


def _is_plain(value: str) -> bool:
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        return False
    if _LONE_SURROGATE.search(value):
        return False
    try:
        json.loads(value)
    except ValueError:
        return True
    except RecursionError:
        return False
    return False


def _json_scalar(value: Any) -> str:
    out = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return _LONE_SURROGATE.sub(_escape_surrogate, out)


def _enc(value: Any) -> str:
    if isinstance(value, str) and _is_plain(value):
        return value
    return _json_scalar(value)


def _slot(value: Any) -> str:
    out = _enc(value)
    return _json_scalar(value) if _FIELD_LINE.match(out) else out


def _render_hits(hits: list[dict[str, Any]]) -> list[str] | None:
    lines: list[str] = []
    for number, hit in enumerate(hits, 1):
        if "title" not in hit or "url" not in hit:
            return None
        extras = [k for k in hit if k not in ("title", "url", "snippet")]
        if not all(_FIELD_KEY.match(k) for k in extras):
            return None
        lines.append(f"{number}. {_enc(hit['title'])}")
        lines.append(_HIT_INDENT + _slot(hit["url"]))
        if "snippet" in hit:
            lines.append(_HIT_INDENT + _slot(hit["snippet"]))
        lines.extend(f"{_HIT_INDENT}{key}: {_enc(hit[key])}" for key in extras)
    return lines


def _is_rendered_hit_list(value: Any) -> bool:
    # ``_is_hit_list`` minus its "some hit carries a snippet" clause: once
    # seen-URL elision has dropped every snippet (a repeated search), the list
    # is still the search's ranked hits and renders the same way (found on the
    # real-path run: an all-seen repeat search fell back to JSON).
    return isinstance(value, list) and bool(value) and all(_is_hit(item) for item in value)


def _render_search(parsed: Any) -> str | None:
    if isinstance(parsed, list):
        hits = _render_hits(parsed) if _is_rendered_hit_list(parsed) else None
        return None if hits is None else "\n".join(hits)
    if not isinstance(parsed, dict) or not all(_FIELD_KEY.match(k) for k in parsed):
        return None
    hit_keys = [k for k, v in parsed.items() if _is_rendered_hit_list(v)]
    if len(hit_keys) != 1:
        return None
    lines: list[str] = []
    for key, value in parsed.items():
        if key == hit_keys[0]:
            hits = _render_hits(value)
            if hits is None:
                return None
            lines.append(f"{key}:")
            lines.extend(hits)
        else:
            lines.append(f"{key}: {_enc(value)}")
    return "\n".join(lines)


def _render_page(parsed: Any) -> str | None:
    if not (
        isinstance(parsed, dict)
        and isinstance(parsed.get("url"), str)
        and isinstance(parsed.get("content"), str)
        and all(_FIELD_KEY.match(k) for k in parsed)
    ):
        return None
    content = parsed["content"]
    if _LONE_SURROGATE.search(content):
        return None  # raw, it could not be sent as UTF-8; JSON keeps the escape
    header = [f"{k}: {_enc(v)}" for k, v in parsed.items() if k != "content"]
    return "\n".join(header) + "\n\n" + content


def render_result_text(text: str, *, tool_name: str = "", keep_prose_links: bool = True) -> str:
    """Render a hit-list or fetched-page JSON result as text; else ``text``.

    See the grammar above. ``text`` is the already-normalized JSON; the
    result is used only when it is a fixed point of the markdown pass, so a
    second ``normalize_tool_text`` over it changes nothing.
    """
    parsed = _strict_json(text)
    if parsed is None:
        return text
    rendered = _render_search(parsed) if is_search_tool(tool_name) else None
    if rendered is None:
        rendered = _render_page(parsed)
    if rendered is None or rendered.lstrip()[:1] in ("{", "["):
        return text
    if normalize_markdown(rendered, keep_prose_links=keep_prose_links) != rendered:
        return text
    return rendered


def _normalize_once(
    text: str,
    keep_prose_links: bool,
    search_snippet_chars: int = 0,
    seen: frozenset[str] = frozenset(),
) -> str:
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
            if isinstance(parsed, dict) and _is_page_payload(parsed):
                # Before the markup walk, so a dropped section's links are
                # never worked on and the walk tidies the blank lines left.
                page: dict[str, Any] = parsed
                content = elide_boilerplate_sections(page["content"])
                if content != page["content"]:
                    parsed = {**page, "content": content}
                    changed.append(True)
            walked = _walk(parsed, keep_prose_links, changed)
            if search_snippet_chars > 0:
                # After the markup walk, as the replay measured it: links and
                # images leave a snippet first, so the cap keeps more words.
                walked = clip_search_hits(walked, search_snippet_chars, changed)
            # ``seen`` is empty unless the tool is a search (the caller gates).
            walked = elide_seen_snippets(walked, seen, changed)
            if not changed:
                return text  # nothing to remove: keep the server's exact bytes
            out = json.dumps(walked, ensure_ascii=False, separators=(",", ":"))
            return _LONE_SURROGATE.sub(_escape_surrogate, out)
    return normalize_markdown(text, keep_prose_links=keep_prose_links)


def normalize_tool_text(
    text: str,
    mode: str = RESULT_NORMALIZATION_DEFAULT,
    *,
    tool_name: str = "",
    search_snippet_chars: int = 0,
    seen_urls: frozenset[str] = frozenset(),
    text_rendering: bool = False,
) -> str:
    """Normalize one tool result's flattened text under ``mode``.

    ``off`` returns ``text`` unchanged (the identity, byte for byte). A JSON
    object or array is walked structurally and re-emitted compactly, but only
    when a string leaf actually changed; JSON with duplicate keys or a number
    that would not re-emit exactly is returned unchanged; anything else takes
    the markdown path. The transform is repeated to a fixed point (at most
    ``MAX_PASSES``), and an output that would not encode as UTF-8 falls back
    to ``text``.

    ``tool_name`` is the tool's bare MCP name; when it names a search
    (``is_search_tool``) and ``search_snippet_chars`` is positive, the JSON
    hit lists it returns are clipped by ``clip_search_hits`` in the same pass.
    ``off`` skips that too: it is the byte-identical escape hatch.

    ``seen_urls`` (``seen_url_key`` keys, a snapshot of the session's
    ``SeenUrlLedger``) drops the snippets of a search's hits the model was
    already shown (``elide_seen_snippets``). A fetched page payload loses its
    boilerplate sections (``elide_boilerplate_sections``). ``text_rendering``
    (``mcp.result_text_rendering``) renders a hit list or a fetched page as
    text after the fixed point (``render_result_text``). All three are pure
    given their inputs, so the result is still deterministic and
    ``f(f(x)) == f(x)`` for the same ``seen_urls``. The defaults here are the
    PR #4525 behaviour; the MCP manager passes the live settings.
    """
    if mode == "off" or not text:
        return text
    if mode not in RESULT_NORMALIZATION_MODES:
        raise ValueError(f"unknown result normalization mode: {mode!r}")
    keep_prose_links = mode == "prose-links"
    search = is_search_tool(tool_name)
    snippet_chars = search_snippet_chars if search else 0
    seen = seen_urls if search else frozenset()
    current = text
    for _ in range(MAX_PASSES):
        following = _normalize_once(current, keep_prose_links, snippet_chars, seen)
        if following == current:
            break
        current = following
    else:
        return text
    if text_rendering:
        # After the fixed point, never inside it: the rendered text is not
        # JSON, so a later pass takes the markdown path, and
        # ``render_result_text`` only returns a text that path leaves alone.
        current = render_result_text(
            current, tool_name=tool_name, keep_prose_links=keep_prose_links
        )
    if current is not text:
        try:
            current.encode("utf-8")
        except UnicodeEncodeError:
            return text
    return current
