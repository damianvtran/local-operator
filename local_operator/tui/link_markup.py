"""Turn a bare URL in markdown source into an autolink, so Rich paints it.

WHY THIS EXISTS
===============

A URL written plainly — ``https://example.com/x`` — is not a link to Rich's
markdown renderer. It sets ``Style(link=...)`` for ``[label](target)`` and for
the explicit autolink ``<target>``, and NOTHING for the bare form, which is the
shape users and models write most. Measured in this tree against rich 15.0.0: a
message holding a markdown link, an autolink and a bare URL produced two link
spans, and the bare one was neither blue, nor underlined, nor clickable.

That is the half of "links don't work in the TUI" that no click handler can
fix, because there is nothing under the pointer to find: the click resolution
in ``TranscriptBlock._link_at`` reads the style of the clicked cell, and a cell
with no link style is indistinguishable from prose. So the bare URL is promoted
to an autolink HERE, before the markdown is rendered, and every downstream
consumer — the painted style, the OSC-8 escape, the click — gets it for free
from the one renderer that already knows how to do all three.

Rich also splits a link span across a fold and repeats the FULL url on both
halves (verified: a 101-character URL folded at 80 columns produced two spans,
both carrying the whole target). So a wrapped URL is clickable on either row,
and this module does not have to reason about width at all.

WHAT IS LEFT ALONE
==================

The transformation is applied to the markdown SOURCE, so anything where a
``<...>`` would change meaning — or where the text is deliberately not prose —
is masked out first:

* **Fenced code** (``` and ``~~~``) and **inline code spans**. A URL in a shell
  command is an argument being shown, not an address being offered; wrapping it
  would put literal angle brackets in the sample, which is exactly what rich
  renders inside code (measured: ``` `curl <https://a.test/x>` ``` painted the
  brackets). Copying the sample would then paste something that does not run.
* **Existing links** — ``[label](target)``, ``<target>`` and reference
  definitions (``[label]: target``). Their targets are already links; a second
  pass over them would nest brackets inside a target and break the parse.

THE URL RULE IS NOT RESTATED HERE
=================================

:func:`~local_operator.tui.link_targets._body_end` and
:func:`~local_operator.tui.link_targets._trim` are imported rather than
respelled. ``/links`` and this module MUST agree about where a URL ends: if the
picker opened ``…/docs`` and a click opened ``…/docs.`` the two routes to the
same link would go to different pages, and the trailing-period case is the
common one in prose. One rule, two callers, as the extraction module's own
docstring argues for its two heads.

``_body_end`` is a SCAN rather than a pattern, and importing it is what buys
this module the cases a regex cannot state: arbitrarily nested parentheses in a
target, an IPv6 literal's ``]``, and a redirector carrying a second URL in its
query string. Re-deriving any of that here would be a second implementation of
the rule whose whole point is that there is only one.
"""

from __future__ import annotations

import re

from local_operator.tui.link_targets import _URL_START_RE, _body_end, _trim, is_openable

__all__ = ["autolink_bare_urls"]

#: Opening/closing fence for a code block. Matched with the same 0-3 space
#: indent allowance ``widgets/assistant._FENCE_RE`` uses, so the two agree
#: about which rows are code.
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: An inline code span: a backtick RUN, closed by a run of the same length.
#: Written as a backreference so ``` ``a ` b`` ``` is one span rather than two
#: — the rule CommonMark states, and the one a naive single-backtick pattern
#: gets wrong on any sample containing a backtick.
_INLINE_CODE_RE = re.compile(r"(`+)(?:.*?)\1", re.DOTALL)

#: ``[label](target)`` / ``![alt](target)`` — the whole construct, so neither
#: the label nor the target is offered to the bare pattern.
_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)\s]*(?:\s+[^)]*)?\)")

#: An existing autolink. Already a link; masked so it is not re-wrapped.
_AUTOLINK_RE = re.compile(r"<[^<>\s]+>")

#: A reference definition line: ``[label]: target``. The target is a link
#: target already, and wrapping it would break the definition.
_REF_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by fenced code blocks.

    Walks lines with fence state rather than pairing markers by regex: a
    ``~~~`` block may contain a ``` ``` ``` line and vice versa, and only the
    marker that OPENED the block can close it. An unterminated fence — a
    streaming message whose closing marker has not arrived — runs to the end of
    the text, which is the reading that keeps a half-arrived code sample from
    being autolinked mid-flush and then un-autolinked when it completes.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    start: int | None = None
    marker = ""
    for line in text.splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        if start is None:
            if match is not None:
                start = offset
                marker = match.group(1)
        elif match is not None:
            run = match.group(1)
            if run[0] == marker[0] and len(run) >= len(marker):
                spans.append((start, offset + len(line)))
                start = None
                marker = ""
        offset += len(line)
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _masked(text: str) -> list[tuple[int, int]]:
    """Every range a bare URL must NOT be promoted inside.

    Fences are collected first and the inline/link patterns are then filtered
    against them: a ``[a](b)`` inside a code sample is not a link, and letting
    its pattern contribute a span there would be harmless but is also
    meaningless. The list is returned sorted so :func:`autolink_bare_urls` can
    test membership with a single walk.
    """
    fences = _fenced_spans(text)

    def _in_fence(start: int) -> bool:
        return any(low <= start < high for low, high in fences)

    spans = list(fences)
    for pattern in (_INLINE_CODE_RE, _MD_LINK_RE, _AUTOLINK_RE, _REF_DEF_RE):
        for match in pattern.finditer(text):
            if not _in_fence(match.start()):
                spans.append((match.start(), match.end()))
    return sorted(spans)


def autolink_bare_urls(text: str) -> str:
    """``text`` with every bare http(s) URL wrapped in ``<...>``.

    Idempotent: a URL this function has already wrapped is matched by
    :data:`_AUTOLINK_RE` on the next pass and masked, so re-rendering the same
    source — which the streaming path does on every flush, and the resize path
    does per width — cannot accumulate brackets.

    Only http/https are promoted, by :func:`~local_operator.tui.link_targets.
    is_openable`. The scheme guard is not really load-bearing here (the click
    boundary re-checks it before anything reaches a browser, and
    :data:`_URL_BODY` only matches those two schemes anyway) but it keeps this
    module from being the place a third scheme quietly becomes clickable.

    The trailing punctuation that belongs to the SENTENCE is left outside the
    brackets: ``see https://a.test/x.`` becomes ``see <https://a.test/x>.`` so
    the rendered row still reads as a sentence and the link still resolves.
    """
    if "://" not in text:
        return text
    masked = _masked(text)
    out: list[str] = []
    cursor = 0
    scan = 0
    while True:
        match = _URL_START_RE.search(text, scan)
        if match is None:
            break
        start = match.start()
        body_to = _body_end(text, match.end())
        if body_to == match.end():
            # ``https://`` with nothing a URL can be made of after it.
            scan = match.end()
            continue
        # Resume past the BODY, not past the scheme, so a redirector's inner
        # URL (``…/r?u=https://b.test/y``) is part of the link it sits in
        # rather than a second link found inside it — the rule
        # :func:`extract_links` states for its own bare pass.
        scan = body_to
        if start < cursor:
            continue
        if any(low <= start < high for low, high in masked):
            continue
        url = _trim(text[start:body_to])
        if not url or not is_openable(url):
            continue
        # `_trim` may have dropped a trailing `.` or `**`; those characters
        # belong to the prose and stay OUTSIDE the brackets, so the rendered
        # sentence is unchanged and only the address is linked.
        out.append(text[cursor:start])
        out.append(f"<{url}>")
        cursor = start + len(url)
    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)
