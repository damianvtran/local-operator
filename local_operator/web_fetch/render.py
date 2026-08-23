"""Body → readable-text rendering for web_fetch.

Four jobs, in one place so the tool, the CLI, and tests share exactly one
classification and one degraded path:

- HTML → markdown, via ``markdownify`` (the ``[fetch]`` extra) when present and
  a pure-stdlib ``html.parser`` fallback when it is not. The fallback is the
  contract that keeps a BARE ``pip install local-operator`` fully functional
  (see :mod:`local_operator.optional` and pyproject's base-deps note): we degrade
  SILENTLY rather than raising, because a usable-if-crude result beats an error.
- JSON → pretty-print, falling back to raw text on malformed JSON.
- Markdown / plain text → whitespace-normalized pass-through.
- Binary (PDF/image/other) → a one-line notice; bytes are never inlined.

The low-quality gate flags — never drops — output that is too short, mostly
navigation chrome, or JS-gated, so the card and the agent can decide whether to
reach for ``browser`` instead.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

from local_operator.web_fetch.models import RenderMethod

#: Subtrees whose text is chrome, not content. Dropped by BOTH renderers so the
#: markdown the model reads is the page, not its navigation. ``<template>`` and
#: ``<svg>`` are here too because their inner text is markup noise.
_STRIP_TAGS = frozenset(
    {"script", "style", "nav", "header", "footer", "aside", "template", "svg", "noscript"}
)

#: Tags that open and close a markdown block. Kept crude on purpose: the stdlib
#: renderer is the fallback, not the good path, and a faithful HTML→markdown
#: converter is exactly what the ``[fetch]`` extra buys.
_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "br", "tr", "table", "ul", "ol", "blockquote"}
)

#: Below this the render is treated as "sparse" — a shell page whose body is
#: JS-loaded, or a redirect stub. Matches omp's isLowQualityOutput threshold.
_LOW_QUALITY_MIN_CHARS = 100

#: A page that is mostly one- or two-word lines is a menu, not prose. Above this
#: ratio of short lines the output is flagged (never dropped) as navigation-heavy.
_SHORT_LINE_RATIO = 0.7
_SHORT_LINE_MAX_WORDS = 3

#: Phrases a JS-gated shell prints where its content would be. Presence flags the
#: result so the agent knows ``browser`` (a real JS session) may be required.
_JS_GATED_MARKERS = (
    "enable javascript",
    "please enable js",
    "requires javascript",
    "javascript is required",
    "turn on javascript",
)

_WHITESPACE_RUN = re.compile(r"[ \t]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")


def _normalize_whitespace(text: str) -> str:
    """Collapse intra-line whitespace runs and cap blank-line runs at one.

    Rendered markdown from any backend accumulates ragged spacing (indentation
    from the source, doubled blank lines between blocks). Normalizing here keeps
    the spilled copy compact so the per-entry spill cap holds more real content.
    """
    lines = [_WHITESPACE_RUN.sub(" ", line).rstrip() for line in text.splitlines()]
    joined = "\n".join(lines)
    return _BLANK_LINE_RUN.sub("\n\n", joined).strip()


class _StdlibMarkdownParser(HTMLParser):
    """A deliberately-crude HTML→markdown converter built on the stdlib.

    This is the fallback that keeps web_fetch working on a bare install where
    ``markdownify`` is absent. It is NOT trying to compete with markdownify — it
    strips the chrome subtrees, keeps heading/list/anchor/pre structure as rough
    markdown, and collapses everything else to text. "Beats raw HTML and beats
    failing" is the whole bar it has to clear (design §5).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        # Depth of the current stripped subtree: while > 0, all data is chrome
        # (script/style/nav/…) and discarded. A depth counter rather than a bool
        # so nested strip tags (a <nav> inside a <header>) close correctly.
        self._strip_depth = 0
        # Inside <pre>/<code> verbatim text must survive whitespace collapsing,
        # so data handling branches on this flag.
        self._pre_depth = 0
        self._list_stack: list[str] = []  # "ul"/"ol" to pick the bullet marker

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _STRIP_TAGS:
            self._strip_depth += 1
            return
        if self._strip_depth:
            return
        if tag in ("pre", "code"):
            self._pre_depth += 1
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("ul", "ol"):
            self._list_stack.append(tag)
            self._out.append("\n")
        elif tag == "li":
            marker = "1. " if (self._list_stack and self._list_stack[-1] == "ol") else "- "
            self._out.append("\n" + marker)
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                # Markdown link syntax opens here and closes on the end tag; the
                # href is stashed so the closing tag can emit ](url).
                self._out.append("[")
                self._pending_href = href
        elif tag in _BLOCK_TAGS:
            self._out.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _STRIP_TAGS:
            self._strip_depth = max(0, self._strip_depth - 1)
            return
        if self._strip_depth:
            return
        if tag in ("pre", "code"):
            self._pre_depth = max(0, self._pre_depth - 1)
        if tag in ("ul", "ol") and self._list_stack:
            self._list_stack.pop()
        if tag == "a":
            href = getattr(self, "_pending_href", None)
            if href is not None:
                self._out.append(f"]({href})")
                self._pending_href = None

    def handle_data(self, data: str) -> None:
        if self._strip_depth:
            return
        if self._pre_depth:
            self._out.append(data)
        elif data.strip():
            self._out.append(data)

    def result(self) -> str:
        return _normalize_whitespace("".join(self._out))


def _render_html_stdlib(html: str) -> str:
    parser = _StdlibMarkdownParser()
    parser.feed(html)
    parser.close()
    return parser.result()


def _render_html_markdownify(html: str) -> str:
    """Render HTML with markdownify after stripping chrome subtrees via bs4.

    Kept behind a local import so the module imports on a bare install: this
    function is only ever called after :func:`html_backend_available` confirmed
    the extra is importable. markdownify's own ``strip`` list drops script/style,
    but nav/header/footer carry real text it would otherwise keep, so bs4
    decomposes those first.
    """
    from bs4 import BeautifulSoup  # local: only on the extra-present path
    from markdownify import markdownify  # local: only on the extra-present path

    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(list(_STRIP_TAGS)):
        element.decompose()
    # ATX headings (``# H1``) read better in a terminal than the underline style
    # markdownify defaults to, and match what the stdlib fallback emits so the
    # two backends produce comparable shapes.
    rendered = markdownify(str(soup), heading_style="ATX")
    return _normalize_whitespace(rendered)


def html_backend_available() -> bool:
    """Whether the ``[fetch]`` extra (markdownify + bs4) can be imported.

    Computed by attempting the import rather than checking an installed-extras
    manifest: the render path degrades on the actual ability to import, which is
    what a bare install or a partially-installed environment really exposes.
    """
    try:
        import bs4  # noqa: F401
        import markdownify  # noqa: F401
    except ImportError:
        return False
    return True


def render_html(html: str, *, force_stdlib: bool = False) -> tuple[str, RenderMethod]:
    """Render HTML to markdown, returning ``(text, method)``.

    ``force_stdlib`` reproduces the bare-install path even when the extra is
    present (config ``render_backend="stdlib"`` and tests). Otherwise markdownify
    is used when importable and the stdlib parser is the SILENT fallback — an
    ImportError here degrades, it never surfaces to the caller, because the whole
    point of the extra is that its absence is invisible to a user.
    """
    if not force_stdlib and html_backend_available():
        try:
            return _render_html_markdownify(html), "markdownify"
        except Exception:
            # A malformed page that trips markdownify/bs4 still has to yield
            # SOMETHING: fall through to the stdlib parser rather than failing
            # the fetch. Parity with optional.py's "sites that can degrade keep
            # their own try/except" rule.
            pass
    return _render_html_stdlib(html), "stdlib"


def render_json(body: str) -> tuple[str, RenderMethod]:
    """Pretty-print JSON; fall back to raw text when it does not parse.

    Mirrors omp's formatJson: a body served as ``application/json`` that is not
    actually valid JSON (an error page with the wrong header, a truncated
    stream) must not error — it passes through as text.
    """
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return _normalize_whitespace(body), "text"
    return json.dumps(parsed, indent=2, ensure_ascii=False), "json"


def render_text(body: str) -> tuple[str, RenderMethod]:
    """Whitespace-normalized pass-through for markdown / plain text."""
    return _normalize_whitespace(body), "text"


def binary_notice(content_type: str, byte_count: int, final_url: str) -> tuple[str, RenderMethod]:
    """A one-line notice for a binary body (PDF/image/other): type, size, URL.

    Binary bytes are never inlined — they would blow context and mean nothing to
    the model. The notice tells the agent what it is and to use ``browser`` for a
    PDF/image it must actually see. Matches omp's buildBinaryNotice.
    """
    size = _human_bytes(byte_count)
    kind = content_type.split(";", 1)[0].strip() or "application/octet-stream"
    return (
        f"[{kind} · {size}] {final_url}\n"
        "Binary content is not inlined. Use `browser` to view it, or download it "
        "with bash if you need the bytes."
    ), "binary"


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} B"


def is_low_quality(text: str) -> bool:
    """Whether rendered ``text`` looks sparse, navigation-heavy, or JS-gated.

    Advisory only — the caller keeps the content and merely flags it, so a false
    positive costs a hint, not data. Three cheap signals (design §6 step 6):
    too short, dominated by short lines, or carrying a JS-gate marker.
    """
    stripped = text.strip()
    if len(stripped) < _LOW_QUALITY_MIN_CHARS:
        return True
    lowered = stripped.lower()
    if any(marker in lowered for marker in _JS_GATED_MARKERS):
        return True
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return True
    short = sum(1 for line in lines if len(line.split()) <= _SHORT_LINE_MAX_WORDS)
    return short / len(lines) > _SHORT_LINE_RATIO
