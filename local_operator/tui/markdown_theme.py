"""Island markdown styling — the markdown ramp for the brand kit (D1/D2).

Two layers. The colour decisions themselves — ``element -> token`` — live in
:mod:`local_operator.tui.bindings` (§2 of ``docs/proposals/theme-colour-
budget.md``); this module wires that table into rich's markdown and syntax
protocols:

- :data:`brand_markdown_theme` — a rich ``Theme`` built from
  :func:`bindings.markdown_theme`, mapping rich-markdown's element styles
  onto the island ramp (headings bold fg, code string-green with no
  background slab, quotes muted, bullets dim, links `signal`). The app
  pushes it onto ``app.console`` once.
- :class:`IslandCodeBlock` — code fences rendered on the island ground with
  brand syntax colors and zero padding, replacing rich's default Monokai
  slab (D2: the slab was the loudest chrome in the transcript).

Installation is idempotent (:func:`install_markdown_theme`); it swaps the
``Markdown`` element table in place and flattens headings to bare styled
text (no panel, no rule line — structure comes from weight and tint).
"""

from __future__ import annotations

from typing import cast

from pygments.token import Token, _TokenType
from rich.markdown import CodeBlock, Heading, Markdown
from rich.style import Style
from rich.syntax import Syntax, SyntaxTheme, TokenType
from rich.text import Text
from rich.theme import Theme

from local_operator.tui import bindings

#: Root of the pygments token hierarchy — the terminal fallback for a token
#: whose whole ancestry is absent from the ramp.
_TOKEN_ROOT = Token


class IslandSyntaxTheme(SyntaxTheme):
    """Code colors built from the island ramp via :mod:`bindings`: keywords
    muted, string green, number/warning amber, comment dim. Replaces the
    Monokai slab palette."""

    def __init__(self) -> None:
        self._styles: dict[_TokenType, Style] = bindings.syntax_styles()

    def get_style_for_token(self, token_type: TokenType) -> Style:
        """Walk up the token hierarchy to the nearest ramp entry.

        Pygments token types are tuples that carry a ``parent``, so a token
        the ramp does not name (``Name.Function.Magic``) resolves to the
        nearest ancestor that it does (``Name.Function``) instead of falling
        all the way to the base ink.

        The parameter is rich's ``TokenType`` (a plain ``tuple[str, ...]``)
        because that is what the abstract method declares; pygments' richer
        ``_TokenType`` is the concrete class that actually arrives, and it is
        the one carrying ``parent``.
        """
        node = cast("_TokenType | None", token_type)
        while node:
            style = self._styles.get(node)
            if style is not None:
                return style
            node = node.parent
        return self._styles[_TOKEN_ROOT]

    def get_background_style(self) -> Style:  # type: ignore[override]
        return Style(bgcolor=bindings.ground_hex("code.background"))


class IslandCodeBlock(CodeBlock):
    """Code fence on the island ground with brand syntax colors, no slab."""

    def __rich_console__(self, console, options):  # type: ignore[override]
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme=IslandSyntaxTheme(),
            word_wrap=True,
            padding=0,
            background_color=bindings.ground_hex("code.background"),
        )


def brand_markdown_theme() -> Theme:
    """Element-style theme for rich Markdown, resolved against the current
    ramp. Rebuild after a theme switch (epoch change).

    The element -> token decisions (including every design rationale — the
    h1 accent, the `hr` rebind, the `signal` inline-code choice) live in
    :mod:`local_operator.tui.bindings` (``docs/proposals/theme-colour-
    budget.md`` §2); see :func:`bindings.markdown_theme` for the note text.
    """
    return bindings.markdown_theme()


_installed = False


def install_markdown_theme() -> None:
    """Swap Markdown's element table to the island rendering (idempotent)."""
    global _installed
    if _installed:
        return
    Markdown.elements = {
        **Markdown.elements,
        "fence": IslandCodeBlock,
        "code_block": IslandCodeBlock,
    }

    def _flat_heading(self: Heading, console, options):  # type: ignore[no-untyped-def]
        text = self.text
        text.justify = "left"
        # Leading ABOVE a heading, scaled by level. Every gap in a rendered
        # answer measured 48.8px — one uniform rhythm from h1 to body prose —
        # so when two levels shared an ink there was no second channel holding
        # the structure up, and a long reply read as one undifferentiated
        # stripe. Space is the cheapest hierarchy available here: it competes
        # with no token, spends nothing from the accent budget, and works
        # identically in all 54 themes because it is not a colour at all.
        #
        # rich yields headings inside a stream of blocks, so a leading blank
        # line is the only lever — there is no margin box to set. h1/h2 get
        # one; h3 and below get none, because a sub-step that pushes air is
        # louder than the section it sits under, and the tail of a document
        # is where density matters most.
        if self.tag in ("h1", "h2"):
            yield Text("")
        yield text

    Heading.__rich_console__ = _flat_heading  # type: ignore[method-assign]
    _installed = True
