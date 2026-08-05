"""Island markdown styling — the markdown ramp for the brand kit (D1/D2).

Two layers, both sourced from the theme's semantic tokens:

- :data:`brand_markdown_theme` — a rich ``Theme`` mapping rich-markdown's
  element styles onto the island ramp (headings bold fg, code string-green
  with no background slab, quotes muted, bullets dim, links accent). The app
  pushes it onto ``app.console`` once.
- :class:`IslandCodeBlock` — code fences rendered on the island ground with
  brand syntax colors and zero padding, replacing rich's default Monokai
  slab (D2: the slab was the loudest chrome in the transcript).

Installation is idempotent (:func:`install_markdown_theme`); it swaps the
``Markdown`` element table in place and flattens headings to bare styled
text (no panel, no rule line — structure comes from weight and tint).
"""

from __future__ import annotations

from rich.markdown import CodeBlock, Heading, Markdown
from rich.style import Style
from rich.syntax import Syntax, SyntaxTheme
from rich.theme import Theme

from local_operator.tui import theme as theme_mod

_C = theme_mod.semantic_color


class IslandSyntaxTheme(SyntaxTheme):
    """Code colors built from the island ramp: keyword accent, string green,
    number/warning amber, comment dim. Replaces the Monokai slab palette."""

    def __init__(self) -> None:
        from pygments.token import (
            Comment,
            Error,
            Generic,
            Keyword,
            Name,
            Number,
            Operator,
            Punctuation,
            String,
            Token,
        )

        self._styles = {
            Token: Style(color=_C("fg")),
            Comment: Style(color=_C("dim")),
            Keyword: Style(color=_C("accent")),
            Keyword.Constant: Style(color=_C("warning")),
            Name: Style(color=_C("fg")),
            Name.Function: Style(color=_C("muted")),
            Name.Class: Style(color=_C("muted")),
            Name.Builtin: Style(color=_C("muted")),
            String: Style(color=_C("string")),
            Number: Style(color=_C("warning")),
            Operator: Style(color=_C("muted")),
            Punctuation: Style(color=_C("dim")),
            Error: Style(color=_C("danger")),
            Generic: Style(color=_C("fg")),
        }

    def get_style_for_token(self, token_type):  # type: ignore[override]
        while token_type:
            if token_type in self._styles:
                return self._styles[token_type]
            token_type = token_type.parent
        from pygments.token import Token

        return self._styles[Token]

    def get_background_style(self) -> Style:  # type: ignore[override]
        return Style(bgcolor=_C("bg"))


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
            background_color=_C("bg"),
        )


def brand_markdown_theme() -> Theme:
    """Element-style theme for rich Markdown, resolved against the current
    ramp. Rebuild after a theme switch (epoch change)."""
    return Theme(
        {
            "markdown.paragraph": Style(color=_C("fg")),
            "markdown.text": Style(color=_C("fg")),
            "markdown.em": Style(color=_C("fg"), italic=True),
            "markdown.strong": Style(color=_C("fg"), bold=True),
            "markdown.code": Style(color=_C("string")),
            "markdown.code_block": Style(color=_C("fg"), bgcolor=_C("bg")),
            "markdown.block_quote": Style(color=_C("muted")),
            "markdown.list": Style(color=_C("fg")),
            "markdown.item.bullet": Style(color=_C("dim")),
            "markdown.item.number": Style(color=_C("dim")),
            "markdown.hr": Style(color=_C("edge")),
            "markdown.h1": Style(color=_C("fg"), bold=True),
            "markdown.h2": Style(color=_C("fg"), bold=True),
            "markdown.h3": Style(color=_C("muted"), bold=True),
            "markdown.h4": Style(color=_C("muted"), bold=True),
            "markdown.h5": Style(color=_C("muted")),
            "markdown.h6": Style(color=_C("muted")),
            "markdown.link": Style(color=_C("signal")),
            "markdown.link_url": Style(color=_C("signal"), underline=True),
        },
        inherit=True,
    )


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
        yield text

    Heading.__rich_console__ = _flat_heading  # type: ignore[method-assign]
    _installed = True
