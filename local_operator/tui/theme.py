"""Brand theme tokens for the Local Operator TUI — the single source of truth.

The TUI renders the product's "island" night: warm near-black ground, warm
bone text, hairline edges, and exactly one green spent sparingly (active
indicator, links, focus). Values are the island palette from the brand kit
(`docs/design-language.md`, "The always-dark islands") plus the light ramp's
paper/ink pair for the future light flag.

Everything downstream (the generated TCSS variables, rich ``Style`` helpers,
widget colors) reads from :data:`BRAND_TOKENS`; no module hard-codes a hex.
A ``theme_epoch`` counter is bumped on every switch so render caches keyed on
theme can be invalidated in one shot (a ``getThemeEpoch``-style pattern).
"""

from __future__ import annotations

from rich.style import Style

#: Brand token ramps, keyed by theme name. Hex values are exact brand kit
#: values and are asserted by tests — do not round or re-solve them.
BRAND_TOKENS: dict[str, dict[str, str]] = {
    # The island night (default). Dark is the product's native ground.
    "dark": {
        "bg": "#14110c",  # island ground
        "surface": "#1e1a14",  # one elevation step: filled cards, input panel
        "raised": "#272219",  # hover rows / selected fills (depth step two)
        "overlay": "#302a20",  # dialogs, active selection
        "sunken": "#0f0c08",  # one step down: the status band ground
        "fg": "#e9e5db",  # island text
        "muted": "#b5afa2",  # secondary text
        "dim": "#837c6d",  # micro-labels, tool rows, separators
        "faint": "#4a4539",  # meta separators, inert hints
        "edge": "#3b3527",  # hairline borders
        "edge-hi": "#4a4231",  # active rail / focused edge
        "accent": "#38c96a",  # the one green: live indicator, focus
        "string": "#57c785",  # string/success green
        "success": "#57c785",
        "amber": "#e0b04b",  # warnings
        "danger": "#ef8078",  # errors
        "signal": "#6ea8d8",  # cool counterweight: links, file paths (NOT green)
        "label": "#b48cd6",  # violet meta: tips, skill labels
    },
    # Warm paper ramp (D22: real kit semantics; danger is NEVER the green).
    "light": {
        "paper": "#f7f4ee",
        "surface": "#efeae0",
        "raised": "#e4ded0",
        "overlay": "#d8d1c0",
        "sunken": "#e5e0d5",
        "ink": "#211e18",
        "muted": "#565147",
        "dim": "#837c6d",
        "faint": "#b3ab98",
        "hairline": "#e5e0d5",
        "edge-hi": "#c9c0a8",
        "accent": "#177b45",
        "string": "#1e7b4e",
        "success": "#1e7b4e",
        "warning": "#8a5800",
        "danger": "#b23a31",
        "signal": "#2b6ea8",
        "label": "#7c5a9e",
    },
}

#: Map the semantic names used by the TCSS and widgets onto each ramp's raw
#: tokens. Dark is exact; light carries the solved kit semantics (D22):
#: success/warning/danger are real ramp members, never silently remapped.
_SEMANTIC_ALIASES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "bg",
        "surface": "surface",
        "raised": "raised",
        "overlay": "overlay",
        "sunken": "sunken",
        "fg": "fg",
        "muted": "muted",
        "dim": "dim",
        "faint": "faint",
        "edge": "edge",
        "edge-hi": "edge-hi",
        "accent": "accent",
        "success": "success",
        "warning": "amber",
        "danger": "danger",
        "string": "string",
        "signal": "signal",
        "label": "label",
    },
    "light": {
        "bg": "paper",
        "surface": "surface",
        "raised": "raised",
        "overlay": "overlay",
        "sunken": "sunken",
        "fg": "ink",
        "muted": "muted",
        "dim": "dim",
        "faint": "faint",
        "edge": "hairline",
        "edge-hi": "edge-hi",
        "accent": "accent",
        "success": "success",
        "warning": "warning",
        "danger": "danger",
        "string": "string",
        "signal": "signal",
        "label": "label",
    },
}

DEFAULT_THEME = "dark"

_current_theme: str = DEFAULT_THEME
#: Bumped on every theme switch; render caches compare against this to know
#: when their cached rows/styles are stale.
_theme_epoch: int = 0


def available_themes() -> list[str]:
    """Names of the built-in theme ramps."""
    return list(BRAND_TOKENS)


def current_theme() -> str:
    """The active theme name."""
    return _current_theme


def get_theme_epoch() -> int:
    """Monotonic counter bumped on every theme switch (cache invalidation)."""
    return _theme_epoch


def set_theme(name: str) -> int:
    """Activate ``name`` and bump the epoch so every render cache invalidates.

    Raises ``KeyError`` for an unknown theme rather than silently falling
    back: a bad theme name is a config bug, not a runtime condition.
    Returns the new epoch.
    """
    global _current_theme, _theme_epoch
    if name not in BRAND_TOKENS:
        raise KeyError(f"unknown theme: {name!r} (have {', '.join(BRAND_TOKENS)})")
    _current_theme = name
    _theme_epoch += 1
    return _theme_epoch


def get_tokens(theme: str | None = None) -> dict[str, str]:
    """Raw brand tokens for ``theme`` (defaults to the current theme)."""
    name = theme or _current_theme
    try:
        return dict(BRAND_TOKENS[name])
    except KeyError:
        raise KeyError(f"unknown theme: {name!r} (have {', '.join(BRAND_TOKENS)})") from None


def semantic_color(semantic: str, theme: str | None = None) -> str:
    """Resolve a semantic name (``bg``, ``accent``, ...) to a hex for a ramp."""
    name = theme or _current_theme
    aliases = _SEMANTIC_ALIASES[name]
    if semantic not in aliases:
        raise KeyError(f"unknown semantic color {semantic!r} for theme {name!r}")
    return BRAND_TOKENS[name][aliases[semantic]]


def tcss_variable_map(theme: str | None = None) -> dict[str, str]:
    """CSS variable name -> hex for both the raw tokens and semantic aliases.

    Textual injects these via ``App.get_css_variables``; the TCSS file then
    references ``var(--lo-<name>)`` and never a literal hex.
    """
    name = theme or _current_theme
    variables: dict[str, str] = {}
    for token, hex_value in get_tokens(name).items():
        variables[f"lo-{token}"] = hex_value
    for semantic in _SEMANTIC_ALIASES[name]:
        variables[f"lo-{semantic}"] = semantic_color(semantic, name)
    return variables


def generate_tcss_vars(theme: str) -> str:
    """Render the CSS variable block for ``theme`` (one declaration per line).

    Textual's TCSS dialect defines variables as ``$name: value;`` (not the
    web ``--name``/``var()`` form), so this emits exactly that — the block is
    prepended verbatim to ``local_operator.tcss`` and every widget selector
    then references ``$lo-<token>``. It is the embeddable form of
    :func:`tcss_variable_map` and is what the theme tests assert against.
    """
    lines = [f"/* local-operator brand tokens: {theme} */"]
    for name, value in tcss_variable_map(theme).items():
        lines.append(f"${name}: {value};")
    return "\n".join(lines) + "\n"


def fg(token: str) -> Style:
    """A rich ``Style`` with the theme's foreground color for ``token``."""
    return Style(color=semantic_color(token))


def bg(token: str) -> Style:
    """A rich ``Style`` with the theme's background color for ``token``."""
    return Style(bgcolor=semantic_color(token))
