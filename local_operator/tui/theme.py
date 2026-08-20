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

import re
from dataclasses import dataclass, field

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
        # State grounds for the tool rows. A tool row's ELEVATION already
        # carries "this is an action"; its TINT carries the outcome. Both
        # sit at roughly the same luminance as `surface` so a failed row
        # never shouts — it just stops being neutral.
        "tint-danger": "#2c1a16",  # failed tool row ground (warm red cast)
        # Selection ground, same idea as `tint-danger`: HUE says the state,
        # elevation only says "this is a row". Pure luminance steps could not
        # carry selection — surface->raised is 1.096:1, which is imperceptible,
        # so the picker's highlight rested entirely on its accent name and hover
        # gave a mouse user almost nothing (D8). A green cast at roughly
        # surface's luminance reads immediately without shouting.
        "tint-select": "#16221a",  # picker's highlighted row (cool green cast)
        # Hover ON the selected row. Hover is additive, so pointing at the
        # highlighted row has to read as MORE than selection alone; reusing a
        # plain elevation step there erased the selection instead.
        "tint-select-hi": "#1b2a1f",
        # The attachment chip's ground — `[Image #1, 1568x200]` in the
        # composer. Cool because `signal` is already the ramp's file/reference
        # hue, and because green and warm are both spoken for one row apart
        # (the accent chevron, and `edge` behind selected prose).
        #
        # This one is NOT iso-luminant like `tint-select`, and could not be:
        # the chip is ~20 cells inside a text field, not a full-width row, and
        # downward is not available — pure black against `surface` tops out at
        # 1.21:1, so there is no well to recess into. 1.37:1 up is the same
        # visible step `edge` makes at 1.42:1, and `signal` still reads 4.98:1
        # on it.
        "tint-attach": "#233448",
        # The same chip while the selection covers it — 1.61:1 above the
        # resting chip, a bigger jump than resting-vs-field, so "selected" is
        # never the weaker of the two reads. `signal` would drop to 3.28:1
        # here, so the selected chip takes `fg` (6.63:1) and the brighten
        # lands in both the ground and the ink.
        "tint-attach-hi": "#325070",
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
        "tint-danger": "#f6e2dd",
        "tint-select": "#dfeadf",
        "tint-select-hi": "#d2e3d2",
        # Paper's attachment chip. The direction FLIPS: `signal` only manages
        # 4.49:1 on `surface` here, so a darker chip ground would push the
        # marker's own ink below AA. The chip goes UP toward `paper` instead
        # — a cool card lifted out of the warm field (4.69:1) — and the
        # selected step goes down and saturated, where `ink` reads 11.22:1.
        "tint-attach": "#e9f0f8",
        "tint-attach-hi": "#c3d6ee",
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
        "tint-danger": "tint-danger",
        "tint-select": "tint-select",
        "tint-select-hi": "tint-select-hi",
        "tint-attach": "tint-attach",
        "tint-attach-hi": "tint-attach-hi",
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
        "tint-danger": "tint-danger",
        "tint-select": "tint-select",
        "tint-select-hi": "tint-select-hi",
        "tint-attach": "tint-attach",
        "tint-attach-hi": "tint-attach-hi",
    },
}

DEFAULT_THEME = "dark"

#: The canonical semantic token set — the ONE vocabulary every widget, the
#: TCSS, and the markdown/syntax ramps speak. A theme is precisely a total
#: function from this set to hexes: `register_theme` rejects a palette that
#: misses a token (a partial theme would fail at render time, on whichever
#: widget happened to ask first) or invents one (a token nothing reads is a
#: palette author's typo, not an extension point).
SEMANTIC_TOKENS: tuple[str, ...] = tuple(_SEMANTIC_ALIASES["dark"])

#: Palette hexes must be exactly ``#rrggbb``. Short forms and named colors are
#: rejected at registration: the TCSS variable block and the terminal-theme
#: mapping both slice these strings positionally.
_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class ThemeSpec:
    """One selectable theme: identity, one-line pitch, and its full ramp.

    ``tokens`` maps every name in :data:`SEMANTIC_TOKENS` to a hex. ``dark``
    declares the ramp's POLARITY — whether elevation lightens (dark ground)
    or darkens (paper ground) as it rises. Its consumer today is the palette
    contrast gate, which checks the elevation ladder in the declared
    direction; nothing at runtime branches on it (the ANSI mapping is built
    from the active ramp's tokens whatever the polarity, and Textual's own
    dark/light flag is left alone). ``description`` is the picker row's
    one-liner, so it is written for a user choosing between thirty rows,
    not for a changelog.
    """

    name: str
    label: str
    description: str
    dark: bool = True
    tokens: dict[str, str] = field(default_factory=dict)


def _builtin_spec(name: str, label: str, description: str, dark: bool) -> ThemeSpec:
    """Wrap a :data:`BRAND_TOKENS` ramp in the registry's spec shape.

    The two brand ramps keep their raw-token + alias form because the raw
    names (``amber``, ``paper``, ``ink``) are pinned by tests and still
    emitted as TCSS variables; everything else registers semantic-only.
    """
    tokens = {
        semantic: BRAND_TOKENS[name][raw] for semantic, raw in _SEMANTIC_ALIASES[name].items()
    }
    return ThemeSpec(name=name, label=label, description=description, dark=dark, tokens=tokens)


#: Every registered theme, keyed by name. Seeded with the two brand ramps;
#: the curated palettes in :mod:`local_operator.tui.palettes` add themselves
#: on first use (see :func:`_registry` — the import is lazy to keep this
#: module importable from the palettes package without a cycle).
_THEMES: dict[str, ThemeSpec] = {
    "dark": _builtin_spec(
        "dark", "Operator Dark", "The island night, the Local Operator default", dark=True
    ),
    "light": _builtin_spec("light", "Operator Light", "Warm paper, brand ink", dark=False),
}

_palettes_loaded = False


def _registry() -> dict[str, ThemeSpec]:
    """The theme registry with the curated palettes folded in (lazy, once).

    Lazy because :mod:`local_operator.tui.palettes` imports THIS module for
    :class:`ThemeSpec`; importing it at module top would be a cycle. The
    palettes never override the two brand ramps — ``register_theme`` refuses
    duplicates, so a palette shadowing ``dark`` is a loud error, not a silent
    rebrand.
    """
    global _palettes_loaded
    if not _palettes_loaded:
        from local_operator.tui import palettes

        # Latched only AFTER the fold completes (review round 1, F3): set
        # before it, a partial registration failure would serve a silently
        # truncated registry to every later caller — themes vanishing with no
        # error, and a saved name in the missing tail reading as user error.
        # The identity skip below is what makes the retry safe: the palette
        # lists are module-level constants, so a second pass sees the SAME
        # spec objects it already folded and skips them, reaching — and
        # re-raising from — the palette that actually failed. Identity, not
        # name: a name-based skip would also silence a genuine cross-family
        # name collision, which must keep raising.
        for spec in palettes.all_palettes():
            if _THEMES.get(spec.name) is not spec:
                register_theme(spec)
        _palettes_loaded = True
    return _THEMES


def register_theme(spec: ThemeSpec) -> None:
    """Add ``spec`` to the registry, validating it is total and new.

    A theme is data, so every mistake a palette author can make is caught
    HERE, at registration, rather than at render time in whichever widget
    first asks for the missing token.
    """
    if spec.name in _THEMES:
        raise ValueError(f"theme {spec.name!r} is already registered")
    missing = [token for token in SEMANTIC_TOKENS if token not in spec.tokens]
    if missing:
        raise ValueError(f"theme {spec.name!r} is missing tokens: {', '.join(missing)}")
    unknown = [token for token in spec.tokens if token not in SEMANTIC_TOKENS]
    if unknown:
        raise ValueError(f"theme {spec.name!r} has unknown tokens: {', '.join(unknown)}")
    malformed = [
        f"{token}={value!r}" for token, value in spec.tokens.items() if not _HEX_RE.fullmatch(value)
    ]
    if malformed:
        raise ValueError(f"theme {spec.name!r} has malformed hexes: {', '.join(malformed)}")
    _THEMES[spec.name] = spec


def theme_spec(name: str | None = None) -> ThemeSpec:
    """The :class:`ThemeSpec` for ``name`` (default: the active theme)."""
    registry = _registry()
    key = name or _current_theme
    try:
        return registry[key]
    except KeyError:
        raise KeyError(f"unknown theme: {key!r} (have {', '.join(registry)})") from None


_current_theme: str = DEFAULT_THEME
#: Bumped on every theme switch; render caches compare against this to know
#: when their cached rows/styles are stale.
_theme_epoch: int = 0


def available_themes() -> list[str]:
    """Names of every registered theme, brand ramps first, then curated."""
    return list(_registry())


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
    registry = _registry()
    if name not in registry:
        raise KeyError(f"unknown theme: {name!r} (have {', '.join(registry)})")
    _current_theme = name
    _theme_epoch += 1
    return _theme_epoch


def get_tokens(theme: str | None = None) -> dict[str, str]:
    """Raw tokens for ``theme`` (defaults to the current theme).

    For the two brand ramps this is the raw :data:`BRAND_TOKENS` vocabulary
    (``amber``, ``paper``…), preserved because tests and the TCSS variable
    block pin it; every curated theme answers with its semantic tokens,
    which ARE its only vocabulary.
    """
    name = theme or _current_theme
    if name in BRAND_TOKENS:
        return dict(BRAND_TOKENS[name])
    return dict(theme_spec(name).tokens)


def semantic_color(semantic: str, theme: str | None = None) -> str:
    """Resolve a semantic name (``bg``, ``accent``, ...) to a hex for a ramp."""
    spec = theme_spec(theme)
    if semantic not in spec.tokens:
        raise KeyError(f"unknown semantic color {semantic!r} for theme {spec.name!r}")
    return spec.tokens[semantic]


def tcss_variable_map(theme: str | None = None) -> dict[str, str]:
    """CSS variable name -> hex for both the raw tokens and semantic aliases.

    Textual injects these via ``App.get_css_variables``; the TCSS file then
    references ``var(--lo-<name>)`` and never a literal hex.
    """
    name = theme or _current_theme
    variables: dict[str, str] = {}
    for token, hex_value in get_tokens(name).items():
        variables[f"lo-{token}"] = hex_value
    for semantic in SEMANTIC_TOKENS:
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
