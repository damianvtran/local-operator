"""Theme token tests — exact brand hexes and the generated TCSS block."""

from __future__ import annotations

import pytest

from local_operator.tui import theme


def test_dark_tokens_exact_hexes() -> None:
    """Dark ramp (the product's island night) must carry the brand hexes."""
    assert theme.BRAND_TOKENS["dark"] == {
        "bg": "#14110c",
        "surface": "#1e1a14",
        "raised": "#272219",
        "overlay": "#302a20",
        "sunken": "#0f0c08",
        "fg": "#e9e5db",
        "muted": "#b5afa2",
        "dim": "#837c6d",
        "faint": "#4a4539",
        "edge": "#3b3527",
        "edge-hi": "#4a4231",
        "accent": "#38c96a",
        "string": "#57c785",
        "success": "#57c785",
        "amber": "#e0b04b",
        "danger": "#ef8078",
        "signal": "#6ea8d8",
        "label": "#b48cd6",
        "tint-danger": "#2c1a16",
        "tint-select": "#16221a",
        "tint-select-hi": "#1b2a1f",
    }


def test_light_tokens_exact_hexes() -> None:
    """Light ramp (warm paper) carries REAL kit semantics (D22): success/
    warning/danger are solved ramp members, never remapped onto the green."""
    assert theme.BRAND_TOKENS["light"] == {
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
    }


def test_generate_tcss_vars_contains_every_token_dark() -> None:
    block = theme.generate_tcss_vars("dark")
    for token, hex_value in theme.BRAND_TOKENS["dark"].items():
        assert f"$lo-{token}: {hex_value};" in block


def test_generate_tcss_vars_contains_every_token_light() -> None:
    block = theme.generate_tcss_vars("light")
    for token, hex_value in theme.BRAND_TOKENS["light"].items():
        assert f"$lo-{token}: {hex_value};" in block


def test_dark_is_default() -> None:
    assert theme.DEFAULT_THEME == "dark"
    # current_theme is a module singleton; restore it after poking.
    original = theme.current_theme()
    theme.set_theme("dark")
    try:
        assert theme.current_theme() == "dark"
    finally:
        theme.set_theme(original)


def test_set_theme_bumps_epoch() -> None:
    before = theme.get_theme_epoch()
    theme.set_theme("light")
    assert theme.get_theme_epoch() == before + 1
    theme.set_theme("dark")
    assert theme.get_theme_epoch() == before + 2


def test_set_theme_unknown_raises() -> None:
    before = theme.get_theme_epoch()
    with pytest.raises(KeyError):
        theme.set_theme("neon")
    # A failed switch must not bump the epoch.
    assert theme.get_theme_epoch() == before


def test_semantic_resolution() -> None:
    assert theme.semantic_color("bg", "dark") == "#14110c"
    assert theme.semantic_color("warning", "dark") == "#e0b04b"
    assert theme.semantic_color("bg", "light") == "#f7f4ee"
    # D22: light danger/warning are REAL semantics, never the accent green.
    assert theme.semantic_color("danger", "light") == "#b23a31"
    assert theme.semantic_color("warning", "light") == "#8a5800"
    assert theme.semantic_color("success", "light") == "#1e7b4e"
    assert theme.semantic_color("success", "light") != theme.semantic_color("accent", "light")
    assert theme.semantic_color("danger", "light") != theme.semantic_color("accent", "light")
    with pytest.raises(KeyError):
        theme.semantic_color("nope", "dark")


def test_fg_bg_styles() -> None:
    fg = theme.fg("accent")
    bg = theme.bg("bg")
    assert fg.color is not None
    assert bg.bgcolor is not None


def test_get_tokens_returns_copy() -> None:
    tokens = theme.get_tokens("dark")
    tokens["bg"] = "#000000"
    assert theme.BRAND_TOKENS["dark"]["bg"] == "#14110c"
