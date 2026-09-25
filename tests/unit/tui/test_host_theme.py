"""The derived ``terminal`` theme clears the same floors as every curated palette."""

from __future__ import annotations

import pytest

from local_operator.tui import host_theme, theme
from tests.unit.tui.test_palette_contrast import _FG_FLOORS, _TINT_CEILING, contrast

# A blue Ghostty-style dark theme, and a light one, as the terminal would reply.
BLUE_REPLY = (
    b"\x1b]10;rgb:d8d8/dede/e9e9\x1b\\\x1b]11;rgb:1b1b/2b2b/4a4a\x1b\\"
    b"\x1b]4;1;rgb:ff/55/55\x07\x1b]4;2;#50fa7b\x07\x1b]4;4;rgb:6666/9999/ffff\x1b\\"
    b"\x1b[?62;22c"
)
LIGHT = ((0x20, 0x20, 0x20), (0xFA, 0xF8, 0xF0), {})


def test_parses_both_reply_forms() -> None:
    fg, bg, ansi = host_theme.parse_replies(BLUE_REPLY)
    assert fg == (0xD8, 0xDE, 0xE9)
    assert bg == (0x1B, 0x2B, 0x4A)
    assert ansi == {1: (255, 0x55, 0x55), 2: (0x50, 0xFA, 0x7B), 4: (0x66, 0x99, 0xFF)}


def test_no_reply_means_no_theme() -> None:
    assert host_theme.parse_replies(b"\x1b[?62;22c") == (None, None, {})


@pytest.mark.parametrize(
    "colors", [host_theme.parse_replies(BLUE_REPLY), LIGHT], ids=["blue-dark", "light"]
)
def test_derived_ramp_is_readable_and_keeps_the_ground(colors) -> None:
    fg, bg, ansi = colors
    spec = host_theme.derive(fg, bg, ansi)
    tokens = spec.tokens
    assert tokens["bg"] == "#{:02x}{:02x}{:02x}".format(*bg)  # the user's ground, untouched
    assert set(tokens) == set(theme.SEMANTIC_TOKENS)
    for token, floor in _FG_FLOORS.items():
        for ground in ("bg", "surface"):
            assert contrast(tokens[token], tokens[ground]) >= floor, (token, ground)
    assert contrast(tokens["danger"], tokens["tint-danger"]) >= 4.0
    assert contrast(tokens["fg"], tokens["edge"]) >= 4.5
    for tint in ("tint-danger", "tint-select", "tint-select-hi"):
        assert contrast(tokens[tint], tokens["bg"]) <= _TINT_CEILING
    assert contrast(tokens["faint"], tokens["bg"]) < contrast(tokens["dim"], tokens["bg"])
    assert tokens["surface"] != tokens["bg"]  # depth survives, unlike an ANSI-default theme


def test_brand_ramps_cannot_be_unregistered() -> None:
    with pytest.raises(ValueError):
        theme.unregister_theme("dark")
