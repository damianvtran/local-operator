"""The derived ``terminal`` theme clears the same floors as every curated palette."""

from __future__ import annotations

import pytest

from local_operator.tui import host_theme, theme
from tests.unit.tui import test_palette_contrast as contrast_suite

# A blue Ghostty-style dark theme, as the terminal would reply (both reply forms).
BLUE_REPLY = (
    b"\x1b]10;rgb:d8d8/dede/e9e9\x1b\\\x1b]11;rgb:1b1b/2b2b/4a4a\x1b\\"
    b"\x1b]4;1;rgb:ff/55/55\x07\x1b]4;2;#50fa7b\x07\x1b]4;4;rgb:6666/9999/ffff\x1b\\"
    b"\x1b[?62;22c"
)
_BLUE = host_theme.parse_replies(BLUE_REPLY)
_LIGHT = ((0x20, 0x20, 0x20), (0xFA, 0xF8, 0xF0), {})
_MID_GREY = ((0xFF, 0xFF, 0xFF), (0x70, 0x70, 0x70), {})
# Every check the curated palettes must clear, reused rather than restated.
_PALETTE_CHECKS = [
    getattr(contrast_suite, name)
    for name in dir(contrast_suite)
    if name.startswith("test_") and name != "test_default_theme_is_operator_dark"
]


def test_parses_both_reply_forms() -> None:
    fg, bg, ansi = _BLUE
    assert fg == (0xD8, 0xDE, 0xE9)
    assert bg == (0x1B, 0x2B, 0x4A)
    assert ansi == {1: (255, 0x55, 0x55), 2: (0x50, 0xFA, 0x7B), 4: (0x66, 0x99, 0xFF)}


def test_no_reply_means_no_theme() -> None:
    assert host_theme.parse_replies(b"\x1b[?62;22c") == (None, None, {})


@pytest.fixture
def registered():
    def register(colors):
        theme.unregister_theme(host_theme.NAME)
        theme.register_theme(host_theme.derive(*colors))
        return theme.theme_spec(host_theme.NAME)

    yield register
    theme.unregister_theme(host_theme.NAME)


@pytest.mark.parametrize("colors", [_BLUE, _LIGHT], ids=["blue-dark", "light"])
def test_derived_ramp_clears_every_palette_check(registered, colors) -> None:
    spec = registered(colors)
    assert host_theme.readable(spec)
    assert spec.tokens["bg"] == "#{:02x}{:02x}{:02x}".format(*colors[1])  # the user's ground
    assert spec.tokens["surface"] != spec.tokens["bg"]  # depth survives, unlike ANSI-default
    for check in _PALETTE_CHECKS:
        check(host_theme.NAME)


def test_a_ramp_that_cannot_be_readable_is_refused() -> None:
    # Neither white nor black reaches 7:1 on #707070, so the gate must say no.
    assert not host_theme.readable(host_theme.derive(*_MID_GREY))


@pytest.mark.parametrize("grey", range(0, 256, 5))
def test_readable_agrees_with_the_palette_suite(registered, grey) -> None:
    """Whatever ``readable`` admits really does clear the suite (the gate is not loose)."""
    for fg in ((255, 255, 255), (0, 0, 0)):
        spec = registered((fg, (grey, grey, grey), {}))
        if host_theme.readable(spec):
            for check in _PALETTE_CHECKS:
                check(host_theme.NAME)


def test_brand_ramps_cannot_be_unregistered() -> None:
    with pytest.raises(ValueError):
        theme.unregister_theme("dark")


@pytest.mark.parametrize("var", ["SSH_TTY", "SSH_CONNECTION"])
def test_a_remote_session_skips_the_probe_unless_the_theme_was_chosen(monkeypatch, var) -> None:
    calls = []
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv(var, "remote")
    monkeypatch.setattr(host_theme, "probe", lambda timeout: calls.append(timeout) or b"")
    assert not host_theme.install("dark")
    assert calls == []
    host_theme.install(host_theme.NAME)
    assert calls == [1.0]


def test_install_registers_the_theme_and_refreshes_the_settings_choices(monkeypatch) -> None:
    from local_operator import settings_io

    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setattr(host_theme, "probe", lambda timeout: BLUE_REPLY)
    theme.unregister_theme(host_theme.NAME)
    settings_io._theme_choices()  # a snapshot taken before the theme exists
    try:
        assert host_theme.install("dark")
        assert host_theme.NAME in {choice.label for choice in settings_io._theme_choices()}
    finally:
        theme.unregister_theme(host_theme.NAME)
        settings_io._theme_choices.cache_clear()
