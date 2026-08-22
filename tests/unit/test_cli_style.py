"""Tests for the plain-print styling/encoding helper (item 15)."""

from __future__ import annotations

import io

from local_operator import cli_style


class _Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class _Pipe(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return False


def test_colour_disabled_when_no_color_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli_style.colour_enabled(_Tty()) is False
    # And paint returns the text untouched.
    assert cli_style.paint("x", cli_style.ERROR, stream=_Tty()) == "x"


def test_colour_disabled_on_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli_style.colour_enabled(_Pipe()) is False


def test_colour_disabled_on_dumb_term(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert cli_style.colour_enabled(_Tty()) is False


def test_colour_enabled_on_real_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert cli_style.colour_enabled(_Tty()) is True
    painted = cli_style.paint("x", cli_style.ERROR, stream=_Tty())
    assert painted.startswith("\033[") and painted.endswith("\033[0m")


class _EncStream:
    """A stream stand-in with a settable ``encoding`` (StringIO's is read-only)."""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding


def test_can_encode_detects_ascii_only_stream():
    ascii_stream = _EncStream("ascii")
    assert cli_style.can_encode("plain", ascii_stream) is True
    assert cli_style.can_encode("box ─╭", ascii_stream) is False


def test_can_encode_treats_unknown_encoding_as_capable():
    assert cli_style.can_encode("box ─╭", _EncStream("")) is True
