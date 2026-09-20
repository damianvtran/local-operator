"""The CLI's "where your password lives" line must name THIS platform's store.

Why a test and not a nicer sentence: both of these prints used to be the literal
string "the login Keychain (service lop-mobile)". That was true when the store
was macOS-only; once ``mobile/auth.py`` grew a Secret Service arm and a DPAPI
arm, the CLI was telling a Linux or Windows user to go and look in a store their
operating system does not have — while the installer's OWN step lines, printed
inches away by the same command, named the right one through
``store_description()``.

Both lines are exercised through ``mobile_command`` rather than by asserting on
the source, because the failure mode is a message the user reads.
"""

from __future__ import annotations

import argparse

import pytest

from local_operator.cli import build_cli_parser, mobile_command
from local_operator.mobile import auth


def _args(*extra: str) -> argparse.Namespace:
    return build_cli_parser().parse_args(["mobile", *extra])


@pytest.fixture
def linux_store(monkeypatch: pytest.MonkeyPatch) -> str:
    """A Linux with no keyring daemon, so the store is the 0600 file.

    Deliberately NOT macOS: the assertion that matters is that the sentence
    moves with the platform, and it cannot move while every run of the test
    sees the platform the message was hardcoded for.
    """
    monkeypatch.setattr(auth, "_PLATFORM", "linux")
    monkeypatch.setattr(auth, "_secret_tool", lambda: None)
    return auth.store_description()


def test_the_password_line_names_the_platform_store(
    linux_store: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-TTY ``lop mobile password`` refuses to print the secret, and says where.

    Non-TTY is the branch under test — under pytest stdout is captured, which
    is exactly the condition (`isatty()` false) that sends the command down the
    "say where it is, never print it" path, and it is also the only path that
    touches no real Keychain.
    """
    assert mobile_command(_args("password")) == 0

    out = capsys.readouterr().out
    assert "Keychain" not in out, "a Linux host has no login Keychain to point at"
    assert f"portal password is in {linux_store}." in out


def test_the_install_line_names_the_platform_store(
    linux_store: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The success summary right after ``install`` uses the same one function."""
    from local_operator.mobile import install as mobile_install

    monkeypatch.setattr(mobile_install, "install", lambda: {"ok": True, "steps": []})

    assert mobile_command(_args("install")) == 0

    out = capsys.readouterr().out
    assert "Keychain" not in out
    assert f"the password is in {linux_store}." in out
