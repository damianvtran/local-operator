"""The CLI's "where your password lives" line must name THIS platform's store.

Why a test and not a nicer sentence: both of these prints used to be the literal
string "the login Keychain (service lop-mobile)". That was true when the store
was macOS-only; once ``mobile/auth.py`` grew a Secret Service arm and a DPAPI
arm, the CLI was telling a Linux or Windows user to go and look in a store their
operating system does not have — while the installer's OWN step lines, printed
inches away by the same command, named the right one through
``store_description()``.

Both lines are exercised through ``mobile_command`` rather than by asserting on
the source, because the failure mode is a message the user reads. ``mobile
status``'s bundle line is held to the same standard and lives here for the same
reason: it reports a state the operator acts on, not a value a caller reads.
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


def test_the_status_line_names_a_missing_bundle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A healthy daemon with no UI must not read as simply "healthy".

    ``status()`` has always carried ``bundle`` (``built`` / ``buildable`` /
    ``missing-sources``), but the human-readable output dropped it — so the exact
    state generations 0.61.13-0.61.16, 0.61.18 and 0.62.0 were flipped into
    printed ``healthy: yes`` while every authed GET answered 503 "mobile web
    bundle not built". The assertion is on the printed line and its remedy, the
    same standard this file holds the password line to: the failure mode is a
    message the operator reads.
    """
    from local_operator.mobile import install as mobile_install

    monkeypatch.setattr(
        mobile_install,
        "status",
        lambda port=0: {
            "installed": True,
            "password_set": True,
            "healthy": True,
            "gate_closed": True,
            "bundle": "buildable",
            "log": "/dev/null",
            "sessions": [],
        },
    )

    assert mobile_command(_args("status")) == 0

    out = capsys.readouterr().out
    assert "healthy:      yes" in out
    # The label leads with the STATE, not the classifier token: `buildable` next to
    # its own negation reads as "fine … not fine" (design round 1, D5). The token
    # stays in the parenthesis, where it is the classifier's own name.
    assert "bundle:       not built (buildable" in out
    assert "lop mobile install" in out, "and the line names the remedy the operator runs"


def test_the_status_line_is_silent_when_the_portal_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a machine that never installed the portal there is no 503 to explain.

    The line exists to make a *hidden outage* visible: `healthy: yes` while the
    phone gets 503. With `installed: no` there is no phone, no portal and no 503,
    so the same line reads as a fault report where the only true advice is "you
    have not set this up yet" — and the reader has to reconcile it with
    `installed: no` (design round 1, D4). Gated on `installed`.
    """
    from local_operator.mobile import install as mobile_install

    monkeypatch.setattr(
        mobile_install,
        "status",
        lambda port=0: {
            "installed": False,
            "password_set": False,
            "healthy": True,
            "gate_closed": True,
            "bundle": "buildable",
            "log": "/dev/null",
            "sessions": [],
        },
    )

    assert mobile_command(_args("status")) == 0

    out = capsys.readouterr().out
    assert "installed:    no" in out
    assert "bundle:" not in out, "no portal means no outage for this line to explain"


def test_the_status_line_names_a_raw_or_missing_web_tree(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`missing-sources` says what is true on its own, without a mislabelled token.

    The two non-servable states need different words: a tree with sources needs
    building, a tree with none has nothing to build from — and telling its reader
    to run `lop mobile install` would be advice that cannot work.
    """
    from local_operator.mobile import install as mobile_install

    monkeypatch.setattr(
        mobile_install,
        "status",
        lambda port=0: {
            "installed": True,
            "password_set": True,
            "healthy": True,
            "gate_closed": True,
            "bundle": "missing-sources",
            "log": "/dev/null",
            "sessions": [],
        },
    )

    assert mobile_command(_args("status")) == 0

    out = capsys.readouterr().out
    assert "bundle:       not built (no web sources to build from)" in out


def test_the_status_line_is_quiet_when_the_bundle_is_built(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The added line is conditional: a servable bundle keeps the output as it was.

    One line, only in the state that needs it — a status command that printed a
    bundle row unconditionally would be a redesign of this output, which is the
    thing the fix was scoped not to do.
    """
    from local_operator.mobile import install as mobile_install

    monkeypatch.setattr(
        mobile_install,
        "status",
        lambda port=0: {
            "installed": True,
            "password_set": True,
            "healthy": True,
            "gate_closed": True,
            "bundle": "built",
            "log": "/dev/null",
            "sessions": [],
        },
    )

    assert mobile_command(_args("status")) == 0

    out = capsys.readouterr().out
    assert "bundle:" not in out, "`built` is the no-op state and says nothing extra"
