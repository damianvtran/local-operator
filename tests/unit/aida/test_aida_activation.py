"""R17/R21: the boot auto-activation signal, one branch per way it can answer.

The predicate's job is a DEFAULT decision — an interactive install gets her
with no configuration, a cloud/automation install is not auto-activated — so
each signal is pinned separately and the failure direction ("cannot tell" =
"no surface") is pinned too.
"""

from __future__ import annotations

import sys

import pytest

from local_operator.aida import activation


class _Tty:
    def isatty(self) -> bool:
        return True


class _Pipe:
    def isatty(self) -> bool:
        return False


def _terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())


def _piped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", _Pipe())
    monkeypatch.setattr(sys, "stdout", _Pipe())


def test_a_terminal_is_a_human_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    _terminal(monkeypatch)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    assert activation.human_surface_present() is True


def test_a_desktop_managed_daemon_is_a_human_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app spawns its backend with a pipe for stdio; the token is the signal."""
    _piped(monkeypatch)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token-for-this-test")
    assert activation.human_surface_present() is True


def test_a_piped_process_without_the_desktop_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cloud/automation shape: agent-runtime-svc pipes, no desktop plane."""
    _piped(monkeypatch)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    assert activation.human_surface_present() is False


def test_one_tty_alone_is_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both streams are the test, mirroring `network.relay._has_terminal`."""
    monkeypatch.setattr(sys, "stdin", _Pipe())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    assert activation.human_surface_present() is False


def test_a_closed_stream_answers_no_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """`isatty` can raise on a closed stream; "cannot tell" must mean "no"."""

    class _Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stdin", _Closed())
    monkeypatch.setattr(sys, "stdout", _Pipe())
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    assert activation.human_surface_present() is False


def test_an_unreadable_desktop_posture_answers_no_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _piped(monkeypatch)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)

    def _boom() -> None:
        raise RuntimeError("no desktop module here")

    monkeypatch.setattr("local_operator.server.desktop.desktop_posture", _boom)
    assert activation.human_surface_present() is False


# ---------------------------------------------------------------------------
# R17's third signal (2026-10-09): the boot's HOME identity.
# ---------------------------------------------------------------------------


def test_the_users_home_answers_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())
    assert activation.home_is_the_users() is True


def test_a_redirected_home_answers_no(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    monkeypatch.setattr(
        "local_operator.supervisors.real_home", lambda: Path("/nonexistent-foreign")
    )
    assert activation.home_is_the_users() is False


def test_an_unknowable_home_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform we cannot interrogate must keep today's behaviour.

    ``None`` from the shared predicate (no passwd database) and a raise from
    the probe both answer True — taking the feature away on "cannot tell" is
    the direction this deliberately does not take; the toast gate documents
    the same asymmetry.
    """
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: None)
    assert activation.home_is_the_users() is True

    def _boom() -> None:
        raise RuntimeError("no home here")

    monkeypatch.setattr("local_operator.supervisors.real_home", _boom)
    assert activation.home_is_the_users() is True
