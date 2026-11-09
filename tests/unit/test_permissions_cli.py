"""``lop permissions`` and ``guide://permissions``.

Pins the two promises the command makes to an operator: ``status`` is read-only
(it plants nothing, signs nothing, and checks Screen Recording by preflight
only), and ``setup`` — the one path that may raise the keychain dialog —
refuses without a terminal and in an isolated environment, before any
``security``/``codesign`` call.
"""

from __future__ import annotations

import argparse
import sys

import pytest

from local_operator import permissions_cli, procname
from local_operator.guides.discovery import discover_guides, make_guide_resolver


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path


@pytest.fixture()
def no_tools(monkeypatch):
    """Fail the test if setup reaches ``security``/``codesign`` or a team signature."""
    calls: list[object] = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError(f"setup must refuse before running a tool: {args!r}")

    monkeypatch.setattr(permissions_cli.subprocess, "run", refuse)
    monkeypatch.setattr(procname, "publish_team_runtime", refuse)
    monkeypatch.setattr(procname, "write_signing_optin", refuse)
    return calls


def _run(action: str | None) -> int:
    return permissions_cli.permissions_command(argparse.Namespace(permissions_command=action))


@pytest.mark.skipif(sys.platform != "darwin", reason="setup is macOS-only")
def test_setup_refuses_without_a_terminal(monkeypatch, capsys, no_tools):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _run("setup") == 2
    assert "interactive" in capsys.readouterr().err
    assert no_tools == []


@pytest.mark.skipif(sys.platform != "darwin", reason="setup is macOS-only")
def test_setup_refuses_an_isolated_environment_even_on_a_tty(monkeypatch, capsys, no_tools):
    class _Tty:
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            return None

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    assert _run("setup") == 2
    err = capsys.readouterr().err
    assert "runtime-root-override" in err or "isolated-home" in err
    assert no_tools == []


@pytest.mark.skipif(sys.platform != "darwin", reason="status reads a macOS identity")
def test_status_plants_nothing_and_never_spawns_when_no_image_exists(isolated, monkeypatch, capsys):
    spawned: list[object] = []
    monkeypatch.setattr(permissions_cli, "_runtime_preflight", lambda: spawned.append(1) or "x")
    assert _run(None) == 0, "bare `lop permissions` is status"
    out = capsys.readouterr().out
    assert "guide://permissions" in out
    assert not (isolated / "runtime").exists(), "status must not plant a runtime image"
    assert spawned == [], "no image means no preflight child"


def test_the_preflight_probe_never_requests_access():
    """Preflight only: a ``CGRequestScreenCaptureAccess`` call would put a dialog up."""
    assert "CGPreflightScreenCaptureAccess" in permissions_cli._PREFLIGHT_PROBE
    assert "Request" not in permissions_cli._PREFLIGHT_PROBE


def test_the_permissions_guide_is_discovered_and_answers_the_operators_questions():
    guides = {guide.name: guide for guide in discover_guides()}
    assert "permissions" in guides
    body = make_guide_resolver(guides)("guide://permissions")
    assert body is not None
    for needle in (
        "lop permissions status",
        "lop permissions setup",
        "Local Operator Runtime",
        "com.local-operator.runtime",
        "tccutil reset",  # named only to forbid it
        "SDK",
        "exec",
        "TUI",
        "−",
    ):
        assert needle in body, needle
    assert "/Users/" not in body, "no machine-private paths in a shipped guide"
