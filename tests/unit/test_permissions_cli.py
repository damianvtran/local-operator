"""``lop permissions`` and ``guide://permissions``.

Pins the two promises the command makes to an operator: ``status`` is read-only
(it plants nothing, signs nothing, and checks Screen Recording by preflight
only), and ``setup`` — the one path that may raise the keychain dialog —
refuses without a terminal and in an isolated environment, before any
``security``/``codesign`` call.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

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
    monkeypatch.setattr(procname, "publish_team_runtime_result", refuse)
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
    # Prose, not the machine code (UX round 1, U8).
    assert "LOCAL_OPERATOR_RUNTIME_ROOT is set" in err or "HOME is redirected" in err
    assert "runtime-root-override" not in err
    assert no_tools == []


@pytest.mark.skipif(sys.platform != "darwin", reason="status reads a macOS identity")
def test_status_plants_nothing_and_never_spawns_when_no_image_exists(isolated, monkeypatch, capsys):
    spawned: list[object] = []
    monkeypatch.setattr(permissions_cli, "_runtime_preflight", lambda *_: spawned.append(1) or "x")
    assert _run(None) == 0, "bare `lop permissions` is status"
    out = capsys.readouterr().out
    assert "guide://permissions" in out
    assert not (isolated / "runtime").exists(), "status must not plant a runtime image"
    assert spawned == [], "no image means no preflight child"
    assert "Image:" not in out, "no path for an image that does not exist (U6)"
    assert "no runtime image yet" in out
    assert "Only Screen Recording is checked" in out, "U7"


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
        # Review round 1: the cost of Always Allow and its undo (U1), the
        # two-builds caveat (R5), and the macOS 15 pane name.
        "any program running as your macOS user",
        "Keychain Access",
        "runtime-signing.json",
        "two** Python builds",
        "Screen & System Audio Recording",
    ):
        assert needle in body, needle
    assert "/Users/" not in body, "no machine-private paths in a shipped guide"


# ---------------------------------------------------------------------------
# Review round 1 (PR #3691): status stays read-only, setup reports honestly.
# ---------------------------------------------------------------------------

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS identity")
_SHA1 = "0123456789ABCDEF0123456789ABCDEF01234567"
_OTHER_SHA1 = "89ABCDEF0123456789ABCDEF0123456789ABCDEF"
_TEAM = "ABCDE12345"


def _planted(monkeypatch, tmp_path, flavor: str) -> Path:
    """``runtime_identity`` reporting an existing image, without building one."""
    image = tmp_path / "image" / "Local Operator"
    image.parent.mkdir(parents=True)
    image.write_text("")
    monkeypatch.setattr(
        procname,
        "runtime_identity",
        lambda *a, **k: {"planted": True, "flavor": flavor, "image": str(image)},
    )
    monkeypatch.setattr(procname, "installed_app", lambda: None)
    return image


@darwin_only
def test_status_preflights_the_reported_image_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """R3 / QA Q1: never through ``spawn_identity`` (it links and can build an image)."""
    from local_operator import macos_disclaim

    image = _planted(monkeypatch, tmp_path, procname.FLAVOR_ADHOC)

    def forbidden(*args, **kwargs):
        raise AssertionError("status must not run the plant ladder")

    for name in ("spawn_identity", "ensure_branded_interpreter", "_plant_runtime", "_link_to"):
        monkeypatch.setattr(procname, name, forbidden)
    executed: list[str] = []

    def spawn(argv, *, executable, stdout, **kwargs):
        executed.append(executable)
        return subprocess.Popen(
            [sys.executable, "-c", "print('denied')"], stdout=stdout, stdin=subprocess.DEVNULL
        )

    monkeypatch.setattr(macos_disclaim, "spawn_disclaimed", spawn)

    assert _run("status") == 0
    out = capsys.readouterr().out
    assert executed == [str(image)], "the preflight ran through the image status reported"
    assert f"Image: {image}" in out
    assert '"Local Operator Runtime"' in out
    assert "Screen & System Audio Recording" in out


@darwin_only
def test_a_path_keyed_image_is_not_called_the_runtime_row(tmp_path, monkeypatch, capsys):
    """N1: only the ad-hoc bundle is named "Local Operator Runtime"."""
    _planted(monkeypatch, tmp_path, "legacy")
    monkeypatch.setattr(permissions_cli, "_runtime_preflight", lambda image: "denied")
    assert _run("status") == 0
    out = capsys.readouterr().out
    assert 'Turn on "Local Operator" in' in out
    assert 'Turn on "Local Operator Runtime"' not in out


@darwin_only
def test_a_hung_preflight_is_bounded_and_reaped(tmp_path, monkeypatch):
    """N2: the read used to block until the child exited, however long."""
    from local_operator import macos_disclaim

    children: list[subprocess.Popen[bytes]] = []

    def spawn(argv, *, executable, stdout, **kwargs):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=stdout,
            stdin=subprocess.DEVNULL,
        )
        children.append(child)
        return child

    monkeypatch.setattr(macos_disclaim, "spawn_disclaimed", spawn)
    monkeypatch.setattr(permissions_cli, "_PREFLIGHT_TIMEOUT_S", 0.5)
    started = time.monotonic()
    try:
        assert permissions_cli._runtime_preflight(sys.executable) == "unknown"
        assert time.monotonic() - started < 10
        assert children[0].poll() is not None, "the hung child is killed and reaped"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


@darwin_only
@pytest.mark.parametrize(
    ("identities", "expected", "absent"),
    [
        ([], "publisher's", "lop permissions setup"),
        (
            [(_SHA1, f"Developer ID Application: Example ({_TEAM})")],
            "lop permissions setup",
            "publisher's",
        ),
    ],
)
def test_status_suggests_setup_only_when_it_can_work(monkeypatch, identities, expected, absent):
    """U2: setup needs the app publisher's key; others are told so, not sent to fail."""
    monkeypatch.setattr(procname, "team_signing_refusal", lambda: None)
    monkeypatch.setattr(procname, "read_signing_optin", lambda: None)
    monkeypatch.setattr(permissions_cli, "_developer_identities", lambda team: identities)
    line = permissions_cli._optin_line(_TEAM)
    assert expected in line
    assert absent not in line


@darwin_only
def test_the_help_footer_names_a_file_a_shell_can_open(monkeypatch, capsys):
    """U3: ``guide://`` resolves only inside a session."""
    assert _run("status") == 0
    footer = [line for line in capsys.readouterr().out.splitlines() if "More help" in line]
    assert footer and "guide://permissions" in footer[0]
    assert Path(permissions_cli._guide_path()).is_file()
    assert permissions_cli._guide_path() in footer[0]


class _Setup:
    """Every setup precondition stubbed; no keychain, codesign or real home."""

    def __init__(self, monkeypatch, tmp_path, capsys):
        self.optin = tmp_path / "home" / ".local-operator" / procname.SIGNING_OPTIN_FILENAME
        self.prompts: list[str] = []
        self.answers: list[str] = []
        self.result = procname.TeamPublish(tmp_path / "team-image", signed=True)
        monkeypatch.setattr(procname, "_optin_path", lambda: self.optin)
        monkeypatch.setattr(procname, "team_signing_refusal", lambda: None)
        monkeypatch.setattr(procname, "installed_app", lambda: tmp_path / "Local Operator.app")
        monkeypatch.setattr(procname, "code_identity", lambda path: {"TeamIdentifier": _TEAM})
        monkeypatch.setattr(
            permissions_cli,
            "_developer_identities",
            lambda team: [(_SHA1, f"Developer ID Application: Example ({_TEAM})")],
        )
        monkeypatch.setattr(procname, "publish_team_runtime_result", lambda *a, **k: self.result)
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

        def answer(prompt: str = "") -> str:
            self.prompts.append(prompt)
            return self.answers.pop(0)

        monkeypatch.setattr("builtins.input", answer)
        self.capsys = capsys

    def record(self) -> dict[str, object] | None:
        return json.loads(self.optin.read_text()) if self.optin.exists() else None

    def always(self) -> object:
        """The recorded ``always_allow_confirmed``; the record must exist."""
        record = self.record()
        assert record is not None, "setup left no opt-in record"
        return record["always_allow_confirmed"]

    def prior(self, identity: str, always: bool) -> None:
        procname.write_signing_optin(identity, _TEAM, always_allow_confirmed=always)


@darwin_only
def test_setup_explains_what_always_allow_grants(monkeypatch, tmp_path, capsys):
    """U1: the cost of "Always Allow" is stated before the operator chooses."""
    rig = _Setup(monkeypatch, tmp_path, capsys)
    rig.answers = ["n"]
    assert _run("setup") == 0
    out = capsys.readouterr().out
    assert "any program running as your macOS user" in out
    assert "Keychain Access" in out
    assert '"Allow" signs this time only' in out


@darwin_only
def test_saying_no_to_always_allow_says_what_it_costs(monkeypatch, tmp_path, capsys):
    """U5: a "no" used to print the same Done line as a "yes"."""
    rig = _Setup(monkeypatch, tmp_path, capsys)
    rig.answers = ["y", "n"]
    assert _run("setup") == 0
    out = capsys.readouterr().out
    assert "will not sign new Python builds" in out
    assert rig.always() is False


@darwin_only
def test_setup_asks_about_always_allow_only_after_it_signed(monkeypatch, tmp_path, capsys):
    """R4: an existing image ran no codesign, so there was no dialog to answer."""
    rig = _Setup(monkeypatch, tmp_path, capsys)
    rig.result = procname.TeamPublish(tmp_path / "team-image", signed=False)
    rig.answers = ["y"]  # the Continue? prompt only
    assert _run("setup") == 0
    out = capsys.readouterr().out
    assert len(rig.prompts) == 1, rig.prompts
    assert "already signed" in out
    # Round 2 (m1/Q2/U9): with no earlier "Always Allow" there is nothing to
    # keep, so the line must not claim one before the allow-once cost line.
    assert "answer is kept" not in out and "answer for this key is kept" not in out
    assert "will not sign new Python builds" in out
    assert rig.always() is False


@darwin_only
def test_an_existing_image_keeps_the_earlier_always_allow_for_the_same_key(
    monkeypatch, tmp_path, capsys
):
    rig = _Setup(monkeypatch, tmp_path, capsys)
    rig.prior(_SHA1, True)
    rig.result = procname.TeamPublish(tmp_path / "team-image", signed=False)
    rig.answers = ["y"]
    assert _run("setup") == 0
    assert rig.always() is True
    out = capsys.readouterr().out
    assert "will not sign new Python builds" not in out
    assert 'Your earlier "Always Allow" answer for this key is kept.' in out


@darwin_only
def test_a_failed_rerun_restores_the_earlier_optin(monkeypatch, tmp_path, capsys):
    """N3: a failed re-run used to delete an earlier, valid opt-in."""
    rig = _Setup(monkeypatch, tmp_path, capsys)
    rig.prior(_OTHER_SHA1, True)
    before = rig.record()
    rig.result = procname.TeamPublish(
        None, signed=True, reason="signing-failed", detail="errSecInternalComponent"
    )
    rig.answers = ["y"]
    assert _run("setup") == 1
    err = capsys.readouterr().err
    assert rig.record() == before
    assert "errSecInternalComponent" in err, "U5: the failure names codesign's error"


@darwin_only
def test_a_failed_first_setup_leaves_no_optin(monkeypatch, tmp_path, capsys):
    rig = _Setup(monkeypatch, tmp_path, capsys)
    rig.result = procname.TeamPublish(None, reason="signing-failed")
    rig.answers = ["y"]
    assert _run("setup") == 1
    assert rig.record() is None
