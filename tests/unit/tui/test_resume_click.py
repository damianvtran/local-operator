"""The click ladder: a running viewer, a stopped app, then a terminal.

Rung 2 is new and it is the one the design's requirement names: clicking a
banner for a session when nothing is running used to open a TERMINAL, which is
the wrong surface when the user has the desktop app installed. Launching it with
the session id is what makes the click land where the conversation actually
lives.

Two properties are load-bearing and both have a silent failure mode:

- **Discovery must be LOUD.** ``spawn_detached`` reports only whether a child
  was STARTED, and a launcher that is not installed starts perfectly well and
  exits 1. So a candidate that cannot work would look like success, later
  candidates would never be tried, and the click would land nowhere. These tests
  assert the fall-through on a non-zero exit and the argvs exactly.
- **A live process must never be killed.** The wait is bounded, and a timeout
  means "still running" — treating it as a failure would try the next candidate
  (a second window) or, worse, kill the app the click just launched.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from local_operator.tui import resume_click


class _Launcher:
    """Records the argv each launch attempt was given, and fakes the outcome.

    ``_launch_once`` is doubled at ``subprocess.Popen`` rather than at
    ``_launch_once`` itself, so the ladder's own ordering, argv construction and
    exit-status reading are all under test — the parts a mock of the whole helper
    would skip.
    """

    def __init__(self, outcomes: dict[str, int | None], monkeypatch) -> None:
        self.attempts: list[list[str]] = []
        self.outcomes = outcomes

        def fake_popen(argv, **_kwargs):
            self.attempts.append(list(argv))
            outcome = self.outcomes.get(argv[0], 1)

            class _Process:
                def wait(self, timeout: float | None = None) -> int:
                    if outcome is None:
                        raise subprocess.TimeoutExpired(argv, timeout or 0.0)
                    return outcome

            return _Process()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)


@pytest.fixture(autouse=True)
def launch_rung_enabled(monkeypatch):
    """Opt this module OUT of the suite-wide launch refusal, deliberately.

    ``tests/conftest.py`` sets ``LOCAL_OPERATOR_NO_DESKTOP_LAUNCH`` for every
    test, because rung 2 discovers the real ``local-operator-ui`` on a
    developer's PATH and would start it. This file is the one place that
    exercises that rung, so it clears the gate — visibly, here, rather than by
    leaving it to each test to remember.
    """
    monkeypatch.delenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, raising=False)


@pytest.fixture
def no_viewer(monkeypatch):
    """No viewer is running, and the terminal spawn is recorded rather than run."""
    spawned: list[str] = []
    monkeypatch.setattr(resume_click, "_route_to_viewer", lambda session_id: False)
    monkeypatch.setattr(
        resume_click, "_spawn_terminal", lambda session_id: spawned.append(session_id) or True
    )
    return spawned


def _no_configured_command(monkeypatch) -> None:
    monkeypatch.setattr(resume_click, "_configured_launch_command", lambda: [])


def test_rung_one_wins_and_nothing_is_launched(monkeypatch, no_viewer):
    """A running viewer is switched in place — no process, no window, no wait."""
    monkeypatch.setattr(resume_click, "_route_to_viewer", lambda session_id: True)
    launched = []
    monkeypatch.setattr(resume_click, "_launch_desktop", lambda session_id: launched.append(1))

    assert resume_click.open_session("a" * 12) is True
    assert launched == []
    assert no_viewer == []


def test_a_configured_launch_command_is_used_verbatim_with_the_session_id(monkeypatch, no_viewer):
    """The user's own answer wins over discovery, and ``{session}`` is substituted."""
    monkeypatch.setattr(
        resume_click, "_configured_launch_command", lambda: ["my-app", "--open={session}", "--new"]
    )
    launcher = _Launcher({"my-app": 0}, monkeypatch)

    assert resume_click.open_session("b" * 12) is True
    assert launcher.attempts == [["my-app", f"--open={'b' * 12}", "--new"]]
    assert no_viewer == []


def test_the_npm_bin_is_preferred_and_the_flag_shape_is_exact(monkeypatch, no_viewer):
    """The argv the app parses, asserted literally — it is a cross-repo contract."""
    _no_configured_command(monkeypatch)
    # Patched on the MODULE: `_launch_desktop` imports `shutil` inside the
    # function, so the seam is the module object rather than an attribute of
    # `resume_click`.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/" + name)
    launcher = _Launcher({"/usr/local/bin/local-operator-ui": 0}, monkeypatch)

    assert resume_click.open_session("c" * 12) is True
    assert launcher.attempts == [["/usr/local/bin/local-operator-ui", "--open-session", "c" * 12]]
    assert no_viewer == []


@pytest.mark.skipif(sys.platform != "darwin", reason="the bundle rung is macOS-only")
def test_the_packaged_bundle_is_launched_by_id_when_no_bin_is_on_path(monkeypatch, no_viewer):
    """A packaged install is not on PATH, so it is addressed by its bundle id.

    ``--args`` is not decoration: ``open`` forwards nothing to the application
    without it, so the flag would be swallowed by LaunchServices and the app
    would come up on the catalogue instead of the conversation.
    """
    _no_configured_command(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    launcher = _Launcher({"open": 0}, monkeypatch)

    assert resume_click.open_session("d" * 12) is True
    assert launcher.attempts == [
        ["open", "-b", "com.local-operator", "--args", "--open-session", "d" * 12]
    ]
    assert no_viewer == []


@pytest.mark.skipif(sys.platform != "darwin", reason="the bundle rung is macOS-only")
def test_a_candidate_that_exits_nonzero_falls_through_to_the_next(monkeypatch, no_viewer):
    """THE LOUD-DISCOVERY PROPERTY, as a test.

    ``open -b`` exits 1 for a bundle that is not installed. Treating "a child
    started" as success would leave the click with nothing on screen and no
    later candidate tried — so the exit status is what decides.
    """
    _no_configured_command(monkeypatch)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/local/bin/local-operator-ui" if name == "local-operator-ui" else None,
    )
    launcher = _Launcher({"/usr/local/bin/local-operator-ui": 1, "open": 0}, monkeypatch)

    assert resume_click.open_session("e" * 12) is True
    assert [argv[0] for argv in launcher.attempts] == [
        "/usr/local/bin/local-operator-ui",
        "open",
    ]
    assert no_viewer == []


def test_a_launcher_still_running_after_the_probe_is_a_success(monkeypatch, no_viewer):
    """A timeout means the app IS running, so nothing else may be tried.

    Two failure modes hide here and both are the reported bug rather than a
    cosmetic one: trying the next candidate opens a SECOND window, and killing
    the process on the timeout shuts down the app the click just launched.
    """
    _no_configured_command(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    launcher = _Launcher({"/bin/local-operator-ui": None}, monkeypatch)

    assert resume_click.open_session("f" * 12) is True
    assert len(launcher.attempts) == 1
    assert no_viewer == []


def test_nothing_installed_falls_through_to_the_terminal(monkeypatch, no_viewer):
    """The original path, reached only when the click has no better home.

    ``pnpm dev`` and a repository checkout are deliberately undiscoverable: a
    second launcher to keep in step buys nothing user-facing, so they land here.
    """
    _no_configured_command(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    if sys.platform == "darwin":
        # The bundle candidate must be tried and refused before the terminal.
        launcher = _Launcher({"open": 1}, monkeypatch)
    else:
        launcher = _Launcher({}, monkeypatch)

    assert resume_click.open_session("a1b2c3d4e5f6") is True
    assert no_viewer == ["a1b2c3d4e5f6"]
    if sys.platform == "darwin":
        assert launcher.attempts, "the bundle candidate was never tried"


def test_a_launcher_that_cannot_start_does_not_eat_the_click(monkeypatch, no_viewer):
    """An OSError from Popen is a candidate that is not there, not a failure."""

    def exploding_popen(argv, **_kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "Popen", exploding_popen)
    _no_configured_command(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)

    assert resume_click.open_session("a1b2c3d4e5f5") is True
    assert no_viewer == ["a1b2c3d4e5f5"]


def test_the_configured_command_is_read_through_the_settings_registry(monkeypatch):
    """One reader, one registry — an edit lands on the next click, not a restart."""
    from local_operator import settings_io

    assert "desktop.launch_command" in settings_io.BY_KEY
    setting = settings_io.BY_KEY["desktop.launch_command"]
    assert setting.path == (
        "desktop",
        "launch_command",
    ), "a literal dotted top-level key would write somewhere nothing reads"
    assert setting.default == resume_click.DESKTOP_LAUNCH_COMMAND_DEFAULT


def test_an_empty_setting_means_discover(monkeypatch):
    """Empty is a real answer ("find it for me"), not a broken command line."""
    monkeypatch.setattr(resume_click, "_configured_launch_command", lambda: [])
    assert resume_click._configured_launch_command() == []


def test_the_session_placeholder_is_the_documented_spelling():
    """A user-facing string: the placeholder is what the setting's help prints."""
    assert resume_click.LAUNCH_SESSION_PLACEHOLDER == "{session}"
    assert resume_click.OPEN_SESSION_FLAG == "--open-session"


def test_the_configured_command_splits_the_way_a_shell_would(monkeypatch):
    """Quoted arguments in the setting must survive to argv intact.

    The registry stores TEXT rather than a comma-separated LIST precisely so an
    argv word may contain a comma, and ``shlex`` is what makes the field a
    command line rather than an approximation of one. A quoted word containing a
    space is the case that separates the two.
    """
    parts = _configured_with(monkeypatch, 'my-app --title "two words, one arg"')
    assert parts == ["my-app", "--title", "two words, one arg"]


def _configured_with(monkeypatch, raw: str) -> list[str]:
    """Read ``desktop.launch_command`` through the real reader, registry doubled."""
    from local_operator.tui import settings as tui_settings

    monkeypatch.setattr(tui_settings, "settings_get", lambda key, default="": raw)
    return resume_click._configured_launch_command()


def test_an_unparseable_command_line_degrades_to_discovery(monkeypatch):
    """An unbalanced quote must not kill the click."""
    assert _configured_with(monkeypatch, "'unbalanced") == []


def test_the_refusal_stops_the_launch_rung_and_falls_through(monkeypatch, no_viewer):
    """The central gate: set, no candidate is even considered.

    This is the escape that keeps a test (or a user who wants a terminal) from
    reaching a real install, so it is asserted at the ladder rather than trusted:
    the discovery is given a working candidate and must not run it.
    """
    monkeypatch.setenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, "1")
    monkeypatch.setattr(
        resume_click, "_configured_launch_command", lambda: ["definitely-a-real-app"]
    )
    launcher = _Launcher({"definitely-a-real-app": 0}, monkeypatch)

    assert resume_click.open_session("a1b2c3d4e5f7") is True
    assert launcher.attempts == [], launcher.attempts
    assert no_viewer == ["a1b2c3d4e5f7"]


def test_the_refusal_is_off_when_the_variable_is_absent(monkeypatch):
    """The gate must be a refusal, not a switch that ships off.

    A user who never sets the variable gets discovery, which is the documented
    default, so the variable's ABSENCE has to reach the candidates.
    """
    monkeypatch.delenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, raising=False)
    monkeypatch.setattr(resume_click, "_configured_launch_command", lambda: ["my-app"])
    launcher = _Launcher({"my-app": 0}, monkeypatch)

    assert resume_click._launch_desktop("a1b2c3d4e5f8") is True
    assert launcher.attempts == [["my-app"]]
