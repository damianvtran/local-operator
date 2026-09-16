"""The click ladder: a running desktop viewer, a stopped app, a running TUI, then a terminal.

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
- **The last rung reports a LANDING, not a spawn (UX round 1, U5).** It used to
  end in a detached `lop --resume` with no terminal attached: nothing appeared,
  and the `True` it returned suppressed the receipt the caller prints, so the
  user was told nothing either. It now opens a window it can really open — the
  detected backend, then macOS's AppleScript Terminal, which launches
  Terminal.app and needs no terminal around this process — and answers `False`
  when none did.
- **A configured launcher that cannot run is not silent (UX round 1, U4).**
  `desktop.launch_command` replaces discovery, so a typo diverts every click to
  a terminal: the writer refuses a value that cannot be executed, and the click
  logs a WARNING for one that reached `config.yml` by hand.
- **A click that cannot land says so WHERE THE USER IS LOOKING (UX round 2,
  U10).** The `lop --resume <id>` receipt goes to stderr, and on a real click
  stderr is `/dev/null`, so the failure branch also raises a best-effort toast
  carrying the same sentence. The write-time refusal answers the WRITER's PATH
  question only after asking the user's (agent review round 1, M3).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from local_operator.tui import resume_click

#: A launcher that really is runnable on any machine, for the tests that have
#: to WRITE a ``desktop.launch_command``. The write is validated now (U4), so a
#: made-up name is refused at the writer — which is the point of that fix, and
#: not what these tests are about.
_REAL_LAUNCHER = sys.executable


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
    """No viewer is running, and the terminal spawn is recorded rather than run.

    The double takes ``**_kwargs`` because the ladder asks about ONE surface at a
    time now (review round 1, R9): ``_route_to_viewer(session_id,
    surface=...)``. A double with the old single-argument shape raised
    ``TypeError`` at the call site instead of returning its answer, so every test
    built on this fixture failed for a reason that had nothing to do with what it
    asserts — and one written to swallow the keyword would have hidden the order
    the ladder now applies, which is why the order is asserted explicitly below.
    """
    spawned: list[str] = []
    monkeypatch.setattr(resume_click, "_route_to_viewer", lambda session_id, **_kwargs: False)
    monkeypatch.setattr(
        resume_click, "_spawn_terminal", lambda session_id: spawned.append(session_id) or True
    )
    return spawned


def _no_configured_command(monkeypatch) -> None:
    monkeypatch.setattr(resume_click, "_configured_launch_command", lambda: [])


def _empty_bin(tmp_path: Path, name: str) -> Path:
    """An empty directory usable as a PATH entry.

    The PATH-shaped tests need a PATH that PROVABLY does not contain the name
    under test, which the machine's own PATH cannot promise (this developer's
    homebrew prefix really does hold ``local-operator-ui``). Empty directories
    make the answer the same on every host.
    """
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return directory


def _fake_launcher(directory: Path, name: str) -> None:
    """An executable file called ``name`` inside ``directory``."""
    launcher = directory / name
    launcher.write_text("#!/bin/sh\nexit 0\n")
    launcher.chmod(0o755)


def _no_login_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Double the login PATH to an empty directory, and the writer's too.

    ``settings_io._user_shell_path`` reads it by running the user's login shell,
    so a test that lets it run depends on this machine's rc files and pays up to
    the helper's ten-second bound for the privilege. Doubling it keeps the
    refusal tests hermetic; the lookup itself is driven for real in
    :func:`test_a_bare_name_the_writer_cannot_resolve_is_accepted_when_the_user_can`,
    against a directory this test owns rather than against a shell.
    """
    from local_operator import settings_io

    monkeypatch.setattr(settings_io, "_user_shell_path", lambda: None)
    monkeypatch.setenv("PATH", str(_empty_bin(tmp_path, "no-bin")))


def _rung_4_fails(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Force every rung to fail, and record what the click REPORTS.

    Doubled at ``detached_notify`` rather than at the process boundary: what a
    user gets IS the argument of that call, and the suite-wide notification gate
    (``tests/conftest.py``) means a call that escaped the double would be a no-op
    rather than a toast on the operator's screen.
    """
    from local_operator.tui import notify

    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(resume_click, "_route_to_viewer", lambda session_id, **_kwargs: False)
    monkeypatch.setattr(resume_click, "_launch_desktop", lambda session_id: False)
    monkeypatch.setattr(resume_click, "_spawn_terminal", lambda session_id: False)
    monkeypatch.setattr(
        notify,
        "detached_notify",
        lambda title, body, **kwargs: posted.append((title, body)) or True,
    )
    return posted


def test_a_click_that_cannot_land_posts_the_receipt_as_a_toast(monkeypatch) -> None:
    """U10 / M4: the receipt has no reader on a real click, so the ladder talks.

    Driven: the notifier is spawned by ``spawn_detached`` with stdin, stdout and
    stderr all on ``/dev/null``, and its own ``NSTask`` inherits them — so
    ``cli.resume_click``'s receipt reached nobody on all three reachable
    failures (ssh, non-darwin, a typo'd ``desktop.launch_command`` hand-edited
    into ``config.yml``), and from the chair each was a click that did nothing
    and said nothing. That is the defect the ladder was rewritten for.

    The sentence is asserted VERBATIM and against the CLI's own wording: two
    reports of one failure disagreeing would be worse than either alone.
    """
    session_id = "sess-target-0001"
    posted = _rung_4_fails(monkeypatch)

    assert resume_click.open_session(session_id) is False

    assert posted == [
        (
            "Local Operator",
            f"could not open a terminal for session {session_id} — "
            f"run: lop --resume {session_id}",
        )
    ]


def test_a_click_that_lands_reports_nothing(monkeypatch) -> None:
    """The toast belongs to the FAILURE branch, and only to that branch.

    Nobody watches a notification's activation target: a click that opened a
    window must not also announce itself.
    """
    posted = _rung_4_fails(monkeypatch)
    monkeypatch.setattr(resume_click, "_spawn_terminal", lambda session_id: True)

    assert resume_click.open_session("sess-target-0002") is True
    assert posted == []


def test_the_failure_toast_is_not_clickable_and_cannot_break_the_click(monkeypatch) -> None:
    """Two properties of the failure toast, both about not making things worse.

    NOT CLICKABLE: passing ``session_id`` is what gives a macOS toast an
    activation, and the action it would post is this same ladder — which has just
    failed. A toast that invites a retry loop is worse than one that names the
    command to run.

    BEST-EFFORT: ``detached_notify`` swallows its own failures, but the ladder
    must not depend on that — a notifier that raises costs the toast and nothing
    else, and the click still answers False so the receipt is still printed.
    """
    from local_operator.tui import notify

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(resume_click, "_route_to_viewer", lambda session_id, **_kwargs: False)
    monkeypatch.setattr(resume_click, "_launch_desktop", lambda session_id: False)
    monkeypatch.setattr(resume_click, "_spawn_terminal", lambda session_id: False)
    monkeypatch.setattr(
        notify,
        "detached_notify",
        lambda title, body, **kwargs: calls.append(kwargs) or True,
    )

    assert resume_click.open_session("sess-target-0003") is False
    assert calls == [{}], "the failure toast must not carry a click action"

    def _raises(*_args, **_kwargs):
        raise RuntimeError("no notification centre here")

    monkeypatch.setattr(notify, "detached_notify", _raises)
    assert resume_click.open_session("sess-target-0004") is False


def test_a_running_desktop_viewer_wins_and_nothing_is_launched(monkeypatch):
    """A running viewer is switched in place — no process, no window, no wait.

    Asked about the DESKTOP surface specifically, because that is the rung the
    operator named and the one this test exists to pin (R9).
    """
    asked: list[str | None] = []

    def route(session_id, *, surface=None):
        asked.append(surface)
        return surface == "desktop"

    monkeypatch.setattr(resume_click, "_route_to_viewer", route)
    launched: list[str] = []
    monkeypatch.setattr(
        resume_click, "_launch_desktop", lambda session_id: launched.append(session_id) or True
    )
    spawned: list[str] = []
    monkeypatch.setattr(
        resume_click, "_spawn_terminal", lambda session_id: spawned.append(session_id) or True
    )

    assert resume_click.open_session("a" * 12) is True
    assert asked == ["desktop"], "the ladder did not ask the UI first"
    assert launched == []
    assert spawned == []


def test_the_installed_app_is_launched_before_any_tui_is_switched(monkeypatch, no_viewer):
    """RUNG 2 BEFORE RUNG 3, which is the whole of R9's ladder.

    The order used to be the other way round: a TUI that happened to be open
    swallowed the click before discovery ever ran, so a user who asked for the app
    got their terminal instead — decided by nothing but incidental focus history.
    """
    order: list[str] = []

    def route(session_id, *, surface=None):
        order.append(surface or "tui-or-any")
        return False

    monkeypatch.setattr(resume_click, "_route_to_viewer", route)
    monkeypatch.setattr(
        resume_click, "_launch_desktop", lambda session_id: order.append("launch") or False
    )
    monkeypatch.setattr(
        resume_click, "_spawn_terminal", lambda session_id: order.append("terminal") or True
    )

    assert resume_click.open_session("a" * 12) is True
    assert order == ["desktop", "launch", "tui-or-any", "terminal"]


def test_a_launched_app_ends_the_ladder_and_a_terminal_is_never_spawned(monkeypatch, no_viewer):
    """The UI rungs short-circuit: a terminal next to a window the click just
    opened is exactly the reported symptom (an orphaned terminal per click)."""
    monkeypatch.setattr(resume_click, "_launch_desktop", lambda session_id: True)
    assert resume_click.open_session("a" * 12) is True
    assert no_viewer == []


def test_a_refused_desktop_is_not_a_destination_even_when_one_is_running(monkeypatch):
    """REVIEW ROUND 2, R13: the refusal has to remove the app from the LADDER.

    ``LOCAL_OPERATOR_NO_DESKTOP_LAUNCH`` gated rungs 1 and 2 and left rung 3
    asking "is anything running?" with no surface narrowing — and
    ``choose_viewer`` prefers a desktop — so a RUNNING app was still the
    destination for a click whose launch the user had forbidden. The test that
    used to stand here asserted the CALL SHAPE (that the scan was not asked about
    the desktop surface) and so could not see it: it stubbed the routing rung out
    entirely.

    This drives the REAL rung — ``resume_click._route_to_viewer`` ->
    ``route_click`` -> ``choose_viewer`` — over synthetic records and asserts the
    OUTCOME: which record a click was actually delivered to, and whether a
    terminal was spawned instead. Only the two true boundaries are doubled, the
    viewer scan (records rather than a directory) and the dial. The control half
    clears the refusal visibly, the way this module opts out of it for the launch
    rung.
    """
    from local_operator.session.runtime import viewer_client
    from local_operator.session.runtime import viewers as viewers_module
    from local_operator.session.runtime.viewer_client import ViewerOutcome
    from local_operator.session.runtime.viewers import (
        DESKTOP_SURFACE,
        TUI_SURFACE,
        ViewerRecord,
    )

    wanted = "a" * 12
    desktop = ViewerRecord(pid=1, surface=DESKTOP_SURFACE, control_port=1, control_key="k" * 64)
    tui = ViewerRecord(pid=2, surface=TUI_SURFACE, control_port=2, control_key="k" * 64)
    monkeypatch.setattr(viewers_module, "scan_viewers", lambda root=None: [desktop, tui])

    delivered: list[int] = []

    async def deliver(record, session_id, **_kwargs):
        delivered.append(record.pid)
        return ViewerOutcome(switched=True)

    monkeypatch.setattr(viewer_client, "deliver_click", deliver)
    launched: list[str] = []
    monkeypatch.setattr(
        resume_click, "_launch_desktop", lambda session_id: launched.append(session_id) or True
    )
    spawned: list[str] = []
    monkeypatch.setattr(
        resume_click, "_spawn_terminal", lambda session_id: spawned.append(session_id) or True
    )

    # CONTROL: with the launch allowed, UI-first still lands on the desktop — the
    # narrowing below must not have changed the unrestricted ladder.
    monkeypatch.delenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, raising=False)
    assert resume_click.open_session(wanted) is True
    assert delivered == [desktop.pid], "the unrestricted ladder did not land on the UI"

    # REFUSED, with a desktop RUNNING: the click goes to the TUI instead.
    monkeypatch.setenv(resume_click.DESKTOP_LAUNCH_REFUSED_ENV, "1")
    delivered.clear()
    assert resume_click.open_session(wanted) is True
    assert delivered == [tui.pid], "a refused desktop still took the click"

    # REFUSED with nothing but a desktop on the wire: the app is not a
    # destination at all, so the ladder falls all the way through to a terminal.
    delivered.clear()
    monkeypatch.setattr(viewers_module, "scan_viewers", lambda root=None: [desktop])
    assert resume_click.open_session(wanted) is True
    assert delivered == [], "a refused desktop took the click with no TUI in sight"
    assert spawned == [wanted]
    assert launched == []


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


def test_an_empty_setting_means_discover(monkeypatch, tmp_path):
    """Empty is a real answer ("find it for me"), not a broken command line.

    Driven through the REAL writer and a real ``config.yml`` like the readers
    below. The gesture the page performs for an ``empty_unsets`` row is a
    RESET, so that is what this drives, and the key is then absent from the
    file — which is also the state a first run is in.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    setting = settings_io.BY_KEY["desktop.launch_command"]
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, setting, _REAL_LAUNCHER)
    settings_io.reset_setting(manager, setting)

    assert resume_click._configured_launch_command() == []


def test_the_session_placeholder_is_the_documented_spelling():
    """A user-facing string: the placeholder is what the setting's help prints."""
    assert resume_click.LAUNCH_SESSION_PLACEHOLDER == "{session}"
    assert resume_click.OPEN_SESSION_FLAG == "--open-session"


def test_the_configured_command_splits_the_way_a_shell_would(tmp_path, monkeypatch):
    """Quoted arguments in the setting must survive to argv intact.

    The registry stores TEXT rather than a comma-separated LIST precisely so an
    argv word may contain a comma, and ``shlex`` is what makes the field a
    command line rather than an approximation of one. A quoted word containing a
    space is the case that separates the two.
    """
    parts = _configured_with(
        monkeypatch, tmp_path, f'{_REAL_LAUNCHER} --title "two words, one arg"'
    )
    assert parts == [_REAL_LAUNCHER, "--title", "two words, one arg"]


def test_the_click_reader_sees_what_the_settings_page_wrote(tmp_path, monkeypatch):
    """Q1, pinned where a doubling of the settings layer cannot hide it again.

    Asserted as the RELATION between the two halves rather than as a constant:
    whatever the settings page's own reader reports is what the click must
    parse. On the head QA rejected, this file read ``[]`` while the page read
    ``'my-app --title "two words" --open={session}'``, because the click was
    reading the DISPLAY fast path — whose cache is built from
    ``display_defaults()`` and therefore holds flat-dotted ``display.*`` flags
    exclusively — against a registry that stores this key NESTED.

    Nothing here is doubled: a real write through ``settings_io``, a real
    ``config.yml``, then the reader as the click runs it.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    setting = settings_io.BY_KEY["desktop.launch_command"]
    # The structural fact the defect turned on, pinned rather than described: a
    # nested key is not a display flag, so the display fast path can never hold
    # it, however the user writes the config.
    assert "desktop.launch_command" not in settings_io.display_defaults()

    manager = ConfigManager(tmp_path)
    written = f'{_REAL_LAUNCHER} --title "two words" --open={{session}}'
    settings_io.write_setting(manager, setting, written)

    assert settings_io.read_setting(manager, setting) == written
    assert resume_click._configured_launch_command() == [
        _REAL_LAUNCHER,
        "--title",
        "two words",
        "--open={session}",
    ]


def _configured_with(monkeypatch, tmp_path, raw: str) -> list[str]:
    """Read ``desktop.launch_command`` through the REAL settings layer.

    Written with ``settings_io.write_setting`` — the one writer ``/settings``,
    ``PATCH /v1/settings`` and ``lop config edit`` all funnel through — into a
    real ``config.yml`` under a redirected config dir, then read back by the
    click's own reader with nothing doubled anywhere. The value must therefore
    be one the writer accepts: it validates a launcher that cannot be run (U4),
    so pass :data:`_REAL_LAUNCHER` rather than an invented name. A value that
    the writer refuses is a DIFFERENT state, and the tests that need it write
    ``config.yml`` directly (see ``_hand_written_launcher``).

    The helper this replaces patched ``tui.settings.settings_get`` FIRST, which
    is how QA round 2's defect shipped (Q1): the double answered a key the
    production reader could never see, so the wiring between the registry and
    the click was never exercised by any test in this file.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["desktop.launch_command"], raw)
    return resume_click._configured_launch_command()


def _hand_written_launcher(monkeypatch, tmp_path, raw: str) -> list[str]:
    """Put ``raw`` in ``config.yml`` DIRECTLY, past the write-time validator.

    Two real states need this: a config written before that validator shipped,
    and one edited by hand. Both reach the click, so the reader's behaviour on
    them is what has to be pinned — through the click's own reader, out of a
    real file, with nothing doubled.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"values": {"desktop": {"launch_command": raw}}})
    )
    return resume_click._configured_launch_command()


def test_an_unparseable_command_line_degrades_to_discovery(monkeypatch, tmp_path):
    """An unbalanced quote must not kill the click.

    Reached only by a hand-edited file now — the writer refuses it (U4) — which
    is exactly why the reader still has to tolerate it.
    """
    assert _hand_written_launcher(monkeypatch, tmp_path, "'unbalanced") == []


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


# ---------------------------------------------------------------------------
# Rung 4: it opens a window, or it says so (UX round 1, U5)
# ---------------------------------------------------------------------------


class _WindowSpawns:
    """Records every process rung 4 started, and the script fed to its stdin.

    Doubled at ``subprocess.Popen`` because BOTH routes land there: the
    AppleScript backends write their script to the child's stdin, and the bare
    detached launch that used to end this rung goes through
    ``proc.spawn_detached``, which is a ``Popen`` too. One recorder therefore
    sees which of the two a click actually chose, which is the whole question
    U5 turns on.
    """

    def __init__(self, monkeypatch) -> None:
        self.argv: list[list[str]] = []
        self.scripts: list[str] = []
        self.waits: list[float | None] = []
        recorder = self

        class _Stdin:
            def write(self, data: bytes) -> None:
                recorder.scripts.append(data.decode("utf-8"))

            def close(self) -> None:
                pass

        class _Process:
            stdin = _Stdin()

            def wait(self, timeout=None) -> int:
                # The child's EXIT STATUS is what ``apple.spawn`` reports since
                # UX round 2 (U11): this double is a real ``osascript`` that
                # opened the window and exited 0. A double with no ``wait`` would
                # fail the rung for the wrong reason.
                recorder.waits.append(timeout)
                return 0

        def fake_popen(argv, **_kwargs):
            recorder.argv.append(list(argv))
            return _Process()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)


def _no_terminal_marker(monkeypatch) -> None:
    """Nothing in this environment identifies a terminal to the registry.

    Doubled at the registry rather than by scrubbing the environment, because
    what the rung has to do with a ``None`` backend is the contract; that a
    stripped environment YIELDS ``None`` is `tests/unit/test_spawn.py`'s.
    """
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: None)


def test_the_last_rung_opens_a_window_it_can_really_open(monkeypatch) -> None:
    """U5: a click must never claim a landing it did not make.

    THE DEFECT, driven. With nothing discoverable the rung fell through to
    ``spawn_detached(["lop", "--resume", <id>])`` and answered True — and that
    child has DEVNULL on all three streams and no terminal attached, so no
    window appeared. True is also what suppresses the receipt, so the user was
    told nothing either: a click indistinguishable from a slow one, on the one
    rung whose entire definition is "there is nothing else to try".

    It now ends on a backend that LAUNCHES a terminal instead of requiring one
    around this process — Terminal.app by AppleScript, which ships with macOS.
    Asserted as the argv a real spawn would run plus the script on its stdin,
    and as the ABSENCE of the bare ``lop --resume`` that used to stand in for a
    window.

    ``sys.platform`` IS FORCED rather than skipped, because the gate this test
    covers is that constant and the unit job runs on ubuntu only — a skipped
    regression guard is the shape of guard this whole change exists to remove.
    Nothing else in the path is platform-specific: the AppleScript backend
    spawns ``osascript`` with a plain ``Popen``, which the recorder catches
    wherever it runs.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    _no_terminal_marker(monkeypatch)
    spawns = _WindowSpawns(monkeypatch)

    assert resume_click._spawn_terminal("a1b2c3d4e5f6") is True
    assert spawns.argv, "nothing was started at all"
    assert spawns.argv[0][0] == "osascript", spawns.argv
    assert spawns.scripts and 'tell application "Terminal"' in spawns.scripts[0]
    # The argv still has to be the resume line: a window that opened somewhere
    # else is not the landing the user was promised.
    assert "--resume" in spawns.argv[0][2]
    assert "a1b2c3d4e5f6" in spawns.argv[0][2]
    bare = [argv for argv in spawns.argv if argv[:2] == ["lop", "--resume"]]
    assert not bare, f"the bare detached launch opened no window: {bare}"


def test_the_last_rung_reports_failure_where_no_window_can_be_promised(monkeypatch) -> None:
    """U5, the other half of the contract: False rather than a false landing.

    ``spawn_detached`` reports whether a child STARTED, so it answered True for
    a click that opened nothing at all. Where no terminal can be launched
    without one around this process — anywhere but macOS — the honest answer is
    False, and ``cli.resume-click`` turns it into the receipt the user needs.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    _no_terminal_marker(monkeypatch)
    spawns = _WindowSpawns(monkeypatch)

    assert resume_click._spawn_terminal("a1b2c3d4e5f6") is False
    assert spawns.argv == []


def test_the_macos_launcher_is_not_offered_over_ssh(monkeypatch) -> None:
    """An ``osascript`` that cannot reach a window server would LIE, not fail.

    It starts, it accepts the script, and the ``tell application`` inside it is
    what fails — so the receipt this rung returns would be the very defect it
    was fixed for, one layer down. ``spawn.fallback`` already draws this exact
    ssh distinction for its message ("no window server over ssh"), so the
    answer here is the same one and for the same reason.

    Driven with the platform forced rather than skipped, so the branch is
    exercised on every platform the suite runs on.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 5000 10.0.0.2 22")
    _no_terminal_marker(monkeypatch)
    spawns = _WindowSpawns(monkeypatch)

    assert resume_click._spawn_terminal("a1b2c3d4e5f6") is False
    assert spawns.argv == []


def test_a_detected_backend_that_refuses_still_reaches_the_visible_fallback(monkeypatch) -> None:
    """A backend that detected and then failed opened NOTHING, so rung 4 has not
    landed yet.

    The detected backend keeps its priority — the user's own terminal wins when
    it is recognised — but an unrecognised-or-failed one is not an answer, and
    an AppleScript that opens Terminal.app is still better than a click that
    reports failure while a window was one line away.
    """

    class _Refusing:
        name = "refusing"

        def spawn(self, launch, env):  # noqa: ANN001
            return False

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda env: _Refusing())
    spawns = _WindowSpawns(monkeypatch)

    assert resume_click._spawn_terminal("a1b2c3d4e5f6") is True
    assert spawns.argv[0][0] == "osascript", spawns.argv


# ---------------------------------------------------------------------------
# The configured launcher: visible when it cannot be used (UX round 1, U4)
# ---------------------------------------------------------------------------


def test_a_launcher_that_cannot_be_run_is_refused_at_write_time(tmp_path, monkeypatch) -> None:
    """U4: the typo is caught while the user is still looking at the field.

    ``desktop.launch_command`` REPLACES discovery — a configured command is the
    user's own answer, not a first try in a chain — so a wrong value sends every
    click to a terminal instead, and the handler that would have noticed runs
    detached from the notification where its only trace was a ``logger.debug``.

    Driven through the REAL writer, the one ``/settings``, ``PATCH
    /v1/settings`` and ``lop config edit`` all funnel through, so no writer can
    store a value the click cannot run.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    setting = settings_io.BY_KEY["desktop.launch_command"]
    manager = ConfigManager(tmp_path)
    # The USER's login PATH is doubled to a directory that cannot contain the
    # names below, so this test does not depend on the machine's login shell (or
    # pay for one): the refusal has to hold when NEITHER path has the name.
    _no_login_path(monkeypatch, tmp_path)

    # The operator's own repro: a path that is not there.
    with pytest.raises(ValueError, match="does not exist"):
        settings_io.write_setting(
            manager, setting, "/tmp/ux-h/bin/does-not-exist --open-session {session}"
        )
    # A bare name that is on neither PATH is the same ``OSError`` at click time,
    # one ``shutil.which`` earlier.
    with pytest.raises(ValueError, match="not on PATH"):
        settings_io.write_setting(manager, setting, "not-an-installed-app")
    # An unbalanced quote parses into nothing, so no candidate is ever built
    # from it and the click silently discovers instead.
    with pytest.raises(ValueError, match="not a valid command line"):
        settings_io.write_setting(manager, setting, "'unbalanced")
    # NOTHING WAS STORED, so the click still discovers the app rather than
    # running a launcher the user cannot use.
    assert settings_io.read_setting(manager, setting) == setting.default

    # ...and a launcher that CAN be run still writes, verbatim.
    settings_io.write_setting(manager, setting, f"{_REAL_LAUNCHER} --open-session {{session}}")
    assert resume_click._configured_launch_command() == [
        _REAL_LAUNCHER,
        "--open-session",
        "{session}",
    ]


def test_a_bare_name_the_writer_cannot_resolve_is_accepted_when_the_user_can(
    tmp_path, monkeypatch
) -> None:
    """M3: the refusal used to answer the WRITER's PATH question, not the user's.

    Driven on the maintainer's machine: with a homebrew PATH the setting's own
    help example (``local-operator-ui --open-session {session}``) wrote fine, and
    with a login-less PATH the SAME value was refused — so a GUI-launched
    settings page, a ``PATCH /v1/settings`` from a service, or ``lop config
    edit`` could not store a launcher the click would have run.

    The name is re-checked against the user's login PATH, and only on the branch
    that is about to refuse (see ``settings_io._user_shell_path``). Both halves
    are asserted: the value is STORED, and the stored argv is what the click
    reader builds from it.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    setting = settings_io.BY_KEY["desktop.launch_command"]
    manager = ConfigManager(tmp_path)
    # The writer's PATH: a directory with nothing in it.
    monkeypatch.setenv("PATH", str(_empty_bin(tmp_path, "writer-bin")))
    # The user's login PATH: a directory holding the npm bin the help names.
    user_bin = _empty_bin(tmp_path, "user-bin")
    _fake_launcher(user_bin, resume_click.DESKTOP_BIN_NAME)
    monkeypatch.setattr(settings_io, "_user_shell_path", lambda: str(user_bin))

    settings_io.write_setting(
        manager, setting, f"{resume_click.DESKTOP_BIN_NAME} --open-session {{session}}"
    )

    assert settings_io.read_setting(manager, setting) == (
        f"{resume_click.DESKTOP_BIN_NAME} --open-session {{session}}"
    )
    assert resume_click._configured_launch_command() == [
        resume_click.DESKTOP_BIN_NAME,
        "--open-session",
        "{session}",
    ]
    # ...and a name on NEITHER path is still refused, which is what keeps the
    # typo guard (U4) intact while the false refusal is gone.
    with pytest.raises(ValueError, match="not on PATH"):
        settings_io.write_setting(manager, setting, "no-such-launcher-anywhere")


def test_every_launcher_rejection_names_a_remedy_in_one_readable_line(
    tmp_path, monkeypatch
) -> None:
    """U12 / D13 and D11's budget, asserted on the strings themselves.

    THE REMEDY IS THE POINT (UX round 2, U12). The rejection REPLACES the row's
    own help while it is on screen, so the sentence that answers "what do I type
    instead" would otherwise be the one thing the error displaced — and this was
    the only rejection on the page that named a fault and a consequence but no
    fix, where the row next door reads "Enter an HTTP or HTTPS server URL…".

    AND IT HAS TO FIT (design round 1, D11). 74 cells is the row's budget at
    80x24, the narrowest width the page measures, so the ADVICE half — fault,
    consequence and remedy — is capped there and is therefore never cut at any
    width the page measures at all. The page sheds the value for that reason
    (``SettingsView._rejection_render``); what is pinned here is that shedding
    the value is always ENOUGH, and that the shape the page relies on is the one
    these messages have.
    """
    from rich.cells import cell_len

    from local_operator import settings_io

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _no_login_path(monkeypatch, tmp_path)
    setting = settings_io.BY_KEY["desktop.launch_command"]
    not_executable = tmp_path / "launcher-not-executable"
    not_executable.write_text("#!/bin/sh\nexit 0\n")
    not_executable.chmod(0o644)

    values = [
        "/tmp/ux-h/bin/does-not-exist --open-session {session}",
        f"{not_executable} --open-session {{session}}",
        "local-operator-ui-not-installed-here",
        # A QUOTED path with a space: one token to the shell grammar the
        # validator parses with, several to ``str.split`` — the shape the
        # page's shed used to fall out of (QA round 1, Q1). It has to be
        # recognised as the interpolated-value shape all the same.
        '"/Applications/Local Operator Canary.app/Contents/MacOS/local-operator-ui"'
        " --open-session {session}",
    ]
    for value in values:
        problem = settings_io.validate(setting, value)
        assert problem, f"{value!r} was accepted"
        split = settings_io.split_value_rejection(problem)
        assert split is not None, problem
        head, advice = split
        # The head is whatever the user typed as the command's first word —
        # `shlex`'s token — so it may contain spaces and must not be
        # re-derived from a token count.
        assert head and advice, problem
        assert cell_len(advice) <= 74, (cell_len(advice), problem)
        # What went wrong, what it costs, and what to do about it.
        assert "clicks open a terminal" in advice, problem
        assert "Clear this to discover the app." in advice, problem
        assert advice.rstrip().endswith("."), problem


def test_empty_is_still_the_default_that_means_discover(tmp_path, monkeypatch) -> None:
    """The validator must not turn "no opinion" into a rejected value.

    ``""`` is the shipped default and it MEANS "discover the app for me", so a
    validator that demanded an executable would make the default un-writable.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    manager = ConfigManager(tmp_path)
    setting = settings_io.BY_KEY["desktop.launch_command"]

    settings_io.write_setting(manager, setting, "")
    assert resume_click._configured_launch_command() == []


def test_a_configured_launcher_that_cannot_run_warns_instead_of_going_quiet(
    tmp_path, monkeypatch, caplog
) -> None:
    """U4's click-time half: a hand-edited ``config.yml`` is still reachable.

    The write-time validator stops the typo at the settings page. A value that
    entered the file by hand — or before that validator shipped — still diverts
    every click, so the handler says so at WARNING rather than leaving the only
    trace in a debug line nobody runs with.
    """
    raw = "/nope/gone --open-session {session}"
    assert _hand_written_launcher(monkeypatch, tmp_path, raw) == [
        "/nope/gone",
        "--open-session",
        "{session}",
    ]
    launcher = _Launcher({"/nope/gone": 1}, monkeypatch)

    with caplog.at_level(logging.WARNING, logger="local_operator.tui.resume_click"):
        assert resume_click._launch_desktop("a1b2c3d4e5f9") is False

    assert launcher.attempts == [["/nope/gone", "--open-session", "a1b2c3d4e5f9"]]
    assert any(
        record.levelno == logging.WARNING and "/nope/gone" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_an_unparseable_hand_written_command_also_warns(tmp_path, monkeypatch, caplog) -> None:
    """The other silent diversion: a value that never parses into argv.

    It degrades to discovery — which is the right behaviour — but a user who
    configured a launcher and is getting something else entirely has to be able
    to find out why.
    """
    assert _hand_written_launcher(monkeypatch, tmp_path, "'unbalanced") == []
    _Launcher({}, monkeypatch)

    with caplog.at_level(logging.WARNING, logger="local_operator.tui.resume_click"):
        assert resume_click._launch_desktop("a1b2c3d4e5fa") is False

    assert any(
        record.levelno == logging.WARNING and "not a valid command line" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------------------------
# The copy on /settings states the order the code takes (UX round 1, U3)
# ---------------------------------------------------------------------------


def test_the_settings_copy_states_the_discovery_order_the_code_takes() -> None:
    """U3: the one surface a user browses to learn the order had it backwards.

    The cursor on **Desktop app -> launch command** read "the packaged bundle,
    then the npm bin" while ``_launch_desktop`` appends the npm bin candidate
    FIRST (``test_the_npm_bin_is_preferred_and_the_flag_shape_is_exact`` pins
    that order) and ``docs/DESKTOP_API.md`` states it the same way as the code.

    Asserted as the ORDER of the two candidates, keyed to the identifiers the
    code actually uses rather than to prose: ``DESKTOP_BIN_NAME`` is the npm
    bin's bin name and must come first, and the bundle candidate exists only on
    darwin, so the copy has to say so (design round 1, D14). A literal sentence
    would pass a rewording that flipped the order back AND would keep the vague
    "the npm bin"/"the packaged bundle" that names neither artifact.

    The BUDGET is asserted against the copy as well: 194 cells of help against a
    `width − 6` slot that is 94 cells at its widest is dead copy at every width
    the page measures (design round 1, D12), so the string has to fit the
    widest one. It is a MINOR here rather than a claim about the frame: the
    rendered line at 100x30 is asserted in
    ``tests/unit/tui/test_settings_view.py``.
    """
    from rich.cells import cell_len

    from local_operator import settings_io

    help_text = settings_io.BY_KEY["desktop.launch_command"].help
    assert resume_click.DESKTOP_BIN_NAME in help_text, help_text
    assert "bundle" not in help_text and "npm bin" not in help_text, help_text
    assert help_text.index(resume_click.DESKTOP_BIN_NAME) < help_text.index("macOS app"), help_text
    # The widest detail row the page paints: a 100-column terminal less the row's
    # own six cells (`SettingsView._detail_width`).
    assert cell_len(help_text) <= 94, (cell_len(help_text), help_text)
