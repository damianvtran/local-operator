"""The fork spawn registry — the safety rules, not the plumbing.

Modelled on ``test_multiplexer.py``: what is pinned here is the set of
properties whose failure is invisible until a user is looking at the wrong
window, or at no window at all.

**No test in this file spawns anything.** Every backend ends in a subprocess
against a GUI application, so the argv is asserted as an exact list and the
verification stops there. A test that mocked ``Popen`` and called that coverage
would assert the mock; whether a window actually opens is a manual check, and
the PR records which emulators were genuinely exercised and which were not.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.multiplexer.broadcast import RESUME_FLAG, resume_argv
from local_operator.spawn.apple import ITerm2Backend, TerminalAppBackend, launch_command
from local_operator.spawn.cmux import (
    PLACEMENT_SURFACE,
    PLACEMENT_WORKSPACE,
    CmuxBackend,
    send_argv,
    surface_argv,
    workspace_argv,
)
from local_operator.spawn.fallback import fallback_receipt
from local_operator.spawn.ghostty import linux_argv, macos_argv
from local_operator.spawn.kitty import direct_argv, remote_argv
from local_operator.spawn.registry import active_backend, backends
from local_operator.spawn.types import ForkLaunch
from local_operator.spawn.wezterm import cli_spawn_argv, start_argv

FORK_ID = "abc123def456"
WORKSPACE_UUID = "11111111-2222-3333-4444-555555555555"
SURFACE_UUID = "66666666-7777-8888-9999-000000000000"

CMUX_ENV = {"CMUX_WORKSPACE_ID": WORKSPACE_UUID, "CMUX_SURFACE_ID": SURFACE_UUID}


@pytest.fixture
def launch() -> ForkLaunch:
    return ForkLaunch(
        session_id=FORK_ID,
        executable="/Users/x/.local/bin/lop",
        argv=("/Users/x/.local/bin/lop", RESUME_FLAG, FORK_ID),
        cwd="/Users/x/projects/thing",
        title="fork · Refactor the loader",
    )


class TestDetectionOrder:
    def test_cmux_wins_over_ghostty_when_both_are_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T10. The real-world case, and the one a naive order gets wrong.

        cmux EMBEDS ghostty and exports ``GHOSTTY_RESOURCES_DIR`` in every
        surface, so both marker sets are present in an ordinary cmux session
        (verified on the development host). A ghostty-first registry would open
        a stray OS window instead of the sidebar workspace the user expects.
        """
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: "/opt/bin/cmux")
        env = {**CMUX_ENV, "GHOSTTY_RESOURCES_DIR": "/Applications/Ghostty.app/Contents"}

        backend = active_backend(env)

        assert backend is not None and backend.name == "cmux"

    def test_ghostty_is_selected_when_there_is_no_cmux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: None)
        backend = active_backend({"GHOSTTY_RESOURCES_DIR": "/x"})
        assert backend is not None and backend.name == "ghostty"

    def test_cmux_markers_without_a_binary_are_not_cmux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ssh-hop case. Every ``CMUX_*`` variable is inherited across a hop
        into a host with no cmux CLI, so the BINARY is the gate — otherwise a
        fork there would spawn a subprocess that can only fail.
        """
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: None)
        assert active_backend(dict(CMUX_ENV)) is None

    def test_no_markers_at_all_selects_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """T15. A bare tty is an ORDINARY outcome, not an error."""
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: None)
        assert active_backend({"TERM": "xterm-256color"}) is None

    def test_a_backend_whose_detect_raises_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T14. A backend author's bug costs that backend, never the fork."""

        def boom(self: Any, env: Any) -> bool:
            raise RuntimeError("backend bug")

        monkeypatch.setattr(CmuxBackend, "detect", boom)
        backend = active_backend({"GHOSTTY_RESOURCES_DIR": "/x"})
        assert backend is not None and backend.name == "ghostty"

    def test_every_backend_declares_the_protocol(self) -> None:
        for backend in backends():
            assert isinstance(backend.name, str) and backend.name
            assert callable(backend.detect)
            assert callable(backend.spawn)


class TestTheLaunchLineIsRestoreAndIdle:
    def test_the_argv_carries_no_prompt_and_no_exec(self, launch: ForkLaunch) -> None:
        """T13. The safety boundary ``resume_argv`` owns, pinned for forks too.

        A fork's opening message rides a sidecar in the fork's own session
        directory precisely so this argv stays a restore-and-idle command: it is
        one copy-paste from a crash-restore binding, which replays unattended to
        every pane at once.
        """
        assert launch.argv == tuple(resume_argv(FORK_ID, launch.executable))
        assert list(launch.argv) == [launch.executable, RESUME_FLAG, FORK_ID]
        joined = " ".join(launch.argv)
        assert "--exec" not in joined
        assert "--prompt" not in joined

    @pytest.mark.parametrize(
        "argv_builder",
        [
            lambda launch: workspace_argv("/opt/bin/cmux", launch),
            lambda launch: surface_argv("/opt/bin/cmux", launch),
            lambda launch: macos_argv(launch),
            lambda launch: linux_argv("/usr/bin/ghostty", launch),
            lambda launch: cli_spawn_argv("/usr/bin/wezterm", launch),
            lambda launch: start_argv("/usr/bin/wezterm", launch),
            lambda launch: remote_argv("/usr/bin/kitty", launch),
            lambda launch: direct_argv("/usr/bin/kitty", launch),
        ],
        ids=[
            "cmux-workspace",
            "cmux-surface",
            "ghostty-macos",
            "ghostty-linux",
            "wezterm-cli",
            "wezterm-start",
            "kitty-remote",
            "kitty-direct",
        ],
    )
    def test_no_backend_can_smuggle_a_prompt_onto_the_command_line(
        self, launch: ForkLaunch, argv_builder: Any
    ) -> None:
        """Every constructed argv, checked for the one thing none may carry.

        Parametrised over the builders rather than asserted once, because the
        invariant is about the SET of backends: a new one added without this
        discipline is the way it would be lost.
        """
        rendered = " ".join(argv_builder(launch))
        assert "--exec" not in rendered
        assert "--prompt" not in rendered


class TestCmuxArgv:
    def test_the_workspace_argv_is_exact(self, launch: ForkLaunch) -> None:
        """T11. Asserted as an exact list, the same discipline as
        ``notify.cmux_command``."""
        assert workspace_argv("/opt/bin/cmux", launch) == [
            "/opt/bin/cmux",
            "new-workspace",
            "--name",
            "fork · Refactor the loader",
            "--cwd",
            "/Users/x/projects/thing",
            "--command",
            "/Users/x/.local/bin/lop --resume abc123def456",
            "--focus",
            "false",
        ]

    def test_the_surface_argv_is_exact(self, launch: ForkLaunch) -> None:
        assert surface_argv("/opt/bin/cmux", launch) == [
            "/opt/bin/cmux",
            "new-surface",
            "--type",
            "terminal",
            "--working-directory",
            "/Users/x/projects/thing",
            "--focus",
            "false",
        ]

    def test_the_send_argv_presses_enter(self, launch: ForkLaunch) -> None:
        """The trailing newline IS the Enter press. Without it the launch line
        sits typed-but-unrun in the new tab, which looks like a hung fork."""
        argv = send_argv("/opt/bin/cmux", "surface:142", launch)
        assert argv[:4] == ["/opt/bin/cmux", "send", "--surface", "surface:142"]
        assert argv[4].endswith("\n")
        assert "--resume abc123def456" in argv[4]

    @pytest.mark.parametrize(
        "argv_builder",
        [workspace_argv, surface_argv],
        ids=["new-workspace", "new-surface"],
    )
    def test_every_cmux_argv_passes_focus_false(
        self, launch: ForkLaunch, argv_builder: Any
    ) -> None:
        """T12. A standing rule, and it deserves a test that fails by name.

        cmux's socket gate only permits focus, window raise and workspace
        switching when a command carries an explicitly truthy ``focus``. A fork
        is something the user asked for while working somewhere else, so it must
        never yank the window they are typing in.
        """
        argv = argv_builder("/opt/bin/cmux", launch)
        assert "--focus" in argv
        assert argv[argv.index("--focus") + 1] == "false"

    def test_a_launcher_path_with_a_space_survives(self) -> None:
        """``shlex.join`` and not ``" ".join``: cmux re-tokenises the command
        string, so an unquoted path with a space would come back as two
        arguments and restore nothing."""
        launch = ForkLaunch(
            session_id=FORK_ID,
            executable="/Applications/My Tools/lop",
            argv=("/Applications/My Tools/lop", RESUME_FLAG, FORK_ID),
            cwd="/Users/x/my projects/thing",
            title="fork",
        )
        argv = workspace_argv("/opt/bin/cmux", launch)
        command = argv[argv.index("--command") + 1]
        assert command == "'/Applications/My Tools/lop' --resume abc123def456"


class TestEmulatorArgv:
    def test_ghostty_on_macos_uses_open_not_the_cli(self, launch: ForkLaunch) -> None:
        """Ghostty's own --help: on macOS, launching the emulator from the CLI
        is NOT supported. ``ghostty -e`` there is a confusing partial failure."""
        assert macos_argv(launch) == [
            "open",
            "-na",
            "Ghostty.app",
            "--args",
            "--working-directory=/Users/x/projects/thing",
            "-e",
            "/Users/x/.local/bin/lop",
            "--resume",
            FORK_ID,
        ]

    def test_ghostty_on_linux_uses_the_cli(self, launch: ForkLaunch) -> None:
        assert linux_argv("/usr/bin/ghostty", launch) == [
            "/usr/bin/ghostty",
            "--working-directory=/Users/x/projects/thing",
            "-e",
            "/Users/x/.local/bin/lop",
            "--resume",
            FORK_ID,
        ]

    def test_wezterm_prefers_the_mux_then_falls_back(self, launch: ForkLaunch) -> None:
        assert cli_spawn_argv("/usr/bin/wezterm", launch) == [
            "/usr/bin/wezterm",
            "cli",
            "spawn",
            "--new-window",
            "--cwd",
            "/Users/x/projects/thing",
            "--",
            "/Users/x/.local/bin/lop",
            "--resume",
            FORK_ID,
        ]
        assert start_argv("/usr/bin/wezterm", launch) == [
            "/usr/bin/wezterm",
            "start",
            "--cwd",
            "/Users/x/projects/thing",
            "--",
            "/Users/x/.local/bin/lop",
            "--resume",
            FORK_ID,
        ]

    def test_kitty_remote_and_direct_forms(self, launch: ForkLaunch) -> None:
        assert remote_argv("/usr/bin/kitty", launch)[:4] == [
            "/usr/bin/kitty",
            "@",
            "launch",
            "--type=os-window",
        ]
        assert direct_argv("/usr/bin/kitty", launch) == [
            "/usr/bin/kitty",
            "--directory",
            "/Users/x/projects/thing",
            "/Users/x/.local/bin/lop",
            "--resume",
            FORK_ID,
        ]

    def test_the_applescript_command_quotes_the_working_directory(self) -> None:
        """A project path with a space in it is ordinary; an unquoted ``cd``
        would land somewhere else entirely."""
        launch = ForkLaunch(
            session_id=FORK_ID,
            executable="/Users/x/.local/bin/lop",
            argv=("/Users/x/.local/bin/lop", RESUME_FLAG, FORK_ID),
            cwd="/Users/x/my projects/thing",
            title="fork",
        )
        assert launch_command(launch) == (
            "cd '/Users/x/my projects/thing' && /Users/x/.local/bin/lop --resume abc123def456"
        )

    def test_the_apple_backends_detect_independently(self) -> None:
        assert TerminalAppBackend().detect({"TERM_PROGRAM": "Apple_Terminal"}) is True
        assert TerminalAppBackend().detect({"ITERM_SESSION_ID": "w0t0p0"}) is False
        assert ITerm2Backend().detect({"ITERM_SESSION_ID": "w0t0p0"}) is True
        assert ITerm2Backend().detect({"TERM_PROGRAM": "Apple_Terminal"}) is False


class TestTheCmuxPlacementIsConfigurable:
    def test_the_placement_selects_which_call_is_made(
        self, monkeypatch: pytest.MonkeyPatch, launch: ForkLaunch
    ) -> None:
        """The workspace form is ONE call that carries the command; the surface
        form needs two, which is why it is not the default."""
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: "/opt/bin/cmux")
        spawned: list[list[str]] = []
        monkeypatch.setattr(
            "local_operator.spawn.cmux.spawn_detached",
            lambda argv, **kwargs: spawned.append(list(argv)) or True,
        )

        CmuxBackend(PLACEMENT_WORKSPACE).spawn(launch, dict(CMUX_ENV))

        assert len(spawned) == 1
        assert spawned[0][1] == "new-workspace"

    def test_an_unknown_placement_falls_back_to_the_workspace_form(
        self, monkeypatch: pytest.MonkeyPatch, launch: ForkLaunch
    ) -> None:
        """A config typo costs the non-default behaviour, never the fork."""
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: "/opt/bin/cmux")
        spawned: list[list[str]] = []
        monkeypatch.setattr(
            "local_operator.spawn.cmux.spawn_detached",
            lambda argv, **kwargs: spawned.append(list(argv)) or True,
        )

        CmuxBackend("nonsense").spawn(launch, dict(CMUX_ENV))

        assert spawned[0][1] == "new-workspace"

    def test_the_registry_threads_the_placement_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("local_operator.spawn.cmux.cmux_binary", lambda: "/opt/bin/cmux")
        backend = active_backend(dict(CMUX_ENV), cmux_placement=PLACEMENT_SURFACE)
        assert isinstance(backend, CmuxBackend)
        assert backend.placement == PLACEMENT_SURFACE


class TestTheFallbackReceipt:
    def test_it_names_the_fork_and_the_command_that_reaches_it(self) -> None:
        """The fork is NEVER lost on this path: the id is in the receipt, the
        session is in the picker, and ``lop --resume <id>`` works."""
        receipt = fallback_receipt(FORK_ID, {"TERM": "dumb"})
        assert FORK_ID in receipt
        assert f"lop --resume {FORK_ID}" in receipt

    def test_ssh_is_named_explicitly(self) -> None:
        """The user knows why there is no window server, and a message that
        says so reads as competent rather than broken."""
        receipt = fallback_receipt(FORK_ID, {"SSH_CONNECTION": "10.0.0.1 22 10.0.0.2 22"})
        assert "ssh" in receipt.lower()
        assert f"lop --resume {FORK_ID}" in receipt

    def test_a_failed_spawn_reads_differently_from_no_backend(self) -> None:
        """Two different situations, two different sentences: "the window did
        not open" versus "this terminal cannot open one"."""
        failed = fallback_receipt(FORK_ID, {"TERM": "dumb"}, failed=True)
        none = fallback_receipt(FORK_ID, {"TERM": "dumb"})
        assert failed != none
        assert FORK_ID in failed and f"lop --resume {FORK_ID}" in failed


class TestPolicyDefaults:
    def test_the_defaults_are_what_the_registry_advertises(self) -> None:
        """The anti-drift test compares these; this pins the reading side."""
        from local_operator import settings_io
        from local_operator.spawn.policy import (
            DEFAULT_FORK_CMUX_PLACEMENT,
            DEFAULT_FORK_MODE,
            fork_cmux_placement,
            fork_mode,
        )

        assert settings_io.BY_KEY["fork.mode"].default == DEFAULT_FORK_MODE
        assert settings_io.BY_KEY["fork.cmux_placement"].default == DEFAULT_FORK_CMUX_PLACEMENT
        assert fork_mode(None) == DEFAULT_FORK_MODE
        assert fork_cmux_placement(None) == DEFAULT_FORK_CMUX_PLACEMENT

    def test_the_paths_are_genuinely_nested(self) -> None:
        """THE trap next door. ``fork.mode`` is a two-element path, not a flat
        dotted key — declaring it flat would write a key nothing reads while
        looking like success from every angle."""
        from local_operator import settings_io

        assert settings_io.BY_KEY["fork.mode"].path == ("fork", "mode")
        assert settings_io.BY_KEY["fork.cmux_placement"].path == ("fork", "cmux_placement")
        assert "fork.mode" not in settings_io.flat_dotted_keys()

    def test_config_values_are_read_from_the_nested_path(self) -> None:
        from local_operator.spawn.policy import fork_cmux_placement, fork_mode

        values = {"fork": {"mode": "switch", "cmux_placement": "surface"}}
        assert fork_mode(values) == "switch"
        assert fork_cmux_placement(values) == "surface"

    @pytest.mark.parametrize(
        "values",
        [
            {"fork": "switch"},  # a string where a mapping belongs
            {"fork": {"mode": "sideways"}},  # not a member of the enum
            {"fork": {"mode": None}},
            {},
        ],
        ids=["not-a-mapping", "unknown-value", "null", "absent"],
    )
    def test_a_malformed_config_degrades_to_the_default(self, values: Any) -> None:
        """This reads a hand-editable YAML file; a typo must not raise on the
        command path."""
        from local_operator.spawn.policy import DEFAULT_FORK_MODE, fork_mode

        assert fork_mode(values) == DEFAULT_FORK_MODE


class TestTheReceiptNamesThePlaceNotTheBackend:
    """D3/U5. The receipt is the only thing pointing at an unfocused fork.

    Deriving the noun from ``backend.name`` told every user a "window" opened.
    Under cmux the default placement is a WORKSPACE — a sidebar row in the
    window they already have — so a user who chose `surface` went looking for a
    window that does not exist, and the AppleScript backends produced "a new
    applescript window", naming a mechanism rather than a place.
    """

    def test_every_backend_names_a_place(self) -> None:
        for backend in backends():
            place = backend.opened_place
            assert place and place[0].islower(), f"{backend.name}: {place!r}"
            assert "applescript" not in place.lower()

    def test_cmux_names_the_placement_it_actually_used(self) -> None:
        assert CmuxBackend(PLACEMENT_WORKSPACE).opened_place == "a new cmux workspace"
        assert CmuxBackend(PLACEMENT_SURFACE).opened_place == "a new surface in this workspace"

    def test_no_backend_calls_a_cmux_placement_a_window(self) -> None:
        """The specific wrong word, pinned so it cannot come back."""
        for placement in (PLACEMENT_WORKSPACE, PLACEMENT_SURFACE):
            assert "window" not in CmuxBackend(placement).opened_place


class TestTerminalAppDoesNotStealFocus:
    """R5/D-focus. `activate` raised Terminal.app over whatever the user was
    typing in, and iTerm2's script never had it."""

    def test_the_terminal_script_does_not_activate(self) -> None:
        from local_operator.spawn.apple import ITERM_SCRIPT, TERMINAL_SCRIPT

        assert "activate" not in TERMINAL_SCRIPT
        assert "activate" not in ITERM_SCRIPT
