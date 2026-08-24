"""Desktop notifications — who gets told, when, and over which wire.

A notification is the only surface this app has that reaches a user who is
looking at a different application, and it is also the only one that can
INTERRUPT them. Both halves of that are load-bearing, and neither is visible in
a frame Textual paints, so the properties are pinned here:

- **The parent's completion is the event; a subagent's is not.** The harness's
  ``task`` tool returns as soon as a child is registered, so a delegating turn
  reaches ``agent_end`` with the work still running — a toast then is a false
  finish, and one per child is five toasts for one task.
- **Waiting notifies on the EDGE, not on the state.** The window title
  re-asserts ``attention`` on every repaint and coalesces; a notification has
  no equivalent of "already on screen", so an unlatched one fires once per
  repaint of a parked turn.
- **A focused session is never notified.** A toast for the window the user is
  already staring at is pure interruption.
- **Nothing model-written reaches the wire intact.** Session names are
  model-generated and both BEL and ESC terminate an OSC string.
"""

from __future__ import annotations

import subprocess
from typing import Any

from local_operator.tui.notify import (
    APP_NAME,
    BEL,
    BODIES,
    CONTEXTS,
    MAX_TITLE_CHARS,
    OSC9_PREFIX,
    OSC99_PREFIX,
    ST,
    Notifier,
    _spawn_detached,
    argv_safe,
    cmux_command,
    cmux_surface_id,
    desktop_notify_command,
    detect_protocol,
    notification_sequence,
    notification_writes,
    notifications_enabled,
    osc99_id,
    sanitize_text,
    should_use_desktop_fallback,
    wrap_tmux_passthrough,
)


class Sink:
    """Collects what the app would have written to the terminal."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def __call__(self, data: str) -> None:
        self.writes.append(data)

    @property
    def joined(self) -> str:
        return "".join(self.writes)


#: A bare OSC 9 terminal, no cmux and no multiplexer. Injected rather than
#: inherited because this suite itself runs inside cmux, whose
#: ``CMUX_SURFACE_ID`` would otherwise capture every delivery and leave the
#: write sink empty — a green run that proved nothing about the wire.
GHOSTTY_ENV = {"GHOSTTY_RESOURCES_DIR": "/Applications/Ghostty.app/Contents/Resources"}


def unfocused(env: dict[str, str] | None = None, platform: str = "darwin") -> tuple[Notifier, Sink]:
    """A notifier in the only state that delivers: built, enabled, unfocused."""
    sink = Sink()
    notifier = Notifier(sink, env=GHOSTTY_ENV if env is None else env, platform=platform)
    notifier.set_focused(False)
    return notifier, sink


# -- protocol detection ------------------------------------------------------


def test_kitty_is_detected_as_osc99() -> None:
    """OSC 99 is kitty's own protocol and the only one carrying a title/body
    split, so kitty must not be collapsed into the OSC 9 majority."""
    assert detect_protocol({"KITTY_WINDOW_ID": "1"}) == "osc99"
    assert detect_protocol({"TERM": "xterm-kitty"}) == "osc99"


def test_ghostty_wezterm_and_iterm_are_detected_as_osc9() -> None:
    assert detect_protocol({"GHOSTTY_RESOURCES_DIR": "/x"}) == "osc9"
    assert detect_protocol({"WEZTERM_PANE": "0"}) == "osc9"
    assert detect_protocol({"ITERM_SESSION_ID": "w0t0p0"}) == "osc9"
    assert detect_protocol({"TERM_PROGRAM": "WarpTerminal"}) == "osc9"


def test_an_unknown_terminal_falls_back_to_the_bell() -> None:
    """BEL is correct rather than merely safe: an unrecognised terminal that
    silently swallowed OSC 9 would give the user nothing at all."""
    assert detect_protocol({"TERM": "xterm-256color"}) == "bell"
    assert detect_protocol({}) == "bell"


# -- the wire ----------------------------------------------------------------


def test_osc9_carries_the_session_name_and_the_state_on_one_line() -> None:
    """OSC 9 has no title/body split, and a notification centre truncates from
    the right — so the session name leads."""
    sequence = notification_sequence("osc9", "Fix quota reporting", "Task complete")
    assert sequence == f"{OSC9_PREFIX}Fix quota reporting: Task complete{ST}"


def test_osc99_sends_title_and_body_as_two_chunks_sharing_an_id() -> None:
    """kitty groups a notification's payloads by ``i=``; ``d=0`` holds display
    until the body arrives, so the toast is shown once rather than twice."""
    sequence = notification_sequence(
        "osc99", "Triage sanctions", "Waiting for approval", "lo-fixed"
    )
    assert sequence == (
        f"{OSC99_PREFIX}i=lo-fixed:u=1:d=0;Triage sanctions{ST}"
        f"{OSC99_PREFIX}i=lo-fixed:p=body;Waiting for approval{ST}"
    )


def test_osc99_without_a_body_displays_immediately() -> None:
    """No second chunk is coming, so holding display with ``d=0`` would leave
    the notification permanently unshown."""
    assert notification_sequence("osc99", "Only a title", "", "lo-fixed") == (
        f"{OSC99_PREFIX}i=lo-fixed:u=1;Only a title{ST}"
    )


def test_the_bell_protocol_writes_a_bell_and_no_text() -> None:
    """BEL cannot carry text, and writing the words anyway would paint them
    over the frame Textual is drawing."""
    assert notification_sequence("bell", "Session", "Task complete") == BEL


def test_tmux_gets_the_passthrough_envelope_and_a_bell() -> None:
    """tmux does not forward OSC 9/99 to the outer terminal, so an unwrapped
    notification is eaten. The BEL is the belt: it raises ``monitor-bell``
    even when ``allow-passthrough`` is off, which is the only signal a
    backgrounded pane can give."""
    writes = notification_writes("osc9", "Session", "Task complete", in_tmux=True)
    assert len(writes) == 2
    assert writes[0].startswith("\x1bPtmux;")
    assert writes[1] == BEL


def test_the_passthrough_envelope_doubles_every_escape() -> None:
    """tmux reads the first un-doubled ESC as the end of its own DCS, which
    truncates the payload and leaks the remainder onto the screen."""
    wrapped = wrap_tmux_passthrough(f"{OSC9_PREFIX}hello{ST}")
    assert wrapped.startswith("\x1bPtmux;\x1b\x1b]9;hello")
    assert wrapped.endswith("\x1b\\")


def test_zellij_gets_the_sequence_and_a_bell_but_no_envelope() -> None:
    """Zellij has no DCS passthrough to wrap in, but it does raise its ``[!]``
    flag on a bare BEL."""
    writes = notification_writes("osc9", "Session", "Task complete", in_zellij=True)
    assert writes == [f"{OSC9_PREFIX}Session: Task complete{ST}", BEL]


def test_the_bell_protocol_never_rings_twice_in_a_multiplexer() -> None:
    """The sequence IS a BEL under this protocol, so appending the multiplexer
    BEL would ring twice for one event."""
    assert notification_writes("bell", "S", "B", in_tmux=True) == [BEL]
    assert notification_writes("bell", "S", "B", in_zellij=True) == [BEL]


# -- the sanitiser -----------------------------------------------------------


def test_control_characters_are_stripped_from_a_model_written_name() -> None:
    """The security property, shared with the window title: a conversation name
    is model-generated, and both BEL and ESC terminate an OSC string — so a
    name carrying either would close the sequence early and leave the rest
    being executed by the terminal."""
    hostile = "Fix \x07 the\x1b]9;pwned\x07 parser"
    cleaned = sanitize_text(hostile)
    assert "\x07" not in cleaned
    assert "\x1b" not in cleaned
    assert cleaned == "Fix the ]9;pwned parser"


def test_a_hostile_name_cannot_escape_the_sequence_it_is_embedded_in() -> None:
    """End-to-end form of the property above.

    The delivered bytes hold exactly ONE notification: one OSC 9 introducer,
    one ST, and no BEL. The injected ``]9;evil`` survives as inert text, which
    is the point — sanitising is not censorship, it is removing the two bytes
    that would have made the terminal read the rest as a command.
    """
    notifier, sink = unfocused()
    notifier.set_label("done\x1b]9;evil\x07")
    notifier.send("complete")
    payload = sink.joined
    assert payload == f"{OSC9_PREFIX}done ]9;evil: {BODIES['complete']}{ST}"
    assert payload.count(OSC9_PREFIX) == 1
    assert payload.count(ST) == 1
    assert BEL not in payload


def test_titles_are_capped() -> None:
    assert len(sanitize_text("x" * 500)) == MAX_TITLE_CHARS


# -- cmux --------------------------------------------------------------------


def test_only_a_uuid_surface_identifies_a_pane() -> None:
    """A workspace holds several surfaces, so a notification sent against one
    would name the wrong pane; the surface UUID is the only marker that ties a
    toast to the session that raised it."""
    real = "773d5e5e-1111-4222-8333-444455556666"
    assert cmux_surface_id({"CMUX_SURFACE_ID": real}) == real
    assert cmux_surface_id({"CMUX_WORKSPACE_ID": real}) is None
    assert cmux_surface_id({"CMUX_SURFACE_ID": "not-a-uuid"}) is None
    assert cmux_surface_id({}) is None


def test_the_cmux_command_targets_the_surface_with_structured_context() -> None:
    surface = "773d5e5e-1111-4222-8333-444455556666"
    assert cmux_command(surface, "Fix quota reporting", "Complete", "Task complete") == [
        "cmux",
        "notify",
        "--surface",
        surface,
        "--title",
        "Fix quota reporting",
        "--subtitle",
        "Complete",
        "--body",
        "Task complete",
    ]


def test_cmux_wins_over_the_in_band_escape(monkeypatch: Any) -> None:
    """cmux hosts Ghostty, so both paths would otherwise fire and the user
    would be told twice about one event. The env below is exactly that: a
    Ghostty terminal that is ALSO a cmux surface, which is the real shape of
    the machine this is developed on."""
    surface = "773d5e5e-1111-4222-8333-444455556666"
    spawned: list[list[str]] = []
    monkeypatch.setattr("local_operator.tui.notify._spawn_detached", spawned.append)
    notifier, sink = unfocused({**GHOSTTY_ENV, "CMUX_SURFACE_ID": surface})
    assert notifier.send("complete") is True
    assert sink.writes == []
    assert spawned == [cmux_command(surface, APP_NAME, CONTEXTS["complete"], BODIES["complete"])]


# -- the Linux D-Bus fallback ------------------------------------------------


def test_the_dbus_fallback_is_only_for_terminals_that_cannot_carry_text() -> None:
    """A terminal that speaks OSC 9/99 already delivered the toast; a second
    one over D-Bus would duplicate it."""
    bus = {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/bus"}
    assert should_use_desktop_fallback("bell", "linux", bus) is True
    assert should_use_desktop_fallback("osc9", "linux", bus) is False
    assert should_use_desktop_fallback("osc99", "linux", bus) is False


def test_the_dbus_fallback_is_linux_only() -> None:
    """macOS terminals all speak OSC 9 or run under cmux, and there is no
    session bus to reach."""
    bus = {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/bus"}
    assert should_use_desktop_fallback("bell", "darwin", bus) is False


def test_the_dbus_fallback_needs_a_reachable_session_bus() -> None:
    assert should_use_desktop_fallback("bell", "linux", {}) is False


def test_the_notify_send_command_is_transient_and_attributed() -> None:
    argv = desktop_notify_command("/usr/bin/notify-send", "Session", "Task complete")
    assert argv[0] == "/usr/bin/notify-send"
    assert "--app-name" in argv and APP_NAME in argv
    assert "--urgency=normal" in argv
    assert argv[-2:] == ["Session", "Task complete"]


# -- who gets told, and when -------------------------------------------------


def test_a_focused_session_is_never_notified() -> None:
    """The user is looking at the frame that already says this."""
    sink = Sink()
    notifier = Notifier(sink, env=GHOSTTY_ENV)  # starts focused
    assert notifier.notify_turn_complete(running_children=0) is False
    assert sink.writes == []


def test_the_notifier_starts_focused() -> None:
    """Textual reports focus only on a CHANGE, so assuming unfocused would
    notify on the very first turn of every session."""
    assert Notifier(Sink(), env=GHOSTTY_ENV).focused is True


def test_a_finished_turn_notifies_once_the_terminal_is_unfocused() -> None:
    notifier, sink = unfocused()
    assert notifier.notify_turn_complete(running_children=0) is True
    assert BODIES["complete"] in sink.joined


def test_a_turn_that_ends_with_children_still_running_stays_quiet() -> None:
    """The ``task`` tool returns as soon as a child is registered, so a
    delegating parent reaches ``agent_end`` with the work still going. The
    model stopped talking; the task did not finish."""
    notifier, sink = unfocused()
    assert notifier.notify_turn_complete(running_children=2) is False
    assert sink.writes == []


def test_the_completion_lands_when_the_last_child_has_settled() -> None:
    """Nothing is lost by staying quiet above: each settled job re-enters the
    conversation as a fresh turn whose own completion is notifiable."""
    notifier, sink = unfocused()
    notifier.notify_turn_complete(running_children=1)
    assert sink.writes == []
    assert notifier.notify_turn_complete(running_children=0) is True
    assert BODIES["complete"] in sink.joined


def test_a_parked_turn_notifies_even_with_children_running() -> None:
    """An unanswered approval blocks THIS turn no matter what else is running,
    and it is the case where a missed notification costs the most: the session
    sits parked indefinitely."""
    notifier, sink = unfocused()
    assert notifier.notify_waiting("approval") is True
    assert BODIES["approval"] in sink.joined


def test_each_kind_says_which_state_it_is() -> None:
    """Four states have distinct details and actionable context: completion is
    unlike a blocked turn, while ask and approval share the need for input."""
    assert len(set(BODIES.values())) == len(BODIES)
    assert CONTEXTS == {
        "complete": "Complete",
        "approval": "Input required",
        "ask": "Input required",
        "error": "Needs attention",
    }
    for kind in ("complete", "approval", "ask", "error"):
        notifier, sink = unfocused()
        notifier.send(kind)  # type: ignore[arg-type]
        assert BODIES[kind] in sink.joined


def test_the_session_name_titles_the_toast() -> None:
    """A user with five sessions open otherwise gets five identical toasts —
    the same failure the window title was built to fix."""
    notifier, sink = unfocused()
    notifier.set_label("Fix quota reporting")
    notifier.send("complete")
    assert "Fix quota reporting" in sink.joined


def test_an_unnamed_session_falls_back_to_the_product_name() -> None:
    notifier, sink = unfocused()
    notifier.send("complete")
    assert APP_NAME in sink.joined


def test_a_disabled_notifier_accepts_state_and_writes_nothing() -> None:
    """A null object rather than an ``Optional`` at each call site, matching
    ``TerminalTitle``."""
    sink = Sink()
    notifier = Notifier(sink, enabled=False, env=GHOSTTY_ENV)
    notifier.set_focused(False)
    notifier.set_label("Fix quota reporting")
    assert notifier.enabled is False
    assert notifier.notify_turn_complete(running_children=0) is False
    assert notifier.notify_waiting("ask") is False
    assert sink.writes == []


# -- the gates ---------------------------------------------------------------


def test_the_env_kill_switch_turns_notifications_off(monkeypatch: Any) -> None:
    """Wanted by anything recording raw terminal output, and by a user who does
    not want to be interrupted, without editing config."""
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda *a, **k: True)
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    assert notifications_enabled() is False


def test_the_config_flag_turns_notifications_off(monkeypatch: Any) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda *a, **k: False)
    assert notifications_enabled() is False


def test_notifications_are_on_by_default(monkeypatch: Any) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default: default)
    assert notifications_enabled() is True


# -- round 1 review: the wire fixes ------------------------------------------


def test_each_osc99_notification_gets_its_own_id() -> None:
    """kitty treats `i=` as an identity: a reused id REPLACES the toast on
    screen. A constant id let a later completion silently overwrite an
    unanswered approval, and made every session on the machine collide."""
    first = notification_sequence("osc99", "Session", "Waiting for approval")
    second = notification_sequence("osc99", "Session", "Task complete")
    assert _osc99_ids(first) != _osc99_ids(second)


def test_the_two_chunks_of_one_notification_share_their_id() -> None:
    """Title and body are one notification; kitty groups them BY that id, so
    differing ids would show two toasts, one of them bodiless."""
    ids = _osc99_ids(notification_sequence("osc99", "Session", "Task complete"))
    assert len(ids) == 2 and ids[0] == ids[1]


def test_osc99_ids_use_only_characters_the_spec_allows() -> None:
    import re

    for _ in range(20):
        assert re.fullmatch(r"[a-zA-Z0-9_+\-.]+", osc99_id())


def _osc99_ids(sequence: str) -> list[str]:
    import re

    return re.findall(r"\x1b\]99;i=([^:;]+)", sequence)


def test_a_dash_leading_session_name_cannot_become_a_notify_send_option() -> None:
    """Sanitisation closes the ESCAPE hole but not the string's SHAPE: a
    model-written name like `--help` or `-u critical` otherwise lands in
    notify-send's option namespace instead of its summary."""
    argv = desktop_notify_command("/usr/bin/notify-send", "--help", "Task complete")
    assert "--" in argv
    assert argv.index("--") < argv.index("--help")
    assert argv[-2:] == ["--help", "Task complete"]


def test_a_dash_leading_session_name_is_neutralised_for_cmux() -> None:
    surface = "773d5e5e-1111-4222-8333-444455556666"
    argv = cmux_command(surface, "-u critical", "Complete", "Task complete")
    assert not argv[argv.index("--title") + 1].startswith("-")


def test_argv_safe_leaves_ordinary_titles_untouched() -> None:
    """It must not mangle the common case: this is a user-facing toast title."""
    assert argv_safe("Fix quota reporting") == "Fix quota reporting"


def test_the_notifier_subprocess_cannot_stall_the_render_loop(monkeypatch: Any) -> None:
    """The hardening `_spawn_detached`'s docstring promises, pinned.

    A notifier is fire-and-forget: without `start_new_session` and fully
    redirected stdio, a hung D-Bus activation or a cmux socket mid-restart
    holds the TUI's loop, and the child's output interleaves into the frame
    Textual is painting.
    """
    seen: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["argv"] = argv
        return object()

    monkeypatch.setattr("local_operator.tui.notify.subprocess.Popen", fake_popen)
    _spawn_detached(["notify-send", "hi"])
    assert seen["start_new_session"] is True
    assert seen["stdin"] == subprocess.DEVNULL
    assert seen["stdout"] == subprocess.DEVNULL
    assert seen["stderr"] == subprocess.DEVNULL


def test_a_failing_notifier_spawn_is_silent(monkeypatch: Any) -> None:
    """The user asked for a task, not for a toast: a missing binary must not
    surface as an error in the session."""

    def boom(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("notify-send")

    monkeypatch.setattr("local_operator.tui.notify.subprocess.Popen", boom)
    _spawn_detached(["notify-send", "hi"])  # must not raise
