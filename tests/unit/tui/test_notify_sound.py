"""The notification sound rides the banner (notification-sounds design §4.1).

Pins three things in ``tui/notify.py``: the sound is asked for only after a
banner leg was taken (never for a focused session, never when banners are
off); kitty's OSC 99 carries ``s=silent`` (base64 ``c2lsZW50``) when lop plays
its own sound or the sound is off, and not when ``system`` is selected;
notify-send gets ``suppress-sound`` under the same rule.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.notifications import sound
from local_operator.tui import notify
from local_operator.tui.notify import (
    SUPPRESS_SOUND_HINT,
    Notifier,
    desktop_notify_command,
    osc99_sequence,
)

KITTY_ENV = {"KITTY_WINDOW_ID": "1", "TERM": "xterm-kitty"}


class Sink:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def __call__(self, data: str) -> None:
        self.writes.append(data)


@pytest.fixture
def played(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    monkeypatch.setattr(notify, "desktop_belongs_to_this_process", lambda: True)
    monkeypatch.setattr(notify, "_play_sound", lambda plan, leg: calls.append((plan.kind, leg)))
    return calls


def _use(monkeypatch: pytest.MonkeyPatch, **values: Any) -> None:
    plan = sound.resolve({"notifications": {"sound": values}}, env={})
    monkeypatch.setattr(notify, "_sound_plan", lambda: plan)


def test_osc99_silent_rides_the_first_chunk_only() -> None:
    wire = osc99_sequence("T", "B", "lo-x", silent=True)
    assert wire.count("s=c2lsZW50") == 1
    assert wire.startswith("\x1b]99;i=lo-x:u=1:s=c2lsZW50:d=0;T")
    assert "s=" not in osc99_sequence("T", "B", "lo-x")


def test_notify_send_suppresses_its_sound_only_when_asked() -> None:
    assert SUPPRESS_SOUND_HINT in desktop_notify_command("n", "T", "B", suppress_sound=True)
    assert SUPPRESS_SOUND_HINT not in desktop_notify_command("n", "T", "B")


def test_an_unfocused_kitty_banner_is_silenced_and_lop_plays(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch)
    sink = Sink()
    notifier = Notifier(sink, env=KITTY_ENV, platform="darwin")
    assert notifier.notify_turn_complete(running_children=0) is True
    assert "s=c2lsZW50" in "".join(sink.writes)
    assert played == [("file", "osc")]


def test_a_focused_session_gets_neither_banner_nor_sound(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch)
    notifier = Notifier(Sink(), env=KITTY_ENV, platform="darwin")
    notifier.set_focused(True)
    assert notifier.notify_turn_complete(running_children=0) is False
    assert played == []


def test_sound_off_still_silences_kitty(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch, enabled=False)
    sink = Sink()
    Notifier(sink, env=KITTY_ENV, platform="darwin").notify_turn_complete(running_children=0)
    assert "s=c2lsZW50" in "".join(sink.writes)
    # The plan is handed over and the player refuses a silent plan itself.
    assert played == [("silent", "osc")]
    assert (
        sound.play_plan(
            sound.resolve({"notifications": {"sound": {"enabled": False}}}, env={}), leg="osc"
        )
        is False
    )


def test_system_leaves_kitty_sound_alone(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch, id="system")
    sink = Sink()
    Notifier(sink, env=KITTY_ENV, platform="darwin").notify_turn_complete(running_children=0)
    assert "s=" not in "".join(sink.writes)


def test_banners_off_means_no_sound(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch)
    env = {**KITTY_ENV, "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1"}
    assert Notifier(Sink(), env=env).notify_turn_complete(running_children=0) is False
    assert played == []


def test_cmux_leg_plays_after_the_toast(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch)
    spawned: list[list[str]] = []
    monkeypatch.setattr(notify, "_spawn_detached", spawned.append)
    env = {
        "CMUX_SURFACE_ID": "6f1d7a3e-0000-4000-8000-000000000001",
        "CMUX_WORKSPACE_ID": "6f1d7a3e-0000-4000-8000-000000000002",
    }
    assert Notifier(Sink(), env=env).notify_turn_complete(running_children=0) is True
    assert spawned and played == [("file", "cmux")]


def test_detached_notify_plays_only_when_a_banner_child_started(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch)
    monkeypatch.setattr(notify, "_raise_detached_banner", lambda *_a: False)
    assert notify.detached_notify("T", "B") is False
    assert played == []
    monkeypatch.setattr(notify, "_raise_detached_banner", lambda *_a: True)
    assert notify.detached_notify("T", "B") is True
    assert played == [("file", "detached")]


def test_detached_linux_banner_carries_the_hint(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch)
    spawned: list[list[str]] = []
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(notify, "_spawn_detached_ok", lambda argv: spawned.append(argv) or True)
    assert notify.detached_notify("T", "B") is True
    assert SUPPRESS_SOUND_HINT in spawned[0]


# -- M1 (review round 1): lop's sound replaces the bare BEL, never the pane flag --

APPLE_TERMINAL_ENV = {"TERM_PROGRAM": "Apple_Terminal"}
UNKNOWN_TERM_ENV = {"TERM": "xterm-unknown-thing"}


@pytest.mark.parametrize("env", [APPLE_TERMINAL_ENV, UNKNOWN_TERM_ENV], ids=["apple", "unknown"])
def test_lops_sound_replaces_the_bel_on_a_bell_only_terminal(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    _use(monkeypatch)
    sink = Sink()
    notifier = Notifier(sink, env=env, platform="darwin")
    assert notifier.protocol == "bell"
    assert notifier.notify_turn_complete(running_children=0) is True
    assert sink.writes == []
    assert played == [("file", "osc")]


@pytest.mark.parametrize("env", [APPLE_TERMINAL_ENV, UNKNOWN_TERM_ENV], ids=["apple", "unknown"])
def test_system_keeps_the_bel_on_a_bell_only_terminal(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    _use(monkeypatch, id="system")
    sink = Sink()
    Notifier(sink, env=env, platform="darwin").notify_turn_complete(running_children=0)
    assert sink.writes == ["\a"]


def test_sound_off_keeps_the_bel(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off means "no sound from lop", not "drop the terminal's own signal"."""
    _use(monkeypatch, enabled=False)
    sink = Sink()
    Notifier(sink, env=APPLE_TERMINAL_ENV, platform="darwin").notify_turn_complete(
        running_children=0
    )
    assert sink.writes == ["\a"]


def test_a_rig_keeps_the_bel_because_lop_will_not_play(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rig guard refuses lop's sound, so the BEL must survive."""
    _use(monkeypatch)
    monkeypatch.setattr(notify, "desktop_belongs_to_this_process", lambda: False)
    sink = Sink()
    Notifier(sink, env=APPLE_TERMINAL_ENV, platform="darwin").notify_turn_complete(
        running_children=0
    )
    assert sink.writes == ["\a"]


@pytest.mark.parametrize(
    "mux", [{"TMUX": "/tmp/tmux-1/default,1,0"}, {"ZELLIJ": "0"}], ids=["tmux", "zellij"]
)
def test_the_pane_flag_bel_is_kept_under_a_multiplexer(
    played: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch, mux: dict[str, str]
) -> None:
    _use(monkeypatch)
    sink = Sink()
    Notifier(sink, env={**APPLE_TERMINAL_ENV, **mux}, platform="darwin").notify_turn_complete(
        running_children=0
    )
    assert sink.writes == ["\a"]
    assert played == [("file", "osc")]
