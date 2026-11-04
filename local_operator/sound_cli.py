"""``lop sound list | preview <id> | doctor``: the notification-sound front door.

WHY A SEPARATE MODULE. ``local_operator/cli.py`` builds its parser on every
``lop`` invocation, so registration here is stdlib-only, the ``disk_cli`` rule.
``notifications.sound`` is itself stdlib-only and loads in the handler.

THE VERBS.

* ``list``: every built-in id with its display name and duration, marking the
  selected one and the default, then the current enabled/volume state. The ids
  are the values ``lop config edit notifications.sound.id`` accepts.
* ``preview <id> [--volume N]``: play one sound now at the configured volume (or
  ``--volume``). It plays even when ``notifications.sound.enabled`` is off, so a
  user can audition before turning sounds on, and it skips the 1.5 s throttle:
  the throttle stops bursts of NOTIFICATIONS, and a preview is not one. It runs
  in the foreground through the same player as a notification
  (``notifications.play``), so ``LOCAL_OPERATOR_NO_SOUND`` and the NULL-backend
  receipt (``LOCAL_OPERATOR_SOUND_BACKEND=null``) behave the same here.
* ``doctor``: what will actually happen on the next notification on THIS
  terminal: lop's plan, which banner leg applies, and whether a terminal or cmux
  will add a sound of its own, with the exact fix. It reads
  ``~/.config/cmux/cmux.json`` read-only and never edits another app's config
  (design §4.4).

The output is operator CLI text, English-only like ``lop disk`` (docs/I18N.md);
only argparse help is keyed (``wire.notification_sound.cli.*``). Sound display
names come from the keyed catalogue through the settings registry, so they
follow the user's language.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from local_operator.i18n.cli_help import help_text
from local_operator.i18n.keys import wire_notification_sound as _ns


def add_parser(
    subparsers: Any, parent_parser: Any = None, *, help_locale: str | None = None
) -> None:
    """Register ``lop sound`` and its verbs (stdlib-only)."""
    parents = [parent_parser] if parent_parser is not None else []
    sound = subparsers.add_parser(
        "sound",
        help=help_text(_ns.cli_help(), help_locale),
        description=help_text(_ns.cli_help(), help_locale),
        parents=parents,
    )
    verbs = sound.add_subparsers(dest="sound_command")
    verbs.add_parser("list", help=help_text(_ns.cli_list_help(), help_locale), parents=parents)
    preview = verbs.add_parser(
        "preview", help=help_text(_ns.cli_preview_help(), help_locale), parents=parents
    )
    preview.add_argument("sound_id", help=help_text(_ns.cli_preview_id_help(), help_locale))
    preview.add_argument(
        "--volume",
        type=int,
        default=None,
        metavar="N",
        help=help_text(_ns.cli_preview_volume_help(), help_locale),
    )
    verbs.add_parser("doctor", help=help_text(_ns.cli_doctor_help(), help_locale), parents=parents)


def _say(text: str = "") -> None:
    print(text)  # i18n: ignore operator CLI output, English-only; lop sound section


def _err(text: str) -> None:
    print(text, file=sys.stderr)  # i18n: ignore operator CLI output, English-only


def _labels() -> dict[str, str]:
    """``id -> localized display name``, through the settings registry's choices."""
    try:
        from local_operator.settings_io import BY_KEY

        return {
            str(choice.value): str(choice.description)
            for choice in BY_KEY["notifications.sound.id"].resolved_choices
        }
    except Exception:  # noqa: BLE001 — names are decoration; ids still list
        return {}


def main(args: argparse.Namespace) -> int:
    """Dispatch ``lop sound <verb>``."""
    verb = getattr(args, "sound_command", None)
    if verb == "list":
        return list_command()
    if verb == "preview":
        return preview_command(args.sound_id, getattr(args, "volume", None))
    if verb == "doctor":
        return doctor_command()
    _err("usage: lop sound {list,preview,doctor}")
    return 2


def list_command() -> int:
    from local_operator.notifications import sound

    values = sound.read_values()
    plan = sound.resolve(values)
    stored = sound.configured(values)
    selected = stored.sound_id
    labels = _labels()
    for entry in sound.builtin_sounds():
        mark = "*" if entry.id == selected else " "
        default = "  (default)" if entry.id == sound.DEFAULT_SOUND_ID else ""
        name = labels.get(entry.id, entry.id)
        _say(f"{mark} {entry.id:17} {name:20} {entry.duration_s:.2f}s{default}")
    mark = "*" if selected == sound.SYSTEM_SOUND_ID else " "
    _say(f"{mark} {sound.SYSTEM_SOUND_ID:17} {labels.get(sound.SYSTEM_SOUND_ID, '')}")
    _say()
    volume = stored.effective_volume
    _say(
        f"selected: {selected}   volume: {volume:g} (gain {sound.gain(volume):.2f})   "
        f"next notification: {_describe(plan)}"
    )
    for problem in _problems(stored, plan):
        _say(f"note: {problem}")
    _say(
        "change it: lop config edit notifications.sound.id <id>   "
        "try one: lop sound preview <id>"
    )
    return 0


def _describe(plan: Any) -> str:
    if plan.kind == "file":
        return f"plays {plan.sound_id}"
    if plan.kind == "system":
        return "terminal/system sound; lop plays nothing"
    return f"silent ({plan.reason})"


def _problems(stored: Any, plan: Any) -> list[str]:
    """Hand-edit problems ``resolve`` silently worked around (review round 1, m4; QA Q2)."""
    from local_operator.notifications import sound

    problems = []
    if (
        stored.enabled
        and stored.sound_id != sound.SYSTEM_SOUND_ID
        and sound.builtin(stored.sound_id) is None
    ):
        problems.append(
            f"notifications.sound.id {stored.sound_id!r} is not a known sound; "
            f"playing {sound.DEFAULT_SOUND_ID} instead (see `lop sound list`)"
        )
    if stored.volume_problem:
        problems.append(stored.volume_problem)
    return problems


def preview_command(sound_id: str, volume: int | None) -> int:
    """Play one sound now.

    It deliberately does NOT apply the rig guard (``desktop_belongs_to_this_process``)
    that gates notification sounds (review round 1, n1). A notification is
    something a RUN emits on its own, so a redirected HOME must not reach the
    speakers. A preview is a command a person typed in order to hear a sound,
    and refusing it would break the one command whose whole job is playing.
    Rigs mute it the way they mute every player: ``LOCAL_OPERATOR_NO_SOUND`` or
    ``LOCAL_OPERATOR_SOUND_BACKEND=null``. The note below says so when the guard
    would have refused, so a rig author sees the choice instead of discovering it.
    """
    from local_operator.notifications import play, sound
    from local_operator.tui.notify import desktop_belongs_to_this_process

    if sound_id == sound.SYSTEM_SOUND_ID:
        _say("`system` means lop plays nothing; your terminal or OS plays its own sound.")
        return 0
    entry = sound.builtin(sound_id)
    if entry is None:
        _err(f"unknown sound {sound_id!r}; choose one of: {', '.join(sound.sound_ids())}")
        return 2
    if volume is not None and not 0 <= volume <= 100:
        _err("--volume must be between 0 and 100")
        return 2
    level = float(volume) if volume is not None else sound.configured().effective_volume
    amplitude = sound.gain(level)
    if os.environ.get(sound.ENV_NO_SOUND):
        _say(f"{sound.ENV_NO_SOUND} is set; not playing {sound_id}.")
        return 0
    if amplitude <= 0:
        _say(f"volume is 0; {sound_id} would be silent.")
        return 0
    if not desktop_belongs_to_this_process(report=False):
        _say(
            "note: HOME is redirected, so notifications would stay silent here; "
            "a preview still plays (set LOCAL_OPERATOR_NO_SOUND=1 to mute it)"
        )
    _say(f"playing {sound_id} at volume {level:g} (gain {amplitude:.2f})")
    code = play.play(entry.path, amplitude, sound_id=entry.id, sha256=entry.sha256, leg="preview")
    if code != 0:
        _err("could not play the sound: no audio device answered (see `lop sound doctor`)")
    return code


#: cmux's own default for ``notifications.sound`` (cmux.schema.json,
#: ``/properties/notifications/properties/sound``): a sound plays unless the
#: user set ``"none"``.
_CMUX_DEFAULT_SOUND = "default"


def _cmux_sound(home: Path) -> str:
    """cmux's own ``notifications.sound`` value, read-only.

    A missing file, a missing key or a malformed file all read as cmux's
    DEFAULT, which plays (review round 1, m2 and m3). That is what cmux itself
    does with them, and "unknown, so no double sound" was the one answer that
    is never true. Every shape is guarded, because this file belongs to
    another program and can hold anything.
    """
    path = home / ".config" / "cmux" / "cmux.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _CMUX_DEFAULT_SOUND
    notifications = data.get("notifications") if isinstance(data, dict) else None
    value = notifications.get("sound") if isinstance(notifications, dict) else None
    return value if isinstance(value, str) and value else _CMUX_DEFAULT_SOUND


def doctor_command() -> int:
    """What the next notification on THIS terminal will sound like, and every
    case where two sounds play, with its fix (review round 1, M1)."""
    from local_operator import terminals
    from local_operator.notifications import sound
    from local_operator.tui import notify

    values = sound.read_values()
    plan = sound.resolve(values)
    if not notify.notifications_enabled():
        # Lead with it (QA Q1): with banners off nothing plays, so a first line
        # reading "plays soft-pulse" would contradict the conclusion.
        _say(
            "lop: silent (banners are off: display.notifications or "
            "LOCAL_OPERATOR_NO_NOTIFICATIONS); a sound only plays with a banner"
        )
        _say(f"  when banners are on: {_describe(plan)}")
        return 0
    env = os.environ
    if terminals.is_ssh(env) and plan.kind == "file":
        # The same rule as `tui/notify._sound_plan` (review round 2, R2-M1): the
        # player would sound on the REMOTE machine, so lop stays out of it
        # entirely and the terminal-side behaviour is exactly as before.
        _say(
            "lop: over SSH: lop plays no sound of its own; your local terminal's "
            "notification sound applies"
        )
        _say(f"  on this machine directly it would play {plan.sound_id}")
        _say("double sound: none expected (only your local terminal signals)")
        return 0
    _say(f"lop: {_describe(plan)}")
    if plan.kind == "file":
        _say(f"  file: {plan.path}")
        _say(f"  gain: {plan.gain:.2f}   at most one sound every {sound.THROTTLE_S:g}s")
    for problem in _problems(sound.configured(values), plan):
        _say(f"  note: {problem}")
    protocol = notify.detect_protocol(env)
    # The SAME condition `Notifier.send` uses to drop the BEL: a file plan that
    # clears the rig guard. Under a redirected HOME lop plays nothing, and the
    # page must not claim its sound replaces anything.
    lop_plays = plan.kind == "file" and notify.desktop_belongs_to_this_process(report=False)
    if plan.kind == "file" and not lop_plays:
        _say(
            "  note: HOME is redirected (a test run, container or sandbox), so lop "
            "plays no notification sound here; the terminal's own signal is left as is"
        )
    in_mux = notify.is_inside_tmux(env) or notify.is_inside_zellij(env)
    mux = "tmux" if notify.is_inside_tmux(env) else "Zellij"
    double: list[str] = []
    if terminals.is_cmux(env):
        value = _cmux_sound(Path.home())
        _say(f"banner leg: cmux (cmux notifications.sound = {value})")
        if lop_plays and value != "none":
            double.append(  # i18n: ignore operator CLI output, English-only
                'cmux plays its own sound too. Fix: set "notifications": {"sound": "none"} '
                "in ~/.config/cmux/cmux.json, or choose `system` here to keep only cmux's."
            )
    else:
        if protocol == "osc99":
            _say("banner leg: kitty OSC 99, sent with s=silent while lop plays its own sound")
        elif protocol == "osc9":
            _say("banner leg: OSC 9 (Ghostty, iTerm2, WezTerm, Warp); it has no sound field")
            if lop_plays and terminals.is_ghostty(env):
                double.append(  # i18n: ignore operator CLI output, English-only
                    "Ghostty plays the system default sound with every OSC 9 banner. "
                    "Choose `system` here to keep only Ghostty's, or turn notification "
                    "sounds off for Ghostty in System Settings > Notifications."
                )
        elif lop_plays and not in_mux:
            _say("banner leg: none in the terminal; lop's sound replaces the terminal bell")
            if terminals.term_program(env) == terminals.TERM_PROGRAM_APPLE:
                # Review round 2, R2-m1: on Terminal.app the BEL is also the
                # unfocused window's only visual cue (Dock bounce and badge).
                _say(
                    "  Terminal.app: without the bell there is no Dock bounce or badge; "
                    "choose `system` to keep them (lop then plays nothing): "
                    "lop config edit notifications.sound.id system"
                )
        else:
            _say("banner leg: terminal bell")
        if protocol == "bell" and sys.platform.startswith("linux"):
            _say("  + notify-send, sent with suppress-sound while lop plays its own sound")
        if in_mux:
            _say(f"  inside {mux}: a terminal bell is also sent, to flag the pane")
            if lop_plays:
                double.append(  # i18n: ignore operator CLI output, English-only
                    f"inside {mux} the terminal bell is kept (it is the only pane flag), "
                    "so an audible bell rings beside lop's sound. Fix: turn the audible "
                    "bell off in your terminal's settings, or choose `system` here "
                    "(lop then plays nothing): lop config edit notifications.sound.id system"
                )
    for line in double:
        _say(f"double sound: {line}")
    if not double:
        _say("double sound: none expected on this terminal")
    return 0
