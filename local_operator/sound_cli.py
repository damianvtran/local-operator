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

    values = sound._config_values()
    plan = sound.resolve(values)
    configured = sound._walk(values, "notifications", "sound", "id")
    selected = configured if isinstance(configured, str) and configured else sound.DEFAULT_SOUND_ID
    labels = _labels()
    for entry in sound.builtin_sounds():
        mark = "*" if entry.id == selected else " "
        default = "  (default)" if entry.id == sound.DEFAULT_SOUND_ID else ""
        name = labels.get(entry.id, entry.id)
        _say(f"{mark} {entry.id:17} {name:20} {entry.duration_s:.2f}s{default}")
    mark = "*" if selected == sound.SYSTEM_SOUND_ID else " "
    _say(f"{mark} {sound.SYSTEM_SOUND_ID:17} {labels.get(sound.SYSTEM_SOUND_ID, '')}")
    _say()
    volume = sound._walk(values, "notifications", "sound", "volume")
    volume = sound.DEFAULT_VOLUME if volume is None else volume
    _say(
        f"selected: {selected}   volume: {volume} (gain {sound.gain(volume):.2f})   "
        f"next notification: {_describe(plan)}"
    )
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


def preview_command(sound_id: str, volume: int | None) -> int:
    from local_operator.notifications import play, sound

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
    if volume is None:
        configured = sound._walk(sound._config_values(), "notifications", "sound", "volume")
        volume = sound.DEFAULT_VOLUME if configured is None else configured
    amplitude = sound.gain(volume)
    if os.environ.get(sound.ENV_NO_SOUND):
        _say(f"{sound.ENV_NO_SOUND} is set; not playing {sound_id}.")
        return 0
    if amplitude <= 0:
        _say(f"volume is 0; {sound_id} would be silent.")
        return 0
    _say(f"playing {sound_id} at volume {volume} (gain {amplitude:.2f})")
    code = play.play(entry.path, amplitude, sound_id=entry.id, sha256=entry.sha256, leg="preview")
    if code != 0:
        _err("could not play the sound: no audio device answered (see `lop sound doctor`)")
    return code


def _cmux_sound(home: Path) -> str | None:
    """cmux's own ``notifications.sound`` value, read-only; ``None`` if unknown."""
    path = home / ".config" / "cmux" / "cmux.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("notifications", {}).get("sound") if isinstance(data, dict) else None
    return value if isinstance(value, str) else ("default" if path.exists() else None)


def doctor_command() -> int:
    from local_operator import terminals
    from local_operator.notifications import sound
    from local_operator.tui import notify

    plan = sound.resolve()
    _say(f"lop: {_describe(plan)}")
    if plan.kind == "file":
        _say(f"  file: {plan.path}")
        _say(f"  gain: {plan.gain:.2f}   at most one sound every {sound.THROTTLE_S:g}s")
    if not notify.notifications_enabled():
        _say(
            "banners: off (display.notifications or LOCAL_OPERATOR_NO_NOTIFICATIONS), "
            "so no sound plays either"
        )
    env = os.environ
    protocol = notify.detect_protocol(env)
    double: list[str] = []
    if terminals.is_cmux(env):
        value = _cmux_sound(Path.home())
        _say(f"banner leg: cmux (cmux notifications.sound = {value or 'unknown'})")
        if plan.kind == "file" and value not in ("none", None):
            double.append(  # i18n: ignore operator CLI output, English-only
                'cmux plays its own sound too. Fix: set "notifications": {"sound": "none"} '
                "in ~/.config/cmux/cmux.json, or choose `system` here to keep only cmux's."
            )
    elif protocol == "osc99":
        _say("banner leg: kitty OSC 99, sent with s=silent while lop plays its own sound")
    elif protocol == "osc9":
        _say("banner leg: OSC 9 (Ghostty, iTerm2, WezTerm, Warp); it has no sound field")
        if plan.kind == "file" and terminals.is_ghostty(env):
            double.append(  # i18n: ignore operator CLI output, English-only
                "Ghostty plays the system default sound with every OSC 9 banner. "
                "Choose `system` here to keep only Ghostty's, or turn notification "
                "sounds off for Ghostty in System Settings > Notifications."
            )
    else:
        _say(
            "banner leg: terminal bell"
            + (
                " + notify-send (sent with suppress-sound)"
                if sys.platform.startswith("linux")
                else ""
            )
        )
    for line in double:
        _say(f"double sound: {line}")
    if not double:
        _say("double sound: none expected on this terminal")
    return 0
