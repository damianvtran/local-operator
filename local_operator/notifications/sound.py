"""Notification sounds: the catalog, the per-notification plan, gain and throttle.

WHAT THIS IS FOR. A desktop banner reaches someone who is looking at another
application. A short, quiet sound reaches someone who is not looking at the
screen at all. lop ships ten built-in sounds (``sounds/*.flac``, described by
``sounds/catalog.json``, exported from the reviewed masters by
``assets/notification-sounds/export.py``). The user picks one, sets its volume
or turns it off through three settings keys:

* ``notifications.sound.enabled`` (BOOL, :data:`DEFAULT_ENABLED`)
* ``notifications.sound.id`` (ENUM: a built-in id or ``system``, :data:`DEFAULT_SOUND_ID`)
* ``notifications.sound.volume`` (INT 0-100, :data:`DEFAULT_VOLUME`)

THE THREE RULES (notification-sounds design §4), and where each one lives:

1. **The sound rides the banner.** Nothing in this module decides whether a
   notification fires. :func:`play_for_banner` is called by the code path that
   has ALREADY raised the banner (``tui/notify.py``: ``Notifier.send`` and
   ``detached_notify``), after it has won the existing one-banner-per-machine
   arbitration. So one event can never produce one banner and two sounds, and
   there is no second arbitration to keep in step with the first.
2. **Who plays.** In the TUI and the detached runtime, lop plays the sound
   itself, in a detached child (:mod:`local_operator.notifications.play`). The
   terminal, cmux or Notification Centre that shows the banner cannot take a
   sound file. ``system`` means lop plays nothing and the terminal's or OS's own
   sound is left alone, which is how notifications behaved before this module
   existed. :meth:`SoundPlan.mutes_host_sound` tells the banner legs that CAN be
   silenced (kitty OSC 99 ``s=silent``, notify-send ``suppress-sound``) whether
   to silence them.
3. **At most one sound every 1.5 s per machine** (:func:`claim_throttle`).
   Several surfaces can each legitimately win a DIFFERENT event within a second
   (a digest plus an item banner, or two sessions finishing together), and a
   burst of chimes is worse than one.

THE VOLUME CURVE. amplitude = (volume / 100) ** 2, so 0 mutes and the slider
follows roughly a 40·log10 dB taper, which is close to how loudness is heard.
Every master peaks at -20 LUFS max momentary, so the default of 70 (0.49,
about -6.2 dB) lands at about -26 LUFS. That is between macOS's Tink (-24.7)
and Pop (-29.0): audible, not intrusive. Measured in the design, §1.2.

FRESHNESS AND COST. :func:`resolve` reads ``config.yml`` on every notification
through :func:`local_operator.config.read_config_values`. Its parse memo turns
an unchanged file into one ``fstat``, so an edit in ``/settings``, ``lop config``
or the desktop app applies to the very next notification (the section is
``LIVE``). It never moves a bad file aside, which a ``ConfigManager`` would do.

STDLIB-ONLY AT IMPORT. ``tui/notify.py`` calls this from the event loop and from
the inbox delivery rung, so importing it must cost nothing: no miniaudio, no
TUI. Only the player child imports miniaudio.

TEST AND RIG SAFETY. ``LOCAL_OPERATOR_NO_SOUND=1`` mutes every player.
``LOCAL_OPERATOR_SOUND_BACKEND=null`` runs the whole path on miniaudio's NULL
device and writes a receipt line to ``<config>/run/notification-sounds.jsonl``,
so QA can prove a sound would have played without speakers. The test suite arms
``LOCAL_OPERATOR_NO_SOUND`` globally (``tests/conftest.py``), beside the
notification kill switch.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: The operator's pick (2026-11-03): "I think I like soft pulse as the default".
DEFAULT_SOUND_ID = "soft-pulse"
#: 70 % -> 0.49 amplitude -> about -26 LUFS max momentary; see the module docstring.
DEFAULT_VOLUME = 70
DEFAULT_ENABLED = True

#: The ``notifications.sound.id`` value meaning "lop plays nothing; leave the
#: terminal's or system's own sound alone".
SYSTEM_SOUND_ID = "system"

#: The machine-wide minimum spacing between two notification sounds (rule 3).
THROTTLE_S = 1.5

#: Mute switch for rigs, CI and captures. Checked here AND in the player child,
#: so a child spawned with a hand-built environment is still muted when the
#: variable reached it.
ENV_NO_SOUND = "LOCAL_OPERATOR_NO_SOUND"
#: ``null``: play through miniaudio's NULL device and write a receipt (QA seam).
ENV_BACKEND = "LOCAL_OPERATOR_SOUND_BACKEND"

#: Both live under ``<config>/run/``, the directory for machine-local runtime
#: state. The config dir is per user, which makes the throttle per user, and the
#: user is who hears the speakers.
THROTTLE_FILE = "notification-sound.last"
RECEIPT_FILE = "notification-sounds.jsonl"

SOUNDS_DIR = Path(__file__).resolve().parent / "sounds"

PlanKind = Literal["file", "system", "silent"]


@dataclasses.dataclass(frozen=True)
class BuiltinSound:
    """One entry of ``sounds/catalog.json``."""

    id: str
    #: The i18n key of the display name. Rendered by the surfaces, never stored.
    label: str
    file: str
    sha256: str
    duration_s: float
    channels: int

    @property
    def path(self) -> Path:
        return SOUNDS_DIR / self.file


@functools.lru_cache(maxsize=1)
def builtin_sounds() -> tuple[BuiltinSound, ...]:
    """The shipped sounds, in picker order.

    Cached for the life of the process: the catalog is package data and cannot
    change under a running interpreter. An unreadable catalog yields ``()``, and
    every caller then falls back to "no file to play". A broken install must not
    raise from a notification path, whose caller asked for a banner, not a sound.
    """
    try:
        raw = json.loads((SOUNDS_DIR / "catalog.json").read_text(encoding="utf-8"))
        return tuple(
            BuiltinSound(
                id=str(entry["id"]),
                label=str(entry["label"]),
                file=str(entry["file"]),
                sha256=str(entry["sha256"]),
                duration_s=float(entry["duration_s"]),
                channels=int(entry["channels"]),
            )
            for entry in raw["sounds"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("notification sound catalog is unreadable", exc_info=True)
        return ()


def builtin(sound_id: str) -> BuiltinSound | None:
    """The built-in sound called ``sound_id``, or ``None``."""
    for sound in builtin_sounds():
        if sound.id == sound_id:
            return sound
    return None


def sound_ids() -> tuple[str, ...]:
    """Every value ``notifications.sound.id`` accepts, in picker order.

    Built-ins, then ``system``, then the user's custom sounds oldest first
    (``custom_sounds``), the order the settings choices and the desktop list use.
    """
    from local_operator.notifications import custom_sounds

    return (
        tuple(sound.id for sound in builtin_sounds())
        + (SYSTEM_SOUND_ID,)
        + tuple(sound.id for sound in custom_sounds.list_sounds())
    )


def playable(sound_id: str) -> tuple[str, Path, str] | None:
    """``(id, file, sha256)`` for a built-in or custom id whose file exists, else ``None``.

    The one lookup both the notification resolver and the desktop's render
    route use, so "which ids can be played" has one answer. A custom id is
    looked up in the index and its path built by ``custom_sounds`` (regex plus
    resolved containment); nothing from a caller is joined to a path here.
    """
    entry = builtin(sound_id)
    if entry is not None:
        return (entry.id, entry.path, entry.sha256) if entry.path.is_file() else None
    from local_operator.notifications import custom_sounds

    found = custom_sounds.resolve_file(sound_id)
    if found is None:
        return None
    row, path = found
    return row.id, path, row.sha256


def gain(volume: Any) -> float:
    """Linear amplitude for a 0-100 volume: ``(v/100)**2``, clamped.

    Clamped rather than rejected because this is the READ side: the registry
    already refuses out-of-range writes, so a value outside 0-100 can only come
    from a hand edit, and a notification is no place to raise about it.
    A non-number falls back to :data:`DEFAULT_VOLUME`.
    """
    if isinstance(volume, bool) or not isinstance(volume, (int, float)) or math.isnan(volume):
        volume = DEFAULT_VOLUME
    level = min(100.0, max(0.0, float(volume))) / 100.0
    return round(level * level, 4)


@dataclasses.dataclass(frozen=True)
class SoundPlan:
    """What should accompany the next banner. Built by :func:`resolve`."""

    #: ``file``: lop plays ``path`` at ``gain``. ``system``: lop plays nothing
    #: and the host's own sound is untouched. ``silent``: nobody plays.
    kind: PlanKind
    sound_id: str
    gain: float = 0.0
    path: Path | None = None
    sha256: str = ""
    #: Why the plan is ``silent`` (for ``lop sound doctor``); empty otherwise.
    reason: str = ""

    @property
    def mutes_host_sound(self) -> bool:
        """Whether a banner leg that CAN be silenced should be silenced.

        True for ``file`` (lop's own sound plays, so the host's would make two)
        and for ``silent`` (the user asked for quiet). False only for
        ``system``, whose whole meaning is "keep the host's sound".
        """
        return self.kind != "system"


def read_values() -> Mapping[str, Any]:
    """The config ``values`` this module reads, for callers that also want
    :func:`configured` and :func:`resolve` over the SAME snapshot (``lop sound``)."""
    return _config_values()


def _config_values() -> Mapping[str, Any]:
    try:
        from local_operator.config import read_config_values
        from local_operator.paths import config_dir

        return read_config_values(config_dir()) or {}
    except Exception:  # noqa: BLE001 — a notification must never fail on config
        logger.debug("could not read the notification sound settings", exc_info=True)
        return {}


def _walk(values: Mapping[str, Any], *path: str) -> Any:
    current: Any = values
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _strict_bool(value: Any, default: bool) -> bool:
    """``settings_io.strict_bool``, imported lazily.

    THE SAME READER AS THE SETTINGS PAGE, deliberately (review round 1, m1): a
    hand-edited ``enabled: "off"`` must read as off here exactly as the page
    paints it, or the page shows "off" while the sound plays. Lazy because
    ``settings_io`` materialises the whole registry (~0.35 s) and this module is
    on the banner path; the import only happens for a non-bool value, which
    the registry never writes, so a real config never pays it.
    """
    if isinstance(value, bool):
        return value
    from local_operator.settings_io import strict_bool

    return strict_bool(value, default)


@dataclasses.dataclass(frozen=True)
class Configured:
    """The three keys as STORED, before any fallback, for ``lop sound`` to report.

    Public so ``sound_cli`` reads the same values ``resolve`` does without
    reaching into this module's private helpers (review round 1, n2).
    """

    #: ``enabled`` after ``strict_bool`` (what the page shows and resolve obeys).
    enabled: bool
    #: The stored id (may name nothing; ``resolve`` then plays the default).
    sound_id: str
    #: The stored volume, or ``None`` when unset or not a number.
    volume: float | None
    #: The volume as it will be USED: clamped to 0-100, default when unusable.
    effective_volume: float
    #: Why the stored volume is not used as written ("" when it is).
    volume_problem: str


def configured(values: Mapping[str, Any] | None = None) -> Configured:
    """Read the three keys the way :func:`resolve` does, without deciding anything."""
    source = _config_values() if values is None else values
    enabled = _strict_bool(_walk(source, "notifications", "sound", "enabled"), DEFAULT_ENABLED)
    raw_id = _walk(source, "notifications", "sound", "id")
    sound_id = raw_id if isinstance(raw_id, str) and raw_id else DEFAULT_SOUND_ID
    raw_volume = _walk(source, "notifications", "sound", "volume")
    volume: float | None = None
    problem = ""
    if raw_volume is None:
        effective = float(DEFAULT_VOLUME)
    elif (
        isinstance(raw_volume, bool)
        or not isinstance(raw_volume, (int, float))
        or math.isnan(raw_volume)
    ):
        effective = float(DEFAULT_VOLUME)
        problem = (
            f"notifications.sound.volume {raw_volume!r} is not a number; using {DEFAULT_VOLUME}"
        )
    else:
        volume = float(raw_volume)
        effective = min(100.0, max(0.0, volume))
        if effective != volume:
            problem = (
                f"notifications.sound.volume {raw_volume!r} is outside 0-100; "
                f"clamped to {effective:g}"
            )
    return Configured(enabled, sound_id, volume, effective, problem)


def resolve(
    values: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None
) -> SoundPlan:
    """The plan for the next notification, read fresh from the config.

    ``values`` and ``env`` are injectable for tests; production passes neither.

    A hand-edited id that names nothing falls back to the DEFAULT sound, not to
    silence. The user's intent ("play a sound") is clear and only its spelling
    is wrong, and a notification that goes quiet with nothing on screen is the
    worse failure. ``lop sound doctor`` reports the fallback (it compares
    :func:`configured`'s id with the plan's).

    A hand-edited out-of-range or non-numeric volume is clamped or defaulted
    rather than refused (the registry already refuses such writes), and the
    silent plan's ``reason`` names the bad value instead of claiming "0".
    """
    source_env = os.environ if env is None else env
    stored = configured(values)
    sound_id = stored.sound_id
    if source_env.get(ENV_NO_SOUND):
        return SoundPlan("silent", sound_id, reason=f"{ENV_NO_SOUND} is set")
    if not stored.enabled:
        return SoundPlan("silent", sound_id, reason="notifications.sound.enabled is off")
    if sound_id == SYSTEM_SOUND_ID:
        return SoundPlan("system", sound_id)
    amplitude = gain(stored.effective_volume)
    if amplitude <= 0.0:
        reason = stored.volume_problem or "notifications.sound.volume is 0"
        return SoundPlan("silent", sound_id, reason=reason)
    # Custom ids resolve through the user's store (C2). A custom id whose row or
    # file has gone (hand-deleted, or a config copied from another machine)
    # takes the same default fallback as a misspelt id, for the same reason.
    found = playable(sound_id)
    if found is None:
        logger.debug("unknown notification sound %r; using %s", sound_id, DEFAULT_SOUND_ID)
        found = playable(DEFAULT_SOUND_ID)
    if found is None:
        return SoundPlan("silent", sound_id, reason="the sound file is missing from this install")
    resolved_id, path, sha = found
    return SoundPlan("file", resolved_id, amplitude, path, sha)


def run_dir() -> Path:
    from local_operator.paths import config_dir

    return config_dir() / "run"


def claim_throttle(now: float | None = None, *, window_s: float = THROTTLE_S) -> bool:
    """Take the machine-wide sound slot; ``False`` if a sound played < 1.5 s ago.

    A ``stat`` and then a touch (``os.utime``) of ``<config>/run/
    notification-sound.last``. The two steps are not atomic, and that is
    accepted: two players racing inside the same microseconds both pass and the
    user hears one extra chime. A lock would buy nothing worth having and adds
    a way to lose a sound to a stale lock. The desktop app's player implements
    the same check on the same file, so the throttle covers every surface on
    the machine.

    Fails OPEN on an unwritable run dir. The throttle exists to stop bursts, and
    refusing every sound because a directory cannot be created would turn a
    tidiness rule into a missing feature.
    """
    stamp = time.time() if now is None else now
    path = run_dir() / THROTTLE_FILE
    try:
        last = path.stat().st_mtime
    except OSError:
        last = None
    if last is not None and 0 <= stamp - last < window_s:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        os.utime(path, (stamp, stamp))
    except OSError:
        logger.debug("could not record the notification sound throttle", exc_info=True)
    return True


def player_argv(plan: SoundPlan, *, leg: str) -> list[str]:
    """argv for the detached player child.

    ``-I`` (isolated mode): the child must not pick up a user's ``PYTHONPATH``
    or ``PYTHONSTARTUP``. The package still resolves, because both a wheel
    install and an editable venv put it on the interpreter's own path.
    """
    return [
        sys.executable,
        "-I",
        "-m",
        "local_operator.notifications.play",
        str(plan.path),
        "--gain",
        f"{plan.gain:.4f}",
        "--sound-id",
        plan.sound_id,
        "--sha256",
        plan.sha256,
        "--leg",
        leg,
    ]


def play_plan(plan: SoundPlan, *, leg: str, throttle: bool = True) -> bool:
    """Start the player for ``plan``; ``True`` when a child was started.

    Never raises, never waits: the caller is a banner path on the event loop.
    """
    if plan.kind != "file" or plan.path is None:
        return False
    if throttle and not claim_throttle():
        return False
    try:
        from local_operator.proc import spawn_detached

        return bool(spawn_detached(player_argv(plan, leg=leg)))
    except Exception:  # noqa: BLE001 — a sound must never affect its caller
        logger.debug("notification sound failed to start", exc_info=True)
        return False


def play_for_banner(leg: str, plan: SoundPlan | None = None) -> bool:
    """Play the configured sound for a banner that HAS been raised.

    Call it only after the banner went out (rule 1). ``plan`` lets a caller that
    already resolved one, to decide the banner's own sound hint, play the same
    plan, so the hint and the sound cannot disagree about the config.
    ``leg`` names the banner route in the QA receipt (``cmux``, ``osc``,
    ``notify-send``, ``detached``).
    """
    return play_plan(resolve() if plan is None else plan, leg=leg)


#: What :func:`preview_plan` decided. ``play``: ``plan`` is a ``file`` plan to
#: start. Every other outcome plays nothing and names why, so each surface can
#: word its own receipt (the CLI prints English operator text, the TUI renders
#: keyed copy) without re-deriving the decision.
PreviewOutcome = Literal["play", "system", "unknown", "muted", "silent", "ssh", "failed"]


@dataclasses.dataclass(frozen=True)
class Preview:
    """The decision for one audition, built by :func:`preview_plan`."""

    outcome: PreviewOutcome
    sound_id: str
    #: The 0-100 level the preview uses (the configured one unless overridden).
    volume: float = 0.0
    gain: float = 0.0
    #: The ``file`` plan to play when ``outcome == "play"``; ``None`` otherwise.
    plan: SoundPlan | None = None


def preview_plan(
    sound_id: str,
    volume: float | None = None,
    *,
    env: Mapping[str, str] | None = None,
    values: Mapping[str, Any] | None = None,
) -> Preview:
    """Decide what auditioning ``sound_id`` would do, playing nothing.

    THE ONE DECISION every preview surface shares (``lop sound preview``, the
    TUI ``/settings`` row and ``/sound preview``), extracted from the CLI so the
    rules cannot drift between them (notification-sounds addendum 2, §1):

    * It ignores ``notifications.sound.enabled``: a person auditions a sound
      before deciding to turn sounds on (C1 QA row 8).
    * It ignores the 1.5 s throttle (the caller passes ``throttle=False``): that
      throttle stops bursts of NOTIFICATIONS, and an audition is not one.
    * It honours ``LOCAL_OPERATOR_NO_SOUND`` and a 0 volume, as every player does.

    ``volume`` is the caller's validated 0-100 override; ``None`` uses the
    configured level. Range checking stays with the caller because each surface
    words the refusal differently (an argparse flag vs the registry's message).
    """
    source_env = os.environ if env is None else env
    if sound_id == SYSTEM_SOUND_ID:
        return Preview("system", sound_id)
    entry = playable(sound_id)
    if entry is None:
        return Preview("unknown", sound_id)
    level = float(volume) if volume is not None else configured(values).effective_volume
    amplitude = gain(level)
    if source_env.get(ENV_NO_SOUND):
        return Preview("muted", sound_id, level, amplitude)
    if amplitude <= 0:
        return Preview("silent", sound_id, level, amplitude)
    playable_id, path, sha = entry
    plan = SoundPlan("file", playable_id, amplitude, path, sha)
    return Preview("play", playable_id, level, amplitude, plan)


def preview(
    sound_id: str,
    volume: float | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Preview:
    """Audition ``sound_id`` now in a DETACHED player; never waits, never raises.

    For callers on an event loop (the TUI). ``lop sound preview`` plays in the
    foreground instead, because a CLI that returned before the sound finished
    would let the shell prompt race it; it shares :func:`preview_plan`.

    OVER SSH it refuses with outcome ``ssh``, the same rule ``tui/notify.py``'s
    ``_sound_plan`` applies to notifications: the player would sound on the
    REMOTE computer's speakers (or nowhere on a headless host), which is never
    what someone browsing a list in their local terminal means. The CLI does not
    refuse, because there the person typed a command on that machine on purpose.

    ``outcome == "failed"`` means the player child could not be started.
    """
    decided = preview_plan(sound_id, volume, env=env)
    if decided.outcome != "play" or decided.plan is None:
        return decided
    from local_operator import terminals

    if terminals.is_ssh(env):
        return dataclasses.replace(decided, outcome="ssh", plan=None)
    # A NEW audition supersedes the one still sounding (UX round 1, U1): a
    # person stepping down the list at reading pace outlives the debounce on
    # every row, and without this each row's player kept going, measured at up
    # to four sounds at once. A sound picker plays only the row you are on.
    stop_preview()
    child = _start_preview_child(decided.plan)
    if child is None:
        return dataclasses.replace(decided, outcome="failed")
    with _PREVIEW_LOCK:
        global _PREVIEW_CHILD
        _PREVIEW_CHILD = child
    return decided


#: The one audition player this process started and may still be sounding.
#: Only auditions are tracked: a NOTIFICATION sound is never cut short by a
#: preview (it is a different event the user did not just choose to replace),
#: so ``play_plan`` keeps its fire-and-forget spawn.
_PREVIEW_CHILD: Any = None
_PREVIEW_LOCK = threading.Lock()


def _start_preview_child(plan: SoundPlan) -> Any:
    """Spawn the detached player for ``plan`` and return its ``Popen`` (or ``None``)."""
    if plan.kind != "file" or plan.path is None:
        return None
    try:
        from local_operator.proc import spawn_detached_child

        return spawn_detached_child(player_argv(plan, leg="preview"))
    except Exception:  # noqa: BLE001 — an audition must never affect its caller
        logger.debug("preview sound failed to start", exc_info=True)
        return None


def stop_preview() -> bool:
    """Stop the audition player this process started, if it is still sounding.

    Called before every new audition and by the TUI on the gestures that end
    one (a commit, ``/sound off``, leaving the list). Signals the EXACT child
    by its ``Popen`` handle (never a pid looked up later, which could have been
    reused) and its process group, with SIGTERM: the player's default handler
    exits at once and the audio device stops with the process. ``True`` when a live player was
    stopped. Never raises.
    """
    global _PREVIEW_CHILD
    with _PREVIEW_LOCK:
        child, _PREVIEW_CHILD = _PREVIEW_CHILD, None
    if child is None:
        return False
    try:
        if child.poll() is not None:
            return False
        if os.name == "posix":
            # The player is a session leader (`spawn_detached` uses setsid), so
            # its pid IS its group id. Signal the GROUP: on the afplay fallback
            # the sound comes from a grandchild that terminating the leader
            # alone would leave playing. `poll()` above just proved the leader
            # unreaped, and a pid is not reused until it is reaped. The only
            # reaper is `proc`'s off-thread `wait()`, which returns only after
            # the leader exits, so the reuse window is the instant between that
            # `poll()` and this signal. That invariant holds only while nothing
            # else `wait()`s on this child: add a second reaper and the pid
            # here can name a stranger's group.
            import signal

            os.killpg(child.pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised on Windows hosts
            child.terminate()
        return True
    except Exception:  # noqa: BLE001 — an already-gone child is the goal state
        logger.debug("could not stop the preview player", exc_info=True)
        return False
