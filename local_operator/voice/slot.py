"""The machine-wide voice slot: which ONE session is voice-enabled (V3).

``<config>/voice/active.json`` holds ``{session_id, enabled_at, instance_id}``
or does not exist. This module is its ONLY writer, and every write runs under a
cross-process file lock in the shape of ``aida/state.py``'s: a
:class:`~local_operator.wakes.lock.WakeWriteLock` pointed at ``<config>/voice/``
with this module's own lock name, so several daemons, windows and the runtime
can share one slot without two of them each believing they hold it.

WHY ONE FILE AND NOT A PER-SESSION FLAG. One-at-a-time is the operator's
requirement. With a flag per session, "enable B" would have to scan every
session and clear the others, and a crash halfway leaves two voices. One slot
makes the switch a single atomic replace: enabling B IS disabling A, and the
answer names A as ``replaced``.

WHY NOT RENDERER STATE. Several windows, a remote-paired backend and (slice 2)
the session's own agent all need the same fact, and only a file under the
config root is visible to all of them.

TOLERANT READS, STRICT WRITES — the aida rule. A missing, unreadable or
foreign-schema file reads as "no session enabled": voice is an affordance, and a
hand-edited file must cost the operator a re-enable, never a crashed route.
Writes are an atomic same-directory ``os.replace`` and raise on failure.

``instance_id`` records WHICH server process wrote the slot. It is
diagnostic: the slot deliberately survives a daemon restart (the operator's
choice of session does not change because the backend was updated), so no
reader treats a foreign ``instance_id`` as stale.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VOICE_DIRNAME = "voice"
ACTIVE_NAME = "active.json"
LOCK_NAME = "slot.lock"

#: The lock's bound. The work under it is one small read and one small
#: replace — sub-millisecond — so a wait this long means a wedged peer, and the
#: route answers ``voice_store_busy`` rather than parking a request on it.
LOCK_WAIT_S = 5.0

#: The lock's own refusal text, for logs and non-HTTP callers. Product words
#: only, the lock module's D7 rule. The desktop route answers with the
#: catalogued ``wire.voice.store_busy`` (same sentence), not with this string.
BUSY_SENTENCE = "Something else on this device is changing voice mode. Try again in a moment."


@dataclass(frozen=True)
class VoiceSlot:
    """The enabled session, as every reader sees it."""

    session_id: str
    #: Epoch MILLISECONDS, the unit ``aida/state.py`` stamps ``created_at`` in.
    enabled_at: int
    instance_id: str = ""

    def public(self) -> dict[str, Any]:
        """The frozen contract's ``active`` object: ``{session_id, enabled_at}``.

        ``instance_id`` stays on disk and off the wire: it names a server
        process, which a renderer has no use for and should not key on.
        """
        return {"session_id": self.session_id, "enabled_at": self.enabled_at}


@dataclass(frozen=True)
class SlotChange:
    """One write's outcome: the slot after it, and the session it displaced."""

    active: VoiceSlot | None
    #: The session that held the slot before an ``enable`` and is no longer in
    #: it. ``None`` when the slot was empty, or already held by the same
    #: session (re-enabling is idempotent and replaces nobody).
    replaced: str | None = None


def voice_dir(config_dir: Path | str) -> Path:
    return Path(config_dir) / VOICE_DIRNAME


def active_path(config_dir: Path | str) -> Path:
    return voice_dir(config_dir) / ACTIVE_NAME


def read_slot(config_dir: Path | str) -> VoiceSlot | None:
    """The enabled session, or ``None``. Never raises (see the module docstring)."""
    path = active_path(config_dir)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("voice: unreadable slot at %s; treating as empty", path, exc_info=True)
        return None
    if not isinstance(data, dict):
        logger.warning("voice: slot at %s is not an object; treating as empty", path)
        return None
    session_id = data.get("session_id")
    enabled_at = data.get("enabled_at")
    instance_id = data.get("instance_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    # ``bool`` is an ``int`` subclass; a ``true`` here is a hand edit, not a time.
    if not isinstance(enabled_at, int) or isinstance(enabled_at, bool):
        return None
    return VoiceSlot(
        session_id=session_id,
        enabled_at=enabled_at,
        instance_id=instance_id if isinstance(instance_id, str) else "",
    )


def is_voice_session(config_dir: Path | str, session_id: str) -> bool:
    """Whether ``session_id`` holds the slot — the check every voice route makes."""
    if not session_id:
        return False
    slot = read_slot(config_dir)
    return slot is not None and slot.session_id == session_id


def _write(config_dir: Path | str, slot: VoiceSlot) -> None:
    path = active_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "session_id": slot.session_id,
                    "enabled_at": slot.enabled_at,
                    "instance_id": slot.instance_id,
                },
                handle,
                separators=(",", ":"),
                sort_keys=True,
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _clear(config_dir: Path | str) -> None:
    try:
        active_path(config_dir).unlink()
    except FileNotFoundError:
        pass


def _lock(config_dir: Path | str, timeout_s: float):
    """The slot's cross-process mutex, reusing the wake lock's fd discipline.

    Imported here rather than at module scope for ``aida/state.py``'s reason:
    the lock module pulls ``procstate``, and a plain slot READ (the hot path)
    must not pay for it.
    """
    from local_operator.wakes.lock import WakeWriteLock

    directory = voice_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return WakeWriteLock(
        directory, timeout_s=timeout_s, name=LOCK_NAME, busy_sentence=BUSY_SENTENCE
    )


def enable(
    config_dir: Path | str,
    session_id: str,
    *,
    instance_id: str = "",
    now_ms: int | None = None,
    timeout_s: float = LOCK_WAIT_S,
) -> SlotChange:
    """Put ``session_id`` in the slot, displacing whoever held it.

    Synchronous and lock-holding: an async caller runs it in a worker
    (``asyncio.to_thread``), the wake lock's documented pattern. Raises
    ``WakeLockBusy``/``WakeLockUnavailable`` when the lock cannot be taken;
    the route answers both as ``voice_store_busy``.

    Re-enabling the session that already holds the slot keeps its original
    ``enabled_at``: nothing changed, so readers must not see a new edge.
    """
    if not session_id:
        raise ValueError("session_id is required")
    lock = _lock(config_dir, timeout_s)
    lock.acquire()
    try:
        current = read_slot(config_dir)
        if current is not None and current.session_id == session_id:
            return SlotChange(active=current, replaced=None)
        slot = VoiceSlot(
            session_id=session_id,
            enabled_at=now_ms if now_ms is not None else int(time.time() * 1000),
            instance_id=instance_id,
        )
        _write(config_dir, slot)
        return SlotChange(active=slot, replaced=current.session_id if current else None)
    finally:
        lock.release()


def disable(
    config_dir: Path | str,
    session_id: str,
    *,
    timeout_s: float = LOCK_WAIT_S,
) -> SlotChange:
    """Clear the slot IF ``session_id`` holds it; otherwise change nothing.

    Scoped to the named session on purpose: a window that still shows A as
    voice-enabled and sends ``disable A`` after another window switched to B
    must not switch B off. The answer always carries the slot as it now is.
    """
    lock = _lock(config_dir, timeout_s)
    lock.acquire()
    try:
        current = read_slot(config_dir)
        if current is None or current.session_id != session_id:
            return SlotChange(active=current, replaced=None)
        _clear(config_dir)
        return SlotChange(active=None, replaced=None)
    finally:
        lock.release()


def slot_token(config_dir: Path | str) -> tuple[int, int, int] | None:
    """A cheap change token for the slot file: ``(inode, size, mtime_ns)`` or ``None``.

    What the machine-wide feed polls on its 1 s probe cadence. The atomic
    replace mints a new inode per write, so two writes inside one mtime tick
    still move the token.
    """
    try:
        stat = os.stat(active_path(config_dir))
    except OSError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)
