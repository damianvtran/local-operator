"""The sidebar's durable pinned-session list.

Kept free of widget imports for the reason :mod:`local_operator.tui.move_targets`
is: reading and toggling a pin are pure functions over a config root, so they
are testable — and reusable by a non-Textual frontend — without importing
Textual to learn which sessions a user pinned.

WHY A FILE AT ALL. Everything else the sidebar ranks by is live state: activity
mtimes, the runtime registry, the attention store. Quit every session and all of
it goes empty. A pin is the opposite — a deliberate, durable statement that
*this* conversation stays at the top until the user says otherwise — so it needs
the smallest durable thing that survives a restart: a capped JSON array of
session ids, newest pin first.

A BARE ARRAY, NOT A VERSIONED OBJECT, deliberately. The shape cannot evolve into
something a reader must interpret, and an unreadable file already degrades to
"no pins", so a version field would buy a migration path for a format that has
nowhere to go.

UNPINNING THE LAST PIN LEAVES THE FILE HOLDING ``[]``, deliberately: an empty
array is this module's resting state and reads back identically to an absent
file, so there is no second write path to keep correct beside the one atomic
replace.

MULTI-PROCESS: ONE WRITER AT A TIME, under a lock file. The index is a whole-list
read-modify-write, so two writers landing inside one window did more than
arbitrate over the id they shared — the later ``os.replace`` discarded every
entry the earlier one had added. That was accepted while two ``lop`` processes
were the only plausible colliders; it stopped being acceptable when the boot
hooks made it routine: Aida's and the Librarian's ensures start together on a
fresh install, and the measured result was a file holding one assistant's pin
while the other session existed with everything else it needed (QA on #2627).
Every mutation therefore takes ``<config>/.sidebar-pins.lock`` through the
house's own bounded, refuse-rather-than-degrade :class:`WakeWriteLock`, the
same lock Aida's store reuses for this reason. The critical section is a small
read plus one atomic replace, so contention is microseconds; the bounded wait
is a wedge guard, not an expected delay.

THE GRANULARITY REMAINS THE WHOLE INDEX. The lock makes the read-modify-write
atomic; it does not make per-id writes independent. Callers that need to know
whether their pin is durable read it back (``read_pins``); a refused lock is
reported as the state observed on disk, never as the state that was asked for.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from local_operator.session.catalog import session_directory_name

logger = logging.getLogger(__name__)

PINS_FILE = "sidebar-pins.json"

#: The DELIBERATE-UNPIN ledger: session ids a person explicitly unpinned. It
#: exists because the bootstraps reassert an assistant's pin (Aida's on create,
#: the Librarian's on every ensure); without this, a person who unpins the
#: Librarian would find the row back at the next launch, which reads as "the
#: pin keeps re-pinning itself". Only a HUMAN write records or clears it — a
#: programmatic reassert (``source="bootstrap"``) leaves the intent alone.
UNPINNED_FILE = "sidebar-pins.unpinned.json"

#: ``set_pin`` provenance. ``"user"`` is the desktop route and every direct
#: caller (the human gesture); ``"bootstrap"`` is an always-on agent's boot
#: reassert, which must neither count as intent nor erase it.
SOURCE_USER = "user"
SOURCE_BOOTSTRAP = "bootstrap"

#: The cross-process mutex over every read-modify-write of the pin file. Dot-led
#: so no reader or config-directory scan mistakes it for the index itself.
LOCK_NAME = ".sidebar-pins.lock"

#: How long a mutation waits for the lock. The work under it is one small read
#: and one atomic replace — sub-millisecond — so a wait this long means a peer
#: is wedged. Kept well below the wake lock's 5 s because one caller is a TUI
#: keypress: a frozen frame is worse than a dropped pin, and the next toggle or
#: boot retries.
LOCK_WAIT_S = 1.0
#: Against ``move_targets.RECENTS_LIMIT = 20``: a pin is a user's deliberate,
#: durable selection rather than a trace of where they have been, so a small
#: multiple of the recents cap is the right order of magnitude.
PINS_LIMIT = 50


def read_pins(config_dir: Path) -> list[str]:
    """The pinned session ids, newest pin first; ``[]`` when unreadable.

    Best-effort by contract. This file is an enhancement to a list that ranks
    perfectly well without it, so every failure mode — absent, truncated,
    holding a JSON object instead of an array, holding non-strings — degrades
    to "no pinned sessions" rather than costing the user the sidebar.

    PRUNED AT READ against the session store: an id whose directory is gone
    (cleanup removed it, or the user deleted it) is dropped here rather than
    rendered as a broken row. Doing it on the read path is what keeps this
    module free of any coordination with ``session/cleanup.py`` — deletion
    needs to know nothing about pins.
    """
    directory = Path(config_dir)
    try:
        raw = json.loads((directory / PINS_FILE).read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.debug("sidebar pins unreadable; continuing without them", exc_info=True)
        return []
    if not isinstance(raw, list):
        return []
    sessions = directory / "sessions"
    return [
        item
        for item in raw
        # A pin is a session id: ONE bare directory name. Checked before the
        # store-prune below, because the prune joins the id onto `sessions/`
        # and `Path.__truediv__` does not keep it there — `sessions / "/tmp"`
        # IS `/tmp`, and `../agents` climbs out of the store. Nothing renders
        # from a bogus entry today (`load_catalog` hydrates none of them), so
        # this is defence in depth: the same rule `session_directory_name`
        # states for discovery metadata, reused rather than re-spelled, so a
        # file this app wrote cannot redirect a read outside `sessions/`.
        if isinstance(item, str)
        and item == Path(item).name
        and session_directory_name(item)
        and (sessions / item).is_dir()
    ]


def explicitly_unpinned(config_dir: Path) -> set[str]:
    """Session ids a person deliberately unpinned; ``set()`` when unusable.

    Pruned against the session store on read for the same reason the pin list
    is: a marker for a conversation that no longer exists has nothing left to
    protect and must not accumulate forever.
    """
    directory = Path(config_dir)
    try:
        raw = json.loads((directory / UNPINNED_FILE).read_text())
    except FileNotFoundError:
        return set()
    except (OSError, ValueError):
        logger.debug("sidebar unpin ledger unreadable; ignoring it", exc_info=True)
        return set()
    if not isinstance(raw, list):
        return set()
    sessions = directory / "sessions"
    return {
        item
        for item in raw
        if isinstance(item, str)
        and item == Path(item).name
        and session_directory_name(item)
        and (sessions / item).is_dir()
    }


def _write_unpinned(directory: Path, ids: set[str]) -> None:
    """Record the deliberate-unpin set, atomically and best-effort (caller locks)."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not ids:
            (directory / UNPINNED_FILE).unlink(missing_ok=True)
            return
        handle_fd, temporary = tempfile.mkstemp(dir=directory, prefix=".sidebar-unpinned-")
        try:
            with os.fdopen(handle_fd, "w") as handle:
                json.dump(sorted(ids)[:PINS_LIMIT], handle)
            os.replace(temporary, directory / UNPINNED_FILE)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        logger.debug("could not record the deliberate unpin", exc_info=True)


def set_pin(
    config_dir: Path,
    session_id: str,
    pinned: bool,
    *,
    source: str = SOURCE_USER,
) -> bool:
    """Put ``session_id`` into the requested pin STATE and return that state.

    DESIRED STATE, NOT A TOGGLE, and idempotence is the whole reason this
    exists beside :func:`toggle_pin`: the TUI's verb is a toggle because it is
    a keypress, while this store now backs an HTTP route whose request can be
    retried after its response is lost. A retried toggle flips the pin back and
    reads to the user as "the pin keeps un-pinning itself" — so the wire carries
    the state the caller wants and this function makes the call safe to repeat.

    A NO-OP WRITES NO PIN FILE: re-pinning a pinned session returns ``True``
    without touching the list, and unpinning an unpinned one returns ``False``
    with no list write. (A HUMAN unpin does record the intent ledger — see
    ``UNPINNED_FILE`` — because that is how the next boot tells an intentional
    removal from the race this store's lock fixes.) Keeping the list untouched
    is what stops a re-pin from REORDERING it: the store is newest-pin-first,
    so an unconditional re-pin of an id already in it would silently move the
    user's order to a retry, and a flaky link would rewrite an order nobody
    asked to change.

    ``source`` separates a human gesture from an always-on agent's boot
    reassert (``SOURCE_BOOTSTRAP``): only a human write records a deliberate
    unpin (or clears one by pinning again), so the next boot can tell a lost
    pin apart from an intent. Default is the human verb, because every existing
    caller is one.

    Never raises, like its neighbour: its caller is a route answering a user who
    pressed something, and a read-only config directory must cost them the pin
    rather than the request.
    """
    directory = Path(config_dir)
    with _pins_lock(directory) as held:
        if not held:
            # A REFUSED LOCK STILL ANSWERS THE DESIRED STATE, and the return
            # has never meant "durable": this module answers a press whose
            # feedback is already on screen, and an unwritable config dir has
            # always cost the pin rather than the action. Nothing is written —
            # that is what the lock guarantee buys — and the next boot's
            # reassert repairs the file.
            return pinned
        current = read_pins(directory)
        if source == SOURCE_USER:
            # A HUMAN write is intent in BOTH directions: pinning clears an
            # earlier deliberate unpin, unpinning records one.
            marked = explicitly_unpinned(directory)
            if pinned and session_id in marked:
                _write_unpinned(directory, marked - {session_id})
            elif not pinned and session_id not in marked:
                _write_unpinned(directory, marked | {session_id})
        if pinned:
            if session_id in current:
                return True
            entries = [session_id, *current]
        else:
            if session_id not in current:
                return False
            entries = [item for item in current if item != session_id]
        _write_pins(directory, entries)
        return pinned


def toggle_pin(config_dir: Path, session_id: str) -> bool:
    """Pin ``session_id`` if it is not pinned, unpin it if it is.

    Returns the NEW state: ``True`` when the session is now pinned, ``False``
    when the pin was removed. A pin re-applied to an already-pinned session is
    an unpin, which is what makes one chord both verbs.

    Unconditional where :func:`set_pin` is not: a keypress names no desired
    state, so an already-pinned id is the unpin the user asked for rather than a
    no-op. A caller that knows the state it wants must use ``set_pin``.
    """
    directory = Path(config_dir)
    with _pins_lock(directory) as held:
        if not held:
            # Same answer as an unwritable directory: the state the toggle
            # WOULD have produced, without the unsafe write.
            return session_id not in read_pins(directory)
        current = read_pins(directory)
        entries = [item for item in current if item != session_id]
        # Nothing was removed, so this is a pin rather than an unpin.
        pinned = len(entries) == len(current)
        marked = explicitly_unpinned(directory)
        if pinned:
            entries.insert(0, session_id)
            if session_id in marked:
                _write_unpinned(directory, marked - {session_id})
        elif session_id not in marked:
            _write_unpinned(directory, marked | {session_id})
        _write_pins(directory, entries)
        return pinned


@contextmanager
def _pins_lock(directory: Path) -> Iterator[bool]:
    """Hold the pin-index mutex; yield whether this caller owns the write.

    Reuses :class:`local_operator.wakes.lock.WakeWriteLock` rather than growing
    a second platform lock — the same reuse Aida's root-level ``ensure.lock``
    documents. The import stays function-local because this module is loaded by
    the TUI's first frame and a label read must never wait on the wake
    subsystem's own imports.

    A refusal (busy for the whole wait, or a directory that cannot hold the
    lock file) yields ``False``: both verbs then answer the state their caller
    asked for and write nothing. Failure direction is deliberate — a missing
    pin is repaired by the next boot (both bootstraps reassert), while a write
    made without the lock is the defect this exists to remove.
    """
    try:
        from local_operator.wakes.lock import (
            WakeLockBusy,
            WakeLockUnavailable,
            WakeWriteLock,
        )

        directory.mkdir(parents=True, exist_ok=True)
        lock = WakeWriteLock(
            directory,
            timeout_s=LOCK_WAIT_S,
            name=LOCK_NAME,
            busy_sentence=(
                "Something else on this device is updating the pinned conversations. "
                "Try again in a moment."
            ),
        )
        try:
            lock.acquire()
        except (WakeLockBusy, WakeLockUnavailable):
            logger.debug("sidebar pins: the pin lock was busy or unavailable", exc_info=True)
            yield False
            return
    except OSError:
        logger.debug("sidebar pins: could not take the pin lock", exc_info=True)
        yield False
        return
    try:
        yield True
    finally:
        lock.release()


def _write_pins(directory: Path, entries: list[str]) -> None:
    """Replace the pin file with ``entries``, capped, atomically, best-effort.

    THE SINGLE WRITE PATH, shared by both verbs, and it owns BOTH halves of the
    discipline: the cap and the atomic replace. A verb that prepared its own
    trimmed list would leave the other one's guarantee to be re-established by
    hand at each call site — which is what the first cut did, and the cap is the
    half that would have gone missing first: a third verb calling this function
    would have inherited no cap at all while every existing test stayed green,
    because the per-verb cap tests exercise the verbs rather than the writer.

    The cap keeps the OLDEST pins out of the file, and ``PINS_LIMIT`` is
    measured from the FRONT because the list is newest-pin-first, so the pin the
    user just made is the one position can never drop.

    Written to a temporary file in the SAME directory and ``os.replace``d over
    the target, the discipline every small index here uses (``config.py``,
    ``move_targets.py``, ``multiplexer/markers.py``): a torn read of this file
    would silently empty a user's pins, and same-directory replace is the only
    form that is atomic.

    Never raises. Both callers answer a user who has already been given feedback
    for their press, so a read-only config directory must cost them the pin and
    not the action.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle_fd, temporary = tempfile.mkstemp(dir=directory, prefix=".sidebar-pins-")
        try:
            with os.fdopen(handle_fd, "w") as handle:
                # Sliced rather than trimmed in place: the caller's list is not
                # this function's to mutate, and a caller that kept it (to report
                # what it wrote, say) must not find it silently shortened.
                json.dump(entries[:PINS_LIMIT], handle)
            os.replace(temporary, directory / PINS_FILE)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        logger.debug("could not record the sidebar pin", exc_info=True)
