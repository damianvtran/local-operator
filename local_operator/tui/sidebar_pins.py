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

MULTI-PROCESS: LAST WRITER WINS, accepted. Two ``lop`` sessions pinning at the
same instant means the second write is what the file holds; no precedent in this
codebase takes a cross-process lock for a small index, and the same-directory
``os.replace`` means a reader never sees a torn file — only an older one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from local_operator.session.catalog import session_directory_name

logger = logging.getLogger(__name__)

PINS_FILE = "sidebar-pins.json"
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


def toggle_pin(config_dir: Path, session_id: str) -> bool:
    """Pin ``session_id`` if it is not pinned, unpin it if it is.

    Returns the NEW state: ``True`` when the session is now pinned, ``False``
    when the pin was removed. A pin re-applied to an already-pinned session is
    an unpin, which is what makes one chord both verbs.

    Written to a temporary file in the SAME directory and ``os.replace``d over
    the target, the discipline every small index here uses (``config.py``,
    ``move_targets.py``, ``multiplexer/markers.py``): a torn read of this file
    would silently empty a user's pins, and same-directory replace is the only
    form that is atomic.

    Never raises. This runs from a keypress the user has already been given
    feedback for, so a read-only config directory must cost them the pin and
    not the session.
    """
    directory = Path(config_dir)
    current = read_pins(directory)
    entries = [item for item in current if item != session_id]
    # Nothing was removed, so this is a pin rather than an unpin.
    pinned = len(entries) == len(current)
    if pinned:
        entries.insert(0, session_id)
    del entries[PINS_LIMIT:]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle_fd, temporary = tempfile.mkstemp(dir=directory, prefix=".sidebar-pins-")
        try:
            with os.fdopen(handle_fd, "w") as handle:
                json.dump(entries, handle)
            os.replace(temporary, directory / PINS_FILE)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        logger.debug("could not record the sidebar pin", exc_info=True)
    return pinned
