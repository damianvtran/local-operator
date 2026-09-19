"""The durable index of ARCHIVED sessions.

WHY A FILE AT ALL, AND WHY HERE. Everything else a listing ranks by is live
state — activity mtimes, the runtime registry, the attention store — so a
restart empties it. An archive is the opposite: a deliberate, durable statement
that this conversation should stop being offered without being destroyed. It
therefore needs the smallest durable thing that survives a restart, exactly as
:mod:`local_operator.tui.sidebar_pins` did for pins.

TWO ALTERNATIVES WERE REJECTED, and the reasons are the constraint this module
is shaped by:

* **A per-session sidecar** (``sessions/<id>/archived.json``) would ride along
  when a session is forked or copied — ``fork._EXCLUDED_SIDECARS`` is an
  allow-list, so a marker nothing remembers to exclude makes every fork of an
  archived conversation archived too, which is a silent inherited state and not
  something a user asked for. It would also have to be added to
  ``retention._SIDECAR_NAMES`` (the canonical list of machine bookkeeping, whose
  spellings ``test_retention.py`` pins) purely to keep it out of every byte and
  activity count. And it would cost one read per ROW on the picker's synchronous
  UI-thread path, where the store is read once per scan.
* **A field on the sidebar pins file.** ``sidebar_pins``'s own docstring refuses
  to become an object — it is a bare array precisely so it cannot evolve into
  something a reader must interpret — and pins and archives are different
  populations with different lifecycles (unpinning is a keystroke, archiving is
  a deliberate act). Two facts in one file means every writer arbitrates over
  both.

WHY THIS ONE IS FREE OF TEXTUAL, like its sibling: reading and writing an
archive is a pure function over a config root, so the server (``/v1/desktop``)
and the TUI share exactly one implementation rather than one per frontend.

A BARE ARRAY, NOT A VERSIONED OBJECT, for ``sidebar_pins``'s reason and one
more: the read is PRUNED AT READ against the session store, so the file holds no
fact a reader must interpret — an entry is either a live session directory or it
is dropped. There is nothing for a version field to migrate.

MULTI-PROCESS: LAST WRITER WINS, accepted, and the granularity is the whole
index rather than one id — the same concession ``sidebar_pins`` records, made
for the same reason (no cross-process lock for a small index, ``os.replace`` in
the same directory so a reader never sees a torn file). It is stated here
because this store now has three writers (the TUI's ``/archive``, the desktop
route, and a future CLI), so the window a collision can land in is wider than
the pins store's was when it made the same trade.

ARCHIVING THE LAST SESSION LEAVES THE FILE HOLDING ``[]``, deliberately: an
empty array is this module's resting state and reads back identically to an
absent file, so there is no second write path to keep correct beside the one
atomic replace.

NOTHING HERE REMOVES A SESSION, and nothing here is consulted by
``session/cleanup.py``'s REMOVAL policy. An archived session is therefore still
eligible for the automatic sweep — archive hides a conversation from the lists,
it does not exempt it from retention. That is a deliberate limit rather than an
oversight: exempting archived sessions from ``max_sessions``/``max_inactive_days``
would make every configured limit unsatisfiable by archiving, which is the one
way a user could fill a disk with a gesture that reads as tidying. The design
record (``docs/design/session-archive.md``) states it where a user would look.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

ARCHIVED_FILE = "archived-sessions.json"
#: Against ``PINS_LIMIT = 50``: archiving is a tidying gesture applied to whole
#: runs of old conversations rather than to a handful of favourites, so the cap
#: is an order of magnitude larger. It is a bound on the FILE (what an unbounded
#: append could grow to), not a policy — a user with 300 archived sessions loses
#: the OLDEST archive records, never a session, and the cap is measured from the
#: front because the list is newest-first.
ARCHIVED_LIMIT = 200


def read_archived(config_dir: Path) -> list[str]:
    """The archived session ids, newest first; ``[]`` when unreadable.

    Best-effort by contract, like ``read_pins``: this file only ever NARROWS a
    listing that ranks perfectly well without it, so every failure mode —
    absent, truncated, holding a JSON object instead of an array, holding
    non-strings — degrades to "nothing is archived" rather than costing the user
    the picker, the sidebar or the search.

    PRUNED AT READ against the session store: an id whose directory is gone
    (retention swept it, or the user deleted it) is dropped here rather than
    rendered as an archived row that resolves to nothing. Doing it on the read
    path is what keeps this module free of any coordination with
    ``session/cleanup.py`` — deletion needs to know nothing about archives, and
    the ``/v1/desktop`` delete route inherits that for free.
    """
    directory = Path(config_dir)
    try:
        raw = json.loads((directory / ARCHIVED_FILE).read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.debug("archived sessions unreadable; continuing without them", exc_info=True)
        return []
    if not isinstance(raw, list):
        return []
    sessions = directory / "sessions"
    return [
        item
        for item in raw
        # A stored entry is a session id: ONE bare directory name. Checked
        # before the store-prune below, because the prune joins the id onto
        # ``sessions/`` and ``Path.__truediv__`` does not keep it there —
        # ``sessions / "/tmp"`` IS ``/tmp``, and ``../agents`` climbs out of the
        # store. The rule is ``session_directory_name``'s, REUSED rather than
        # re-spelled so a file this app wrote cannot redirect a read outside
        # ``sessions/``; it is imported here rather than at module scope because
        # ``session.catalog`` imports ``resume``, which imports this module.
        if isinstance(item, str)
        and item == Path(item).name
        and _is_session_id(item)
        and (sessions / item).is_dir()
    ]


def _is_session_id(value: str) -> bool:
    from local_operator.session.catalog import session_directory_name

    return session_directory_name(value)


def archived_ids(config_dir: Path) -> frozenset[str]:
    """The archived ids as a set, for the membership test a listing runs per row.

    The scan takes this rather than a list so a store with 200 archived sessions
    does not turn its per-row predicate into a linear search.
    """
    return frozenset(read_archived(config_dir))


def set_archived(config_dir: Path, session_id: str, archived: bool) -> bool:
    """Put ``session_id`` into the requested archive STATE and return that state.

    DESIRED STATE, NOT A TOGGLE, for ``set_pin``'s reason: this backs an HTTP
    route, and a retried toggle flips the archive back — the user reports "the
    archive keeps un-archiving itself". The wire carries the state the caller
    wants so a retry lands on the same state.

    A NO-OP WRITES NOTHING, in both directions. That is what keeps a re-archive
    from REORDERING the list (the store is newest-first, so an unconditional
    re-archive of an id already in it would move the order to a retry) and,
    more importantly, what keeps two frontends pressing the same state from
    writing the whole index against each other for no change.

    Never raises, like its sibling: its caller is a route or a keypress handler
    answering a user who pressed something, and a read-only config directory
    must cost them the archive rather than the request.
    """
    directory = Path(config_dir)
    current = read_archived(directory)
    if archived:
        if session_id in current:
            return True
        entries = [session_id, *current]
    else:
        if session_id not in current:
            return False
        entries = [item for item in current if item != session_id]
    _write_archived(directory, entries)
    return archived


def _write_archived(directory: Path, entries: list[str]) -> None:
    """Replace the archive file with ``entries``, capped, atomically, best-effort.

    THE SINGLE WRITE PATH, so the cap and the atomic replace cannot be lost by a
    verb that prepared its own list — the defect ``sidebar_pins._write_pins``
    records from its own first cut, where the cap was the half that would have
    gone missing first.

    Same-directory temporary plus ``os.replace``, the discipline every small
    index here uses: a torn read of this file would silently empty a user's
    archive, and same-directory replace is the only atomic form.

    Never raises, for the reason :func:`set_archived` states.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle_fd, temporary = tempfile.mkstemp(dir=directory, prefix=".archived-sessions-")
        try:
            with os.fdopen(handle_fd, "w") as handle:
                # Sliced rather than trimmed in place: the caller's list is not
                # this function's to mutate.
                json.dump(entries[:ARCHIVED_LIMIT], handle)
            os.replace(temporary, directory / ARCHIVED_FILE)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        logger.debug("could not record the archived session", exc_info=True)
