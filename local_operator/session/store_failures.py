"""Which of three very different store failures a desktop request actually met.

WHY THIS EXISTS. The shared failure ladder in ``routes/desktop_sessions.py``
answered EVERY ``sqlite3.Error`` -- lock contention, a full disk, an unopenable
store, a corrupt store -- with one sentence: *"Read state is busy right now. It
will catch up on its own."*, raised ``from None`` with no log record anywhere.
On 2026-09-17 the boot volume hit zero bytes free. ``SQLITE_CANTOPEN`` --
SQLite unable to create the file it needs -- arrived at that ladder and the
operator was told the read state was momentarily busy and would heal itself,
while the one action that could not help was appended by the client ("Send it
again") and the server logged nothing at all. Attributing it took an hour of
log archaeology across ``runtime.log``.

Three conditions, three answers, because they need three different actions:

* :data:`STORE_BUSY` -- genuine contention. Transient and retryable; the client
  keeps its retry hint and the user does nothing.
* :data:`STORE_OUT_OF_SPACE` -- the volume is full. Retrying cannot help and the
  user CAN act: free space, then send again. This is the case that was
  misreported, and the one the repro in ``scripts/repro-enospc-send.py``
  demonstrates end to end.
* :data:`STORE_UNAVAILABLE` -- the store could not be opened, read or written
  for any other reason. Retrying will not help either, but the remedy is
  checking the machine, not the disk gauge.

WHAT THE CODES ASSERT, BECAUSE A RENDERER HAS TO SAY SOMETHING. The two
non-retryable codes mean *this request was not admitted*: the store write is
what failed, so no durable row exists for the message and there is nothing for
the user to reconcile. A client may state that plainly -- "this message was not
sent" -- rather than hedging about whether it reached the agent (UX round 1,
U4, cross-repo: the renderer's held-claim line was contradicting the sentence
above it). Two honest limits on that claim:

* It is the STORE that did not take the write. The SQLite stores are
transactional, so a failed statement leaves no row; that is what makes "nothing
was written" true here rather than merely hopeful.
* The one nuance is the ``ENOSPC`` ``OSError`` arm below, where the failing
write is a file append rather than a transaction: an earlier write in the same
message may have completed (measured: an attachment blob written before its
sidecar failed). The message still was not admitted and the remedy is
unchanged, but "nothing was written" is a claim about the ADMISSION, not about
every byte the request touched.

WHY THE DISCRIMINATOR IS ``sqlite_errorname`` AND NOT THE MESSAGE. SQLite's own
text is prose that changes between releases, is localized by wrappers, and --
the trap here -- is AMBIGUOUS where it matters most: a full volume and a
read-only directory BOTH produce ``OperationalError('unable to open database
file')`` with ``SQLITE_CANTOPEN``, measured, so the text and the errorname
together cannot separate them. Python 3.11+ exposes the certified code
(``sqlite_errorcode`` / ``sqlite_errorname``), which is stable, and this repo
runs 3.12-3.14.

WHY ``SQLITE_CANTOPEN`` NEEDS A SECOND SIGNAL. ``SQLITE_FULL`` says "out of
space" outright, but the failure the operator actually hit was ``CANTOPEN``,
which is equally the shape of a missing directory and of a permissions problem.
So for the connect/IO family the volume's own free space decides, read with
``shutil.disk_usage`` -- a READ. It deliberately does not probe by writing: an
error path may never perform the operation that is failing, and a detection path
may never mutate the filesystem (``browser_bridge/state.py``'s ``state_path``
docstring records the incident that rule comes from -- a full disk turned a
read-only path computation into a second ``OSError`` raised from inside the
handler for the first one).

``FULL_VOLUME_FLOOR_BYTES`` is sized from the measurement, not from taste: on a
bounded 30 MB APFS image SQLite refused a create with 1.29 MB still reported
free (the repro's own capture), so a floor at or below a megabyte would have
classified the operator's incident as ``store_unavailable`` and kept saying
"check the machine" about a full disk. Any volume with less than this much free
is full in the sense that matters to a desktop app writing a journal and a
transcript.

NOT THE CLIENT'S COPY. No sentence here is the exception's own text: a store
error names file paths, and the rule that a store's wording never reaches the
renderer is the one the ladder already applies to ``ConnectionError``. The real
exception is logged server-side instead, with the request path and session, by
the ladder.

WHY THIS LIVES UNDER ``session/`` RATHER THAN ``server/utils/`` (agent review
round 1, R1; UX round 1, U2). The TUI's ``/notifications`` meets the same three
conditions the desktop ladder does, and it must answer them the same way -- but
it also must not import ``local_operator.server`` to do it, which is a layering
rule the TUI keeps today. One classifier, two consumers, is the only shape that
stops the two surfaces from disagreeing about what "busy" means; the server path
keeps its import through the delegating ``server/utils/store_failures.py`` shim,
so no server caller changed.
"""

from __future__ import annotations

import errno
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

#: Genuine lock contention on the shared stores. Retryable, and the client's
#: retry hint is CORRECT for it -- which is why it is the only one of the three
#: that keeps the sentence the app already relays.
STORE_BUSY = "store_busy"

#: The volume holding the store is full, so NOTHING became durable for the
#: request that met it (see the module docstring: the SQLite stores are
#: transactional). Not retryable as-is, and actionable.
STORE_OUT_OF_SPACE = "store_out_of_space"

#: The store could not be opened, read or written for any other reason, and
#: again nothing became durable for the request. "Retrying will not help" is
#: literal: the condition is not going to clear between two identical sends.
#:
#: THE TOKEN IS KEPT DELIBERATELY, against a rename this tree already made once.
#: ``session/errors.py``'s ``SessionStoreUnavailable`` is ``session_store_unavailable``
#: precisely because a bare ``store_unavailable`` is ALREADY the MCP credentials
#: tool's answer for a failure to write a SECRET (that docstring states the
#: collision at length, and it is the reason the sibling got the
#: ``<subsystem>_unavailable`` spelling). This ladder's token therefore collides
#: too, on a fourth store. It is kept because the desktop contract pins it
#: VERBATIM -- the renderer in ``local-operator-ui`` withholds its retry hint by
#: matching this exact string, and both sides were pinned before the collision
#: was noticed. Renaming it now is a coordinated two-repo change, not a tidy-up:
#: an unmatched code silently reinstates the retry hint the incident was about.
#: Recorded in the PR thread as a follow-up; changing it here alone would be the
#: silent divergence the contract exists to prevent.
STORE_UNAVAILABLE = "store_unavailable"

#: Unchanged, and only the contention case carries it now. The shipped app
#: matches nothing in this sentence; it withholds its generic retry hint for the
#: two codes that cannot be helped by retrying.
BUSY_MESSAGE = "Read state is busy right now. It will catch up on its own."

#: Names the disk and the remedy, in the order they apply -- and names WHERE the
#: disk is, which the first version of this sentence did not do (design round 1,
#: D1). "This computer is out of disk space" is true and useless on a machine
#: with several volumes and a relocated config root; the renderer deliberately
#: paints the backend's sentence verbatim rather than keeping a second copy of
#: it, so the destination has to be stated here or nowhere. ``{root}`` is filled
#: by :func:`out_of_space_message` with the config root the request actually used.
#:
#: Naming it does NOT breach the rule that a store's own text never reaches the
#: client (the arm in ``routes/desktop_sessions.py`` states it at length). That
#: rule exists to stop SQLite's message -- which carries whatever paths its
#: failure happened to pass through -- being echoed; this is a path THIS process
#: chose, and the operator needs it.
OUT_OF_SPACE_MESSAGE = (
    "This computer is out of disk space, so the message could not be written. "
    "Free some space on the volume holding {root} and send it again."
)

#: Says retrying will not help, and names the place to look. Same reason as
#: above for naming it: "check this machine" is a destination nobody can reach.
UNAVAILABLE_MESSAGE = (
    "The session store could not be read or written. Retrying will not help; "
    "check {root} and the disk it is on."
)

#: Below this much free space, a store that could not create its journal or WAL
#: is reporting the disk, not a permissions problem. See the module docstring
#: for the measurement this number comes from.
FULL_VOLUME_FLOOR_BYTES = 16 * 1024 * 1024

#: ``SQLITE_FULL`` says it outright; nothing else has to be consulted for it.
_OUT_OF_SPACE_ERRONAMES = frozenset({"SQLITE_FULL"})

#: Contention, in the two spellings SQLite uses. Both are retryable.
_BUSY_ERRONAMES = frozenset(
    {
        "SQLITE_BUSY",
        "SQLITE_BUSY_RECOVERY",
        "SQLITE_BUSY_SNAPSHOT",
        "SQLITE_LOCKED",
        "SQLITE_LOCKED_SHAREDCACHE",
    }
)

#: The family that means "SQLite could not create or write the file it needed".
#: Ambiguous on its own -- a full volume, an absent directory and a read-only
#: directory all land here (measured: the read-only case reports CANTOPEN too) --
#: so these consult the volume. Prefixes rather than exact names because the
#: IOERR family is reported with a suffix per failing operation
#: (``SQLITE_IOERR_WRITE``, ``SQLITE_IOERR_FSYNC``, ...).
_AMBIENT_ERRONAME_PREFIXES = (
    "SQLITE_CANTOPEN",
    "SQLITE_IOERR",
    "SQLITE_READONLY",
    "SQLITE_PERM",
)

#: ``OSError`` errnos that mean this machine has nowhere left to put the bytes.
#: ``EDQUOT`` is a quota where ``ENOSPC`` is the volume; both are the disk's
#: answer rather than the code's.
_SPACE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})


@dataclass(frozen=True)
class StoreFailure:
    """One classified store failure: what to answer, and how to log it."""

    status: int
    code: str
    message: str
    #: Contention is routine and expected to clear, so it is logged at WARNING;
    #: the other two are conditions an operator has to act on and are logged at
    #: ERROR. Both levels are visible at the default ``LOG_LEVEL`` of WARNING,
    #: which is the point -- the record this module exists to produce must be
    #: there in the log the operator is already reading.
    level: int
    #: Whether the record carries the exception's traceback. TRUE for the two
    #: conditions an operator has to act on, where the stack -- which store,
    #: which statement -- IS the finding. FALSE for contention: a lock that clears
    #: on its own is a routine event, and a full traceback per retry is noise that
    #: buries the records worth reading (review round 1, R5). The record still
    #: names the code, the route and the session either way, so a contention that
    #: does NOT clear is still attributable from the log.
    traceback: bool


def display_root(root: str | Path | None) -> str:
    """The config root as the operator would type it: ``~``-relative when it can be.

    A refusal that says "the volume holding /Users/someone/.local-operator" is
    addressable but noisy, and the same sentence rendered for a relocated or
    temporary root (an isolated run, a test) must stay truthful about the
    directory in use -- so the path is shortened when it is under the home
    directory and left alone when it is not.
    """
    try:
        path = Path(root) if root is not None else Path.home() / ".local-operator"
    except TypeError:
        # Not a path at all (a double in a test, an injected root of another
        # shape): the generic phrase is better than a repr of it.
        return "the local-operator configuration directory"
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    # ``path`` IS the home directory (a test config root, an isolated run), and
    # a bare ``~/.`` is not what anybody types.
    if not relative.parts:
        return "~"
    return "~/" + str(relative)


def out_of_space_message(root: str | Path | None = None) -> str:
    """The 507 sentence, naming the volume to free space on (design round 1, D1)."""
    return OUT_OF_SPACE_MESSAGE.format(root=display_root(root))


def unavailable_message(root: str | Path | None = None) -> str:
    """The 500 sentence, naming the directory to check."""
    return UNAVAILABLE_MESSAGE.format(root=display_root(root))


def _busy() -> StoreFailure:
    return StoreFailure(503, STORE_BUSY, BUSY_MESSAGE, logging.WARNING, traceback=False)


def _out_of_space(root: str | Path | None = None) -> StoreFailure:
    return StoreFailure(
        507, STORE_OUT_OF_SPACE, out_of_space_message(root), logging.ERROR, traceback=True
    )


def _unavailable(root: str | Path | None = None) -> StoreFailure:
    return StoreFailure(
        500, STORE_UNAVAILABLE, unavailable_message(root), logging.ERROR, traceback=True
    )


def volume_is_full(root: str | Path | None) -> bool:
    """Whether the volume holding ``root`` has no room left to write into.

    A read, never a probe write: see the module docstring. A root that cannot be
    measured answers ``False`` -- "the disk is full" is a claim that must be
    earned, and a root this process cannot stat says nothing about the disk.
    """
    try:
        usage = shutil.disk_usage(root if root is not None else Path.home())
    except OSError:
        return False
    return usage.free < FULL_VOLUME_FLOOR_BYTES


def sqlite_store_failure(error: sqlite3.Error, root: str | Path | None = None) -> StoreFailure:
    """Classify a ``sqlite3`` error. Total: every one of them is one of the three.

    There is deliberately no "unknown sqlite error" branch left unanswered. The
    bug this module exists to remove was a catch-all that answered one class's
    copy to all of them; an unclassified error defaulting to the *transient*
    sentence is that same bug wearing a smaller hat, so the default here is
    :data:`STORE_UNAVAILABLE`.
    """
    errorname = str(getattr(error, "sqlite_errorname", "") or "")
    if errorname in _BUSY_ERRONAMES:
        return _busy()
    if errorname in _OUT_OF_SPACE_ERRONAMES:
        return _out_of_space(root)
    if errorname.startswith(_AMBIENT_ERRONAME_PREFIXES):
        # Could not create or write the file it needed, for one of three
        # reasons. The volume separates the disk-full one from the rest.
        return _out_of_space(root) if volume_is_full(root) else _unavailable(root)
    return _unavailable(root)


def store_failure(error: BaseException, root: str | Path | None = None) -> StoreFailure | None:
    """Classify ``error`` as a store failure, or ``None`` if it is not one.

    ``None`` is the caller's signal to re-raise untouched rather than to invent
    an answer: this ladder sits under every desktop control-plane route, and
    widening it to answer for arbitrary ``OSError``s would swallow exactly the
    failures whose own routes have deliberately better words for them
    (``desktop_sessions.move_session`` answers a bad target with a 409 naming the
    path, and an unmounted volume has to keep reaching the user as it does).
    """
    if isinstance(error, sqlite3.Error):
        return sqlite_store_failure(error, root)
    if isinstance(error, OSError) and error.errno in _SPACE_ERRNOS:
        # The non-sqlite writes on the send path -- the transcript append, the
        # attachment store -- raise this rather than a sqlite error, and a
        # message that could not be persisted is the same condition to the user.
        return _out_of_space(root)
    return None
