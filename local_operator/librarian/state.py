"""The Librarian's durable state, and the lock that serialises her mutations.

Her per-install facts live in one small JSON file under
``<config>/librarian/`` rather than in config keys. None of them is a setting a
user authors: which session id she lives in, when she was last paused, and the
cadence phase the engine keeps are runtime-managed bookkeeping, and putting
them in ``settings_io`` would invite hand edits to values the runtime is the
writer of. The settings she DOES expose (``librarian.enabled``, ``.name``,
``.cadence.*``) are read through :mod:`local_operator.librarian.proactive` and
:mod:`local_operator.librarian.bootstrap`, never cached here.

Two files, one owner each:

- ``state.json``   — ``{"schema_version", "session_id", "created_at",
  "paused_at", "cadence_phase", "cadence_pending", "cadence_held",
  "extras", "last_run_at", "catch_up_for"}``. Identity, pause bookkeeping, the
  cadence phase ledger and the bounded-extra ledger.
- ``ensure.lock``  — the cross-process mutex over every read-modify-write of
  ``state.json``.

**Import-light by contract.** The readers of this module are boot paths (the
TUI, the server lifespan) and *every session open and wake persist* — the
latter only ever takes the cheap branch (:func:`is_librarian_session` is one
stat). Importing anything heavier than the stdlib here would put it on the
session construction path for every conversation on the machine. The one
non-stdlib import is ``local_operator.wakes.lock``, itself stdlib-only apart
from ``procstate``, and it is imported for the lock's fd discipline rather
than re-implemented here (see :func:`locked`).

**Tolerant reads, deliberate writes.** Readers treat a missing file as "not
configured" and a corrupt one as absent-with-a-log — a hand-edited or
truncated state file must cost the Librarian an ensure, never the session that
asked about her. Writers are strict (atomic ``os.replace``, exceptions
propagate) so a caller that must not silently lose a write (the bootstrap's
create) can tell success from failure and the best-effort callers (pause
receipts) can wrap it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding every Librarian file. A flat name
#: (not dotted) because unlike the per-session sidecars this is a directory a
#: person may reasonably open and read; the lock file inside it is the only
#: dotted member.
LIBRARIAN_DIRNAME = "librarian"

STATE_NAME = "state.json"
EXTRAS_NAME = "extras.json"
LOCK_NAME = "ensure.lock"

#: Bumped only on an incompatible change to ``state.json``'s shape. Readers
#: that do not understand a future schema treat the file as absent, exactly as
#: they treat a corrupt one: her session can be re-ensured, and guessing at
#: half-understood state is how a bootstrap duplicates a conversation.
STATE_SCHEMA = 1

#: How long a mutation waits for the lock before giving up. Shorter than the
#: wake-write lock's bound because the work under it is a handful of tiny
#: file writes, never a transcript parse: sub-millisecond in practice, so a
#: wait this long means a peer is wedged, and the callers that matter (boot,
#: the desktop's ``open``) should answer rather than park.
LOCK_WAIT_S = 5.0

#: The environment kill switch, mirroring ``LOCAL_OPERATOR_NO_AIDA`` and
#: ``LOCAL_OPERATOR_NO_HERDR``. A deployment that must never carry the
#: Librarian (a harness-only automation install) exports it; truthiness is
#: tested per the house convention, with ``0``/``false``/``no``/``off``
#: meaning "on".
ENV_DISABLE = "LOCAL_OPERATOR_NO_LIBRARIAN"

_FALSY = {"0", "false", "no", "off", ""}


def env_disabled() -> bool:
    """Whether ``LOCAL_OPERATOR_NO_LIBRARIAN`` disables her for this process.

    Read from the environment on every call (never cached) for the same reason
    ``paths.config_dir`` is: tests and launchers set it after import.
    """
    return os.environ.get(ENV_DISABLE, "").strip().lower() not in _FALSY


def librarian_dir(config_dir: Path | str) -> Path:
    return Path(config_dir) / LIBRARIAN_DIRNAME


def state_path(config_dir: Path | str) -> Path:
    return librarian_dir(config_dir) / STATE_NAME


def extras_path(config_dir: Path | str) -> Path:
    return librarian_dir(config_dir) / EXTRAS_NAME


def read_json(path: Path, *, what: str) -> dict[str, Any] | None:
    """One small JSON object, or ``None`` when absent or unusable.

    Unusable means unreadable, not an object, or without the expected schema —
    all treated identically for the reason the module docstring gives: every
    one of these files is rebuildable state, and a reader that raised would
    take down the session open (or the boot) that asked.
    """
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning(
            "librarian: unreadable %s at %s; treating as absent", what, path, exc_info=True
        )
        return None
    if not isinstance(data, dict):
        logger.warning("librarian: %s at %s is not an object; treating as absent", what, path)
        return None
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic replace of one small JSON file. Raises on failure.

    The staged write + ``os.replace`` discipline every small index in this
    tree uses: a reader must never see a torn file, and the same-directory
    replace is what makes that true on every platform we run on. The temp
    name starts with ``.`` so nothing scanning the directory reads it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_state(config_dir: Path | str) -> dict[str, Any] | None:
    """Her state, or ``None`` when never configured (or unusable)."""
    state = read_json(state_path(config_dir), what="state")
    if state is None:
        return None
    if state.get("schema_version") != STATE_SCHEMA:
        logger.warning("librarian: skipping state with unknown schema_version at %s", config_dir)
        return None
    return state


def write_state(config_dir: Path | str, state: dict[str, Any]) -> None:
    payload = dict(state)
    payload["schema_version"] = STATE_SCHEMA
    write_json(state_path(config_dir), payload)


def session_id_of(config_dir: Path | str) -> str | None:
    state = read_state(config_dir)
    if state is None:
        return None
    session_id = state.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def read_extras(config_dir: Path | str) -> list[Any]:
    """The pending ``librarian-extra-*`` requests, tolerant by contract.

    ``{"requests": [...]}`` with each entry the wake tool's own request
    grammar (``in``/``at`` plus an optional message). Unlike Aida's tray this
    one is READ and then CLEARED explicitly by the engine after it has armed
    the rows: a crash between the two duplicates a request rather than losing
    it, which is the safe direction for a one-a-day budget.
    """
    data = read_json(extras_path(config_dir), what="extras") or {}
    requests = data.get("requests")
    return list(requests) if isinstance(requests, list) else []


def clear_extras(config_dir: Path | str) -> None:
    """Remove the tray. Missing is success; an unreadable one is removed too."""
    try:
        extras_path(config_dir).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("librarian: could not clear the extras tray", exc_info=True)


def is_librarian_session(config_dir: Path | str, session_id: str) -> bool:
    """Whether ``session_id`` is the session her state names.

    THE cheap identity check, asked by every session open and every wake
    persist. ONE STAT for every other session on the machine: the state file's
    absence answers before any open, and only the install that has her pays the
    short read.
    """
    if not session_id:
        return False
    try:
        if not state_path(config_dir).is_file():
            return False
    except OSError:
        return False
    return session_id_of(config_dir) == session_id


def update_state(config_dir: Path | str, **fields: Any) -> dict[str, Any]:
    """Read-modify-write of ``state.json``. Caller holds the lock.

    Creates the file (with ``created_at``) when absent, so callers that layer
    bookkeeping (``paused_at``, the extra ledger) onto a fresh install do not
    each need their own "does it exist" branch. A corrupt existing file is
    replaced — the writers below are authoritative about the fields they set,
    and refusing to write over unreadable state would strand every later
    mutation behind one bad byte.
    """
    state = read_state(config_dir) or {}
    state.setdefault("created_at", int(time.time() * 1000))
    state.update(fields)
    write_state(config_dir, state)
    return state


@contextmanager
def locked(config_dir: Path | str, *, timeout_s: float = LOCK_WAIT_S) -> Iterator[None]:
    """Hold the cross-process mutex over every Librarian file, synchronously.

    Delegates to :class:`local_operator.wakes.lock.WakeWriteLock` — the same
    bounded, non-blocking, refuse-rather-than-degrade fd discipline — pointed
    at ``<config>/librarian/`` with this module's own file name. Reusing the
    class (rather than copying it or importing its private helpers) is what
    keeps the two locks' platform handling from drifting.

    A contended acquire raises :class:`~local_operator.wakes.lock.WakeLockBusy`
    or ``WakeLockUnavailable``. Callers treat that as a refusal — boot paths
    degrade to "try again next time", ops answer with the retry sentence.
    """
    lock = wake_lock(config_dir, timeout_s=timeout_s)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def wake_lock(config_dir: Path | str, *, timeout_s: float = LOCK_WAIT_S):
    """The same lock as an OBJECT, for callers that hold it across awaits.

    ``ensure_session`` creates her session inside the lock and its transcript
    append is a coroutine, so it cannot use the context manager without
    parking the event loop in ``acquire``'s retry sleep. It takes this and
    follows ``WakeWriteLock``'s own documented pattern instead: acquire in
    ``asyncio.to_thread``, work, release in ``to_thread`` — so a contended
    acquire waits on a worker while the loop keeps painting.
    """
    from local_operator.wakes.lock import WakeWriteLock

    directory = librarian_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return WakeWriteLock(directory, timeout_s=timeout_s, name=LOCK_NAME)


def note_lock_refusal(what: str, exc: BaseException) -> None:
    """One quiet line for a lock refusal, at the level its retryability deserves.

    THE REFUSAL IS NORMAL: several attended writers take this lock at once (the
    TUI launch hook, the server lifespan, the desktop route), so a
    ``WakeLockBusy`` is the documented "a peer held it; re-run" answer rather
    than a defect. Logged without ``exc_info`` so a working first-run boot does
    not print a traceback into the operator's log (the clean-log contract in
    ``tests/e2e/test_tui_boot_e2e.py`` greps for exactly that).
    """
    logger.info("librarian: %s was refused by a held lock: %s", what, exc)


def op_refusal_sentence(verb: str, name: str, exc: BaseException) -> str:
    """The ONE sentence an attended surface answers a refused control op with.

    ``pause``/``resume`` are reachable from the desktop route and the receipts;
    both meet the same refusal: a peer holds the Librarian store lock for the
    whole wait, so :func:`locked` raises before the op runs. The COPY is the
    lock module's own (``str(exc)``: ``WakeLockBusy`` carries its busy
    sentence); what this function owns is the SHAPE, in one place, so the
    surfaces cannot drift into two accounts of one refusal.
    """
    return f"could not {verb} {name}: {exc}"
