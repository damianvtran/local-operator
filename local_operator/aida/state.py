"""Aida's durable state, and the lock that serialises her mutations.

Her per-install facts live in small JSON files under ``<config>/aida/`` rather
than in config keys, because none of them is a setting a user authors: which
session id she lives in, when she was last paused, and how much escalation
budget she has spent today are runtime-managed bookkeeping, and putting them in
``settings_io`` would invite hand edits to values the runtime is the writer of.

Four files, one owner each:

- ``state.json``          — ``{"schema_version", "session_id", "created_at",
  "paused_at", "extras": {"day", "armed"}}``. The session identity R7 is built
  on, plus pause bookkeeping and the escalation budget ledger.
- ``escalate.json``       — her escalation in-tray; written by Aida during a
  turn, consumed by the engine (see :mod:`local_operator.aida.proactive`).
- ``onboarding.json``     — the greeting ledger (see
  :mod:`local_operator.aida.onboarding`; slice B extends it with the nudge
  ledger).
- ``ensure.lock``         — the cross-process mutex over every read-modify-write
  of the three files above.

**Import-light by contract.** The readers of this module are boot paths (the
TUI, the server lifespan) and *every session open and wake persist* — the
latter only ever takes the cheap branch (:func:`is_aida_session` is one stat).
Importing anything heavier than the stdlib here would put it on the session
construction path for every conversation on the machine. The one non-stdlib
import is ``local_operator.wakes.lock``, itself stdlib-only apart from
``procstate``, and it is imported for the lock's fd discipline rather than
re-implemented here (see :func:`locked`).

**Tolerant reads, deliberate writes.** Readers treat a missing file as "not
configured" and a corrupt one as absent-with-a-log — a hand-edited or
truncated state file must cost Aida an ensure, never the session that asked
about her. Writers are strict (atomic ``os.replace``, exceptions propagate) so
a caller that must not silently lose a write (the bootstrap's create) can tell
success from failure and the best-effort callers (pause receipts) can wrap it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding every Aida file. A flat name (not
#: dotted) because unlike the per-session sidecars this is a directory a
#: person may reasonably open and read; the lock file inside it is the only
#: dotted member.
AIDA_DIRNAME = "aida"

STATE_NAME = "state.json"
ONBOARDING_NAME = "onboarding.json"
ESCALATE_NAME = "escalate.json"
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
#: a `/aida` keystroke) should answer rather than park.
LOCK_WAIT_S = 5.0

#: The environment kill switch, mirroring ``LOCAL_OPERATOR_NO_HERDR`` and
#: ``LOCAL_OPERATOR_NO_NOTIFICATIONS``. A deployment that must never carry
#: Aida (a harness-only automation install) exports it; truthiness is tested
#: per the house convention, with ``0``/``false``/``no``/``off`` meaning "on".
ENV_DISABLE = "LOCAL_OPERATOR_NO_AIDA"


_FALSY = {"0", "false", "no", "off", ""}


def env_disabled() -> bool:
    """Whether ``LOCAL_OPERATOR_NO_AIDA`` disables her for this process.

    Read from the environment on every call (never cached) for the same reason
    ``paths.config_dir`` is: tests and launchers set it after import.
    """
    return os.environ.get(ENV_DISABLE, "").strip().lower() not in _FALSY


def aida_dir(config_dir: Path | str) -> Path:
    return Path(config_dir) / AIDA_DIRNAME


def state_path(config_dir: Path | str) -> Path:
    return aida_dir(config_dir) / STATE_NAME


def onboarding_path(config_dir: Path | str) -> Path:
    return aida_dir(config_dir) / ONBOARDING_NAME


def escalate_path(config_dir: Path | str) -> Path:
    return aida_dir(config_dir) / ESCALATE_NAME


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
        logger.warning("aida: unreadable %s at %s; treating as absent", what, path, exc_info=True)
        return None
    if not isinstance(data, dict):
        logger.warning("aida: %s at %s is not an object; treating as absent", what, path)
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
        logger.warning("aida: skipping state with unknown schema_version at %s", config_dir)
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


def is_aida_session(config_dir: Path | str, session_id: str) -> bool:
    """Whether ``session_id`` is the session her state names.

    THE cheap identity check, asked by every session open (see
    :func:`local_operator.aida.proactive.filter_on_load`) and every wake
    persist. One stat + one short read on the machine whose state exists; one
    ``FileNotFoundError`` on every other machine and every other session.
    """
    if not session_id:
        return False
    return session_id_of(config_dir) == session_id


def update_state(config_dir: Path | str, **fields: Any) -> dict[str, Any]:
    """Read-modify-write of ``state.json``. Caller holds the lock.

    Creates the file (with ``created_at``) when absent, so callers that layer
    bookkeeping (``paused_at``, the budget ledger) onto a fresh install do not
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
    """Hold the cross-process mutex over every Aida file, synchronously.

    Delegates to :class:`local_operator.wakes.lock.WakeWriteLock` — the same
    bounded, non-blocking, refuse-rather-than-degrade fd discipline — pointed
    at ``<config>/aida/`` with this module's own file name. Reusing the class
    (rather than copying it or importing its private helpers) is what keeps
    the two locks' platform handling from drifting; the ``name`` override is
    the one extension it needed.

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

    directory = aida_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return WakeWriteLock(directory, timeout_s=timeout_s, name=LOCK_NAME)


def note_lock_refusal(what: str, exc: BaseException) -> None:
    """One quiet line for a lock refusal, at the level its retryability deserves.

    THE REFUSAL IS NORMAL, and it was being logged as a defect. These locks are
    taken by several ATTENDED writers at once — the TUI launch hook's ensure
    task, the app's first-run route, the runtime's reconcile, the wake
    supervisor — so :class:`~local_operator.wakes.lock.WakeLockBusy` is the
    answer ``wakes.lock`` documents for a peer that held the lock for the whole
    wait: "a peer's temporary hold and re-running is the fix". The call sites
    below used to let it reach their broad ``except Exception`` arms, which log
    ``exc_info=True``, so a first-run boot that was working correctly printed a
    full traceback into the operator's log; the clean-log contract in
    ``tests/e2e/test_tui_boot_e2e.py`` watched for exactly that (CI ``tui-e2e
    (ubuntu-latest, 1)``, run 37886200214, and the reviewer's held-lock
    reproducer).

    THE SPLIT IS THE POINT, not the silence: a busy lock is a miss for this tick
    (the loser of an arm writes the same row the winner is writing) and logs at
    INFO; :class:`~local_operator.wakes.lock.WakeLockUnavailable` will refuse
    again until the store's permissions change, so it logs at WARNING — with the
    exception's own sentence, which names the remedy — and still without a
    stack, because the failure is the store rather than this call path. Every
    OTHER exception keeps its caller's stack: a defect must stay loud.
    """
    from local_operator.wakes.lock import WakeLockBusy

    if isinstance(exc, WakeLockBusy):
        logger.info("aida: %s deferred: the store lock is held by a peer; retried next time", what)
        return
    logger.warning("aida: %s refused: the store lock could not be created (%s)", what, exc)


def consume_escalations(config_dir: Path | str) -> list[Any]:
    """Read ``escalate.json`` and DELETE it in one step; return its ``wakes`` list.

    Read-and-unlink rather than read-and-log-a-marker: the file is a one-shot
    request in-tray, and a request that stayed on disk would be re-armed on
    every later reconcile. The window between the read and the unlink has no
    await in it, so within one process the consume is atomic against the
    event loop; across processes the loser of a race sees ``ENOENT`` and
    returns empty, which is the correct "someone else has it" answer.

    Tolerant by contract: unusable content is logged, removed, and answered
    with an empty list, because a stuck in-tray would otherwise re-warn on
    every persist for the life of the install.
    """
    path = escalate_path(config_dir)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.warning("aida: unreadable escalate.json at %s; dropping it", path, exc_info=True)
        _unlink_quietly(path)
        return []
    _unlink_quietly(path)
    if not isinstance(data, dict):
        logger.warning("aida: escalate.json is not an object; dropping it")
        return []
    wakes = data.get("wakes")
    if not isinstance(wakes, list):
        return []
    return list(wakes)


def restore_escalations(config_dir: Path | str, requests: Sequence[Any]) -> None:
    """Put UNARMED requests back in ``escalate.json``, ahead of anything new.

    The counterpart of :func:`consume_escalations` for a drain that stopped
    early (review round 1, M1a): an owner that appears mid-drain refuses the
    arm with a 503, and the requests the sweep had already taken out of the
    tray must not evaporate — the owner's own reconcile reads the FILE, so
    "leave the rest for that owner" is only implementable by writing them back.

    MERGED rather than overwritten: a turn running in the owner process may
    have appended to the tray between the consume and this write, and its
    request is newer than the ones being restored — so the remainder goes
    FIRST and the file's current content follows it. Tolerant like its
    sibling: an unreadable current tray is treated as empty (it was going to
    be dropped by the next consume anyway), and a failed write is logged
    rather than raised, because the alternative is failing the /aida resume
    this drain serves.
    """
    path = escalate_path(config_dir)
    existing: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        current = data.get("wakes") if isinstance(data, dict) else None
        if isinstance(current, list):
            existing = list(current)
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logger.warning("aida: unreadable escalate.json at %s; replacing it", path, exc_info=True)
    try:
        write_json(path, {"wakes": [*requests, *existing]})
    except OSError:
        logger.warning(
            "aida: could not restore %d escalation request(s)", len(requests), exc_info=True
        )


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("aida: could not remove %s", path, exc_info=True)
