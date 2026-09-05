"""The ONE session-cleanup policy, and the only code allowed to remove a
session directory.

WHY THIS MODULE EXISTS, AND WHY IT IS OFF
=========================================

Every earlier automatic cleanup under ``sessions/`` is gone. The first
generation (age/count/byte ceilings) evicted a running session's transcript
out from under it. The second (an "empty directory" reaper) grew, in #576,
into a backfill that removed any directory whose transcript held no
``"role": "user"`` row — behind an opt-out toggle that ``/settings`` wrote
under a nested key and the reaper read under a flat one, so it never
worked. The runtime exit path carried a fourth (#622, an ``rmdir`` on
lease-only directories). On the night the incident was noticed, 225 of an
operator's 244 named sessions — 296,617 model calls of history — were gone.

The forensics could not pin every deletion on one of those reapers: none of
the logged reaps had a model call, and no reaper could remove a directory
holding a transcript. That is why this module is built around the
assumption that **something outside its own judgement may try to delete
the real store**, and refuses on the actor's behalf:

- **Nothing runs unless** ``session.cleanup.enabled`` **is true.** The
  default is false. Every limit below is inert while it is false, whatever
  value it holds — :func:`cleanup_from_config` returns before it lists the
  store.
- **The store must be marked.** :func:`remove_session_dir` refuses any
  target that is not directly under a ``sessions/`` directory carrying the
  :data:`STORE_MARKER_NAME` file, and refuses any target not under the
  config dir the process was given. A store under ``/tmp`` or a test's
  ``tmp_path`` has the marker only if its creator wrote it — the deliberate
  opt-in — and the operator's real store gets it once, from
  :func:`mark_store` on first use and from the config migration.
- **Every removal is logged at WARNING with the session id, the policy
  that chose it and the reason**, and appended to ``sessions/.cleanup-log.
  jsonl`` — an always-on, per-store record, because the earlier reaps were
  only discoverable by hunting through per-process log files.
- **Hard guards apply even when enabled.** A session with a live claim,
  lease or pid; one with an armed wake; one with unread spooled mail; the
  session being built right now; and the :data:`RECENT_KEEP` most recently
  active sessions are never candidates. The guards are checked per
  directory at decision time, not from a cached listing.
- **Fail closed.** Any guard that cannot be evaluated (unreadable file,
  stat error, import failure) keeps the directory.

The enforcement is ``tests/unit/session/test_no_session_deletion.py``: it
walks ``local_operator/`` and fails, naming file:line, on any ``rmtree``/
``rmdir``/``rename``/``replace``/``move`` outside this module whose
argument could be a session directory; and it asserts this module removes
nothing when disabled, with every limit set to 1 over a 50-session store.

The policy is also runnable by hand: ``lop sessions cleanup --dry-run``
prints what it WOULD remove, and ``lop sessions cleanup`` runs it. Neither
requires ``enabled`` — an explicit command is the user's judgement — but
both honour every hard guard.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_operator.session.retention import (
    _SIDECAR_NAMES,
    SESSIONS_DIRNAME,
    TRANSCRIPT_FILENAME,
    _activity_mtime,
    _is_claimed,
    _process_alive,
)

logger = logging.getLogger(__name__)

#: The path of the cleanup block under ``values``, spelled ONCE. The
#: ``settings_io`` rows use ``CLEANUP_PATH + (leaf,)`` and the consumer reads
#: ``ConfigManager.get_nested_value(CLEANUP_PATH + (leaf,))``; both sides share
#: this tuple so the flat-vs-nested mismatch that silenced the #576 opt-out
#: cannot be reintroduced by a typo on one side.
CLEANUP_PATH: tuple[str, ...] = ("session", "cleanup")

#: Defaults, one per leaf. These are the consumer's defaults — the registry
#: (``settings_io``) is checked against them by
#: ``test_every_default_matches_its_consumer``. ``enabled`` is False and every
#: limit is "unlimited": a config that says nothing gets no cleanup.
DEFAULT_ENABLED = False
DEFAULT_MAX_SESSIONS = 0
DEFAULT_MAX_INACTIVE_DAYS = 0
DEFAULT_MAX_TOTAL_BYTES = 0
DEFAULT_REMOVE_EMPTY = False

#: Marker file that a ``sessions/`` directory must carry before
#: :func:`remove_session_dir` will remove anything inside it. Written by
#: :func:`mark_store` when the harness creates or migrates its own store.
#: A directory without it is either not a local-operator store, or a store
#: from before the marker existed — both mean "refuse". Dotted so it never
#: reads as a session id to a listing.
STORE_MARKER_NAME = ".local-operator-store"

#: Append-only record of every removal, inside the store it describes. One
#: JSON object per line: ``{"at", "session", "policy", "reason", "dry_run"}``.
CLEANUP_LOG_NAME = ".cleanup-log.jsonl"

#: The N most recently active sessions are never candidates, whatever the
#: policy says. 10 matches the resume picker's first page: a session the user
#: can see at the top of ``/resume`` must not vanish between two launches.
RECENT_KEEP = 10


@dataclass(frozen=True)
class CleanupPolicy:
    """The five knobs, as read from config or the CLI. ``enabled`` gates
    everything; a limit of 0 means "no limit of that kind"."""

    enabled: bool = DEFAULT_ENABLED
    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_inactive_days: int = DEFAULT_MAX_INACTIVE_DAYS
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    remove_empty: bool = DEFAULT_REMOVE_EMPTY

    @property
    def has_any_limit(self) -> bool:
        return bool(
            self.max_sessions or self.max_inactive_days or self.max_total_bytes or self.remove_empty
        )


@dataclass(frozen=True)
class Candidate:
    """One directory the policy chose, and why. Returned so the CLI's dry run
    and the tests can assert on decisions rather than scrape log lines."""

    session: str
    policy: str
    reason: str


@dataclass
class CleanupResult:
    scanned: int = 0
    #: Directories the policy chose. In a dry run these were NOT removed.
    chosen: list[Candidate] = field(default_factory=list)
    removed: list[Candidate] = field(default_factory=list)
    #: ``(session, guard)`` for every directory a hard guard kept that a limit
    #: would otherwise have taken. Reported so a dry run shows what is
    #: protected, not only what is doomed.
    protected: list[tuple[str, str]] = field(default_factory=list)
    errors: int = 0
    dry_run: bool = False
    #: Why nothing ran, when nothing ran ("disabled", "no limits", "no store").
    skipped: str | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def policy_from_config(config_manager: Any) -> CleanupPolicy:
    """Read the policy through :meth:`ConfigManager.get_nested_value` on
    :data:`CLEANUP_PATH` — the exact path ``settings_io`` writes.

    Duck-typed so a test can hand in any object with the method. A manager
    without ``get_nested_value`` (a stub predating it) yields the disabled
    default rather than guessing from ``get_config_value``: a reader that
    silently used the flat accessor is the bug this module replaces.
    """
    getter: Callable[..., Any] | None = getattr(config_manager, "get_nested_value", None)
    if getter is None:
        return CleanupPolicy()

    def leaf(name: str, default: Any) -> Any:
        try:
            return getter(CLEANUP_PATH + (name,), default)
        except Exception:  # noqa: BLE001 — a broken config is not a reason to run
            return default

    return CleanupPolicy(
        enabled=bool(leaf("enabled", DEFAULT_ENABLED)),
        max_sessions=_coerce_int(leaf("max_sessions", DEFAULT_MAX_SESSIONS), DEFAULT_MAX_SESSIONS),
        max_inactive_days=_coerce_int(
            leaf("max_inactive_days", DEFAULT_MAX_INACTIVE_DAYS), DEFAULT_MAX_INACTIVE_DAYS
        ),
        max_total_bytes=_coerce_int(
            leaf("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES), DEFAULT_MAX_TOTAL_BYTES
        ),
        remove_empty=bool(leaf("remove_empty", DEFAULT_REMOVE_EMPTY)),
    )


# ---------------------------------------------------------------------------
# The store marker
# ---------------------------------------------------------------------------


def store_marker_path(sessions_dir: Path) -> Path:
    return sessions_dir / STORE_MARKER_NAME


def mark_store(sessions_dir: Path) -> None:
    """Stamp ``sessions_dir`` as a local-operator store.

    Idempotent and best-effort: a store that cannot be marked simply stays
    outside cleanup's reach, which is the safe side. Called when the harness
    creates its own store (session construction) and by the config migration,
    never by cleanup itself — the policy must not be able to authorise its
    own target.
    """
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        marker = store_marker_path(sessions_dir)
        if not marker.exists():
            marker.write_text(
                "This directory is a local-operator session store. "
                "Its presence lets `lop sessions cleanup` remove sessions here "
                "when the user has enabled the policy.\n",
                encoding="utf-8",
            )
    except OSError as exc:
        logger.debug("session cleanup: cannot mark store %s: %s", sessions_dir, exc)


def _refusal(target: Path, config_dir: Path | None) -> str | None:
    """Why :func:`remove_session_dir` must not touch ``target``; ``None`` if it may.

    Three independent checks, each sufficient to refuse: the target must be
    directly under a ``sessions/`` directory; that directory must carry the
    store marker; and, when a ``config_dir`` is given, it must be THAT config
    dir's store. Paths are resolved so a symlink into the real store cannot
    launder itself through a marked scratch store.
    """
    try:
        resolved = target.resolve(strict=True)
    except OSError:
        return "target does not exist or cannot be resolved"
    parent = resolved.parent
    if parent.name != SESSIONS_DIRNAME:
        return f"not directly under a '{SESSIONS_DIRNAME}/' directory"
    if not store_marker_path(parent).is_file():
        return f"store carries no {STORE_MARKER_NAME} marker"
    if config_dir is not None:
        try:
            expected = (config_dir / SESSIONS_DIRNAME).resolve()
        except OSError:
            return "config dir cannot be resolved"
        if parent != expected:
            return f"store {parent} is not this process's store {expected}"
    if resolved.is_symlink() or not resolved.is_dir():
        return "not a directory"
    return None


def remove_session_dir(
    target: Path,
    *,
    config_dir: Path | None,
    policy: str,
    reason: str,
    dry_run: bool = False,
) -> bool:
    """THE ONLY ``rmtree`` OF A SESSION DIRECTORY IN THIS CODEBASE.

    Refuses unless :func:`_refusal` clears the target; logs the refusal at
    WARNING so an attempt against an unmarked or foreign store is visible.
    Logs every removal at WARNING and appends it to the store's
    :data:`CLEANUP_LOG_NAME` BEFORE the ``rmtree``, so a crash mid-removal
    still leaves the record. Returns whether the directory was (or, in a dry
    run, would have been) removed.
    """
    why_not = _refusal(target, config_dir)
    if why_not is not None:
        logger.warning(
            "session cleanup: REFUSED to remove %s (%s); policy=%s reason=%s",
            target,
            why_not,
            policy,
            reason,
        )
        return False
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session": target.name,
        "policy": policy,
        "reason": reason,
        "dry_run": dry_run,
    }
    _append_cleanup_log(target.parent, record)
    if dry_run:
        logger.warning(
            "session cleanup (dry run): would remove %s; policy=%s reason=%s",
            target.name,
            policy,
            reason,
        )
        return True
    logger.warning("session cleanup: removing %s; policy=%s reason=%s", target.name, policy, reason)
    shutil.rmtree(target)
    return True


def _append_cleanup_log(sessions_dir: Path, record: dict[str, Any]) -> None:
    try:
        with (sessions_dir / CLEANUP_LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        # The removal still proceeds — the WARNING log line above carries the
        # same facts — but say so, because the jsonl is the record the user
        # is told to look at.
        logger.warning("session cleanup: cannot append to %s: %s", CLEANUP_LOG_NAME, exc)


# ---------------------------------------------------------------------------
# Hard guards
# ---------------------------------------------------------------------------


def _lease_owner_alive(directory: Path) -> bool | None:
    """Whether the ``.execution-lease`` names a live pid; ``None`` = no lease."""
    lease = directory / ".execution-lease"
    try:
        raw = lease.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return True  # unreadable: assume owned
    try:
        pid = json.loads(raw).get("pid")
    except (ValueError, AttributeError):
        return True  # corrupt lease: assume owned
    if not isinstance(pid, int):
        return True
    return _process_alive(pid)


def _has_wake(config_dir: Path, session: str) -> bool:
    try:
        from local_operator.wakes.store import entry_path

        return entry_path(config_dir, session).exists()
    except Exception:  # noqa: BLE001 — unprovable is "yes"
        return True


def _has_spooled_mail(directory: Path) -> bool:
    try:
        from local_operator.session.runtime.inbox import INBOX_NAME

        spool = directory / INBOX_NAME
        return spool.exists() and spool.stat().st_size > 0
    except Exception:  # noqa: BLE001 — unprovable is "yes"
        return True


def _guard(directory: Path, config_dir: Path, now: float) -> str | None:
    """The hard guards, in one place. Returns the guard's name when the
    directory must be kept, ``None`` when the policy may consider it."""
    try:
        if _is_claimed(directory, now):
            return "claimed by a live process"
    except OSError:
        return "claim unreadable"
    owned = _lease_owner_alive(directory)
    if owned:
        return "leased by a live process"
    if _has_wake(config_dir, directory.name):
        return "has an armed wake"
    if _has_spooled_mail(directory):
        return "has unread spooled mail"
    return None


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def _has_transcript(directory: Path) -> bool:
    try:
        return (directory / TRANSCRIPT_FILENAME).stat().st_size > 0
    except OSError:
        return False


def _dir_bytes(directory: Path) -> int:
    total = 0
    try:
        for entry in directory.rglob("*"):
            try:
                if entry.is_file() and entry.name not in _SIDECAR_NAMES:
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


@dataclass
class _Entry:
    path: Path
    activity: float
    has_transcript: bool
    size: int


def run_cleanup(
    config_dir: Path,
    policy: CleanupPolicy,
    *,
    live_dir: Path | None = None,
    now: float | None = None,
    dry_run: bool = False,
    explicit: bool = False,
) -> CleanupResult:
    """Apply ``policy`` to ``config_dir/sessions``.

    ``explicit`` is the CLI's flag: the user typed the command, so
    ``policy.enabled`` is not consulted — the command IS the consent. Every
    other caller leaves it False and gets nothing unless the config says so.

    Selection order, deliberately from least to most aggressive, with each
    limit re-checking the guards on its own candidates:

    1. ``remove_empty`` — directories with no non-empty transcript.
    2. ``max_inactive_days`` — last activity (:func:`_activity_mtime`, which
       ignores sidecars) older than the limit.
    3. ``max_sessions`` — beyond the N most recently active, oldest first.
    4. ``max_total_bytes`` — least recently active first until under budget.

    The :data:`RECENT_KEEP` newest sessions are excluded from every limit.
    """
    result = CleanupResult(dry_run=dry_run)
    if not explicit and not policy.enabled:
        result.skipped = "disabled"
        return result
    if not policy.has_any_limit:
        result.skipped = "no limits configured"
        return result
    sessions_dir = config_dir / SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        result.skipped = "no store"
        return result

    moment = now if now is not None else time.time()
    live_resolved: Path | None = None
    if live_dir is not None:
        try:
            live_resolved = live_dir.resolve()
        except OSError:
            live_resolved = None

    entries: list[_Entry] = []
    try:
        children = [child for child in sessions_dir.iterdir() if child.is_dir()]
    except OSError as exc:
        logger.warning("session cleanup: cannot scan %s: %s", sessions_dir, exc)
        result.errors += 1
        return result
    for child in children:
        result.scanned += 1
        try:
            if live_resolved is not None and child.resolve() == live_resolved:
                result.protected.append((child.name, "the current session"))
                continue
            stamp = child.stat().st_mtime
            activity = _activity_mtime(child, stamp)
            entries.append(_Entry(child, activity, _has_transcript(child), _dir_bytes(child)))
        except OSError:
            continue

    entries.sort(key=lambda entry: entry.activity, reverse=True)
    recent = {entry.path.name for entry in entries[:RECENT_KEEP]}
    chosen: dict[str, Candidate] = {}

    def consider(entry: _Entry, policy_name: str, reason: str) -> bool:
        name = entry.path.name
        if name in chosen:
            return True
        if name in recent:
            result.protected.append((name, f"one of the {RECENT_KEEP} most recent"))
            return False
        guard = _guard(entry.path, config_dir, moment)
        if guard is not None:
            result.protected.append((name, guard))
            return False
        chosen[name] = Candidate(name, policy_name, reason)
        return True

    if policy.remove_empty:
        for entry in entries:
            if not entry.has_transcript:
                consider(entry, "remove_empty", "no transcript")

    if policy.max_inactive_days:
        cutoff = moment - policy.max_inactive_days * 86400.0
        for entry in entries:
            if entry.activity < cutoff:
                idle_days = (moment - entry.activity) / 86400.0
                consider(
                    entry,
                    "max_inactive_days",
                    f"inactive {idle_days:.1f}d > {policy.max_inactive_days}d",
                )

    if policy.max_sessions:
        survivors = [entry for entry in entries if entry.path.name not in chosen]
        excess = survivors[policy.max_sessions :]
        for entry in reversed(excess):  # oldest first
            consider(
                entry,
                "max_sessions",
                f"beyond the {policy.max_sessions} most recently active",
            )

    if policy.max_total_bytes:
        survivors = [entry for entry in entries if entry.path.name not in chosen]
        total = sum(entry.size for entry in survivors)
        for entry in reversed(survivors):  # least recently active first
            if total <= policy.max_total_bytes:
                break
            if consider(
                entry,
                "max_total_bytes",
                f"store {total} B > {policy.max_total_bytes} B",
            ):
                total -= entry.size

    result.chosen = list(chosen.values())
    for candidate in result.chosen:
        target = sessions_dir / candidate.session
        try:
            done = remove_session_dir(
                target,
                config_dir=config_dir,
                policy=candidate.policy,
                reason=candidate.reason,
                dry_run=dry_run,
            )
        except OSError as exc:
            logger.warning("session cleanup: cannot remove %s: %s", candidate.session, exc)
            result.errors += 1
            continue
        if done:
            result.removed.append(candidate)
            if not dry_run:
                _forget_wake_entry(config_dir, candidate.session)
    return result


def _forget_wake_entry(config_dir: Path, session: str) -> None:
    """A removed session cannot have a wake fire for it; drop its index entry.
    Unreachable in practice (an armed wake is a hard guard) but keeps the
    index from pointing at nothing if the guard was bypassed by hand."""
    try:
        from local_operator.wakes.store import remove_entry

        remove_entry(config_dir, session)
    except Exception:  # noqa: BLE001
        pass


def cleanup_from_config(
    config_manager: Any, config_dir: Path, *, live_dir: Path | None = None
) -> CleanupResult:
    """The startup entry point. Reads the policy and runs it — which, with the
    default config, means reading five keys and returning."""
    policy = policy_from_config(config_manager)
    if not policy.enabled:
        return CleanupResult(skipped="disabled")
    result = run_cleanup(config_dir, policy, live_dir=live_dir)
    if result.removed:
        logger.warning(
            "session cleanup: removed %d of %d sessions (see %s)",
            len(result.removed),
            result.scanned,
            config_dir / SESSIONS_DIRNAME / CLEANUP_LOG_NAME,
        )
    return result
