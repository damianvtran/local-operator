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
walks ``local_operator/`` and fails, naming file:line, on any call named
``rmtree``/``rmdir``/``removedirs``/``rename``/``renames``/``replace``/
``move``/``unlink``/``remove`` outside this module — through any import
alias and on any receiver — unless allow-listed with a reason; and it
asserts this module removes nothing when disabled, with every limit set to
1 over a 50-session store.

The policy is also runnable by hand: ``lop sessions cleanup --dry-run``
lists what the limits WOULD remove (it may do so with the switch off, and
says so), and ``lop sessions cleanup`` runs it — only with ``enabled:
true``, or with ``--force`` after listing and a typed confirmation. Both
honour every hard guard.

The store marker is a guard against foreign and unmarked targets and the
CLI on a store nothing has booted; it is NOT a second gate on the harness's
own startup pass, which marks its store in ``_prepare`` before maintenance
runs. That pass is gated by ``enabled`` alone (QA round 1, Q4).
"""

from __future__ import annotations

import json
import logging
import os
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

#: Append-only record of every REAL removal, inside the store it describes.
#: One JSON object per line: ``{"at", "session", "title", "policy", "reason",
#: "actor", "pid"}``. Dry runs are NOT recorded — a rehearsal in the same file
#: as the losses made the log unreadable (13 of 22 rows in one QA store were
#: rehearsals). ``actor`` is ``"startup"`` (the maintenance pass, gated on
#: ``enabled``) or ``"cli"`` (``lop sessions cleanup``), which is the first
#: question after "why": who did it.
CLEANUP_LOG_NAME = ".cleanup-log.jsonl"

#: The LAST startup removal, as one JSON object beside the log: ``{"at",
#: "actor", "pid", "removed": N, "scanned": N, "policies": {name: count},
#: "record": path, "acknowledged": bool}``. Written only by a pass that
#: removed ≥1 session; overwritten by the next such pass. This is how a boot
#: that removed sessions gets to SAY SO on screen: the maintenance pass runs
#: in the runtime process, after the first frame, and the TUI may be a
#: viewer attached over a socket — so the fact is put on disk where every
#: viewer of that store can read it, and the first one to report it flips
#: ``acknowledged`` so the same removal is announced once, not on every
#: ``/resume``. Deliberately a file, not a notification bus: one fact, one
#: reader shape, durable across the process that produced it (UX round 1,
#: U1; the incident's launches painted an identical splash after 225
#: removals).
LAST_CLEANUP_NAME = "last-cleanup.json"

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
    and the tests can assert on decisions rather than scrape log lines.

    Carries what a user needs to JUDGE the decision, not only the id: a
    12-hex id says nothing, "Research thread 7 · 29d · 28 kB" does (UX round
    1, U4). ``title`` is resolved the way the resume picker resolves it.
    """

    session: str
    policy: str
    reason: str
    title: str = ""
    idle_days: float = 0.0
    size_bytes: int = 0


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
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _coerce_bool(value: Any, default: bool) -> bool:
    """A REAL boolean or the default — never ``bool(value)``.

    ``enabled: "false"`` in a hand-edited YAML is a non-empty string, and
    ``bool("false")`` is True: the one leaf where "garbage means the default"
    did not hold was the master switch (review round 1, R1-6). An operator
    frightened by the incident will hand-edit exactly this key, so anything
    that is not literally ``true``/``false`` (or the YAML 1.1 spellings a
    human types: yes/no/on/off, 0/1) reads as the default, which is OFF.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


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
        enabled=_coerce_bool(leaf("enabled", DEFAULT_ENABLED), DEFAULT_ENABLED),
        max_sessions=_coerce_int(leaf("max_sessions", DEFAULT_MAX_SESSIONS), DEFAULT_MAX_SESSIONS),
        max_inactive_days=_coerce_int(
            leaf("max_inactive_days", DEFAULT_MAX_INACTIVE_DAYS), DEFAULT_MAX_INACTIVE_DAYS
        ),
        max_total_bytes=_coerce_int(
            leaf("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES), DEFAULT_MAX_TOTAL_BYTES
        ),
        remove_empty=_coerce_bool(leaf("remove_empty", DEFAULT_REMOVE_EMPTY), DEFAULT_REMOVE_EMPTY),
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
    actor: str,
    title: str = "",
    dry_run: bool = False,
) -> bool:
    """THE ONLY ``rmtree`` OF A SESSION DIRECTORY IN THIS CODEBASE.

    Refuses unless :func:`_refusal` clears the target; logs the refusal at
    WARNING so an attempt against an unmarked or foreign store is visible.
    Logs every real removal at WARNING (naming the record file) and appends
    it to the store's :data:`CLEANUP_LOG_NAME` BEFORE the ``rmtree``, so a
    crash mid-removal still leaves the record. A dry run refuses and decides
    exactly as a real run would but writes nothing and logs at DEBUG — the
    CLI prints the decisions itself, and a WARNING per rehearsal doubled
    every line in a terminal (UX round 1, U3). Returns whether the directory
    was (or, in a dry run, would have been) removed.
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
    if dry_run:
        logger.debug(
            "session cleanup (dry run): would remove %s; policy=%s reason=%s",
            target.name,
            policy,
            reason,
        )
        return True
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session": target.name,
        "title": title,
        "policy": policy,
        "reason": reason,
        "actor": actor,
        "pid": os.getpid(),
    }
    _append_cleanup_log(target.parent, record)
    logger.warning(
        "session cleanup: removing %s%s; policy=%s reason=%s actor=%s (record: %s)",
        target.name,
        f" ({title})" if title else "",
        policy,
        reason,
        actor,
        target.parent / CLEANUP_LOG_NAME,
    )
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


def _session_title(directory: Path) -> str:
    """The name ``/resume`` would show for this directory, or ``""``.

    Resolved through ``resume.session_name`` so the CLI list and the picker
    agree on what a session is called. Import-lazy: ``resume`` is heavy and
    this runs only for the directories the policy CHOSE, never for the scan.
    """
    try:
        from local_operator.resume import session_name

        return session_name(directory, max_chars=48)
    except Exception:  # noqa: BLE001 — a name is a courtesy, never a reason to fail
        return ""


def run_cleanup(
    config_dir: Path,
    policy: CleanupPolicy,
    *,
    live_dir: Path | None = None,
    now: float | None = None,
    dry_run: bool = False,
    force: bool = False,
    actor: str = "startup",
) -> CleanupResult:
    """Apply ``policy`` to ``config_dir/sessions``.

    THE MASTER SWITCH GOVERNS EVERY CALLER. Round 1 let the CLI run with
    ``enabled: false`` on the theory that typing the command was consent;
    QA (Q1), UX (U2) and review (R1-5) each showed why that is the incident's
    shape: ``/settings`` leaves the limits in the file when the switch is
    turned OFF, so a user who read "off: nothing ever removes a session
    directory" and then ran the command "to see what it would do" lost 16 of
    34 sessions. Now ``enabled: false`` means nothing is removed by anyone
    unless ``force`` is set — and the CLI sets it only for ``--force``, after
    listing and confirming. A dry run with the switch off still LISTS what
    the limits would take (safe and useful) and reports ``skipped`` so the
    caller can say the switch is off.

    ``actor`` is recorded in the cleanup log: ``"startup"`` for the
    maintenance pass, ``"cli"`` for ``lop sessions cleanup``.

    Selection order, deliberately from least to most aggressive, with each
    limit re-checking the guards on its own candidates:

    1. ``remove_empty`` — directories with no non-empty transcript.
    2. ``max_inactive_days`` — last activity (:func:`_activity_mtime`: the
       transcript's or the mail spool's mtime, never a sidecar's) older than
       the limit.
    3. ``max_sessions`` — beyond the N most recently active, oldest first.
    4. ``max_total_bytes`` — least recently active first until under budget.

    The :data:`RECENT_KEEP` newest sessions are excluded from every limit.
    Ties on the activity clock break on the directory NAME so a dry run and
    the real run pick the same directories on any filesystem (R1-11).
    """
    result = CleanupResult(dry_run=dry_run)
    if not policy.enabled and not force and not dry_run:
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

    # Newest first; equal stamps fall back to the name so the order is a
    # property of the store, not of ``iterdir`` on this filesystem.
    entries.sort(key=lambda entry: (-entry.activity, entry.path.name))
    recent = {entry.path.name for entry in entries[:RECENT_KEEP]}
    chosen: dict[str, Candidate] = {}

    # A directory several limits would take is reported once, under the first
    # guard that saved it; without this a dry run listed the same session
    # per limit.
    protected_seen: set[str] = set()

    def protect(name: str, guard: str) -> None:
        if name not in protected_seen:
            protected_seen.add(name)
            result.protected.append((name, guard))

    def consider(entry: _Entry, policy_name: str, reason: str) -> bool:
        name = entry.path.name
        if name in chosen:
            return True
        if name in recent:
            protect(name, f"one of the {RECENT_KEEP} most recent")
            return False
        guard = _guard(entry.path, config_dir, moment)
        if guard is not None:
            protect(name, guard)
            return False
        chosen[name] = Candidate(
            name,
            policy_name,
            reason,
            title=_session_title(entry.path) if entry.has_transcript else "",
            idle_days=max(0.0, (moment - entry.activity) / 86400.0),
            size_bytes=entry.size,
        )
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
    if dry_run and not policy.enabled and not force:
        # Listed, not removed, and the caller is told why: the switch is off.
        result.skipped = "disabled"
    for candidate in result.chosen:
        target = sessions_dir / candidate.session
        try:
            done = remove_session_dir(
                target,
                config_dir=config_dir,
                policy=candidate.policy,
                reason=candidate.reason,
                actor=actor,
                title=candidate.title,
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
    result = run_cleanup(config_dir, policy, live_dir=live_dir, actor="startup")
    if result.removed:
        logger.warning(
            "session cleanup: removed %d of %d sessions (see %s)",
            len(result.removed),
            result.scanned,
            config_dir / SESSIONS_DIRNAME / CLEANUP_LOG_NAME,
        )
        write_last_cleanup(config_dir / SESSIONS_DIRNAME, result, actor="startup")
    return result


def write_last_cleanup(sessions_dir: Path, result: CleanupResult, *, actor: str) -> None:
    """Record a removing pass in :data:`LAST_CLEANUP_NAME` for the TUI to announce."""
    policies: dict[str, int] = {}
    for candidate in result.removed:
        policies[candidate.policy] = policies.get(candidate.policy, 0) + 1
    payload = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "actor": actor,
        "pid": os.getpid(),
        "removed": len(result.removed),
        "scanned": result.scanned,
        "policies": policies,
        "record": str(sessions_dir / CLEANUP_LOG_NAME),
        "acknowledged": False,
    }
    try:
        (sessions_dir / LAST_CLEANUP_NAME).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("session cleanup: cannot write %s: %s", LAST_CLEANUP_NAME, exc)


def take_unannounced_cleanup(sessions_dir: Path) -> dict[str, Any] | None:
    """The last removing pass if no viewer has announced it yet, marking it
    announced; ``None`` otherwise.

    Read-then-rewrite without a lock: two viewers adopting the same store in
    the same instant could both announce, which is the harmless direction.
    A malformed or unreadable file is treated as "nothing to announce" — the
    jsonl and the WARNING still hold the facts.
    """
    path = sessions_dir / LAST_CLEANUP_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("acknowledged") or not payload.get("removed"):
        return None
    payload["acknowledged"] = True
    try:
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    return payload


def format_cleanup_notice(payload: dict[str, Any]) -> str:
    """The one-line transcript notice for a removing pass."""
    removed = int(payload.get("removed") or 0)
    policies = payload.get("policies") or {}
    by = ", ".join(f"{count} by {name}" for name, count in sorted(policies.items()))
    noun = "session" if removed == 1 else "sessions"
    record = str(payload.get("record") or CLEANUP_LOG_NAME)
    home = os.path.expanduser("~")
    if record.startswith(home + os.sep):
        record = "~" + record[len(home) :]
    return (
        f"session cleanup removed {removed} {noun} at launch"
        + (f" ({by})" if by else "")
        + f" — record: {record}"
        + " · preview next time: lop sessions cleanup --dry-run"
        + " · turn off: /settings › Session cleanup"
    )
