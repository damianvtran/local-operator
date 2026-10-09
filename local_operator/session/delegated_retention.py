"""Retention for DELEGATED sessions: the second class of ``session.cleanup``.

WHY A SECOND CLASS
==================

``session.cleanup.*`` was one global policy, off by default, that counted every
directory under ``sessions/`` the same. On the operator's real store
(2026-10-09: 19,397 sessions, 71.2 GB) 90% of the bytes were subagent
transcripts and 17,579 hidden-origin sessions had been idle for more than 48 hours
— machine bookkeeping nobody opens, which the one policy could only remove by
being switched on for the user's own conversations too. So the store is split
by ONE predicate, ``resume.is_user_session_origin``:

* PARENT class — everything the sidebar lists (no ``origin.json``, ``fork``,
  ``agent-workstream``). Governed by the existing five keys, OFF by default.
  ``cleanup.run_cleanup`` now iterates ONLY this class.
* DELEGATED class — every other origin (``subagent``, ``agent-shell``,
  ``agent-config`` and any future value, because the predicate is opt-OUT of
  visibility). Governed by ``session.cleanup.delegated.*``, ON by default, with a
  bounded age (2..720 hours). This module is that class's whole policy.

``cleanup.remove_session_dir`` remains the ONLY ``rmtree`` of a session
directory (``tests/unit/session/test_no_session_deletion.py``); this module
decides, it never deletes.

THE SAFETY RULES (every one fails CLOSED: cannot tell => keep)
==============================================================

A delegated session is kept when ANY of these holds:

1. It is not older than the window. Age is hours since the last ACTIVITY
   (``retention.session_activity``: transcript or spooled mail, never a sidecar).
   A directory that was never active (no transcript) is clocked by
   ``created_at.json``, then the directory's mtime; no clock => keep.
2. A hard guard holds (``cleanup._guard``): live claim or lease, armed wake, armed
   monitor, unread spooled mail, open ask. A child with running descendants is
   covered by the claim: children run IN their parent's process, which holds
   ``.session.pid`` on every directory it owns, so a roster ``running`` flag is
   deliberately not trusted on its own (it goes stale when a process dies).
3. PARENT STILL ACTIVE. Parents reach children through ``hub peek/resume``, which
   read the child's directory as recorded in the PARENT's
   ``subagent-roster.v1.json`` (``records[].session_dir``); ``origin.json`` has no
   parent link for legacy children. A delegated session is protected when ANY
   session listing it in its roster is live OR has activity inside the window
   (so a child outlives its newest parent by at most ``max_age_hours``). An idle
   parent older than the window protects nothing. The set is built only from
   rosters of sessions that are live or recent — never all of them (the real store
   has ~800, some 6 MB) — and followed transitively through protected delegated
   sessions. An unreadable roster of such a parent makes the set unknowable, so
   the WHOLE pass is skipped (logged), not guessed. Accepted trade-off: a
   resumed, ancient parent that asks for a reaped child gets the existing "gone"
   outcome (``comms.py`` already models gone children).
4. An OPEN PROJECT lists it (``<config>/projects/<id>.json``, ``sessions``, status
   not ``done``/``archived``). Children of a project-linked parent are NOT
   additionally pinned — only rule 3 applies to them. An unreadable project file
   skips the pass for the same reason as an unreadable roster.
5. Its ``scratchpad/`` holds a git repository (file or directory ``.git``, depth
   <= 4) that is dirty, has commits on no remote, a stash, or cannot be
   inspected in 5 s; or the search exceeds its entry cap. Removing the session
   removes the scratchpad, so work that exists nowhere else is not removed.
6. It is the current session, or it moved after the scan (re-stat at removal).

There is NO PR/MR registry in the harness, so a session whose work is only
referenced from a pull request cannot be protected on that basis. That is a
limit of what the harness knows, not a promise this module makes.

DRAINING A BACKLOG WITHOUT A SPIKE
==================================

Defaulting ON means the first run on a long-lived store faces thousands of
removals (17.6k here). A pass therefore removes in BATCHES with a yield after
every removal and a wall-clock budget; a pass that runs out records how many
candidates it did not reach and the caller (the store-maintenance thread in
``session_factory``) comes back after a short pause, re-reading the policy, so
turning the switch off stops the drain at the next batch. Once nothing remains
the same thread wakes hourly: a 48-hour window needs steady-state sweeps, and a
startup-only pass never runs again in a long-lived runtime.

THE ONE-TIME NOTICE
===================

``sessions/.delegated-retention.json`` counts what has been removed and records
whether the user was told. The first pass that removes anything sets
``first_removal_at``; ONE viewer announces it and flips ``notice_acknowledged``;
every later removal is silent (the jsonl log keeps the record). It is a file
separate from ``last-cleanup.json`` on purpose: that record is re-armed by
every removing pass of the PARENT class, and reusing it would announce each
hourly delegated sweep.

Import-light on purpose (stdlib + ``retention`` + ``cleanup``): it runs on the
store-maintenance thread of every runtime, and ``resume`` / ``wakes`` / ``asks``
load lazily inside the functions that need them.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    MAX_DELEGATED_MAX_AGE_HOURS,
    MIN_DELEGATED_MAX_AGE_HOURS,
    Candidate,
    CleanupPolicy,
    CleanupResult,
    _claimed,
    _forget_monitor_entry,
    _forget_wake_entry,
    _guard,
    _is_delegated_dir,
    _lease_runtime_alive,
    _write_record,
    policy_from_config,
    remove_session_dir,
)
from local_operator.session.retention import (
    SESSIONS_DIRNAME,
    TRANSCRIPT_FILENAME,
    _process_alive,
    session_activity_path,
)

logger = logging.getLogger(__name__)

#: The ``policy`` string recorded in ``.cleanup-log.jsonl`` for these removals.
DELEGATED_POLICY = "delegated_max_age"

#: The notice/progress record. See the module docstring for why it is its own file.
STATE_NAME = ".delegated-retention.json"

#: Sessions removed between two checks of (stop, budget, live policy). 100 keeps
#: the check rate around one per second at the measured removal speed while
#: bounding how many more removals a "turn it off" can be late by.
BATCH_SIZE = 100

#: Wall-clock budget of ONE pass, scan included. 30 s lets a pass make real
#: progress on a backlog yet hand the thread back well inside the time a
#: departing runtime waits on its own drain.
PASS_BUDGET_S = 30.0

#: Yield after every removal so a drain of thousands never saturates the disk the
#: user's own session is writing its transcript to.
REMOVAL_PAUSE_S = 0.01

#: Pause between passes while a backlog remains, and the steady-state period.
DRAIN_RESUME_S = 5.0
STEADY_SWEEP_S = 3600.0
#: Re-try interval when another process holds the sweep lock.
LOCK_BUSY_RETRY_S = 60.0

#: Scratchpad search bounds (rule 5). Depth counts directories below
#: ``scratchpad/``; the entry cap bounds a pathological tree, and exceeding it
#: KEEPS the session. 20,000 entries is ~10x a large legitimate scratch area.
SCRATCH_MAX_DEPTH = 4
SCRATCH_MAX_ENTRIES = 20_000
SCRATCH_MAX_REPOS = 8
SCRATCH_SKIP_DIRS = frozenset({"node_modules", ".venv", ".tox", "site-packages", "__pycache__"})
GIT_TIMEOUT_S = 5.0

#: Size estimates stop walking a directory after this many entries; the figure
#: is then a lower bound. Sizes are an after-the-fact estimate for the notice
#: record, never an input to a decision.
SIZE_MAX_ENTRIES = 2000

#: How many in-window delegated names a result carries (for the dry run's kept list).
WINDOW_NAMES_CAP = 500

ROSTER_NAME = "subagent-roster.v1.json"
_PROJECT_CLOSED = frozenset({"done", "archived"})


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class DelegatedResult(CleanupResult):
    """:class:`CleanupResult` plus what a draining caller needs."""

    #: Hidden-origin directories seen (any age).
    delegated_total: int = 0
    #: Hidden-origin directories inside the age window — kept without a row.
    within_window: int = 0
    #: Over-age candidates the pass did not get to (budget, stop or the switch
    #: turned off); the caller schedules another pass while this is > 0.
    remaining: int = 0
    budget_exhausted: bool = False
    hours: int = 0
    #: Bytes removed, a LOWER BOUND (see :data:`SIZE_MAX_ENTRIES`).
    freed_bytes: int = 0
    #: Seconds the scan alone took (reported by the CLI and the timing evidence).
    scan_seconds: float = 0.0
    #: Directories skipped as non-sessions or unreadable.
    unreadable: int = 0
    #: Delegated sessions kept for being inside the window (capped list).
    window_names: list[str] = field(default_factory=list)


class _Default:
    """Type of :data:`UNBOUNDED`."""


#: Pass ``budget_s=UNBOUNDED`` for no wall-clock budget (``None`` means "the default").
UNBOUNDED = _Default()


@dataclass
class _Cand:
    path: Path
    clock: float
    has_transcript: bool


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def _clock_without_activity(directory: Path) -> float | None:
    """The age clock of a directory that was never worked in, or ``None``.

    ``created_at.json`` first (the birth stamp the transcript constructor writes —
    a bare float), then its mtime, then the directory's. The directory's mtime is
    a LATER time than the birth (it moves with every entry added), which only ever
    makes the session look younger: the safe side. ``None`` means no clock could
    be established and the caller keeps the directory.
    """
    marker = directory / "created_at.json"
    try:
        raw = marker.read_text(encoding="utf-8").strip()
        parsed: Any = json.loads(raw)
        if isinstance(parsed, Mapping):
            parsed = parsed.get("created_at", parsed.get("at"))
        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) and parsed > 0:
            return float(parsed)
    except (OSError, ValueError):
        pass
    for candidate in (marker, directory):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return None


#: THE class predicate lives in ``cleanup`` (the parent-class scan needs it to
#: EXCLUDE these directories and ``cleanup`` cannot import this module); re-exported
#: so both classes provably share one classifier.
is_delegated_dir = _is_delegated_dir


def bounded_dir_bytes(directory: Path, max_entries: int = SIZE_MAX_ENTRIES) -> int:
    """Bytes under ``directory`` counting at most ``max_entries`` entries.

    A lower bound once capped. Deliberately not ``cleanup._dir_bytes`` (an
    unbounded ``rglob`` that also follows into ``node_modules``): this runs only
    for the sessions about to be removed, so its cost is bounded per removal.
    """
    total = 0
    seen = 0
    stack = [os.fspath(directory)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    seen += 1
                    if seen > max_entries:
                        return total
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


# ---------------------------------------------------------------------------
# Rule 5: scratchpad git repositories
# ---------------------------------------------------------------------------


def _git_env() -> dict[str, str]:
    """``env -u XPC_FLAGS`` + no system config + no prompts + no index writes.

    ``XPC_FLAGS=0x2`` breaks name resolution for children of a launchd-spawned
    tool on this host; ``GIT_OPTIONAL_LOCKS=0`` keeps ``git status`` from
    refreshing the index of a repository we are inspecting, not changing.
    """
    env = {k: v for k, v in os.environ.items() if k != "XPC_FLAGS"}
    env.update(
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0",
    )
    return env


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        env=_git_env(),
        check=False,
    )


def _repo_has_unsaved_work(repo: str) -> bool:
    """True unless ``repo`` is provably clean and fully on a remote. Raises nothing."""
    try:
        status = _git(repo, "status", "--porcelain")
        if status.returncode != 0 or status.stdout.strip():
            return True
        unpushed = _git(repo, "rev-list", "-n1", "--branches", "--not", "--remotes")
        if unpushed.returncode != 0 or unpushed.stdout.strip():
            return True
        if _git(repo, "symbolic-ref", "-q", "HEAD").returncode != 0:
            # Detached HEAD: commits made there are on no branch, so ``--branches``
            # above cannot see them.
            head = _git(repo, "rev-list", "-n1", "HEAD", "--not", "--remotes", "--branches")
            if head.returncode != 0 or head.stdout.strip():
                return True
        if _git(repo, "rev-parse", "-q", "--verify", "refs/stash").returncode == 0:
            return True
        return False
    except (OSError, subprocess.SubprocessError, ValueError):
        # git missing, timed out, or any other failure: cannot prove clean.
        return True


def scratchpad_git_hazard(session_dir: Path) -> str | None:
    """The first scratchpad repo that blocks removal (a description), else ``None``.

    Bounded: directories below ``scratchpad/`` to depth :data:`SCRATCH_MAX_DEPTH`,
    never following symlinks, never entering :data:`SCRATCH_SKIP_DIRS` or a
    ``.git`` directory, at most :data:`SCRATCH_MAX_ENTRIES` entries and
    :data:`SCRATCH_MAX_REPOS` repositories. Exceeding either cap returns a
    description too: the answer to "could I not look?" is KEEP.
    """
    root = os.path.join(os.fspath(session_dir), "scratchpad")
    try:
        if not os.path.isdir(root) or os.path.islink(root):
            return None
    except OSError:
        return "scratchpad cannot be inspected"
    visited = 0
    repos: list[str] = []
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as entries:
                children = list(entries)
        except OSError:
            return f"scratchpad directory {current} cannot be read"
        names = {entry.name for entry in children}
        visited += len(children)
        if visited > SCRATCH_MAX_ENTRIES:
            return f"scratchpad has more than {SCRATCH_MAX_ENTRIES} entries to search"
        if ".git" in names:
            repos.append(current)
            if len(repos) > SCRATCH_MAX_REPOS:
                return f"scratchpad holds more than {SCRATCH_MAX_REPOS} git repositories"
        if depth >= SCRATCH_MAX_DEPTH:
            continue
        for entry in children:
            try:
                if (
                    entry.is_dir(follow_symlinks=False)
                    and entry.name != ".git"
                    and entry.name not in SCRATCH_SKIP_DIRS
                ):
                    stack.append((entry.path, depth + 1))
            except OSError:
                return f"scratchpad entry {entry.path} cannot be inspected"
    for repo in repos:
        if _repo_has_unsaved_work(repo):
            return f"scratchpad repo {repo} has uncommitted/unpushed work"
    return None


# ---------------------------------------------------------------------------
# Rules 3 and 4: parents (rosters) and open projects
# ---------------------------------------------------------------------------


class _Unknowable(Exception):
    """A protection source that cannot be read: the pass must not guess."""


#: ``path -> ((mtime_ns, size), payload)``. Bounded; keyed on the file's stat so a
#: rewritten roster is re-read and an unchanged one costs a stat across passes.
_FILE_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}
_FILE_CACHE_MAX = 1024


def _cached_json(path: Path) -> Any:
    """Parsed JSON of ``path`` cached on ``(mtime_ns, size)``; raises :class:`_Unknowable`."""
    key = os.fspath(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise _Unknowable(f"{path.name}: {exc.strerror or exc}") from exc
    stamp = (info.st_mtime_ns, info.st_size)
    hit = _FILE_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        payload = _read_json(path)
    except (OSError, ValueError) as exc:
        raise _Unknowable(f"{path.parent.name}/{path.name}: {exc}") from exc
    if len(_FILE_CACHE) >= _FILE_CACHE_MAX:
        _FILE_CACHE.clear()
    _FILE_CACHE[key] = (stamp, payload)
    return payload


def roster_children(roster: Path) -> set[str]:
    """Session ids a parent's roster records as its children.

    Raises :class:`_Unknowable` for a roster that exists but cannot be read or is
    not an object — the caller must not treat that as "no children".
    """
    payload = _cached_json(roster)
    if not isinstance(payload, dict):
        raise _Unknowable(f"{roster.parent.name}/{ROSTER_NAME}: not an object")
    records = payload.get("records")
    if records is None:
        return set()
    if not isinstance(records, list):
        raise _Unknowable(f"{roster.parent.name}/{ROSTER_NAME}: records is not a list")
    names: set[str] = set()
    for record in records:
        if isinstance(record, dict):
            directory = record.get("session_dir")
            if isinstance(directory, str) and directory:
                names.add(os.path.basename(directory.rstrip("/\\")))
    return names


def open_project_sessions(config_dir: Path) -> set[str]:
    """Session ids listed by any project whose status is not done/archived.

    A project row missing ``status`` counts as OPEN (cannot prove it closed).
    Raises :class:`_Unknowable` for a project file that exists and cannot be read.
    """
    projects = config_dir / "projects"
    try:
        entries = [e for e in os.scandir(projects) if e.name.endswith(".json") and e.is_file()]
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise _Unknowable(f"projects/: {exc.strerror or exc}") from exc
    protected: set[str] = set()
    for entry in entries:
        payload = _cached_json(Path(entry.path))
        if not isinstance(payload, dict):
            raise _Unknowable(f"projects/{entry.name}: not an object")
        status = payload.get("status")
        if isinstance(status, str) and status.strip().lower() in _PROJECT_CLOSED:
            continue
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            protected.update(item for item in sessions if isinstance(item, str) and item)
    return protected


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def _is_live(directory: Path, now: float) -> bool:
    """A live claim or lease on ``directory`` (closed: unreadable => live)."""
    try:
        if _claimed(directory, now):
            return True
        return bool(_lease_runtime_alive(directory))
    except Exception:  # noqa: BLE001 — cannot tell: live
        return True


@dataclass
class _Scan:
    candidates: list[_Cand] = field(default_factory=list)
    #: Sessions (any class) that are live or inside the window: the parents whose
    #: rosters protect children.
    sources: dict[str, Path] = field(default_factory=dict)
    #: Hidden-origin directory names, for the transitive step.
    hidden: set[str] = field(default_factory=set)
    delegated_total: int = 0
    within_window: int = 0
    unreadable: int = 0
    stopped: bool = False
    #: Directories examined (every class), reported as ``scanned``.
    total: int = 0
    #: Names of delegated sessions kept for being inside the window, capped: the
    #: dry run names them (an honest "kept" list) without holding 17k strings.
    window_names: list[str] = field(default_factory=list)


def _scan(
    sessions_dir: Path,
    cutoff: float,
    now: float,
    should_stop: Callable[[], bool] | None,
    deadline: float | None = None,
) -> _Scan:
    """One pass over the store: no sizes, no rglob, a handful of syscalls per directory.

    Per directory: two stats for the activity clock; for an OLD one, one small read
    of ``origin.json``; for every directory one stat of the roster (only parents
    have one); liveness is probed only for an OLD directory that has a roster.
    """
    scan = _Scan()
    try:
        entries = [e for e in os.scandir(sessions_dir) if not e.name.startswith(".")]
    except OSError as exc:
        logger.warning("session cleanup: cannot scan %s: %s", sessions_dir, exc)
        scan.unreadable += 1
        return scan
    for index, entry in enumerate(entries):
        if index % 256 == 0 and (
            (should_stop is not None and should_stop())
            or (deadline is not None and time.monotonic() > deadline)
        ):
            scan.stopped = True
            return scan
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            scan.unreadable += 1
            continue
        scan.total += 1
        path = Path(entry.path)
        activity = session_activity_path(entry.path)
        has_roster = os.path.exists(os.path.join(entry.path, ROSTER_NAME))
        if activity is not None and activity >= cutoff:
            # Inside the window: a potential parent whatever its class, and for
            # a hidden one a session kept without a row. The origin is read only
            # to count it.
            scan.sources[entry.name] = path
            try:
                if is_delegated_dir(path):
                    scan.hidden.add(entry.name)
                    scan.delegated_total += 1
                    scan.within_window += 1
                    if len(scan.window_names) < WINDOW_NAMES_CAP:
                        scan.window_names.append(entry.name)
            except Exception:  # noqa: BLE001
                scan.unreadable += 1
            continue
        try:
            hidden = is_delegated_dir(path)
        except Exception:  # noqa: BLE001 — cannot classify: not ours
            scan.unreadable += 1
            continue
        if has_roster and _is_live(path, now):
            scan.sources[entry.name] = path
        if not hidden:
            continue
        scan.hidden.add(entry.name)
        scan.delegated_total += 1
        clock = activity if activity is not None else _clock_without_activity(path)
        if clock is None or clock >= cutoff:
            # Rule 1's last clause: no clock, no removal.
            scan.within_window += 1
            if len(scan.window_names) < WINDOW_NAMES_CAP:
                scan.window_names.append(entry.name)
            continue
        has_transcript = False
        try:
            has_transcript = os.stat(os.path.join(entry.path, TRANSCRIPT_FILENAME)).st_size > 0
        except OSError:
            pass
        scan.candidates.append(_Cand(path, clock, has_transcript))
    scan.candidates.sort(key=lambda cand: (cand.clock, cand.path.name))
    return scan


def _protected_children(scan: _Scan) -> set[str]:
    """Rule 3: every session named in the roster of a live/recent session, transitively.

    Transitive through protected DELEGATED sessions: a kept child's own children
    are reachable by resuming it, so they are kept too. Rosters are read only for
    sources and for newly protected hidden sessions.
    """
    protected: set[str] = set()
    queue = list(scan.sources.items())
    seen = {name for name, _ in queue}
    while queue:
        name, path = queue.pop()
        roster = path / ROSTER_NAME
        if not os.path.exists(roster):
            continue
        for child in roster_children(roster):
            protected.add(child)
            if child in scan.hidden and child not in seen:
                seen.add(child)
                queue.append((child, path.parent / child))
    return protected


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

_WARNED_KEEP: set[tuple[str, str]] = set()


def _evaluate(
    cand: _Cand,
    *,
    config_dir: Path,
    now: float,
    cutoff: float,
    parents: set[str],
    projects: set[str],
    live_resolved: Path | None,
) -> str | None:
    """Why ``cand`` must be kept (a short phrase), or ``None`` if it may go."""
    name = cand.path.name
    if name in parents:
        return "parent session still active"
    if name in projects:
        return "linked to an open project"
    if live_resolved is not None:
        try:
            if cand.path.resolve() == live_resolved:
                return "the current session"
        except OSError:
            return "cannot resolve path"
    guard = _guard(cand.path, config_dir, now)
    if guard is not None:
        return guard
    hazard = scratchpad_git_hazard(cand.path)
    if hazard is not None:
        if (name, hazard) not in _WARNED_KEEP:
            _WARNED_KEEP.add((name, hazard))
            logger.warning("session cleanup: keeping %s: %s", name, hazard)
        return "scratchpad git repo has uncommitted/unpushed work"
    fresh = session_activity_path(os.fspath(cand.path))
    if fresh is not None and fresh >= cutoff:
        return "active since the scan"
    return None


def run_delegated_pass(
    config_dir: Path,
    policy_provider: Callable[[], CleanupPolicy],
    *,
    live_dir: Path | None = None,
    now: float | None = None,
    dry_run: bool = False,
    force: bool = False,
    actor: str = "startup",
    batch_size: int | None = None,
    budget_s: "float | None | _Default" = None,
    removal_pause_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    only: Iterable[str] | None = None,
) -> DelegatedResult:
    """One scan-and-remove pass over the DELEGATED class.

    ``policy_provider`` is called before every batch, so a switch turned off (or a
    window widened) takes effect at the next batch rather than at the next pass.
    A dry run evaluates EVERY candidate (so its kept list is honest) and removes
    nothing; a real run evaluates lazily, batch by batch, until the budget.

    ``only`` restricts the candidates to those names: the CLI previews, asks, then
    applies with the SAME ``now`` and ``only=<previewed names>``, so a confirmed
    removal can never include a session the user was not shown (the parent class's
    ``apply_cleanup`` invariant). Every rule is still re-evaluated at removal.
    """
    # Resolved at CALL time so the module constants are the tuning surface (a
    # default bound at import would make them inert). ``budget_s=UNBOUNDED`` is
    # the CLI's: a person who typed the command waits for all of it.
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    removal_pause_s = REMOVAL_PAUSE_S if removal_pause_s is None else removal_pause_s
    budget: float | None
    if budget_s is None:
        budget = PASS_BUDGET_S
    elif isinstance(budget_s, _Default):
        budget = None
    else:
        budget = budget_s
    result = DelegatedResult(dry_run=dry_run)
    sessions_dir = config_dir / SESSIONS_DIRNAME
    policy = policy_provider()
    result.hours = max(
        MIN_DELEGATED_MAX_AGE_HOURS,
        min(MAX_DELEGATED_MAX_AGE_HOURS, policy.delegated_max_age_hours),
    )
    if not policy.delegated_enabled and not force and not dry_run:
        result.skipped = "disabled"
        return result
    if not policy.delegated_enabled:
        result.skipped = "disabled"  # a dry run still lists; the caller says so
    if not sessions_dir.is_dir():
        result.skipped = "no store"
        return result

    started = time.monotonic()
    deadline = None if budget is None or dry_run else started + budget
    moment = now if now is not None else time.time()
    cutoff = moment - result.hours * 3600.0
    live_resolved: Path | None = None
    if live_dir is not None:
        try:
            live_resolved = live_dir.resolve()
        except OSError:
            live_resolved = None

    # The SCAN IS NOT BUDGETED, only stoppable. A pass that gave up mid-scan would
    # leave nothing recorded and no backlog to resume, i.e. a drain that never
    # continues; the scan is O(directories) in a handful of stats each (measured in
    # the PR), so the budget is spent where the cost is: removals.
    scan = _scan(sessions_dir, cutoff, moment, should_stop, None)
    result.scan_seconds = time.monotonic() - started
    result.scanned = scan.total
    result.window_names = scan.window_names
    result.delegated_total = scan.delegated_total
    result.within_window = scan.within_window
    result.unreadable = scan.unreadable
    if scan.stopped:
        result.skipped = "stopped during the scan"
        return result
    try:
        parents = _protected_children(scan)
        projects = open_project_sessions(config_dir)
    except _Unknowable as exc:
        logger.warning("session cleanup: delegated pass skipped, cannot read %s", exc)
        result.skipped = f"skipped: cannot read {exc}"
        result.errors += 1
        return result

    todo = scan.candidates
    if only is not None:
        wanted = set(only)
        todo = [cand for cand in todo if cand.path.name in wanted]
    position = 0
    freed = 0
    recorded = 0
    in_batch = 0
    while position < len(todo):
        if should_stop is not None and should_stop():
            break
        if not dry_run and in_batch >= batch_size:
            # A BATCH IS ``batch_size`` REMOVALS, not ``batch_size`` candidates
            # examined. Candidates are oldest-first and the oldest are exactly the
            # ones that tend to be kept for good (a dirty scratchpad repo, a parent
            # that stays active), so counting examined rows let a handful of
            # permanent keeps at the head of the queue consume every batch and the
            # drain removed nothing, pass after pass (seen in the drain evidence).
            # Counting removals also gives the progress guarantee: a pass that has
            # a removable candidate removes at least ``batch_size`` of them before
            # the budget can end it, however small the budget.
            if len(result.removed) > recorded:
                _record_progress(
                    sessions_dir,
                    removed=result.removed,
                    hours=result.hours,
                    remaining=max(0, len(todo) - position),
                    actor=actor,
                    already=recorded,
                )
                recorded = len(result.removed)
            if deadline is not None and time.monotonic() > deadline:
                result.budget_exhausted = True
                break
            # Between batches: the live policy. Disabled stops the drain here.
            live = policy_provider()
            if not live.delegated_enabled and not force:
                result.skipped = "disabled during the pass"
                break
            hours = max(
                MIN_DELEGATED_MAX_AGE_HOURS,
                min(MAX_DELEGATED_MAX_AGE_HOURS, live.delegated_max_age_hours),
            )
            if hours > result.hours:
                # A WIDER window only ever protects more: tighten the cutoff.
                result.hours = hours
                cutoff = moment - hours * 3600.0
            in_batch = 0
        cand = todo[position]
        position += 1
        if cand.clock >= cutoff:
            result.protected.append((cand.path.name, "inside the (widened) window"))
            continue
        keep = _evaluate(
            cand,
            config_dir=config_dir,
            now=moment,
            cutoff=cutoff,
            parents=parents,
            projects=projects,
            live_resolved=live_resolved,
        )
        if keep is not None:
            result.protected.append((cand.path.name, keep))
            continue
        idle_hours = max(0.0, (moment - cand.clock) / 3600.0)
        # Sized here, for the rows about to be shown or removed only: a bounded
        # scandir walk, never the scan's cost (see bounded_dir_bytes).
        size = bounded_dir_bytes(cand.path)
        row = Candidate(
            cand.path.name,
            DELEGATED_POLICY,
            f"idle over {result.hours}h",
            title=_title(cand),
            idle_days=idle_hours / 24.0,
            size_bytes=size,
            origin=_origin(cand.path),
            active=cand.has_transcript,
        )
        result.chosen.append(row)
        if dry_run:
            result.removed.append(row)
            continue
        try:
            done = remove_session_dir(
                cand.path,
                config_dir=config_dir,
                policy=DELEGATED_POLICY,
                reason=row.reason,
                actor=actor,
                title=row.title,
            )
        except OSError as exc:
            logger.warning("session cleanup: cannot remove %s: %s", cand.path.name, exc)
            result.errors += 1
            continue
        if done:
            freed += size
            in_batch += 1
            result.removed.append(row)
            _forget_wake_entry(config_dir, cand.path.name)
            _forget_monitor_entry(config_dir, cand.path.name)
            if removal_pause_s > 0:
                time.sleep(removal_pause_s)
    if not dry_run and len(result.removed) > recorded:
        _record_progress(
            sessions_dir,
            removed=result.removed,
            hours=result.hours,
            remaining=max(0, len(todo) - position),
            actor=actor,
            already=recorded,
        )
    result.remaining = max(0, len(todo) - position)
    if dry_run:
        # The honest kept list names the sessions the age rule itself saved, last
        # (capped at WINDOW_NAMES_CAP; the totals line carries the full count).
        result.protected.extend(
            (name, f"active within the last {result.hours}h") for name in scan.window_names
        )
    result.freed_bytes = freed
    if not dry_run and result.removed:
        logger.warning(
            "session cleanup: removed %d delegated sessions older than %dh (%d left to consider; "
            "see %s)",
            len(result.removed),
            result.hours,
            result.remaining,
            sessions_dir / CLEANUP_LOG_NAME,
        )
    return result


def _title(cand: _Cand) -> str:
    if not cand.has_transcript:
        return ""
    try:
        from local_operator.session.cleanup import _session_title

        return _session_title(cand.path)
    except Exception:  # noqa: BLE001 — a name is a courtesy
        return ""


def _origin(path: Path) -> str:
    try:
        from local_operator.resume import session_origin

        return session_origin(path) or "user"
    except Exception:  # noqa: BLE001
        return "subagent"


# ---------------------------------------------------------------------------
# Progress record and the one-time notice
# ---------------------------------------------------------------------------


def read_state(sessions_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((sessions_dir / STATE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_progress(
    sessions_dir: Path,
    *,
    removed: list[Candidate],
    hours: int,
    remaining: int,
    actor: str,
    already: int,
) -> None:
    """Fold this pass's removals into the progress record (batch-granular).

    ``removed`` is the pass's cumulative list; ``already`` is how many of them an
    earlier batch of THIS pass already folded in, so each removal is counted once.
    """
    fresh = len(removed) - already
    if fresh <= 0:
        return
    state = read_state(sessions_dir)
    state.setdefault("first_removal_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    state.setdefault("first_removal_pid", os.getpid())
    state.setdefault("max_age_hours", hours)
    state.setdefault("notice_acknowledged", False)
    total = state.get("removed_total")
    state["removed_total"] = (total if isinstance(total, int) else 0) + fresh
    prior = state.get("freed_bytes_estimate")
    batch_freed = sum(c.size_bytes for c in removed[already:])
    state["freed_bytes_estimate"] = (prior if isinstance(prior, int) else 0) + batch_freed
    # A lower bound (see SIZE_MAX_ENTRIES): the notice never quotes it as a figure.
    state["drain_remaining"] = remaining
    state["last_removal_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["actor"] = actor
    _write_record(sessions_dir / STATE_NAME, state)


def take_unannounced_delegated_notice(
    sessions_dir: Path, *, runtime_pid: int | None = None, defer_to_writer: bool = True
) -> dict[str, Any] | None:
    """The progress record if no viewer has announced it yet, marking it announced.

    Same viewer rule as ``cleanup.take_unannounced_cleanup`` (the runtime that
    removed defers to its own viewer while it lives) with one switch for a
    viewer that has no runtime of its own, the desktop server:
    ``defer_to_writer=False`` takes it at once. ONE announcement per store, ever;
    a malformed or absent record announces nothing.
    """
    path = sessions_dir / STATE_NAME
    state = read_state(sessions_dir)
    removed = state.get("removed_total")
    if (
        not state
        or state.get("notice_acknowledged")
        or not isinstance(removed, int)
        or isinstance(removed, bool)
        or removed <= 0
    ):
        return None
    writer = state.get("first_removal_pid")
    if (
        defer_to_writer
        and isinstance(writer, int)
        and not isinstance(writer, bool)
        and writer != runtime_pid
        and writer != os.getpid()
        and _process_alive(writer)
    ):
        return None
    announced = dict(state)
    state["notice_acknowledged"] = True
    if not _write_record(path, state):
        logger.debug("session cleanup: %s is not writable; the notice will repeat", path)
    return announced


#: How the notice names the removal record: home-relative, as the parent
#: class's notice does (``cleanup.format_cleanup_notice``).
_RECORD_DISPLAY = "~/.local-operator/sessions/" + CLEANUP_LOG_NAME


def format_delegated_notice(payload: Any) -> str:
    """The one plain sentence group for the one-time announcement. Total on any shape."""
    if not isinstance(payload, dict):
        payload = {}
    try:
        count = int(payload.get("removed_total") or 0)
    except (TypeError, ValueError):
        count = 0
    try:
        hours = int(payload.get("max_age_hours") or 48)
    except (TypeError, ValueError):
        hours = 48
    remaining = payload.get("drain_remaining")
    so_far = " so far" if isinstance(remaining, int) and remaining > 0 else ""
    noun = "session" if count == 1 else "sessions"
    record = _RECORD_DISPLAY
    return "\n".join(
        (
            f"Cleaned up {count:,} delegated {noun}{so_far} (subagents and background runs) "
            f"older than {hours} hours to save disk space.",
            "Your own conversations were not touched.",
            "Change or turn this off in Settings > Delegated work.",
            f"Record: {record}",
        )
    )


def notice_wire(payload: dict[str, Any]) -> dict[str, Any]:
    """The desktop wire form of the one-time notice (additive on ``GET /v1/desktop/sessions``).

    ``message`` is the finished sentence group (the same text the terminal shows),
    so a client can render it verbatim; the numbers ride beside it for a client
    that wants its own layout. ``in_progress`` is the "so far" flag.
    """
    remaining = payload.get("drain_remaining")
    return {
        "message": format_delegated_notice(payload),
        "removed": payload.get("removed_total"),
        "max_age_hours": payload.get("max_age_hours"),
        "in_progress": isinstance(remaining, int)
        and not isinstance(remaining, bool)
        and remaining > 0,
        "first_removal_at": payload.get("first_removal_at"),
        "freed_bytes_estimate": payload.get("freed_bytes_estimate"),
        "record": _RECORD_DISPLAY,
    }


# ---------------------------------------------------------------------------
# Live policy and the sweep loop
# ---------------------------------------------------------------------------

#: Cross-process mutex for a REAL pass, beside the store (not inside it: the store
#: directory is listed as sessions). Two runtimes launched together would otherwise
#: both scan and both drain, double-counting the progress record.
SWEEP_LOCK_NAME = ".delegated-retention.lock"

#: A completed, backlog-free sweep younger than this makes a launching runtime skip
#: its first pass: 25 concurrent sessions each scanning a 20k-directory store at
#: launch is the cost this stamp exists to remove. Well under STEADY_SWEEP_S so
#: the hourly cadence still holds when launches are rare.
FRESH_SWEEP_S = 900.0


def _acquire_sweep_lock(config_dir: Path) -> int | None:
    from local_operator.procstate import O_BINARY
    from local_operator.wakes.lock import _try_lock

    fd = os.open(config_dir / SWEEP_LOCK_NAME, os.O_CREAT | os.O_RDWR | O_BINARY, 0o600)
    try:
        if _try_lock(fd):
            return fd
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise


def _release_sweep_lock(fd: int) -> None:
    from local_operator.wakes.lock import _unlock

    try:
        _unlock(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _stamp_sweep(sessions_dir: Path, remaining: int) -> None:
    """Record that a real pass finished, and how much backlog it left."""
    state = read_state(sessions_dir)
    state["last_sweep_at"] = time.time()
    state["drain_remaining"] = remaining
    _write_record(sessions_dir / STATE_NAME, state)


def _sweep_is_fresh(sessions_dir: Path) -> bool:
    state = read_state(sessions_dir)
    stamp = state.get("last_sweep_at")
    remaining = state.get("drain_remaining")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return False
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining != 0:
        return False
    return 0 <= time.time() - float(stamp) < FRESH_SWEEP_S


def run_sweeps(
    config_dir: Path,
    policy_provider: Callable[[], CleanupPolicy],
    *,
    live_dir: Path | None,
    should_stop: Callable[[], bool],
    wait: Callable[[float], bool],
    actor: str = "startup",
) -> str:
    """Pass, wait, pass again — until stopped or the switch is off. Returns why it ended.

    ``wait(seconds)`` must return True when the caller is stopping (it is
    ``threading.Event.wait``). The loop ends when the delegated switch is off at
    a sweep boundary — the keys are Scope.NEW_LAUNCH, so switching it back on
    takes effect at the next launch, as the settings page says — and a pass that is
    already running notices the switch at its next batch. A drain that is not
    finished comes back after :data:`DRAIN_RESUME_S`; a finished one after
    :data:`STEADY_SWEEP_S`.
    """
    sessions_dir = config_dir / SESSIONS_DIRNAME
    while not should_stop():
        if not policy_provider().delegated_enabled:
            return "disabled"
        delay = STEADY_SWEEP_S
        try:
            fd = _acquire_sweep_lock(config_dir)
        except OSError:
            logger.debug("delegated retention: sweep lock unavailable", exc_info=True)
            return "lock unavailable"
        if fd is None:
            delay = LOCK_BUSY_RETRY_S
        else:
            try:
                if not _sweep_is_fresh(sessions_dir):
                    result = run_delegated_pass(
                        config_dir,
                        policy_provider,
                        live_dir=live_dir,
                        actor=actor,
                        should_stop=should_stop,
                    )
                    if result.skipped is None or result.skipped == "disabled during the pass":
                        _stamp_sweep(sessions_dir, result.remaining)
                    if result.remaining or result.budget_exhausted:
                        delay = DRAIN_RESUME_S
                    elif result.skipped and result.skipped.startswith("skipped:"):
                        delay = LOCK_BUSY_RETRY_S * 10
            except Exception:  # noqa: BLE001 — housekeeping never fails a session
                logger.warning("session cleanup: delegated sweep failed", exc_info=True)
            finally:
                _release_sweep_lock(fd)
        if wait(delay):
            return "stopped"
    return "stopped"


class _SnapshotConfig:
    """``get_nested_value`` over the config watcher's last good ``values``."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = values

    def get_nested_value(self, path: Iterable[str], default: Any = None) -> Any:
        current: Any = self._values
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


def live_policy_provider(config_manager: Any, config_dir: Path) -> Callable[[], CleanupPolicy]:
    """A provider that re-reads the policy on every call, without a destructive reload.

    ``ConfigManager`` snapshots ``config.yml`` at construction and its
    ``_load_config`` can move a malformed file aside, so re-instantiating one on a
    background thread is exactly the hazard ``config_watch`` documents. The
    process watcher already holds the last GOOD parse and follows the file, so it is
    the source when the process has one; the passed manager is the fallback.
    """

    def provider() -> CleanupPolicy:
        try:
            from local_operator.config_watch import existing_watcher

            watcher = existing_watcher(config_dir)
            if watcher is not None:
                return policy_from_config(_SnapshotConfig(watcher.values))
        except Exception:  # noqa: BLE001 — fall back to the manager
            logger.debug("delegated retention: no live config snapshot", exc_info=True)
        return policy_from_config(config_manager)

    return provider
