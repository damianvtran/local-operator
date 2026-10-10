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
decides, and the CONTENT layer's removals run through ``cleanup``'s guarded
remover (``remove_scratchpad_entry``) — no deletion lives outside that module.

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

THE CONTENT LAYER (the carve-out)
=================================

The record pass above decides whether a delegated SESSION may go. This module
also reclaims the CONTENT of delegated sessions the record pass KEEPS. The
authority is the disk desk's rule, operator/Aida-confirmed
(``~/tools/disk-hygiene/docs/tick-role.md``, "Reclaimable class — stale
session scratchpad trees"):

    A stored session's scratchpad ``tree``, ``worktree`` or ``node_modules``
    copy is reclaimable when the owning PR/branch is MERGED, the session is
    more than 24 hours old, and it is not live.

    A tree is a candidate only when merged AND clean AND ownerless AND stale
    — a dirty tree is not a candidate even when merged.

Rules 3-5 keep the RECORD of a session whose work may still matter (an active
parent, an open project, a scratchpad repo with uncommitted/unpushed work);
they were never about bytes nobody can reach. So, for a delegated session
past the class window whose record survived the record pass, the scratchpad
is walked — bounded, symlinks never followed, ``.git`` never matched by shape
— and reclaimed by three arms:

* MERGED-CLEAN GIT TREES. A directory that is a git tree (its ``.git``
  resolves) is reclaimed WHOLE only when its HEAD resolves, its status is
  clean, and its HEAD is an ancestor of the remote trunk (``origin/HEAD``,
  else ``origin/main``, else ``origin/master``; no trunk keeps). Where the
  tree's objects live decides HOW it may go:

  - ``.git`` FILE naming a linked worktree (``<shared>/.git/worktrees/<name>``):
    the objects belong to the SHARED repository, so a bundle would duplicate
    that store into the pad (measured: the operator's lo-before tree, 133 MB,
    bundled 205 shared refs into 264 MB — the pad would GROW). After the
    merged+clean checks the tree is removed with ``git -C <shared> worktree
    remove <tree>`` WITHOUT ``--force`` (git's own refusal is the last word),
    and only once ``git worktree list`` of the shared repository LISTS this
    exact path — the belt that keeps a COPY of a worktree (whose ``.git``
    still points at the original) from acting on a registration that is not
    its own. No bundle; the registration is pruned by the removal itself. A
    pointer that is malformed, dangling, names a submodule
    (``.git/modules/``), or a tree the shared repository does not list: KEPT.
  - ``.git`` DIR (an independent clone): callers continue to the uniqueness
    logic — no commit of HEAD may be absent from every remote, and the
    sidebar refs (``refs/heads/*`` and ``refs/stash``) are checked against
    the same remotes: when every tip is reachable the tree goes as-is, and
    when any is not, its refnames are rescued into a bundle beside the tree
    (``reap-rescue-<tree>.bundle``: ``git bundle create`` + ``git bundle
    verify``) BEFORE the tree may go — any bundle failure keeps the tree.
  - a ``.git`` SYMLINK: kept (the content layer never follows links).

  The "objects also exist in the main repo" shortcut is NOT automated: there
  is no generic way to locate that repo on an arbitrary machine, and a wrong
  guess would authorise a lossy delete, so the bundle arm is the general safe
  path and the per-machine determination stays a human one. Dirty, unmerged,
  no-remote and unresolvable trees are KEPT WHOLE and nothing inside them is
  touched.
* BUILD SHAPES. A directory whose name matches the ``scratchpad`` write
  policy's refused SEGMENT vocabulary, or a file whose name matches its
  refused SUFFIX vocabulary, is removed. The same arms on the same sides the
  write refusal uses — parents by segment, leaves by suffix — so the pass can
  only ever take what a write could not have put there, ``.git`` is never
  taken by shape, and ONE documented exemption outranks the vocabulary: a
  path with a pad-relative segment whose case-folded name starts with
  ``evidence`` is never taken by this arm (measured: a real
  ``docs/evidence/session-load-central-cache/`` holding scripts and bench
  results matched the ``-cache`` segment suffix). An exempted entry is kept
  WHOLE as a directory and the walk still descends into it — rules (a) and
  (c) carry no evidence exemption, so a big stale log or a merged-clean tree
  inside evidence is still judged by its own arm. The vocabulary is imported
  LAZILY from ``local_operator.scratchpad`` so this module keeps its import
  weight.
* STALE OUTPUT. A file with suffix in :data:`STALE_OUTPUT_SUFFIXES` of at
  least :data:`STALE_OUTPUT_MIN_BYTES`, untouched for
  :data:`STALE_OUTPUT_GRACE_S`, is removed (the desk's measured 122 MB
  ``desktop-suite.log`` family).

HARD GUARDS: nothing is content-reclaimed for the current session, a live
claim or lease, an armed wake or monitor, a session inside the window, a
non-delegated origin — and any guard evaluation that fails keeps the pad and
counts an error. Parent-still-active, open projects and the git hazard are
NOT content guards: they keep the record, and that carve-out is the point.
Unread spooled mail and open asks are not content guards either: neither is a
running process, the record stays (reopening still finds the conversation),
and the merged+clean+stale class is exactly what is safe to take from a
session that may later be reopened. The pad root itself is never removed,
only its subdirs where the rules took their contents; a still-failing removal
keeps the entry and counts an error (fail closed). A rescue bundle is the one
thing the pass WRITES back into a pad — it lives until the record is reaped,
which is the sanctioned path for it (the record guards are what kept the
session at all). The phase re-evaluates the kept pads every pass, like the
record pass re-evaluates its keeps: a pad with nothing to take costs one
bounded walk, a pad that took something is smaller next time, and the pass
budget with the batch gate bounds the tail.

RECONCILIATION WITH THE DISK LADDER (two scratchpad policies, on purpose)
========================================================================

Two policies trim scratchpad content on this machine, and they are
deliberate halves rather than drift:

* the disk ladder's runtime-hygiene prune (``~/tools/disk-hygiene``, e.g.
  ``src/stored_scratchpad_prune.py``) runs with the standing scope
  ``--min-mb 300 --min-idle-days 0.5`` — machine-local, operator-approved,
  tuned for THIS host's disk pressure, and it removes blob-level candidates
  with no knowledge of what the session was for;
* THIS module's content layer is the shipped, default-on retention policy:
  window = the class's ``max_age_hours``, and it takes only merged-clean git
  trees, build shapes and stale output, with the record guards left standing.

The thresholds differ because the audiences do: the desk automates a
MACHINE-LOCAL disk emergency and can be aggressive under its own approval,
while this policy ships to every machine and must be safe by default — hence
the much narrower vocabulary (merged+clean+fully-pushed only, a 12-hour
grace, ``.git`` only ever via a whole-tree rule).

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
every later removal is silent (the jsonl log keeps the record). The TUI
consumes it on show (``take_unannounced_delegated_notice``); the desktop
renderer acknowledges it explicitly (``acknowledge_delegated_notice``), because
its list route is polled by many readers — the app's own attach and auth probes
among them — so a read must only PEEK (``peek_unannounced_delegated_notice``;
the UI PR's round-1 review). It is a file
separate from ``last-cleanup.json`` on purpose: that record is re-armed by
every removing pass of the PARENT class, and reusing it would announce each
hourly delegated sweep.

Import-light on purpose (stdlib + ``retention`` + ``cleanup``): it runs on the
store-maintenance thread of every runtime, and ``resume`` / ``wakes`` / ``asks``
/ ``scratchpad`` load lazily inside the functions that need them.
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
    _append_cleanup_log,
    _claimed,
    _forget_monitor_entry,
    _forget_wake_entry,
    _guard,
    _has_armed_monitor,
    _has_armed_wake,
    _is_delegated_dir,
    _lease_runtime_alive,
    _write_record,
    policy_from_config,
    remove_scratchpad_entry,
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

#: The ``policy`` string for CONTENT reclaims (the carve-out below): one row
#: per PAD in the same log, so an auditor can tell a removed session from a
#: kept session whose scratchpad released its build output.
DELEGATED_SCRATCHPAD_POLICY = "delegated_scratchpad"

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

#: Content-phase walk bounds, mirroring the scratchpad search's style
#: (``SCRATCH_MAX_DEPTH``/``_ENTRIES``): a pad that cannot be looked at
#: COMPLETELY is kept by the caller, so these caps fail closed. Depth counts
#: directories below ``scratchpad/``; 8 clears every measured shape (a clone's
#: ``tree`` at 2, a QA repo under ``r3/`` at 2, a pytest tmpdir chain) with
#: margin, and the entry/repo caps bound a pathological tree.
CONTENT_MAX_DEPTH = 8
CONTENT_MAX_ENTRIES = 20_000
CONTENT_MAX_REPOS = 8

#: STALE OUTPUT (the desk's measured 122 MB ``desktop-suite.log`` family):
#: these suffixes, at least this size, untouched for this long, are removed.
STALE_OUTPUT_SUFFIXES: tuple[str, ...] = (
    ".log",
    ".cpuprofile",
    ".heapsnapshot",
    ".trace",
    ".prof",
    ".dmp",
)
STALE_OUTPUT_MIN_BYTES = 10 * 1024 * 1024
STALE_OUTPUT_GRACE_S = 12 * 3600.0

#: The three class phrases a content row's reason is built from, in the fixed
#: order they are displayed and counted (see :func:`_PadPlan.classes`).
CONTENT_CLASS_MERGED = "merged-clean clone"
CONTENT_CLASS_BUILD = "build shapes"
CONTENT_CLASS_STALE = "stale output"

#: A rescue bundle's name, beside the tree it preserves refs of:
#: ``<pad>/reap-rescue-<tree>.bundle``.
RESCUE_BUNDLE_PREFIX = "reap-rescue-"
RESCUE_BUNDLE_SUFFIX = ".bundle"

#: Bound for the content layer's HEAVY git operations — a rescue bundle's
#: create and verify, and ``git worktree remove``. Measured on this host: a
#: full dev clone's 207 unique refs bundled in 14.9 s (266 MB) — orders of
#: magnitude over :data:`GIT_TIMEOUT_S`, whose 5 s budget is sized for
#: ``status``/``rev-parse`` probes; the first bound tried (5 s) killed a create
#: mid-write and left a stale lock that blocked every retry. 300 s clears the
#: measured cost with margin under fleet load; a fired bound keeps the tree
#: (fail closed) and the next pass retries from a cleaned slate.
HEAVY_GIT_TIMEOUT_S = 300.0

#: How many unique commits the rescue check will list before it gives up and
#: KEEPS the tree (fail closed). One bounded ``rev-list`` walk answers every
#: tip's uniqueness at once; a ``merge-base --is-ancestor`` loop would be one
#: process per (tip, remote) pair — measured ~0.5 s each, so ~7 minutes for a
#: full dev clone's 885 x 213. 20,000 is ~30x the largest measured case
#: (lo-before's 634).
CONTENT_MAX_UNIQUE_COMMITS = 20_000

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
    #: Pads whose scratchpad content this pass reclaimed; in a dry run,
    #: "would reclaim" rows. ONE row per pad (never per file).
    content: list[ContentRow] = field(default_factory=list)
    #: ``(session, reason)`` for every content candidate the pass DECLINED —
    #: unmerged/dirty/no-remote trees, unreadable subtrees, failed rescue
    #: bundles — so a dry run audits what was left, not only what was taken.
    content_kept: list[tuple[str, str]] = field(default_factory=list)
    #: Pads the content phase did not reach (budget or stop); > 0 schedules
    #: the drain's next pass, like :attr:`remaining` does for the record pass.
    content_remaining: int = 0
    content_budget_exhausted: bool = False
    #: Why the content phase did not run, when it did not ("disabled").
    content_skipped: str | None = None


class _Default:
    """Type of :data:`UNBOUNDED`."""


#: Pass ``budget_s=UNBOUNDED`` for no wall-clock budget (``None`` means "the default").
UNBOUNDED = _Default()


@dataclass
class _Cand:
    path: Path
    clock: float
    has_transcript: bool


@dataclass
class ContentRow:
    """One pad's content reclaim. In a dry run, a "would reclaim" row.

    Fields mirror :class:`cleanup.Candidate` where the meanings are the same
    (``session`` / ``title`` / ``policy`` / ``reason``) and add what the
    content pass knows and record rows do not: the reclaimed byte estimate,
    the entry count, which classes were taken, and the rescue bundle, when
    unique refs had to be preserved beside the pad.
    """

    session: str
    policy: str = DELEGATED_SCRATCHPAD_POLICY
    reason: str = ""
    title: str = ""
    #: A stat-based LOWER-BOUND estimate of the reclaimed bytes (see
    #: :func:`_bounded_dir_stats`), never an input to any decision.
    bytes: int = 0
    entries: int = 0
    #: Or the recorded one; every content row is a delegated origin by
    #: construction, so this is what the CLI's origin column prints.
    origin: str = "subagent"
    #: The classes taken, in the fixed display order (merged-clean clone,
    #: build shapes, stale output) — what the CLI's per-class counts read.
    classes: tuple[str, ...] = ()
    #: The rescue bundle kept beside the pad (an absolute path) and its size,
    #: when the uniqueness check found refs that live nowhere else.
    rescue_bundle: str = ""
    rescue_bundle_bytes: int = 0
    #: How the removal was carried out: ``"remove"`` (the guarded pad-level
    #: path), ``"worktree-remove"`` (a linked worktree removed through its
    #: shared repository), or ``"bundle-rescue"`` (an independent clone whose
    #: unique refs were bundled first). The cleanup-log row carries it so a
    #: reader can tell which of the three actually ran.
    method: str = "remove"


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


def _bounded_dir_stats(directory: Path, max_entries: int = SIZE_MAX_ENTRIES) -> tuple[int, int]:
    """``(bytes, entries)`` under ``directory``, counting at most ``max_entries``.

    Both figures are LOWER BOUNDS once capped. One walk serves both the size
    estimate a notice records and the entry count a cleanup-log row carries, so
    the two can never disagree about what was measured. Deliberately not
    ``cleanup._dir_bytes`` (an unbounded ``rglob`` that also follows into
    ``node_modules``): this runs per removed item, so its cost is bounded.
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
                        return total, max_entries
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total, seen


def bounded_dir_bytes(directory: Path, max_entries: int = SIZE_MAX_ENTRIES) -> int:
    """Bytes under ``directory`` counting at most ``max_entries`` entries.

    A lower bound once capped. Deliberately not ``cleanup._dir_bytes`` (an
    unbounded ``rglob`` that also follows into ``node_modules``): this runs only
    for the sessions about to be removed, so its cost is bounded per removal.
    """
    return _bounded_dir_stats(directory, max_entries)[0]


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


def _git(repo: str, *args: str, timeout: float = GIT_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
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
# The content layer: what a KEPT record's scratchpad may still release
# ---------------------------------------------------------------------------


@dataclass
class _TreeDecision:
    """Rule (c) for ONE git tree: reclaim it, or keep it (and why)."""

    reclaim: bool
    keep: str | None = None
    #: Refnames a rescue bundle must carry before the tree may go. Empty when
    #: every local ref is reachable from the remotes (no bundle needed).
    bundle_refs: list[str] = field(default_factory=list)
    #: The SHARED repository when this tree is a linked worktree (its ``.git``
    #: is a FILE naming ``<shared>/.git/worktrees/<name>``): such a tree is
    #: removed by ``git worktree remove`` in ``<shared>``, never bundled —
    #: its refs and objects ARE the shared store's, and a bundle would only
    #: duplicate the operator's whole clone into the pad.
    worktree: Path | None = None


def _read_gitdir_link(entry: Path) -> str | None:
    """The target of a ``.git`` FILE's ``gitdir:`` pointer, or ``None``.

    The pointer is resolved against the file's own directory (git writes it
    relative in some layouts) and returned as a path string; ``None`` covers
    everything malformed or unreadable, which the caller treats as keep.
    """
    try:
        content = entry.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    raw = content[len("gitdir:") :].strip()
    if not raw:
        return None
    target = Path(raw)
    if not target.is_absolute():
        target = entry.parent / target
    return os.fspath(target)


def _linked_worktree_shared(repo: Path) -> tuple[Path | None, str | None]:
    """Where a linked worktree's objects live, when ``repo`` is one.

    Returns ``(shared, None)`` for a ``.git`` FILE naming
    ``<shared>/.git/worktrees/<name>`` (the shape ``git worktree add``
    writes), ``(None, why)`` for every other ``.git`` FILE — a submodule
    pointer (``.git/modules/``), a separate-git-dir link, anything malformed
    — and ``(None, None)`` for a ``.git`` DIRECTORY (an independent clone,
    which the clone path judges). A ``.git`` SYMLINK is a keep: the content
    layer never follows links, not even to decide what a tree is.
    """
    git_entry = repo / ".git"
    if git_entry.is_symlink():
        return None, "the .git entry is a symlink"
    if not git_entry.is_file():
        return None, None
    link = _read_gitdir_link(git_entry)
    if link is None:
        return None, "the .git link is malformed"
    if "/modules/" in link.replace(os.sep, "/"):
        return None, "a submodule pointer (never touched)"
    parts = Path(link).parts
    for i in range(len(parts) - 3, -1, -1):
        if parts[i] == ".git" and parts[i + 1] == "worktrees" and len(parts) == i + 3:
            return Path(*parts[:i]), None
    return None, "the gitdir link does not name a linked worktree"


def _worktree_listed(listing: str, repo: Path) -> bool:
    """Whether ``repo`` is one of the worktree paths ``git worktree list`` shows.

    Both sides are realpath-ed so a store reached through a symlinked root
    still matches; only the porcelain format's ``worktree <path>`` lines are
    read. This is the belt that keeps a COPY of a worktree — whose ``.git``
    still names the original's gitdir — from acting on the original's
    registration: the copy's own path is not in the list.
    """
    wanted = os.path.realpath(os.fspath(repo))
    for line in listing.splitlines():
        if line.startswith("worktree "):
            candidate = line[len("worktree ") :].strip()
            if candidate and os.path.realpath(candidate) == wanted:
                return True
    return False


def _remove_linked_worktree(shared: Path, tree: Path) -> str | None:
    """``git worktree remove`` for one reclaimable linked worktree; None on success.

    WITHOUT ``--force``: git's own refusal (dirty, locked, unlisted, missing)
    is the last word and comes back as the keep-reason, exactly like a failed
    bundle. The removal also prunes the registration, so a success leaves
    nothing behind to retry.
    """
    done = _git(
        os.fspath(shared), "worktree", "remove", os.fspath(tree), timeout=HEAVY_GIT_TIMEOUT_S
    )
    if done.returncode == 0:
        return None
    lines = done.stderr.strip().splitlines()
    detail = lines[0][:160] if lines else "no output"
    return f"git worktree remove failed for {tree.name}: {detail}"


def _rescue_bundle_path(root: str, relative: str) -> Path:
    """``<pad>/reap-rescue-<tree>.bundle`` for a tree ``relative`` to the pad.

    A tree's path separators flatten to ``-`` so a nested tree gets a name of
    its own (``r3/qa-frames`` -> ``reap-rescue-r3-qa-frames.bundle``) rather
    than colliding with a top-level tree that shares its basename.
    """
    flattened = relative.replace(os.sep, "-")
    return Path(root) / f"{RESCUE_BUNDLE_PREFIX}{flattened}{RESCUE_BUNDLE_SUFFIX}"


def _tree_reclaim_decision(repo: Path) -> _TreeDecision:
    """Rule (c): may this git tree be reclaimed WHOLE, and does it need a bundle?

    The desk's test (``tick-role.md``, "Reclaimable class — stale session
    scratchpad trees") as tightened for this pass: HEAD must resolve, the
    worktree must be clean, and HEAD must be an ancestor of the remote trunk
    (``origin/HEAD`` -> ``origin/main`` -> ``origin/master``; no trunk keeps).
    The tree's ``.git`` then decides HOW it may go: a linked worktree is
    removed through its shared repository (no bundle — its refs and objects
    are the shared store's) and only when that repository LISTS this path; an
    independent clone must additionally have no commit of HEAD on no remote,
    and its sidebar refs (``refs/heads/*`` and ``refs/stash``) are checked
    against the same remotes for uniqueness — when any tip is unreachable its
    refname comes back so the caller rescues it into a bundle BEFORE the tree
    may go. Dirty, unmerged, no-remote and unresolvable trees all come back as
    keep — anything that cannot be PROVEN reclaimable is kept, and this
    function never raises (a repository it cannot inspect is a keep).
    """
    try:
        return _tree_reclaim_checks(repo)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return _TreeDecision(False, f"cannot inspect the tree ({exc.__class__.__name__})")


def _tree_reclaim_checks(repo: Path) -> _TreeDecision:
    """The decision body for :func:`_tree_reclaim_decision`; may raise on a bad repo."""
    path = os.fspath(repo)
    shared, why_not = _linked_worktree_shared(repo)
    if why_not is not None:
        return _TreeDecision(False, why_not)
    head = _git(path, "rev-parse", "-q", "--verify", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        return _TreeDecision(False, "HEAD does not resolve (or the gitdir link is broken)")
    status = _git(path, "status", "--porcelain")
    if status.returncode != 0:
        return _TreeDecision(False, "the working tree cannot be inspected")
    if status.stdout.strip():
        return _TreeDecision(False, "uncommitted changes")
    trunk: str | None = None
    for ref in (
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/main",
        "refs/remotes/origin/master",
    ):
        probe = _git(path, "rev-parse", "-q", "--verify", ref)
        if probe.returncode == 0 and probe.stdout.strip():
            trunk = ref
            break
    if trunk is None:
        return _TreeDecision(
            False, "no remote trunk (origin/HEAD, origin/main or origin/master) to compare"
        )
    ancestor = _git(path, "merge-base", "--is-ancestor", "HEAD", trunk)
    if ancestor.returncode == 1:
        return _TreeDecision(False, "HEAD is not merged into the remote trunk")
    if ancestor.returncode != 0:
        return _TreeDecision(False, "cannot decide whether HEAD is merged")
    if shared is not None:
        # A LINKED WORKTREE: its refs and objects ARE the shared repository's,
        # so there is nothing to rescue and nothing to bundle — only git's own
        # removal, gated on the shared repository LISTING this exact path (the
        # belt that keeps a COPY of a worktree, whose .git still points at the
        # original, from acting on a registration that is not its own).
        listed = _git(os.fspath(shared), "worktree", "list", "--porcelain")
        if listed.returncode != 0:
            return _TreeDecision(
                False,
                "the shared repository cannot be inspected (missing or dangling pointer)",
            )
        if not _worktree_listed(listed.stdout, repo):
            return _TreeDecision(False, "not registered in the shared repository's worktree list")
        return _TreeDecision(True, worktree=shared)
    on_remote = _git(path, "log", "HEAD", "--not", "--remotes", "--oneline")
    if on_remote.returncode != 0:
        return _TreeDecision(False, "cannot decide whether HEAD is on a remote")
    if on_remote.stdout.strip():
        return _TreeDecision(False, "HEAD carries commits that are on no remote")
    # The sidebar refs (heads + stash) against the same remotes. The per-tip
    # question — "is this tip reachable from any remote ref?" — is answered by
    # ONE ``rev-list --branches [refs/stash] --not --remotes`` walk: the output
    # is the set of commits no remote ref reaches, and a tip is unique exactly
    # when it appears there. (``--no-walk`` is NOT usable here: git documents
    # it as having no effect when a range is given, and ``--not`` IS one —
    # verified on git 2.55, the full walk came out anyway.) The walk is still
    # bounded: over :data:`CONTENT_MAX_UNIQUE_COMMITS` the list cannot be
    # proven complete, so the tree is kept (fail closed).
    stash = _git(path, "rev-parse", "-q", "--verify", "refs/stash")
    refs = ["rev-list", "--max-count", str(CONTENT_MAX_UNIQUE_COMMITS + 1), "--branches"]
    if stash.returncode == 0 and stash.stdout.strip():
        refs.append("refs/stash")
    refs += ["--not", "--remotes"]
    unique = _git(path, *refs)
    if unique.returncode != 0:
        return _TreeDecision(False, "cannot test the local refs against the remotes")
    unique_commits = unique.stdout.split()
    if len(unique_commits) > CONTENT_MAX_UNIQUE_COMMITS:
        return _TreeDecision(False, "more unique commits than the rescue check will list")
    if not unique_commits:
        return _TreeDecision(True)
    listing = _git(
        path, "for-each-ref", "--format=%(objectname) %(refname)", "refs/heads", "refs/stash"
    )
    if listing.returncode != 0:
        return _TreeDecision(False, "cannot list the local refs to rescue")
    wanted = set(unique_commits)
    names: list[str] = []
    for line in listing.stdout.splitlines():
        object_id, _, refname = line.partition(" ")
        if refname and object_id in wanted:
            names.append(refname)
    if not names:
        return _TreeDecision(False, "unique commits exist but no ref names them to rescue")
    return _TreeDecision(True, bundle_refs=names)


def _write_rescue_bundle(repo: Path, bundle_path: Path, refnames: list[str]) -> str | None:
    """Create + verify the unique-refs bundle beside ``repo``; ``None`` on success.

    The rescue is the WHOLE price of removing a tree whose local refs are not
    all reachable from its remotes: the bundle is written and verified before
    the tree is given up, and ANY failure (a non-zero ``git bundle create`` or
    ``git bundle verify``, a fired :data:`HEAVY_GIT_TIMEOUT_S`, git missing)
    returns a reason string the caller logs and keeps the tree for. A failed
    attempt is CLEANED UP first — its partial ``.bundle`` and git's
    ``<bundle>.lock`` — so the next pass starts from nothing rather than
    meeting ``File exists`` on a lock a killed create left behind (measured: a
    5 s bound firing mid-create left exactly that, and every later attempt
    failed on it; this bundle target has exactly one writer — the sweep, under
    its cross-process lock, or a single CLI run — so clearing its stale lock
    is self-healing, never a race). A dry run never calls this — it reports
    the planned bundle from :class:`_PadPlan` without writing.
    """
    lock = Path(os.fspath(bundle_path) + ".lock")

    def _clean_up() -> None:
        for leftover in (bundle_path, lock):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:  # pragma: no cover — best effort
                pass

    try:
        lock.unlink(missing_ok=True)
        create = _git(
            os.fspath(repo),
            "bundle",
            "create",
            os.fspath(bundle_path),
            *refnames,
            timeout=HEAVY_GIT_TIMEOUT_S,
        )
        if create.returncode != 0:
            lines = create.stderr.strip().splitlines()
            detail = lines[0][:160] if lines else "no output"
            _clean_up()
            return f"git bundle create failed for {repo.name}: {detail}"
        verify = _git(
            os.fspath(repo),
            "bundle",
            "verify",
            os.fspath(bundle_path),
            timeout=HEAVY_GIT_TIMEOUT_S,
        )
        if verify.returncode != 0:
            _clean_up()
            return f"git bundle verify failed for {repo.name}"
        return None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        _clean_up()
        return f"cannot write the rescue bundle for {repo.name} ({exc.__class__.__name__})"


class _PadAbort(Exception):
    """The pad could not be looked at completely: nothing is taken from it."""


@dataclass
class _PlannedRemoval:
    """One entry a pad plan will take, with its stat-based estimate."""

    path: Path
    cls: str
    bytes: int
    entries: int
    #: How the removal is carried out: ``"remove"`` (the guarded pad-level
    #: path) or ``"worktree-remove"`` (``git -C <shared_repo> worktree
    #: remove``, for a linked worktree whose objects are the shared store's).
    method: str = "remove"
    #: The shared repository for ``method == "worktree-remove"``.
    shared_repo: Path | None = None


@dataclass
class _PadPlan:
    """What the content pass will take from ONE pad, decided but not yet done."""

    removals: list[_PlannedRemoval] = field(default_factory=list)
    #: A rescue bundle to write before the tree it belongs to may go: the tree,
    #: the target bundle path, and the refnames that live only in that tree.
    bundle_tree: Path | None = None
    bundle_path: Path | None = None
    bundle_refs: list[str] = field(default_factory=list)
    #: Why specific candidates were left — an unmerged tree, a broken gitdir
    #: link, an aborted cap. The caller records these so the dry run shows what
    #: it declined, not only what it took.
    kept: list[str] = field(default_factory=list)

    def classes(self) -> tuple[str, ...]:
        """The distinct classes taken, in the fixed display order."""
        taken = {removal.cls for removal in self.removals}
        return tuple(
            name
            for name in (CONTENT_CLASS_MERGED, CONTENT_CLASS_BUILD, CONTENT_CLASS_STALE)
            if name in taken
        )


def _has_evidence_segment(relative: str) -> bool:
    """Whether any pad-relative segment starts with ``evidence`` (case-folded).

    ONE documented exemption to the build-shape arm: an entry under — or
    named by — a segment like ``evidence/`` is never shape-matched, because
    evidence outranks the shape vocabulary. Measured on a real pad whose
    ``docs/evidence/session-load-central-cache/`` held scripts and bench
    results and matched the ``-cache`` segment suffix. Rule (a), stale
    output, is unaffected.
    """
    return any(part.lower().startswith("evidence") for part in Path(relative).parts)


def _plan_pad_content(pad: Path, *, now: float) -> _PadPlan | None:
    """Walk one pad (bounded, symlinks never followed) and plan its reclaim.

    ``None`` when the session has no ``scratchpad/`` directory (or that root
    is a symlink — never followed); otherwise a plan whose ``removals`` may be
    empty. Raises :class:`_PadAbort` for a pad that cannot be looked at
    COMPLETELY (an unreadable subtree, an exceeded cap): the caller keeps the
    pad and counts the error, because "could not look" is keep, not guess.

    The rules and their order are the module docstring's "THE CONTENT LAYER":
    a directory that is a git tree is judged by rule (c) — reclaimed or kept
    WHOLE, never descended into for the other rules; other directories
    matching the write policy's refused SEGMENT vocabulary are taken whole;
    files are judged on the refused SUFFIX vocabulary and on the stale-output
    arm. ``.git`` entries are never matched by the shape arm (the vocabulary
    lists them; the carve-out skips them), and a pad ROOT that is itself a git
    tree is kept whole — the root is never a removal candidate, and deleting
    selected content under a repository this pass may not judge would dirty
    it.
    """
    from local_operator.scratchpad import (  # LAZY: the write path's vocabulary, one copy
        _is_refused_segment,
        _refused_suffix,
    )

    root = os.path.join(os.fspath(pad), "scratchpad")
    if not os.path.isdir(root) or os.path.islink(root):
        return None
    plan = _PadPlan()
    if os.path.lexists(os.path.join(root, ".git")):
        # The PAD ROOT is itself a git tree. It is never a removal candidate
        # (only entries below it are), and deleting selected content under a
        # repository this pass may not judge would dirty it — keep it whole.
        plan.kept.append("the scratchpad root is itself a git tree")
        return plan
    if os.path.lexists(os.path.join(os.fspath(pad), ".git")):
        # The SESSION directory is a repository whose worktree includes the
        # pad: same reasoning, one level up.
        plan.kept.append("the session directory is itself a git tree")
        return plan
    visited = 0
    repos = 0
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise _PadAbort(f"scratchpad directory {current} cannot be read") from exc
        visited += len(children)
        if visited > CONTENT_MAX_ENTRIES:
            raise _PadAbort(f"scratchpad has more than {CONTENT_MAX_ENTRIES} entries to search")
        for entry in children:
            name = entry.name
            try:
                if entry.is_symlink():
                    continue  # never followed, never shape-matched
                if name == ".git":
                    # Rule (b) never deletes a .git; rule (c) owns every tree,
                    # at the directory that holds this entry.
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if os.path.lexists(os.path.join(entry.path, ".git")):
                        repos += 1
                        if repos > CONTENT_MAX_REPOS:
                            raise _PadAbort(
                                "scratchpad holds more than "
                                f"{CONTENT_MAX_REPOS} git repositories to inspect"
                            )
                        _plan_tree(entry.path, root, plan)
                        continue
                    if _is_refused_segment(name):
                        relative = os.path.relpath(entry.path, root)
                        if _has_evidence_segment(relative):
                            # Kept as a directory — and the walk still DESCENDS
                            # into it, because rules (a) and (c) carry no
                            # evidence exemption: a big stale log or a
                            # merged-clean tree inside evidence is still
                            # judged by its own arm.
                            plan.kept.append(
                                f"{relative}: evidence path (never taken by the shape arm)"
                            )
                        else:
                            stats = _bounded_dir_stats(Path(entry.path))
                            plan.removals.append(
                                _PlannedRemoval(Path(entry.path), CONTENT_CLASS_BUILD, *stats)
                            )
                            continue
                    if depth >= CONTENT_MAX_DEPTH:
                        continue  # beyond the cap: left in place (the keep side)
                    stack.append((entry.path, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue  # fifos, sockets: never touched
                if _refused_suffix(name) is not None:
                    relative = os.path.relpath(entry.path, root)
                    if _has_evidence_segment(relative):
                        plan.kept.append(
                            f"{relative}: evidence path (never taken by the shape arm)"
                        )
                        continue
                    info = entry.stat(follow_symlinks=False)
                    plan.removals.append(
                        _PlannedRemoval(Path(entry.path), CONTENT_CLASS_BUILD, info.st_size, 1)
                    )
                    continue
                if name.lower().endswith(STALE_OUTPUT_SUFFIXES):
                    info = entry.stat(follow_symlinks=False)
                    if (
                        info.st_size >= STALE_OUTPUT_MIN_BYTES
                        and now - info.st_mtime >= STALE_OUTPUT_GRACE_S
                    ):
                        plan.removals.append(
                            _PlannedRemoval(Path(entry.path), CONTENT_CLASS_STALE, info.st_size, 1)
                        )
            except OSError as exc:
                raise _PadAbort(f"scratchpad entry {entry.path} cannot be inspected") from exc
    return plan


def _plan_tree(path: str, root: str, plan: _PadPlan) -> None:
    """Rule (c) for one git tree at ``path``: plan it, or record why it stays."""
    relative = os.path.relpath(path, root)
    decision = _tree_reclaim_decision(Path(path))
    if not decision.reclaim:
        plan.kept.append(f"{relative}: {decision.keep}")
        return
    stats = _bounded_dir_stats(Path(path))
    plan.removals.append(
        _PlannedRemoval(
            Path(path),
            CONTENT_CLASS_MERGED,
            *stats,
            method="worktree-remove" if decision.worktree is not None else "remove",
            shared_repo=decision.worktree,
        )
    )
    if decision.bundle_refs:
        plan.bundle_tree = Path(path)
        plan.bundle_path = _rescue_bundle_path(root, relative)
        plan.bundle_refs = decision.bundle_refs


@dataclass
class _ContentOutcome:
    """What one pad's content evaluation decided, and what it cost."""

    row: ContentRow | None = None
    kept: list[str] = field(default_factory=list)
    #: Removal attempts that raised (the entry is kept; counted like a record
    #: pass error).
    errors: int = 0
    #: Refusals from the remover's own guards (kept; not counted as errors,
    #: exactly as the record pass treats a refused removal).
    refused: int = 0


def _content_guard(
    cand: _Cand, *, config_dir: Path, now: float, cutoff: float, live_resolved: Path | None
) -> str | None:
    """The content phase's OWN guards: why this pad must not be touched at all.

    Narrower than the record pass's :func:`_evaluate` BY DESIGN — the carve-out
    is the point: parent-still-active, open projects and the scratchpad git
    hazard keep the RECORD; only the current session, a live claim or lease, an
    armed wake or monitor, the (possibly widened) window and the class
    predicate refuse CONTENT. Spooled mail and open asks keep the record but
    not the content: neither is a running process, and the desk's class
    (merged+clean+stale) is exactly what is safe to take from a session that
    may later be reopened. The caller wraps this call: a probe that raises
    keeps the pad and counts an error (fail closed).
    """
    if cand.clock >= cutoff:
        return "inside the (widened) window"
    if live_resolved is not None:
        try:
            if cand.path.resolve() == live_resolved:
                return "the current session"
        except OSError:
            return "cannot resolve path"
    if _claimed(cand.path, now):
        return "claimed by a live process"
    if _lease_runtime_alive(cand.path):
        return "leased by a live process"
    if _has_armed_wake(config_dir, cand.path.name):
        return "armed wake for this session"
    if _has_armed_monitor(config_dir, cand.path.name):
        return "armed monitor for this session"
    if not _is_delegated_dir(cand.path):
        return "not a delegated session"
    return None


def _reclaim_pad_content(
    cand: _Cand,
    *,
    config_dir: Path,
    now: float,
    cutoff: float,
    live_resolved: Path | None,
    sessions_dir: Path,
    dry_run: bool,
) -> _ContentOutcome:
    """Evaluate — and in a real pass execute — the content reclaim for ONE pad.

    The hard guards run first; a refusal comes back as ``kept`` and nothing
    below is touched. Then the pad is planned (bounded walk, no writes); then,
    in a real pass, each planned removal goes through
    ``cleanup.remove_scratchpad_entry`` (the one guarded removal shape) —
    except a linked worktree's, which goes through ``git worktree remove`` in
    its shared repository (its objects are not the pad's to bundle). A rescue
    bundle is written and verified BEFORE the tree it belongs to is removed;
    any bundle failure keeps THAT tree (its planned removal is dropped from
    the plan) and counts an error, while the pad's other planned removals
    still proceed. A refusal or failure of the git removal keeps that tree and
    counts an error the same way.
    """
    outcome = _ContentOutcome()
    guard = _content_guard(
        cand, config_dir=config_dir, now=now, cutoff=cutoff, live_resolved=live_resolved
    )
    if guard is not None:
        outcome.kept.append(guard)
        return outcome
    try:
        plan = _plan_pad_content(cand.path, now=now)
    except OSError as exc:  # the walk could not complete: keep, count it
        raise _PadAbort(f"scratchpad cannot be inspected ({exc.__class__.__name__})") from exc
    if plan is None:
        return outcome
    outcome.kept.extend(plan.kept)
    if not plan.removals:
        return outcome
    bundle_path = ""
    bundle_bytes = 0
    if plan.bundle_tree is not None and plan.bundle_path is not None:
        if dry_run:
            bundle_path = os.fspath(plan.bundle_path)
        else:
            failure = _write_rescue_bundle(plan.bundle_tree, plan.bundle_path, plan.bundle_refs)
            if failure is not None:
                outcome.kept.append(f"{os.path.basename(os.fspath(plan.bundle_tree))}: {failure}")
                outcome.errors += 1
                logger.warning("session cleanup: %s; keeping the tree", failure)
                plan.removals = [
                    removal for removal in plan.removals if removal.path != plan.bundle_tree
                ]
            else:
                bundle_path = os.fspath(plan.bundle_path)
                try:
                    bundle_bytes = os.stat(plan.bundle_path).st_size
                except OSError:  # pragma: no cover — it was just written
                    bundle_bytes = 0
    if not plan.removals:
        return outcome
    taken: list[_PlannedRemoval] = []
    for removal in plan.removals:
        if dry_run:
            taken.append(removal)
            continue
        if removal.method == "worktree-remove" and removal.shared_repo is not None:
            # A LINKED WORKTREE: git removes it, and the pad-level remover is
            # for entries that are the pad's own — this one is the shared
            # repository's to take down, registration included.
            try:
                failure = _remove_linked_worktree(removal.shared_repo, removal.path)
            except (OSError, subprocess.SubprocessError) as exc:
                failure = f"cannot run git worktree remove ({exc.__class__.__name__})"
            if failure is not None:
                outcome.errors += 1
                outcome.kept.append(f"{os.path.basename(os.fspath(removal.path))}: {failure}")
                logger.warning("session cleanup: %s; keeping the tree", failure)
                continue
            taken.append(removal)
            continue
        try:
            removed = remove_scratchpad_entry(
                removal.path, config_dir=config_dir, sessions_dir=sessions_dir
            )
        except OSError as exc:
            outcome.errors += 1
            outcome.kept.append(
                f"{os.path.relpath(removal.path, os.fspath(cand.path))}: cannot be removed"
            )
            logger.warning("session cleanup: cannot remove %s: %s", removal.path, exc)
            continue
        if removed:
            taken.append(removal)
        else:
            outcome.refused += 1
    if not taken:
        return outcome
    classes = tuple(
        cls
        for cls in (CONTENT_CLASS_MERGED, CONTENT_CLASS_BUILD, CONTENT_CLASS_STALE)
        if any(removal.cls == cls for removal in taken)
    )
    outcome.row = ContentRow(
        session=cand.path.name,
        reason=" + ".join(f"{name} past the window" for name in classes),
        title=_title(cand),
        bytes=sum(removal.bytes for removal in taken),
        entries=sum(removal.entries for removal in taken),
        origin=_origin(cand.path),
        classes=classes,
        method=_taken_method(taken, bundle_path),
        rescue_bundle=bundle_path,
        rescue_bundle_bytes=bundle_bytes,
    )
    return outcome


def _taken_method(taken: list[_PlannedRemoval], bundle_path: str) -> str:
    """The executed removal method for a pad's row (see :class:`ContentRow`)."""
    if any(removal.method == "worktree-remove" for removal in taken):
        return "worktree-remove"
    if bundle_path:
        return "bundle-rescue"
    return "remove"


def _content_log_record(row: ContentRow, *, actor: str) -> dict[str, Any]:
    """The ONE ``.cleanup-log.jsonl`` row a pad's reclaim leaves.

    Same core keys as a record removal's row (``at``/``session``/``title``/
    ``policy``/``reason``/``actor``/``pid``) so one reader handles both; the
    content-specific keys are additive, and ``rescue_bundle`` is present only
    when a bundle was actually kept beside the pad.
    """
    record: dict[str, Any] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session": row.session,
        "title": row.title,
        "policy": row.policy,
        "reason": row.reason,
        "actor": actor,
        "pid": os.getpid(),
        "bytes": row.bytes,
        "entries": row.entries,
        "classes": list(row.classes),
        "method": row.method,
    }
    if row.rescue_bundle:
        record["rescue_bundle"] = row.rescue_bundle
        record["rescue_bundle_bytes"] = row.rescue_bundle_bytes
    return record


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

    ``only`` restricts the RECORD candidates to those names: the CLI previews,
    asks, then applies with the SAME ``now`` and ``only=<previewed names>``, so
    a confirmed removal can never include a session the user was not shown (the
    parent class's ``apply_cleanup`` invariant). Every rule is still
    re-evaluated at removal. The CONTENT phase is NOT confined by ``only``:
    its rows are exactly what the preview showed for the sessions that stay,
    and confining it to the removed names would silently drop the reclaims the
    operator was shown.
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
    refused = 0
    while position < len(todo):
        if should_stop is not None and should_stop():
            break
        if deadline is not None and time.monotonic() > deadline and refused >= batch_size:
            # THE ONE WALK THE BATCH GATE CANNOT BOUND, bounded (review F2). A
            # batch boundary is ``batch_size`` SUCCESSFUL removals, so a pass
            # whose removals keep failing — a foreign/read-only store, persistent
            # EACCES — never reached one, and walked the whole candidate list
            # with the deadline never consulted. A full batch of REFUSED attempts
            # is the same evidence in the failure direction: once it exists the
            # deadline can end the pass, reporting the backlog for the drain to
            # come back to. (A walk of KEPT candidates deliberately stays
            # governed by the batch gate alone: ending every pass at the same
            # head-of-queue keeps would stall the drain, the failure the
            # removal-counting batch exists to prevent.)
            result.budget_exhausted = True
            break
        if not dry_run and in_batch >= batch_size:
            # A BATCH IS ``batch_size`` REMOVALS, not ``batch_size`` candidates
            # examined. Candidates are oldest-first and the oldest are exactly the
            # ones that tend to be kept for good (a dirty scratchpad repo, a parent
            # that stays active), so counting examined rows let a handful of
            # permanent keeps at the head of the queue consume every batch and the
            # drain removed nothing, pass after pass (seen in the drain evidence).
            # Counting removals also gives the progress guarantee: a pass that
            # can complete a batch of removals does so before the budget can end
            # it, however small the budget (the top-of-loop refusal window is the
            # only earlier exit, review F2).
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
            refused += 1
            continue
        if done:
            freed += size
            in_batch += 1
            result.removed.append(row)
            _forget_wake_entry(config_dir, cand.path.name)
            _forget_monitor_entry(config_dir, cand.path.name)
            if removal_pause_s > 0:
                time.sleep(removal_pause_s)
        else:
            # A refusal (unmarked/foreign store, symlink, EACCES class; the
            # remover logged its reason): evidence toward the top-of-loop window.
            refused += 1
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

    # ------------------------------------------------------------------
    # The CONTENT layer (the carve-out; see the module docstring): the pads of
    # the sessions the record pass did NOT remove, in the same sweep, under the
    # same budget and policy cadence, with the same batching shape — a batch is
    # ``batch_size`` reclaimed pads, the F2 window is a batch of failed ones,
    # and a walk of kept pads stays governed by the batch gate alone.
    # ------------------------------------------------------------------
    content_runs = force or (policy.delegated_enabled and policy.delegated_scratchpad_enabled)
    if content_runs:
        removed_names = {row.session for row in result.removed}
        content_pool = [cand for cand in scan.candidates if cand.path.name not in removed_names]
        content_position = 0
        content_failed = 0
        content_in_batch = 0
        content_recorded = 0
        while content_position < len(content_pool):
            if should_stop is not None and should_stop():
                break
            if (
                deadline is not None
                and time.monotonic() > deadline
                and content_failed >= batch_size
            ):
                # The record loop's F2 window, mirrored: a pass whose content
                # removals keep failing must not walk the whole pool.
                result.content_budget_exhausted = True
                break
            if not dry_run and content_in_batch >= batch_size:
                if deadline is not None and time.monotonic() > deadline:
                    result.content_budget_exhausted = True
                    break
                live = policy_provider()
                if not (force or (live.delegated_enabled and live.delegated_scratchpad_enabled)):
                    result.content_skipped = "disabled during the pass"
                    break
                content_in_batch = 0
            cand = content_pool[content_position]
            content_position += 1
            name = cand.path.name
            try:
                outcome = _reclaim_pad_content(
                    cand,
                    config_dir=config_dir,
                    now=moment,
                    cutoff=cutoff,
                    live_resolved=live_resolved,
                    sessions_dir=sessions_dir,
                    dry_run=dry_run,
                )
            except _PadAbort as abort:
                result.errors += 1
                result.content_kept.append((name, str(abort)))
                continue
            except Exception:  # noqa: BLE001 — fail closed; count it
                result.errors += 1
                result.content_kept.append((name, "cannot evaluate the scratchpad guards"))
                continue
            result.errors += outcome.errors
            content_failed += outcome.errors + outcome.refused
            result.content_kept.extend((name, reason) for reason in outcome.kept)
            if outcome.row is None:
                continue
            result.content.append(outcome.row)
            if dry_run:
                continue
            content_in_batch += 1
            _append_cleanup_log(sessions_dir, _content_log_record(outcome.row, actor=actor))
            if len(result.content) > content_recorded:
                _record_content_progress(
                    sessions_dir, rows=result.content, already=content_recorded
                )
                content_recorded = len(result.content)
            logger.warning(
                "session cleanup: reclaimed scratchpad content in %s (%s; %s)",
                name,
                outcome.row.reason,
                sessions_dir / CLEANUP_LOG_NAME,
            )
            if removal_pause_s > 0:
                time.sleep(removal_pause_s)
        result.content_remaining = max(0, len(content_pool) - content_position)
    else:
        result.content_skipped = "disabled"
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


def _record_content_progress(sessions_dir: Path, *, rows: list[ContentRow], already: int) -> None:
    """Fold this pass's CONTENT reclaims into the progress record (batch-granular).

    ADDITIVE keys only, so a reader of the record pass's keys reads exactly
    what it read before this layer existed: ``content_reclaimed_total``,
    ``content_freed_bytes_estimate`` (a lower bound; never quoted as a figure)
    and ``content_last_removal_at``. The one-time notice keys are deliberately
    NOT touched: that notice counts removed SESSIONS, and a content reclaim
    removes none — ``removed_total`` staying put is what keeps an old viewer's
    announcement honest.
    """
    fresh = len(rows) - already
    if fresh <= 0:
        return
    state = read_state(sessions_dir)
    total = state.get("content_reclaimed_total")
    state["content_reclaimed_total"] = (
        total if isinstance(total, int) and not isinstance(total, bool) else 0
    ) + fresh
    prior = state.get("content_freed_bytes_estimate")
    state["content_freed_bytes_estimate"] = (
        prior if isinstance(prior, int) and not isinstance(prior, bool) else 0
    ) + sum(row.bytes for row in rows[already:])
    state["content_last_removal_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_record(sessions_dir / STATE_NAME, state)


def _unannounced_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """The record IF it is still unannounced, else ``None``. Read-only.

    The one validity rule shared by the peek, the consume and the ack paths: a
    record that is absent, unreadable (``read_state`` gives ``{}``), already
    acknowledged, or carries no positive removal count announces nothing.
    """
    removed = state.get("removed_total")
    if (
        not state
        or state.get("notice_acknowledged")
        or not isinstance(removed, int)
        or isinstance(removed, bool)
        or removed <= 0
    ):
        return None
    return dict(state)


def peek_unannounced_delegated_notice(sessions_dir: Path) -> dict[str, Any] | None:
    """The notice record while nobody has announced it yet, WITHOUT consuming it.

    A read that flips nothing, for surfaces a GET must not mutate: the desktop
    list route is polled by many readers — the app's own attach and auth probes
    among them — and a consume-on-read there ate the once-per-store notice
    before the renderer could render it. The GET returns the notice while the
    record stays unacknowledged; the renderer flips it explicitly through
    ``acknowledge_delegated_notice``.
    """
    return _unannounced_payload(read_state(sessions_dir))


def take_unannounced_delegated_notice(
    sessions_dir: Path, *, runtime_pid: int | None = None, defer_to_writer: bool = True
) -> dict[str, Any] | None:
    """The progress record if no viewer has announced it yet, marking it announced.

    Same viewer rule as ``cleanup.take_unannounced_cleanup`` (the runtime that
    removed defers to its own viewer while it lives) with one switch for a
    viewer that has no runtime of its own: ``defer_to_writer=False`` takes it
    at once. ONE announcement per store, ever; a malformed or absent record
    announces nothing. The TUI's consume-on-show uses this; the desktop does
    NOT (a GET must not consume — it peeks and acknowledges separately).
    """
    path = sessions_dir / STATE_NAME
    state = read_state(sessions_dir)
    announced = _unannounced_payload(state)
    if announced is None:
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
    state["notice_acknowledged"] = True
    if not _write_record(path, state):
        logger.debug("session cleanup: %s is not writable; the notice will repeat", path)
    return announced


def acknowledge_delegated_notice(sessions_dir: Path) -> bool:
    """Flip ``notice_acknowledged`` so the notice is never announced again.

    The write half of the desktop pair (``peek_unannounced_delegated_notice``
    is the read half): the renderer calls this once it has SHOWN the notice,
    because the list route itself must not consume it. Idempotent — a second
    call, an already-acknowledged record, or no record at all all answer
    ``True``, the state the caller asked for (the sibling pin/archive
    convention); a client that needs the store's own answer re-reads the
    listing. A store this process cannot write logs and the notice repeats on
    a later read rather than erroring here.
    """
    state = read_state(sessions_dir)
    if state and state.get("notice_acknowledged") is not True:
        state["notice_acknowledged"] = True
        if not _write_record(sessions_dir / STATE_NAME, state):
            logger.debug(
                "session cleanup: %s is not writable; the notice will repeat",
                sessions_dir / STATE_NAME,
            )
    return True


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

    The GET PEEKS (it never consumes; the renderer acknowledges through the ack
    route), so the same payload rides every listed read until acknowledged.
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
                        _stamp_sweep(sessions_dir, result.remaining + result.content_remaining)
                    if (
                        result.remaining
                        or result.budget_exhausted
                        or result.content_remaining
                        or result.content_budget_exhausted
                    ):
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
