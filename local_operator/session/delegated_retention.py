"""Retention for DELEGATED sessions: the second class of ``session.cleanup``.

WHY A SECOND CLASS
==================

``session.cleanup.*`` was one global policy, off by default, that counted every
directory under ``sessions/`` the same. On the operator's real store
(2026-10-09: 19,397 sessions, 71.2 GB) 90% of the bytes were subagent
transcripts and 17,579 hidden-origin sessions had been idle for more than 48 hours
— machine bookkeeping nobody opens, which the one policy could only remove by
being switched on for the user's own conversations too. So the store is split
by TWO questions that must stay separate: the class split asks the LISTING
question ``resume.is_user_session_origin``, while what this pass may REAP is
``cleanup.deletable_origins()`` (a DESTRUCTION question; deriving one from the
other made every origin minted later deletable by default):

* PARENT class — everything the sidebar lists (no ``origin.json``, ``fork``,
  ``agent-workstream``). Governed by the existing five keys, OFF by default.
  ``cleanup.run_cleanup`` now iterates ONLY this class.
* DELEGATED class — every other origin (``subagent``, ``agent-shell``,
  ``agent-config`` and any future value, because visibility is opt-OUT). Governed
  by ``session.cleanup.delegated.*``, ON by default, with a bounded age
  (2..720 hours). This module is that class's whole policy.
* KEPT BY ORIGIN — hidden (so not the parent class) but not named in
  ``cleanup.deletable_origins()``: ``agent-config`` today, and every origin a
  later build mints until someone names it there. Neither policy removes these;
  the scan counts them (``kept_by_origin``) and carries a capped name list
  (``kept_by_origin_names``) so the preview can show WHICH ones were kept.

``cleanup.remove_session_dir`` remains the ONLY ``rmtree`` of a session
directory (``tests/unit/session/test_no_session_deletion.py``); this module
decides, and the CONTENT layer's removals run through ``cleanup``'s guarded
remover (``remove_scratchpad_entry``) — no deletion lives outside that module.

THE SAFETY RULES (every one fails CLOSED: cannot tell => keep)
==============================================================

A delegated session is kept when ANY of these holds:

0. Its origin is not one retention may reap (``cleanup.deletable_origins()``). The
   set is an ALLOW-LIST, so an unknown value, a newer build's origin, and the
   operator's own ``agent-config`` runs are kept — ambiguity costs disk, never a
   session nobody can recover.
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
they were never about bytes nobody can reach. So, for every delegated session
whose ACTIVITY is older than the record window (``max_age_hours``) and whose
record survives the record pass — an open project, an active parent or a
scratchpad repo is what keeps it — the scratchpad is walked; bounded, symlinks
never followed, ``.git`` never matched by shape — and reclaimed by four arms:

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
  - ``.git`` DIR (an independent clone): a clone whose object store serves
    registered worktrees is KEPT whole ("has registered worktrees" — a
    linked worktree's own HEAD lives per-worktree, invisible to the ref
    walks, so removing the store would leave a worktree this pass may keep an
    object-less husk). Otherwise no commit of HEAD may be absent from every
    remote, and the local refs (``refs/heads/*``, ``refs/stash`` and
    ``refs/tags/*``) are checked against the same remotes: when every tip is
    reachable the tree goes as-is, and when any is not, its refnames are
    rescued into a bundle beside the tree (``reap-rescue-<tree>.bundle``:
    ``git bundle create`` + ``git bundle verify``) BEFORE the tree may go —
    any bundle failure keeps THAT tree (per-tree: a pad with several
    reclaimable clones gets one bundle each, and a failed bundle keeps only
    its own tree). A ref the commit walk cannot speak for — a tag whose
    peeled target is not a commit (a blob or tree) — keeps the tree, checked
    BEFORE the empty-walk return, so an empty walk never authorises one. A
    rescue bundle carries no HEAD (bundles have none): recover a preserved
    ref with ``git clone -b <branch> <bundle> <dest>``.
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
* NON-GIT COPIES (OFF by default, ``session.cleanup.delegated.non_git_copies.
  enabled``). A plain directory ANYWHERE under ``scratchpad/`` (bounded to
  :data:`CONTENT_MAX_DEPTH`) is a candidate when it holds at least
  :data:`NON_GIT_COPY_MIN_BYTES`, has no ``.git`` anywhere below it and no
  refused segment name. It is tried TOP-DOWN: a directory that proves is taken
  file-by-file and its children are not tried; one that does not is descended
  into and each child tried in turn, so the largest provable subtree wins
  (measured: ``qa-1232/`` matches 0% because it mixes in session artifacts,
  while ``qa-1232/base-repo/`` matches 956 of its 975 MB). A directory with a
  ``.git`` anywhere below it is not a candidate and this arm does not descend
  into it — the merged-clean arm judges that repository (measured ``r7``,
  ``r1``, ``audio-home``).
  THE CLASS BOUNDARY: a non-git directory whose every byte can be shown to
  exist at the same path in a MERGED COMMIT is reproducible and may go; a
  non-git directory with no provenance is NOT reproducible and must stay.
  The proof is per FILE: git's blob id is computed here
  (``sha1("blob <size>\0" + content)``, streamed in 1 MiB chunks, no process
  per file) and matched by relative path against the reference repositories'
  merged trees, tried as a UNION: a file proves when ANY reference has that
  path at that blob, and the share is computed against the union. The union is
  the measured norm — most of the 50-155 MB copies on this store are of a
  DIFFERENT repository than the first one an operator names — and the class
  report names which reference proved each candidate. References are the
  repositories found inside the same pad (bounded) plus
  ``non_git_copies.reference_repos`` from config; a reference QUALIFIES only
  when its object format is SHA-1 (the ids computed here are SHA-1; a
  ``--object-format=sha256`` repository would otherwise match nothing and read
  as a 0% copy), its trunk (``origin/HEAD``, else ``origin/main``, else
  ``origin/master``) resolves, and its tree listing fits the bounds. Only
  COMMITTED, MERGED object content is read, so a dirty working tree cannot
  taint the proof — the operator's own ``~/local-operator-ui`` is dirty — and
  a ``--filter=blob:none`` partial clone is never asked for missing blobs
  (``ls-tree`` reads trees only, no network). A candidate is reclaimed only
  when matched bytes are >= :data:`NON_GIT_COPY_MIN_SHARE` of its bytes AND
  >= :data:`NON_GIT_COPY_MIN_BYTES`: one removal per matched FILE, and the
  unmatched files (session artifacts, logs, generated reports) stay with a
  keep line naming their count, bytes and up to three examples. Below the
  share the WHOLE candidate is kept — the arm must never take bytes it cannot
  prove. An UNREADABLE file or directory inside a candidate refuses that
  candidate with ``cannot read <rel>`` while the pad's other arms still run —
  it is never an error that aborts the pad (measured: a mode-000 fixture inside
  a real candidate). A nested ``.git`` means the directory is not a plain copy:
  the candidate is refused and the walk descends with this arm DISABLED below
  it, so the merged-clean arm judges that repository itself (a real pad held 7
  unmerged repos and this arm must never delete a repository that arm would
  keep). Blob ids are cached per pad (bounded), because a refused parent
  re-walks the files its proving child hashes. The evidence exemption does
  NOT apply here: it exists because the shape arm matches by NAME, while this
  arm proves every byte, so ``qa-1232/base-repo/docs/evidence`` (945 MB of
  tracked content) is exactly what it should reclaim. Empty directories are
  left behind (only matched files are removed). TWO VACUOUS GUARDS, because
  the failure that motivated this arm was a broken instrument read as a pass:
  a walk that yields zero regular files while an INDEPENDENT measure shows the
  tree holds bytes raises :class:`NonGitCopyScanError` (measured: a shell
  pipeline reported "0 files in a 1.0 GB copy", and the corrected pass via
  ``git hash-object --stdin-paths`` timed out at 1,800 s — hence ONE
  ``ls-tree`` per reference plus streaming hashes, never a git process per
  file). The measure is what makes the guard correct in both directions: a
  scratch HOME's empty directory scaffold has entries and ZERO bytes, and is
  descended into like any sub-threshold tree; and
  :func:`scan_non_git_copies` raises when a >= 25 GiB store has NO reachable
  scratchpad directory — a store whose pads WERE scanned and hold no copies is
  an honest empty report, while an unreadable store always raises rather than
  reporting "nothing to reclaim". A candidate over :data:`NON_GIT_COPY_MAX_FILES` /
  :data:`NON_GIT_COPY_MAX_ENTRIES` is refused (keep), and the real pass
  defers a candidate past :data:`NON_GIT_COPY_HASH_BUDGET_BYTES` to the next
  pass instead of hashing unbudgeted. WHY OFF BY DEFAULT: the other arms
  match a shape the write policy itself refuses or a tree git can prove is
  merged; a plain copy has neither property, so this one earns its trust per
  file and needs a reference repository — both facts an operator should opt
  into after a ``--report-non-git-copies`` preview. OPERATOR HOLDS: the ids in
  ``non_git_copies.hold_sessions`` are refused BEFORE their candidates are
  walked — reason ``operator hold`` — in the pass and in the class report
  alike, and the hold outranks proof: a held item is never a removal even when
  every byte is provable. It exists because a reservation stated in
  conversation does not bind an automated pass; the harness has no other
  per-session hold, and this arm must not have to infer one from a path.

HARD GUARDS: nothing is content-reclaimed for the current session, a live
claim or lease, an armed wake or monitor, a non-delegated origin, a session
with no activity clock, or a session whose activity is inside the record
window — and any guard evaluation that fails keeps the pad and counts an
error. THE AGE GATE IS BACK (2026-10-19): live-adjacent content is not ours,
and a transcript touched an hour ago means the session is being worked in even
when no claim or wake is visible. The per-command pad cap and the watchdog
bound live growth; reclamation serves sessions past the record window that
are still KEPT by a parent, an open project or the git guard, where the
transcript must stay and only cache-shaped content may go.
Parent-still-active, open projects and the git hazard are NOT content guards:
they keep the record, and that carve-out is the point. Unread spooled mail and
open asks ARE content guards (restored 2026-10-19, per the incident
requirement): a pad that can still hand the user a question, or mail they have
not read, is untouchable even when the record survives — the merged+clean+stale
class is not worth destroying a live conversation with the user. The pad root
itself is never removed,
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
  it runs against non-live delegated pads PAST the record window (a parent,
  an open project or a scratchpad repo is what keeps such a pad's record),
  and it takes only merged-clean git trees, build shapes and
  stale output, with the LIVENESS guards left standing.

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

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    DEFAULT_DELEGATED_MAX_AGE_HOURS,
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
    _has_open_asks,
    _has_spooled_mail,
    _is_delegated_dir,
    _is_deletable_dir,
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

#: WHERE a reclaim is attributable after the fact: one JSON manifest per pad
#: under ``sessions/.cleanup-manifests/``, written BEFORE the first removal and
#: rewritten with the completed subset after the loop. The aggregate cleanup
#: log row carries only counts, so without this a loss cannot name the path
#: (the incident this exists for: two pads were emptied and the log could not
#: say what had been in them). A write failure never fails the removal — the
#: log row records ``manifest_error`` instead.
CONTENT_MANIFEST_DIRNAME = ".cleanup-manifests"
CONTENT_MANIFEST_SCHEMA = "cleanup.removal.v1"

#: A dry run has no manifest, so ``ContentRow.paths`` (and therefore
#: ``--json``) carries the planned paths instead — bounded so one pathological
#: pad cannot make the serialised result enormous; ``paths_total`` keeps the
#: count. Real rows leave ``paths`` empty and point at the manifest.
CONTENT_ROW_PATHS_CAP = 200

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

#: The four class phrases a content row's reason is built from, in the fixed
#: order they are displayed and counted (see :func:`_PadPlan.classes`).
CONTENT_CLASS_MERGED = "merged-clean clone"
CONTENT_CLASS_BUILD = "build shapes"
CONTENT_CLASS_STALE = "stale output"
#: A plain non-git directory COPY of a merged tree (the fourth arm). Its own
#: phrase rather than "merged-clean clone" because nothing here is a clone: a
#: removal is per FILE, proven against a reference repository, and the row must
#: not claim a git check that never ran on this directory.
CONTENT_CLASS_COPY = "non-git copy"

#: --- The NON-GIT COPY arm -------------------------------------------------
#: Below this a copy candidate is not worth the per-file walk and hash churn.
#: The measured class on the real store (2026-10-20) is all >= 100 MB direct
#: children of ``scratchpad/`` (41 of them, ~9.4 GB); 50 MB is 2x under that.
NON_GIT_COPY_MIN_BYTES = 50 * 1024 * 1024
#: Matched bytes / total bytes. Below it the directory is NOT a copy of a
#: merged tree and the WHOLE candidate is kept: a mongod data dir measured ~1%
#: (its files merely share a few paths), a real QA copy measured 98%. A share
#: test rather than an exact match because a copy legitimately carries session
#: artifacts (logs, generated reports) that exist in no committed tree.
NON_GIT_COPY_MIN_SHARE = 0.90
#: Per-candidate bounds; exceeding either is a REFUSAL (keep the candidate),
#: never a partial guess. Measured: the largest real copy held 24,151 files
#: (975 MB), so these leave ~8x headroom and still bound a pathological tree.
NON_GIT_COPY_MAX_FILES = 200_000
NON_GIT_COPY_MAX_ENTRIES = 400_000
#: Hash-cache entries per PAD (dropped between pads). A refused parent re-walks
#: the files its proving child will hash (measured 2x on a 975 MB copy), so one
#: pass reuses ids; the cap keeps that memory bounded at roughly 300k entries.
NON_GIT_COPY_HASH_CACHE = 300_000
#: Hashing budget for ONE real pass: 975 MB hashed in 6.7 s measured, so 768 MB
#: bounds a pass to a few seconds of hashing. A candidate whose bytes do not fit
#: the remaining budget is DEFERRED to the next pass (reason "hash budget") and
#: sets :attr:`DelegatedResult.copy_budget_exhausted`; a dry run is unbudgeted.
NON_GIT_COPY_HASH_BUDGET_BYTES = 768 * 1024 * 1024
#: The store-level vacuous-scan floor (see :func:`scan_non_git_copies`): a store
#: this large whose scan found ZERO copy children is the bug this guard exists
#: for, not a clean store. 25 GiB is roughly a quarter of the measured store.
NON_GIT_COPY_STORE_MIN_BYTES = 25 * 1024 * 1024 * 1024

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
    #: Hidden-origin directories kept with no row: inside the age window, or with
    #: no USABLE clock (a missing marker, or an implausible birth stamp — see
    #: :func:`_clock_without_activity`). Both mean "cannot tell it is old", so both
    #: are kept, and the preview says which counts they are in.
    within_window: int = 0
    #: Hidden-origin directories OLD ENOUGH to go but kept because their origin is
    #: not in ``cleanup.deletable_origins()`` (``agent-config`` today, and any origin
    #: a newer build mints). Neither class removes them; counted so the preview's
    #: totals are explainable rather than mysterious.
    kept_by_origin: int = 0
    #: Names of the ``kept_by_origin`` group (capped like ``window_names``), so a
    #: dry run can point the operator at sessions that never appear in the sidebar.
    kept_by_origin_names: list[str] = field(default_factory=list)
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
    #: --- The non-git COPY arm, counted for the CLI's report block. Only the
    #: PASS-integrated arm's numbers: the read-only class report
    #: (:func:`scan_non_git_copies`) carries its own, unsized-by-gate totals.
    #: Candidates are TOP-LEVEL attempts (a refused parent whose child proves
    #: counts once, at the parent); reclaimed bytes are every proven file,
    #: nested rows included.
    copy_candidates: int = 0
    copy_candidate_bytes: int = 0
    #: Bytes a real run actually TOOK (taken removals only; a dry run's plan).
    copy_reclaimed_bytes: int = 0
    copy_budget_exhausted: bool = False
    #: Why the arm did not run, when it did not: ``None``, ``"disabled"`` (its
    #: config switch or a master switch is off), ``"no reference repository"``
    #: (it ran, nothing was configured and no in-pad repository qualified), or
    #: ``"holds unreadable"`` (the hold list was present but malformed, so the
    #: arm refused to guess).
    copy_skipped: str | None = None


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
    #: when the uniqueness check found refs that live nowhere else. The
    #: FIRST bundle and the TOTAL bytes of all of them; the list of every
    #: bundle is :attr:`rescue_bundles`.
    rescue_bundle: str = ""
    rescue_bundle_bytes: int = 0
    #: Every rescue bundle this pad kept, one per reclaimable clone tree with
    #: unique refs (review F1 made these per-tree). Empty when none was
    #: needed; ``rescue_bundle`` mirrors the first for older readers.
    rescue_bundles: tuple[str, ...] = ()
    #: How the removal was carried out: ``"remove"`` (the guarded pad-level
    #: path), ``"worktree-remove"`` (a linked worktree removed through its
    #: shared repository), or ``"bundle-rescue"`` (an independent clone whose
    #: unique refs were bundled first). The cleanup-log row carries it so a
    #: reader can tell which of the three actually ran.
    method: str = "remove"
    #: What ran the pass (``startup`` / ``cli``) and why (``session-startup`` /
    #: ``manual``); the log row carries the pair so a reclaim that was never
    #: requested by a person is distinguishable from one that was.
    trigger: str = "startup"
    #: Absolute path of this pad's manifest, when one was written. Empty for a
    #: dry run (no file is written) and when every manifest write failed.
    manifest: str = ""
    #: Why the manifest is missing or incomplete; the log row carries it
    #: rather than swallowing a failed write.
    manifest_error: str = ""
    #: Path-level provenance carried IN the row when the on-disk manifest
    #: cannot name the taken paths: a dry run (no file), a failed manifest
    #: write, or a failed COMPLETED rewrite whose manifest still says
    #: ``completed: []`` — the cleanup-log row then carries these paths instead
    #: of a bare count. Bounded by :data:`CONTENT_ROW_PATHS_CAP`; a complete
    #: manifest leaves this empty and points at the file.
    paths: tuple[str, ...] = ()
    #: How many paths the manifest records (or the dry run planned); ``paths``
    #: may hold only the first :data:`CONTENT_ROW_PATHS_CAP` of them.
    paths_total: int = 0
    #: How old the activity clock was at guard time, in hours. ``None`` only
    #: for rows built by callers that did not run a guard (tests, CLI display).
    activity_age_hours: float | None = None


@dataclass
class NonGitCopyCandidate:
    """One directory the READ-ONLY class report evaluated as a copy candidate.

    Candidates are RECURSIVE: a directory is tried first, and only when it does
    not prove are its children tried (the largest provable subtree wins). Rows
    for nested attempts carry ``top_level=False`` so the totals can name the
    top-most directories an operator sees without counting a parent's bytes
    twice. Each row carries the numbers that justify the decision — total
    bytes, proven bytes, which reference proved them, the unmatched remainder
    with examples — so the proof can be audited before the arm is switched on.
    """

    session: str
    #: ``sessions/<id>/scratchpad/...`` relative to the sessions root.
    path: str
    bytes: int = 0
    #: Bytes whose path+blob id exists in at least one qualifying merged tree.
    matched_bytes: int = 0
    unmatched_files: int = 0
    unmatched_bytes: int = 0
    #: Up to three unmatched relative paths (what the keep line names).
    examples: tuple[str, ...] = ()
    #: The reference repositories (realpaths) whose merged trees proved at least
    #: one file. The proof is a UNION: a file proves if ANY of them has it.
    proved_by: tuple[str, ...] = ()
    #: Hours since the session's last activity; ``None`` means no usable clock.
    age_hours: float | None = None
    #: Whether the session is inside the report's window. A proof that passes
    #: inside the window is reported SEPARATELY from "would reclaim now": the
    #: measured class is 12-30 h old against a 48 h default, and a report that
    #: hid those would say "nothing to do" while 9 GB sits there.
    within_window: bool = False
    #: Whether the session is on the operator's hold list. A hold outranks
    #: proof: the row says ``operator hold`` and is never a would-reclaim.
    held: bool = False
    #: The scan could not evaluate this candidate at all (the instrument's
    #: contradiction guard fired). Recorded as a row, never allowed to abort the
    #: whole store report.
    error: bool = False
    #: Whether the path+blob proof passed (share and floor), independent of age.
    proof_ok: bool = False
    #: The proof passed but the retention window keeps the session: the row is
    #: WAITING, distinct from a refusal (design C1's labelled states). A
    #: pad-root guard clears it: that row is kept, not waiting.
    waiting: bool = False
    #: Whether no ANCESTOR was already attempted (see the totals' docstring).
    top_level: bool = True
    #: ``would reclaim now``: proof passed, the session is past the window, and
    #: it is a delegated, reapable origin. Liveness guards (claim/lease, armed
    #: wake, current session) are evaluated at pass time and are NOT part of a
    #: read-only report.
    would_reclaim: bool = False
    #: Why it would not, when it would not ("no reference repository",
    #: "contains a git repository", "only 12% of its bytes ... are in a merged
    #: tree", "proof ok, inside the 48h window", "not a delegated session", ...).
    #: Empty when it would reclaim.
    reason: str = ""


@dataclass
class NonGitCopyReport:
    """The whole-store, READ-ONLY non-git-copy class report (``--report-non-git-copies``).

    It enumerates every ``sessions/*/scratchpad`` subtree that has the candidate
    SHAPE — no age or liveness gate, because its job is to show the class the
    pass-integrated arm governs once switched on, including sessions the record
    window would keep. It changes nothing and writes nothing.

    ``count`` / ``candidate_bytes`` are TOP-LEVEL attempts only (the direct
    children the operator thinks in). The STATE figures count distinct
    SESSIONS, not rows: one session can hold several provable subtrees, and a
    summary that says "2 pad(s)" for two subtrees of one session is a number
    nobody can act on. ``kept_candidates`` is candidates again — a top-level
    directory in no reclaim/waiting/held session — which is why the parts need
    not add up to ``count``.
    """

    candidates: list[NonGitCopyCandidate] = field(default_factory=list)
    window_hours: float = float(DEFAULT_DELEGATED_MAX_AGE_HOURS)
    references: tuple[str, ...] = ()
    #: ``None``; ``"no store"`` when ``sessions/`` does not exist yet (a fresh
    #: install); ``"no reference repository"`` when nothing was configured and
    #: no in-pad repository qualified; ``"holds unreadable"`` when the hold
    #: list could not be parsed.
    skipped: str | None = None
    #: Accumulated only on the zero-candidate path (the vacuous guard); 0 when
    #: the scan had candidates and never needed the figure.
    store_bytes: int = 0
    #: Candidates the instrument could not evaluate (a raised
    #: :class:`NonGitCopyScanError`). The CLI exits 3 when this is non-zero.
    errors: int = 0

    @property
    def top_level(self) -> list[NonGitCopyCandidate]:
        return [row for row in self.candidates if row.top_level]

    @property
    def count(self) -> int:
        return len(self.top_level)

    @property
    def candidate_bytes(self) -> int:
        return sum(row.bytes for row in self.top_level)

    @property
    def reclaimable_sessions(self) -> int:
        """Distinct SESSIONS with a would-reclaim row.

        Sessions, not rows: one session can hold several provable subtrees
        (``qa-1232`` and ``qa-1232/base-repo`` both descend from one pad), and
        "2 pad(s)" for one session's two subtrees is a number nobody can act on.
        """
        return len({row.session for row in self.candidates if row.would_reclaim})

    @property
    def reclaimable_bytes(self) -> int:
        return sum(row.matched_bytes for row in self.candidates if row.would_reclaim)

    @property
    def within_window_sessions(self) -> int:
        """Distinct sessions whose proof passed but whose window blocks them."""
        return len({row.session for row in self.candidates if row.waiting})

    @property
    def within_window_bytes(self) -> int:
        return sum(row.matched_bytes for row in self.candidates if row.waiting)

    @property
    def held_sessions(self) -> int:
        return len({row.session for row in self.candidates if row.held})

    @property
    def kept(self) -> list[NonGitCopyCandidate]:
        """Top-level candidates in no reclaim/waiting/held session (design C1).

        The kept figure is a CANDIDATE count beside session-state counts on
        purpose: a refused top-level directory whose nested child is provable
        is still kept as a candidate, and the child's state is counted in its
        own unit. That is why the parts need not add up to ``count``.
        """
        busy = {
            row.session for row in self.candidates if row.would_reclaim or row.waiting or row.held
        }
        return [row for row in self.top_level if row.session not in busy]

    @property
    def kept_candidates(self) -> int:
        return len(self.kept)

    @property
    def kept_bytes(self) -> int:
        return sum(row.bytes for row in self.kept)

    @property
    def refused(self) -> int:
        return sum(
            1 for row in self.candidates if not row.proof_ok and not row.held and not row.error
        )

    def as_dict(self) -> dict[str, Any]:
        """The ``--json`` object, every row included (the text listing caps rows)."""
        return {
            "candidates": [asdict(row) for row in self.candidates],
            "count": self.count,
            "rows": len(self.candidates),
            "candidate_bytes": self.candidate_bytes,
            "would_reclaim_now": self.reclaimable_sessions,
            "reclaimable_bytes": self.reclaimable_bytes,
            "proof_ok_within_window": self.within_window_sessions,
            "withheld_bytes": self.within_window_bytes,
            "held": self.held_sessions,
            "kept_candidates": self.kept_candidates,
            "kept_bytes": self.kept_bytes,
            "refused": self.refused,
            "errors": self.errors,
            "window_hours": self.window_hours,
            "references": list(self.references),
            "skipped": self.skipped,
            "store_bytes": self.store_bytes,
        }


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


#: The earliest birth stamp this build will BELIEVE: 2020-01-01T00:00:00Z. Below
#: it is a units error or a truncated epoch (``1.7e8`` is 1975), not a birth —
#: the harness did not exist to write one. Believing it would rank the session as
#: maximally OLD, which is the one direction that removes data.
MIN_PLAUSIBLE_BIRTH = 1_577_836_800.0

#: How far AHEAD of the wall clock a birth stamp may be and still count as a clock
#: (NTP skew, a writer on another host, a test's fixed ``now``): one day. A far
#: future value keeps the session anyway (a negative age is inside every window),
#: so this bound is not about danger — it is that a garbage number which happens
#: to sort sanely is still a number nobody wrote, and trusting it puts an invented
#: timestamp into the candidate ordering.
MAX_BIRTH_SKEW_S = 86_400.0


def _clock_without_activity(directory: Path, now: float) -> float | None:
    """The age clock of a directory that was never worked in, or ``None``.

    ``created_at.json`` first (the birth stamp the transcript constructor writes —
    a bare float), then its mtime, then the directory's. The directory's mtime is
    a LATER time than the birth (it moves with every entry added), which only ever
    makes the session look younger: the safe side. ``None`` means no clock could
    be established and the caller keeps the directory.

    A PRESENT but IMPLAUSIBLE stamp (below :data:`MIN_PLAUSIBLE_BIRTH`, more than
    :data:`MAX_BIRTH_SKEW_S` ahead, zero, negative, ``NaN``) is not a clock to
    second-guess: it is corrupt, and the answer is ``None`` — keep. It must NOT
    fall back to mtime, because the mtime can be stale too (the marker is exactly
    what a truncated writer left behind) and ranking a corrupt session old is the
    inverse of safe. Only a MISSING or unparseable marker takes the mtime
    fallback, and there the stamp never existed to contradict it.

    ``now`` is the pass's injected clock, never ``time.time()``: the upper bound
    must move with the caller's notion of "now" or a test that pins time cannot
    exercise the boundary deterministically (and the scan already has one
    ``moment``; a second, independently-read clock is how the two drift).
    """
    marker = directory / "created_at.json"
    try:
        raw = marker.read_text(encoding="utf-8").strip()
        parsed: Any = json.loads(raw)
        if isinstance(parsed, Mapping):
            parsed = parsed.get("created_at", parsed.get("at"))
        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
            try:
                stamp = float(parsed)
            except OverflowError:  # a JSON integer too large for a float
                return None
            if MIN_PLAUSIBLE_BIRTH <= stamp <= now + MAX_BIRTH_SKEW_S:
                return stamp
            return None
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
#: so both classes provably share one classifier. The same holds for the REAP
#: predicate, which is the narrower question — a directory can be in this class and
#: still not deletable (``agent-config``).
is_delegated_dir = _is_delegated_dir
is_deletable_dir = _is_deletable_dir


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


#: Environment variables that REDIRECT which repository or object store git
#: answers about. ``git -C <repo>`` is not authoritative while any of them is
#: set: measured with ``GIT_DIR=<repoB>/.git``, ``_merged_tree(repoA)`` returned
#: repoB's tree, and the proof would have judged one repository against another
#: repository's content. Scrub them on every call; a polluted parent shell must
#: never be able to redirect a proof about a copy.
_GIT_ENV_SCRUBBED = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    }
)


def _git_env() -> dict[str, str]:
    """``env -u XPC_FLAGS`` + no system config + no prompts + no index writes.

    ``XPC_FLAGS=0x2`` breaks name resolution for children of a launchd-spawned
    tool on this host; ``GIT_OPTIONAL_LOCKS=0`` keeps ``git status`` from
    refreshing the index of a repository we are inspecting, not changing. The
    repo-location variables in :data:`_GIT_ENV_SCRUBBED` are removed because a
    caller's environment must not be able to point a proof at another repo.
    """
    env = {k: v for k, v in os.environ.items() if k != "XPC_FLAGS" and k not in _GIT_ENV_SCRUBBED}
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
    than colliding with a top-level tree that shares its basename. The
    flattening cannot keep ``a/b`` and ``a-b`` apart, which is what
    :func:`_unique_rescue_bundle_path` is for.
    """
    flattened = relative.replace(os.sep, "-")
    return Path(root) / f"{RESCUE_BUNDLE_PREFIX}{flattened}{RESCUE_BUNDLE_SUFFIX}"


def _unique_rescue_bundle_path(root: str, relative: str, taken: set[str]) -> Path:
    """A bundle path no other tree of this pad has claimed (review N2).

    ``_rescue_bundle_path`` flattens separators, so ``a/b`` and ``a-b`` map to
    one name; a pad may carry several reclaimable trees and each needs its own
    bundle, so the second user of a name takes ``-2`` (then ``-3``...).
    """
    base = _rescue_bundle_path(root, relative)
    if base.name not in taken:
        taken.add(base.name)
        return base
    stem = base.name[: -len(RESCUE_BUNDLE_SUFFIX)]
    serial = 2
    while True:
        candidate = base.with_name(f"{stem}-{serial}{RESCUE_BUNDLE_SUFFIX}")
        if candidate.name not in taken:
            taken.add(candidate.name)
            return candidate
        serial += 1


def _tree_reclaim_decision(repo: Path) -> _TreeDecision:
    """Rule (c): may this git tree be reclaimed WHOLE, and does it need a bundle?

    The desk's test (``tick-role.md``, "Reclaimable class — stale session
    scratchpad trees") as tightened for this pass: HEAD must resolve, the
    worktree must be clean, and HEAD must be an ancestor of the remote trunk
    (``origin/HEAD`` -> ``origin/main`` -> ``origin/master``; no trunk keeps).
    The tree's ``.git`` then decides HOW it may go: a linked worktree is
    removed through its shared repository (no bundle — its refs and objects
    are the shared store's) and only when that repository LISTS this path; an
    independent clone that serves registered worktrees is kept whole, and any
    other clone must additionally have no commit of HEAD on no remote, with
    its local refs (``refs/heads/*``, ``refs/stash`` and ``refs/tags/*``)
    checked against the same remotes for uniqueness — when any tip is
    unreachable its refname comes back so the caller rescues it into a bundle
    BEFORE the tree may go. Dirty, unmerged, no-remote and unresolvable trees
    all come back as keep — anything that cannot be PROVEN reclaimable is
    kept, and this function never raises (a repository it cannot inspect is a
    keep).
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
    # A clone whose object store SERVES REGISTERED WORKTREES is kept whole: a
    # linked worktree's own HEAD lives per-worktree (invisible to the ref
    # walks below), and removing the store would leave a worktree this pass
    # keeps an object-less husk (fail closed).
    registrations = _git(path, "worktree", "list", "--porcelain")
    if registrations.returncode != 0:
        return _TreeDecision(False, "the worktree registrations cannot be inspected")
    if sum(1 for line in registrations.stdout.splitlines() if line.startswith("worktree ")) > 1:
        return _TreeDecision(False, "has registered worktrees")
    on_remote = _git(path, "log", "HEAD", "--not", "--remotes", "--oneline")
    if on_remote.returncode != 0:
        return _TreeDecision(False, "cannot decide whether HEAD is on a remote")
    if on_remote.stdout.strip():
        return _TreeDecision(False, "HEAD carries commits that are on no remote")
    # The local refs (heads + stash + tags) against the same remotes. The
    # per-tip question — "is this tip reachable from any remote ref?" — is
    # answered by ONE ``rev-list --branches --tags [refs/stash] --not
    # --remotes`` walk: the output is the set of commits no remote ref
    # reaches, and a tip is unique exactly when it appears there. Tags are
    # durable refs and belong on the positive side: a tag-only commit (the
    # branch was deleted) is otherwise invisible and the tree would go with no
    # bundle (review F2). (``--no-walk`` is NOT usable here: git documents it
    # as having no effect when a range is given, and ``--not`` IS one —
    # verified on git 2.55, the full walk came out anyway.) The walk is still
    # bounded: over :data:`CONTENT_MAX_UNIQUE_COMMITS` the list cannot be
    # proven complete, so the tree is kept (fail closed).
    stash = _git(path, "rev-parse", "-q", "--verify", "refs/stash")
    refs = ["rev-list", "--max-count", str(CONTENT_MAX_UNIQUE_COMMITS + 1), "--branches", "--tags"]
    if stash.returncode == 0 and stash.stdout.strip():
        refs.append("refs/stash")
    refs += ["--not", "--remotes"]
    unique = _git(path, *refs)
    if unique.returncode != 0:
        return _TreeDecision(False, "cannot test the local refs against the remotes")
    unique_commits = unique.stdout.split()
    if len(unique_commits) > CONTENT_MAX_UNIQUE_COMMITS:
        return _TreeDecision(False, "more unique commits than the rescue check will list")
    # The listing runs BEFORE the "nothing unique" return (review F2b / QA Q1):
    # the commit walk silently ignores a tag whose target is not a commit, so
    # an empty walk must never authorise a tree whose refs include one — the
    # listing is the only thing that SEES them (measured: a lightweight blob
    # tag on an otherwise fully-pushed clone reclaimed the tree with no
    # bundle, and the local-only blob was lost).
    listing = _git(
        path,
        "for-each-ref",
        "--format=%(objectname)%00%(objecttype)%00%(*objectname)%00%(*objecttype)%00%(refname)",
        "refs/heads",
        "refs/stash",
        "refs/tags",
    )
    if listing.returncode != 0:
        return _TreeDecision(False, "cannot list the local refs to rescue")
    wanted = set(unique_commits)
    names: list[str] = []
    for line in listing.stdout.splitlines():
        # NUL-delimited fields: ``%(objectname)``, ``%(objecttype)``, the
        # peeled object and type (empty for non-tags), ``%(refname)``. A
        # space-separated format would collapse the empty peeled fields and
        # make positional parsing ambiguous (measured: a non-tag line came
        # back two fields, and the tag rescue silently dropped every ref).
        fields = line.split("\x00")
        if len(fields) != 5:
            continue
        object_id, object_type, peeled, peeled_type, refname = fields
        if (peeled_type or object_type) != "commit":
            # An annotated tag's objectname is the TAG object; its commit is
            # the peeled one (``%(*objectname)``, empty for non-tags), which
            # is what the commit set can speak for. A ref whose PEELED target
            # is not a commit (a tag on a blob or tree) is outside the commit
            # walk's vocabulary — kept, not rescued (fail closed), whether or
            # not the walk found unique commits.
            return _TreeDecision(False, "a local ref points at a non-commit object")
        tip = peeled or object_id
        if tip in wanted:
            names.append(refname)
    if not unique_commits:
        return _TreeDecision(True)
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


class NonGitCopyScanError(RuntimeError):
    """The scan produced an IMPOSSIBLE result; the caller must not read it as a pass.

    Raised by the walk's vacuous guard (a non-empty candidate tree that
    produced zero regular files) and by the store-level guard (a >= 25 GiB
    scratchpad store whose scan found zero copy children). Both mean the
    instrument failed, not that there is nothing to reclaim — the measured
    incident was a shell pipeline reporting "0 files in a 1.0 GB copy" and a
    reader taking it as success.
    """


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
    #: This tree's OWN rescue bundle: the target path and the refnames that
    #: live only in it. PER-TREE on purpose (review F1): one pad may carry
    #: several reclaimable clones, and a bundle is written and verified
    #: before ITS OWN tree goes — a failure keeps only that tree.
    bundle_path: Path | None = None
    bundle_refs: list[str] = field(default_factory=list)


@dataclass
class _PadPlan:
    """What the content pass will take from ONE pad, decided but not yet done."""

    removals: list[_PlannedRemoval] = field(default_factory=list)
    #: Bundle FILENAMES already planned in this pad, so two trees whose
    #: flattened names collide (``a/b`` vs ``a-b``) still get one bundle each
    #: (review N2).
    bundle_names: set[str] = field(default_factory=set)
    #: Why specific candidates were left — an unmerged tree, a broken gitdir
    #: link, an aborted cap. The caller records these so the dry run shows what
    #: it declined, not only what it took.
    kept: list[str] = field(default_factory=list)
    #: --- The non-git COPY arm's counters for this pad. Kept bytes includes
    #: files left inside a reclaimed candidate (unmatched artifacts) as well as
    #: whole candidates refused or deferred, so ``candidate_bytes - kept_bytes``
    #: is the proven-reclaimed figure and the two can never disagree.
    copy_candidates: int = 0
    copy_candidate_bytes: int = 0
    copy_deferred: int = 0

    def classes(self) -> tuple[str, ...]:
        """The distinct classes taken, in the fixed display order."""
        taken = {removal.cls for removal in self.removals}
        return tuple(
            name
            for name in (
                CONTENT_CLASS_MERGED,
                CONTENT_CLASS_BUILD,
                CONTENT_CLASS_STALE,
                CONTENT_CLASS_COPY,
            )
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


def _plan_pad_content(
    pad: Path, *, now: float, copy_arm: "_CopyArm | None" = None
) -> _PadPlan | None:
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

    ``copy_arm`` is the non-git COPY arm, or ``None`` when its switch is off —
    in which case this function behaves exactly as it did before the arm
    existed. When it is on, each non-git directory the walk reaches is offered
    to :func:`_plan_copy` BEFORE the walk descends; the walk descends only when
    the copy planner says so (a nested repository, a refusal, or a directory
    that is not a candidate), so today's shape/stale rules keep seeing exactly
    what they saw before, except inside a directory this arm reclaimed.
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
    if copy_arm is not None:
        # The hash cache lives for ONE pad (a refused parent re-walks the files
        # its proving child hashes), and a pad with no qualifying reference has
        # nothing this arm can prove against: leave the walk exactly as it was
        # before the arm existed and report the condition once at pass level.
        copy_arm.begin_pad()
        if pad.name in copy_arm.holds:
            # An OPERATOR HOLD outranks proof and costs nothing: the pad is
            # refused before a reference is resolved or a candidate walked, and
            # the reason is recorded so the dry run explains the silence.
            plan.kept.append(
                "copy arm: operator hold "
                "(session.cleanup.delegated.non_git_copies.hold_sessions)"
            )
            copy_arm = None
        elif not _has_copy_shaped_child(Path(root)):
            # NO COPY-SHAPED CHILD: the arm could not prove anything here, so it
            # must not pay the scratchpad walk and one ``ls-tree`` per reference
            # (review A2; measured: every pad paid them, most pads had nothing).
            copy_arm = None
        elif not copy_arm.trees(pad):
            refusal = copy_arm.pad_refusal(pad)
            plan.kept.append(
                f"copy arm: {refusal}"
                if refusal
                else "copy arm: no qualifying reference repository "
                "(none configured, or no merged trunk to prove against)"
            )
            copy_arm = None
    visited = 0
    repos = 0
    # ``(path, depth, copy_ancestor, copy_allowed)``: the last two carry the
    # copy arm down the walk — ``copy_ancestor`` marks a subtree whose top-most
    # directory was already attempted (so a nested proof is not counted as a new
    # candidate), and ``copy_allowed`` is cleared below a nested repository.
    stack: list[tuple[str, int, bool, bool]] = [(root, 0, False, True)]
    while stack:
        current, depth, copy_ancestor, copy_allowed = stack.pop()
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
                    refused_name = _is_refused_segment(name)
                    if refused_name:
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
                    # The COPY arm owns plain directories at any depth, and
                    # never one the shape arm already took whole by name. It
                    # decides BEFORE the descent: a proving directory is
                    # planned file-by-file and its children are not tried; a
                    # refused or sub-threshold one descends so a copy one level
                    # down can prove (the largest provable subtree wins). See
                    # :func:`_plan_copy` for the verdicts.
                    if (
                        copy_arm is not None
                        and copy_allowed
                        and depth < CONTENT_MAX_DEPTH
                        and not refused_name
                    ):
                        verdict = _plan_copy(
                            entry.path, pad, plan, copy_arm, counted=not copy_ancestor
                        )
                        if verdict == _COPY_BELOW:
                            # Not a candidate at THIS size, but its children may
                            # hold the copy (measured: the copy sits under a run
                            # dir like ``r3/normal-prev``): descend unchanged.
                            stack.append((entry.path, depth + 1, copy_ancestor, copy_allowed))
                        elif verdict in (_COPY_DESCEND, _COPY_BLOCKED):
                            stack.append(
                                (
                                    entry.path,
                                    depth + 1,
                                    True,
                                    verdict == _COPY_DESCEND,
                                )
                            )
                        continue
                    if depth >= CONTENT_MAX_DEPTH:
                        continue  # beyond the cap: left in place (the keep side)
                    stack.append((entry.path, depth + 1, copy_ancestor, copy_allowed))
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
    bundle_path: Path | None = None
    if decision.bundle_refs:
        bundle_path = _unique_rescue_bundle_path(root, relative, plan.bundle_names)
    plan.removals.append(
        _PlannedRemoval(
            Path(path),
            CONTENT_CLASS_MERGED,
            *stats,
            method="worktree-remove" if decision.worktree is not None else "remove",
            shared_repo=decision.worktree,
            bundle_path=bundle_path,
            bundle_refs=decision.bundle_refs,
        )
    )


# ---------------------------------------------------------------------------
# The non-git COPY arm (content rule (d)) and its read-only class report
# ---------------------------------------------------------------------------
#
# WHY THIS IS A SEPARATE MACHINERY FROM THE MERGED-CLEAN ARM. That arm asks git
# about a REPOSITORY ("is HEAD merged and clean?") and removes the whole tree on
# one answer. This one has no repository to ask: it computes git's blob id per
# file and matches it by relative path against a reference repository's merged
# tree, because the directory is a plain copy that may carry anything. One
# `ls-tree` per reference plus streaming local hashes is the whole cost model —
# per-file `git hash-object` was measured at 1,800 s on one 1 GB copy before it
# timed out, and `git cat-file --batch-check` tried the network on a partial
# clone.
#
# EVERY function here is READ-ONLY: the report never removes (the pass plans
# removals separately) and no helper writes a ref, an index or a bundle.


def _mb(value: int) -> str:
    """A decision reason's byte figure — MB for copy-sized values, never "0 MB".

    A keep line's remainder can be a few bytes (a marker file); rounding that
    to ``0 MB`` would read as "nothing kept" beside a file count that says
    otherwise.
    """
    if value >= 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):.1f} GB"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.0f} MB"
    if value >= 1024:
        return f"{value / 1024:.0f} kB"
    return f"{value} B"


def _percent(part: int, whole: int) -> str:
    """A share for a keep reason: ``0%`` only when truly zero, else ``<1%``.

    A nonzero fraction that truncates to 0 ("0% of its bytes (4 MB of 655 MB)")
    reads as a bug in the instrument, not as a refusal with a real remainder.
    """
    if part <= 0 or whole <= 0:
        return "0%"
    value = int(part * 100 / whole)
    return "<1%" if value == 0 else f"{value}%"


def _blob_sha1(path: Path, size: int) -> str:
    """Git's blob id for ``path``, streaming — the exact object git would hash.

    ``sha1("blob <size>\\0" + content)`` in 1 MiB chunks. No process per file:
    a 24k-file copy would otherwise spawn 24k gits, which is the measured
    failure mode this arm exists to avoid repeating. A file whose size changed
    after the walk hashes to something that matches no tree, so the race keeps
    the file (the safe side).
    """
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with open(path, "rb") as handle:
        remaining = size
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _merged_tree(repo: str, *, format_checked: bool = False) -> tuple[str, dict[str, str]] | None:
    """``(trunk sha, {relative path: blob sha})`` for a repo's MERGED trunk.

    ``None`` when the repo does not qualify: no trunk resolving (``origin/HEAD``,
    else ``origin/main``, else ``origin/master``), a non-``sha1`` object format
    (the blob ids this arm computes are SHA-1; a SHA-256 repository would
    silently match nothing — see :func:`_reference_format_refusal`), a
    ``ls-tree`` failure, or a listing past the parse bounds. ``format_checked``
    lets a caller that already asked :func:`_reference_format_refusal` avoid a
    second ``git rev-parse`` per reference per pad (review: it ran twice).

    Reads TREE OBJECTS ONLY — never blobs and never the working tree — for two
    measured reasons: the operator's own ``~/local-operator-ui`` is dirty, and
    a dirty worktree must not disqualify
    (or taint) a proof about COMMITTED content; and a ``--filter=blob:none``
    partial clone must never be asked to fetch a missing blob (``ls-tree``
    cannot). No clean-worktree requirement belongs here on purpose.
    """
    if not format_checked and _reference_format_refusal(repo) is not None:
        return None
    trunk = ""
    for name in ("origin/HEAD", "origin/main", "origin/master"):
        try:
            resolved = _git(repo, "rev-parse", "--verify", f"{name}^{{commit}}")
        except (OSError, subprocess.SubprocessError):
            return None
        if resolved.returncode == 0:
            lines = resolved.stdout.strip().splitlines()
            if lines and lines[0].strip():
                trunk = lines[0].strip()
                break
    if not trunk:
        return None
    try:
        listing = _git(repo, "ls-tree", "-r", "--format=%(objectname) %(path)", trunk)
    except (OSError, subprocess.SubprocessError):
        return None
    if listing.returncode != 0:
        return None
    # Bound AFTER capture: ``_git`` returns a completed process, so the parse
    # bound is what keeps a pathological repo out of memory/time, and the 5 s
    # timeout bounds the capture (measured 0.055 s for 24,461 entries).
    if len(listing.stdout) > 64 * 1024 * 1024:
        return None
    tree: dict[str, str] = {}
    for index, line in enumerate(listing.stdout.splitlines()):
        if index >= 400_000:
            return None
        objectname, separator, path = line.partition(" ")
        if separator:
            tree[path] = objectname
    if not tree:
        return None
    return trunk, tree


def _reference_format_refusal(repo: str) -> str | None:
    """Why ``repo`` cannot be a reference because of its object format, or None.

    The proof computes a SHA-1 blob id itself (streaming, no subprocess). A
    repository initialized with ``--object-format=sha256`` stores different ids,
    so every file would compare unequal and the candidate would report a false
    "0% match" instead of telling the operator the reference is unusable. An
    unreadable format answer is NOT a refusal here: ``_merged_tree``'s other
    checks decide that case.
    """
    try:
        found = _git(repo, "rev-parse", "--show-object-format")
    except (OSError, subprocess.SubprocessError):
        return None
    if found.returncode != 0:
        return None
    value = found.stdout.strip()
    if value and value != "sha1":
        return f"reference object format is {value}"
    return None


def _repos_in(directory: Path) -> list[Path]:
    """Git trees under ``directory``: at most :data:`CONTENT_MAX_REPOS`, sorted.

    The copy arm's in-pad reference source: a pad that holds a clone beside a
    plain copy of the same project can prove the copy against that clone.
    Bounded by the content walk's depth/entry caps AND by the same repository
    cap the walk uses, so a pad holding hundreds of repos cannot turn one proof
    into hundreds of ``ls-tree`` calls. Entries are sorted and pushed reversed
    so the DFS visits them in ascending name order: with a cap, WHICH
    repositories count must not depend on filesystem enumeration order.
    """
    repos: list[Path] = []
    stack: list[tuple[str, int]] = [(os.fspath(directory), 0)]
    seen = 0
    while stack and len(repos) < CONTENT_MAX_REPOS:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError:
            continue
        descend: list[tuple[str, int]] = []
        for entry in children:
            seen += 1
            if seen > CONTENT_MAX_ENTRIES:
                return repos
            try:
                if entry.is_symlink():
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name == ".git":
                    continue
                if os.path.lexists(os.path.join(entry.path, ".git")):
                    repos.append(Path(entry.path))
                    if len(repos) >= CONTENT_MAX_REPOS:
                        return repos
                    continue
            except OSError:
                continue
            if depth + 1 > CONTENT_MAX_DEPTH:
                continue
            descend.append((entry.path, depth + 1))
        # Push REVERSED so the DFS pops the first (ascending) name next: the
        # cap then selects the same repositories on every filesystem.
        stack.extend(reversed(descend))
    return repos


def _copy_shaped_children(children: Iterable[os.DirEntry[str]]) -> bool:
    """Whether any depth-1 entry is one the copy arm can reach (unsized).

    The gate MUST mirror the walk's own reachability (review 1). The walk
    keeps a refused-segment directory whole — but when its relative path has
    an evidence segment it still DESCENDS into it, so a copy under
    ``evidence-build/base-repo`` is attempted with this arm. A gate that
    stopped at the refused name resolved no references and silently skipped
    the pad (reproduced: a 55 MB provable copy left untouched). A depth-1
    directory is therefore copy-shaped when it is not a git tree and either is
    not a refused segment or carries an evidence segment.
    """
    from local_operator.scratchpad import _is_refused_segment  # LAZY: write vocabulary

    for entry in children:
        try:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            if entry.name == ".git":
                continue
            if os.path.lexists(os.path.join(entry.path, ".git")):
                continue
            if _is_refused_segment(entry.name) and not _has_evidence_segment(entry.name):
                continue
        except OSError:
            continue
        return True
    return False


def _has_copy_shaped_child(scratchpad: Path) -> bool:
    """Whether a scratchpad directory holds a child the copy arm could try.

    A cheap, UNSIZED gate used before any reference is resolved: a pad with no
    such child has nothing the arm could prove, so it must not pay the
    scratchpad walk and the ``ls-tree`` calls. See
    :func:`_copy_shaped_children` for the shape itself.
    """
    try:
        with os.scandir(scratchpad) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
    except OSError:
        return False
    return _copy_shaped_children(children)


def _reference_repos(scratchpad: Path, references: Iterable[str]) -> list[Path]:
    """Configured reference paths plus the repos found inside ``scratchpad``.

    Deduplicated by realpath, ordered configured-first so an operator's named
    references take effect even when a pad happens to carry its own clones.
    Only real directories are returned; a missing path costs nothing.
    """
    found: list[Path] = []
    seen: set[str] = set()
    for raw in (*references, *(os.fspath(path) for path in _repos_in(scratchpad))):
        try:
            real = os.path.realpath(raw)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        if os.path.isdir(real):
            found.append(Path(real))
    return found


@dataclass
class _HashCache:
    """Blob ids reused across one PAD's candidates, never across pads.

    WHY IT EXISTS: with recursive candidates a refused parent walks the same
    bytes its proving child will walk again (measured 2x on a 975 MB copy). The
    key carries size and mtime_ns as well as the realpath, so a file that
    changed between two candidate walks is re-hashed instead of being served a
    stale id. :data:`NON_GIT_COPY_HASH_CACHE` caps the memory; beyond it new
    digests are simply not retained (the walk keeps working, it just re-reads).
    """

    entries: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def blob(self, path: Path, size: int, mtime_ns: int) -> tuple[str, int]:
        """``(blob id, bytes actually read)``; a hit reads nothing.

        The second value is what the real pass charges to its hash budget: a
        recursive candidate walk re-offers the same files its refused parent
        already hashed, and charging those bytes again would defer work that
        cost nothing this pass (measured 2x on ``qa-1232``).
        """
        key = (os.path.realpath(path), size, mtime_ns)
        cached = self.entries.get(key)
        if cached is not None:
            return cached, 0
        cached = _blob_sha1(path, size)
        if len(self.entries) < NON_GIT_COPY_HASH_CACHE:
            self.entries[key] = cached
        return cached, size


@dataclass
class _CopyProof:
    """The per-file verdict for ONE copy candidate, before any removal."""

    total_bytes: int = 0
    entries: int = 0
    #: ``(absolute path, candidate-relative posix path, size)`` for proven files.
    matched: list[tuple[Path, str, int]] = field(default_factory=list)
    #: ``(relative posix path, size)`` for the rest.
    unmatched: list[tuple[str, int]] = field(default_factory=list)
    matched_bytes: int = 0
    #: The reference repositories that proved at least one file (realpaths), so
    #: a union proof says WHICH trees it used.
    proved_by: tuple[str, ...] = ()
    #: Bytes actually hashed (the real pass's budget is charged this, not the
    #: candidate's total: a path present in no tree is never read).
    hashed_bytes: int = 0
    #: Why the candidate is not reclaimed; empty means it is.
    reason: str = ""
    below_floor: bool = False
    nested_git: bool = False
    cap_exceeded: bool = False
    unreadable: bool = False
    deferred: bool = False

    @property
    def ok(self) -> bool:
        return not self.reason


#: The verdicts :func:`_copy_outcome` hands back. ``TAKEN`` and ``KEEP`` stop
#: the walk at this directory; ``DESCEND`` tries its children (the largest
#: provable subtree wins, top-down); ``BLOCKED`` descends but disables the copy
#: arm below (the merged-clean arm owns a nested repository).
_COPY_TAKEN = "taken"
_COPY_KEEP = "keep"
_COPY_DESCEND = "descend"
_COPY_BLOCKED = "blocked"
_COPY_BELOW = "below"


def _copy_outcome(proof: _CopyProof) -> str:
    """Map a proof to the walk's next move (see the verdict constants).

    A directory that proves is TAKEN whole-subtree file-by-file. A refusal due
    to an unreadable file/dir or an exceeded cap is KEPT and NOT descended: the
    walk could not look at it completely, and a partial guess (or a descent
    past the pad's own entry cap) is exactly what this arm never does. A share
    or reference refusal DESCENDS, because the copy may sit one level down
    (measured: ``qa-1232/`` matches nothing, ``qa-1232/base-repo/`` matches
    956 of 975 MB). A nested repository BLOCKS the arm below it.
    """
    if proof.ok:
        return _COPY_TAKEN
    if proof.nested_git:
        return _COPY_BLOCKED
    if proof.cap_exceeded or proof.unreadable or proof.deferred:
        return _COPY_KEEP
    return _COPY_DESCEND


def _prove_copy(
    candidate: Path,
    *,
    trees: tuple[tuple[str, dict[str, str]], ...],
    cache: _HashCache | None = None,
    budget: int | None = None,
    min_bytes: int = NON_GIT_COPY_MIN_BYTES,
) -> _CopyProof:
    """Walk one candidate and prove its files against the merged reference trees.

    ``trees`` is a UNION: a file proves when ANY (repository, merged tree) pair
    has its relative path at its blob id, and the share is computed against the
    union. That is the measured norm — most 50-155 MB copies on the real store
    are of a DIFFERENT repository than the first one an operator names.

    Raises :class:`NonGitCopyScanError` for the VACUOUS case — the walk saw
    entries but zero regular files. That is not "nothing to reclaim": it is the
    instrument failing (measured: a pipeline reported "0 files in a 1.0 GB
    copy" and a reader took it as success). Every other failure is a REFUSAL
    recorded on the returned proof, never an exception.
    """
    proof = _CopyProof()
    files: list[tuple[Path, str, int, int]] = []
    stack: list[str] = [os.fspath(candidate)]
    entries = 0
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                children = list(scan)
        except OSError:
            proof.unreadable = True
            proof.reason = f"cannot read {os.path.relpath(current, candidate)}"
            return proof
        for entry in children:
            entries += 1
            if entries > NON_GIT_COPY_MAX_ENTRIES:
                proof.cap_exceeded = True
                proof.reason = f"more than {NON_GIT_COPY_MAX_ENTRIES} entries"
                return proof
            try:
                if entry.is_symlink():
                    continue  # never followed: a link is not proof of a copy
                if entry.name == ".git":
                    # A repository anywhere below makes this NOT a plain copy.
                    # The merged-clean arm judges it; the copy arm must never
                    # delete a repository that arm would keep.
                    proof.nested_git = True
                    proof.reason = "contains a git repository"
                    return proof
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue  # fifos, sockets: never touched
                if len(files) >= NON_GIT_COPY_MAX_FILES:
                    proof.cap_exceeded = True
                    proof.reason = f"more than {NON_GIT_COPY_MAX_FILES} files"
                    return proof
                info = entry.stat(follow_symlinks=False)
                relative = os.path.relpath(entry.path, candidate).replace(os.sep, "/")
                files.append((Path(entry.path), relative, info.st_size, info.st_mtime_ns))
                proof.total_bytes += info.st_size
            except OSError:
                proof.unreadable = True
                proof.reason = f"cannot read {entry.name}"
                return proof
    proof.entries = entries
    if entries and not files:
        # THE GUARD IS ABOUT BYTES, NOT ENTRIES (found on the real store: a
        # scratch HOME's empty ``Library`` scaffold has dozens of empty dirs and
        # zero bytes — a legitimate tree, not an impossible result). Our own
        # walk computes files and bytes together, so a contradiction has to
        # come from a SECOND source: measure the tree independently and raise
        # only when it holds bytes that the walk did not see as files (the
        # measured incident: "0 files in a 1.0 GB copy"). Zero bytes is an empty
        # scaffold: below the floor, descended into, never an error.
        measured = _bounded_dir_stats(candidate, max_entries=NON_GIT_COPY_MAX_ENTRIES)[0]
        if measured > 0:
            raise NonGitCopyScanError(
                f"{candidate}: walk produced zero files from a non-empty tree "
                f"({measured} bytes measured independently)"
            )
        proof.below_floor = True
        return proof
    if proof.total_bytes < min_bytes:
        proof.below_floor = True
        return proof
    if not trees:
        proof.reason = "no reference repository"
        return proof
    if budget is not None and proof.total_bytes > budget:
        proof.deferred = True
        proof.reason = "hash budget"
        return proof
    # THE BEST CASE THIS CANDIDATE CAN REACH is the bytes of files whose PATH
    # appears in at least one merged tree; a file at an absent path can never
    # match. When even that bound cannot clear the floor, refuse WITHOUT
    # reading: the verdict is identical and the reads (measured: gigabytes on
    # this store's refused copies) are pure cost.
    hashable_bytes = sum(
        size
        for _absolute, relative, size, _mtime in files
        if any(relative in tree for _, tree in trees)
    )
    if hashable_bytes < NON_GIT_COPY_MIN_BYTES or (
        hashable_bytes < NON_GIT_COPY_MIN_SHARE * proof.total_bytes
    ):
        # ONE refusal phrase for both bounds (design C3): the early bound can
        # only ever say an UPPER bound, so it says "at most" and then the same
        # "match a merged tree" a reader sees on the hashed verdict below.
        proof.reason = (
            f"at most {_percent(hashable_bytes, proof.total_bytes)} of its bytes "
            f"({_mb(hashable_bytes)} of {_mb(proof.total_bytes)}) match a merged tree"
        )
        return proof
    proved: set[str] = set()
    for absolute, relative, size, mtime_ns in files:
        expected = [(name, tree[relative]) for name, tree in trees if relative in tree]
        if not expected:
            proof.unmatched.append((relative, size))
            continue
        try:
            if cache is not None:
                blob, read = cache.blob(absolute, size, mtime_ns)
            else:
                blob, read = _blob_sha1(absolute, size), size
        except OSError:
            proof.unreadable = True
            proof.reason = f"cannot read {relative}"
            return proof
        # ONLY ACTUAL READS ARE CHARGED (review A3): a cache hit is free, so a
        # refused parent that already hashed a subtree does not make its
        # proving child defer on budget it did not spend.
        proof.hashed_bytes += read
        if any(objectname == blob for _, objectname in expected):
            for name, objectname in expected:
                if objectname == blob:
                    proved.add(name)
            proof.matched.append((absolute, relative, size))
            proof.matched_bytes += size
        else:
            proof.unmatched.append((relative, size))
    proof.proved_by = tuple(sorted(proved))
    share = proof.matched_bytes / proof.total_bytes if proof.total_bytes else 0.0
    if share < NON_GIT_COPY_MIN_SHARE or proof.matched_bytes < NON_GIT_COPY_MIN_BYTES:
        proof.reason = (
            f"only {_percent(proof.matched_bytes, proof.total_bytes)} of its bytes "
            f"({_mb(proof.matched_bytes)} of {_mb(proof.total_bytes)}) match a merged tree"
        )
    return proof


@dataclass
class _CopyArm:
    """One pass's copy-arm state: references, merged-tree cache and hash budget."""

    references: tuple[str, ...] = ()
    #: Whether any reference was CONFIGURED (as opposed to found in a pad) —
    #: distinguishes "nothing to prove against because nobody named one" from
    #: "the operator configured references and they did not qualify".
    configured: bool = False
    #: Remaining hash budget in bytes; ``None`` = unbudgeted (a dry run).
    budget: int | None = None
    #: ``(realpath, trunk sha) -> merged tree``, per pass.
    merged: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    resolved: dict[str, tuple[str, dict[str, str]] | None] = field(default_factory=dict)
    pad_trees: dict[str, tuple[tuple[str, dict[str, str]], ...]] = field(default_factory=dict)
    qualified: set[str] = field(default_factory=set)
    #: ``pad path -> reason`` when a reference was refused for its OBJECT
    #: FORMAT (design D4). Kept so a pad that proves nothing can say WHY its
    #: references did not qualify instead of a generic "no reference".
    pad_refusals: dict[str, str] = field(default_factory=dict)
    budget_exhausted: bool = False
    #: Session ids the operator reserved from this arm (config
    #: ``non_git_copies.hold_sessions``). A hold outranks proof: a held pad's
    #: candidates are refused before any hashing.
    holds: tuple[str, ...] = ()
    hash_cache: _HashCache = field(default_factory=_HashCache)

    def begin_pad(self) -> None:
        """Drop the per-pad hash cache; the next pad starts clean."""
        self.hash_cache = _HashCache()

    def trees(self, pad: Path) -> tuple[tuple[str, dict[str, str]], ...]:
        key = os.fspath(pad)
        cached = self.pad_trees.get(key)
        if cached is None:
            found: list[tuple[str, dict[str, str]]] = []
            refusal = ""
            for repo in _reference_repos(pad / "scratchpad", self.references):
                real = os.path.realpath(os.fspath(repo))
                format_refusal = _reference_format_refusal(real)
                if format_refusal is not None:
                    refusal = refusal or format_refusal
                    continue
                merged = self._tree_for(real)
                if merged is not None:
                    found.append((real, merged))
            cached = tuple(found)
            self.pad_trees[key] = cached
            if refusal:
                self.pad_refusals[key] = refusal
        return cached

    def pad_refusal(self, pad: Path) -> str:
        """Why the pad's references were unusable, or "" (design D4)."""
        return self.pad_refusals.get(os.fspath(pad), "")

    def _tree_for(self, repo: str) -> dict[str, str] | None:
        real = os.path.realpath(repo)
        if real in self.resolved:
            resolved = self.resolved[real]
        else:
            resolved = _merged_tree(real, format_checked=True)
            self.resolved[real] = resolved
        if resolved is None:
            return None
        trunk, tree = resolved
        self.qualified.add(real)
        cached = self.merged.get((real, trunk))
        if cached is not None:
            return cached
        self.merged[(real, trunk)] = tree
        return tree

    def spend(self, hashed: int) -> None:
        if self.budget is not None:
            self.budget = max(0, self.budget - hashed)


def _plan_copy(path: str, pad: Path, plan: _PadPlan, arm: _CopyArm, *, counted: bool) -> str:
    """Rule (d) for ONE directory the walk has reached. Returns a verdict.

    The walk offers every non-git directory it reaches, top-down, so the largest
    provable subtree wins: a directory that proves is planned file-by-file and
    its children are not walked; a refused or sub-threshold one is descended
    into, and each child is offered in turn (measured: ``qa-1232/`` matches
    nothing while ``qa-1232/base-repo/`` matches 956 of 975 MB).

    ``counted`` is False when an ANCESTOR was already attempted: the reclaimed
    bytes still count, but the candidate total names the top-most directories an
    operator sees, not every nested attempt (whose bytes a parent already
    included).

    Evidence-named segments are NOT exempt at the candidate test: the exemption
    exists because the shape arm matches by NAME, while this arm proves every
    byte, and a copy's ``docs/evidence`` is tracked content like any other.
    """
    candidate = Path(path)
    name = candidate.name
    # Hash-budget deferral is charged before the walk's own cost is known: the
    # proof returns a deferred verdict, which this records and does not descend
    # past (the candidate is retried next pass).
    proof = _prove_copy(candidate, trees=arm.trees(pad), cache=arm.hash_cache, budget=arm.budget)
    arm.spend(proof.hashed_bytes)
    if proof.below_floor:
        return _COPY_BELOW
    if counted:
        plan.copy_candidates += 1
        plan.copy_candidate_bytes += proof.total_bytes
    outcome = _copy_outcome(proof)
    if outcome == _COPY_TAKEN:
        for absolute, _relative, size in proof.matched:
            plan.removals.append(_PlannedRemoval(absolute, CONTENT_CLASS_COPY, size, 1))
        if proof.unmatched:
            examples = ", ".join(relative for relative, _ in proof.unmatched[:3])
            kept = proof.total_bytes - proof.matched_bytes
            plan.kept.append(
                f"{name}: {len(proof.unmatched)} unmatched file(s), {_mb(kept)} kept "
                f"(e.g. {examples})"
            )
        return _COPY_TAKEN
    if outcome == _COPY_BLOCKED:
        plan.kept.append(f"{name}: {proof.reason} (the merged-clean arm judges it)")
        return _COPY_BLOCKED
    if outcome == _COPY_KEEP:
        if proof.deferred:
            arm.budget_exhausted = True
            plan.copy_deferred += 1
            plan.kept.append(f"{name}: {proof.reason} (deferred to the next pass)")
        else:
            plan.kept.append(f"{name}: {proof.reason}")
        return _COPY_KEEP
    plan.kept.append(f"{name}: {proof.reason}")
    return _COPY_DESCEND


def _tree_bytes_at_least(path: Path, floor: int) -> int:
    """Bytes under ``path``, stopping as soon as ``floor`` is reached.

    A sanity instrument, not an accounting figure: the store-level vacuous
    guard only needs to know whether the store is at least as large as the
    floor, so this never walks a whole multi-terabyte tree.
    """
    total = 0
    stack = [os.fspath(path)]
    while stack and total < floor:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                children = list(entries)
        except OSError:
            continue
        for entry in children:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    if total >= floor:
                        return total
            except OSError:
                continue
    return total


def _session_age_hours(session_dir: Path, moment: float) -> float | None:
    """Hours since the session's last activity, or ``None`` for no usable clock.

    The same clock chain the record pass uses (activity path, then
    ``_clock_without_activity``'s birth-stamp fallbacks); a ``None`` here means
    the report cannot say the session is past the window, and it is reported as
    inside it (the conservative direction).
    """
    clock = session_activity_path(os.fspath(session_dir))
    if clock is None:
        clock = _clock_without_activity(session_dir, moment)
    if clock is None or clock > moment:
        return None
    return max(0.0, (moment - clock) / 3600.0)


def scan_non_git_copies(
    sessions_dir: Path,
    *,
    references: Iterable[str] = (),
    #: Session ids to hold. ``None`` means the CONFIGURED list was unreadable:
    #: the report refuses to guess and returns ``skipped="holds unreadable"``.
    holds: Iterable[str] | None = (),
    now: float | None = None,
    window_hours: float = float(DEFAULT_DELEGATED_MAX_AGE_HOURS),
    min_bytes: int = NON_GIT_COPY_MIN_BYTES,
    min_store_bytes: int = NON_GIT_COPY_STORE_MIN_BYTES,
) -> NonGitCopyReport:
    """The READ-ONLY, whole-store class report for plain non-git directory copies.

    It is the CLI's ``--report-non-git-copies`` instrument and the arm's own
    candidate enumeration: every ``sessions/*/scratchpad`` is walked top-down,
    with NO age or liveness gate for enumeration (the class exists whether or
    not a pass would reach it now). Each candidate directory gets the SAME
    :func:`_prove_copy` the pass uses, against the UNION of the configured
    references and the pad's own repositories, and one of four rows:

    * ``would reclaim`` — the proof passed AND the session is past the window;
    * ``proof ok, inside the window`` — provable, but young (the measured class
      is 12-30 h old against a 48 h default, so a report that hid these would
      say "nothing to do" while 9 GB sits there);
    * kept with a reason (share too low, no reference, a nested repository,
      unreadable, over a cap, a pad root the pass refuses, an operator hold);
    * the same, for a session the pass may not touch (non-delegated origin).

    IT APPLIES THE PAD-ROOT GUARDS TOO (review B1): a pad whose ``scratchpad/``
    root is itself a git tree, or whose session directory is, is a row the PASS
    would keep whole — the report must not call it ``would reclaim``.

    THE STORE-LEVEL VACUOUS GUARD LIVES HERE, and it means "the scan could not
    see the store", never "the store has no copies": a store with no reachable
    scratchpad directory that holds at least ``min_store_bytes`` (default
    25 GiB) is the instrument failing, and it is sized ONLY on that path. A
    store whose pads were scanned and genuinely hold no plain copies is an
    honest empty report. An unreadable ``sessions/`` always raises: returning
    an empty report would present "nothing to reclaim" for a store nobody
    looked at. This is a sanity instrument, not a policy: it removes nothing
    and writes nothing.
    """
    moment = time.time() if now is None else now
    configured = tuple(
        os.path.expanduser(part) for part in references if isinstance(part, str) and part.strip()
    )
    report = NonGitCopyReport(window_hours=float(window_hours), references=configured)
    if holds is None:
        # The caller could not read the hold list. Do not scan: an empty hold
        # set here would present held items as reclaimable.
        report.skipped = "holds unreadable"
        return report
    hold_set = {hold.strip() for hold in holds if isinstance(hold, str) and hold.strip()}
    from local_operator.scratchpad import (  # LAZY: the write path's vocabulary
        _is_refused_segment,
    )

    pads: list[Path] = []
    try:
        with os.scandir(sessions_dir) as entries:
            sessions = [entry for entry in entries if entry.is_dir(follow_symlinks=False)]
    except FileNotFoundError:
        # A FRESH INSTALL has no sessions directory yet: an honest empty
        # report, not an error. Only a store that EXISTS and cannot be read is
        # the instrument failing (review 2).
        report.skipped = "no store"
        return report
    except OSError as exc:
        # AN UNREADABLE STORE IS NEVER AN EMPTY STORE (review B3a): "nothing to
        # reclaim" for a directory nobody could list is the silent-success
        # shape this guard exists for.
        raise NonGitCopyScanError(
            f"store cannot be read: {sessions_dir} ({exc.__class__.__name__})"
        ) from exc
    for session in sessions:
        if session.name.startswith("."):
            continue
        pad = Path(session.path) / "scratchpad"
        if os.path.isdir(pad) and not os.path.islink(pad):
            pads.append(pad)
    # ONE report-level merged-tree cache: a configured reference is the same
    # tree for every pad (2,900 pads x 2 references was 5,800 `git ls-tree`
    # subprocesses), and an in-pad repository is resolved once for the pad that
    # carries it. Measured 2026-10-20: the uncached report took 22 min cold and
    # over 30 min warm on a 13 GB class over 2,900 pads.
    merged_cache: dict[str, tuple[str, dict[str, str]] | None] = {}

    def resolve(repo: str) -> tuple[str, dict[str, str]] | None:
        real = os.path.realpath(repo)
        if real not in merged_cache:
            merged_cache[real] = _merged_tree(real, format_checked=True)
        return merged_cache[real]

    configured_trees: list[tuple[str, dict[str, str]]] = []
    configured_refusal = ""
    for repo in configured:
        refusal = _reference_format_refusal(repo)
        if refusal is not None:
            configured_refusal = configured_refusal or refusal
            continue
        resolved = resolve(repo)
        if resolved is not None:
            configured_trees.append((os.path.realpath(repo), resolved[1]))
    qualified = bool(configured_trees)

    for pad in pads:
        session_dir = pad.parent
        age_hours = _session_age_hours(session_dir, moment)
        within_window = age_hours is None or age_hours <= float(window_hours)
        delegated = _is_delegated_dir(session_dir)
        deletable = delegated and _is_deletable_dir(session_dir)
        held = session_dir.name in hold_set
        # THE PAD-ROOT GUARDS, applied here exactly as the pass applies them
        # (review B1): a report row that says "would reclaim" for a pad the pass
        # keeps whole is a false action.
        root_guard = ""
        if os.path.lexists(os.path.join(pad, ".git")):
            root_guard = "the scratchpad root is itself a git tree"
        elif os.path.lexists(os.path.join(session_dir, ".git")):
            root_guard = "the session directory is itself a git tree"
        # THE CHEAP GATE FIRST: a pad with no depth-1 candidate shape needs no
        # in-pad repository walk at all. That walk (up to CONTENT_MAX_ENTRIES)
        # for every pad was the other half of the 30-minute cost. Scan once and
        # KEEP the failure: an unreadable pad used to vanish silently
        # (candidates=0, errors=0, exit 0), so the operator never learned the
        # scan could not look at it (review 3).
        try:
            with os.scandir(pad) as entries:
                pad_children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            report.errors += 1
            report.candidates.append(
                NonGitCopyCandidate(
                    session=session_dir.name,
                    path=os.path.relpath(pad, sessions_dir).replace(os.sep, "/"),
                    error=True,
                    reason=f"scratchpad cannot be read ({exc.__class__.__name__})",
                )
            )
            continue
        if not _copy_shaped_children(pad_children):
            continue
        trees = list(configured_trees)
        pad_refusal = configured_refusal
        if not held:
            seen = {name for name, _ in trees}
            for repo in _repos_in(pad):
                real = os.path.realpath(os.fspath(repo))
                if real in seen:
                    continue
                format_refusal = _reference_format_refusal(real)
                if format_refusal is not None:
                    pad_refusal = pad_refusal or format_refusal
                    continue
                resolved = resolve(real)
                if resolved is None:
                    continue
                seen.add(real)
                trees.append((real, resolved[1]))
                qualified = True
        trees = tuple(trees)
        cache = _HashCache()
        stack: list[tuple[str, int, bool, bool]] = [(os.fspath(pad), 0, False, True)]
        while stack:
            current, depth, ancestor, allowed = stack.pop()
            try:
                with os.scandir(current) as entries:
                    children = sorted(entries, key=lambda entry: entry.name)
            except OSError:
                continue  # unreadable dir: its parent already recorded the keep
            for entry in children:
                try:
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        continue
                    if entry.name == ".git":
                        continue
                    if os.path.lexists(os.path.join(entry.path, ".git")):
                        continue  # the merged-clean arm's own candidate
                    if _is_refused_segment(entry.name):
                        relative = os.path.relpath(entry.path, os.fspath(pad))
                        if not _has_evidence_segment(relative):
                            continue  # the shape arm takes it whole
                        # A REFUSED EVIDENCE NAME IS STILL DESCENDED INTO: the
                        # pass keeps the directory and lets the copy arm try
                        # its children. Mirror that here or the report would
                        # hide the very copy the pass reclaims (review 1).
                        if not allowed or depth >= CONTENT_MAX_DEPTH:
                            continue
                        stack.append((entry.path, depth + 1, ancestor, allowed))
                        continue
                except OSError:
                    continue
                if not allowed or depth >= CONTENT_MAX_DEPTH:
                    continue
                counted = not ancestor
                try:
                    if held:
                        # A HOLD IS STRUCTURAL: the proof is not even attempted
                        # (the measured held copy is 975 MB), and the row must
                        # read as a hold, never as a would-reclaim or as a proof
                        # that happened to fail.
                        proof = _prove_copy(Path(entry.path), trees=(), min_bytes=min_bytes)
                    else:
                        proof = _prove_copy(
                            Path(entry.path), trees=trees, cache=cache, min_bytes=min_bytes
                        )
                except NonGitCopyScanError as exc:
                    # ONE BROKEN CANDIDATE MUST NOT KILL THE STORE REPORT: record
                    # a row naming the failure, count it, and keep scanning. The
                    # CLI exits 3 on ``errors`` after printing the rest.
                    report.errors += 1
                    report.candidates.append(
                        NonGitCopyCandidate(
                            session=session_dir.name,
                            path=os.path.relpath(entry.path, sessions_dir).replace(os.sep, "/"),
                            held=held,
                            top_level=counted,
                            error=True,
                            reason=str(exc),
                        )
                    )
                    continue
                if proof.below_floor:
                    stack.append((entry.path, depth + 1, ancestor, True))
                    continue
                if held:
                    outcome = _COPY_KEEP
                    proof_ok = False
                    waiting = False
                    would = False
                    reason = "operator hold"
                else:
                    outcome = _copy_outcome(proof)
                    proof_ok = proof.ok
                    # WAITING means the PASS would take it as soon as the
                    # window passes: a non-delegated or non-reapable session
                    # never would, so it is a keep with its own reason (review
                    # 4). Otherwise the waiting summary counted rows the pass
                    # can never act on.
                    waiting = (
                        proof_ok and within_window and not root_guard and delegated and deletable
                    )
                    would = proof_ok and not within_window and deletable and not root_guard
                    if root_guard:
                        # The pass keeps the whole pad before any copy proof;
                        # the row is a keep, never a would-reclaim (review B1).
                        reason = root_guard
                    elif not proof_ok:
                        reason = proof.reason
                        if reason == "no reference repository" and pad_refusal:
                            reason = pad_refusal
                    elif not delegated:
                        reason = "not a delegated session"
                    elif not deletable:
                        reason = "origin is not reapable"
                    elif within_window:
                        reason = f"proof ok, inside the {float(window_hours):g}h window"
                    else:
                        reason = ""
                report.candidates.append(
                    NonGitCopyCandidate(
                        session=session_dir.name,
                        path=os.path.relpath(entry.path, sessions_dir).replace(os.sep, "/"),
                        bytes=proof.total_bytes,
                        matched_bytes=proof.matched_bytes,
                        unmatched_files=len(proof.unmatched),
                        unmatched_bytes=proof.total_bytes - proof.matched_bytes,
                        examples=tuple(relative for relative, _ in proof.unmatched[:3]),
                        proved_by=proof.proved_by,
                        age_hours=age_hours,
                        within_window=within_window,
                        waiting=waiting,
                        held=held,
                        proof_ok=proof_ok,
                        top_level=counted,
                        would_reclaim=would,
                        reason=reason,
                    )
                )
                if outcome == _COPY_DESCEND:
                    stack.append((entry.path, depth + 1, True, True))
                elif outcome == _COPY_BLOCKED:
                    stack.append((entry.path, depth + 1, True, False))
    if not report.candidates and not pads:
        # THE STORE-LEVEL VACUOUS GUARD means "the scan could not SEE the
        # store" (review B3b), not "the store has no copies": zero scratchpad
        # directories were reachable over a store that holds the floor's bytes.
        # A pad that WAS scanned and holds no plain copies is an honest empty
        # report above, never an exit 3.
        report.store_bytes = _tree_bytes_at_least(sessions_dir, min_store_bytes)
        if report.store_bytes >= min_store_bytes:
            raise NonGitCopyScanError(
                "store scan could not see any scratchpad directory over a >= "
                f"{min_store_bytes // (1024 * 1024 * 1024)} GB store"
            )
        return report
    if not configured and not qualified:
        report.skipped = "no reference repository"
    return report


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
    #: The non-git COPY arm's counters for this pad (zero when the arm is off);
    #: rolled into :class:`DelegatedResult` by the caller.
    copy_candidates: int = 0
    copy_candidate_bytes: int = 0
    copy_reclaimed_bytes: int = 0
    copy_deferred: int = 0


@dataclass(frozen=True)
class _ContentGuard:
    """Why a pad's content may not be taken, and the clock the guard measured.

    ``reason`` is ``None`` only when the pad may be planned; ``activity`` is the
    transcript clock's value at guard time (retention's ONE clock: transcript or
    spooled mail), carried so the log row can record how old the session was
    when the pass decided it was old enough.
    """

    reason: str | None = None
    activity: float | None = None

    def age_hours(self, now: float) -> float | None:
        if self.activity is None:
            return None
        return max(0.0, (now - self.activity) / 3600.0)


def _content_guard(
    cand: _Cand, *, config_dir: Path, now: float, cutoff: float, live_resolved: Path | None
) -> _ContentGuard:
    """The content phase's OWN guards: why this pad must not be touched at all.

    Narrower than the record pass's :func:`_evaluate` BY DESIGN — the carve-out
    is the point: parent-still-active, open projects and the scratchpad git
    hazard keep the RECORD; only the current session, a live claim or lease, an
    armed wake or monitor, unread spooled mail, open asks and the REAP predicate
    refuse CONTENT. The mail and ask guards are the 2026-10-19 incident
    requirement: a pad that can still hand the user a question, or mail they
    have not read, outranks reclaiming cache-shaped bytes.

    THE AGE GATE IS RESTORED (2026-10-19). The policy that removed it reclaimed
    a build shape from a transcript touched an hour ago as long as no claim,
    lease or wake was visible — live-adjacent content, and a contradiction of
    the incident requirement: never remove content from a session with a live
    runtime or one whose transcript was touched inside the retention window. A
    session with no activity clock is KEPT (fail closed): "cannot tell it is
    old" is not "old". The per-command pad cap and the watchdog bound live
    growth; reclamation serves a session PAST the window whose record is still
    kept by a parent, an open project or the git guard — the transcript stays,
    only cache-shaped content goes. The re-stat also judges against the PASS
    CLOCK (``now``) first: a session whose activity moved after the scan is
    live work in progress whatever its age. The caller wraps this call: a probe
    that raises keeps the pad and counts an error (fail closed).
    """
    fresh = session_activity_path(os.fspath(cand.path))
    if fresh is not None and fresh >= now:
        return _ContentGuard("active since the scan", fresh)
    if live_resolved is not None:
        try:
            if cand.path.resolve() == live_resolved:
                return _ContentGuard("the current session", fresh)
        except OSError:
            return _ContentGuard("cannot resolve path", fresh)
    if _claimed(cand.path, now):
        return _ContentGuard("claimed by a live process", fresh)
    if _lease_runtime_alive(cand.path):
        return _ContentGuard("leased by a live process", fresh)
    if _has_armed_wake(config_dir, cand.path.name):
        return _ContentGuard("armed wake for this session", fresh)
    if _has_armed_monitor(config_dir, cand.path.name):
        return _ContentGuard("armed monitor for this session", fresh)
    # The incident requirement: a pad that can still hand the user a question
    # or mail is untouchable, not only one with an armed surface. Both probes
    # fail closed (an unreadable ask log or inbox answers "yes").
    if _has_spooled_mail(cand.path):
        return _ContentGuard("has unread spooled mail", fresh)
    if _has_open_asks(cand.path):
        return _ContentGuard("has open asks", fresh)
    if not _is_deletable_dir(cand.path):
        return _ContentGuard("origin is not reapable", fresh)
    if fresh is None:
        return _ContentGuard("no activity clock", None)
    if fresh >= cutoff:
        return _ContentGuard("inside the retention window", fresh)
    return _ContentGuard(None, fresh)


def _manifest_path(sessions_dir: Path, session: str, now: float) -> Path:
    """This pad's manifest location: UTC-stamped so runs sort chronologically."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    return sessions_dir / CONTENT_MANIFEST_DIRNAME / f"{stamp}-{session}.json"


def _write_cleanup_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` atomically (temp beside the target, then rename).

    The temp file shares the directory so the rename is same-filesystem and
    atomic; a reader either sees the previous manifest or the new one, never a
    half-written JSON file. Raises ``OSError`` (or a JSON encoding error) for
    the caller to record and continue: a manifest failure never fails a
    removal the operator asked for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def _manifest_classes(removals: Iterable[_PlannedRemoval]) -> tuple[str, ...]:
    seen = {removal.cls for removal in removals}
    return tuple(
        cls
        for cls in (
            CONTENT_CLASS_MERGED,
            CONTENT_CLASS_BUILD,
            CONTENT_CLASS_STALE,
            CONTENT_CLASS_COPY,
        )
        if cls in seen
    )


def _manifest_entries(removals: Iterable[_PlannedRemoval]) -> list[dict[str, Any]]:
    return [
        {
            "path": os.fspath(removal.path),
            "bytes": removal.bytes,
            "class": removal.cls,
            "method": removal.method,
        }
        for removal in removals
    ]


def _manifest_payload(
    cand: _Cand,
    *,
    now: float,
    actor: str,
    trigger: str,
    planned: list[_PlannedRemoval],
    completed: list[_PlannedRemoval],
) -> dict[str, Any]:
    """The manifest body: every planned path, then the subset actually taken.

    The top-level ``classes``/``bytes``/``entries`` describe the list that is
    meaningful at write time: the PLAN before the first removal, the COMPLETED
    subset after the loop. ``planned`` always holds the full plan, so a process
    that dies mid-pass leaves an attributable record of everything it meant to
    take.
    """
    described = completed if completed else planned
    classes = _manifest_classes(described)
    return {
        "schema": CONTENT_MANIFEST_SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "session": cand.path.name,
        "origin": _origin(cand.path),
        "actor": actor,
        "trigger": trigger,
        "policy": DELEGATED_SCRATCHPAD_POLICY,
        "reason": " + ".join(classes),
        "classes": list(classes),
        "bytes": sum(removal.bytes for removal in described),
        "entries": sum(removal.entries for removal in described),
        "planned": _manifest_entries(planned),
        "completed": _manifest_entries(completed),
    }


def _reclaim_pad_content(
    cand: _Cand,
    *,
    config_dir: Path,
    now: float,
    cutoff: float,
    live_resolved: Path | None,
    sessions_dir: Path,
    dry_run: bool,
    actor: str,
    trigger: str,
    copy_arm: "_CopyArm | None" = None,
) -> _ContentOutcome:
    """Evaluate — and in a real pass execute — the content reclaim for ONE pad.

    The hard guards run first; a refusal comes back as ``kept`` and nothing
    below is touched. Then the pad is planned (bounded walk, no writes); then,
    in a real pass, the manifest's ``planned`` list is written BEFORE the first
    removal (so a crash mid-pass still names every path the pass meant to
    take), each planned removal goes through
    ``cleanup.remove_scratchpad_entry`` (the one guarded removal shape) —
    except a linked worktree's, which goes through ``git worktree remove`` in
    its shared repository (its objects are not the pad's to bundle) — and the
    manifest is rewritten with the ``completed`` subset after the loop. A
    rescue bundle is written and verified BEFORE the tree it belongs to is
    removed, PER TREE: any bundle failure keeps only its own tree (that planned
    removal is dropped and counted) while the pad's other removals proceed. A
    refusal or failure of the git removal keeps that tree and counts an error
    the same way. A manifest write failure never fails the removal; the log row
    carries ``manifest_error`` so the loss is still attributable as far as it
    can be.
    """
    outcome = _ContentOutcome()
    guard = _content_guard(
        cand, config_dir=config_dir, now=now, cutoff=cutoff, live_resolved=live_resolved
    )
    if guard.reason is not None:
        outcome.kept.append(guard.reason)
        return outcome
    try:
        plan = _plan_pad_content(cand.path, now=now, copy_arm=copy_arm)
    except NonGitCopyScanError as exc:
        # The instrument produced an impossible result: the pad is KEPT and the
        # message reaches ``content_kept``, never a silent zero-removal plan.
        raise _PadAbort(str(exc)) from exc
    except OSError as exc:  # the walk could not complete: keep, count it
        raise _PadAbort(f"scratchpad cannot be inspected ({exc.__class__.__name__})") from exc
    if plan is None:
        return outcome
    outcome.kept.extend(plan.kept)
    # Rolled even when nothing was planned: a share-refused candidate is a keep
    # the CLI's report block must count, and its zero removals are the point.
    outcome.copy_candidates = plan.copy_candidates
    outcome.copy_candidate_bytes = plan.copy_candidate_bytes
    outcome.copy_deferred = plan.copy_deferred
    if not plan.removals:
        return outcome
    manifest_path = _manifest_path(sessions_dir, cand.path.name, now)
    manifest_error = ""
    # ``manifest_written``: a file exists to point at (the planned write, or a
    # later successful rewrite). ``completed_written``: that file's
    # ``completed`` list is the real one. They differ when the planned write
    # succeeded and the completed rewrite failed — the row must then carry the
    # taken paths itself, because the manifest on disk says completed=[]
    # (review m1).
    manifest_written = False
    completed_written = False
    if not dry_run:
        try:
            _write_cleanup_manifest(
                manifest_path,
                _manifest_payload(
                    cand,
                    now=now,
                    actor=actor,
                    trigger=trigger,
                    planned=plan.removals,
                    completed=[],
                ),
            )
            manifest_written = True
        except Exception as exc:
            manifest_error = f"planned: {exc.__class__.__name__}"
            logger.warning("session cleanup: cannot write manifest %s: %s", manifest_path, exc)
    taken: list[_PlannedRemoval] = []
    bundles: list[tuple[str, int]] = []
    for removal in plan.removals:
        if dry_run:
            taken.append(removal)
            continue
        if removal.bundle_path is not None:
            # PER-TREE, and before ITS OWN tree may go (review F1): one pad may
            # carry several reclaimable clones, so a failure here keeps only
            # its own tree while the pad's other removals proceed.
            failure = _write_rescue_bundle(removal.path, removal.bundle_path, removal.bundle_refs)
            if failure is not None:
                outcome.errors += 1
                outcome.kept.append(f"{os.path.basename(os.fspath(removal.path))}: {failure}")
                logger.warning("session cleanup: %s; keeping the tree", failure)
                continue
            try:
                bundle_size = os.stat(removal.bundle_path).st_size
            except OSError:  # pragma: no cover — it was just written
                bundle_size = 0
            bundles.append((os.fspath(removal.bundle_path), bundle_size))
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
    # THE COUNTER IS WHAT WAS TAKEN, NOT WHAT WAS PLANNED (review B2): a pad
    # whose files are read-only reports bytes that actually left, not the
    # planned figure (measured: 57,671,686 planned against 6 bytes taken).
    outcome.copy_reclaimed_bytes = sum(
        removal.bytes for removal in taken if removal.cls == CONTENT_CLASS_COPY
    )
    if not taken:
        # The planned manifest (if written) stays: it names what the pass meant
        # to take, which is the honest record when every removal was kept
        # (a failed bundle, a refused entry) and nothing was logged.
        return outcome
    classes = _manifest_classes(taken)
    if not dry_run:
        try:
            _write_cleanup_manifest(
                manifest_path,
                _manifest_payload(
                    cand,
                    now=now,
                    actor=actor,
                    trigger=trigger,
                    planned=plan.removals,
                    completed=taken,
                ),
            )
            manifest_written = True
            completed_written = True
        except Exception as exc:
            suffix = f"completed: {exc.__class__.__name__}"
            manifest_error = f"{manifest_error}; {suffix}" if manifest_error else suffix
            logger.warning("session cleanup: cannot rewrite manifest %s: %s", manifest_path, exc)
    planned_bundles = [os.fspath(removal.bundle_path) for removal in taken if removal.bundle_path]
    written = dict(bundles)
    bundle_list = tuple(planned_bundles)
    bundle_bytes = 0 if dry_run else sum(written.get(path, 0) for path in planned_bundles)
    outcome.row = ContentRow(
        session=cand.path.name,
        # The classes are the finding the operator acts on; the manifest says
        # which paths within them.
        reason=" + ".join(classes),
        title=_title(cand),
        bytes=sum(removal.bytes for removal in taken),
        entries=sum(removal.entries for removal in taken),
        origin=_origin(cand.path),
        classes=classes,
        method=_taken_method(taken, bundle_list[0] if bundle_list else ""),
        rescue_bundle=bundle_list[0] if bundle_list else "",
        rescue_bundle_bytes=bundle_bytes,
        rescue_bundles=bundle_list,
        trigger=trigger,
        manifest=os.fspath(manifest_path) if manifest_written else "",
        manifest_error=manifest_error,
        # Path-level provenance in-row whenever the on-disk manifest cannot
        # name the taken paths itself: a dry run writes none, a failed planned
        # write leaves none, and a failed completed rewrite leaves one whose
        # ``completed`` is empty (review m1).
        paths=(
            tuple(os.fspath(removal.path) for removal in taken[:CONTENT_ROW_PATHS_CAP])
            if dry_run or not completed_written
            else ()
        ),
        paths_total=len(taken),
        activity_age_hours=guard.age_hours(now),
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
    when a bundle was actually kept beside the pad. ``paths`` is an int (the
    count) when the manifest's ``completed`` list is the record, and a bounded
    list of the taken paths when the row had to carry them (no manifest, or an
    incomplete one) — with ``paths_total`` alongside so a capped list is not
    read as the whole story.
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
        # Provenance: the aggregate counts alone cannot name a lost path, so
        # the row points at the manifest that can, and names what ran it. With
        # no manifest (a failed write) the row carries the taken paths itself
        # instead of only a count — path-level provenance for every removal.
        "trigger": row.trigger,
        "manifest": row.manifest,
    }
    if row.paths:
        record["paths"] = list(row.paths)
        record["paths_total"] = row.paths_total
    else:
        record["paths"] = row.paths_total
    if row.manifest_error:
        record["manifest_error"] = row.manifest_error
    if row.activity_age_hours is not None:
        record["activity_age_hours"] = round(row.activity_age_hours, 2)
    if row.rescue_bundle:
        record["rescue_bundle"] = row.rescue_bundle
        record["rescue_bundle_bytes"] = row.rescue_bundle_bytes
        record["rescue_bundles"] = list(row.rescue_bundles or (row.rescue_bundle,))
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
    #: Every REAPABLE delegated session the scan saw, with its activity clock
    #: (0.0 when there is no usable one), not only the past-window candidates.
    #: The CONTENT layer's pool: it gates on the activity clock too (restored
    #: 2026-10-19), so an in-window or clockless pad must still be VISIBLE — its
    #: kept reason then names the window or the missing clock instead of the pad
    #: silently vanishing from the plan. "Reapable" excludes origins
    #: ``cleanup.deletable_origins()`` never names, which the content layer may
    #: not take content from either.
    delegated: list[_Cand] = field(default_factory=list)
    #: Sessions (any class) that are live or inside the window: the parents whose
    #: rosters protect children.
    sources: dict[str, Path] = field(default_factory=dict)
    #: Hidden-origin directory names, for the transitive step.
    hidden: set[str] = field(default_factory=set)
    delegated_total: int = 0
    within_window: int = 0
    #: Hidden and over-age, but its origin is not one retention may reap.
    kept_by_origin: int = 0
    #: Names of that group, capped: they have no sidebar row, so the preview is
    #: the only place an operator can find them.
    kept_by_origin_names: list[str] = field(default_factory=list)
    unreadable: int = 0
    stopped: bool = False
    #: Directories examined (every class), reported as ``scanned``.
    total: int = 0
    #: Names of delegated sessions kept for being inside the window or for having
    #: no usable clock, capped: the dry run names them (an honest "kept" list)
    #: without holding 17k strings.
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
    of ``origin.json`` — twice if it is hidden, once for the class split and once
    for the reap set (:func:`cleanup.deletable_origins`), a deliberate second
    question rather than caching one answer and reusing it as the other; for every
    directory one stat of the roster (only parents have one); liveness is probed
    only for an OLD directory that has a roster.
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
                    # The CONTENT pool is the REAPABLE delegated class: an
                    # origin retention may never remove (``agent-config``) has
                    # nothing the content layer may take either, and putting it
                    # in the pool would fill the preview's kept list with one
                    # "origin is not reapable" row per such session (measured:
                    # 600 rows in the CLI's own capped-listing fixture).
                    if _is_deletable_dir(path):
                        scan.delegated.append(_Cand(path, activity, False))
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
        clock = activity if activity is not None else _clock_without_activity(path, now)
        # Read ONCE and reused for the record's rule 0 and the content pool: the
        # content layer now runs inside the window too, so it needs the same
        # answer this branch already asks below.
        deletable = _is_deletable_dir(path)
        if deletable:
            # 0.0 and not a sentinel: a candidate with no clock must still
            # reach the CONTENT layer so its kept reason names the missing
            # clock (the age gate restored there); the value only feeds the
            # preview's ordering.
            scan.delegated.append(_Cand(path, clock if clock is not None else 0.0, False))
        if clock is None or clock >= cutoff:
            # Rule 1's last clause: no clock, no removal.
            scan.within_window += 1
            if len(scan.window_names) < WINDOW_NAMES_CAP:
                scan.window_names.append(entry.name)
            continue
        if not deletable:
            # Rule 0: over-age and hidden, but this origin is NOT one retention
            # may reap (``cleanup.deletable_origins()``). Unknown must fail CLOSED:
            # a value minted by a newer build is kept until somebody names it
            # there, and an origin the operator owns (``agent-config``) stays put
            # because reaping it is irreversible while keeping it costs disk.
            scan.kept_by_origin += 1
            if len(scan.kept_by_origin_names) < WINDOW_NAMES_CAP:
                scan.kept_by_origin_names.append(entry.name)
            continue
        has_transcript = False
        try:
            has_transcript = os.stat(os.path.join(entry.path, TRANSCRIPT_FILENAME)).st_size > 0
        except OSError:
            pass
        scan.candidates.append(_Cand(path, clock, has_transcript))
    scan.candidates.sort(key=lambda cand: (cand.clock, cand.path.name))
    scan.delegated.sort(key=lambda cand: (cand.clock, cand.path.name))
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
    if not _is_deletable_dir(cand.path):
        # Re-asked at DECISION time rather than inherited from the scan: a marker
        # rewritten since then (or one the scan raced) must keep the session, the
        # same way every other guard is re-evaluated per directory.
        return "origin is not reapable"
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
    trigger: str = "startup",
    batch_size: int | None = None,
    budget_s: "float | None | _Default" = None,
    removal_pause_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    only: Iterable[str] | None = None,
    content_only: Iterable[str] | None = None,
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
    re-evaluated at removal. ``content_only`` is the CONTENT phase's own
    confinement (review F4): the CLI passes the preview's would-reclaim names,
    so a confirmed content reclaim can never include a pad the operator was not
    shown — a pad whose guard cleared since the preview is reclaimed on a later
    pass, and shown then. ``None`` leaves the phase unconfined (startup passes,
    tests). ``actor`` and ``trigger`` are recorded on every reclaim (the log row
    and the pad's manifest), so a loss can be traced to what ran the pass.
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
    result.kept_by_origin = scan.kept_by_origin
    result.kept_by_origin_names = scan.kept_by_origin_names
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
    # The copy arm rides the content phase's master switches and has NO
    # ``--force`` shortcut: ``--force`` stays the scratchpad switch's escape
    # hatch, while an arm that needs a reference repository earns its run from
    # its own config switch (and its preview is
    # ``--report-non-git-copies``).
    copy_runs = (
        policy.delegated_enabled
        and policy.delegated_scratchpad_enabled
        and policy.delegated_non_git_copies_enabled
        # A PRESENT but unreadable hold list means "I cannot tell what is held":
        # the arm does not run at all. Reading it as an empty list would reap
        # exactly the session the operator reserved.
        and not policy.delegated_non_git_copy_holds_unreadable
    )
    copy_arm = (
        _CopyArm(
            references=policy.delegated_non_git_copy_references,
            configured=bool(policy.delegated_non_git_copy_references),
            # A DRY RUN IS UNBUDGETED, like the rest of the preview: it removes
            # nothing, and it must report the whole class rather than the first
            # slice a real pass can afford to hash.
            budget=None if dry_run else NON_GIT_COPY_HASH_BUDGET_BYTES,
            holds=policy.delegated_non_git_copy_holds,
        )
        if copy_runs
        else None
    )
    if policy.delegated_non_git_copy_holds_unreadable:
        result.copy_skipped = "holds unreadable"
    else:
        result.copy_skipped = "disabled" if copy_arm is None else None
    if content_runs:
        removed_names = {row.session for row in result.removed}
        # EVERY delegated session, not only the past-window candidates: the
        # content guard itself names the refusal (window or missing clock), so the
        # plan/kept list stays auditable. Parent directories are never in this
        # pool (``_scan.delegated``), which is what keeps the user's own
        # conversations audit-only.
        content_pool = [cand for cand in scan.delegated if cand.path.name not in removed_names]
        if content_only is not None:
            # The preview's own set (review F4): an apply reclaims from exactly
            # the pads the operator was shown, never a pad that became eligible
            # between the preview and the confirmation.
            wanted_content = set(content_only)
            content_pool = [cand for cand in content_pool if cand.path.name in wanted_content]
        content_position = 0
        content_failed = 0
        content_in_batch = 0
        content_recorded = 0
        while content_position < len(content_pool):
            if should_stop is not None and should_stop():
                break
            # THE WALL-CLOCK STOP, once per pad (review m5). The pool is now
            # every reapable delegated session, so a steady-state sweep can walk
            # thousands of pads; the old deadline check only ran at a batch
            # boundary (100 removals or 100 failures), which a store with little
            # reclaimable content never reaches. The record pass's batch gate
            # stays as the failure-window arm below; this is what actually bounds
            # the sweep in wall time.
            # ``content_position`` guards the PROGRESS guarantee the batch gate
            # exists for: a pass with a spent budget still completes one pad, so
            # a tiny budget can never loop forever reclaiming nothing. Every
            # later pad re-checks, which is the once-per-pad bound (review m5).
            if deadline is not None and time.monotonic() > deadline and content_position:
                result.content_budget_exhausted = True
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
                if copy_arm is not None:
                    # Re-read the COPY arm's own policy at the batch boundary,
                    # like the switches above: a switch turned off stops it, an
                    # unreadable hold list stops it, and a hold ADDED mid-pass
                    # binds the next batch (a hold is the operator's safety
                    # statement, not a launch-time snapshot).
                    if not live.delegated_non_git_copies_enabled:
                        copy_arm = None
                        result.copy_skipped = "disabled"
                    elif live.delegated_non_git_copy_holds_unreadable:
                        copy_arm = None
                        result.copy_skipped = "holds unreadable"
                    else:
                        copy_arm.holds = live.delegated_non_git_copy_holds
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
                    actor=actor,
                    trigger=trigger,
                    copy_arm=copy_arm,
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
            result.copy_candidates += outcome.copy_candidates
            result.copy_candidate_bytes += outcome.copy_candidate_bytes
            result.copy_reclaimed_bytes += outcome.copy_reclaimed_bytes
            if outcome.copy_deferred:
                result.copy_budget_exhausted = True
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
    if copy_arm is not None:
        result.copy_budget_exhausted = result.copy_budget_exhausted or copy_arm.budget_exhausted
        if not copy_arm.configured and not copy_arm.qualified:
            # It ran and nothing could ever be proved: say so once, instead of
            # leaving every candidate's "no reference repository" to imply it.
            result.copy_skipped = "no reference repository"
    if dry_run:
        # The honest kept list names the sessions the age rule itself saved, last
        # (capped at WINDOW_NAMES_CAP; the totals line carries the full count).
        # The reason covers both members of the bucket: a session inside the
        # window, and one with NO USABLE clock (no marker, or an implausible birth
        # stamp). Calling the second "active recently" would state something the
        # scan does not know.
        result.protected.extend(
            (name, f"inside the {result.hours}h window or no usable clock")
            for name in scan.window_names
        )
        # Same cap, same place: these were never in the age rule at all (their
        # origin is not reapable) and they have no sidebar row, so the dry run is
        # the only surface that can tell the operator WHICH ones the count means.
        result.protected.extend((name, "origin not reapable") for name in scan.kept_by_origin_names)
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
    trigger: str = "startup",
) -> str:
    """Pass, wait, pass again — until stopped or the switch is off. Returns why it ended.

    ``wait(seconds)`` must return True when the caller is stopping (it is
    ``threading.Event.wait``). The loop ends when the delegated switch is off at
    a sweep boundary — the keys are Scope.NEW_LAUNCH, so switching it back on
    takes effect at the next launch, as the settings page says — and a pass that is
    already running notices the switch at its next batch. A drain that is not
    finished comes back after :data:`DRAIN_RESUME_S`; a finished one after
    :data:`STEADY_SWEEP_S`. ``actor``/``trigger`` name this host in the content
    manifests and cleanup-log rows (``session-startup`` for this thread).
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
                        trigger=trigger,
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
