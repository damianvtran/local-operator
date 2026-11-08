"""Bounded, fail-closed disk-pressure reclaim (``lop disk reclaim``, issue #2676).

WHY THIS MODULE EXISTS
    Under acute disk pressure the product already owned guarded removers
    (``remove_scratchpad_entry``, ``prune_generations``) but offered no
    sanctioned, measure-first front door for them, so an ad-hoc sweep with a bare
    ``shutil.rmtree`` was the path of least resistance — and one such sweep
    deleted a live runtime generation. This module is that front door for the
    three classes of reclaimable disk the incident named. It is a PLANNER plus a
    thin APPLIER, never a second deleter:

    * **linked git worktrees** (new here) — removed ONLY through
      ``git worktree remove <path>`` WITHOUT ``--force``; git's own refusal is
      the last word. Tracked or unmerged work is never "cache": a worktree is
      reclaimable only when every predicate in :func:`assess_worktree` holds.
    * **the Go build cache** — removed ONLY through ``go clean -cache`` with
      ``GOCACHE`` pinned to the path this module validated.
    * **delegated-session scratchpad content** — composed with the existing
      ``run_delegated_pass`` machinery (same policy, retention window, force
      semantics and guards as ``lop sessions cleanup --scratchpads-only``);
      nothing is re-implemented or weakened here.
    * **operator-named project build output** (``--build-output PATH``) — the
      fourth category and the ONE place this module deletes anything itself, for
      the reason set out under BUILD OUTPUT below.

    There is no ``shutil.rmtree``, ``os.remove`` or ``rm`` anywhere in this
    file, and none may be added. Every removal belongs to a guarded remover that
    owns its own refusal rules — ``git worktree remove``, ``go clean -cache``,
    the scratchpad pass — with exactly ONE documented exception: build output is
    a directory tree that no tool-level remover owns (``cargo clean`` knows only
    the ``target`` it created, and nothing collects ``.next``/``build``/
    ``dist``), so :func:`remove_build_output` exists. It owns these guards, and
    they are its whole justification:

    * it re-validates EVERY predicate at removal time and SKIPS, never removes,
      an item that stopped qualifying — reporting which predicate failed;
    * it refuses a symlink and a mount point at removal time as well, because the
      state that was previewed is not necessarily the state on disk;
    * it never follows a symlink out of the tree (a post-order walk that unlinks
      link entries instead of descending them);
    * it measures allocated bytes immediately before removal and records that
      measurement, so the receipt's preview figure and the removal figure can be
      compared;
    * it is the only place in this module that removes anything, and it is
      reachable only from :func:`apply`, for items the preview already approved.

BUILD OUTPUT: WHY THE FOURTH CATEGORY IS GATED ON PROVENANCE
    Build output is the one class of reclaimable space with no tool that owns
    its removal, so the safety argument cannot be "the tool refused". It has to
    be "this directory is provably build output of a provably identified
    project, and provably idle". Everything unusual about this category follows
    from that. The asymmetry that shapes it is the module's own: a deleted build
    tree costs a rebuild, while a deleted piece of someone's work — or a
    directory removed out from under a running compile — is not recoverable.

    PROVENANCE IS THE GATE, AND IT IS DATA. :data:`BUILD_OUTPUT_SHAPES` maps a
    directory name to the manifests that PROVE a project of that kind owns it:
    ``target`` beside a ``Cargo.toml``, ``.next`` beside a ``package.json``, and
    ``build``/``dist`` beside one of the manifests listed there. A directory is
    only even CONSIDERED as a candidate when its parent — the project root the
    operator named — holds one of that shape's manifests. That gate is what
    makes this safe for a non-technical user: someone who happens to keep a
    folder called ``build`` or ``dist`` under ``~/projects`` has no manifest
    there, so it is never a candidate at all — whatever it contains and however
    large it is. The manifest is what makes the deletion PROVABLE rather than
    guessed from a name, and it is why adding an ecosystem is one row in that
    table rather than a new code path, a new module or a new command. A manifest
    is not the only defence for the folders that are not projects at all: see
    :data:`PROTECTED_DATA_DIRS`.

    The operator names project DIRECTORIES (``--build-output PATH``), never
    shape directories, and nothing is discovered or defaulted — the same rule as
    ``--repo``. Naming none means this category is not planned at all.

    Every predicate fails toward KEEP, and each refusal is an ``Item`` with
    ``action=keep``, a stable code and a :data:`KEEP_HINTS` entry:

    * a ``.git`` entry (directory OR the FILE a linked worktree carries) at the
      candidate's own root or anywhere below it — a cloned repository inside
      build output keeps its history only there — or the candidate BEING a git
      directory (``git init --bare <shape>``, ``--separate-git-dir``) —
      ``git-repository``;
    * a candidate that is a symlink, or a path that traverses one — ``symlink``;
    * a candidate that is a mount point, or whose walk crosses onto another
      filesystem — ``mount-point``, because that other volume's contents are not
      ours to delete;
    * a project root that is ``$HOME`` or a standard data directory
      (:data:`PROTECTED_DATA_DIRS`), and a candidate that CONTAINS one of those
      directories — ``protected-location``: one stray manifest is not evidence
      that somebody's own folder is build output;
    * tracked files inside it (``git-tracked``), or a shape git does not ignore
      (``git-not-ignored``); a git that cannot answer is ``git-unknown`` and
      keeps, never "nothing is tracked";
    * a toolchain process attributable to it or to its project root, or a
      process whose cwd is at/under it or which holds files open beneath it —
      ``toolchain-running`` / ``handles-open``; either probe failing is
      ``probe-failed`` and keeps. This command's own process is excluded from
      both tables: it is a generic ``python`` producer sitting in whatever
      directory the operator ran it from, so counting it kept the very tree the
      command was asked about;
    * a walk that hit its entry bound — ``idle-unproven``, because a truncated
      walk cannot prove the tree idle.

    The ``.git`` guards run BEFORE the git probes on purpose: the probes ask from
    the PARENT, and ``git -C <repo>/.worktrees ls-files`` happily answers about
    the parent repository, so a confident answer about the WRONG repository is
    the dangerous case. The ``.git``-inside guard is what closes it.

    THE IDLE WINDOW IS THIS CATEGORY'S OWN: :data:`BUILD_OUTPUT_IDLE_MINUTES`
    (six hours), not the module's 60-minute floor. ``--min-idle-minutes`` may
    raise it and can never lower it, so the effective window is
    ``max(--min-idle-minutes, 360)`` and the value used is stamped into every
    item's evidence.

    NO SETTINGS, NO i18n, NO NEW NAMESPACE, deliberately: the three things an
    operator would want configurable are the command's existing verbs. The
    threshold IS this category's idle window; dry-run IS the command's default
    (the preview); and "enabled" IS naming ``--build-output`` at all. Adding a
    config section would create a second place where a scope decision lives,
    and this module's whole premise is that the operator names the scope. The
    output is English-only operator CLI text, like the rest of this command and
    the sessions-cleanup section (docs/I18N.md), so there is nothing to translate.

CONTRACT
    * Preview first. :func:`plan` reads and measures; it removes nothing.
    * Sizes are ``st_blocks * 512`` ALLOCATED bytes from a BOUNDED walk. They are
      a measurement, not a promise: hardlinks, shared git objects, APFS clones
      and block rounding all break "allocated == reclaimed". A truncated walk
      yields a LOWER bound and says so. The only honest "what was reclaimed"
      figure is the before/after ``shutil.disk_usage`` pair in the receipt.
    * Apply (:func:`apply`) acts ONLY on the previewed items and re-validates
      every predicate at removal time; an item that stopped qualifying is
      skipped and reported, never removed.
    * Every uncertainty keeps the thing: an unreadable directory, a hit walk
      bound, an unreachable ``lsof``, an unexpected exception — each is a KEEP
      with an owner-actionable reason.
    * The operator names the repositories (``--repo``). Nothing is discovered by
      wandering the filesystem, and the primary checkout is never a candidate.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# The git helpers are reused, not copied: ``_git`` runs ``git -C <repo>`` with
# the hardened environment (``XPC_FLAGS`` dropped, repo-redirecting variables
# scrubbed, system config and prompts disabled, ``GIT_OPTIONAL_LOCKS=0`` so an
# inspection never refreshes the index it is judging). A second environment
# recipe beside it would be a way for the two to drift.
from local_operator.session.delegated_retention import _git

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: The idle floor, in minutes. ``--min-idle-minutes`` may RAISE it and can never
#: lower it: an hour is the shortest quiet period that still separates "an
#: agent between turns" from "abandoned". Enforced in :func:`validate_min_idle`.
MIN_IDLE_MINUTES_FLOOR = 60
DEFAULT_MIN_IDLE_MINUTES = 60

#: Entries the working-tree walk may visit. Same order of magnitude as the
#: content layer's recency bound (``CONTENT_RECENCY_MAX_ENTRIES``): large enough
#: for an ordinary checkout with its dependency directories, small enough that a
#: pathological tree cannot stall the command. PAST IT THE MTIME IS UNPROVEN AND
#: THE WORKTREE IS KEPT — "could I not look?" answers keep, never reclaim.
WALK_MAX_ENTRIES = 250_000

#: Seconds for a cheap git/lsof/go query, and for the ones that scale with the
#: repository (``status``, ``worktree remove``, ``go clean``).
QUICK_TIMEOUT_S = 30.0
HEAVY_TIMEOUT_S = 300.0

#: The text Go writes to ``<GOCACHE>/README``. Its presence is the proof that a
#: directory IS a Go build cache rather than whatever ``GOCACHE`` was pointed at.
GO_CACHE_MARKER = "This directory holds cached build artifacts from the Go build system"

RECEIPT_DIRNAME = "disk-reclaim-receipts"
RECEIPT_SCHEMA = 1

#: How a removal was authorised, stamped into EVERY receipt this module writes
#: (intent, checkpoints, final). It lives here, next to the schema, because it is
#: part of the durable audit record rather than a CLI-only detail: the whole point
#: of the receipt is that the run can be reconstructed from the FILE, so the file
#: must say whether a typed prompt was answered or ``--yes`` bypassed it. There is
#: deliberately no default — a caller that cannot say how removal was authorised
#: is refused (see :func:`apply`) instead of writing a receipt that guesses.
CONFIRMATION_TYPED = "typed"
CONFIRMATION_YES = "--yes"
CONFIRMATION_KINDS = (CONFIRMATION_YES, CONFIRMATION_TYPED)

#: Printed in every receipt and every preview: the one sentence that stops a
#: reader from treating a measurement as a result.
MEASUREMENT_NOTE = (
    "Sizes are allocated bytes (st_blocks*512) from a bounded walk — a measurement, "
    "not a promise: hardlinks, shared git objects, APFS clones and block rounding "
    "mean allocated or logical bytes are NOT reclaimed bytes. A truncated walk is a "
    "lower bound. The disk_usage before/after readings are the only figure of what "
    "was actually freed."
)

CATEGORY_WORKTREE = "worktree"
CATEGORY_GO_CACHE = "go-build-cache"
CATEGORY_SCRATCHPAD = "scratchpad"
CATEGORY_BUILD_OUTPUT = "build-output"

#: The idle window for build output, in minutes: this category's OWN default,
#: not the module's 60-minute floor. The floor exists to separate "an agent
#: between turns" from "abandoned" for a WORKTREE, whose activity is a person or
#: an agent editing files. A build tree is written in bursts by a toolchain that
#: a wrapper may be driving from elsewhere, so an hour of quiet is not evidence
#: that nothing is building into it. ``--min-idle-minutes`` may RAISE this window
#: and can never lower it (see :func:`build_output_idle_minutes`), which keeps the
#: floor rule and this default composing instead of competing.
BUILD_OUTPUT_IDLE_MINUTES = 6 * 60

#: Directory name -> the manifests that PROVE a project of that kind owns it.
#:
#: THIS TABLE IS THE GENERALISATION. A candidate directory is never judged by its
#: NAME: it is only even considered when its parent — the project root the
#: operator named — holds one of the manifests listed here for that shape. A
#: folder called ``build`` with no manifest beside it is somebody's data, and no
#: amount of resemblance changes that. Adding an ecosystem is one row here.
BUILD_OUTPUT_SHAPES: dict[str, tuple[str, ...]] = {
    "target": ("Cargo.toml",),
    ".next": ("package.json",),
    "build": (
        "CMakeLists.txt",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "meson.build",
        "Makefile",
        "package.json",
        "pyproject.toml",
        "setup.py",
    ),
    "dist": (
        "pyproject.toml",
        "setup.py",
        "package.json",
        "Cargo.toml",
        "tsconfig.json",
    ),
}

#: The executables whose presence near a candidate means a build MAY be writing
#: it. Deliberately NOT ``node``, ``npm``, ``pnpm`` or ``yarn``: they are
#: interpreters and package managers, far too common among a fleet of concurrent
#: sessions to be evidence of anything, and a measured run had a plain ``node``
#: process with its cwd at the repository freeze ``<repo>/target`` for as long as
#: it lived — a guard that fires on an idle machine is a guard that gets switched
#: off. Nothing is lost, because the producer a build runs IS itself in this set
#: for as long as the build runs (``next build`` is
#: ``node .../node_modules/.bin/next``) and :func:`is_producer_process` checks
#: EVERY argv element, not just ``argv[0]``.
BUILD_OUTPUT_PRODUCERS: frozenset[str] = frozenset(
    {
        # target
        "cargo",
        "rustc",
        "rustdoc",
        "clippy-driver",
        "sccache",
        # .next
        "next",
        "vite",
        "tsc",
        "webpack",
        "esbuild",
        # build
        "cmake",
        "make",
        "ninja",
        "gradle",
        "mvn",
        "maven",
        "xcodebuild",
        # dist
        "python",
        "pip",
        "uv",
        "setuptools",
    }
)

#: The subset of :data:`BUILD_OUTPUT_PRODUCERS` that is attributed by CWD ONLY.
#:
#: These are the generic interpreter and package-manager names a Python build
#: runs through, and they are separated for a MEASURED reason: ``lop disk reclaim
#: --build-output <root>`` is itself a ``python`` process whose argv CONTAINS the
#: project root, so attributing them by argv blocked every reclaim this command
#: made — a guard that fires on an idle machine is a guard that gets switched off.
#: cwd is still evidence for them (``python -m build`` and ``pip install .`` run
#: IN the project), and a specific toolchain keeps full argv attribution, where a
#: path in the command line really is a build being pointed at a directory.
BUILD_OUTPUT_GENERIC_PRODUCERS: frozenset[str] = frozenset(
    {
        "python",
        "pip",
        "uv",
        "setuptools",
    }
)

#: Directory names directly under ``$HOME`` that hold the user's own data rather
#: than a project. ``$HOME`` itself and these are never build-output project
#: roots, and a candidate that CONTAINS one of them is kept: the provenance gate
#: proves a project owns a shape directory, it says nothing about a folder the
#: operator happens to keep documents in, and one stray ``~/package.json`` would
#: otherwise make ``~/dist`` — or ``~/Documents/build`` — provably build output.
#: The Go-cache category already refuses ``$HOME`` for the same reason.
PROTECTED_DATA_DIRS: frozenset[str] = frozenset(
    {"Documents", "Desktop", "Downloads", "Movies", "Music", "Pictures"}
)

#: What :func:`producer_kind` answers for one process.
PRODUCER_SPECIFIC = "specific"
PRODUCER_GENERIC = "generic"

ACTION_RECLAIM = "reclaim"
ACTION_KEEP = "keep"

# Item outcomes. ``planned`` -> ``removed`` | ``skipped`` | ``failed`` on apply;
# ``kept`` never changes.
OUTCOME_PLANNED = "planned"
OUTCOME_KEPT = "kept"
OUTCOME_REMOVED = "removed"
OUTCOME_SKIPPED = "skipped"
OUTCOME_FAILED = "failed"

#: Entries in a worktree's gitdir that mean a multi-step git operation is
#: half-done (the state is NOT in the commit graph).
_OPERATION_MARKERS = (
    "rebase-merge",
    "rebase-apply",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)

#: Keep codes -> the owner's next step. The CLI groups kept rows by code so the
#: owner can act on a whole class at once.
KEEP_HINTS: dict[str, str] = {
    "primary": "the primary checkout is never a candidate",
    "bare": "a bare repository entry is never a candidate",
    "uninspectable": "git cannot inspect it; run `git -C <path> status` to see why",
    "invalid-registration": (
        "the registration is invalid, missing or prunable; `git worktree prune` is the "
        "owner's call — this command never deletes it"
    ),
    "locked": "unlock with `git worktree unlock <path>` once the lock's reason is over",
    "cwd": "this process runs inside it (or it is the --repo you named)",
    "in-use": "a process has its working directory inside it; stop that process first",
    "in-use-unknown": "lsof could not establish which processes use it; rerun where lsof works",
    "recent": "touched within the idle window; it is active — rerun after it has been quiet",
    "head-moved": "HEAD changed between the preview and removal; re-preview to see the new state",
    "ignored-content": (
        "gitignored files (e.g. .env, a nested repo) exist only in this directory and would "
        "be deleted with it; move or delete them deliberately, then rerun"
    ),
    "operation-in-progress": (
        "a rebase/merge/cherry-pick/revert/bisect is in progress; finish or abort it, then rerun"
    ),
    "idle-unproven": "too large or unreadable to prove idle within the walk bound",
    "dirty": "commit, stash or discard the changes (`git -C <path> status`) then rerun",
    "unpushed": "push the HEAD commit (or merge it) so a remote ref contains it, then rerun",
    "no-trunk-or-remote": "no remote-tracking ref contains HEAD; fetch/push, then rerun",
    "refused": "see the reason; nothing was touched",
    "policy-off": "enable the setting named in the reason, or pass --force for one run",
    "guarded": "kept by the scratchpad guards (see the reason)",
    "git-repository": (
        "the directory IS a repository, contains one below it, or is a linked-worktree "
        "root; a branch, its history and any uncommitted work would go with it"
    ),
    "symlink": "a symlink (or a path through one) is never followed or removed",
    "mount-point": (
        "it is the root of its own filesystem, or the tree crosses onto another one; "
        "that volume's contents are not ours"
    ),
    "protected-location": (
        "it is (or contains) your home directory or one of your standard data "
        "directories; a folder there is your data, never build output"
    ),
    "git-tracked": "git tracks files inside it, so part of it is committed content",
    "git-not-ignored": (
        "git does not ignore it, so it is not provably build output; ignore it in "
        ".gitignore or delete it deliberately, then rerun"
    ),
    "git-unknown": (
        "git could not answer (missing binary, timeout, unreadable repository); rerun "
        "where git works"
    ),
    "toolchain-running": (
        "a build toolchain process is working in it; let it finish and rerun (a removed "
        "build directory corrupts an in-flight build)"
    ),
    "handles-open": (
        "a process holds files open inside it or is working in it; stop that process "
        "first"
    ),
    "probe-failed": (
        "the liveness or provenance probes could not answer (ps/lsof/git unavailable or "
        "unreadable); rerun where they work"
    ),
    "no-manifest": (
        "the manifest that proved this shape is gone since the preview; re-preview to see "
        "the new state"
    ),
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Item:
    """One candidate (reclaim) or one explained keep.

    ``allocated_bytes`` is a MEASUREMENT (see :data:`MEASUREMENT_NOTE`);
    ``bytes_truncated`` marks it a lower bound. ``evidence`` carries the facts
    the decision rested on (HEAD sha, containing remote refs, idle seconds…) so a
    receipt reader does not have to trust the verdict.
    """

    id: str
    category: str
    group: str
    path: str
    action: str
    code: str
    reason: str
    allocated_bytes: int | None = None
    bytes_truncated: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    outcome: str = OUTCOME_PLANNED
    detail: str = ""

    @property
    def hint(self) -> str:
        return KEEP_HINTS.get(self.code, "")

    def as_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["hint"] = self.hint
        return data


@dataclass
class Selection:
    """What the operator asked for. Nothing is added by default."""

    repos: tuple[str, ...] = ()
    go_build_cache: bool = False
    scratchpads: bool = False
    #: Operator-named PROJECT directories whose build output may be reclaimed.
    #: Never defaulted and never discovered, exactly like ``repos``: naming none
    #: means the category is not planned at all.
    build_output: tuple[str, ...] = ()
    min_idle_minutes: int = DEFAULT_MIN_IDLE_MINUTES
    force: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Re-check mutable selections at every public planning/apply boundary."""
        # The floor lives where the predicate runs, not only in the CLI: this
        # module is the sanctioned entry point other callers are told to use, so
        # a Selection that asks for less than the floor cannot even be built.
        problem = validate_min_idle(self.min_idle_minutes)
        if problem:
            raise ValueError(problem)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Plan:
    """The preview. ``refusals`` are whole-category refusals (e.g. an unsafe GOCACHE)."""

    selection: Selection
    items: list[Item] = field(default_factory=list)
    refusals: list[dict[str, str]] = field(default_factory=list)
    #: The moment every time-based predicate was judged against.
    now: float = 0.0
    #: Directories whose filesystems get a ``disk_usage`` reading.
    probe_paths: list[str] = field(default_factory=list)

    @property
    def reclaimable(self) -> list[Item]:
        return [i for i in self.items if i.action == ACTION_RECLAIM]

    @property
    def kept(self) -> list[Item]:
        return [i for i in self.items if i.action == ACTION_KEEP]


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def validate_min_idle(value: int) -> str | None:
    """``None`` when ``value`` is acceptable, else the refusal message.

    The floor is a SAFETY constant, not a default: a lower value is rejected
    rather than clamped, so a script that typed ``--min-idle-minutes 5`` learns
    it did not get what it asked for instead of silently getting an hour.
    """
    if value < MIN_IDLE_MINUTES_FLOOR:
        return (
            f"--min-idle-minutes must be at least {MIN_IDLE_MINUTES_FLOOR} "
            f"(got {value}); the floor cannot be lowered"
        )
    return None


def utc_stamp(moment: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if moment is None else moment, timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )


def _iso(moment: float) -> str:
    return datetime.fromtimestamp(moment, timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Bounded filesystem measurement
# --------------------------------------------------------------------------


@dataclass
class Walk:
    """Result of one bounded walk: allocated bytes, newest mtime, and how far it saw."""

    allocated_bytes: int = 0
    newest_mtime_ns: int = 0
    entries: int = 0
    #: The entry bound was hit: bytes are a LOWER bound and "no recent entry" is
    #: NOT established.
    truncated: bool = False
    #: An entry newer than the cutoff was found (the walk stops there; bytes are
    #: partial and must not be quoted).
    recent: bool = False
    #: An entry could not be stat-ed/listed: the mtime is unproven.
    error: str = ""
    #: ``.git`` entries (file or directory) found BELOW the root: a nested
    #: repository or submodule whose history lives only inside this tree.
    nested_git: list[str] = field(default_factory=list)
    #: ``stop_at_mount`` only: a directory inside the tree that is a mount point.
    #: The walk STOPPED there, so bytes are a lower bound and nothing past it was
    #: measured — another filesystem's contents are not this tree's to delete.
    mount_crossed: str = ""
    #: ``prune_names`` only: directories the walk refused to descend into. Empty
    #: is the only answer that leaves the tree fully measured.
    pruned: list[str] = field(default_factory=list)


def walk_allocated(
    root: str,
    *,
    cutoff_ns: int | None = None,
    max_entries: int = WALK_MAX_ENTRIES,
    skip_top_level: frozenset[str] = frozenset(),
    stop_at_mount: bool = False,
    prune_names: frozenset[str] = frozenset(),
) -> Walk:
    """Iterative, symlink-never-followed walk of ``root`` with a hard entry bound.

    One pass answers both questions the callers have — "how much is allocated"
    and "what is the newest mtime" — so the two can never describe different
    populations. Directories are stat-ed too (their mtime moves when a file is
    added or removed beside them). With ``cutoff_ns`` the walk stops at the first
    entry at-or-after it: nothing later can make the tree look idler.

    ``skip_top_level`` names entries directly under ``root`` to ignore (a
    worktree's ``.git`` pointer/admin dir, judged separately). Nested ``.git``
    directories (submodules, vendored repos) are NOT skipped: activity there is
    activity.

    ``stop_at_mount`` and ``prune_names`` are the build-output category's two
    edges, opt-in so the worktree and Go-cache walks keep their existing meaning.
    A mount point below ``root`` ENDS the walk: its bytes belong to another
    filesystem and nothing past it is measured. A directory named in
    ``prune_names`` is not descended into and is recorded in
    :attr:`Walk.pruned` — never silently, because a walk that did not cover the
    tree cannot be used to justify deleting all of it.
    """
    result = Walk()
    try:
        top = os.lstat(root)
    except OSError as exc:
        result.error = f"{root}: {exc.strerror or exc}"
        return result
    result.newest_mtime_ns = top.st_mtime_ns
    if cutoff_ns is not None and top.st_mtime_ns >= cutoff_ns:
        result.recent = True
        return result
    stack = [root]
    first = True
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as scan:
                listing = list(scan)
        except OSError as exc:
            result.error = f"{directory}: {exc.strerror or exc}"
            return result
        for entry in listing:
            if first and entry.name in skip_top_level:
                continue
            if result.entries >= max_entries:
                result.truncated = True
                return result
            result.entries += 1
            if not first and entry.name == ".git" and len(result.nested_git) < 5:
                result.nested_git.append(entry.path)
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                result.error = f"{entry.path}: {exc.strerror or exc}"
                return result
            if stop_at_mount and stat.S_ISDIR(st.st_mode):
                # STOP rather than descend: `_remove_tree` would walk onto the
                # other volume and delete what is not ours. A probe that cannot
                # answer is the same keep as an unreadable directory.
                try:
                    crossed = os.path.ismount(entry.path)
                except OSError as exc:
                    result.error = f"{entry.path}: {exc.strerror or exc}"
                    return result
                if crossed:
                    result.mount_crossed = entry.path
                    return result
            if stat.S_ISDIR(st.st_mode) and entry.name in prune_names:
                if len(result.pruned) < 5:
                    result.pruned.append(entry.path)
                continue
            result.allocated_bytes += st.st_blocks * 512
            if st.st_mtime_ns > result.newest_mtime_ns:
                result.newest_mtime_ns = st.st_mtime_ns
            if cutoff_ns is not None and st.st_mtime_ns >= cutoff_ns:
                result.recent = True
                return result
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
            except OSError as exc:
                result.error = f"{entry.path}: {exc.strerror or exc}"
                return result
        first = False
    return result


# --------------------------------------------------------------------------
# Process-in-use signal (lsof)
# --------------------------------------------------------------------------


def process_cwds(timeout: float = QUICK_TIMEOUT_S) -> list[tuple[int, str]] | None:
    """``[(pid, realpath-of-cwd)]`` for every process lsof can see, or ``None``.

    DECISION (documented in docs/disk-pressure-reclaim.md): "in use" means *a
    process has its working directory inside the worktree*. That is the one
    signal that is cheap (one ``lsof -d cwd`` for the whole fleet, not a
    ``+D`` descent per tree), deterministic, and the failure shape of the real
    incident class (a shell or agent sitting in a checkout). Open file
    descriptors elsewhere in the tree are NOT checked, nor are processes owned
    by other users lsof cannot see — the idle floor and the lock are the
    backstops, and the docs say so.

    FAIL CLOSED: if lsof is missing, times out, fails, or does not even list
    THIS process (a canary: our own cwd must appear, otherwise the listing is
    not the whole picture), the answer is ``None`` and every worktree that
    reaches the check is kept as "in-use state unknown".
    """
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["lsof", "-n", "-P", "-w", "-d", "cwd", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # lsof exits 1 when it could not inspect some process but still lists the
    # rest; only a listing that contains our own pid is trusted.
    if proc.returncode not in (0, 1):
        return None
    rows: list[tuple[int, str]] = []
    pid: int | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("p"):
            try:
                pid = int(line[1:])
            except ValueError:
                pid = None
        elif line.startswith("n") and pid is not None:
            rows.append((pid, os.path.realpath(line[1:])))
    if not any(p == os.getpid() for p, _ in rows):
        return None
    return rows


# --------------------------------------------------------------------------
# Linked worktrees
# --------------------------------------------------------------------------


class GitUnavailable(Exception):
    """git could not answer a question the plan cannot proceed without."""


@dataclass
class Registration:
    """One ``worktree`` stanza of ``git worktree list --porcelain``."""

    path: str
    head: str = ""
    branch: str = ""
    detached: bool = False
    bare: bool = False
    #: ``None`` unlocked; otherwise the lock reason (possibly empty).
    locked: str | None = None
    #: ``None`` not prunable; otherwise git's reason text.
    prunable: str | None = None


def _run_git(
    repo: str, *args: str, timeout: float = QUICK_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    """``_git`` that turns a launch failure/timeout into :class:`GitUnavailable`."""
    try:
        return _git(repo, *args, timeout=timeout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise GitUnavailable(f"git {args[0] if args else ''} failed: {exc}") from exc


def list_registrations(repo: str) -> list[Registration]:
    """The repository's registered worktrees, primary first.

    Plain (non ``-z``) porcelain, parsed by stanza. A path containing a newline
    would be mis-split, but a mis-split path does not exist on disk and is
    therefore kept as an invalid registration — it fails closed. The FIRST
    stanza is always the main working tree (or the bare repository), which is
    how the primary is identified without guessing.
    """
    done = _run_git(repo, "worktree", "list", "--porcelain")
    if done.returncode != 0:
        raise GitUnavailable(f"git worktree list failed: {done.stderr.strip()[:200]}")
    regs: list[Registration] = []
    current: Registration | None = None
    for line in done.stdout.splitlines():
        if not line.strip():
            current = None
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current = Registration(path=value)
            regs.append(current)
        elif current is None:
            continue
        elif key == "HEAD":
            current.head = value
        elif key == "branch":
            current.branch = value
        elif key == "detached":
            current.detached = True
        elif key == "bare":
            current.bare = True
        elif key == "locked":
            current.locked = value
        elif key == "prunable":
            current.prunable = value
    return regs


@dataclass
class RepoContext:
    """Everything one repository's candidates are judged against."""

    repo: str  # realpath of the --repo argument
    primary: str  # realpath of the main working tree
    common_dir: str  # realpath of the shared git dir
    registrations: list[Registration]
    now: float
    min_idle_s: float
    cwd: str
    #: ``None`` when lsof could not establish the table (fail closed).
    cwds: list[tuple[int, str]] | None
    trunk: str | None


def _resolve_trunk(repo: str) -> str | None:
    """The remote trunk ref to test "merged" against, or ``None``.

    ``origin/HEAD`` first (what the remote says its default is), then
    ``origin/main`` and ``origin/master``. Remote-tracking refs are as last
    FETCHED — this command does no network I/O, so a trunk that moved since is
    not seen (conservative for "merged": an old trunk contains less).
    """
    sym = _run_git(repo, "symbolic-ref", "-q", "refs/remotes/origin/HEAD")
    if sym.returncode == 0 and sym.stdout.strip():
        target = sym.stdout.strip()
        if _run_git(repo, "rev-parse", "-q", "--verify", target).returncode == 0:
            return target
    for ref in ("refs/remotes/origin/main", "refs/remotes/origin/master"):
        if _run_git(repo, "rev-parse", "-q", "--verify", ref).returncode == 0:
            return ref
    return None


def build_repo_context(
    repo_arg: str,
    *,
    now: float,
    min_idle_minutes: int,
    cwds: list[tuple[int, str]] | None,
) -> RepoContext:
    """Resolve ``--repo`` into a context. Raises :class:`GitUnavailable` if it is no repo.

    The path must be a git working tree (or its linked worktree) the operator
    named explicitly; ``--repo`` is never defaulted or discovered.
    """
    problem = validate_min_idle(min_idle_minutes)
    if problem:
        # Second line of defence behind ``Selection``: the predicate itself
        # refuses a window under the floor, whoever built the context.
        raise ValueError(problem)
    repo = os.path.realpath(os.path.expanduser(repo_arg))
    if not os.path.isdir(repo):
        raise GitUnavailable(f"{repo_arg}: not a directory")
    common = _run_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common.returncode != 0 or not common.stdout.strip():
        raise GitUnavailable(f"{repo_arg}: not a git repository")
    regs = list_registrations(repo)
    if not regs:
        raise GitUnavailable(f"{repo_arg}: git lists no worktrees")
    return RepoContext(
        repo=repo,
        primary=os.path.realpath(regs[0].path),
        common_dir=os.path.realpath(common.stdout.strip()),
        registrations=regs,
        now=now,
        min_idle_s=min_idle_minutes * 60.0,
        cwd=os.path.realpath(os.getcwd()),
        cwds=cwds,
        trunk=_resolve_trunk(repo),
    )


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _keep(ctx: RepoContext, path: str, code: str, reason: str, **evidence: Any) -> Item:
    return Item(
        id=f"{CATEGORY_WORKTREE}:{path}",
        category=CATEGORY_WORKTREE,
        group=ctx.repo,
        path=path,
        action=ACTION_KEEP,
        code=code,
        reason=reason,
        evidence={"repo": ctx.repo, **evidence},
        outcome=OUTCOME_KEPT,
    )


def assess_worktree(ctx: RepoContext, reg: Registration) -> Item:
    """Decide ONE registered worktree: a reclaim Item or an explained keep.

    Predicates, cheapest and most decisive first. ALL must hold to reclaim; the
    first that fails is the reported reason. Any exception is a keep.

    1. not the primary checkout, not a bare entry (never removed, by identity)
    2. the registration is valid and the directory is a real linked worktree of
       THIS repository (guards against a copy whose ``.git`` file still names the
       original's gitdir) and git can inspect it
    3. not locked
    4. not this process's cwd / the ``--repo`` argument / containing the cwd
    5. no process has its cwd inside it (lsof; failure => keep, "state unknown")
    6. idle >= the window, from the newest of {gitdir HEAD, index, logs/HEAD} and
       a bounded walk of the working tree (``.git`` skipped); an unreached bound
       or unreadable entry => keep
    7. clean: ``status --porcelain --untracked-files=all --ignored`` empty;
       ignored content, nested git entries and in-progress operations KEEP it
    8. HEAD is contained in a remote-tracking ref ("pushed") — reported as
       "merged" when that ref is the remote trunk. A checked-out branch must
       still sit at HEAD. Refs are as last fetched; no network is used.

    COMMITS ONLY LIVE ON REFS: removing a worktree deletes its working files,
    its per-worktree HEAD/index/reflog, and nothing in the shared object store
    or ref namespace. So the proof that no work is lost is that HEAD (and the
    branch at it) is reachable from a remote — which is what (8) establishes —
    plus that the working tree has no uncommitted changes (7).
    """
    path = os.path.realpath(reg.path)
    try:
        return _assess(ctx, reg, path)
    except GitUnavailable as exc:
        return _keep(ctx, path, "uninspectable", f"git could not inspect it: {exc}")
    except Exception as exc:  # noqa: BLE001 — fail closed on ANY unexpected error
        return _keep(
            ctx, path, "uninspectable", f"unexpected {type(exc).__name__} while inspecting: {exc}"
        )


def _assess(ctx: RepoContext, reg: Registration, path: str) -> Item:
    if path == ctx.primary:
        return _keep(ctx, path, "primary", "primary checkout")
    if reg.bare:
        return _keep(ctx, path, "bare", "bare repository entry")
    if reg.prunable is not None:
        return _keep(ctx, path, "invalid-registration", f"prunable registration: {reg.prunable}")
    if not os.path.isdir(reg.path) or os.path.islink(reg.path):
        return _keep(ctx, path, "invalid-registration", "directory missing or a symlink")
    git_entry = os.path.join(reg.path, ".git")
    if os.path.islink(git_entry) or not os.path.isfile(git_entry):
        return _keep(ctx, path, "invalid-registration", ".git is not a plain gitdir-pointer file")
    # The listed path is a real linked worktree of THIS repository only if its
    # own idea of the common dir and toplevel agree with the registration.
    gitdir = _run_git(path, "rev-parse", "--path-format=absolute", "--absolute-git-dir")
    common = _run_git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    top = _run_git(path, "rev-parse", "--show-toplevel")
    if gitdir.returncode or common.returncode or top.returncode:
        return _keep(ctx, path, "uninspectable", "git cannot resolve its git dir")
    gitdir_real = os.path.realpath(gitdir.stdout.strip())
    if (
        os.path.realpath(common.stdout.strip()) != ctx.common_dir
        or os.path.dirname(gitdir_real) != os.path.join(ctx.common_dir, "worktrees")
        or os.path.realpath(top.stdout.strip()) != path
    ):
        return _keep(
            ctx, path, "invalid-registration", "its .git does not belong to this repository"
        )

    if reg.locked is not None:
        why = reg.locked.strip() or "no reason given"
        return _keep(ctx, path, "locked", f"locked ({why})", lock_reason=why)

    if _inside(ctx.cwd, path):
        return _keep(ctx, path, "cwd", "the current working directory is inside it")
    if path == ctx.repo:
        return _keep(ctx, path, "cwd", "it is the --repo argument itself")

    if ctx.cwds is None:
        return _keep(
            ctx, path, "in-use-unknown", "in-use state unknown (lsof could not list process cwds)"
        )
    users = sorted({pid for pid, cwd in ctx.cwds if _inside(cwd, path)})
    if users:
        return _keep(
            ctx,
            path,
            "in-use",
            f"in use: {len(users)} process(es) have their cwd inside it (pid {users[0]}"
            + (", …" if len(users) > 1 else "")
            + ")",
            pids=users[:10],
        )

    # IDLE. Admin files first (a `git checkout`/commit/fetch touches them even if
    # the tree looks quiet), then the bounded tree walk. HEAD is mandatory.
    head_file = os.path.join(gitdir_real, "HEAD")
    admin_mtimes: list[int] = []
    for name in ("HEAD", "index", os.path.join("logs", "HEAD")):
        try:
            admin_mtimes.append(os.lstat(os.path.join(gitdir_real, name)).st_mtime_ns)
        except FileNotFoundError:
            if os.path.join(gitdir_real, name) == head_file:
                return _keep(ctx, path, "idle-unproven", "admin HEAD file unreadable")
        except OSError as exc:
            return _keep(ctx, path, "idle-unproven", f"admin file unreadable: {exc}")
    cutoff_ns = int((ctx.now - ctx.min_idle_s) * 1_000_000_000)
    newest_admin = max(admin_mtimes)
    if newest_admin >= cutoff_ns:
        return _keep(
            ctx,
            path,
            "recent",
            f"git admin files touched {_age(ctx.now, newest_admin)} ago "
            f"(window {_fmt_s(ctx.min_idle_s)})",
        )
    walk = walk_allocated(path, cutoff_ns=cutoff_ns, skip_top_level=frozenset({".git"}))
    if walk.recent:
        return _keep(
            ctx,
            path,
            "recent",
            f"working tree touched within the idle window ({_fmt_s(ctx.min_idle_s)})",
        )
    if walk.error:
        return _keep(ctx, path, "idle-unproven", f"cannot prove idle: {walk.error}")
    if walk.truncated:
        return _keep(
            ctx,
            path,
            "idle-unproven",
            f"more than {WALK_MAX_ENTRIES} entries; idle time unproven",
        )
    nested_git = list(walk.nested_git)
    newest = max(newest_admin, walk.newest_mtime_ns)
    idle_s = ctx.now - newest / 1e9

    # A rebase/merge/cherry-pick/revert/bisect in progress leaves a clean-looking
    # tree whose real state (the todo list, the half-applied commit) lives only in
    # the gitdir that `worktree remove` deletes with it.
    for marker in _OPERATION_MARKERS:
        if os.path.lexists(os.path.join(gitdir_real, marker)):
            return _keep(
                ctx, path, "operation-in-progress", f"git operation in progress ({marker})"
            )

    # CLEAN, WITH `--ignored`. `git worktree remove` (no --force) will happily
    # delete ignored files, and removal destroys NOTHING but working-tree contents
    # (branch refs and objects stay in the shared repository) — so the ignored
    # files are exactly the part of the tree this command could lose for good: a
    # `.env`, a local config, a nested repository with unpushed commits under an
    # ignored `tmp/`. Ignored content therefore KEEPS the worktree. There is no
    # build-cache vocabulary carve-out on purpose: a carve-out is a second
    # classification to keep correct, and the operator can delete a cache dir
    # deliberately in seconds; losing a secret cannot be undone.
    status = _run_git(
        path,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored",
        timeout=HEAVY_TIMEOUT_S,
    )
    if status.returncode != 0:
        return _keep(ctx, path, "uninspectable", "git status failed")
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    changed = [line for line in lines if not line.startswith("!! ")]
    ignored = [line[3:] for line in lines if line.startswith("!! ")]
    if changed:
        return _keep(
            ctx,
            path,
            "dirty",
            f"{len(changed)} uncommitted/untracked change(s)",
            dirty_paths=len(changed),
        )
    if ignored:
        shown = ", ".join(ignored[:5]) + (" …" if len(ignored) > 5 else "")
        return _keep(
            ctx,
            path,
            "ignored-content",
            f"{len(ignored)} gitignored entr{'y' if len(ignored) == 1 else 'ies'} "
            f"would be deleted with it: {shown}",
            ignored_entries=ignored[:20],
        )
    if nested_git:
        return _keep(
            ctx,
            path,
            "ignored-content",
            "contains a nested git repository/submodule whose history lives only here: "
            + ", ".join(os.path.relpath(g, path) for g in nested_git[:3]),
            nested_git=nested_git[:5],
        )

    # PUSHED / MERGED.
    head = _run_git(path, "rev-parse", "--verify", "-q", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        return _keep(ctx, path, "uninspectable", "HEAD does not resolve (unborn?)")
    sha = head.stdout.strip()
    if reg.branch:
        tip = _run_git(path, "rev-parse", "--verify", "-q", reg.branch)
        if tip.returncode != 0 or tip.stdout.strip() != sha:
            return _keep(ctx, path, "uninspectable", f"{reg.branch} does not sit at HEAD")
    contains = _run_git(
        path,
        "for-each-ref",
        "--contains",
        sha,
        "--format=%(refname)",
        "--count=20",
        "refs/remotes",
    )
    if contains.returncode != 0:
        return _keep(ctx, path, "uninspectable", "cannot decide whether a remote contains HEAD")
    remote_refs = [r for r in contains.stdout.split() if not r.endswith("/HEAD")]
    if not remote_refs:
        where = reg.branch or "detached HEAD"
        return _keep(
            ctx,
            path,
            "unpushed",
            f"HEAD {sha[:12]} ({where}) is contained in no remote-tracking ref",
            head=sha,
        )
    merged = False
    if ctx.trunk:
        anc = _run_git(path, "merge-base", "--is-ancestor", sha, ctx.trunk)
        if anc.returncode == 0:
            merged = True
        elif anc.returncode != 1:
            return _keep(ctx, path, "uninspectable", "cannot decide whether HEAD is merged")
    sized = walk_allocated(path, skip_top_level=frozenset({".git"}))
    basis = (
        f"merged into {ctx.trunk.removeprefix('refs/remotes/')}"
        if merged and ctx.trunk
        else f"pushed (contained in {remote_refs[0].removeprefix('refs/remotes/')})"
    )
    return Item(
        id=f"{CATEGORY_WORKTREE}:{path}",
        category=CATEGORY_WORKTREE,
        group=ctx.repo,
        path=path,
        action=ACTION_RECLAIM,
        code="reclaim",
        reason=f"{basis}; clean; idle {_fmt_s(idle_s)}",
        allocated_bytes=sized.allocated_bytes,
        bytes_truncated=sized.truncated or bool(sized.error),
        evidence={
            "repo": ctx.repo,
            "head": sha,
            "branch": reg.branch or None,
            "basis": "merged" if merged else "pushed",
            "remote_refs": remote_refs[:5],
            "trunk": ctx.trunk,
            "idle_seconds": round(idle_s),
            "min_idle_seconds": int(ctx.min_idle_s),
        },
    )


def _age(now: float, mtime_ns: int) -> str:
    return _fmt_s(max(0.0, now - mtime_ns / 1e9))


def _fmt_s(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def plan_worktrees(
    repo_arg: str,
    *,
    now: float,
    min_idle_minutes: int,
    cwds: list[tuple[int, str]] | None,
) -> list[Item]:
    """Assess every registered worktree of one repository."""
    ctx = build_repo_context(repo_arg, now=now, min_idle_minutes=min_idle_minutes, cwds=cwds)
    return [assess_worktree(ctx, reg) for reg in ctx.registrations]


def revalidate_worktree(item: Item, *, min_idle_minutes: int) -> Item:
    """Re-run EVERY predicate for a previewed item, from scratch, right now.

    Fresh registrations, fresh lsof table, fresh clock. The result must also
    still be a reclaim at the SAME HEAD the operator was shown; a worktree that
    moved to another commit between preview and apply is a different candidate
    and is skipped, not removed.
    """
    now = time.time()
    repo = str(item.evidence.get("repo", ""))
    try:
        ctx = build_repo_context(
            repo, now=now, min_idle_minutes=min_idle_minutes, cwds=process_cwds()
        )
    except GitUnavailable as exc:
        return dataclasses.replace(item, action=ACTION_KEEP, code="uninspectable", reason=str(exc))
    for reg in ctx.registrations:
        if os.path.realpath(reg.path) == item.path:
            fresh = assess_worktree(ctx, reg)
            if fresh.action == ACTION_RECLAIM and fresh.evidence.get("head") != item.evidence.get(
                "head"
            ):
                return dataclasses.replace(
                    fresh,
                    action=ACTION_KEEP,
                    code="head-moved",
                    reason="HEAD changed since the preview",
                )
            return fresh
    return dataclasses.replace(
        item, action=ACTION_KEEP, code="invalid-registration", reason="no longer registered"
    )


def remove_worktree(item: Item) -> str | None:
    """``git worktree remove <path>`` from the repository; ``None`` on success.

    WITHOUT ``--force``, on purpose: dirty, locked, missing or unlisted trees are
    git's to refuse, and its refusal is returned as the failure detail.
    """
    primary_arg = str(item.evidence.get("repo", ""))
    try:
        done = _run_git(primary_arg, "worktree", "remove", item.path, timeout=HEAVY_TIMEOUT_S)
    except GitUnavailable as exc:
        return str(exc)
    if done.returncode != 0:
        first = (done.stderr.strip().splitlines() or ["no output"])[0][:200]
        return f"git worktree remove refused: {first}"
    if os.path.exists(item.path):
        return "git reported success but the directory still exists"
    return None


# --------------------------------------------------------------------------
# Go build cache
# --------------------------------------------------------------------------


#: Variables a Go invocation here may inherit. EVERYTHING else is dropped:
#: ``GOFLAGS`` in particular can widen ``go clean -cache`` (``-modcache`` also
#: deletes the module cache — reproduced in review R1-3), and ``GOENV`` /
#: ``go env -w`` files can inject the same flags. Home and temp locations are
#: needed for Go to find its own defaults.
_GO_ENV_PASSTHROUGH = ("PATH", "HOME", "USER", "TMPDIR", "LOCALAPPDATA", "APPDATA")


def _go_env(*, resolve: bool, gocache: str | None = None) -> dict[str, str]:
    """An explicit, minimal environment for a Go subprocess.

    ``resolve=True`` (for ``go env GOCACHE``) additionally forwards the variables
    that decide WHERE Go looks for its cache and ``go env -w`` file, because the
    answer must be the cache the operator's ``go build`` really uses.
    ``resolve=False`` (for ``go clean -cache``) sets ``GOENV=off`` so no config
    file can add flags, clears ``GOFLAGS`` and pins ``GOCACHE`` to the validated
    path. ``GOTOOLCHAIN=local`` stops Go from fetching a toolchain to run.
    """
    env = {k: os.environ[k] for k in _GO_ENV_PASSTHROUGH if k in os.environ}
    env["GOTOOLCHAIN"] = "local"
    env["GOFLAGS"] = ""
    if resolve:
        for key in ("GOENV", "GOCACHE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "GOROOT"):
            if key in os.environ:
                env[key] = os.environ[key]
    else:
        env["GOENV"] = "off"
        if gocache is not None:
            env["GOCACHE"] = gocache
    return env


def resolve_go_cache() -> tuple[str | None, str]:
    """``(validated_path, "")`` or ``(None, refusal_reason)``.

    FAIL CLOSED. ``GOCACHE`` is operator/environment-controlled text that ends up
    as the target of a delete, so it is accepted only when ALL hold: ``go`` is on
    PATH and ``go env GOCACHE`` answers; the value is absolute and not ``off``;
    it is a real directory (not a symlink); it is not ``/``, the home directory
    or any ancestor of home, and does not contain the config directory; and it
    carries Go's own marker ``README``. The marker is the proof it is a Go build
    cache rather than a directory someone pointed ``GOCACHE`` at.
    """
    go = shutil.which("go")
    if not go:
        return None, "go is not installed or not on PATH"
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [go, "env", "GOCACHE"],
            capture_output=True,
            text=True,
            timeout=QUICK_TIMEOUT_S,
            check=False,
            env=_go_env(resolve=True),
            cwd=tempfile.gettempdir(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"`go env GOCACHE` failed: {exc}"
    raw = done.stdout.strip()
    if done.returncode != 0 or not raw:
        return None, "`go env GOCACHE` did not return a path"
    if "\n" in raw or "\x00" in raw:
        return None, "GOCACHE is not a single path"
    if not os.path.isabs(raw):
        return None, f"GOCACHE is not an absolute path ({raw!r})"
    if os.path.islink(raw):
        return None, "GOCACHE is a symlink"
    if not os.path.isdir(raw):
        return None, "GOCACHE does not exist (nothing to reclaim)"
    real = os.path.realpath(raw)
    home = os.path.realpath(os.path.expanduser("~"))
    from local_operator.paths import config_dir

    cfg = os.path.realpath(os.fspath(config_dir()))
    if real == os.sep or _inside(home, real):
        return None, "GOCACHE is the filesystem root, the home directory or one of its parents"
    if _inside(cfg, real):
        return None, "GOCACHE contains the Local Operator config directory"
    marker = os.path.join(real, "README")
    try:
        with open(marker, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(2048)
    except OSError:
        text = ""
    if GO_CACHE_MARKER not in text:
        return None, "GOCACHE lacks the Go build-cache marker README; not a Go cache"
    return real, ""


def plan_go_cache() -> tuple[Item | None, dict[str, str] | None]:
    """The Go cache as a reclaim Item (measure only), or a whole-category refusal."""
    path, why = resolve_go_cache()
    if path is None:
        refusal = {"category": CATEGORY_GO_CACHE, "reason": why}
        item = Item(
            id=f"{CATEGORY_GO_CACHE}:unresolved",
            category=CATEGORY_GO_CACHE,
            group="go",
            path="",
            action=ACTION_KEEP,
            code="refused",
            reason=why,
            outcome=OUTCOME_KEPT,
        )
        return item, refusal
    walk = walk_allocated(path)
    return (
        Item(
            id=f"{CATEGORY_GO_CACHE}:{path}",
            category=CATEGORY_GO_CACHE,
            group="go",
            path=path,
            action=ACTION_RECLAIM,
            code="reclaim",
            reason="Go build cache (cache-only; rebuilt on demand by `go build`)",
            allocated_bytes=walk.allocated_bytes,
            bytes_truncated=walk.truncated or bool(walk.error),
            evidence={"via": "go clean -cache"},
        ),
        None,
    )


def clean_go_cache(item: Item) -> str | None:
    """``go clean -cache`` with ``GOCACHE`` pinned to the validated path.

    Re-resolves first: the path must STILL validate and still equal the previewed
    one. ``None`` on success, else the failure/refusal detail.
    """
    path, why = resolve_go_cache()
    if path is None:
        return f"no longer qualifies: {why}"
    if path != item.path:
        return f"GOCACHE changed since the preview ({item.path} -> {path})"
    go = shutil.which("go")
    if not go:
        return "go is not on PATH"
    env = _go_env(resolve=False, gocache=path)
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [go, "clean", "-cache"],
            capture_output=True,
            text=True,
            timeout=HEAVY_TIMEOUT_S,
            check=False,
            env=env,
            cwd=tempfile.gettempdir(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"go clean -cache failed: {exc}"
    if done.returncode != 0:
        first = (done.stderr.strip().splitlines() or ["no output"])[0][:200]
        return f"go clean -cache failed: {first}"
    return None


# --------------------------------------------------------------------------
# Scratchpads (composition with the existing guarded machinery)
# --------------------------------------------------------------------------


def _scratchpad_policy(root: Path) -> Any:
    from local_operator.config import ConfigManager
    from local_operator.session.cleanup import policy_from_config

    return policy_from_config(ConfigManager(root))


def plan_scratchpads(
    root: Path, *, force: bool, now: float
) -> tuple[list[Item], dict[str, str] | None]:
    """Preview via the SAME dry-run pass ``lop sessions cleanup --scratchpads-only`` runs.

    Same config policy, same retention window (``max_age_hours``), same force
    semantics, same guards (live claim/lease, armed wake/monitor, unread mail,
    current session, manifests) — they live inside ``run_delegated_pass`` and
    are neither re-implemented nor overridable here. The content gate mirrors
    the sessions command: with the delegated/scratchpad switches off the
    category is refused unless ``--force`` (which overrides the SWITCHES only).
    """
    from local_operator.logger import file_logging
    from local_operator.session.delegated_retention import run_delegated_pass

    policy = _scratchpad_policy(root)
    allowed = force or (policy.delegated_enabled and policy.delegated_scratchpad_enabled)
    if not allowed:
        from local_operator.cli import _content_switches_off

        why = (
            f"scratchpad content reclaim is off ({_content_switches_off(policy)}); "
            "turn it on or pass --force to run once"
        )
        return [
            Item(
                id=f"{CATEGORY_SCRATCHPAD}:policy",
                category=CATEGORY_SCRATCHPAD,
                group="sessions",
                path=str(root / "sessions"),
                action=ACTION_KEEP,
                code="policy-off",
                reason=why,
                outcome=OUTCOME_KEPT,
            )
        ], {"category": CATEGORY_SCRATCHPAD, "reason": why}
    with file_logging():
        preview = run_delegated_pass(
            root,
            lambda: policy,
            dry_run=True,
            force=force,
            actor="cli",
            trigger="manual",
            now=now,
            only=(),
        )
    items: list[Item] = []
    for row in preview.content:
        items.append(
            Item(
                id=f"{CATEGORY_SCRATCHPAD}:{row.session}",
                category=CATEGORY_SCRATCHPAD,
                group="sessions",
                path=str(root / "sessions" / row.session / "scratchpad"),
                action=ACTION_RECLAIM,
                code="reclaim",
                reason=", ".join(row.classes) or row.reason or "scratchpad build/cache content",
                allocated_bytes=row.bytes,
                # The pass's own estimate is a stat-based LOWER bound
                # (SIZE_MAX_ENTRIES), so it is always labelled as one.
                bytes_truncated=True,
                evidence={"session": row.session, "classes": list(row.classes)},
            )
        )
    for name, reason in preview.content_kept:
        items.append(
            Item(
                id=f"{CATEGORY_SCRATCHPAD}:{name}",
                category=CATEGORY_SCRATCHPAD,
                group="sessions",
                path=str(root / "sessions" / name / "scratchpad"),
                action=ACTION_KEEP,
                code="guarded",
                reason=reason,
                outcome=OUTCOME_KEPT,
                evidence={"session": name},
            )
        )
    return items, None


def apply_scratchpads(root: Path, items: list[Item], *, force: bool, now: float) -> None:
    """Apply ONLY the previewed pads through ``run_delegated_pass(content_only=…)``.

    The pass re-evaluates every guard at removal time; a pad that stopped
    qualifying simply is not in the result's ``content`` and is reported as
    skipped. Item outcomes are updated in place.
    """
    from local_operator.logger import file_logging
    from local_operator.session.delegated_retention import UNBOUNDED, run_delegated_pass

    names = [str(i.evidence["session"]) for i in items]
    policy = _scratchpad_policy(root)
    try:
        with file_logging():
            result = run_delegated_pass(
                root,
                lambda: policy,
                force=force,
                actor="cli",
                trigger="manual",
                now=now,
                budget_s=UNBOUNDED,
                only=(),
                content_only=names,
            )
    except Exception as exc:  # noqa: BLE001 — fail closed, report every pad
        for item in items:
            item.outcome = OUTCOME_FAILED
            item.detail = f"scratchpad pass raised {type(exc).__name__}: {exc}"
        return
    done = {row.session: row for row in result.content}
    for item in items:
        row = done.get(str(item.evidence["session"]))
        if row is None:
            item.outcome = OUTCOME_SKIPPED
            item.detail = "no longer qualifies at removal time (a guard cleared or tripped)"
        else:
            item.outcome = OUTCOME_REMOVED
            item.allocated_bytes = row.bytes
    if result.errors:
        for item in items:
            if item.outcome == OUTCOME_REMOVED:
                item.detail = f"pass reported {result.errors} error(s); see the log"


# --------------------------------------------------------------------------
# Build output — the fourth category
# --------------------------------------------------------------------------


def build_output_idle_minutes(min_idle_minutes: int) -> int:
    """The effective idle window for build output, in minutes.

    ``max`` and not the flag alone: the module-wide rule is that
    ``--min-idle-minutes`` may RAISE a window and can never lower it, and this
    category's own default is :data:`BUILD_OUTPUT_IDLE_MINUTES`. Composing the
    two keeps both promises. The value used is stamped into every item's evidence
    so a receipt reader can see which window judged it instead of inferring it.
    """
    return max(int(min_idle_minutes), BUILD_OUTPUT_IDLE_MINUTES)


def read_process_table() -> list[tuple[int, tuple[str, ...]]] | None:
    """``[(pid, argv)]`` for every process ``ps`` reports, or ``None``.

    ``ps -axo pid=,command=`` because it is present on every supported platform
    and cheap. ``ps`` reports no cwd and does not delimit argv, so argv is
    approximated by whitespace splitting and the cwd is filled in from
    :func:`read_open_handles` (one ``lsof`` sweep already has it). Both
    approximations fail in the SAFE direction: a path containing a space can make
    a process unattributable by argv, and the cwd rule then covers it.

    FAIL CLOSED: a missing ``ps``, a timeout, a non-zero exit or a parse that
    yields no rows is ``None`` ("cannot tell"), never "nothing is running" —
    every supported platform has at least one process, so an empty table means
    the output was not what we expect.

    THIS COMMAND'S OWN PROCESS IS EXCLUDED. ``lop disk reclaim --build-output
    <root>`` is a generic ``python`` producer, so its argv is never what
    attributes it and the cwd rule decides — which made the planner itself the
    live build it was looking for whenever it was run from inside the named
    root. The row is dropped here, in the reader, so no attribution path can see
    it; :func:`read_open_handles` drops the same pid for the same reason.
    """
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=QUICK_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    rows: list[tuple[int, tuple[str, ...]]] = []
    own_pid = os.getpid()
    for line in done.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, rest = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            return None
        if pid == own_pid:
            continue
        rows.append((pid, tuple(rest.split())))
    return rows or None


def read_open_handles() -> tuple[str, ...] | None:
    """Every open path plus every process cwd in ONE ``lsof`` sweep, or ``None``.

    A failed ``lsof`` means "everything may be in use", never "nothing is": a
    stale disk is recoverable, a build directory deleted under another session's
    compile is not.

    The exit-status rule is the one already proven in :func:`process_cwds`:
    ``lsof`` exits non-zero when it could not stat some files — normal on a
    shared machine — while still listing thousands of handles, so a usable parse
    is trusted and only an EMPTY parse is ``None``. A row counts only when its
    PID column is an integer, because the header splits into the same nine
    columns as a data row and counting it made ``('NAME',)`` — a confident
    "nothing is open" — out of output that carried no handle at all.

    cwd rows come back as ``pid:<n>:cwd=<path>`` sentinels: attribution needs the
    cwd, ``ps`` does not carry it, and a second sweep would double the most
    expensive call in a plan.

    Names are returned as ``lsof`` reports them (normalised, not resolved): the
    candidate side is already a real path, so the common case matches, and a
    handle opened through a DIFFERENT symlinked route to the same file is a
    documented blind spot of the same shape the worktree category documents for
    its own lsof use. Resolving tens of thousands of names per plan would cost
    more than the guard is worth on the machine this exists for.

    TWO PIDS ARE EXCLUDED — this process and the ``lsof`` child it just spawned.
    Both sit in whatever directory the operator ran the command from, ``lsof``
    reports its OWN cwd (measured: a ``lsof`` row for the sweep itself), and a
    cwd sentinel inside the candidate was a confident ``handles-open`` on a tree
    nobody was using. Excluding exactly those pids keeps every OTHER process in
    that same directory attributable, which is the direction this guard exists
    for. The child's pid is only knowable from :class:`subprocess.Popen`, which
    is why this is not a ``subprocess.run``.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            ["lsof", "-n", "-P", "-w"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        stdout, _ = proc.communicate(timeout=QUICK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    own_pids = {os.getpid(), proc.pid}
    paths: list[str] = []
    for line in (stdout or "").splitlines():
        # NAME may contain spaces, so split the fixed leading columns and keep
        # the remainder whole: COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME
        fields = line.split(None, 8)
        if len(fields) != 9:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if pid in own_pids:
            continue
        name = fields[8].strip()
        if not name:
            continue
        if fields[3] == "cwd":
            paths.append(f"pid:{fields[1]}:cwd={name}")
        else:
            paths.append(os.path.normpath(name))
    return tuple(paths) or None


def cwds_by_pid(handles: Iterable[str]) -> dict[int, str]:
    """The cwd sentinels from :func:`read_open_handles`, keyed by pid."""
    out: dict[int, str] = {}
    for entry in handles:
        if not entry.startswith("pid:"):
            continue
        pid_text, _, tail = entry[len("pid:") :].partition(":cwd=")
        try:
            out[int(pid_text)] = tail
        except ValueError:
            continue
    return out


#: Characters that may legitimately precede a path inside one argv element (the
#: flag glue in ``--out-dir=/w/target/debug``), and characters that may follow
#: it. Used by :func:`path_in_arg`.
_ARG_SEPARATORS = "=:, \t('\""
_PATH_FOLLOWERS = "/:, '\")"


def path_in_arg(arg: str, targets: Iterable[str]) -> bool:
    """Whether ``arg`` CONTAINS one of ``targets`` as a whole path.

    Needed because argv elements carry values glued to a flag or an ``=``
    (``--out-dir=/w/target/debug``), where an exact or prefix test on the element
    misses. The match is boundary-aware — the target must start the element or
    follow a separator, and end it or be followed by one — so a sibling checkout
    whose name merely EXTENDS this one (``repo-old`` against ``repo``) is not
    attributed to it. That precision matters on a machine running ~25 concurrent
    sessions: an over-match here blocks a real reclaim every time a neighbour
    compiles, which is how a reclaimer becomes a no-op nobody trusts.

    Deliberately attribute-only: it decides whether a build MAY be running, so a
    false positive costs one kept candidate while a false negative corrupts
    somebody's compile.
    """
    for target in targets:
        if not target:
            continue
        start = arg.find(target)
        while start != -1:
            end = start + len(target)
            before_ok = start == 0 or arg[start - 1] in _ARG_SEPARATORS
            after_ok = end == len(arg) or arg[end] in _PATH_FOLLOWERS
            if before_ok and after_ok:
                return True
            start = arg.find(target, start + 1)
    return False


def producer_kind(argv: Iterable[str]) -> str | None:
    """Which kind of build producer, if any, this process's argv names.

    Every element is checked, not just ``argv[0]``: ``next build`` runs as
    ``node .../node_modules/.bin/next`` and a wrapper invocation is common.
    Checking more elements can only produce more blocks, which is the safe
    direction — EXCEPT for the generic names, which are attributed by cwd only
    (see :data:`BUILD_OUTPUT_GENERIC_PRODUCERS`).

    A VERSIONED interpreter counts as its base name: ``python3.12``, ``python3``
    and ``pip3`` are the producer names with a version glued on, and matching only
    the bare spelling would silently drop the ``dist`` producers on the machine
    this runs on.
    """
    for arg in argv:
        base = os.path.basename(arg)
        if base in BUILD_OUTPUT_PRODUCERS:
            name: str | None = base
        else:
            stem = base.rstrip("0123456789.")
            name = stem if stem and stem != base and stem in BUILD_OUTPUT_PRODUCERS else None
        if name is None:
            continue
        return PRODUCER_GENERIC if name in BUILD_OUTPUT_GENERIC_PRODUCERS else PRODUCER_SPECIFIC
    return None


def is_producer_process(argv: Iterable[str]) -> bool:
    """Whether any argv element names a build producer of either kind."""
    return producer_kind(argv) is not None


#: How much of the PROJECT ROOT may count for attribution.
#:
#: Only two levels are needed here (the three-level rule the reference reaper
#: carries exists for scan roots it DISCOVERS; this category's roots are named by
#: the operator, so the ambiguous case is home itself). See
#: :func:`build_output_parent_scope`.
_PARENT_SCOPE_FULL = "full"
_PARENT_SCOPE_NONE = "none"


def build_output_parent_scope(project_root: str) -> str:
    """Which parent-attribution scope a named project root may use.

    ``NONE`` when the root IS the home directory, ``FULL`` otherwise.

    Home is not a scope at all: it CONTAINS every project its owner has, so a
    parent-prefix rule there attributes any toolchain process anywhere under it
    and makes the category a permanent no-op on an idle machine. A directory the
    operator NAMED as a project root is the opposite claim — this is the
    repository whose build was named — so the whole of it counts, including the
    exact ``cwd == project root`` case: a build running in the repository must
    block its own ``target/``, which is the whole point of the guard.
    """
    home = os.path.realpath(os.path.expanduser("~"))
    return _PARENT_SCOPE_NONE if os.path.realpath(project_root) == home else _PARENT_SCOPE_FULL


def process_touches(
    argv: Iterable[str],
    cwd: str | None,
    *,
    candidate: str,
    project_root: str,
    parent_scope: str = _PARENT_SCOPE_FULL,
    ignore_argv: bool = False,
) -> bool:
    """Whether ONE process is attributable to ``candidate``.

    Three ways: an argv element equal to or under the candidate; an argv element
    equal to or under the candidate's PROJECT ROOT (``cargo`` is invoked with
    ``--manifest-path <root>/Cargo.toml`` and ``rustc`` with ``--out-dir
    <root>/target/debug/deps``, both of which name the root rather than the
    candidate verbatim); or a cwd at/under that root.

    Scoped by ``parent_scope`` so a root that CONTAINS everything (home) cannot
    swallow the whole tree: see :func:`build_output_parent_scope`. The candidate
    term is never scoped — a process naming the build directory itself is
    evidence wherever the directory is.

    Every path is compared in BOTH spellings, as given and resolved. On macOS
    ``/var`` is a symlink to ``/private/var``, so a candidate the operator named
    through the one spelling and a cwd ``lsof`` reports through the other would
    otherwise never match — the guard would fail OPEN, which is the one direction
    it must not fail.
    """
    candidate_forms = {os.path.normpath(candidate), os.path.realpath(candidate)}
    root_forms = {os.path.normpath(project_root), os.path.realpath(project_root)}
    for arg in () if ignore_argv else argv:
        if any(arg == text or arg.startswith(text + os.sep) for text in candidate_forms):
            return True
        if path_in_arg(arg, tuple(candidate_forms)):
            return True
        if parent_scope == _PARENT_SCOPE_FULL:
            if any(arg == text or arg.startswith(text + os.sep) for text in root_forms):
                return True
            if path_in_arg(arg, tuple(root_forms)):
                return True
    if cwd:
        cwd_text = os.path.normpath(cwd)
        if cwd_text in candidate_forms:
            return True
        if parent_scope == _PARENT_SCOPE_FULL and any(
            cwd_text == text or cwd_text.startswith(text + os.sep) for text in root_forms
        ):
            return True
    return False


@dataclass
class BuildProbes:
    """The live-build and provenance probes for build output, injectable.

    Each answers ``True`` / ``False`` / ``None`` for ONE candidate, and ``None`` —
    "the instrument could not be trusted" — BLOCKS. Tests construct this with
    lambdas; production uses :func:`default_build_probes`. Mirrors the shape the
    worktree category gets from :func:`process_cwds`, with the tri-state the
    build-output predicates need.
    """

    toolchain_running: Callable[[str, str], bool | None]
    handles_open: Callable[[str], bool | None]
    tracked_files: Callable[[str], bool | None]
    is_git_ignored: Callable[[str], bool | None]
    refresh: Callable[[], None] = lambda: None


class _Tables:
    """The two process tables a plan reads, loaded once and refreshable."""

    def __init__(self) -> None:
        self.processes: list[tuple[int, tuple[str, ...]]] | None = None
        self.handles: tuple[str, ...] | None = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        self.processes = read_process_table()
        self.handles = read_open_handles()
        self.loaded = True

    def refresh(self) -> None:
        self.processes = None
        self.handles = None
        self.loaded = False


def toolchain_touches(
    candidate: str,
    project_root: str,
    *,
    processes: Iterable[tuple[int, tuple[str, ...]]],
    handles: Iterable[str],
) -> bool:
    """Whether any process in ``processes`` is attributable to ``candidate``.

    Split out from :func:`default_build_probes` so the attribution rule — in
    particular the generic-producer asymmetry — is testable against a synthetic
    process table instead of whatever happens to be running on the machine.
    """
    scope = build_output_parent_scope(project_root)
    cdws = cwds_by_pid(handles)
    for pid, argv in processes:
        kind = producer_kind(argv)
        if kind is None:
            continue
        if process_touches(
            argv,
            cdws.get(pid),
            candidate=candidate,
            project_root=project_root,
            parent_scope=scope,
            ignore_argv=kind == PRODUCER_GENERIC,
        ):
            return True
    return False


def default_build_probes() -> BuildProbes:
    """The production probes, bound to one refreshable snapshot of the tables.

    One ``ps`` and one ``lsof`` per snapshot rather than per candidate: the
    snapshot is refreshed immediately before a removal (see
    :func:`revalidate_build_output`), so the answer that decides a delete is read
    at removal time while a preview of N candidates still costs two probes.
    """
    tables = _Tables()

    def toolchain_running(candidate: str, project_root: str) -> bool | None:
        tables.load()
        if tables.processes is None:
            return None
        return toolchain_touches(
            candidate, project_root, processes=tables.processes, handles=tables.handles or ()
        )

    def handles_open(candidate: str) -> bool | None:
        tables.load()
        if tables.handles is None:
            return None
        exact = os.path.normpath(candidate)
        # Both spellings, for the reason :func:`process_touches` documents: a
        # one-spelling comparison against ``lsof`` output fails OPEN on macOS.
        forms = {exact, os.path.realpath(candidate)}
        prefixes = {text + os.sep for text in forms}
        for entry in tables.handles:
            # A ``pid:<n>:cwd=<path>`` sentinel still names a path in use: a
            # process sitting IN the directory holds it as much as one with a
            # file open beneath it.
            path_text = entry.split(":cwd=", 1)[1] if entry.startswith("pid:") else entry
            path_text = os.path.normpath(path_text)
            if path_text in forms or any(path_text.startswith(p) for p in prefixes):
                return True
        return False

    return BuildProbes(
        toolchain_running=toolchain_running,
        handles_open=handles_open,
        tracked_files=probe_tracked_files,
        is_git_ignored=probe_git_ignored,
        refresh=tables.refresh,
    )


#: git's own words for "there is no repository at or above here". The
#: discriminator for exit 128 — see :func:`inside_any_repository`.
_NOT_A_REPOSITORY = "not a git repository"


def build_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str] | None:
    """One git query through this module's hardened helper, or ``None``.

    ``None`` means git could not be trusted to answer at all (missing binary, a
    timeout, an undecodable stream), and every caller turns it into a KEEP. The
    helper itself is reused from ``delegated_retention`` rather than copied: a
    second environment recipe (``XPC_FLAGS`` dropped, repo-redirecting variables
    scrubbed, prompts and system config off) would be a way for the two to drift.
    """
    try:
        return _git(cwd, *args, timeout=QUICK_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def inside_any_repository(cwd: str) -> bool | None:
    """Whether git considers ``cwd`` part of a repository: ``True``/``False``/``None``.

    The discriminator for exit 128, which is NEVER evidence by itself: ``git -C
    <parent> ls-files`` exits 128 both when the parent is genuinely outside every
    repository AND when it is a repository git refuses to describe from outside —
    the case where a linked worktree's own ``.git`` sat inside a candidate and a
    confident "nothing is tracked" answered about the WRONG repository.
    ``rev-parse --show-toplevel`` is the query because it is read-only, cheap (one
    extra call on the 128 branch only) and answered for a bare repository or a
    ``GIT_DIR`` override as well as a worktree.
    """
    done = build_git(cwd, "rev-parse", "--show-toplevel")
    if done is None:
        return None
    if done.returncode == 0:
        return True
    if done.returncode == 128 and _NOT_A_REPOSITORY in (done.stderr or ""):
        return False
    return None


def probe_tracked_files(candidate: str) -> bool | None:
    """Whether any git-TRACKED file lives beneath ``candidate``.

    ``True`` is the dangerous answer: a tracked file there means part of what
    would be deleted is committed content, so nothing is removed regardless of
    the shape's name. ``ls-files`` runs from the PARENT with the bare directory
    name as pathspec (no absolute-pathspec edge cases) and lists recursively,
    which is the "must be empty" test the contract states.

    Outside any repository the answer is ``False`` — nothing there can be tracked
    — which is why a ``target/`` in a plain directory stays reclaimable. That
    ``False`` needs PROOF, though: a 128 becomes ``False`` only when
    :func:`inside_any_repository` says so, and every other failure (git missing, a
    timeout, an undecodable stream, a corrupt repo, a permissions wall) is
    ``None`` — cannot tell, so the candidate is kept.
    """
    parent = os.path.dirname(candidate)
    done = build_git(parent, "ls-files", "--", os.path.basename(candidate))
    if done is None:
        return None
    if done.returncode == 128:
        return False if inside_any_repository(parent) is False else None
    if done.returncode != 0:
        return None
    return bool(done.stdout.strip())


def probe_git_ignored(candidate: str) -> bool | None:
    """Whether ``candidate`` is git-IGNORED, which is the shape's real claim.

    The guard this buys is narrow but real: an untracked AND not-ignored ``dist/``
    is a directory ``git status`` shows by hand — build output in a repository
    that chose not to ignore it is an owner decision, so it is kept rather than
    assumed. Tracked content is caught by :func:`probe_tracked_files`; this is the
    neighbouring case.

    Outside any repository the answer is ``True``: no ignore rule exists to
    violate and nothing can be tracked, so the manifest is the whole safety case
    (and the tracked probe independently confirms the second half). Same proof
    requirement as that probe: a 128 becomes ``True`` only when
    :func:`inside_any_repository` confirms there is genuinely no repository, and
    is otherwise ``None``.
    """
    parent = os.path.dirname(candidate)
    done = build_git(parent, "check-ignore", "-q", "--", os.path.basename(candidate))
    if done is None:
        return None
    if done.returncode == 0:
        return True
    if done.returncode == 1:
        return False
    if done.returncode == 128:
        return True if inside_any_repository(parent) is False else None
    return None


#: The entries every git DIRECTORY carries: the HEAD pointer, the object store
#: and the ref namespace. Used by :func:`is_git_directory`.
_GIT_DIR_ENTRIES = ("HEAD", "objects", "refs")


def has_git_entry(candidate: str) -> bool | None:
    """Whether ``candidate`` carries a ``.git`` entry (directory OR file).

    ``True`` means the candidate is a repository or a linked-worktree root, not
    build output. A linked worktree's ``.git`` is a FILE pointing at the parent
    repository, so a directory-only check misses exactly the case that would take
    a worktree's branch and uncommitted work with it — the repro is ``git worktree
    add .worktrees/target -b feat/wip``, where the worktree's own basename is a
    shape name and its parent is a real project root. ``None`` (the entry could
    not be stat-ed) keeps the candidate; only ``FileNotFoundError`` is ``False``.
    """
    try:
        os.lstat(os.path.join(candidate, ".git"))
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return True


def is_git_directory(candidate: str) -> bool | None:
    """Whether ``candidate`` IS a git directory rather than containing one.

    The residual :func:`has_git_entry` cannot see: ``git init --bare <shape>``
    creates no ``.git`` entry at all — the directory itself holds ``HEAD``,
    ``objects/`` and ``refs/`` — so a repository guard that only looks for
    ``.git`` answers "not a repository" and the objects and refs go with it. The
    same layout is what ``git init --separate-git-dir`` leaves when the separate
    store is the candidate.

    A pure filesystem test on purpose: no subprocess, so no locale, timeout or
    missing-binary answer to distrust, and every layout git creates carries these
    three entries.
    """
    for name in _GIT_DIR_ENTRIES:
        try:
            os.lstat(os.path.join(candidate, name))
        except FileNotFoundError:
            return False
        except OSError:
            return None
    return True


def symlink_in_path(path: str, *, stop_at: str) -> str | None:
    """The first component of ``path`` (down from ``stop_at``) that is a symlink.

    A candidate is kept when it IS a symlink and when its path TRAVERSES one: the
    operator named a real project directory, and a symlink in between means the
    bytes that would be deleted are not provably the ones the manifests were found
    beside. ``OSError`` propagates — a component that cannot be inspected is
    "cannot tell", and the caller keeps the candidate.
    """
    current = os.path.abspath(stop_at)
    if stat.S_ISLNK(os.lstat(current).st_mode):
        return current
    rest = os.path.relpath(os.path.abspath(path), current)
    if rest in (".", ""):
        return None
    for part in rest.split(os.sep):
        current = os.path.join(current, part)
        if stat.S_ISLNK(os.lstat(current).st_mode):
            return current
    return None


def _in_config_dir(path: str, config_root: Path) -> bool:
    """Whether ``path`` is the config dir or anything below it.

    Both spellings, like every other path comparison here: ``config_dir()`` is
    what the caller resolved and the named root is what the operator typed.
    """
    root = os.path.abspath(os.path.expanduser(os.fspath(config_root)))
    forms = (os.path.abspath(path), os.path.realpath(path))
    roots = (root, os.path.realpath(root))
    return any(_inside(form, base) for form in forms for base in roots)


def protected_location_reason(project_root: str) -> str | None:
    """Why ``project_root`` may not be a build-output root, or ``None`` when it may.

    ``$HOME`` and the standard data directories below it
    (:data:`PROTECTED_DATA_DIRS`) are not projects, so one stray manifest there
    must not turn them into one: ``~/package.json`` would otherwise make
    ``~/dist`` "provably owned build output", and the same manifest dropped in
    ``~/Documents`` would do it to ``~/Documents/build``. The Go-cache category
    refuses ``$HOME`` for the same reason. A named project INSIDE one of these
    directories is still fine — that is the operator naming a project.

    Both spellings are compared, like every path comparison in this module: the
    root is what the operator typed (``abspath``, never ``realpath`` — see
    :func:`plan_build_output`) and home is what the environment says either way.
    """
    home = os.path.expanduser("~")
    home_forms = {os.path.abspath(home), os.path.realpath(home)}
    root_forms = {os.path.abspath(project_root), os.path.realpath(project_root)}
    for form in root_forms:
        if form in home_forms:
            return (
                f"{project_root} is your home directory; name the project directory "
                "inside it instead"
            )
        for home_form in home_forms:
            if (
                os.path.dirname(form) == home_form
                and os.path.basename(form) in PROTECTED_DATA_DIRS
            ):
                return (
                    f"{project_root} is a standard data directory; the folders in it "
                    "are your data, never build output — name the project directory "
                    "inside it"
                )
    return None


def manifests_present(project_root: str, manifests: Iterable[str]) -> tuple[str, ...]:
    """Which of one shape's manifests exist as files in ``project_root``.

    An ``OSError`` on the check is not evidence, so that manifest does not count:
    the provenance gate can only open on a manifest that was SEEN. An empty answer
    means "this is not that kind of project" and the directory is not a candidate.
    """
    found: list[str] = []
    for name in manifests:
        try:
            if os.path.isfile(os.path.join(project_root, name)):
                found.append(name)
        except OSError:
            continue
    return tuple(found)


def _keep_build_output(candidate: str, code: str, reason: str, **evidence: Any) -> Item:
    """One explained keep for the build-output category.

    ``project_root`` travels in ``evidence`` rather than as an argument: most call
    sites in :func:`assess_build_output` spread one evidence dict that already
    carries it, and a named parameter would then be bound twice.
    """
    return Item(
        id=f"{CATEGORY_BUILD_OUTPUT}:{candidate}",
        category=CATEGORY_BUILD_OUTPUT,
        group=str(evidence.get("project_root") or os.path.dirname(candidate)),
        path=candidate,
        action=ACTION_KEEP,
        code=code,
        reason=reason,
        outcome=OUTCOME_KEPT,
        evidence=dict(evidence),
    )


def assess_build_output(
    candidate: str,
    *,
    project_root: str,
    manifests: tuple[str, ...],
    now: float,
    min_idle_minutes: int,
    probes: BuildProbes,
) -> Item:
    """Decide ONE candidate build directory: a reclaim Item or an explained keep.

    Every predicate fails toward KEEP. The order is cheap-first and
    safety-ordered. Symlink and mount point are two ``lstat`` calls and answer a
    question no name can. The ``.git`` guards come BEFORE the git probes on
    purpose: those probes ask from the PARENT, and for a worktree whose own
    basename is a shape (``.worktrees/target``) the parent's answer describes a
    different repository entirely — a confident answer about the wrong repository
    is the dangerous case, and the ``.git``-inside guard is what closes it. Age is
    next, because the bounded walk also stops early the moment a tree is provably
    too young. The three structural findings that same walk can turn up — a mount
    point it had to stop at, a standard data directory it refused to descend into
    (:data:`PROTECTED_DATA_DIRS`), a nested ``.git`` — are judged immediately
    after it: every one means the tree was either not fully measured or holds
    something no rebuild can recreate. git provenance follows, and the live-build
    probes are last: they are the most expensive, and their "cannot tell" is the
    one that costs most to get wrong.
    """
    window = build_output_idle_minutes(min_idle_minutes)
    cutoff_ns = int((now - window * 60) * 1e9)
    evidence: dict[str, Any] = {
        "project_root": project_root,
        "manifests": list(manifests),
        "idle_minutes": window,
    }

    try:
        linked = symlink_in_path(candidate, stop_at=project_root)
    except OSError as exc:
        return _keep_build_output(
            candidate,
            "probe-failed",
            f"the path could not be inspected ({exc.strerror or exc})",
            **evidence,
        )
    if linked:
        return _keep_build_output(
            candidate,
            "symlink",
            f"a path component is a symlink ({linked}); nothing is followed or removed",
            **evidence,
        )

    try:
        mounted = os.path.ismount(candidate)
    except OSError as exc:  # a probe that cannot answer is not a go-ahead
        return _keep_build_output(
            candidate,
            "probe-failed",
            f"the mount probe could not answer ({type(exc).__name__})",
            **evidence,
        )
    if mounted:
        return _keep_build_output(
            candidate,
            "mount-point",
            "it is the root of its own filesystem; removing it would walk onto another volume",
            **evidence,
        )

    git_entry = has_git_entry(candidate)
    if git_entry is None:
        return _keep_build_output(
            candidate, "probe-failed", "its .git entry could not be stat-ed", **evidence
        )
    if git_entry:
        return _keep_build_output(
            candidate,
            "git-repository",
            "it contains a .git entry, so it is a repository or a linked-worktree root",
            **evidence,
        )
    git_dir = is_git_directory(candidate)
    if git_dir is None:
        return _keep_build_output(
            candidate,
            "probe-failed",
            "its git-directory entries could not be stat-ed",
            **evidence,
        )
    if git_dir:
        return _keep_build_output(
            candidate,
            "git-repository",
            "it IS a git directory (bare or --separate-git-dir layout), not build output",
            **evidence,
        )

    walk = walk_allocated(
        candidate,
        cutoff_ns=cutoff_ns,
        stop_at_mount=True,
        prune_names=PROTECTED_DATA_DIRS,
    )
    evidence["newest_mtime"] = (
        _iso(walk.newest_mtime_ns / 1e9) if walk.newest_mtime_ns else None
    )
    if walk.error:
        return _keep_build_output(
            candidate,
            "idle-unproven",
            f"the tree could not be read ({walk.error}), so idle cannot be proven",
            **evidence,
        )
    if walk.mount_crossed:
        return _keep_build_output(
            candidate,
            "mount-point",
            "the walk crosses onto another filesystem at "
            f"{os.path.relpath(walk.mount_crossed, candidate)}; that volume's "
            "contents are not ours to delete",
            **evidence,
        )
    if walk.pruned:
        return _keep_build_output(
            candidate,
            "protected-location",
            "it contains a standard data directory ("
            + ", ".join(os.path.relpath(path, candidate) for path in walk.pruned[:3])
            + "), whose contents are the user's own data",
            **evidence,
        )
    if walk.nested_git:
        return _keep_build_output(
            candidate,
            "git-repository",
            "it contains a nested .git ("
            + ", ".join(os.path.relpath(path, candidate) for path in walk.nested_git[:3])
            + "), so a repository's history lives only inside it",
            **evidence,
        )
    if walk.recent:
        return _keep_build_output(
            candidate,
            "recent",
            f"an entry was written within the last {window // 60}h; rerun after it is quiet",
            **evidence,
        )
    if walk.truncated:
        return _keep_build_output(
            candidate,
            "idle-unproven",
            f"the walk hit its {WALK_MAX_ENTRIES}-entry bound, so idle cannot be proven",
            bytes_lower_bound=walk.allocated_bytes,
            **evidence,
        )

    tracked = probes.tracked_files(candidate)
    if tracked is None:
        return _keep_build_output(
            candidate,
            "git-unknown",
            "git could not say whether files inside it are tracked",
            **evidence,
        )
    if tracked:
        return _keep_build_output(
            candidate,
            "git-tracked",
            "git tracks files inside it, so part of it is committed content",
            **evidence,
        )
    ignored = probes.is_git_ignored(candidate)
    if ignored is None:
        return _keep_build_output(
            candidate,
            "git-unknown",
            "git could not say whether it is ignored",
            **evidence,
        )
    if not ignored:
        return _keep_build_output(
            candidate,
            "git-not-ignored",
            "git does not ignore it, so it is not provably build output",
            **evidence,
        )

    handles = probes.handles_open(candidate)
    if handles is None:
        return _keep_build_output(
            candidate,
            "probe-failed",
            "lsof could not establish which files are open",
            **evidence,
        )
    if handles:
        return _keep_build_output(
            candidate,
            "handles-open",
            "a process holds files open inside it",
            **evidence,
        )
    running = probes.toolchain_running(candidate, project_root)
    if running is None:
        return _keep_build_output(
            candidate,
            "probe-failed",
            "the process table could not be read, so a live build cannot be ruled out",
            **evidence,
        )
    if running:
        return _keep_build_output(
            candidate,
            "toolchain-running",
            "a build toolchain process is working in it",
            **evidence,
        )

    return Item(
        id=f"{CATEGORY_BUILD_OUTPUT}:{candidate}",
        category=CATEGORY_BUILD_OUTPUT,
        group=project_root,
        path=candidate,
        action=ACTION_RECLAIM,
        code="reclaim",
        reason=(
            f"{os.path.basename(candidate)}/ proven by {'/'.join(manifests)} and "
            f"idle for at least {window // 60}h"
        ),
        allocated_bytes=walk.allocated_bytes,
        bytes_truncated=walk.truncated,
        evidence=evidence,
    )


def plan_build_output(
    paths: Iterable[str],
    *,
    now: float,
    min_idle_minutes: int,
    config_root: Path | None = None,
    probes: BuildProbes | None = None,
) -> tuple[list[Item], list[dict[str, str]]]:
    """Assess every operator-named project directory. Returns ``(items, refusals)``.

    A named path that does not exist, is not a directory, or LOOKS like a build
    directory itself (its basename is a shape name and it holds no manifest of its
    own) is a whole-category refusal keyed by that path: the operator asked about
    something this command cannot answer, and a silent empty preview would read as
    "nothing to reclaim". That is the difference between a refusal and a keep —
    a keep is a CANDIDATE this plan decided not to touch, and a directory with no
    manifest is not a candidate at all, so it is never even listed.

    Nothing is discovered. The candidates are the fixed shape names inside each
    named directory, one level deep, and no shape becomes a candidate unless the
    directory holds a manifest that proves the project owns it.

    A named root inside the config dir (``config_root``) is refused by name. The
    sessions store and every session scratchpad live there, this category's
    remover is the one deleter in this module, and the scratchpads category
    already reclaims what belongs below that path under its own guards (live
    claim/lease, retention window, manifests). Refusing here is what lets the
    no-session-deletion guard prove that nothing this module deletes can reach a
    session directory.

    A named root that is ``$HOME`` or one of the standard data directories below
    it is refused for the reason :func:`protected_location_reason` gives: the
    provenance gate proves a project owns a shape directory, and it must not be
    possible to make somebody's own folder into one by leaving a manifest in it.
    """
    resolved = probes if probes is not None else default_build_probes()
    items: list[Item] = []
    refusals: list[dict[str, str]] = []
    seen: set[str] = set()
    every_manifest = tuple(
        dict.fromkeys(name for names in BUILD_OUTPUT_SHAPES.values() for name in names)
    )
    for raw in paths:
        given = os.fspath(raw)
        # ``abspath``, NOT ``realpath``: resolving a named root would silently
        # redirect the delete at whatever a symlink points to, so a symlinked
        # root is kept by the predicate instead of being followed here. The
        # probes compare both spellings (see :func:`process_touches`), so nothing
        # is lost by keeping this one.
        root = os.path.abspath(os.path.expanduser(given))
        # ONE directory however it is spelled, the same way `--repo P --repo P`
        # collapses: enumerating it twice would double-count bytes and make apply
        # report the second copy as a failure.
        if root in seen:
            continue
        seen.add(root)
        if config_root is not None and _in_config_dir(root, config_root):
            refusals.append(
                {
                    "category": CATEGORY_BUILD_OUTPUT,
                    "path": given,
                    "reason": (
                        f"{given} is inside the config dir; session scratchpads are "
                        "reclaimed by --scratchpads under its own guards"
                    ),
                }
            )
            continue
        protected = protected_location_reason(root)
        if protected:
            refusals.append(
                {"category": CATEGORY_BUILD_OUTPUT, "path": given, "reason": protected}
            )
            continue
        if not os.path.isdir(root):
            refusals.append(
                {
                    "category": CATEGORY_BUILD_OUTPUT,
                    "path": given,
                    "reason": f"{given} is not a directory (name a project directory)",
                }
            )
            continue
        if os.path.basename(root) in BUILD_OUTPUT_SHAPES and not manifests_present(
            root, every_manifest
        ):
            refusals.append(
                {
                    "category": CATEGORY_BUILD_OUTPUT,
                    "path": given,
                    "reason": (
                        f"{given} is named like build output and holds no project manifest; "
                        "name the directory that contains it"
                    ),
                }
            )
            continue
        for name, manifests in BUILD_OUTPUT_SHAPES.items():
            candidate = os.path.join(root, name)
            try:
                mode = os.lstat(candidate).st_mode
            except FileNotFoundError:
                continue
            except OSError:
                items.append(
                    _keep_build_output(
                        candidate,
                        "probe-failed",
                        "it could not be inspected",
                        project_root=root,
                        manifests=list(manifests),
                    )
                )
                continue
            if not (stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
                # A plain file that happens to carry a shape's name is not a
                # build directory and there is nothing to reclaim in it.
                continue
            found = manifests_present(root, manifests)
            if not found:
                # THE PROVENANCE GATE: no manifest, no candidate — not a keep,
                # not a refusal, simply not this category's business.
                continue
            items.append(
                assess_build_output(
                    candidate,
                    project_root=root,
                    manifests=found,
                    now=now,
                    min_idle_minutes=min_idle_minutes,
                    probes=resolved,
                )
            )
    return items, refusals


def revalidate_build_output(
    item: Item, *, min_idle_minutes: int, probes: BuildProbes | None = None
) -> Item:
    """Re-run EVERY predicate for a previewed item, from scratch, right now.

    Called at removal time and nowhere else, so the answer that decides a delete
    is read then rather than carried over from the preview: the gap between the
    preview and the delete is the race this closes. The probes are refreshed
    first for the same reason — one ``ps`` and one ``lsof`` of the moment we are
    about to delete in.
    """
    if probes is not None:
        probes.refresh()
    root = item.evidence.get("project_root") or os.path.dirname(item.path)
    protected = protected_location_reason(root)
    if protected:
        return _keep_build_output(
            item.path, "protected-location", protected, project_root=root
        )
    manifests = tuple(
        item.evidence.get("manifests")
        or BUILD_OUTPUT_SHAPES.get(os.path.basename(item.path), ())
    )
    if not manifests:
        return _keep_build_output(
            item.path,
            "no-manifest",
            "no manifest in the table proves this shape any more",
            project_root=root,
        )
    found = manifests_present(root, manifests)
    if not found:
        return _keep_build_output(
            item.path,
            "no-manifest",
            f"the manifest that proved it is gone ({', '.join(manifests)})",
            project_root=root,
        )
    fresh = assess_build_output(
        item.path,
        project_root=root,
        manifests=found,
        now=time.time(),
        min_idle_minutes=min_idle_minutes,
        probes=probes if probes is not None else default_build_probes(),
    )
    fresh.id = item.id
    fresh.outcome = item.outcome
    return fresh


def _raise_walk_error(exc: OSError) -> None:
    """``os.walk`` error hook that RAISES: an unreadable directory is a failure."""
    raise exc


def _remove_tree(root: str) -> None:
    """Delete one tree bottom-up, never following a symlink out of it.

    ``os.walk(topdown=False, followlinks=False)`` visits the deepest directory
    first, so every directory is already empty when its own ``rmdir`` runs. A
    link entry — to a file or to a directory — is UNLINKED, never descended:
    that is the "never follows a symlink out of the tree" guarantee stated as
    code rather than as intent, and it holds even for a link created between the
    preview and this call.

    ``onerror`` re-raises instead of being ignored: the default swallows an
    unreadable subdirectory, which would turn "could not look" into "part of it
    was removed". An entry that vanished underneath us is fine
    (``FileNotFoundError`` is a no-op); every other ``OSError`` reaches the
    caller, which reports a partial removal instead of a receipt claiming space
    it never got back.
    """
    for directory, dirnames, filenames in os.walk(
        root, topdown=False, followlinks=False, onerror=_raise_walk_error
    ):
        for name in dirnames + filenames:
            entry = os.path.join(directory, name)
            try:
                if stat.S_ISLNK(os.lstat(entry).st_mode):
                    os.unlink(entry)
                elif os.path.isdir(entry):
                    os.rmdir(entry)
                else:
                    os.unlink(entry)
            except FileNotFoundError:
                continue
    try:
        os.rmdir(root)
    except FileNotFoundError:
        return


def remove_build_output(
    item: Item,
    *,
    min_idle_minutes: int,
    config_root: Path | None = None,
    probes: BuildProbes | None = None,
) -> str | None:
    """Remove ONE previewed build-output tree. ``None`` on success.

    The single tree remover this module documents in its docstring, and it owns
    every guard that makes that acceptable:

    1. :func:`revalidate_build_output` re-runs EVERY predicate from scratch, and
       an item that no longer qualifies is SKIPPED with the failing predicate
       named — never removed "because it was approved earlier";
    2. the shape of the entry itself is re-checked against ``lstat``: a symlink
       or a mount point that appeared since the preview is refused, and so is a
       path that stopped being a directory — plus a mount point that appeared
       INSIDE the tree, which the measurement walk stops at;
    3. the tree is measured again immediately before deletion, and that
       measurement is what the receipt carries for this item — the preview's
       figure and the removal's figure are different numbers, and the reader can
       see both;
    4. :func:`_remove_tree` never follows a symlink out of the tree.

    The return value is a detail string: a prefix of ``no longer qualifies`` or
    ``refusing`` means the item was skipped (it stopped qualifying), anything else
    is a failure. Both make the run exit 1, which is the honest code for "this
    previewed item did not get reclaimed".
    """
    if config_root is not None and _in_config_dir(item.path, config_root):
        return "refusing: it is inside the config dir"
    fresh = revalidate_build_output(item, min_idle_minutes=min_idle_minutes, probes=probes)
    if fresh.action != ACTION_RECLAIM:
        return f"no longer qualifies: [{fresh.code}] {fresh.reason}"

    try:
        mode = os.lstat(item.path).st_mode
    except OSError as exc:
        return f"refusing: it could not be stat-ed at removal time ({exc.strerror or exc})"
    if stat.S_ISLNK(mode):
        return "refusing: it is a symlink at removal time"
    if not stat.S_ISDIR(mode):
        return "refusing: it is no longer a directory at removal time"
    try:
        if os.path.ismount(item.path):
            return "refusing: it is a mount point at removal time"
    except OSError as exc:
        return f"refusing: the mount probe failed ({exc.strerror or exc})"

    walk = walk_allocated(item.path, stop_at_mount=True)
    if walk.error:
        return f"refusing: the tree could not be measured ({walk.error})"
    if walk.mount_crossed:
        # The same race as the mount check above, one level down: the walk stops at
        # the boundary, so nothing past it was measured — and `_remove_tree` would
        # happily descend onto the other volume.
        return (
            "refusing: a mount point appeared inside it at removal time "
            f"({walk.mount_crossed})"
        )
    item.allocated_bytes = walk.allocated_bytes
    item.bytes_truncated = walk.truncated
    item.evidence["allocated_bytes_at_removal"] = walk.allocated_bytes
    item.evidence["bytes_truncated_at_removal"] = walk.truncated
    item.evidence["entries_at_removal"] = walk.entries
    try:
        _remove_tree(item.path)
    except OSError as exc:
        return f"removal failed part-way ({exc.strerror or exc}); the tree is now partial"
    if os.path.lexists(item.path):
        # A remover that cannot prove its own end state reports failure rather
        # than a receipt claiming space it did not get back.
        return "removal returned without error but the path still exists"
    return None


# --------------------------------------------------------------------------
# Plan / apply / receipt
# --------------------------------------------------------------------------


def disk_readings(paths: Iterable[str]) -> list[dict[str, Any]]:
    """``shutil.disk_usage`` for each distinct device the paths live on."""
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for path in paths:
        probe = path
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        try:
            device = os.stat(probe).st_dev
            if device in seen:
                continue
            usage = shutil.disk_usage(probe)
        except OSError:
            continue
        seen.add(device)
        out.append(
            {
                "path": probe,
                "device": device,
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "read_at": _iso(time.time()),
            }
        )
    return out


def plan(selection: Selection, config_root: Path, *, now: float | None = None) -> Plan:
    """Build the preview. Reads and measures; removes nothing; never raises on a candidate."""
    selection.validate()
    moment = time.time() if now is None else now
    result = Plan(selection=selection, now=moment)
    result.probe_paths.append(os.fspath(config_root))
    if selection.repos:
        cwds = process_cwds()
        # ONE repository however it is spelled: `--repo P --repo P` or a linked
        # worktree of P both resolve to the same shared git dir. Enumerating it
        # twice would double-count bytes and make apply report the second copy
        # of an already-removed worktree as a failure (a false exit 1).
        seen_common: set[str] = set()
        for repo_arg in selection.repos:
            try:
                ctx = build_repo_context(
                    repo_arg,
                    now=moment,
                    min_idle_minutes=selection.min_idle_minutes,
                    cwds=cwds,
                )
                if ctx.common_dir in seen_common:
                    continue
                seen_common.add(ctx.common_dir)
                items = [assess_worktree(ctx, reg) for reg in ctx.registrations]
            except GitUnavailable as exc:
                # A --repo that is not a repository (or git itself is broken) is
                # a whole-repo refusal: nothing under it is enumerated, nothing
                # is guessed in its place.
                result.refusals.append(
                    {"category": CATEGORY_WORKTREE, "repo": repo_arg, "reason": str(exc)}
                )
                continue
            result.items.extend(items)
            result.probe_paths.append(os.path.realpath(os.path.expanduser(repo_arg)))
    if selection.go_build_cache:
        item, refusal = plan_go_cache()
        if item is not None:
            result.items.append(item)
            if item.path:
                result.probe_paths.append(item.path)
        if refusal:
            result.refusals.append(refusal)
    if selection.scratchpads:
        items, refusal = plan_scratchpads(config_root, force=selection.force, now=moment)
        result.items.extend(items)
        if refusal:
            result.refusals.append(refusal)
    if selection.build_output:
        items, refusals = plan_build_output(
            selection.build_output,
            now=moment,
            min_idle_minutes=selection.min_idle_minutes,
            config_root=config_root,
        )
        result.items.extend(items)
        result.refusals.extend(refusals)
        # The operator's named directories get the disk reading, not the individual
        # candidates: the honest "what was freed" figure is about the filesystem
        # each project lives on, and a candidate is already gone by the time the
        # AFTER reading is taken.
        for named in selection.build_output:
            result.probe_paths.append(os.path.realpath(os.path.expanduser(os.fspath(named))))
    return result


def _receipt_payload(
    plan_: Plan,
    *,
    mode: str,
    status: str,
    started_at: float,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]] | None,
    finished_at: float | None,
) -> dict[str, Any]:
    items = [i.as_dict() for i in plan_.items]
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "disk-reclaim-receipt",
        "mode": mode,
        "status": status,
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at) if finished_at else None,
        "selection": plan_.selection.as_dict(),
        "measurement_note": MEASUREMENT_NOTE,
        "disk_usage_before": before,
        "disk_usage_after": after,
        "refusals": plan_.refusals,
        "items": items,
        "summary": summarize(plan_),
    }


def summarize(plan_: Plan) -> dict[str, Any]:
    by_outcome: dict[str, int] = {}
    measured = 0
    truncated = False
    for item in plan_.items:
        by_outcome[item.outcome] = by_outcome.get(item.outcome, 0) + 1
        if item.action == ACTION_RECLAIM and item.allocated_bytes:
            measured += item.allocated_bytes
            truncated = truncated or item.bytes_truncated
    return {
        "reclaimable": len(plan_.reclaimable),
        "kept": len(plan_.kept),
        "by_outcome": by_outcome,
        "measured_allocated_bytes": measured,
        "measured_is_lower_bound": truncated,
        "measured_is_not_reclaimed_bytes": True,
    }


def preview_payload(plan_: Plan) -> dict[str, Any]:
    return _receipt_payload(
        plan_,
        mode="dry-run",
        status="preview",
        started_at=plan_.now,
        before=disk_readings(plan_.probe_paths),
        after=None,
        finished_at=None,
    )


def write_receipt_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a receipt atomically, through the SANCTIONED session manifest writer.

    Delegates to ``delegated_retention._write_cleanup_manifest`` rather than
    carrying its own tmp+fsync+``os.replace``+``unlink`` sequence: that writer
    is already the reviewed/allow-listed shape for exactly this operation
    (temp beside target, fsync, rename, cleanup on failure), and duplicating
    it would grow the no-session-deletion allow-list for no behaviour gain.

    The receipt directory is created 0o700 here first, so the whole receipt
    tree stays private. The manifest writer's temp file inherits the process
    umask on rename (0o644 typically), which is acceptable: a receipt carries
    only paths and sizes, never secrets, inside that private directory.
    """
    from local_operator.session.delegated_retention import _write_cleanup_manifest

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_cleanup_manifest(path, payload)


@dataclass
class ApplyResult:
    payload: dict[str, Any]
    receipt_path: Path | None
    exit_code: int


def apply(
    plan_: Plan,
    config_root: Path,
    *,
    confirmation: str,
    on_progress: Callable[[Item], None] | None = None,
) -> ApplyResult:
    """Apply the previewed reclaim items, re-validating each at removal time.

    ``confirmation`` is one of :data:`CONFIRMATION_KINDS` and REQUIRED, not
    defaulted: it is stamped into every receipt write below, including the
    INTENT written before the first removal, so the audit record says how the run
    was authorised even if the process dies mid-sweep. The CLI passes ``--yes``
    or ``typed``; a caller that cannot say which applies gets ``ValueError``
    rather than a receipt that claims the wrong provenance.

    ORDER OF EFFECTS: (1) take the before reading, (2) write the receipt as an
    INTENT (every item still ``planned``) — if that fails nothing is touched and
    the result is a refusal (exit 2), because an unrecorded removal is exactly
    what this command exists to avoid; (3) per item: re-validate, remove through
    the guarded remover, rewrite the receipt with the outcome; (4) take the
    after reading and finalise. An unexpected exception on an item keeps it
    (``failed``) and the loop continues.

    Exit codes, in precedence order: 2 a category was refused outright (reported,
    nothing there touched) or the intent receipt could not be written; else 1 any
    previewed item was skipped or failed, or a receipt update failed; else 0.
    """
    # A caller may change Selection after preview. Reject before even writing
    # an intent, not only when a worktree happens to reach revalidation.
    plan_.selection.validate()
    if confirmation not in CONFIRMATION_KINDS:
        raise ValueError(
            f"confirmation must be one of {CONFIRMATION_KINDS} (got {confirmation!r}); "
            "the receipt records it"
        )
    started = time.time()

    def recorded(payload: dict[str, Any]) -> dict[str, Any]:
        # Stamped on the document itself, not only on the value the CLI prints
        # afterwards, so the file on disk and stdout cannot disagree about how
        # the removal was confirmed.
        payload["confirmed"] = True
        payload["confirmation"] = confirmation
        return payload

    path = config_root / RECEIPT_DIRNAME / f"{utc_stamp(started)}-{os.getpid()}.json"
    before = disk_readings(plan_.probe_paths)
    try:
        write_receipt_atomic(
            path,
            recorded(
                _receipt_payload(
                    plan_,
                    mode="apply",
                    status="intent",
                    started_at=started,
                    before=before,
                    after=None,
                    finished_at=None,
                )
            ),
        )
    except OSError as exc:
        payload = recorded(
            _receipt_payload(
                plan_,
                mode="apply",
                status="refused",
                started_at=started,
                before=before,
                after=None,
                finished_at=time.time(),
            )
        )
        payload["error"] = f"could not write the intent receipt ({exc}); nothing was removed"
        return ApplyResult(payload, None, 2)

    receipt_warnings: list[str] = []

    def checkpoint(status: str = "in-progress") -> None:
        # A failed mid-run receipt write must not stop removal bookkeeping, but it
        # must not be silent either: the file on disk is now OLDER than reality,
        # so the failure is recorded, surfaced by the CLI and turns the exit code
        # into a failure.
        try:
            write_receipt_atomic(
                path,
                recorded(
                    _receipt_payload(
                        plan_,
                        mode="apply",
                        status=status,
                        started_at=started,
                        before=before,
                        after=None,
                        finished_at=None,
                    )
                ),
            )
        except OSError as exc:
            receipt_warnings.append(f"receipt checkpoint failed ({exc}); the file may be stale")

    # Worktrees, Go cache and build output are per-item; scratchpads are one
    # batched, confinement-by-name pass.
    build_probes = default_build_probes()
    scratch_items = [
        i for i in plan_.items if i.category == CATEGORY_SCRATCHPAD and i.action == ACTION_RECLAIM
    ]
    for item in plan_.items:
        if item.action != ACTION_RECLAIM or item.category == CATEGORY_SCRATCHPAD:
            continue
        try:
            if item.category == CATEGORY_WORKTREE:
                fresh = revalidate_worktree(item, min_idle_minutes=plan_.selection.min_idle_minutes)
                if fresh.action != ACTION_RECLAIM:
                    item.outcome = OUTCOME_SKIPPED
                    item.detail = f"no longer qualifies: [{fresh.code}] {fresh.reason}"
                else:
                    error = remove_worktree(item)
                    item.outcome = OUTCOME_FAILED if error else OUTCOME_REMOVED
                    item.detail = error or ""
            elif item.category == CATEGORY_GO_CACHE:
                error = clean_go_cache(item)
                if error and error.startswith(("no longer qualifies", "GOCACHE changed")):
                    item.outcome = OUTCOME_SKIPPED
                else:
                    item.outcome = OUTCOME_FAILED if error else OUTCOME_REMOVED
                item.detail = error or ""
            elif item.category == CATEGORY_BUILD_OUTPUT:
                # The one remover this module owns; see its docstring for the
                # guards it re-applies rather than trusting the preview.
                error = remove_build_output(
                    item,
                    min_idle_minutes=plan_.selection.min_idle_minutes,
                    config_root=config_root,
                    probes=build_probes,
                )
                if error and error.startswith(("no longer qualifies", "refusing")):
                    item.outcome = OUTCOME_SKIPPED
                else:
                    item.outcome = OUTCOME_FAILED if error else OUTCOME_REMOVED
                item.detail = error or ""
            else:  # pragma: no cover — unknown category: keep
                item.outcome = OUTCOME_SKIPPED
                item.detail = "unknown category"
        except Exception as exc:  # noqa: BLE001 — fail closed: keep and report
            item.outcome = OUTCOME_FAILED
            item.detail = f"unexpected {type(exc).__name__}: {exc}"
        if on_progress:
            on_progress(item)
        checkpoint()
    if scratch_items:
        apply_scratchpads(config_root, scratch_items, force=plan_.selection.force, now=plan_.now)
        for item in scratch_items:
            if on_progress:
                on_progress(item)
        checkpoint()

    failed = any(
        i.outcome in (OUTCOME_FAILED, OUTCOME_SKIPPED)
        for i in plan_.items
        if i.action == ACTION_RECLAIM
    )
    failed = failed or bool(receipt_warnings)
    status = "partial" if failed else "complete"
    finished = time.time()
    payload = recorded(
        _receipt_payload(
            plan_,
            mode="apply",
            status=status,
            started_at=started,
            before=before,
            after=disk_readings(plan_.probe_paths),
            finished_at=finished,
        )
    )
    payload["receipt_path"] = os.fspath(path)
    if receipt_warnings:
        payload["receipt_warnings"] = receipt_warnings
        failed = True
    try:
        write_receipt_atomic(path, payload)
    except OSError as exc:
        payload["error"] = f"could not finalise the receipt ({exc})"
        failed = True
    # PRECEDENCE: a refused category (2) outranks item failures (1), which
    # outrank success (0). A script that sees 2 knows part of what it asked for
    # was refused outright even when something else also went wrong.
    code = 2 if plan_.refusals else (1 if failed else 0)
    return ApplyResult(payload, path, code)
