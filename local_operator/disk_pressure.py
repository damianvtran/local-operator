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

    There is no ``shutil.rmtree``, ``os.remove`` or ``rm`` anywhere in this
    file, and none may be added: every removal belongs to a guarded remover that
    owns its own refusal rules.

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


def walk_allocated(
    root: str,
    *,
    cutoff_ns: int | None = None,
    max_entries: int = WALK_MAX_ENTRIES,
    skip_top_level: frozenset[str] = frozenset(),
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

    # Worktrees and Go cache are per-item; scratchpads are one batched,
    # confinement-by-name pass.
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
