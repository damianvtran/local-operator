"""Standing build-cache reaper: reclaim regenerable build output, only when idle.

WHY THIS EXISTS
---------------
On 2026-11-07 this machine's disk hit 99% full (4 GiB free) because one agent
lane ran two full Rust rebuilds and left 12 GiB of ``target/`` behind in a
shared service checkout. Nothing prunes build output: the toolchain never
collects it, ``lop-disk-guard.sh`` only watches for a runaway *build*, and a
human notices at 99% — when the failure mode is unrelated sessions dying
mid-write. This module is the standing reaper for that class of space: it
removes regenerable build output, and it logs every removal so the space is
accountable after the fact.

The asymmetry that shapes every rule below: a disk that is too full is
recoverable by deleting; a build directory removed out from under a running
compile is not (it corrupts an in-flight build for whoever owns it), and a
deleted piece of someone's *work* is not recoverable at all. So every check is
tri-state and **"cannot tell" means SKIP** — never "nothing is running".

WHAT COUNTS AS BUILD OUTPUT, AND WHY EACH SHAPE IS (OR IS NOT) ON THE LIST
------------------------------------------------------------------------
Membership is a judgement about the directory's *contract*, not about its size
or how regenerable it looks. That distinction is the whole safety argument, and
it is not academic: the disk lane measured ``out/`` at 14.83 GiB in one repo on
the same night this reaper was specified — capture frames from merged pull
requests, i.e. **evidence**, merely looking like packaging output. "Large and
regenerable-looking" is therefore not the test, and ``out/`` is deliberately
not on any list.

Default ON:

``target``
    Rust/Cargo. The toolchain itself owns this path (``CARGO_TARGET_DIR``
    default), creates it as a sibling of ``Cargo.toml``, and treats its
    contents as a pure function of source + flags + toolchain: deleting it
    costs a rebuild and nothing else. Cargo also documents it as disposable,
    and ``cargo clean`` exists precisely for it. It is also the shape with the
    measured harm — the 12 GiB above.

``.next``
    Next.js. Written by the framework's own build into a dot-directory at the
    project root; ``.next`` is in Next's default ``.gitignore`` stanza, and no
    ecosystem treats a dot-directory named ``.next`` as hand-authored source.
    Dot-prefixed is the same reason ``.venv`` / ``.git`` need no argument.

Default OFF, behind one opt-in key (``include_extra_shapes``), because their
semantics vary by ecosystem and a wrong guess destroys source:

``build``
    CMake, Gradle, Maven and Xcode write here — but so do a great many
    hand-named source directories, and ``build`` is a plain (non-dot) word a
    developer types on purpose. In a Gradle or CMake repo it is unambiguously
    output; in the next checkout it is somebody's ``build/`` of scripts. A
    default-on rule would have to pick one of those and be wrong about the
    other, so this one is the user's call.

``dist``
    Usually packaging output (``npm run build``, Python ``sdist``/``wheel``,
    ``dist/`` in a Rust binary crate) — but it is *sometimes a committed
    artifact directory*: a repo that ships its built bundle has ``dist/`` under
    version control, which is a decision only its owner can make. The
    git-tracked rule below already protects a committed one, but an
    *untracked-and-unignored* ``dist/`` is an owner decision too, so the shape
    waits for the same opt-in as ``build``.

An opt-in shape is only as safe as the producers the live-build probe knows
about: ``_is_toolchain_process`` answering ``False`` means "nothing is
running", not "cannot tell", and the file-handle probe cannot stand in for it
(a compiler between translation units holds nothing open). So
``TOOLCHAIN_EXECUTABLES`` carries the producer of **every** shape this module
can reap, not just the default two — see the constant's own comment for the
measured asymmetry that made that mandatory.

Never, by construction (not by configuration): ``node_modules`` (its own
package manager owns the lifecycle, and a re-install is not cheap), ``.git``
(work lives there), ``.venv`` (recreating is minutes of work and a wrong
deletion is invisible until the next command fails), the config dir
(credentials and transcripts — private, and ``~/.local-operator`` also holds
every session scratchpad), ``~/Library`` (application state),
``~/Trash``/``.Trash`` (the user's own recoverable copy is not ours to empty),
and the standard data directories (``~/Documents``, ``~/Desktop``,
``~/Downloads``, ``~/Movies``, ``~/Music``, ``~/Pictures``): a ``target/``
under one of those is the user's own tree, and nothing about the name makes
its contents regenerable.

Two guards are properties of the DIRECTORY rather than of its name, so no
shape can opt out of them. A candidate that contains a ``.git`` entry — a
directory, or the ``.git`` FILE a linked worktree carries — is a repository or
worktree root, not build output, and is skipped as ``git-repository``:
removing it takes a branch and any uncommitted work with it. A candidate that
is itself a mount point is skipped as ``mount-point``: ``rmtree`` would walk
onto another volume and delete what it finds there.

A CONFIG THAT CANNOT BE TRUSTED STOPS THE TICK
---------------------------------------------
``read_settings`` never invents a configuration. A *missing* ``config.yml`` is
the documented defaults, but a config that exists and cannot be read or
validated raises :class:`ReaperConfigError` and the tick deletes nothing — the
run is refused with a named error instead. The alternative (fall back to the
module defaults) fails in the direction that destroys data: the defaults are
``enabled=True, dry_run=False``, so a corrupt file would turn a switched-OFF
reaper into a live one.

THE LIVE-BUILD GUARD IS THE PROCESS AND HANDLE PROBE, NOT THE MTIME
------------------------------------------------------------------
Two independent questions decide a candidate:

1. **Is anything using it right now?** — per candidate, from the process table
   and the open-file table. This is the load-bearing guard, and it is
   *per-candidate* rather than global on measured grounds: while the 12 GiB of
   ``target/`` sat idle, this box had a real cargo-clippy+rustc build running in
   *another* worktree (``~/block-of-the-dead/.worktrees/prep-day/target/debug``).
   A "any toolchain process anywhere → skip everything" rule would have made the
   reaper a permanent no-op on a machine that runs ~25 concurrent agent
   sessions, which is the same as not having it.

   A candidate is blocked when an open handle exists anywhere beneath it, or a
   toolchain process is attributable to it — its argv mentions the candidate (or
   the worktree that contains it), or its cwd is at/under that worktree. Each of
   those answers ``True`` / ``False`` / ``None``; ``None`` ("ps failed",
   "lsof missing", "could not parse") blocks. A process table that cannot be
   read at all aborts the tick, because then nothing can be attributed.

2. **Is it old?** — the age floor (``age_hours``, default 6). This is defence
   in depth for the window between a finished build and the next one, never the
   primary guard: ``pytest-tmp-reaper`` measured a legitimately running suite
   still working in a 13-hour-old directory, so age alone deletes live work.

   Age is the NEWEST mtime in the tree, not the directory's own — measured:
   ``~/workspace/pr3154-review`` carries a newest file stamped 1970-01-01, so a
   directory-mtime rule would call it 56 years old and delete it instantly.
   Directories contribute their mtime too (a tree of empty directories has no
   files to read), which can only make a candidate look *younger*, the safe
   direction. The walk is bounded; if the bound is hit before the tree is
   enumerated the age is UNKNOWN and the candidate is skipped as
   ``cannot-tell``. A partial maximum understates recency, which is the unsafe
   direction, so there is no fallback to it.

BYTES ARE ALLOCATED BLOCKS, NOT APPARENT SIZE
---------------------------------------------
``reclaimed`` is ``st_blocks * 512`` summed over the tree's files. Apparent
size overcounts hardlinked and APFS-clone trees badly (measured on this box:
``~/workspace`` reported 44.36 GiB by ``du`` while its children summed to
46.76 GiB), and this machine's worktrees are clone-based by policy, so apparent
size would systematically overstate what the reaper gave back. Both numbers are
reported; ``df`` free before/after is the honest cross-check, since it is the
only figure that includes metadata and allocation rounding.

SHAPE OF THE MODULE
-------------------
**Stdlib only at module scope**, like :mod:`local_operator.buildwatch`: this is
invoked from launchd and must not pay for the harness's import graph. Anything
heavy — :mod:`local_operator.config`, :mod:`local_operator.settings_io` — is
imported function-locally inside :func:`read_settings`, the one place that needs
it.

The four tunables live in the ``maintenance.build_cache`` config section (see
:func:`read_settings` and the ``SETTINGS`` registry): ``enabled``,
``age_hours``, ``dry_run`` and ``include_extra_shapes``. The defaults are the
module constants at the top, and ``tests/unit/test_settings_io.py``'s
``_consumer_defaults()`` pins the registry rows to them so the page cannot
advertise a default the reaper does not use.

``--root`` exists so the reaper can be rehearsed and bounded: without it there
is no way to point a manual run at a fixture tree or at one checkout, and
pointing ``HOME`` at a scratch tree is exactly what this harness forbids for
rigs. It is a CLI option only — not a settings key — so the registry stays at
four rows.
"""

from __future__ import annotations

import argparse
import dataclasses
import json as jsonlib
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Iterator, Sequence

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Defaults. These are the REAL defaults: ``local_operator/settings_io.py``
# carries the literals its page shows, and tests/unit/test_settings_io.py's
# ``_consumer_defaults()`` maps each registry row onto the constant here.
# --------------------------------------------------------------------------

#: Whether the standing reaper runs at all. ON by default: the failure it
#: prevents (a full disk breaking unrelated sessions mid-write) is worse than
#: the cost of a rebuild, and it only ever removes provably idle output.
DEFAULT_ENABLED = True

#: Age floor, hours since the newest write in the tree. Six hours is
#: ``pytest-tmp-reaper``'s measured choice for the same class of problem, and
#: low enough that an overnight build is reclaimable by morning.
DEFAULT_AGE_HOURS = 6

#: Report without deleting. OFF by default; a scheduled run has to actually
#: reclaim to be worth scheduling, and every removal is logged either way.
DEFAULT_DRY_RUN = False

#: Whether ``build``/``dist`` are candidates too. OFF — see the module docstring
#: for why these two, alone, are the user's call rather than ours.
DEFAULT_INCLUDE_EXTRA_SHAPES = False

#: Shape directory names pruned with no opt-in.
DEFAULT_SHAPES: tuple[str, ...] = ("target", ".next")

#: Shape directory names that only become candidates under
#: ``maintenance.build_cache.include_extra_shapes``.
EXTRA_SHAPES: tuple[str, ...] = ("build", "dist")

#: Directory names never descended into and never candidates, at any depth.
#: ``node_modules`` is here for cost as much as safety (it is the largest thing
#: on most of this fleet and is never a build shape), ``.git`` because work
#: lives there, ``.venv`` because recreating one is minutes of work whose loss
#: is invisible until the next command fails.
PRUNED_DIRNAMES: frozenset[str] = frozenset({".git", "node_modules", ".venv"})

#: Absolute paths (and their subtrees) excluded from the scan. The config dir
#: also covers every session scratchpad (``sessions/*/scratchpad``) and the
#: encrypted store; ``Trash`` is listed in both spellings (the macOS one and the
#: conventional one).
#:
#: The six data directories are here because a reaper with DEFAULT settings
#: reaped ``~/Documents/target``, ``~/Documents/thesis/dist``, ``~/Movies/build``
#: and ``~/Desktop/target`` on a fixture home: rule 1 says never anything under a
#: data directory, and for these the *shape name* is no evidence at all — a
#: ``build/`` in ``~/Documents`` is a person's tree, not a compiler's output.
#: Excluded by path rather than by name, so a checkout inside one of them is
#: protected too: build output is regenerable anywhere, and "it was in my
#: Documents" is not a risk worth pricing.
PRUNED_SUBTREES_AT_HOME: tuple[str, ...] = (
    "Library",
    "Trash",
    ".Trash",
    "Documents",
    "Desktop",
    "Downloads",
    "Movies",
    "Music",
    "Pictures",
)

#: Scan depth, counted from each root (root = 0). Six reaches
#: ``~/src/org/repo/crate/target`` with room to spare; a deeper tree is more
#: likely to be a vendored copy than an agent worktree.
MAX_DEPTH = 6

#: Wall-clock budget for one tick's SCAN, seconds. Hitting it sets
#: ``incomplete_scan`` — candidates already proved idle are still pruned, which
#: is the honest middle: a partial scan must not become a full skip.
DEFAULT_TIME_BUDGET_S = 90.0

#: Bounds on the per-candidate tree walk (age + bytes). 200k entries or 20 s is
#: an order of magnitude above the measured largest ``target/`` on this box; a
#: tree past it is skipped as unmeasurable rather than guessed at.
WALK_MAX_ENTRIES = 200_000
WALK_BUDGET_S = 20.0

#: Subprocess timeouts. A hung ``lsof`` on a machine at 99% disk is exactly the
#: case this tool exists for, so no probe may block a tick indefinitely.
PROCESS_PROBE_TIMEOUT_S = 15.0
GIT_PROBE_TIMEOUT_S = 10.0

#: Log file name under :func:`local_operator.paths.log_dir` — macOS
#: ``~/Library/Logs/local-operator/build-cache-reaper.log``, elsewhere the
#: platform log dir or ``$LOCAL_OPERATOR_CONFIG_DIR/logs``.
LOG_FILENAME = "build-cache-reaper.log"

#: Executable names that mean "a build may be running".
#:
#: The list covers the producer of EVERY shape this module can reap, because a
#: name that is missing here is not a gap in coverage — it is a confident FALSE
#: from :func:`_is_toolchain_process` ("nothing is running"), which is the
#: dangerous answer, and the file-handle probe is no substitute: a compiler
#: between translation units holds nothing open. Measured on this box with
#: otherwise identical stand-ins, an in-flight build named ``cargo`` kept its
#: ``target/`` while one named ``cmake`` or ``gradle`` lost its ``build/``.
#:
#: ``clippy-driver`` and ``sccache`` compile and cache on behalf of cargo;
#: ``next`` and the JS bundlers run through ``node``/``npm``/``pnpm``/``yarn``,
#: hence the package managers. ``javac`` is in the list because Gradle and Maven
#: compile through it. Checking more names can only produce more blocks, which is
#: the safe direction, and attribution stays per-candidate
#: (:func:`_process_touches`), so a neighbour's build still does not freeze this
#: one.
TOOLCHAIN_EXECUTABLES: frozenset[str] = frozenset(
    {
        # The two DEFAULT shapes (``target``, ``.next``).
        "cargo",
        "rustc",
        "rustdoc",
        "clippy-driver",
        "sccache",
        "next",
        # ``build`` / ``dist`` producers, reachable behind
        # ``include_extra_shapes``: CMake/Make/Ninja, the JVM pair (through
        # javac), Xcode, and the JS bundlers/type-checkers plus their package
        # managers.
        "cmake",
        "make",
        "ninja",
        "gradle",
        "mvn",
        "maven",
        "javac",
        "xcodebuild",
        "vite",
        "tsc",
        "webpack",
        "node",
        "npm",
        "pnpm",
        "yarn",
    }
)

#: Characters that may legitimately precede a path inside one argv element (the
#: flag glue in ``--out-dir=/w/target/debug``), and characters that may follow it.
#: See :func:`target_in_arg` for why the match is boundary-aware.
_ARG_SEPARATORS = "=:, \t('\""
_PATH_FOLLOWERS = "/:, '\")"

# Config keys, spelled once. The registry rows carry the same dotted keys.
KEY_ENABLED = "maintenance.build_cache.enabled"
KEY_AGE_HOURS = "maintenance.build_cache.age_hours"
KEY_DRY_RUN = "maintenance.build_cache.dry_run"
KEY_INCLUDE_EXTRA_SHAPES = "maintenance.build_cache.include_extra_shapes"

# Skip reasons. Strings rather than an enum because they land verbatim in the
# log line and in ``--json``, where a human reads them.
SKIP_SYMLINK = "symlink"
SKIP_FRESH = "fresh"
SKIP_TOOLCHAIN = "toolchain-running"
SKIP_HANDLES = "handles-open"
SKIP_GIT_TRACKED = "git-tracked"
SKIP_GIT_NOT_IGNORED = "git-not-ignored"
#: The candidate IS a repository or linked-worktree root (it carries a ``.git``
#: entry). Named separately from ``git-tracked`` because the fix differs: nothing
#: about the directory's contents is wrong, the directory itself is not output.
SKIP_GIT_REPOSITORY = "git-repository"
SKIP_MOUNT_POINT = "mount-point"
SKIP_CANNOT_TELL = "cannot-tell"
SKIP_ERROR = "error"

# Exit codes. ``0`` is a completed tick even when it removed nothing; the two
# non-zero codes are both "the run told you why it could not do its job", which
# a scheduled job needs in order to be diagnosable at all.
EXIT_OK = 0
#: The configuration exists but could not be read or validated. Nothing was
#: scanned and nothing was deleted.
EXIT_CONFIG_ERROR = 2
#: Every scan root was unreadable (missing, or a permission wall). A scheduled
#: run that reports "0 candidates" for this is indistinguishable from "nothing to
#: do", so it says so and exits non-zero.
EXIT_NO_READABLE_ROOT = 3


# --------------------------------------------------------------------------
# Report records
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TreeScan:
    """What one bounded walk of a candidate tree learned.

    ``stopped`` names the bound that ended the walk early (``"entries"`` or
    ``"time"``) and is the ONLY thing that makes the numbers partial: a value of
    ``None`` means the whole tree was enumerated. ``saw_fresh`` is set the moment
    an entry newer than the age cutoff is seen, which lets :func:`scan_tree`
    stop early on the common case (a live build directory) without pretending
    the partial totals mean anything.
    """

    newest_mtime: float | None
    allocated_bytes: int
    apparent_bytes: int
    entries: int
    stopped: str | None
    saw_fresh: bool


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A candidate the scan proved idle and old enough to remove."""

    path: Path
    shape: str
    age_hours: float
    allocated_bytes: int
    apparent_bytes: int


@dataclasses.dataclass(frozen=True)
class Removal:
    """A candidate that was actually deleted."""

    path: Path
    shape: str
    age_hours: float
    allocated_bytes: int
    apparent_bytes: int


@dataclasses.dataclass(frozen=True)
class Skip:
    """A candidate that was not touched, and why.

    ``detail`` names the bound or probe that produced a ``cannot-tell`` (or the
    specific guard that matched), so a tree that is permanently unmeasurable is
    visible in the log instead of silently unreapable.
    """

    path: Path
    shape: str
    reason: str
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class UnreadableRoot:
    """A scan root the walk could not open, and why.

    Reported rather than swallowed: ``--root`` at a ``chmod 000`` directory (or
    at a path that does not exist) used to print ``0 candidate(s)``, exit 0 and
    leave ``incomplete_scan=false`` — a scheduled job hitting a permission wall
    looked exactly like a machine with nothing to reap.
    """

    path: Path
    detail: str


@dataclasses.dataclass(frozen=True)
class ReapReport:
    """One tick's outcome, in the shape both ``--json`` and the log need."""

    dry_run: bool
    roots: tuple[Path, ...]
    age_hours: int
    include_extra_shapes: bool
    candidates: tuple[Candidate, ...] = ()
    removed: tuple[Removal, ...] = ()
    skipped: tuple[Skip, ...] = ()
    free_before: int = 0
    free_after: int = 0
    incomplete_scan: bool = False
    aborted: str = ""
    unreadable_roots: tuple[UnreadableRoot, ...] = ()
    #: Tick-level warnings that are not about one candidate — currently the "a
    #: real reap with no ``--root`` scans the whole home" notice. Carried in the
    #: report so the log records it too: a warning that only reached a terminal
    #: would be invisible for the scheduled run that most needs it.
    warnings: tuple[str, ...] = ()

    @property
    def all_roots_unreadable(self) -> bool:
        """Whether every root failed, i.e. the tick proved nothing at all."""
        return bool(self.roots) and len(self.unreadable_roots) == len(self.roots)


    @property
    def reclaimed_bytes(self) -> int:
        return sum(r.allocated_bytes for r in self.removed)

    @property
    def would_reclaim_bytes(self) -> int:
        return sum(c.allocated_bytes for c in self.candidates)

    def to_json_dict(self) -> dict[str, object]:
        """The ``--json`` payload. Paths as strings; bytes as integers."""

        def path_text(p: Path) -> str:
            return str(p)

        return {
            "dry_run": self.dry_run,
            "roots": [path_text(r) for r in self.roots],
            "age_hours": self.age_hours,
            "include_extra_shapes": self.include_extra_shapes,
            "candidates": [
                {
                    "path": path_text(c.path),
                    "shape": c.shape,
                    "age_hours": round(c.age_hours, 3),
                    "allocated_bytes": c.allocated_bytes,
                    "apparent_bytes": c.apparent_bytes,
                }
                for c in self.candidates
            ],
            "removed": [
                {
                    "path": path_text(r.path),
                    "shape": r.shape,
                    "age_hours": round(r.age_hours, 3),
                    "allocated_bytes": r.allocated_bytes,
                    "apparent_bytes": r.apparent_bytes,
                }
                for r in self.removed
            ],
            "skipped": [
                {
                    "path": path_text(s.path),
                    "shape": s.shape,
                    "reason": s.reason,
                    "detail": s.detail,
                }
                for s in self.skipped
            ],
            "reclaimed_bytes": self.reclaimed_bytes,
            "would_reclaim_bytes": self.would_reclaim_bytes,
            "free_before_bytes": self.free_before,
            "free_after_bytes": self.free_after,
            "incomplete_scan": self.incomplete_scan,
            "aborted": self.aborted,
            "unreadable_roots": [
                {"path": path_text(root.path), "detail": root.detail}
                for root in self.unreadable_roots
            ],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReaperSettings:
    """The four tunables, resolved once per tick.

    Defaults are the module constants above. :func:`read_settings` fills this
    from the registry; a caller that already knows the values (the CLI, a test)
    constructs one directly.
    """

    enabled: bool = DEFAULT_ENABLED
    age_hours: int = DEFAULT_AGE_HOURS
    dry_run: bool = DEFAULT_DRY_RUN
    include_extra_shapes: bool = DEFAULT_INCLUDE_EXTRA_SHAPES


class ReaperConfigError(RuntimeError):
    """A ``config.yml`` that EXISTS but could not be trusted.

    Raised rather than falling back to the module defaults, because those defaults
    are ``enabled=True, dry_run=False``: a fallback turns a corrupt file into a
    LIVE reaper. :func:`run_cli` turns exactly this into a refusal with a
    non-zero exit, and the tick deletes nothing.
    """


#: A config key that is not in the file at all — distinct from a stored ``None``.
_ABSENT = object()


def _stored_value(values: object, path: Sequence[str]) -> object:
    """Walk the stored config along ``path``; ``_ABSENT`` when it is not there.

    The RAW value, which :func:`local_operator.settings_io.read_setting`
    deliberately does not expose (it normalises a bool and substitutes the row's
    own default for anything it cannot parse). Type validation needs the raw form:
    ``enabled: "flase"`` normalises to the DEFAULT switch, i.e. silently ON, which
    is precisely the direction this module must never fail in.
    """
    node = values
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def read_settings(config_dir: Path | str | None = None) -> ReaperSettings:
    """Resolve the four keys from config; a MISSING file means the defaults.

    Read at tick time — not import time — which is why the section's ``Scope``
    in the registry is ``LIVE``: editing a key takes effect on the next reap
    with no relaunch, and the reaper has no long-lived process to reload.

    TWO FAILURE MODES, DELIBERATELY DIFFERENT. No ``config.yml`` at all is the
    module defaults: an installed reaper with no config yet has to work. A
    ``config.yml`` that exists and cannot be parsed, read, or type-checked raises
    :class:`ReaperConfigError` instead, and the caller refuses the tick. The
    swallow-and-default behaviour this replaces failed toward REAPING (see the
    module docstring): a corrupt file silently armed a reaper the user had
    switched off.

    ``settings_io.read_setting`` still supplies every VALUE rather than a
    hand-rolled YAML scan, so a key can never be read one way here and another way
    on the settings page; :func:`_stored_value` is only the type gate.
    """
    try:
        # Function-local: the harness's import graph is exactly what this module
        # must not pull in at launchd start (see the module docstring).
        import yaml

        from local_operator import settings_io
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir as resolve_config_dir
    except Exception:  # noqa: BLE001 — an unusable import graph is not a bad config
        logger.warning("build-cache: settings registry unavailable; using defaults", exc_info=True)
        return ReaperSettings()

    root = Path(config_dir) if config_dir is not None else resolve_config_dir()
    config_file = root / "config.yml"
    try:
        config_file.stat()
    except FileNotFoundError:
        return ReaperSettings()
    except OSError as exc:
        raise ReaperConfigError(f"{config_file}: {type(exc).__name__}") from exc

    # Parse it OURSELVES, before ConfigManager sees it. The manager QUARANTINES a
    # file it cannot parse (renames it to ``config.yml.bad.<stamp>``) and carries
    # on with the shipped defaults, so asking it afterwards cannot tell "the user
    # set nothing" from "the user's file was moved aside". A scheduled run that
    # cannot read its own switch must say so, not rearm itself.
    try:
        with open(config_file, "r", encoding="utf-8") as handle:
            document = yaml.load(handle, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReaperConfigError(f"{config_file}: {type(exc).__name__}: {exc}") from exc
    if document is not None and not isinstance(document, dict):
        raise ReaperConfigError(
            f"{config_file}: top level is {type(document).__name__}, expected a mapping"
        )
    document_values = (document or {}).get("values")
    if document_values is not None and not isinstance(document_values, dict):
        raise ReaperConfigError(
            f"{config_file}: values: is {type(document_values).__name__}, expected a mapping"
        )

    try:
        manager = ConfigManager(config_dir=root)
        stored = manager.get_config().values
    except Exception as exc:  # noqa: BLE001 — an unreadable config is fatal
        raise ReaperConfigError(f"{config_file}: {type(exc).__name__}: {exc}") from exc

    resolved = ReaperSettings()
    for key in (KEY_ENABLED, KEY_AGE_HOURS, KEY_DRY_RUN, KEY_INCLUDE_EXTRA_SHAPES):
        setting = settings_io.resolve_key(key)
        if setting is None:  # a renamed registry row must be loud, never a guess
            raise ReaperConfigError(f"{key}: not in the settings registry")
        raw = _stored_value(stored, setting.path)
        if raw is _ABSENT:
            # Not in the file: the row's own default, which
            # ``tests/unit/test_settings_io.py::_consumer_defaults()`` pins to the
            # module constants above.
            continue
        try:
            value = settings_io.read_setting(manager, setting)
        except Exception as exc:  # noqa: BLE001 — a value the page would reject
            raise ReaperConfigError(f"{key}: {exc}") from exc
        if key == KEY_AGE_HOURS:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ReaperConfigError(f"{key}: expected a whole number of hours, got {raw!r}")
            if not 1 <= raw <= 720:
                # The registry row's own bound (``minimum=1, maximum=720``). A
                # hand edit outside it is a config this reaper will not guess at.
                raise ReaperConfigError(f"{key}: must be between 1 and 720, got {raw!r}")
            resolved = dataclasses.replace(resolved, age_hours=int(value))
        else:
            if not isinstance(raw, bool):
                raise ReaperConfigError(f"{key}: expected true or false, got {raw!r}")
            field = {
                KEY_ENABLED: "enabled",
                KEY_DRY_RUN: "dry_run",
                KEY_INCLUDE_EXTRA_SHAPES: "include_extra_shapes",
            }[key]
            resolved = dataclasses.replace(resolved, **{field: bool(value)})
    return resolved


def enabled_shapes(include_extra: bool) -> tuple[str, ...]:
    """The candidate shape names for this tick."""
    return DEFAULT_SHAPES + EXTRA_SHAPES if include_extra else DEFAULT_SHAPES


# --------------------------------------------------------------------------
# Scan roots
# --------------------------------------------------------------------------


def resolve_roots(
    requested: Sequence[str | Path] | None = None, home: Path | None = None
) -> tuple[Path, ...]:
    """Decide which trees to scan: a pure function of (args, home).

    No ``--root`` means the user's home. Any ``--root`` REPLACES that default
    for the invocation, repeatable, de-duplicated and order-preserving so a
    manual run can be bounded to one checkout or rehearsed against a fixture
    tree without touching ``HOME`` (which this harness forbids for rigs).

    Purity is deliberate: a test drives the whole CLI without mutating the
    environment or the real home, and the resolution cannot drift between the
    dry run the operator inspects and the real run that follows.

    ``realpath`` on the roots only — a root that IS a symlink (a symlinked
    checkout) is resolved once, while every candidate found beneath it is still
    refused if it is a symlink (see :func:`iter_candidate_dirs`).
    """
    base = Path(home) if home is not None else Path.home()
    if not requested:
        return (Path(os.path.realpath(str(base))),)
    seen: set[str] = set()
    roots: list[Path] = []
    for raw in requested:
        text = os.path.realpath(str(Path(str(raw)).expanduser()))
        if text in seen:
            continue
        seen.add(text)
        roots.append(Path(text))
    return tuple(roots)


def _pruned_absolute_paths(home: Path) -> tuple[str, ...]:
    """Absolute subtrees excluded from every scan, as realpath strings.

    The config dir comes from :func:`local_operator.paths.config_dir` rather
    than a hardcoded ``~/.local-operator``: the override exists so an isolated
    run touches nothing outside one directory, and a reaper that ignored it
    would delete another process's scratchpad while proving it was isolated.
    The session scratchpad root is excluded separately because a session can
    set it anywhere.
    """
    pruned: list[str] = []
    try:
        from local_operator.paths import CONFIG_DIR_ENV, config_dir

        pruned.append(os.path.realpath(str(config_dir())))
        if not os.environ.get(CONFIG_DIR_ENV):
            # Belt and braces for the shared default: a reaper running with a
            # relocated config dir must still never prune the real one.
            pruned.append(os.path.realpath(str(home / ".local-operator")))
    except Exception:  # noqa: BLE001 — losing an exclusion must not lose the others
        logger.warning("build-cache: could not resolve the config dir", exc_info=True)
    scratch = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
    if scratch:
        pruned.append(os.path.realpath(str(Path(scratch).expanduser())))
    for name in PRUNED_SUBTREES_AT_HOME:
        pruned.append(os.path.realpath(str(home / name)))
    return tuple(dict.fromkeys(pruned))


def _is_pruned(path_text: str, pruned: tuple[str, ...]) -> bool:
    """Whether an absolute path is one of the excluded subtrees."""
    return any(path_text == p or path_text.startswith(p + os.sep) for p in pruned)


def unreadable_root(root: Path) -> UnreadableRoot | None:
    """Whether ``root`` can be opened at all, and why not.

    :func:`iter_candidate_dirs` swallows an ``OSError`` per directory on purpose
    (one permission wall deep in a tree must not abort the scan of everything
    else), but swallowing it at a ROOT made an unreadable or nonexistent
    ``--root`` indistinguishable from nothing to do: exit 0, ``0 candidate(s)``,
    ``incomplete_scan=false``. A scheduled job that hits a permission wall has to
    be able to say so, so each root is probed once here and reported by name.
    """
    try:
        with os.scandir(root) as iterator:
            next(iterator, None)
    except FileNotFoundError:
        return UnreadableRoot(root, "does not exist")
    except NotADirectoryError:
        return UnreadableRoot(root, "not a directory")
    except PermissionError:
        return UnreadableRoot(root, "permission denied")
    except OSError as exc:
        return UnreadableRoot(root, exc.__class__.__name__)
    return None


def iter_candidate_dirs(
    roots: Sequence[Path],
    *,
    shapes: Sequence[str],
    pruned: Sequence[str] = (),
    max_depth: int = MAX_DEPTH,
    budget_s: float = DEFAULT_TIME_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[tuple[Path, str, bool]]:
    """Yield ``(path, shape, budget_tripped)`` for every candidate directory.

    DEPTH-first over each root through an explicit LIFO stack (``stack.pop()``
    from the end) — not breadth-first, which an earlier revision of this
    docstring claimed. Nothing here depends on the order: the scan has a
    wall-clock budget rather than a depth or breadth priority, and the measured
    full-home walk finishes inside it (12.7 s, ``incomplete_scan=False``).

    Never follows a symlink (an
    ``os.scandir`` walk with ``follow_symlinks=False``, and a symlinked entry is
    not descended into and never yielded as a candidate): a shared
    ``node_modules``-shaped symlink target is exactly how this class of tool
    breaks a machine.

    ``PRUNED_DIRNAMES`` and the absolute ``pruned`` subtrees are cut before
    descent, so the walk costs what the interesting trees cost and no more.

    The generator yields ``budget_tripped=True`` once as its last item when the
    wall-clock ``budget_s`` is spent; the caller turns that into
    ``incomplete_scan`` and still prunes what it already proved idle.

    A directory that cannot be opened is skipped silently HERE — one permission
    wall inside a tree must not cost the whole scan — which is why a ROOT's
    readability is checked separately, by :func:`unreadable_root`, before the
    walk starts.
    """
    deadline = clock() + budget_s
    for root in roots:
        root_text = str(root)
        stack: list[tuple[str, int]] = [(root_text, 0)]
        while stack:
            if clock() > deadline:
                yield Path(root_text), "", True
                return
            current, depth = stack.pop()
            if depth >= max_depth:
                continue
            try:
                with os.scandir(current) as iterator:
                    entries = list(iterator)
            except OSError:
                continue
            for entry in entries:
                if entry.is_symlink():
                    # A symlinked shape is REPORTED and refused, not silently
                    # skipped: a shared build directory behind per-worktree
                    # symlinks is how a reaper deletes every worktree's output at
                    # once, and the operator reading the log needs to see that the
                    # space is still there and why nothing touched it.
                    if entry.name in shapes:
                        yield Path(os.path.join(current, entry.name)), entry.name, False
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                entry_path = os.path.join(current, entry.name)
                if entry.name in PRUNED_DIRNAMES or _is_pruned(entry_path, tuple(pruned)):
                    continue
                if entry.name in shapes:
                    yield Path(entry_path), entry.name, False
                    # A shape directory can nest (target/debug/target is noise),
                    # and walking inside it costs more than it can ever find.
                    continue
                stack.append((entry_path, depth + 1))


# --------------------------------------------------------------------------
# Bounded tree walk: age and bytes in one pass
# --------------------------------------------------------------------------


def scan_tree(
    path: Path,
    *,
    cutoff: float | None = None,
    max_entries: int = WALK_MAX_ENTRIES,
    budget_s: float = WALK_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
    stop_when_fresh: bool = True,
) -> TreeScan:
    """Walk ``path`` once for its newest mtime and its allocated bytes.

    Iterative (no recursion limit on a deep vendored tree), symlink-safe
    (``follow_symlinks=False`` everywhere; a symlink contributes its own lstat
    and is never descended into), and bounded in two dimensions.

    Bytes are ``st_blocks * 512`` over REGULAR FILES only: allocated blocks are
    what the filesystem actually gave back (see the module docstring; apparent
    size overstates clone/hardlink trees, which is what this fleet's worktrees
    are). Directory entries contribute their mtime but not their size — a
    directory's "size" is metadata, and counting it would put a number in the
    reclaimed figure that no deletion can return.

    ``cutoff`` is the age boundary as an epoch second. When an entry newer than
    it is seen, ``saw_fresh`` is set and — with ``stop_when_fresh`` — the walk
    returns at once: the verdict is already "too young", and re-walking a live
    multi-gigabyte build tree to produce totals nobody will read is the one
    avoidable cost in the tick. The partial totals such a walk returns are
    never used; the caller treats ``saw_fresh`` as the whole answer.
    """
    started = clock()
    newest: float | None = None
    allocated = 0
    apparent = 0
    entries_seen = 0
    stopped: str | None = None
    saw_fresh = False

    try:
        root_stat = os.stat(path, follow_symlinks=False)
        newest = root_stat.st_mtime
        if cutoff is not None and root_stat.st_mtime > cutoff:
            saw_fresh = True
    except OSError:
        pass

    stack: list[str] = [str(path)]
    while stack and not (saw_fresh and stop_when_fresh):
        if clock() - started > budget_s:
            stopped = "time"
            break
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError:
            continue
        for entry in entries:
            entries_seen += 1
            if entries_seen > max_entries:
                stopped = "entries"
                break
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            mtime = entry_stat.st_mtime
            if newest is None or mtime > newest:
                newest = mtime
            if cutoff is not None and mtime > cutoff:
                saw_fresh = True
            mode = entry_stat.st_mode
            if stat.S_ISREG(mode):
                apparent += entry_stat.st_size
                allocated += getattr(entry_stat, "st_blocks", 0) * 512
            elif stat.S_ISDIR(mode):
                stack.append(entry.path)
            if saw_fresh and stop_when_fresh:
                break
        if stopped is not None:
            break

    return TreeScan(
        newest_mtime=newest,
        allocated_bytes=allocated,
        apparent_bytes=apparent,
        entries=entries_seen,
        stopped=stopped,
        saw_fresh=saw_fresh,
    )


# --------------------------------------------------------------------------
# Probes. Every one is tri-state, and None means SKIP.
# --------------------------------------------------------------------------

#: ``True``/``False``/``None`` (cannot tell). The alias documents the contract
#: at every use site; the probes are injectable so tests drive all three
#: answers without spawning a process or opening a file.
TriState = Callable[[Path], "bool | None"]


@dataclasses.dataclass(frozen=True)
class ProcessInfo:
    """One row of the process table, as much as attribution needs."""

    pid: int
    argv: tuple[str, ...]
    cwd: str | None


@dataclasses.dataclass(frozen=True)
class _Snapshot:
    """The two tables a tick reads, cached and refreshable.

    ``ps`` and ``lsof`` are each ONE sweep per tick rather than one per
    candidate (``pytest-tmp-reaper`` proves the shape: a per-directory sweep is
    both slower and racier). ``refresh()`` drops the cache, and the tick calls
    it immediately before each removal: the snapshot narrows the candidate list,
    while the reap-time re-read is what actually decides — a build can start
    during the scan, which is the race the proven script exists to close.
    """

    processes: "tuple[ProcessInfo, ...] | None" = dataclasses.field(default=None)
    handles: "tuple[str, ...] | None" = dataclasses.field(default=None)
    _loaded: bool = dataclasses.field(default=False, compare=False)

    def refresh(self) -> None:
        object.__setattr__(self, "processes", None)
        object.__setattr__(self, "handles", None)
        object.__setattr__(self, "_loaded", False)

    def load(self) -> None:
        if self._loaded:
            return
        object.__setattr__(self, "processes", _read_process_table())
        object.__setattr__(self, "handles", _read_open_handles())
        object.__setattr__(self, "_loaded", True)


def _is_mount_point(path: Path) -> bool:
    """Whether ``path`` is the root of its own filesystem.

    ``os.path.ismount`` rather than a hand-rolled ``st_dev`` comparison: it is
    the same check, and it answers ``False`` rather than raising for a path it
    cannot stat (a candidate that cannot be stat-ed is already ``error``).
    ``rmtree`` across a mount walks onto another volume and deletes whatever it
    finds there, so a candidate that IS one is skipped by name.
    """
    return os.path.ismount(path)


def _has_git_entry(candidate: Path) -> bool | None:
    """Whether ``candidate`` carries a ``.git`` entry (directory OR file).

    ``True`` means the candidate is a repository or a linked-worktree root, not
    build output: a worktree's ``.git`` is a FILE pointing at the parent repo, so
    a directory-only check misses exactly the case that took a worktree's branch
    and uncommitted work with it. ``None`` (the entry could not be stat-ed) skips
    the candidate; only ``FileNotFoundError`` answers ``False``.
    """
    try:
        os.lstat(candidate / ".git")
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return True


@dataclasses.dataclass(frozen=True)
class Probes:
    """The four injectable safety probes plus a cache refresh hook.

    Each probe answers ``True`` / ``False`` / ``None`` for ONE candidate, and
    ``None`` ("the instrument could not be trusted") blocks — see the module
    docstring. Tests construct this with lambdas; production uses
    :func:`default_probes`.
    """

    toolchain_running: TriState
    handles_open: TriState
    is_git_ignored: TriState
    tracked_files: TriState
    #: Whether the candidate is itself a filesystem root. Injectable like the
    #: rest: a test cannot mount a volume, and the guard still has to be provable.
    is_mount_point: Callable[[Path], bool] = _is_mount_point
    refresh: Callable[[], None] = lambda: None
    process_table_readable: Callable[[], "bool | None"] = lambda: True


def _read_process_table() -> tuple[ProcessInfo, ...] | None:
    """Read the process table once. ``None`` means "could not tell".

    ``ps -axo pid=,command=`` because it is present on every supported platform
    and cheap. ``ps`` does not report a cwd and does not delimit argv, so argv is
    approximated by whitespace splitting and the cwd is filled in from
    :func:`_read_open_handles` (one ``lsof`` sweep already reports it). Both
    approximations fail in the SAFE direction: a path containing a space can
    make a process unattributable by argv, and the cwd rule then covers it.

    A zero-length table is a failure, not an empty machine: every supported
    platform has at least one process, so an empty parse means the output was
    not what we expect.
    """
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=PROCESS_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    rows: list[ProcessInfo] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, rest = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            return None
        argv = tuple(rest.split())
        rows.append(ProcessInfo(pid=pid, argv=argv, cwd=None))
    if not rows:
        return None
    return tuple(rows)


def _read_open_handles() -> tuple[str, ...] | None:
    """Read every open path and every process cwd in ONE ``lsof`` sweep.

    Returns the absolute paths, or ``None`` when the reading cannot be trusted —
    and a failed ``lsof`` means "everything may be in use", never "nothing is":
    a stale disk is recoverable, a build directory removed under another
    session's compile is not.

    The exit-status rule is the proven one from ``pytest-tmp-reaper``: ``lsof``
    exits non-zero when it could not stat some files (permission denied is
    normal on a shared machine) while still listing thousands of handles, so a
    non-zero status with a usable parse is trusted, and a non-zero status with
    nothing parsed is ``None``. Treating every non-zero status as failure would
    make this a permanent no-op on exactly the machine it was built for.

    cwd rows are returned too, as ``pid:<n>:cwd=<path>`` sentinels: attribution
    needs the cwd, ``ps`` does not carry it, and a second sweep would double the
    most expensive call in the tick. Sentinels rather than a second structure
    because this is the type the caller already threads around.

    A HEADER-ONLY or unparseable table is ``None``, never a short list. The
    header row (``COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME``) splits
    into the same nine columns as a data row, so counting rows without rejecting
    it made ``('NAME',)`` — a confident "nothing is open" — out of output that
    carried no handle at all. A row only counts when its PID column is an integer.
    """
    try:
        completed = subprocess.run(
            ["lsof", "-n"], capture_output=True, text=True, timeout=PROCESS_PROBE_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        # NAME may contain spaces, so split the fixed leading columns and keep
        # the remainder whole: COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME
        fields = line.split(None, 8)
        if len(fields) != 9:
            continue
        try:
            int(fields[1])
        except ValueError:
            # The header, or a status/count line some builds append. Not a
            # handle: counting it would make an empty table look usable.
            continue
        fd = fields[3]
        name = fields[8].strip()
        if fd == "cwd":
            paths.append(f"pid:{fields[1]}:cwd={name}")
        else:
            paths.append(name)
    if not paths:
        # Nothing parsed at all. An empty table is unreadable, never "nothing is
        # open": every supported platform has processes with open files, so a
        # zero-length parse means the output was not what we expect.
        return None
    return tuple(paths)


def _cwd_by_pid(handles: Sequence[str]) -> dict[int, str]:
    """The cwd sentinels from :func:`_read_open_handles`, keyed by pid."""
    out: dict[int, str] = {}
    for entry in handles:
        if not entry.startswith("pid:"):
            continue
        rest = entry[len("pid:") :]
        pid_text, _, tail = rest.partition(":cwd=")
        try:
            out[int(pid_text)] = tail
        except ValueError:
            continue
    return out


def _is_toolchain_process(argv: Sequence[str]) -> bool:
    """Whether any argv element names a build toolchain executable.

    Every element is checked, not just ``argv[0]``: the ``.next`` producer runs
    as ``node .../node_modules/.bin/next`` and a wrapper invocation is common.
    Checking more elements can only produce more blocks, which is the safe
    direction.
    """
    for arg in argv:
        base = os.path.basename(arg)
        if base in TOOLCHAIN_EXECUTABLES:
            return True
    return False


def _process_touches(touching: ProcessInfo, candidate: Path, *, parent_term: bool = True) -> bool:
    """Whether ONE process is attributable to ``candidate``.

    Three ways, matching the brief's contract: an argv element equal to or under
    the candidate, an argv element equal to or under the candidate's parent (the
    worktree — ``cargo`` is invoked with ``--manifest-path <worktree>/Cargo.toml``
    and rustc with ``--out-dir <worktree>/target/debug/deps``, both of which
    mention the worktree rather than the candidate verbatim), or a cwd at/under
    the worktree.

    ``parent_term=False`` drops the parent (and its cwd rule), leaving only what
    names the candidate itself. That is required when the parent is a SCAN ROOT
    or the home directory: ``worktree = candidate.parent`` would otherwise
    attribute every toolchain process anywhere beneath home, and the observed
    result was ``skipped /Users/damian/.next: toolchain-running`` with no build
    anywhere near it — twice, on an idle machine. A real project root keeps the
    term (it is the worktree the compiler was pointed at); a directory that
    merely CONTAINS many projects is not evidence about any one of them.
    """
    worktree = str(candidate.parent)
    targets = (str(candidate), worktree) if parent_term else (str(candidate),)
    for arg in touching.argv:
        for target in targets:
            if arg == target or arg.startswith(target + os.sep):
                return True
        if target_in_arg(arg, targets):
            return True
    cwd = touching.cwd
    if cwd:
        if cwd == str(candidate):
            return True
        if parent_term and (cwd == worktree or cwd.startswith(worktree + os.sep)):
            return True
    return False


def _parent_is_attribution_scope(candidate: Path, roots: Sequence[Path], home: Path) -> bool:
    """Whether ``candidate``'s parent may be used for process attribution.

    False when the parent is a scan root or the home directory — see
    :func:`_process_touches` for the measured false positives that made this
    necessary.
    """
    parent = candidate.parent
    if parent == home:
        return False
    return not any(parent == root for root in roots)


def target_in_arg(arg: str, targets: Sequence[str]) -> bool:
    """Whether ``arg`` CONTAINS one of ``targets`` as a whole path.

    Needed because argv elements carry values glued to a flag or a ``=``
    (``--out-dir=/w/target/debug``), where an exact/prefix test on the element
    misses. The match is boundary-aware — the target must start the element or
    follow a separator, and end the element or be followed by a separator — so a
    sibling checkout whose name merely EXTENDS this one (``repo-old`` against
    ``repo``) is not attributed to it. That precision matters on a fleet where
    ~25 concurrent sessions build different worktrees of the same repository: an
    over-match here would block a real reap every time a neighbour compiled,
    which is how a reaper becomes a no-op nobody trusts.

    Deliberately attribute-only: it decides whether a build MAY be running, so a
    false positive costs one skipped candidate while a false negative corrupts
    someone's compile.
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


def default_probes(roots: Sequence[Path] = (), home: Path | None = None) -> Probes:
    """The production probes, bound to one refreshable snapshot.

    ``roots``/``home`` are what tell process attribution whether a candidate's
    PARENT is a usable scope: for a candidate sitting directly in a scan root or
    in the home directory the parent term is dropped (see
    :func:`_process_touches`), because "somewhere under home" is not evidence
    that a build is using THIS directory. A caller that omits them (a test
    exercising one directory) gets the full attribution rule.
    """
    snapshot = _Snapshot()
    scope_home = Path(home) if home is not None else Path.home()

    def toolchain_running(candidate: Path) -> bool | None:
        snapshot.load()
        table = snapshot.processes
        if table is None:
            return None
        cdws = _cwd_by_pid(snapshot.handles or ())
        parent_term = _parent_is_attribution_scope(candidate, roots, scope_home)
        for process in table:
            if not _is_toolchain_process(process.argv):
                continue
            enriched = ProcessInfo(
                pid=process.pid, argv=process.argv, cwd=cdws.get(process.pid)
            )
            if _process_touches(enriched, candidate, parent_term=parent_term):
                return True
        return False

    def handles_open(candidate: Path) -> bool | None:
        snapshot.load()
        handles = snapshot.handles
        if handles is None:
            return None
        prefix = str(candidate) + os.sep
        exact = str(candidate)
        for entry in handles:
            # A ``pid:<n>:cwd=<path>`` sentinel still names a path in use: a
            # process sitting IN the directory holds it just as much as one with
            # a file open beneath it.
            path_text = entry.split(":cwd=", 1)[1] if entry.startswith("pid:") else entry
            if path_text == exact or path_text.startswith(prefix):
                return True
        return False

    def process_table_readable() -> bool | None:
        snapshot.load()
        return snapshot.processes is not None

    return Probes(
        toolchain_running=toolchain_running,
        handles_open=handles_open,
        is_git_ignored=git_ignored_probe,
        tracked_files=git_tracked_files_probe,
        refresh=snapshot.refresh,
        process_table_readable=process_table_readable,
    )


def _run_git(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run one git query, or ``None`` when git itself could not be trusted.

    ``LC_ALL=C`` so a diagnostic that this module READS (see
    :func:`_not_a_repository`) arrives in the one spelling it matches; an
    undecodable stream is ``None`` ("cannot tell") rather than an exception out
    of the reaper's only subprocess helper.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=GIT_PROBE_TIMEOUT_S,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None


#: git's own words for "there is no repository here", matched below. The check is
#: the only locale-stable thing about a 128: every other fatal error (a corrupt
#: repository, an unsupported ``core.repositoryformatversion``, a permissions
#: wall, a bad ``GIT_DIR``) exits 128 too, and treating all of them as "no
#: repository" is how a reaper answers "nothing is tracked" about a directory it
#: could not read at all.
_NOT_A_REPOSITORY = "not a git repository"


def _inside_any_repository(cwd: Path) -> bool | None:
    """Whether git considers ``cwd`` part of a repository: True/False/None.

    The discriminator for 128. ``git -C <candidate.parent> ls-files`` exits 128
    both when the parent genuinely is outside every repository AND when the
    directory is a repository of its own that git refuses to describe from
    outside — the case that deleted a linked worktree whole. So a 128 is never
    evidence by itself: this asks the follow-up question ("is there a repository
    at/above here?") and only a recognised "no" becomes ``False``.

    ``rev-parse --show-toplevel`` is the query because it is: (a) read-only, (b)
    cheap on git's common path — the fast path below is ONE extra call on the
    128 branch only, so an in-repository tree costs exactly what it did before —
    and (c) answered for a bare repository or a ``GIT_DIR`` override as well as
    a worktree.
    """
    completed = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if completed is None:
        return None
    if completed.returncode == 0:
        return True
    if completed.returncode == 128 and _NOT_A_REPOSITORY in (completed.stderr or ""):
        return False
    return None


def git_tracked_files_probe(candidate: Path) -> bool | None:
    """Whether any git-TRACKED file lives beneath ``candidate``.

    ``True`` is the dangerous answer: a tracked file there means part of what we
    would delete is committed content, so nothing is removed regardless of the
    shape's name. ``git ls-files`` with the bare directory name as pathspec (run
    from the parent, so no absolute-pathspec edge cases) lists tracked files
    recursively, which is the "must be empty" test the safety rule states.

    Outside any repository the answer is ``False`` — nothing there can be
    tracked — which is why a ``target/`` in a plain directory (the common case
    for a scratch build) stays reapable. That ``False`` needs PROOF, though: a
    128 is only "no repository" when :func:`_inside_any_repository` says so, and
    every other failure (git missing, a timeout, an undecodable stream, a corrupt
    repo, a permissions wall) is ``None`` — cannot tell, so the candidate is
    skipped. Reading ``128`` as ``False`` is what let a git worktree whose own
    ``.git`` file sat INSIDE the candidate be reported as having nothing tracked.
    """
    completed = _run_git(["ls-files", "--", candidate.name], candidate.parent)
    if completed is None:
        return None
    if completed.returncode == 128:
        return False if _inside_any_repository(candidate.parent) is False else None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def git_ignored_probe(candidate: Path) -> bool | None:
    """Whether ``candidate`` is git-IGNORED, which is the shape's real claim.

    The guard this buys is narrow but real: an untracked AND not-ignored
    ``dist/`` is somebody's directory that ``git status`` shows by hand — build
    output in a repository that chose not to ignore it is an owner decision, so
    it is skipped rather than assumed. Tracked content is already caught by
    :func:`git_tracked_files_probe`; this is the neighbouring case.

    Outside any repository the answer is ``True``: no ignore rule exists to
    violate and nothing can be tracked, so the shape name is the whole safety
    case (and the tracked probe independently confirms the second half). Same
    proof requirement as the tracked probe: a 128 becomes ``True`` only when
    :func:`_inside_any_repository` confirms there is genuinely no repository, and
    is otherwise ``None``.
    """
    completed = _run_git(["check-ignore", "-q", "--", candidate.name], candidate.parent)
    if completed is None:
        return None
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    if completed.returncode == 128:
        return True if _inside_any_repository(candidate.parent) is False else None
    return None


# --------------------------------------------------------------------------
# Free space
# --------------------------------------------------------------------------


def free_bytes(paths: Sequence[Path]) -> int:
    """Free bytes across the DISTINCT filesystems holding ``paths``; 0 if none.

    ``shutil.disk_usage`` rather than parsing ``df``: the same number without a
    platform-specific parser. The figure used to be measured on
    ``scan_roots[0]`` alone, which is the wrong disk whenever the run covers two
    volumes — and the ``df`` cross-check in the module docstring then compares
    against a number from another filesystem. Every root counts here,
    DE-DUPLICATED by ``st_dev`` so the common one-volume run (several ``--root``s
    on the same disk, or the home default) reports that disk once rather than a
    multiple of it, and a two-volume run reports what both have free.

    An unreadable path contributes 0: this is a report figure, never a guard, so
    a measuring failure must not be able to stop a reap.
    """
    seen: set[int] = set()
    total = 0
    for path in paths:
        try:
            device = os.stat(path).st_dev
        except OSError:
            continue
        if device in seen:
            continue
        seen.add(device)
        try:
            total += shutil.disk_usage(str(path)).free
        except OSError:
            continue
    return total


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Verdict:
    """One candidate's fate, before anything is touched."""

    removable: bool
    reason: str = ""
    detail: str = ""
    age_hours: float = 0.0
    allocated_bytes: int = 0
    apparent_bytes: int = 0


def _evaluate(
    candidate: Path,
    *,
    shape: str,
    cutoff: float,
    now: float,
    probes: Probes,
    clock: Callable[[], float] = time.monotonic,
) -> _Verdict:
    """Decide one candidate. Every uncertainty resolves to "skip".

    Order is cheap-first and safety-ordered: a symlink is never considered at
    all; the two PROPERTY guards (mount point, ``.git`` entry) are two ``stat``
    calls and answer a question no shape name can — whether the directory is
    build output in the first place; age is next (the bounded walk can also stop
    early the moment a tree is provably too young); git next (two ``subprocess``
    calls, no tree walk); the live-build probes last because they are the ones
    worth re-reading at reap time anyway.
    """
    # 1. Never walk through or delete a symlinked candidate. `target` or `.next`
    # as a symlink is a shared target across worktrees — deleting "it" would
    # delete every worktree's build at once, and deleting the link would leave
    # the space in place. Both are wrong; skip the shape entirely.
    try:
        if candidate.is_symlink():
            return _Verdict(False, SKIP_SYMLINK)
        if not candidate.is_dir():
            return _Verdict(False, SKIP_ERROR, "not a directory")
    except OSError as exc:
        return _Verdict(False, SKIP_ERROR, exc.__class__.__name__)

    # 1b. A mount point is another volume's contents, not ours to empty.
    try:
        mounted = probes.is_mount_point(candidate)
    except OSError as exc:  # a probe that cannot answer is not a go-ahead
        return _Verdict(False, SKIP_CANNOT_TELL, f"mount-probe: {exc.__class__.__name__}")
    if mounted:
        return _Verdict(False, SKIP_MOUNT_POINT)

    # 1c. A directory carrying a `.git` entry is a repository or a linked-worktree
    # root. It is not build output whatever it is called, and this is checked
    # BEFORE the git probes because those run from the PARENT: for a worktree
    # whose own name is a shape (`.worktrees/target`), the parent answer
    # describes a different repository entirely.
    repo_entry = _has_git_entry(candidate)
    if repo_entry is None:
        return _Verdict(False, SKIP_CANNOT_TELL, "dot-git-probe")
    if repo_entry:
        return _Verdict(False, SKIP_GIT_REPOSITORY, "contains .git")

    # 2. Age, from the newest mtime in the tree (see the module docstring).
    tree = scan_tree(candidate, cutoff=cutoff, clock=clock)
    if tree.saw_fresh:
        return _Verdict(False, SKIP_FRESH, f"newest entry within {_human_hours(cutoff, now)}")
    if tree.stopped is not None:
        # A partial maximum understates recency; there is no fallback.
        return _Verdict(False, SKIP_CANNOT_TELL, f"age-walk-{tree.stopped}")
    if tree.newest_mtime is None:
        # The tree could not be stat-ed at all (a race with the owner, or a
        # permission wall). No mtime means the age floor was never applied, so
        # there is nothing to compare against the cutoff — skip rather than
        # report a zero age that reads as "brand new" while removing it.
        return _Verdict(False, SKIP_CANNOT_TELL, "age-unknown")
    age_hours = max(0.0, (now - tree.newest_mtime) / 3600.0)

    # 3. Git: nothing tracked may be deleted, and an unignored shape is the
    # owner's call rather than ours.
    tracked = probes.tracked_files(candidate)
    if tracked is None:
        return _Verdict(False, SKIP_CANNOT_TELL, "git-tracked-probe")
    if tracked:
        return _Verdict(False, SKIP_GIT_TRACKED)
    ignored = probes.is_git_ignored(candidate)
    if ignored is None:
        return _Verdict(False, SKIP_CANNOT_TELL, "git-ignore-probe")
    if not ignored:
        return _Verdict(False, SKIP_GIT_NOT_IGNORED)

    # 4. The load-bearing guard: nothing may be using it right now, and "cannot
    # tell" is not "nothing is using it".
    handles = probes.handles_open(candidate)
    if handles is None:
        return _Verdict(False, SKIP_CANNOT_TELL, "handle-probe")
    if handles:
        return _Verdict(False, SKIP_HANDLES)
    running = probes.toolchain_running(candidate)
    if running is None:
        return _Verdict(False, SKIP_CANNOT_TELL, "toolchain-probe")
    if running:
        return _Verdict(False, SKIP_TOOLCHAIN)

    return _Verdict(
        True,
        age_hours=age_hours,
        allocated_bytes=tree.allocated_bytes,
        apparent_bytes=tree.apparent_bytes,
    )


def _human_hours(cutoff: float, now: float) -> str:
    return f"{max(0.0, (now - cutoff) / 3600.0):.1f}h"


# --------------------------------------------------------------------------
# The tick
# --------------------------------------------------------------------------


def reap(
    *,
    roots: Sequence[Path] | None = None,
    settings: ReaperSettings | None = None,
    probes: Probes | None = None,
    now: float | None = None,
    home: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    budget_s: float = DEFAULT_TIME_BUDGET_S,
    default_roots: bool = False,
) -> ReapReport:
    """One reap tick: scan, decide, delete what is proved idle, report.

    Never touches a directory the scan did not fully prove idle: an aborted tick
    (the process table unreadable) removes nothing at all, and a candidate whose
    probes answer ``None`` is skipped with ``cannot-tell``.

    ``refresh`` is called immediately before each deletion, so the live-build
    answer that decides is read at reap time rather than carried over from the
    scan — the race ``pytest-tmp-reaper`` closes and the reason its comment
    says the lsof check "is the real control".

    A removal that raises :class:`OSError` (permissions, a file re-created
    underneath) is recorded as a skip, never as reclaimed space: the report's
    bytes must be space the filesystem actually gave back, and the ``df``
    cross-check in the caller is the second opinion.

    Each root is probed once up front (:func:`unreadable_root`) and reported by
    name: an unreadable or nonexistent root produces no candidates and no skips,
    so without this it is indistinguishable from a machine with nothing to reap.

    ``default_roots`` says the caller's ``roots`` were resolved from the HOME
    default rather than from ``--root`` — the caller is the one that knows, since
    a CLI resolves ``~`` before calling here. It only drives the live-fire
    warning (see below); ``roots=None`` implies it.
    """
    scan_roots = tuple(roots) if roots else resolve_roots(None, home)
    resolved_settings = settings if settings is not None else read_settings()
    scope_home = Path(home) if home is not None else Path.home()
    resolved_probes = (
        probes if probes is not None else default_probes(scan_roots, home=scope_home)
    )
    moment = time.time() if now is None else now
    cutoff = moment - resolved_settings.age_hours * 3600.0
    shapes = enabled_shapes(resolved_settings.include_extra_shapes)
    pruned = _pruned_absolute_paths(scope_home)

    unreadable: list[UnreadableRoot] = []
    for root in scan_roots:
        problem = unreadable_root(root)
        if problem is not None:
            unreadable.append(problem)

    # A real (non-dry) tick over the HOME default reaps everything. That is the
    # intended scheduled behaviour, and it is also how a hand-run "usage check"
    # removed 12 directories: the warning is for the human who typed the command.
    warnings: list[str] = []
    unbounded = default_roots or not roots
    if unbounded and not resolved_settings.dry_run:
        warnings.append(
            "real reap over the default roots ("
            + ", ".join(str(root) for root in scan_roots)
            + ") — pass --root to bound it, or --dry-run to preview"
        )

    free_before = free_bytes(scan_roots)

    aborted = ""
    readable = resolved_probes.process_table_readable()
    if readable is None:
        # Without a process table nothing can be attributed, so a tick in which
        # every candidate would answer "cannot tell" is better reported as one
        # abort than as N identical skips.
        aborted = "aborted: cannot read process table"

    candidates: list[Candidate] = []
    removed: list[Removal] = []
    skipped: list[Skip] = []
    incomplete = False

    if not aborted:
        for path, shape, budget_tripped in iter_candidate_dirs(
            scan_roots,
            shapes=shapes,
            pruned=pruned,
            budget_s=budget_s,
            clock=clock,
        ):
            if budget_tripped:
                incomplete = True
                break
            verdict = _evaluate(
                path,
                shape=shape,
                cutoff=cutoff,
                now=moment,
                probes=resolved_probes,
                clock=clock,
            )
            if not verdict.removable:
                skipped.append(
                    Skip(path=path, shape=shape, reason=verdict.reason, detail=verdict.detail)
                )
                continue
            candidate = Candidate(
                path=path,
                shape=shape,
                age_hours=verdict.age_hours,
                allocated_bytes=verdict.allocated_bytes,
                apparent_bytes=verdict.apparent_bytes,
            )
            candidates.append(candidate)
            if resolved_settings.dry_run:
                continue
            # Re-read the live-build answer IMMEDIATELY before deleting: the
            # snapshot narrowed the list, this decides.
            resolved_probes.refresh()
            recheck = _reap_recheck(path, shape, probes=resolved_probes)
            if recheck is not None:
                skipped.append(recheck)
                continue
            try:
                shutil.rmtree(str(path))
            except OSError as exc:
                skipped.append(
                    Skip(path=path, shape=shape, reason=SKIP_ERROR, detail=exc.__class__.__name__)
                )
                continue
            removed.append(
                Removal(
                    path=path,
                    shape=shape,
                    age_hours=candidate.age_hours,
                    allocated_bytes=candidate.allocated_bytes,
                    apparent_bytes=candidate.apparent_bytes,
                )
            )

    free_after = free_bytes(scan_roots)

    return ReapReport(
        dry_run=resolved_settings.dry_run,
        roots=scan_roots,
        age_hours=resolved_settings.age_hours,
        include_extra_shapes=resolved_settings.include_extra_shapes,
        candidates=tuple(candidates),
        removed=tuple(removed),
        skipped=tuple(skipped),
        free_before=free_before,
        free_after=free_after,
        incomplete_scan=incomplete,
        aborted=aborted,
        unreadable_roots=tuple(unreadable),
        warnings=tuple(warnings),
    )


def _reap_recheck(path: Path, shape: str, *, probes: Probes) -> "Skip | None":
    """The just-before-deletion live check. ``None`` means "go ahead".

    Deliberately identical to the scan-time rule and deliberately re-read: the
    cost is one ``lsof``/``ps`` per deletion (rare) against the cost of deleting
    a directory a build claimed during the scan (unrecoverable). Any
    non-``False`` answer — including ``None`` — returns a skip.
    """
    handles = probes.handles_open(path)
    if handles is None:
        return Skip(
            path=path,
            shape=shape,
            reason=SKIP_CANNOT_TELL,
            detail="handle-probe-recheck",
        )
    if handles:
        return Skip(path=path, shape=shape, reason=SKIP_HANDLES, detail="recheck")
    running = probes.toolchain_running(path)
    if running is None:
        return Skip(
            path=path,
            shape=shape,
            reason=SKIP_CANNOT_TELL,
            detail="toolchain-probe-recheck",
        )
    if running:
        return Skip(path=path, shape=shape, reason=SKIP_TOOLCHAIN, detail="recheck")
    return None


# --------------------------------------------------------------------------
# Reporting: the human-readable log and the CLI surfaces
# --------------------------------------------------------------------------


def human_bytes(count: int) -> str:
    """Render a byte count for a log line (base 1024, one decimal)."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} B"  # pragma: no cover - unreachable


def _stamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))


def log_lines(report: ReapReport, *, now: float | None = None) -> list[str]:
    """The log records for one tick: one line per removal, then one summary.

    ``log_lines`` is separate from :func:`write_log` so the record a test
    asserts on is the record the reaper writes — a formatter tested through its
    own copy guards nothing.
    """
    moment = time.time() if now is None else now
    stamp = _stamp(moment)
    lines: list[str] = []
    if report.aborted:
        lines.append(f"{stamp} {report.aborted}")
    for warning in report.warnings:
        lines.append(f"{stamp} warning {warning}")
    for root in report.unreadable_roots:
        lines.append(f"{stamp} warning root-unreadable path={root.path} detail={root.detail}")
    for removal in report.removed:
        lines.append(
            f"{stamp} removed shape={removal.shape} path={removal.path} "
            f"allocated={human_bytes(removal.allocated_bytes)} "
            f"({removal.allocated_bytes} B) "
            f"apparent={human_bytes(removal.apparent_bytes)} "
            f"age={removal.age_hours:.1f}h"
        )
    lines.append(
        f"{stamp} summary dry_run={'true' if report.dry_run else 'false'} "
        f"candidates={len(report.candidates)} removed={len(report.removed)} "
        f"skipped={len(report.skipped)} "
        f"reclaimed={human_bytes(report.reclaimed_bytes)} ({report.reclaimed_bytes} B) "
        f"would_reclaim={human_bytes(report.would_reclaim_bytes)} "
        f"free_before={human_bytes(report.free_before)} "
        f"free_after={human_bytes(report.free_after)} "
        f"unreadable_roots={len(report.unreadable_roots)} "
        f"incomplete_scan={'true' if report.incomplete_scan else 'false'}"
    )
    return lines


def log_path() -> Path:
    """Where the reaper appends its records.

    ``paths.log_dir()`` rather than a hardcoded ``~/Library/Logs/...``: the
    override is what lets an isolated run keep its log inside one directory, and
    the platform table behind it already puts macOS logs under
    ``~/Library/Logs/local-operator`` (the path the operator reads after the
    fact) and everything else under the config dir's / XDG state log location.
    """
    from local_operator.paths import log_dir

    return log_dir() / LOG_FILENAME


def write_log(
    report: ReapReport, *, now: float | None = None, path: Path | None = None
) -> Path | None:
    """Append this tick's records to the log and mirror them to ``logging``.

    Returns the path written, or ``None`` when the log could not be created —
    logging is a diagnostic and must never make a successful reap look failed
    (same rule as :func:`local_operator.paths.ensure_log_dir`). The module
    logger gets every line unconditionally, so a launchd job's stdout carries
    the same records when the file cannot be written.

    Not rotated: records are one line per removal plus one summary per tick, so
    a hourly job writes well under a megabyte a year, and a rotation that renamed
    the file would move a human's earlier evidence out of the path they know.
    """
    from local_operator.paths import ensure_log_dir

    lines = log_lines(report, now=now)
    for line in lines:
        logger.info("%s", line)
    target = path if path is not None else log_path()
    if ensure_log_dir() is None:
        logger.warning("build-cache: log directory unavailable; records went to the logger only")
        return None
    try:
        with open(target, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    except OSError:
        logger.warning("build-cache: could not append to %s", target, exc_info=True)
        return None
    return target


def format_report(report: ReapReport) -> str:
    """The human-readable terminal summary. English: developer-facing output."""
    out: list[str] = []
    if report.aborted:
        out.append(report.aborted)
    for warning in report.warnings:
        out.append(f"warning: {warning}")
    for root in report.unreadable_roots:
        # Named per root, and counted in the summary line below: a root that
        # cannot be opened is the ONE failure that otherwise looks like success.
        out.append(f"warning: root {root.path} is unreadable ({root.detail})")
    mode = "dry run" if report.dry_run else "reaped"
    if report.removed:
        for removal in report.removed:
            out.append(
                f"removed {removal.path} ({removal.shape}, "
                f"{human_bytes(removal.allocated_bytes)}, age {removal.age_hours:.1f}h)"
            )
    elif report.candidates:
        for candidate in report.candidates:
            out.append(
                f"would remove {candidate.path} ({candidate.shape}, "
                f"{human_bytes(candidate.allocated_bytes)}, age {candidate.age_hours:.1f}h)"
            )
    if report.skipped:
        for skip in report.skipped:
            suffix = f" [{skip.detail}]" if skip.detail else ""
            out.append(f"skipped {skip.path}: {skip.reason}{suffix}")
    out.append(
        f"{mode}: {len(report.candidates)} candidate(s), {len(report.removed)} removed, "
        f"{human_bytes(report.reclaimed_bytes)} reclaimed, "
        f"free {human_bytes(report.free_before)} -> {human_bytes(report.free_after)}"
        + (" (scan incomplete)" if report.incomplete_scan else "")
        + (
            f" ({len(report.unreadable_roots)} unreadable root(s))"
            if report.unreadable_roots
            else ""
        )
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def run_cli(
    *,
    dry_run: bool = False,
    json_output: bool = False,
    roots: Sequence[str | Path] | None = None,
    home: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run one reap from parsed CLI options and return the process exit code.

    ``--dry-run`` FORCES a dry run regardless of config: the flag exists so an
    operator can inspect what a scheduled run would do without editing the
    setting, and a config value that could silently turn a requested rehearsal
    into a real deletion would be the wrong precedence.

    ``0`` on a completed tick, including "nothing to do" and including an aborted
    tick — the abort is reported in the output and the log, and the invocation
    itself succeeded. The two non-zero codes are the failures a SCHEDULED job has
    to be able to distinguish from success, both of which used to exit 0:

    * ``EXIT_CONFIG_ERROR`` — ``config.yml`` exists but could not be read or
      validated. Nothing was scanned and nothing was deleted.
    * ``EXIT_NO_READABLE_ROOT`` — every scan root was missing or unreadable, so
      the tick proved nothing. Reported per root in the output and the log.

    Argparse's own ``2`` for a malformed invocation is the one remaining code;
    ``EXIT_CONFIG_ERROR`` shares it deliberately (the same "the run never
    started" class) and the message on stderr says which.
    """
    from dataclasses import replace

    try:
        settings = read_settings()
    except ReaperConfigError as exc:
        # Refuse rather than reap on guessed settings. "Fall back to the defaults"
        # here would mean "delete with a configuration nobody wrote", because the
        # defaults are enabled=True and dry_run=False.
        print(f"build-cache: refusing to run — {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if dry_run and not settings.dry_run:
        settings = replace(settings, dry_run=True)
    if not settings.enabled and not dry_run:
        # A disabled reaper is not a failure: say so and exit clean so the
        # launchd job stays green while the user has it switched off.
        print("build-cache: disabled (maintenance.build_cache.enabled is false)", file=sys.stderr)
        return EXIT_OK
    scan_roots = resolve_roots(roots, home)
    report = reap(
        roots=scan_roots,
        settings=settings,
        home=home,
        clock=clock,
        # Resolved HERE (``~`` expansion, ``realpath``), so only this layer knows
        # the operator gave no ``--root`` — which is what the live-fire warning in
        # ``reap`` is about.
        default_roots=not roots,
    )
    write_log(report)
    for warning in report.warnings:
        print(f"build-cache: {warning}", file=sys.stderr)
    if json_output:
        print(jsonlib.dumps(report.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return EXIT_NO_READABLE_ROOT if report.all_roots_unreadable else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point: ``python -m local_operator.build_cache_reaper``.

    The ``lop build-cache reap`` subcommand parses its own options in
    :mod:`local_operator.cli` and calls :func:`run_cli` directly, so both
    surfaces share one implementation and one set of defaults. Kept here so the
    reaper can be exercised (and scheduled) without the full CLI import graph.
    """
    parser = argparse.ArgumentParser(
        prog="lop build-cache reap",
        description=(
            "Prune regenerable build-cache directories when nothing is using them. "
            "Removals are logged; a dry run reports the same candidates and bytes "
            "without deleting."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the report as JSON instead of a summary.",
    )
    parser.add_argument(
        "--root",
        action="append",
        metavar="PATH",
        dest="roots",
        default=None,
        help=(
            "Scan this directory instead of the home directory. Repeatable; each "
            "use replaces the home default for this run, so a rehearsal against a "
            "fixture tree or a run bounded to one checkout needs no HOME change."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_cli(
        dry_run=args.dry_run,
        json_output=args.json_output,
        roots=args.roots,
    )


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
