#!/usr/bin/env python3
"""Map a diff to the unit tests it affects, run them, and print PR evidence.

WHY THIS EXISTS
---------------
Pull-request CI used to run the whole unit suite (six shards, ~25 min each, 89%
of the workflow's runner time) on every push, and the job queue it created was
the slowest part of every review cycle (median 15.6 min, p90 77 min of waiting
before a job even started; measured 2026-10-24..11-07). The unit suite now runs
on `main` only (`.github/workflows/full-suite.yml`, post-merge and nightly), and
a PR carries TARGETED local evidence instead: the agent runs the tests its diff
can reach and pastes the command and result under `## Targeted test evidence`.

This module is the "which tests" half of that bargain. It is deliberately
SIMPLE and EXPLAINABLE, not complete:

* the changed test files themselves;
* `test_<stem>*.py` for every changed module `<stem>`, in the mirroring
  `tests/unit/<pkg>` directory (or below/above it, never `tests/unit`
  tree-wide), and `test_<pkg>_<stem>*.py` anywhere;
* tests that DIRECTLY import a changed module (depth 1 — not the transitive
  closure: through hub modules that closure is most of the tree, which is the
  whole-tree run this replaces);
* a changed `conftest.py` selects its own directory;
* a path none of those place selects the nearest `tests/unit/<subdir>` that
  mirrors its package directory — NEVER `tests/unit` itself; a path with no
  such directory is reported as unplaced and selects nothing extra;
* always `GUARD_TESTS`, the tree-wide ratchets that PR CI also runs.

What it is NOT: `scripts/ci_scope.py`'s scoped `test` job, which proves a
selection complete over the full parse graph and falls back to the whole tree
whenever it cannot. That proof is the right tool for an authoritative gate and
the wrong one here — on this tree it almost always falls back, and its graph
build alone exceeded 300 s under fleet load (measured 2026-11-07). Completeness
now comes from the `main` full-suite run, which files an issue when red; this
helper's job is a fast, judged first pass whose reasoning is printed per test,
so the agent can widen or narrow it deliberately.

A LARGE SELECTION IS KEPT, NOT REPLACED. Above `WARN_SELECTION_FILES` the
helper prints a warning that the agent should narrow it by judgement, and still
runs exactly what it selected. A silent whole-tree fallback is the behaviour
this replaces.

Stdlib only, like `ci_scope.py`, so a broken dependency set can never decide
what runs. It reuses `ci_scope`'s git helpers (merge base, both sides of a
rename, untracked files) so the two tools agree about what "the diff" is.

Usage:
    .venv/bin/python scripts/affected_tests.py                 # list, base = merge-base origin/main
    .venv/bin/python scripts/affected_tests.py --since <rev>   # another base
    .venv/bin/python scripts/affected_tests.py path/a.py ...   # explicit paths, no git
    .venv/bin/python scripts/affected_tests.py --run           # run through run_bounded.py
    .venv/bin/python scripts/affected_tests.py --evidence      # run + markdown for the PR
    .venv/bin/python scripts/affected_tests.py --guards        # the guard list, one per line
    .venv/bin/python scripts/affected_tests.py --evidence --workers 2  # explicit xdist width
    .venv/bin/python scripts/affected_tests.py --evidence --chunk      # one pytest per target
    make test-affected                                         # == --evidence
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

# Imported as `scripts.affected_tests` (tests) or run as a script, whose own
# directory is then `sys.path[0]`; no sys.path mutation either way.
if __package__:
    from . import ci_scope
else:  # pragma: no cover - exercised by the subprocess tests
    import ci_scope

#: THE GUARD LIST — the one place it is spelled. `ci.yml`'s `guards` job runs
#: `--guards`, this helper appends it to every selection, and
#: `tests/unit/test_ci_hygiene.py` asserts the job reads it from here.
#:
#: Membership rule: a test that scans the WHOLE tree (or a workflow/packaging
#: manifest) for a forbidden shape, so a change ANYWHERE can break it and no
#: path-based selection would pick it. These are the regressions targeted local
#: runs miss by construction — measured in the 2026-11 flaky sweep, PR CI's
#: full suite caught undeclared mutating calls only through
#: `test_sync_copy_set.py` and `test_no_session_deletion.py`. Behavioural tests
#: of one module do not belong here, however important: the diff selects them.
#:
#: Local wall time for the whole list (984 tests), measured 2026-11-07 on this
#: 14-core host under load average 110-260, isolated HOME, through
#: run_bounded: n=3, 158 / 114 / 99 s, median 114 s, max 158 s. CI's `guards`
#: job is sized from that (see its `timeout-minutes`). Keep the list fast: a
#: large behavioural file joins as its inventory NODES, not whole.
GUARD_TESTS: tuple[str, ...] = (
    # CI/workflow structure, including the release-safety jobs it pins.
    "tests/unit/test_ci_hygiene.py",
    "tests/unit/test_publish_workflow.py",
    # Startup import-graph ratchets.
    "tests/unit/test_import_graph.py",
    "tests/unit/test_startup_imports.py",
    # Undeclared mutating calls / deletion paths (escapes: 4 of the 7 GUARD).
    "tests/unit/network/test_sync_copy_set.py",
    "tests/unit/session/test_no_session_deletion.py",
    # Tree-wide censuses: transcript probes, cwd producers, env reads,
    # Pydantic annotation shadowing.
    "tests/unit/session/test_transcript_reads_are_archive_aware.py",
    "tests/unit/session/runtime/test_cwd_omission_census.py",
    "tests/unit/test_ambient_env_isolation.py",
    "tests/unit/test_pydantic_annotation_shadowing.py",
    # Shipping contracts: package-data globs cover the files a wheel needs.
    "tests/unit/i18n/test_package_data.py",
    "tests/unit/skills/test_builtin_package_data.py",
    "tests/unit/mobile/test_web_package_data.py",
    # i18n catalogue snapshots, non-translatable-sink counts, ledger shards
    # (escapes: #3449, feat/i18n-relay-hi-catalogues).
    "tests/unit/i18n/test_catalogues.py",
    "tests/unit/i18n/test_non_translatable_sinks.py",
    "tests/unit/i18n/test_ledger_shards.py",
    # Inventory NODES inside large behavioural files — the whole files are
    # Textual-pilot heavy, so only the registry/field-count pins ride along.
    # Slash registry + argument-mode inventory (escape: /sound).
    "tests/unit/tui/test_approvals_ux.py::test_the_registry_states_which_commands_offer_values",
    "tests/unit/tui/test_command_picker.py::test_bare_slash_suggests_the_full_registry_in_order",
    # Viewer-protocol member inventory (escape: viewer_only 75 -> 76).
    "tests/unit/session/test_viewer_protocol.py"
    "::test_the_viewer_protocol_covers_what_only_the_facade_has",
    "tests/unit/session/test_viewer_protocol.py"
    "::test_the_quoted_isinstance_member_count_is_still_true",
    # Generated-file freshness, incl. extension/ui-vendor/** (escape:
    # fix/browser-ax-value regenerated a bundle the PR never touched).
    "tests/unit/browser_bridge/test_protocol.py::test_generated_types_are_current",
)

#: Above this many selected FILES the helper warns (and still runs them all).
#: 150 is ~7% of the 2,179 files under tests/unit on 2026-11-07: past that the
#: selection has stopped being "the tests this change touches" and is usually a
#: hub module's importer fan-out, which judgement should narrow.
WARN_SELECTION_FILES = 150

#: The unit tree this helper selects from. e2e is a separate, opt-in stage.
UNIT_ROOT = "tests/unit"

#: INFRA paths: a change here alters isolation, ordering, bounding or test
#: selection for EVERY test, so no targeted selection is evidence for it and a
#: local whole-tree run is what this helper exists to avoid. The helper prints
#: a `WIDE:` directive to dispatch full-suite.yml on the branch instead. The
#: escaped-regression analysis (35 red PR runs, 2026-11) found isolation
#: breaks from exactly these files that no import/name selection reached.
#: Exact paths; `conftest.py` anywhere under tests/ is matched by name below.
INFRA_PATHS: frozenset[str] = frozenset(
    {
        "conftest.py",
        "scripts/run_bounded.py",
        "scripts/ci_scope.py",
        # Root-conftest-loaded and `-p`-loaded pytest plugins.
        "tests/ci_order_owned_first.py",
        "tests/shard_memory_dump.py",
        "tests/shard_stall_watchdog.py",
        "tests/tree_identity.py",
    }
)

#: Basenames too generic to grep for (they would select most of the tree).
_GENERIC_BASENAMES = frozenset({"__init__.py", "conftest.py", "__main__.py", "README.md"})

#: The `base` sentinel for explicit positional paths (no diff range).
EXPLICIT = "explicit"

#: run_bounded's bound for `--run`, seconds. Generous on purpose — a bound that
#: fires on a legitimately slow, loaded host is a false red — and overridable.
DEFAULT_TIMEOUT = 3600

#: WORKER CAP for `--run`/`--evidence`. The repo's addopts say `-n auto`, and
#: even through conftest's `pytest_xdist_auto_num_workers` cap (2..8) that is
#: too wide for a targeted run on this shared host: measured 2026-11-08, a
#: 12-file docs/seed diff selected 109 targets, launched xdist on a 14-core /
#: 36 GB machine, and the machine memory guard SIGKILLed the run four times at
#: 1.1-2.0 GB per process group, so no targeted evidence could be produced at
#: all. A targeted selection is small by construction, so a narrow pool loses
#: little wall time and keeps the group under the guard. An explicit `-n N`
#: on the command line overrides addopts' `-n auto` (pytest takes the last).
MAX_DEFAULT_WORKERS = 4
#: At or below this many targets xdist's worker start-up costs more than it
#: buys, and `-n0` is also the cheapest memory shape: run in-process.
SERIAL_TARGETS = 3
#: Per-session override, same precedence slot as `--workers` but below it, so a
#: memory-starved agent can pin the width once (`LOP_AFFECTED_WORKERS=0`)
#: without threading a flag through `make test-affected`.
WORKERS_ENV = "LOP_AFFECTED_WORKERS"

_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")


@dataclass
class Selection:
    """What to run and why. `reasons` maps each target to its first reason."""

    paths: tuple[str, ...]
    targets: tuple[str, ...]
    reasons: Mapping[str, str]
    unplaced: tuple[str, ...]
    warnings: tuple[str, ...] = field(default=())
    #: INFRA paths in the diff; non-empty means "dispatch the full suite".
    wide: tuple[str, ...] = field(default=())


def _is_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return (
        rel.startswith("tests/")
        and name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


def _module_of(rel: str) -> str | None:
    """The dotted name a repo `.py` file is imported as, or None."""
    if not rel.endswith(".py"):
        return None
    return ci_scope._module_name(rel)


def _index(rel: str, source: str) -> tuple[frozenset[str], str]:
    """A test's direct imports and its string literals, from ONE parse.

    `select` walks all ~2,200 unit test files on every call, so the AST parse
    is the bulk of its cost; parsing each file once (not once per index) is
    what keeps a single-path call cheap (review R3/QA Q4 on #3592).
    """
    try:
        tree = ast.parse(source, filename=rel)
    except (SyntaxError, ValueError):
        # See the two helpers: no imports, and raw source as the literals.
        return frozenset(), source
    return _imported_names(rel, tree), _string_literals(tree)


def _imported_names(rel: str, tree: ast.AST) -> frozenset[str]:
    """Dotted names a file imports, DIRECT only.

    `from a.b import c` yields both `a.b` and `a.b.c`, because `c` may be a
    submodule; an over-match there costs one extra test file, an under-match a
    missed one. Relative imports are resolved against the file's package.
    Unparseable files import nothing (the changed-test rule still selects them,
    and pytest reports the syntax error).
    """
    package = ci_scope._package_parts(rel)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                head = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                head = node.module or ""
            if head:
                names.add(head)
                names.update(f"{head}.{alias.name}" for alias in node.names if alias.name != "*")
    return frozenset(names)


def _string_literals(tree: ast.AST) -> str:
    """A test's string constants, docstrings excluded, one per line.

    DATA-AS-TEXT matching runs over these, not the raw source: a test that
    READS `AGENTS.md` spells it in a literal (`ROOT / "AGENTS.md"`), while
    155 tests merely cite it in a docstring or comment — measured on this
    branch, where raw-source matching selected 190 files for a four-word
    AGENTS.md edit. Unparseable files fall back to the raw source (over-select).
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return "\n".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    )


def _unit_test_files(root: Path) -> list[str]:
    base = root / UNIT_ROOT
    if not base.is_dir():
        return []
    return sorted(
        p.relative_to(root).as_posix()
        for p in base.rglob("*.py")
        if _is_test_file(p.relative_to(root).as_posix())
    )


def is_infra(rel: str) -> bool:
    """Whether `rel` is an INFRA path (see `INFRA_PATHS`)."""
    return rel in INFRA_PATHS or (rel.startswith("tests/") and rel.endswith("/conftest.py"))


def _mirror_parts(rel: str) -> tuple[str, ...]:
    """`rel`'s directory as the `tests/unit/<...>` path that would mirror it.

    `local_operator/tunnels/cli.py` -> (`tunnels`,); `local_operator/cli.py`
    -> (); `tests/unit/pkg/helpers.py` -> (`pkg`,); `scripts/x.py` ->
    (`scripts`,). Shared by the name rule and `_nearest_test_dir`.
    """
    parts = rel.split("/")[:-1]
    if parts and parts[0] in ("local_operator", "tests"):
        parts = parts[1:]
        if parts and parts[0] == "unit":
            parts = parts[1:]
    return tuple(parts)


def _in_mirror_scope(test: str, pkg: tuple[str, ...]) -> bool:
    """Whether `test` sits in `tests/unit/<pkg>`, below it, or in an ancestor
    short of `tests/unit`; for `pkg == ()`, in `tests/unit` itself only."""
    test_dir = tuple(test.split("/")[2:-1])  # below `tests/unit`
    if not pkg:
        return not test_dir
    return test_dir[: len(pkg)] == pkg or (bool(test_dir) and pkg[: len(test_dir)] == test_dir)


def _named_after(test: str, pkg: tuple[str, ...], stem: str) -> bool:
    """Whether `test` is named for module `stem` of mirrored package `pkg`.

    SCOPED, not tree-wide (review R1/QA Q1 on #3592): `test_<stem>*.py`
    counts only where it can be about THIS module, i.e. in the mirroring
    directory `tests/unit/<pkg>`, below it, or in one of its ancestors short
    of `tests/unit` itself. A top-level module (`pkg == ()`) mirrors
    `tests/unit` and matches only that directory's own files. Elsewhere a
    test must carry the package in its name, `test_<pkg[-1]>_<stem>*.py`.
    Without the scope a generic stem pulled every same-named test in the
    tree: `tunnels/cli.py` selected 42 `i18n/test_cli_*` files and
    `team_bridge/store.py` selected `analytics/test_store.py`.
    """
    name = test.rsplit("/", 1)[-1]
    prefixes = [f"test_{stem}"] if _in_mirror_scope(test, pkg) else []
    if pkg:
        prefixes.append(f"test_{pkg[-1]}_{stem}")
    return any(
        name.startswith(prefix) and (len(name) == len(prefix) + 3 or name[len(prefix)] in "_.")
        for prefix in prefixes
    )


def _shared_basenames(root: Path) -> frozenset[str]:
    """Basenames that more than one tracked file carries (`GUIDE.md` x33,
    `wire.settings.json` x15 on 2026-11-07). Without git every basename
    counts as shared, which only narrows the bare-basename needle."""
    rc, out, _ = ci_scope._git(["ls-files", "-z"], cwd=root)
    if rc != 0:
        return frozenset({"*"})
    seen: set[str] = set()
    shared: set[str] = set()
    for path in out.split("\0"):
        name = path.rsplit("/", 1)[-1]
        (shared if name in seen else seen).add(name)
    return frozenset(shared)


def _text_needles(rel: str, *, shared_basename: bool = True) -> tuple[re.Pattern[str], ...]:
    """What a test that reads `rel` AS DATA would spell in its string literals.

    * The repo-relative path, always. For a `.py` file the dotted module name
      too (`importlib.import_module("local_operator.x")`, `monkeypatch.setattr
      ("local_operator.x.f", ...)`). A `.py` file gets NO bare-basename needle:
      the import rule covers its real readers, and `cli.py`/`store.py` occur in
      unrelated literals (`"local_operator/cli.py"` matched `tunnels/cli.py`;
      review R1/QA Q1 on #3592).
    * A non-Python file's basename unless generic. Bare when no other tracked
      file shares it; when one does (`shared_basename`), only after its own
      parent directory or after an ancestor plus a glob
      (`glob("guides/*/GUIDE.md")` for `local_operator/guides/x/GUIDE.md`),
      because `en/wire.settings.json` in a catalogue test is not a reader of
      `i18n/audit-history/hi/wire.settings.json` (QA Q2 on #3592).
    * A non-Python file's parent directory as a PATH SEGMENT, only inside a
      path-shaped literal: `/`-bounded within a literal that contains a `/`
      (`f"local_operator/agent_seeds/{name}.md"`). Never a bare word: the
      parent of `i18n/audit-history/hi/x.json` is `hi`, and `"hi"` is a test
      message in ~200 files (QA Q2 on #3592). The segment needs a path
      segment before it or a `/` after it, so the slash command `"/hi"` is
      not a path. One exception, a WHOLE literal naming the parent when its
      name is a compound identifier (contains `_` or `-`): `tmp_path /
      "agent_seeds"` is how test_agent_profiles.py spells the seeds it copies,
      `... / "guides" / "team-bridge"` how test_setup_cli.py reaches its
      guide, and a compound name is never an English word in a test message.
    """
    needles = [re.compile(re.escape(rel))]
    module = _module_of(rel)
    if module:
        needles.append(re.compile(r"(?<![\w.])" + re.escape(module) + r"(?![\w])"))
        return tuple(needles)
    parts = rel.split("/")
    name = parts[-1]
    if name not in _GENERIC_BASENAMES:
        bare = r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])"
        if not shared_basename:
            needles.append(re.compile(bare))
        elif len(parts) >= 3:
            # Its own parent (`team-bridge/GUIDE.md`), or an ancestor and a
            # glob (`guides/*/GUIDE.md`, `audit-history/**/x.json`). Another
            # sibling's path (`/guides/browser/GUIDE.md`) is not a reader.
            parent = re.escape(parts[-2])
            dirs = "|".join(re.escape(d) for d in parts[1:-1])
            needles.append(
                re.compile(
                    rf"(?:^|/){parent}/{bare}|(?:^|/)(?:{dirs})/[^\n]*\*[^\n]*{bare}",
                    re.MULTILINE,
                )
            )
    if len(parts) >= 3:
        # Over string LITERALS (see `_string_literals`), one per line, so `^`
        # and `$` bound one literal and the required `/` sits inside it.
        segment = re.escape(parts[-2])
        needles.append(
            re.compile(
                rf"(?:^|/){segment}/|[^/\s]/{segment}(?=$|/)",
                re.MULTILINE,
            )
        )
        if re.search(r"[_-]", parts[-2]):
            needles.append(re.compile(rf"^{segment}$", re.MULTILINE))
    return tuple(needles)


def _nearest_test_dir(root: Path, rel: str) -> str | None:
    """The deepest `tests/unit/<pkg path>` mirroring `rel`'s directory.

    `local_operator/tui/styles/app.tcss` -> `tests/unit/tui/styles` if it
    exists, else `tests/unit/tui`. Stops BEFORE `tests/unit` itself: that would
    be the whole tree, which this helper never selects implicitly.
    """
    parts = _mirror_parts(rel)
    while parts:
        candidate = root / UNIT_ROOT / Path(*parts)
        if candidate.is_dir() and next(candidate.rglob("test_*.py"), None) is not None:
            return candidate.relative_to(root).as_posix()
        parts = parts[:-1]
    return None


def select(root: Path, paths: Sequence[str], *, include_guards: bool = True) -> Selection:
    """The targeted node set for `paths` (repo-relative), with reasons."""
    root = root.resolve()
    tests = _unit_test_files(root)
    imports: dict[str, frozenset[str]] = {}
    sources: dict[str, str] = {}
    for rel in tests:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        imports[rel], sources[rel] = _index(rel, text)

    reasons: dict[str, str] = {}
    unplaced: list[str] = []
    shared: frozenset[str] | None = None

    def _is_shared(rel: str) -> bool:
        # Lazily: one `git ls-files` per call, and only when a non-Python
        # path needs the basename rule.
        nonlocal shared
        if shared is None:
            shared = _shared_basenames(root)
        name = rel.rsplit("/", 1)[-1]
        return "*" in shared or name in shared

    def add(target: str, why: str) -> None:
        reasons.setdefault(target, why)

    wide = [rel for rel in paths if is_infra(rel)]
    for rel in paths:
        placed = False
        if rel in ("conftest.py", "tests/conftest.py", f"{UNIT_ROOT}/conftest.py"):
            # A root conftest reaches every test and has no tests of its own:
            # reported as WIDE, never widened to the tree. Other INFRA paths
            # (run_bounded, ci_scope, plugins) still select their own tests.
            continue
        if _is_test_file(rel):
            if (root / rel).is_file():
                add(rel, "changed test file")
            # A deleted test file selects nothing; its absence is the change.
            placed = True
        elif rel.endswith("/conftest.py") and rel.startswith("tests/"):
            directory = rel.rsplit("/", 1)[0]
            # The root conftests reach every test; selecting their directory
            # would be the implicit whole-tree run, so they fall through to the
            # unplaced report instead (the main full-suite run covers them).
            if directory not in ("tests", UNIT_ROOT) and (root / directory).is_dir():
                add(directory, f"its conftest.py changed ({rel})")
                placed = True
        module = _module_of(rel) if not _is_test_file(rel) else None
        if module:
            stem = rel.rsplit("/", 1)[-1][:-3]
            if stem == "__init__":
                stem = module.rsplit(".", 1)[-1]
            pkg = _mirror_parts(rel)
            # A package's `__init__` is tested both beside the package
            # (`tests/unit/test_tunnels.py`) and inside it (`tests/unit/tunnels/`).
            scopes = (pkg, pkg[:-1]) if rel.endswith("/__init__.py") and pkg else (pkg,)
            for test in tests:
                if any(_named_after(test, scope, stem) for scope in scopes):
                    add(test, f"named after {rel}")
                    placed = True
                if module in imports[test]:
                    add(test, f"imports {module}")
                    placed = True
        # DATA-AS-TEXT: tests that read this file as data (a seed prompt, a
        # GUIDE.md, a catalogue) import nothing from it; their source names it.
        if not _is_test_file(rel):
            shared_name = not rel.endswith(".py") and _is_shared(rel)
            needles = _text_needles(rel, shared_basename=shared_name)
            # A shared basename as a WHOLE literal (`base / "GUIDE.md"`) is how
            # a test reads EVERY file of that kind (guides/test_guides.py
            # validates all guides through `discover_guides()`), so it counts
            # for tests in the package's mirror scope. Only under
            # `local_operator/`: `tests/unit` mirrors the package, not repo-root
            # data such as `i18n/audit-history/`, where `"wire.settings.json"`
            # in i18n/test_messages.py is a fixture of its own (QA Q2).
            scoped: re.Pattern[str] | None = None
            # Bound on every path so the `scoped is not None` guard below is the
            # only gate on reading it; `()` is never consulted while `scoped` is None.
            pkg: tuple[str, ...] = ()
            name = rel.rsplit("/", 1)[-1]
            if shared_name and rel.startswith("local_operator/") and name not in _GENERIC_BASENAMES:
                scoped = re.compile(rf"^{re.escape(name)}$", re.MULTILINE)
                pkg = _mirror_parts(rel)
            for test in tests:
                if test == rel:
                    continue
                if any(n.search(sources[test]) for n in needles) or (
                    scoped is not None
                    and _in_mirror_scope(test, pkg)
                    and scoped.search(sources[test])
                ):
                    add(test, f"names {rel} in its source")
                    placed = True
        if not placed:
            nearest = _nearest_test_dir(root, rel)
            if nearest:
                add(nearest, f"nearest test directory for {rel} (no name or import match)")
            else:
                unplaced.append(rel)

    if include_guards:
        for guard in GUARD_TESTS:
            add(guard, "guard (tree-wide ratchet, always run)")

    targets = tuple(sorted(reasons))
    warnings: list[str] = []
    files = sum(1 for t in targets if t.split("::", 1)[0].endswith(".py"))
    if files > WARN_SELECTION_FILES:
        warnings.append(
            f"{files} test files selected (> {WARN_SELECTION_FILES}): this is usually a hub "
            "module's importer fan-out. The full selection is kept; narrow it by "
            "judgement (pass the paths you mean) if it is more than the change warrants."
        )
    return Selection(
        paths=tuple(paths),
        targets=targets,
        reasons={t: reasons[t] for t in targets},
        unplaced=tuple(unplaced),
        warnings=tuple(warnings),
        wide=tuple(wide),
    )


def wide_directive(wide: Sequence[str]) -> str:
    """The `WIDE:` line for an infra change, with the exact dispatch command.

    A recommendation, not a local run: the whole tree locally is the cost
    this helper removes, and full-suite.yml runs it on clean runners in
    shards. `workflow_dispatch` runs on any ref, in its own concurrency group,
    and does not file or close the `ci-main-red` issue off `main`.
    """
    return (
        f"WIDE: infra change ({', '.join(wide)}), recommend dispatching full-suite.yml "
        "via workflow_dispatch on this branch: "
        'gh workflow run full-suite.yml --ref "$(git branch --show-current)"'
    )


def resolve_workers(
    targets: int,
    *,
    override: int | None = None,
    env: Mapping[str, str] | None = None,
    cpu_count: int | None = None,
) -> tuple[int, str]:
    """The xdist worker count for a run, and the term that decided it.

    Precedence: `--workers` > `$LOP_AFFECTED_WORKERS` > the default cap
    `min(MAX_DEFAULT_WORKERS, cpu_count, targets)`, which drops to 0 (`-n0`,
    in-process) at `SERIAL_TARGETS` or fewer. `targets` counts selection
    entries (a directory is one entry), so the cap can only err narrow. The
    reason string goes into the evidence block: a reader must be able to tell
    a capped run from an operator's explicit choice.
    """
    if override is not None:
        return max(0, override), "--workers"
    raw = (env if env is not None else os.environ).get(WORKERS_ENV, "").strip()
    if raw:
        try:
            return max(0, int(raw)), f"${WORKERS_ENV}"
        except ValueError:
            # A typo must not silently re-widen the run to `-n auto`; fall
            # through to the cap and say so in the reason.
            pass
    if targets <= SERIAL_TARGETS:
        return 0, f"default: <= {SERIAL_TARGETS} targets run in-process"
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    workers = min(MAX_DEFAULT_WORKERS, cpus, targets)
    reason = f"default cap: min({MAX_DEFAULT_WORKERS}, {cpus} cpus, {targets} targets)"
    if raw:
        reason += f"; ignored non-integer ${WORKERS_ENV}={raw!r}"
    return workers, reason


def pytest_command(targets: Iterable[str], python: str, workers: int | None = None) -> list[str]:
    """The pytest argv, with the TUI suite's colour-capable terminal.

    `env -u NO_COLOR TERM=xterm-256color` is the documented spelling (AGENTS.md
    "TUI tests need a colour-capable terminal"): Textual pilot tests fail under
    NO_COLOR for a reason CI never sees. `workers` (see `resolve_workers`)
    becomes an explicit `-n`, which beats addopts' `-n auto`; None leaves
    addopts in charge (the listing-only path prints that form).
    """
    width = [] if workers is None else ["-n", str(workers)]
    return [
        "env",
        "-u",
        "NO_COLOR",
        "TERM=xterm-256color",
        python,
        "-m",
        "pytest",
        "-q",
        *width,
        *targets,
    ]


def bounded_command(command: Sequence[str], python: str, timeout: int) -> list[str]:
    """Wrap in run_bounded.py so a hung run is group-reaped (command-status:
    we need pytest's status, and cleanup-required refuses on Darwin)."""
    wrapper = str(Path(__file__).resolve().parent / "run_bounded.py")
    return [python, wrapper, "--contract=command-status", "--timeout", str(timeout), "--", *command]


def parse_counts(output: str) -> dict[str, int]:
    """pytest's final summary counts (`3 passed, 1 failed in 2.1s`)."""
    counts: dict[str, int] = {}
    for line in reversed(output.splitlines()):
        found = _SUMMARY.findall(line)
        if found and (" in " in line or "passed" in line or "failed" in line):
            for number, word in found:
                key = "error" if word == "errors" else word
                counts[key] = counts.get(key, 0) + int(number)
            break
    return counts


@dataclass
class ChunkedResult:
    """Aggregate of `--chunk`: one serial pytest invocation per target."""

    rc: int
    counts: dict[str, int]
    invocations: int
    failed: list[str]


def run_chunked(
    targets: Sequence[str],
    run_one: Callable[[str], tuple[int, str]],
) -> ChunkedResult:
    """Run each target alone, serially, and sum the pytest summaries.

    `--chunk` exists for the host where even a capped pool trips the memory
    guard: one process group at a time, each short-lived, so peak RSS is one
    test file's, not the selection's. `run_one(target)` returns (rc, output);
    injected so tests never fan out real subprocesses.

    Exit status: pytest's 5 ("no tests collected") is normal for a lone target
    whose tests are all deselected (`-m "not e2e"`), which a combined run would
    never report, so it only surfaces when EVERY chunk collected nothing.
    Otherwise the first real non-zero status wins and every failing target is
    named, so one red file cannot hide behind the others.
    """
    counts: dict[str, int] = {}
    codes: list[int] = []
    failed: list[str] = []
    for target in targets:
        rc, output = run_one(target)
        codes.append(rc)
        if rc not in (0, 5):
            failed.append(target)
        for key, number in parse_counts(output).items():
            counts[key] = counts.get(key, 0) + number
    real = [rc for rc in codes if rc not in (0, 5)]
    if real:
        rc = real[0]
    elif codes and all(rc == 5 for rc in codes):
        rc = 5
    else:
        rc = 0
    return ChunkedResult(rc=rc, counts=counts, invocations=len(codes), failed=failed)


def evidence_markdown(
    *,
    base: str,
    head: str,
    selection: Selection,
    command: Sequence[str],
    rc: int,
    counts: Mapping[str, int],
    wall: float,
    workers: tuple[int, str] | None = None,
    chunked: ChunkedResult | None = None,
) -> str:
    verdict = "PASS" if rc == 0 else f"FAIL (rc={rc})"
    tally = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "no pytest summary parsed"
    files = sum(1 for t in selection.targets if t.split("::", 1)[0].endswith(".py"))
    dirs = len(selection.targets) - files
    guards = sum(1 for t in selection.targets if t in GUARD_TESTS)
    named = ", ".join(f"`{p}`" for p in selection.unplaced)
    unplaced = f" ({len(selection.unplaced)} unplaced: {named})" if selection.unplaced else ""
    lines = [
        "## Targeted test evidence",
        "",
        # Explicit paths have no diff range, but the selection still depends
        # on the tree it was computed in: name the commit (QA Q3 on #3592).
        (
            f"- range: explicit paths at `{_short(head)}`"
            if base == EXPLICIT
            else f"- range: `{_short(base)}..{_short(head)}`"
        ),
        f"- changed paths: {len(selection.paths)}" + unplaced,
        f"- selected: {files} test files + {dirs} directories "
        f"(guards included: {guards}/{len(GUARD_TESTS)})",
        f"- result: **{verdict}** — {tally}",
        f"- wall time: {wall:.1f} s",
    ]
    if workers is not None:
        lines.append(f"- workers: `-n {workers[0]}` ({workers[1]})")
    if chunked is not None:
        # Chunked counts are a SUM of separate sessions, not one pytest run:
        # the block must say so, or a reader compares it with a combined run.
        lines.append(
            f"- mode: **chunked** — {chunked.invocations} serial pytest invocations, "
            "one per target, counts summed; the command below ran once per target"
        )
        if chunked.failed:
            lines.append("- failing targets: " + ", ".join(f"`{t}`" for t in chunked.failed))
    lines += ["", "```sh", " ".join(_display(command)), "```"]
    lines.extend(f"\n> warning: {w}" for w in selection.warnings)
    if selection.wide:
        lines.append(f"\n> {wide_directive(selection.wide)}")
    return "\n".join(lines) + "\n"


def _short(rev: str) -> str:
    """A 12-char SHA, keeping a `+worktree` (uncommitted changes) suffix."""
    sha, plus, rest = rev.partition("+")
    return (sha[:12] if re.fullmatch(r"[0-9a-f]{13,}", sha) else sha) + plus + rest


def _display(command: Sequence[str]) -> list[str]:
    return [shlex.quote(token) for token in command]


def _git_out(root: Path, *args: str) -> str:
    rc, out, _ = ci_scope._git(list(args), cwd=root)
    return out.strip() if rc == 0 else ""


def _head_identity(root: Path) -> str:
    """HEAD's SHA, `+worktree` when the tree has uncommitted changes."""
    head = _git_out(root, "rev-parse", "HEAD") or "HEAD"
    if _git_out(root, "status", "--porcelain"):
        head += "+worktree"
    return head


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("paths", nargs="*", help="explicit repo-relative paths (skips git)")
    parser.add_argument(
        "--since", default="origin/main", help="base rev; its merge base with HEAD is used"
    )
    parser.add_argument(
        "--root", default=None, help="repository root (default: this script's repo)"
    )
    parser.add_argument(
        "--guards",
        action="store_true",
        help="print GUARD_TESTS one per line (with --run: run only them)",
    )
    parser.add_argument(
        "--no-guards", action="store_true", help="omit the guard list from the selection"
    )
    parser.add_argument(
        "--run", action="store_true", help="run the selection through run_bounded.py"
    )
    parser.add_argument(
        "--evidence", action="store_true", help="run, then print the PR evidence block"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter for pytest (default: this one)"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="run_bounded bound, seconds"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"xdist workers for --run/--evidence (0 = in-process); default "
            f"${WORKERS_ENV}, else min({MAX_DEFAULT_WORKERS}, cpus, targets), "
            f"0 at <= {SERIAL_TARGETS} targets"
        ),
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="run one pytest per target, serially, and sum the counts (lowest memory)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve() if args.root else ci_scope.default_root(Path(__file__))

    if args.guards and not (args.run or args.evidence):
        print("\n".join(GUARD_TESTS))
        return 0

    if args.guards:
        # `--guards --run`: the guard list alone, as ci.yml's `guards` job runs
        # it (`make check-changed` uses this spelling for that job).
        paths = []
        base = head = "guards"
    elif args.paths:
        paths = [ci_scope._norm(p) for p in args.paths]
        base = EXPLICIT
        head = _head_identity(root)
    else:
        resolved = ci_scope.resolve_base(args.since, cwd=root)
        if resolved is None:
            print(
                f"affected_tests: cannot resolve base {args.since!r}; pass paths explicitly",
                file=sys.stderr,
            )
            return 2
        collected = ci_scope.collect_paths(resolved, local=True, cwd=root)
        if collected is None:
            print("affected_tests: git diff failed; pass paths explicitly", file=sys.stderr)
            return 2
        paths, base = collected, resolved
        head = _head_identity(root)

    selection = select(root, paths, include_guards=not args.no_guards)
    stream = sys.stderr if args.evidence else sys.stdout
    print(f"changed paths: {len(paths)}; selected targets: {len(selection.targets)}", file=stream)
    for target in selection.targets:
        print(f"  {target}  # {selection.reasons[target]}", file=stream)
    for rel in selection.unplaced:
        print(
            f"unplaced: {rel} (no name/import match and no mirroring tests/unit directory)",
            file=stream,
        )
    for warning in selection.warnings:
        print(f"WARNING: {warning}", file=stream)
    if selection.wide:
        print(wide_directive(selection.wide), file=stream)

    if not (args.run or args.evidence):
        command = pytest_command(selection.targets, args.python)
        print("command: " + " ".join(_display(command)), file=stream)
        return 0
    if not selection.targets:
        print("nothing selected; nothing to run", file=stream)
        return 0

    # A chunk is one target, so the default resolves to `-n0` per chunk; an
    # explicit --workers / env value still applies to each chunk as given.
    workers = resolve_workers(1 if args.chunk else len(selection.targets), override=args.workers)
    print(f"workers: -n {workers[0]} ({workers[1]})", file=stream)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}

    def run(targets: Sequence[str]) -> tuple[int, str]:
        proc = subprocess.Popen(
            bounded_command(
                pytest_command(targets, args.python, workers[0]), args.python, args.timeout
            ),
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        captured: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            stream.write(line)
        return proc.wait(), "".join(captured)

    started = time.monotonic()
    chunked: ChunkedResult | None = None
    if args.chunk:
        chunked = run_chunked(selection.targets, lambda target: run([target]))
        rc, counts = chunked.rc, chunked.counts
        command = pytest_command(["<target>"], args.python, workers[0])
    else:
        rc, output = run(selection.targets)
        counts = parse_counts(output)
        command = pytest_command(selection.targets, args.python, workers[0])
    wall = time.monotonic() - started
    if args.evidence:
        print(
            evidence_markdown(
                base=base,
                head=head,
                selection=selection,
                command=command,
                rc=rc,
                counts=counts,
                wall=wall,
                workers=workers,
                chunked=chunked,
            )
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
