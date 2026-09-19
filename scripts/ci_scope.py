#!/usr/bin/env python3
"""Decide which CI jobs a diff can possibly affect, and run them locally.

WHY THIS EXISTS
---------------
`ci.yml` used to run its whole job set on every pull request. A one-line docs
edit paid for the five-shard unit matrix, a Windows run, a macOS TUI run, a
live-LLM sanity run and a dependency audit — 16 checks for a file no gate
reads. The operator's motivating PR (#1238, `docs/store/release-record.md`
only) is the measured case.

The fix is not "run less on small PRs", which is unverifiable taste. It is a
*classifier* that answers one question per job: does anything in this diff
appear in what that job verifies? This module is that classifier, and it is
the SINGLE SOURCE OF TRUTH for both answers: `ci.yml`'s `changes` job exports
one output per flag from here, and `make check-changed` runs the same module
to pick the local gates. A second, hand-written mapping in either place is the
defect this design exists to prevent (asserted by `tests/unit/test_ci_hygiene.py`).

DESIGN RULES, AND WHY EACH ONE IS LOAD-BEARING
----------------------------------------------
Fail OPEN, never closed. A path must *match* an inert rule to be skippable;
anything unrecognised counts as live. The opposite shape — a denylist of code
paths — goes green on the path nobody enumerated, which is exactly the
pathology both AGENTS.md files name: a guard nothing runs is indistinguishable
from no guard. Two independent fail-open layers exist: this module (which
returns every flag `true` when it cannot resolve a diff) and the workflow
(every gate reads `<flag> != 'false'`, so an output that was never written runs
the job rather than skipping it).

Skipping must be LEGIBLE. `--summary` writes the base SHA, every changed path
with the category it got, every flag with its reason, and the resulting
run/skip job list into `$GITHUB_STEP_SUMMARY`. Without that, "the PR was green"
stops meaning anything, because a skipped guard leaves no trace.

Stdlib only (`tomllib`-free — no parsing of the manifest beyond the version
line). A classifier that needs an install is a classifier that goes red for
reasons unrelated to classification, the same argument
`scripts/version-bump-guard.mjs` makes for itself.

INVOCATION
----------
CI (`--github-output`/`--summary` are the Actions channel files)::

    python scripts/ci_scope.py --event "$GITHUB_EVENT_NAME" --base "$sha" \
        --root "$GITHUB_WORKSPACE" \
        --github-output "$GITHUB_OUTPUT" --summary "$GITHUB_STEP_SUMMARY"

`--root` is passed EXPLICITLY by the workflow even though `default_root`
resolves the repository on its own, and that redundancy is the point: the step
runs a COPY of this file from `$RUNNER_TEMP`, so a module that trusted
`__file__` would compute `/home/runner/work` as its repository, `git diff` would
exit 128 there, and the fail-open branch would set every flag true on every pull
request — correct classifier, dead gate. Two independent fixes beat one clever
default; see `default_root` for the resolution order.

The report goes to **stdout always** (the run log) and additionally to
`--summary` when given. Both, because the summary is what a reviewer is told to
read and the log is where anyone debugging a run actually looks.

Local (index + working tree + untracked, then run the selected gates)::

    python scripts/ci_scope.py --since "$(git merge-base origin/main HEAD)" --run

`--run` prints each job it will NOT run (see `LOCAL_EXCLUSIONS`) with the reason,
so a local green is never quietly narrower than it looks.

Exit status is 0 for every classification outcome, including the fail-open
ones — this module's verdict is only "how much to run", and refusing would red
a PR for an infrastructure reason and teach people to bypass the gate. The one
exception is `--run`, which reports the gates' own exit status. A malformed
flag value is *not* a classification outcome: the `$GITHUB_OUTPUT` writer
raises rather than emitting an empty string (see `_flag_value`).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------
# One name per kind of path, produced by `category_of` with FIRST MATCH WINS in
# the order documented in the module's category table below. The categories
# exist so the flag predicates can be written as set membership over *kinds*
# rather than as a second list of prefixes in each predicate — the drift that
# would otherwise let "is this a Python change" and "does the unit suite care"
# disagree.

CAT_CI = "ci"
CAT_PYTHON = "python"
CAT_WEB = "web"
CAT_TESTS = "tests"
CAT_SCRIPTS = "scripts"
CAT_AUX_PYTHON = "aux_python"
CAT_EXTENSION = "extension"
CAT_MANIFEST = "manifest"
CAT_DEPS_LOCK = "deps_lock"
CAT_GATE_CONFIG = "gate_config"
CAT_DOCS = "docs"
CAT_OTHER = "other"

CATEGORIES = (
    CAT_CI,
    CAT_PYTHON,
    CAT_WEB,
    CAT_TESTS,
    CAT_SCRIPTS,
    CAT_AUX_PYTHON,
    CAT_EXTENSION,
    CAT_MANIFEST,
    CAT_DEPS_LOCK,
    CAT_GATE_CONFIG,
    CAT_DOCS,
    CAT_OTHER,
)

#: Categories that are INERT for every Python job (lint, type-check, the unit
#: matrix, tui-e2e). `docs` is the inert set proper: a markdown file under
#: `docs/` or a root-level `README.md` is read by no gate — `flake8 .` does
#: walk the tree, which is why a `docs/**/*.py` file is deliberately NOT in
#: this category (it falls through to `other`, i.e. live).
#:
#: `web` is `local_operator/mobile/web/**`: it ships as package data
#: (`[tool.setuptools.package-data]`) but no unit test reads it, its `dist/` is
#: gitignored so a source-only change alters no committed artifact, and
#: `mobile-web.yml` owns it behind its own paths filter. Web-only is therefore
#: inert for the Python jobs — pinned by `test_web_only_is_inert_for_the_python_jobs`.
PY_INERT_CATEGORIES = frozenset({CAT_DOCS, CAT_WEB})

#: The two test files `filesystem-boundaries-windows` exists to run under
#: native Windows semantics (`ci.yml`'s `Verify native filesystem boundaries`
#: step). It is the only place they run on Windows, so a diff confined to
#: `tests/**` must NOT blanket-skip the job: the `windows` flag is keyed on the
#: paths the job actually reads, not on "is this under tests/".
WINDOWS_TEST_PATHS = frozenset(
    {
        "tests/unit/test_agent_import_boundary.py",
        "tests/unit/server/test_edit_workspace_boundary.py",
    }
)

#: The `conftest.py` chain those two files load. A change to one of them can
#: change the job's behaviour without touching either test file, so they are
#: `windows` inputs for the same reason the files are.
WINDOWS_CONFTEST_PATHS = frozenset(
    {
        "conftest.py",
        "tests/conftest.py",
        "tests/unit/server/conftest.py",
    }
)

#: What the job's pytest run imports THROUGH that chain, which the path list
#: above does not cover. `conftest.py` does `from tests import
#: shard_stall_watchdog` at module scope, and pytest imports the `__init__.py`
#: markers for both protected modules — so a change to any of these can change
#: the Windows job's behaviour without touching a file it names. Narrow, since
#: Linux runs `tests/shard_stall_watchdog.py` too, but it is the same "flag
#: narrower than the guard's real input" pattern this classifier exists to
#: avoid, and the fix is a name in a set.
WINDOWS_LOADED_PATHS = frozenset(
    {
        "tests/shard_stall_watchdog.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/unit/server/__init__.py",
    }
)

#: `context-budget` builds the real tool surface through these two scripts, so
#: a change to either is a change to what the job measures.
BUDGET_SCRIPTS = frozenset(
    {
        "scripts/bench_context_budget.py",
        "scripts/real_tool_surface.py",
    }
)

#: `cli-sanity` EXECUTES this script (`ci.yml`: `python
#: scripts/check_streaming_contract.py run.jsonl`), and nothing else in the repo
#: imports it — so without naming it here a scripts-only diff skips the job's
#: only CI surface: the diff that changes a guard is the diff that skips it.
#:
#: A named input set rather than adding `scripts/**` to the `cli` predicate, and
#: the choice is deliberate: widening to all of `scripts/**` would run both
#: live-LLM jobs (real spend) plus the audit on every scripts-only diff, which
#: the per-class cost analysis rates as safe to skip. This is the same shape
#: `BUDGET_SCRIPTS` already uses for `context-budget`.
CLI_SANITY_SCRIPTS = frozenset({"scripts/check_streaming_contract.py"})

GATE_CONFIG_PATHS = frozenset({"Makefile", ".flake8", "setup.cfg", "tox.ini"})
MANIFEST_PATHS = frozenset({"pyproject.toml"})
DEPS_LOCK_PATHS = frozenset({"uv.lock"})

#: A root-level markdown file (no directory component). `^[^/]+\.md$` is the
#: whole rule: `docs/**/*.md` is handled by the `docs/` branch below, and a
#: markdown file inside any other directory is NOT inert (it lives next to code
#: the gates read, and no such directory has an inertness argument recorded).
ROOT_MARKDOWN_RE = re.compile(r"^[^/]+\.md$")

#: `[+-]version = ` as `ci.yml`'s `version-bump-guard` spells it
#: (`grep -E '^[+-]version[[:space:]]*='`). Reusing that shape rather than
#: deriving a new one keeps the classifier's idea of "the version line" and the
#: guard's idea of it identical; if the guard's syntax changes, this must
#: change with it.
VERSION_LINE_RE = re.compile(r"^version[ \t]*=")


def _norm(path: str) -> str:
    """Repo-relative POSIX form of a path from git.

    `git diff --name-status` reports forward slashes on every platform, but the
    Windows runner can hand us a backslash through a caller; normalising here
    means the prefix rules cannot silently miss on one platform.
    """
    p = path.replace("\\", "/").strip()
    # Only an exact leading `./` is stripped. `lstrip("./")` would eat the dot
    # of `.github/workflows/ci.yml` and drop the whole diff to `other`.
    return p[2:] if p.startswith("./") else p


def category_of(path: str) -> str:
    """Classify one repo-relative path. First match wins (spec §2.1).

    The order is the contract, not an implementation detail: `docs/**/*.py`
    must reach the `other` (live) branch rather than the inert `docs` one, and
    a root `*.md` must be inert while `local_operator/prompts_md/x.md` is live
    Python package data.
    """
    p = _norm(path)
    if not p:
        return CAT_OTHER
    if p == ".github" or p.startswith(".github/"):
        # D14: where the gating itself lives. `flags_for` overrides every flag
        # to `true` for this category rather than deriving it, because "which
        # jobs run" must not be decided by a file that a PR could edit to
        # decide it.
        return CAT_CI
    if p == "local_operator" or p.startswith("local_operator/"):
        if p.startswith("local_operator/mobile/web/"):
            return CAT_WEB
        # ANY extension: `prompts_md/*.md`, `guides/*/*.md`,
        # `agent_seeds/*.md` and `tui/*.tcss` are shipped package data that the
        # Python code reads at runtime, so an extension list here would be a
        # hole.
        return CAT_PYTHON
    if p == "tests" or p.startswith("tests/"):
        return CAT_TESTS
    if p.startswith("scripts/"):
        return CAT_SCRIPTS
    if p.startswith(("benchmarks/", "bench/", "examples/")):
        return CAT_AUX_PYTHON
    if p.startswith("extension/"):
        return CAT_EXTENSION
    if p in MANIFEST_PATHS:
        return CAT_MANIFEST
    if p in DEPS_LOCK_PATHS:
        return CAT_DEPS_LOCK
    if p in GATE_CONFIG_PATHS:
        return CAT_GATE_CONFIG
    if p.startswith("docs/"):
        return CAT_DOCS if not p.endswith((".py", ".pyi")) else CAT_OTHER
    if ROOT_MARKDOWN_RE.match(p):
        return CAT_DOCS
    return CAT_OTHER


# --------------------------------------------------------------------------
# Flags, the jobs they gate, and the local commands those jobs run
# --------------------------------------------------------------------------

#: The outputs the `changes` job exports. Order is the order they are written
#: to `$GITHUB_OUTPUT` and the order they are listed in the summary.
FLAGS = (
    "lint",
    "types",
    "budget",
    "unit",
    "tui",
    "windows",
    "audit",
    "cli",
    "server",
)

#: job id in `ci.yml` -> the flags that gate it. Every flag here must be one of
#: `FLAGS` and the job's `if:` must read each of them (asserted A1).
JOB_FLAGS: dict[str, tuple[str, ...]] = {
    "lint": ("lint",),
    "type-check": ("types",),
    "context-budget": ("budget",),
    "filesystem-boundaries-windows": ("windows",),
    "test": ("unit",),
    "pip-audit": ("audit",),
    "tui-e2e": ("tui",),
    "cli-sanity": ("cli",),
    "server-sanity": ("server",),
}

#: Jobs that take no scope flag, each with the reason. Everything in `ci.yml`
#: is either here or in `JOB_FLAGS`; a job in neither is a job that pays full
#: price unnoticed (asserted A1).
UNGATED_JOBS: dict[str, str] = {
    "changes": (
        "always runs: it IS the classifier, and every gated job `needs:` it, so "
        "an event condition here would collaterally skip the whole workflow on "
        "push to main (D7)"
    ),
    "version-bump-guard": (
        "gated on the EVENT (pull_request), not on a scope flag: it must not "
        "become able to be skipped by the classifier whose output a PR could "
        "influence, and one check name per fact is the honest checks list (D12)"
    ),
    "coverage-report": (
        "no `if:` at all, `needs: test` only: GitHub's 'a skipped need skips the "
        "dependent' is the wanted behaviour — a docs-only PR must never combine "
        "zero shard artifacts, and an `always()` added for symmetry would go red "
        "on an empty combine (D6)"
    ),
}

#: Dependency pairs that may use the PERMISSIVE clause
#: (`needs.D.result == 'success' || needs.D.result == 'skipped'`) instead of the
#: strict `needs.D.result == 'success'`. The invariant the test enforces (A4):
#: for each `(job, dependency)` pair, either the module guarantees
#: `types(job) => types(dependency)` (then the strict clause is required) or the
#: pair is listed here WITH A NON-EMPTY REASON (then the permissive clause is
#: required).
#:
#: Local-operator needs NONE, and that is a property of the predicates rather
#: than a hope: `unit == lint == types`, `tui == unit`, and `cli == server ==
#: audit`, so `cli ⊆ {lint, types, unit, audit}` holds by construction and every
#: dependency clause in ci.yml can be the strict form. The equality `cli ==
#: audit` is what keeps the two live-LLM sanity jobs honest — with a narrow
#: `audit` they would be gated on a job that a `local_operator/**`-only diff
#: skips, i.e. silently disarmed on the most common PR in the repo. The UI repo
#: needs one permissive pair (`npx-sanity-check` → `audit`), which is where this
#: allowlist earns its keep.
PERMISSIVE_DEPS: dict[tuple[str, str], str] = {}

#: job id -> the commands a developer runs for that job, in CI order. These are
#: the LOCAL, safe spellings: `python -m` for a module the interpreter owns
#: (never `.venv/bin/flake8`, whose install-time shebang exits 126 after its
#: owning worktree is deleted, and whose failure a pipeline swallows — the
#: #423 mechanism) and `uvx` for the tool versions CI pins but the dev extra
#: deliberately does not carry (black, isort). `_invoked_tool` maps each back
#: to the tool the job actually runs, which is what the drift assertion
#: compares against ci.yml.
#: The wrapper that bounds a gate and reaps its process group (see
#: scripts/run_bounded.py). It is invoked through the interpreter, so the #423
#: shebang rule holds for it too; `_invoked_tool` sees through it so the drift
#: assertion still compares the TOOL the job runs against ci.yml.
BOUNDED_WRAPPER_NAME = "run_bounded.py"

#: The local bound for a wrapped gate, in seconds. **Deliberately NOT ci.yml's
#: `timeout-minutes: 15`**: a whole-tree `pyright` measures 508 s on a quiet host
#: and 1170 s under load on this fleet, and CI's 15 minutes also cover checkout,
#: dependency install and that job's protocol-sync step. A bound that fires on a
#: legitimately slow host reds a gate CI would pass, which is a worse failure than
#: a slow gate, so the local bound is twice CI's provision and a fired bound
#: prints its remedy (see `run_jobs`). The `type-check` Makefile target defaults
#: to the same number and a test asserts the two agree.
BOUNDED_GATE_TIMEOUT = 1800

JOB_COMMANDS: dict[str, tuple[str, ...]] = {
    "lint": (
        ".venv/bin/python -m flake8 .",
        "uvx --from black==26.1.0 black --check .",
        "uvx isort==5.13.2 --check .",
    ),
    "type-check": (
        # Bounded and group-reaped. `pyright` is a Python wrapper around an
        # npm/node analyzer that runs as a SEPARATE process, so a bound that ends
        # only the leader can leave the analyzer behind — measured on the fleet as
        # node children re-parented to `ppid 1` holding 2.28 GB and 1.50 GB, one
        # alive 81 minutes after its parent died. `timeout(1)` is not the villain
        # here (on this host it signals the group); what it cannot cover is a
        # leader that exits while a descendant lives on, which the wrapper sweeps.
        # The bound is `BOUNDED_GATE_TIMEOUT`, above: generous on purpose, because
        # timing out a legitimately slow gate is worse than waiting for it.
        f".venv/bin/python scripts/{BOUNDED_WRAPPER_NAME} --timeout {BOUNDED_GATE_TIMEOUT} -- "
        ".venv/bin/python -m pyright --pythonpath .venv/bin/python .",
        # The protocol-sync check is a *step of this job* in ci.yml, not a job
        # of its own: a Python-only protocol edit must fail even when the
        # paths-gated extension workflow does not run.
        ".venv/bin/python -m local_operator.browser_bridge.gen_ts --check",
    ),
    "context-budget": (".venv/bin/python scripts/bench_context_budget.py --verbose",),
    "test": (
        # The colour-capable environment is part of the documented gate: without
        # it the Textual pilot tests run against a NO_COLOR terminal and fail
        # for a reason CI never sees (AGENTS.md, "TUI tests need a
        # colour-capable terminal").
        "env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/unit -q",
    ),
    "tui-e2e": (
        "env -u NO_COLOR TERM=xterm-256color " ".venv/bin/python -m pytest tests/e2e -m e2e -n0 -q",
    ),
}

#: Jobs in `JOB_FLAGS` that deliberately have no local command, each with the
#: reason it cannot be one. `--run` prints these so a skipped job is legible
#: rather than silently absent.
LOCAL_EXCLUSIONS: dict[str, str] = {
    "filesystem-boundaries-windows": (
        "native Windows drive/junction semantics; the job exists precisely "
        "because a POSIX test with os.name mocked is not evidence (ci.yml "
        "'Canonical containment must use native Windows drive/junction "
        "semantics')"
    ),
    "cli-sanity": (
        "live-LLM job: needs OPENROUTER_API_KEY and spends real tokens, so it "
        "is never part of a default local run"
    ),
    "server-sanity": ("live-LLM job: same secrets and spend as cli-sanity"),
    "pip-audit": (
        "no faithful local spelling. CI is `pypa/gh-action-pip-audit@v1.1.0`, "
        "which does `pip install .` and then runs `pip-audit … --desc "
        "--vulnerability-service pypi <dir>` in a HERMETIC venv it builds "
        "itself; the local equivalent (`.venv/bin/python -m pip_audit .`) "
        "cannot build that venv on this host — `ve.create(ve_dir)` dies with "
        "`ensurepip … <Signals.SIGABRT: 6>`, rc=1, reproducibly and on a clean "
        "tree — so a local 'audit failure' would be a false red for a check "
        "the job may well pass. Excluded rather than left as a gate that cries "
        "wolf; the audit stays a required CI check and this classifier cannot "
        "skip it on any dependency-touching diff."
    ),
}

#: Human-readable predicate per flag, used in `--summary` so the reason a job
#: ran (or did not) is written down where a reviewer reads it.
FLAG_REASONS: dict[str, str] = {
    "lint": "at least one changed path outside the inert set (docs/**, web-only)",
    "types": "at least one changed path outside the inert set (docs/**, web-only)",
    "budget": "a path that can change the assembled start-of-session context",
    "unit": "at least one changed path outside the inert set (docs/**, web-only)",
    "tui": "same predicate as `unit`, deliberately (scripts/** is a tui input)",
    "windows": "a path the Windows job reads, or a manifest/CI/gate config",
    "audit": "a Python, manifest, lockfile, CI or unrecognised path",
    "cli": (
        "a Python, manifest, lockfile, CI or unrecognised path, or the "
        "streaming-contract script cli-sanity executes"
    ),
    "server": "same predicate as `cli`",
}


#: Ungated jobs that still inherit a SKIP from a job they `need:` — GitHub's
#: default ("a skipped need skips the dependent"). `coverage-report` is the case
#: D6 pins: it combines the shard artifacts, so on a diff where `test` was
#: deliberately skipped it must not run a combine over zero artifacts. `changes`
#: and `version-bump-guard` depend on nothing and always run.
#:
#: Modelled here rather than left implicit because the `--summary` job list is
#: the only place the ACTUAL per-run job set is written down (D13); a plan that
#: listed a job as running when GitHub will skip it would be a false report.
INHERITS_SKIP: dict[str, tuple[str, ...]] = {"coverage-report": ("test",)}


class ScopeError(RuntimeError):
    """A flag could not be rendered as `true`/`false`.

    Raised rather than degraded: an EMPTY output read by `!= 'false'` still runs
    the job, but an empty output in the Actions UI is indistinguishable from a
    classifier that never ran — and the one outcome this scheme must never
    produce is a green PR that silently ran nothing.
    """


def _flag_value(flag: str, flags: Mapping[str, bool]) -> str:
    """Render one flag for `$GITHUB_OUTPUT`; RAISE on anything but true/false."""
    try:
        value = flags[flag]
    except KeyError as exc:  # pragma: no cover - guarded by FLAGS/JOB_FLAGS tests
        raise ScopeError(f"{flag!r} is not a known flag; refusing to write it") from exc
    if value is True:
        return "true"
    if value is False:
        return "false"
    raise ScopeError(
        f"flag {flag!r} rendered as {value!r}; only true/false may reach "
        "$GITHUB_OUTPUT (an unknown/empty value is not a scope decision)"
    )


# --------------------------------------------------------------------------
# release_bump: a version-only manifest edit
# --------------------------------------------------------------------------


def _diff_changed_lines(diff_text: str) -> tuple[list[str], list[str]]:
    """`+`/`-` body lines of a unified diff, with the file headers removed."""
    removed: list[str] = []
    added: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return removed, added


def is_release_bump(paths: Sequence[str], diff: str | None) -> bool:
    """True when the change is nothing but a version bump in `pyproject.toml`.

    `diff` is the unified diff of `pyproject.toml` ALONE (as
    `git diff <base> <target> -- pyproject.toml` produces). Passing a
    whole-repo diff is safe — the other files' `+`/`-` lines make this return
    False — it simply will not detect a bump.

    The test is the LINE PAIR, not the file: a dependency, metadata or
    `[project.scripts]` edit in the same file must keep the whole matrix and
    the audit, because it changes what gets installed. Matching on the filename
    alone is how a dependency bump would silently drop the audit and every
    shard from the PR that changed the install.
    """
    if set(_norm(p) for p in paths) != set(MANIFEST_PATHS):
        return False
    if not diff:
        return False
    removed, added = _diff_changed_lines(diff)
    if not removed or not added:
        return False
    return all(VERSION_LINE_RE.match(line) for line in removed + added)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def flags_for(
    categories: Iterable[str],
    paths: Iterable[str],
    release_bump: bool,
) -> dict[str, bool]:
    """The flag vector for a set of categories. Split out so the tests can pin
    a predicate without constructing a diff."""
    cats = set(categories)
    path_set = set(paths)

    if CAT_CI in cats:
        return _all(True)
    if release_bump:
        return _all(False)
    if CAT_OTHER in cats:
        return _all(True)

    live = any(cat not in PY_INERT_CATEGORIES for cat in cats)
    unit = live
    # `cli == server == audit` deliberately (see PERMISSIVE_DEPS): the two
    # live-LLM jobs `need:` pip-audit, so an `audit` narrower than `cli` would
    # let a `local_operator/**`-only diff skip the audit and therefore skip the
    # sanity jobs that depend on it. `CLI_SANITY_SCRIPTS` is shared by all three
    # so the equality survives the named-script widening (R2), and `cli` still
    # implies `lint`/`types`/`unit` because that script is not inert.
    cli = bool(cats & {CAT_PYTHON, CAT_MANIFEST, CAT_DEPS_LOCK, CAT_CI, CAT_OTHER}) or bool(
        path_set & CLI_SANITY_SCRIPTS
    )
    audit = cli
    windows = (
        CAT_PYTHON in cats
        or bool(cats & {CAT_CI, CAT_MANIFEST, CAT_GATE_CONFIG, CAT_OTHER})
        or bool(path_set & (WINDOWS_TEST_PATHS | WINDOWS_CONFTEST_PATHS | WINDOWS_LOADED_PATHS))
    )
    budget = bool(cats & {CAT_PYTHON, CAT_MANIFEST, CAT_DEPS_LOCK}) or bool(
        path_set & BUDGET_SCRIPTS
    )
    return {
        "lint": live,
        "types": live,
        "budget": budget,
        "unit": unit,
        "tui": unit,
        "windows": windows,
        "audit": audit,
        "cli": cli,
        "server": cli,
    }


def is_permissive_dependency(job: str, dependency: str) -> bool:
    """Whether `job` may treat a SKIPPED `dependency` as acceptable (A4).

    For every other `(job, dependency)` pair the module must guarantee
    `types(job) => types(dependency)`, so a skipped dependency means the
    dependency was not needed in the first place. A pair where that implication
    does NOT hold must be declared in `PERMISSIVE_DEPS` with a reason, and its
    workflow clause must accept `skipped` as well as `success` — otherwise a
    deliberately skipped cheap gate silently skips the expensive job it guards.
    """
    return (job, dependency) in PERMISSIVE_DEPS


def _all(value: bool) -> dict[str, bool]:
    return {flag: value for flag in FLAGS}


def classify(paths: Sequence[str], diff: str | None = None) -> dict[str, bool]:
    """Flag vector for a change set.

    `paths` are repo-relative changed paths (both sides of a rename — a rename
    OUT of `docs/` into `local_operator/` must set the Python flags, so
    classifying only the new path is a defect). `diff` is the `pyproject.toml`
    diff, used only to refine the `manifest` category into `release_bump`.

    The returned mapping also carries `release_bump`, which is not a CI output
    (the `version-bump-guard` job is ungated) but is the classification the
    summary prints, and is what "a bump PR runs only the guard" refers to.
    """
    norm = [_norm(p) for p in paths]
    cats = [category_of(p) for p in norm]
    release_bump = is_release_bump(norm, diff)
    flags = flags_for(cats, norm, release_bump)
    flags["release_bump"] = release_bump
    return flags


def categories_of(paths: Sequence[str]) -> list[tuple[str, str]]:
    """`(path, category)` pairs, in the order given, for the summary."""
    return [(p, category_of(p)) for p in paths]


# --------------------------------------------------------------------------
# Git plumbing
# --------------------------------------------------------------------------


def _git(args: Sequence[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def resolve_base(rev: str, cwd: Path | None = None) -> str | None:
    """The merge base of `rev` and HEAD, falling back to `rev` itself.

    `--since origin/main` is how a developer thinks about it; the merge base is
    what makes the answer "this branch's own work". `--since <sha>` (an already
    resolved base) round-trips: the merge base of an ancestor and HEAD is that
    ancestor.

    `cwd` is the resolved repository root, like every other git call in this
    module. It used to be omitted, which made this the one function that read the
    PROCESS cwd — so a perfectly good `--root` plus a valid base still failed to
    resolve whenever the process happened to be started outside a repository,
    and the fail-open branch fired even though the caller had named the repo.
    """
    rc, out, _ = _git(["merge-base", rev, "HEAD"], cwd=cwd)
    if rc == 0 and out.strip():
        return out.strip().splitlines()[0].strip()
    rc, out, _ = _git(["rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=cwd)
    if rc == 0 and out.strip():
        return out.strip()
    return None


def _parse_name_status(stdout: str) -> list[str]:
    """Changed paths from `git diff --name-status`, BOTH sides of a rename."""
    paths: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") or status.startswith("C"):
            # R100<TAB>old<TAB>new — the OLD path is what moved out of the
            # inert set, so dropping it would let `docs/a.md -> local_operator/a.py`
            # classify as a docs-only diff.
            paths.extend(_norm(p) for p in fields[1:3] if p)
        elif len(fields) > 1:
            paths.append(_norm(fields[1]))
    return paths


def collect_paths(base: str, local: bool, cwd: Path | None = None) -> list[str] | None:
    """Changed paths between `base` and HEAD (CI) or the working tree (local).

    Returns None when git itself failed, which the caller turns into the
    fail-open path rather than into an empty (and therefore all-skipping)
    change set.
    """
    target = [] if local else ["HEAD"]
    rc, out, _ = _git(["diff", "--name-status", "-M", base, *target], cwd=cwd)
    if rc != 0:
        return None
    paths = _parse_name_status(out)
    if local:
        # Untracked files are changes a local gate must see (a new test file
        # nobody staged still has to be linted) and are invisible to `git diff`.
        rc, out, _ = _git(["ls-files", "--others", "--exclude-standard"], cwd=cwd)
        if rc != 0:
            return None
        paths.extend(_norm(p) for p in out.splitlines() if p.strip())
    # De-duplicate but keep first-seen order so the summary reads stably.
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def manifest_diff(base: str, local: bool, cwd: Path | None = None) -> str | None:
    """The `pyproject.toml` diff, or None when git failed (fail open)."""
    target = [] if local else ["HEAD"]
    rc, out, _ = _git(["diff", base, *target, "--", "pyproject.toml"], cwd=cwd)
    if rc != 0:
        return None
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _warning(message: str, title: str) -> None:
    """An Actions annotation, plus the same text on stderr for a local run.

    Printing it is the point: an unresolvable base must be VISIBLE rather than
    a silent full run, or the next person reads a full job set as "the
    classifier decided everything was affected".
    """
    print(f"::warning title={title}::{message}")
    print(f"warning: {message}", file=sys.stderr)


def job_plan(flags: Mapping[str, bool]) -> dict[str, str]:
    """job id -> 'run' | 'skip' for the whole workflow, in ci.yml order."""
    plan: dict[str, str] = {job: "run" for job in UNGATED_JOBS}
    for job, job_flags in JOB_FLAGS.items():
        plan[job] = "run" if all(flags[f] for f in job_flags) else "skip"
    for job, dependencies in INHERITS_SKIP.items():
        if any(plan.get(dependency) == "skip" for dependency in dependencies):
            plan[job] = "skip"
    return plan


def summary_lines(
    *,
    event: str,
    base: str | None,
    base_label: str,
    paths: Sequence[str],
    flags: Mapping[str, bool],
    note: str | None,
) -> list[str]:
    """The `$GITHUB_STEP_SUMMARY` body (D13)."""
    lines = [
        "## Change classification",
        "",
        f"- event: `{event}`",
        f"- diff base: `{base_label}`" + (f" (`{base}`)" if base else ""),
    ]
    if note:
        lines.append(f"- **{note}**")
    lines.extend(["", f"### Changed paths ({len(paths)})", ""])
    if paths:
        for path in paths:
            lines.append(f"- `{path}` → `{category_of(path)}`")
    else:
        lines.append("- (none)")
    lines.extend(["", "### Flags", ""])
    for flag in FLAGS:
        value = "true" if flags[flag] else "false"
        lines.append(f"- `{flag}` = **{value}** — {FLAG_REASONS[flag]}")
    release_bump = bool(flags.get("release_bump"))
    lines.append(
        f"- `release_bump` = **{'true' if release_bump else 'false'}** — "
        + (
            "the diff is nothing but the `version =` line pair in "
            "pyproject.toml, so only `version-bump-guard` has anything to check"
            if release_bump
            else "this is not a version-only `pyproject.toml` diff, so the "
            "manifest keeps every gate a manifest change earns"
        )
    )
    lines.extend(["", "### Jobs", ""])
    for job, verdict in job_plan(flags).items():
        if job in UNGATED_JOBS:
            why = UNGATED_JOBS[job]
            if verdict == "skip":
                why = "skipped because a job it `need:`s was skipped: " + ", ".join(
                    INHERITS_SKIP.get(job, ())
                )
        else:
            why = ", ".join(f"{f}={flags[f]}" for f in JOB_FLAGS[job])
        lines.append(f"- `{job}`: **{verdict}** — {why}")
    lines.append("")
    lines.append(
        "A skipped job is a claim, not a pass: read the flag row above before "
        "treating this run as evidence."
    )
    return lines


# --------------------------------------------------------------------------
# Local execution
# --------------------------------------------------------------------------


def _invoked_tool(command: str) -> str:
    """The tool a local JOB_COMMANDS entry actually runs.

    Strips the LOCAL-safe prefix (`env -u … VAR=…`, `python -m`, `uvx [--from
    x==1]`) down to the tool name, which is what the drift assertion compares
    against the job's `run:` blocks in ci.yml. The prefix cannot be compared
    verbatim: the local spelling is deliberately different from CI's (that is
    the #423 rule), so the spec's "first two tokens" form would either be
    unverifiable or would force the unsafe bare console script back in.
    """
    tokens = shlex.split(command)
    i = 0
    if tokens[:1] == ["env"]:
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token in ("-i", "-0", "--ignore-environment", "--null"):
                i += 1
            elif token in ("-u", "--unset"):
                i += 2
            elif token.startswith("-") or "=" in token:
                # `-u NO_COLOR` / `TERM=xterm-256color`; neither is the tool.
                i += 1
            else:
                break
    if i >= len(tokens):
        return ""
    head = tokens[i]
    tail = tokens[i + 1 :]
    if Path(head).name.startswith("python"):
        if tail[:1] == ["-m"] and len(tail) > 1:
            return tail[1]
        if tail and Path(tail[0]).name == BOUNDED_WRAPPER_NAME:
            # `.venv/bin/python scripts/run_bounded.py --timeout N -- <gate>`
            # bounds and reaps a gate; the tool it reports is the WRAPPED one.
            # Reporting the wrapper would compare its name against ci.yml and
            # fail the drift assertion whenever a gate is correctly bounded.
            inner = _unwrap_bounded(tail[1:])
            return _invoked_tool(shlex.join(inner)) if inner else ""
        return Path(tail[0]).name if tail else "python"
    if head == "uvx":
        if tail[:1] == ["--from"]:
            tail = tail[2:]
        if not tail:
            return "uvx"
        return tail[0].split("==")[0]
    return Path(head).name


def _unwrap_bounded(tokens: Sequence[str]) -> list[str]:
    """The command inside `run_bounded.py --timeout N -- <command>`.

    Shared by `_invoked_tool` (which the drift assertion reads) and the tests,
    so the wrapper's argument shape has one reader rather than two.
    """
    rest = list(tokens)
    if "--" in rest:
        return rest[rest.index("--") + 1 :]
    # No separator: the wrapper was invoked without a command, so there is no tool
    # to report. Counting flags instead sliced into the command's own arguments
    # whenever the PAYLOAD carried a `--timeout`/`--grace` of its own — flags
    # before the separator happened to come out right — which is how a directory
    # (`tests`) once got compared against ci.yml.
    for flag in ("--timeout", "--grace"):
        if flag in rest:
            rest = rest[rest.index(flag) + 2 :]
    return rest


def _command_bound(command: str) -> str | None:
    """The bound a WRAPPED command carries, or None when rc=124 is not that bound.

    Reported from the COMMAND rather than from `BOUNDED_GATE_TIMEOUT` (the Makefile
    carries its own), and `None` unless the command really does go through the
    wrapper with a `--timeout`: rc=124 out of anything else is not this bound, and
    a message that claims it is sends a developer to turn a knob that never fired
    (review round 2, N3).
    """
    tokens = shlex.split(command)
    wrapped = any(Path(token).name == BOUNDED_WRAPPER_NAME for token in tokens)
    if wrapped and "--timeout" in tokens:
        return f"--timeout {tokens[tokens.index('--timeout') + 1]}s"
    return None


def run_jobs(jobs: Sequence[str], root: Path) -> int:
    """Run each selected job's local commands in order; return a shell status."""
    failures: list[str] = []
    for job in jobs:
        for command in JOB_COMMANDS[job]:
            print(f"\n=== {job}: {command}", flush=True)
            proc = subprocess.run(command, shell=True, cwd=str(root), check=False)
            if proc.returncode != 0:
                print(f"!!! {job} failed (rc={proc.returncode}): {command}")
                bound = _command_bound(command)
                if proc.returncode == 124 and bound is not None:
                    # This is the BOUND firing, not the gate failing, and the
                    # difference decides what a developer does next: on a loaded
                    # host a whole-tree pyright can legitimately exceed the bound,
                    # and re-running (or raising it) is the fix — never reading it
                    # as a red.
                    print(
                        f"    rc=124 is the BOUND ({bound}) firing, not the gate "
                        "failing: the gate was still working. Re-run it; if it fires again, "
                        "raise the bound (`make type-check BOUND_TIMEOUT=<seconds>`, or "
                        f"BOUNDED_GATE_TIMEOUT in scripts/{Path(__file__).name} for "
                        "`make check-changed`)."
                    )
                failures.append(f"{job}: {command} (rc={proc.returncode})")
    print()
    if failures:
        print(f"{len(failures)} command(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all selected gates passed")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def default_root(module_file: Path) -> Path:
    """The repository to classify, resolved WITHOUT trusting where this file is.

    `Path(__file__).resolve().parent.parent` is the tempting default and it is
    wrong in exactly the place that matters: the `changes` step copies this
    module to `$RUNNER_TEMP` and runs the copy, so the file's grandparent is
    `/home/runner/work` — one level ABOVE the checkout and inside no repository.
    `root` is also the `cwd` of every git call — `resolve_base` included, so an
    explicit `--root` decides base resolution as well as the diff. That default
    made `git diff` exit 128 on a runner, and the fail-open branch set every flag
    true on every pull request: the classifier was right and never engaged, which
    is the failure this whole change exists to remove.

    So: the git top level of the INVOCATION directory first (the step runs with
    the workspace as its cwd), then the tree this file was shipped inside, then
    the invocation directory. All three branches are load-bearing: with the cwd
    inside no repository, the middle branch is what turns a copy of this file
    shipped inside a checkout back into a working run. The CI step also passes
    `--root "$GITHUB_WORKSPACE"` explicitly, deliberately belt-and-braces: that
    flag is what makes the workflow independent of this function's cleverness.
    """
    rc, out, _ = _git(["rev-parse", "--show-toplevel"])
    if rc == 0 and out.strip():
        return Path(out.strip()).resolve()
    candidate = module_file.resolve().parent.parent
    rc, out, _ = _git(["rev-parse", "--show-toplevel"], cwd=candidate)
    if rc == 0 and out.strip():
        return Path(out.strip()).resolve()
    return Path.cwd().resolve()


def _github_event(args: argparse.Namespace) -> str | None:
    """The Actions event name, or None when this is a LOCAL run.

    The distinction is load-bearing rather than cosmetic. "Every flag true" is
    the answer for a real CI EVENT that is not a pull request (`push` to `main`
    must stay an unconditional full run — D5). A local `make check-changed` has
    NO event, and must classify: keying the short-circuit on "the resolved event
    is not `pull_request`" with a `local` default would make the local command a
    slower spelling of "run everything", which is the defect this module exists
    to remove.
    """
    return args.event if args.event is not None else os.environ.get("GITHUB_EVENT_NAME")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ci_scope.py",
        description="Classify a diff into the CI jobs it can affect.",
    )
    parser.add_argument(
        "--event",
        default=None,
        help=(
            "the Actions event name ($GITHUB_EVENT_NAME). Anything other than "
            "`pull_request` runs everything: `main` is the safety net for the "
            "narrowed PR matrix, and it is the one place a misclassification is "
            "ever contradicted (D5)."
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help="exact diff base commit (CI: HEAD^1). Diffed against HEAD only.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "base ref for a LOCAL run (e.g. origin/main): resolved to its merge "
            "base with HEAD, and the diff includes the index, the working tree "
            "and untracked files."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "skip classification: every flag true. The fallback when no "
            "classifier exists at the base revision."
        ),
    )
    parser.add_argument(
        "--github-output",
        default=None,
        help="write `flag=value` lines here ($GITHUB_OUTPUT).",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="append the human-readable report here ($GITHUB_STEP_SUMMARY).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the local commands of every selected job (the `make check-changed` body).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "retained for existing callers; the report is now always printed to "
            "stdout, so this flag changes nothing."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help="repo root to inspect/run in (default: this file's parent's parent).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve() if args.root else default_root(Path(__file__))
    ci_event = _github_event(args)
    event = ci_event or "local"

    note: str | None = None
    base: str | None = None
    base_label = "(none)"
    paths: list[str] = []
    diff: str | None = None

    if args.all:
        flags = _all(True)
        flags["release_bump"] = False
        note = "`--all`: every job runs (classification not attempted)"
    elif ci_event is not None and ci_event != "pull_request":
        flags = _all(True)
        flags["release_bump"] = False
        base_label = event
        note = (
            f"event `{event}` is not a `pull_request`: every job runs. `main` is "
            "the safety net for the narrowed PR matrix, so its run must stay "
            "unconditional (D5)."
        )
    else:
        rev = args.base or args.since
        if not rev:
            flags = _all(True)
            flags["release_bump"] = False
            note = "no `--base`/`--since` given: every job runs"
            _warning(
                "no diff base was given, so this diff cannot be classified. "
                "Running every job rather than guessing.",
                "Change classification unavailable",
            )
        else:
            base = resolve_base(rev, cwd=root)
            base_label = rev
            if base is None:
                flags = _all(True)
                flags["release_bump"] = False
                note = f"base `{rev}` could not be resolved: every job runs"
                _warning(
                    f"the diff base `{rev}` could not be resolved, so this diff "
                    "cannot be classified. Running every job rather than guessing.",
                    "Change classification unavailable",
                )
            else:
                local = args.base is None
                collected = collect_paths(base, local, cwd=root)
                diff = manifest_diff(base, local, cwd=root) if collected is not None else None
                if collected is None or diff is None:
                    flags = _all(True)
                    flags["release_bump"] = False
                    note = "`git diff` failed: every job runs"
                    _warning(
                        "`git diff` failed, so this diff cannot be classified. "
                        "Running every job rather than guessing.",
                        "Change classification unavailable",
                    )
                else:
                    paths = collected
                    flags = classify(paths, diff)
                    if CAT_OTHER in {category_of(p) for p in paths}:
                        _warning(
                            "this diff contains path(s) the classifier does not "
                            "recognise, so every job runs: "
                            + ", ".join(p for p in paths if category_of(p) == CAT_OTHER),
                            "Unrecognised path",
                        )

    lines = summary_lines(
        event=event,
        base=base,
        base_label=base_label,
        paths=paths,
        flags=flags,
        note=note,
    )

    text = "\n".join(lines)
    # The report goes to the LOG unconditionally and to the step summary as well
    # when `--summary` is given. D13 asks for both: the summary is what a
    # reviewer is told to read, and the log is where anyone debugging a run
    # actually looks. `--verbose` is retained for existing callers and is now
    # redundant — it used to be the only way to see the report locally.
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            for flag in FLAGS:
                # The value is rendered by a function that RAISES rather than
                # emitting '' — an unwritten output leaves `!= 'false'` true and
                # runs the job, but a *malformed* one is not a decision at all.
                handle.write(f"{flag}={_flag_value(flag, flags)}\n")

    if args.run:
        plan = job_plan(flags)
        selected = [
            job for job, verdict in plan.items() if verdict == "run" and job in JOB_COMMANDS
        ]
        for job, reason in sorted(LOCAL_EXCLUSIONS.items()):
            print(f"skipped locally: {job} — {reason}")
        return run_jobs(selected, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
