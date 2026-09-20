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
import ast
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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

#: What the two `xplat-probe-*` jobs read from `scripts/`. The battery
#: (`scripts/xplat_probe.py`) is stdlib-only and self-driving, so `scripts/**`
#: as a whole is deliberately NOT an input — the same choice `CLI_SANITY_SCRIPTS`
#: records, for the same reason: a `shard_tests.py` change cannot alter a single
#: probe, while widening to every script would put a Windows runner and a full
#: battery on every scripts-only diff.
#:
#: The last three are not helpers beside the probe, they are its own reads:
#: `--driver tui` imports `scripts.probe_isolation` and
#: `scripts.visual_capture` in the child that boots the app, so a change to
#: either changes what the `tui.boot` probe measures. `xplat_linux_matrix.sh` is
#: the rig the extra DISTROS (Mint, and any other image) are read through — the
#: jobs do not execute it, but it is the Linux leg's own tooling, so a change to
#: it belongs on the diff that also exercises the legs.
XPLAT_SCRIPT_INPUTS = frozenset(
    {
        "scripts/xplat_probe.py",
        "scripts/xplat_report.py",
        "scripts/xplat_linux_matrix.sh",
        "scripts/probe_isolation.py",
        "scripts/visual_capture.py",
    }
)

#: `scripts/xplat/**` is the container rig the extra Linux distros are built
#: from (`Dockerfile.probe` and its dockerignore). A prefix rather than a name
#: list: the rig grows files (a second image, a helper script) and a name list
#: would silently stop covering them.
XPLAT_SCRIPT_PREFIXES = ("scripts/xplat/",)

#: The probe's `tui.boot` step boots the REAL app against the fake session the
#: test suite owns — `--driver tui` does `from tests.unit.tui.test_app_pilot
#: import FakeSession, _factory` — so a diff confined to `tests/**` must NOT
#: blanket-skip these jobs. That is the same trap `WINDOWS_TEST_PATHS` records,
#: in the other direction (there the job runs the test; here it imports it).
#: The `__init__.py` markers are named because the dotted import needs the
#: package chain to exist, not because the probe reads them itself.
#:
#: `conftest.py` is an OVER-APPROXIMATION and is declared as one: the battery is
#: not a pytest run, so nothing imports it. It is listed because the test package
#: is a live input to this job and the pytest bootstrap is the one path a reader
#: would otherwise have to reason about case by case; the cost of the false
#: positive is one battery run, and the cost of the false negative is a leg that
#: never runs on the diff that broke it. Anything that can show a conftest change
#: cannot alter a probe reading may drop it.
XPLAT_TEST_INPUTS = frozenset(
    {
        "tests/unit/tui/test_app_pilot.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/unit/tui/__init__.py",
        "conftest.py",
    }
)

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
    "xplat",
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
    "xplat-probe-linux": ("xplat",),
    "xplat-probe-windows": ("xplat",),
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
    # Both `xplat-probe-*` legs carry the SAME local spelling, and that is the
    # honest shape rather than a copy-paste: the pair differs only by the OS leg
    # CI supplies, and a developer's machine has exactly one OS. The command is
    # the battery on THIS host — the same reading the job takes of its own
    # runner, which is precisely the relationship `test`'s local command has to
    # an ubuntu runner it does not reproduce either. Excluding them instead would
    # be the false claim in the other direction: the battery runs locally (it is
    # self-isolating, needs no network and no secrets), and it is the cheapest
    # way to see a probe regression before pushing. `run_jobs` runs the shared
    # spelling ONCE, because a second run of a several-minute battery answers
    # nothing the first did not — the Linux leg's richer local rig
    # (`scripts/xplat_linux_matrix.sh`, which needs Docker and rebuilds images)
    # is deliberately not wired into `--run`: see `scripts/xplat/README.md`.
    "xplat-probe-linux": (".venv/bin/python scripts/xplat_probe.py",),
    "xplat-probe-windows": (".venv/bin/python scripts/xplat_probe.py",),
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

#: Job id -> the local command that must NOT be narrowed, with the reason it
#: stays whole-tree. Every scoped job's command is built from its
#: `JOB_COMMANDS` entry (see `scope_plan`), so there is still one place that
#: spells what a gate runs — this map only says which of them may take a file
#: list, and it is empty today: nothing that is left is a candidate.
UNSCOPED_JOBS: dict[str, str] = {}

#: Jobs whose local command is narrowed to the files the diff touched, in
#: `JOB_COMMANDS` order. Anything not listed here runs exactly as CI spells it.
SCOPED_JOBS: tuple[str, ...] = ("lint", "type-check", "test", "tui-e2e")

#: The scripted input each scoped job's command narrows, replaced by the
#: selected file list. It is ALSO the tree a job's test universe comes from, so
#: there is one map rather than a marker in one place and a tree in another. A
#: command that no longer contains its marker means `JOB_COMMANDS` moved under
#: this module's feet, which `_narrow` refuses rather than guessing.
SCOPE_MARKERS: dict[str, str] = {
    "lint": ".",
    "type-check": ".",
    "test": "tests/unit",
    "tui-e2e": "tests/e2e",
}

#: The closure arm for `type-check`. A file-list pyright analyzes ONLY the files
#: it is given — measured: an error in an imported but unlisted module is not
#: reported, while a signature change in a listed file's dependency IS reported
#: in the listed file — so the list must be the changed files plus their
#: transitive reverse dependents, and its cost tracks that set. When naming them
#: would drag most of the program in, the narrowed command is the whole-tree
#: command with extra steps, and this arm says so instead of pretending.
SCOPE_MAX_CLOSURE_FRACTION = 0.5

#: The trees the import graph parses. `benchmarks/**` and `docs/**` are outside
#: it on purpose: neither is imported by the suite, and a changed file there is
#: a `scope_barriers` trigger rather than something to resolve.
GRAPH_TREES: tuple[str, ...] = ("local_operator", "tests", "scripts")

#: Trees the graph parses IN ADDITION to `GRAPH_TREES`, because a GATE reads them:
#: `pyright` analyzes these (they are not in its `exclude`) and they import
#: `local_operator.*`, so a change inside the covered trees can break them. A
#: scoped `type-check` list that omits them is therefore not "complete for what this
#: change can break" (#1322 QA round 1, Q-1: measured 20 files pyright analyzes and
#: the graph did not, 8 of them importers of `local_operator`).
ANALYZED_TREES: tuple[str, ...] = ("benchmarks/osworld_v2_adapter",)

#: Everything the graph parses: the suite's trees plus the trees a gate analyses.
PARSED_TREES: tuple[str, ...] = (*GRAPH_TREES, *ANALYZED_TREES)

#: What black/isort/flake8 read. `.pyi` is deliberately absent: a stub change is a
#: `scope_barriers` trigger (the graph does not parse stubs), so it never reaches
#: the lint target filter — listing it here would describe an arm no input can
#: hit, which reads as coverage this does not have.
LINT_SUFFIXES: tuple[str, ...] = (".py",)

#: The two fraction arms a narrowed TEST selection must stay under. Weight is
#: the primary arm because this suite's cost is not spread evenly over its
#: files: `tests/unit/tui` is 82.3% of the measured serial weight while being
#: 25% of the files, so a file-count arm alone would wave through a selection
#: that costs as much as the whole run. The file-count arm is the fallback for a
#: tree with no duration manifest, and a bound on pytest's own collection cost.
SCOPE_MAX_WEIGHT_FRACTION = 0.25
SCOPE_MAX_FILE_FRACTION = 0.5

#: Relative weight of a test file that `tests/durations.json` does not mention.
#: The manifest names its own fallback; this is used only when the file is
#: missing, and the arm is reported as unavailable in that case.
DURATIONS_PATH = "tests/durations.json"

# --------------------------------------------------------------------------
# Barriers: what makes a narrowed selection unsound rather than merely slow
# --------------------------------------------------------------------------
# The shape is a WHITELIST, not a denylist. A changed path may narrow a gate only
# when it is a `.py` file the import graph covers, or documentation no gate
# reads. EVERYTHING else — the cases named below, and anything nobody thought
# about — runs the whole-tree command and prints the path that stopped it.
# Inverting this is how a narrowed run goes green on the file that mattered.
#
# The named cases, each for a reason the static graph cannot express:
#
# * a CI/gating file decides what runs at all, and the classifier must not
#   narrow the gates it edits;
# * a conftest decides collection and fixtures for a whole subtree;
# * a gate/manifest config changes what the gate commands themselves do;
# * a package `__init__` changes a module's surface and pytest's collection
#   semantics without appearing as an import of anything;
# * a shared test-helper tree is not a contract any import edge encodes;
# * the entry points are the process's own entry, which every test that boots the
#   app inherits (and the runtime composes itself there from names the graph
#   cannot see: measured `importlib.import_module(<computed name>)` sites in
#   `local_operator/agents.py`, `session_factory.py`, `providers/registry.py`,
#   `optional.py` and `evaluation/adapters/discovery.py`);
# * a tree the suite reads by PATH rather than by import — `extension/`, where
#   `tests/unit/browser_bridge/test_extension_version_skew.py` reads the
#   committed files and no import edge exists to select with;
# * package and test data read at run time, which imports nothing;
# * a vendored tree whose lint behaviour its own exclude lists own;
# * a stub file, a deleted module, or Python outside the covered trees.
STRUCTURAL_PATHS = frozenset(
    {
        "Makefile",
        ".flake8",
        "setup.cfg",
        "tox.ini",
        "pyproject.toml",
        "uv.lock",
    }
)
#: Path prefixes, each with the reason it gives when it fires.
STRUCTURAL_PREFIXES: tuple[tuple[str, str], ...] = (
    (".github/", "CI and gating files — the classifier must not narrow the gates it edits"),
    (
        "extension/",
        "a tree the suite reads by PATH, not by import — no graph edge can select for it",
    ),
    (
        "benchmarks/osworld_v2_adapter/src/evaluation_examples/",
        "vendored upstream release helpers, byte-identical to their recorded SHA",
    ),
    (
        "tests/helpers/",
        "a shared test-helper tree — no import edge is a contract for what it does",
    ),
)

#: Console-script targets (`[project.scripts]` in pyproject.toml) plus the
#: module `python -m local_operator` executes. A change here is a change to the
#: process's own entry, which every test that boots the real app inherits.
#: The files a test reaches by importing the APPLICATION, and the closure behind
#: them — the one reference class no static analysis can bound. Measured on this
#: tree (measured on `b0577cdd`, 2026-09-20): 448 of the 508 `local_operator/**`
#: modules are inside this closure, so a
#: change to any of them makes almost every test a candidate and the whole-tree
#: command is the only honest answer. Restricted to three roots rather than "all
#: of `local_operator/**`" on purpose: the ~60 modules OUTSIDE the closure are
#: exactly the class where narrowing still pays, and rounds 1-4 found every false
#: green in the closure's shadow rather than in it (#1322, decision 2026-09-19).
BOOT_ROOTS: tuple[str, ...] = (
    "local_operator/cli.py",
    "local_operator/tui/app.py",
    "local_operator/session_factory.py",
)

ENTRY_POINT_PATHS = frozenset(
    {
        "local_operator/__main__.py",
        "local_operator/cli.py",
    }
)

#: Human-readable predicate per flag, used in `--summary` so the reason a job
#: ran (or did not) is written down where a reviewer reads it.
FLAG_REASONS: dict[str, str] = {
    "lint": "at least one changed path outside the inert set (docs/**, web-only)",
    "types": "at least one changed path outside the inert set (docs/**, web-only)",
    "budget": "a path that can change the assembled start-of-session context",
    "unit": "at least one changed path outside the inert set (docs/**, web-only)",
    "tui": "same predicate as `unit`, deliberately (scripts/** is a tui input)",
    "windows": "a path the Windows job reads, or a manifest/CI/gate config",
    "xplat": (
        "a path the probe battery reads: the package it walks and imports, its "
        "own scripts, or the test-suite app double it boots the TUI against"
    ),
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
    # `CAT_PYTHON` is the broad half on purpose: the battery AST-scans every
    # `local_operator/**/*.py` for POSIX-only module-level imports and then
    # imports EVERY module it can walk (`pkgutil.walk_packages`), so a change to
    # any file in the package is a change to what the battery measures — a
    # narrower predicate would skip the legs on exactly the PRs they exist for.
    # `CAT_DEPS_LOCK` is absent because these jobs install from
    # `pyproject.toml` (`pip install -e ".[dev]"`), not from `uv.lock`, and
    # `CAT_GATE_CONFIG` because neither leg runs a linter or the Makefile.
    xplat = (
        bool(cats & {CAT_PYTHON, CAT_MANIFEST, CAT_CI, CAT_OTHER})
        or bool(path_set & (XPLAT_SCRIPT_INPUTS | XPLAT_TEST_INPUTS))
        or any(p.startswith(XPLAT_SCRIPT_PREFIXES) for p in path_set)
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
        "xplat": xplat,
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


#: Force git to emit paths as raw bytes rather than C-quoting them. With the
#: default `core.quotePath`, a non-ASCII path arrives as
#: `"local_operator/probe_\303\274n\303\257code.py"`, which names no real file:
#: the path is reported mangled, `category_of` reads it as `CAT_OTHER`, and a
#: one-file change escalates to the whole matrix for a reason that does not apply
#: (fail-open, but the wrong answer and an inapplicable explanation). A path
#: containing a NEWLINE is still mangled by the line-based parse below and still
#: fails open — that is the residual, and `-z` is the fix if it ever bites.
_GIT_RAW_PATHS = ("-c", "core.quotePath=false")


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
    rc, out, _ = _git([*_GIT_RAW_PATHS, "diff", "--name-status", "-M", base, *target], cwd=cwd)
    if rc != 0:
        return None
    paths = _parse_name_status(out)
    if local:
        # Untracked files are changes a local gate must see (a new test file
        # nobody staged still has to be linted) and are invisible to `git diff`.
        rc, out, _ = _git([*_GIT_RAW_PATHS, "ls-files", "--others", "--exclude-standard"], cwd=cwd)
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
# File-level scoping: the LOCAL run path only
# --------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------
# `classify` above answers one question per JOB, and that is the right
# granularity for CI: a job runs or it does not. Locally it is the wrong one,
# because the commands those jobs spell are whole-tree — `flake8 .`,
# `pyright … .`, `pytest tests/unit -q` — and the inner loop pays for them
# several times per PR. Measured on this host under fleet load: flake8 . 28.7 s,
# isort --check . 15.6 s, black --check . 3.2 s, and a full local unit run
# 40-55 min wall (107.7 min of measured serial weight, 82.3% of it under
# tests/unit/tui). For a two-file PR almost none of that touches the change.
#
# So this section answers the same question one level down, for the LOCAL run
# only: which FILES can this diff affect? CI is untouched — `ci.yml` still runs
# `flake8 .`, `pyright` over the project and every shard, and that stays the
# authoritative gate. A local green was never evidence about the whole tree;
# what changes is that it is now LEGIBLE about how much narrower it is.
#
# WHAT MAKES A SELECTION SOUND
# ----------------------------
# The selection is an UNDER-APPROXIMATION of "what the change can break", so it
# is only ever allowed to run when the approximation is safe:
#
# * a changed file the graph cannot place, an unreadable tree, a structural path
#   (`scope_barriers`), or a selection above the fraction arms runs the whole
#   command as CI spells it, and prints WHICH trigger fired;
# * the graph is built by PARSING, never importing: importing would execute
#   module-level code, need a `.venv`-shaped dependency this stdlib-only
#   classifier must not have, and turn a syntax error in an unrelated module
#   into a classification crash;
# * a computed-name dynamic import (`importlib.import_module(name)`) is a hole
#   the graph cannot close. Resolvable ones are resolved (literal strings, and
#   the literal head of an f-string). The rest are collected as `unnamed` and
#   PRINTED with every scoped run, because a limit that is not stated is the
#   quiet narrowing this design exists to avoid.
#
# The honest limit, stated here rather than buried: this suite's tests import
# the assembled app, so a change anywhere the app (or a test conftest) imports
# selects most of the tree by construction — measured, the median module has
# 93% of the suite's weight behind it — and then the fraction arms send the run
# back to whole-tree. Scoping pays on the diffs it can pay on: a changed test
# file, `scripts/**`, and the subtrees the app does not import (the server, the
# evaluation harness). That asymmetry is a property of the import shape, not of
# the thresholds, and the fallback is what keeps it from being a guess.

#: A dotted path literal is a candidate module name only if it looks like one.
DOTTED_NAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")

#: A string constant that could NAME a Python file: a path ending in a source
#: extension, or a bare basename. This is how the suite reaches a file it does
#: not import — `sys.executable` + a path, or `-m` — which no import statement
#: records, so a scoped `test` job that ignored these names would go green while
#: CI's `test` job failed (#1322 review, BLOCKER 1).
#:
#: The optional leading `[/.~]*` is load-bearing, not tidiness: an f-string leaves
#: its literal segments as separate constants, so the natural `f"{ROOT}/scripts/x.py"`
#: contributes `/scripts/x.py`, and the equally ordinary `"./x.py"`, `"../x.py"`,
#: `"~/x.py"` and `/absolute/path/x.py` spellings all begin with a separator
#: (`.{1,2}/` is two characters, so a single optional one is not enough).
#: Without it NONE of them were collected or resolved, so the file they name
#: selected nothing, the gate went green, and — unlike a computed import — nothing
#: was printed to say so (#1322 round 2, MAJOR 1). `_name_target` is what decides
#: whether a collected name lands on a real file; the two must admit the same
#: spellings, or the collection gate silently drops the case before resolution.
#:
#: A WINDOWS-style spelling (`"scripts" + "\\x.py"`) is deliberately NOT admitted:
#: the separator class is POSIX-only, so such a literal is collected by neither
#: branch and no edge is built for it. The gates here are POSIX, the repo has no
#: backslash paths, and admitting `\\` would also resolve a string that merely
#: CONTAINS one (a regex, a Windows-only fixture) as a repo path. Stated rather
#: than fixed (round 3, NIT): a Windows CI leg would want it fixed together with
#: the path normalisation that has to accompany it.
PY_PATH_RE = re.compile(r"[/.~]*[A-Za-z_][\w./\-]*\.py[io]?\Z")

#: Dynamic-import callables: `importlib.import_module(...)`, `__import__(...)`.
_DYNAMIC_IMPORT_NAMES = ("import_module", "__import__")


def _graph_files(root: Path) -> tuple[list[str], list[str]]:
    """Every .py the graph parses, and every tree it could not reach.

    A missing tree is reported rather than ignored: a graph built over half the
    repository would select too little and say nothing about it.
    """
    files: list[str] = []
    unreadable: list[str] = []
    for tree in PARSED_TREES:
        base = root / tree
        if not base.is_dir():
            # A missing `GRAPH_TREES` entry is reported: a graph over half the
            # repository selects too little and must say so. A missing
            # `ANALYZED_TREES` entry is not — those trees exist for a gate's benefit,
            # and a checkout without them (as every fixture repository is) has
            # nothing for the graph to cover.
            if tree in GRAPH_TREES:
                unreadable.append(f"{tree}/ (not a directory)")
            continue
        for path in sorted(base.rglob("*.py")):
            files.append(path.relative_to(root).as_posix())
    root_conftest = root / "conftest.py"
    if root_conftest.is_file():
        files.append("conftest.py")
    return files, unreadable


def _package_parts(rel: str) -> list[str]:
    """The dotted parts of the package a file LIVES IN.

    `local_operator/tools/foo.py` lives in `local_operator.tools` — that is the
    anchor a relative import is resolved against, and it is NOT the module's own
    name. An `__init__.py` IS its package, so it keeps every part; a normal
    module drops its stem.
    """
    parts = rel[:-3].split("/")
    if parts[-1] != "__init__":
        parts = parts[:-1]
    return parts


def _module_name(rel: str) -> str:
    """The dotted name a file is importable AS (an `__init__` is its package)."""
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_module(root: Path, dotted: str) -> str | None:
    """Resolve a dotted module name to a repo-relative file, or None.

    Only files are considered: a name that resolves to neither `<path>.py` nor
    `<path>/__init__.py` is not in this repository (a third-party or stdlib
    module), which is exactly the distinction the graph needs.
    """
    if not dotted:
        return None
    base = root.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:  # pragma: no cover - root is a real ancestor here
                return None
    return None


def _repo_basenames(files: Iterable[str]) -> Mapping[str, tuple[str, ...]]:
    """Covered files indexed by basename.

    A name reference may carry no directory at all (`"visual_gallery.py"`) or a
    directory that is not this checkout's (an absolute path built from
    `__file__`, or one that reached the test through `sys.executable`). The
    basename is then the only part that resolves, and it may resolve to more
    than one file: the edge is an OBLIGATION to run a test, so every candidate
    is kept — over-including is the fail-closed direction here.
    """
    index: dict[str, list[str]] = {}
    for rel in files:
        index.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    return {name: tuple(sorted(paths)) for name, paths in index.items()}


def _name_target(
    root: Path,
    literal: str,
    basenames: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """The covered files a string constant can NAME, or an empty tuple.

    Two spellings, because the suite uses both for a file it never imports: a
    dotted module name (`python -m local_operator.exec_worker`, a registry name)
    and a path (`sys.executable` + `scripts/visual_gallery.py`). A path is
    normalised the way a reader would read it — empty and `.` segments dropped,
    then leading `..`/`~` segments dropped — and every remaining suffix is tried,
    so `scripts/x.py`, `./scripts/x.py`, `../scripts/x.py`, `~/scripts/x.py`,
    `/scripts/x.py` and `/abs/path/scripts/x.py` all land on the same file; a
    bare basename falls back to the basename index.
    `test_every_spelling_of_a_repo_path_is_collected_and_resolved` asserts that
    list, because a spelling the resolver cannot see is a silent
    false green and the real-tree property test iterates resolution OUTPUT, so it
    cannot catch one.
    """
    resolved: set[str] = set()
    if DOTTED_NAME_RE.match(literal):
        target = _resolve_module(root, literal)
        if target:
            resolved.add(target)
    if PY_PATH_RE.match(literal):
        # Drop empty and `.` segments (from `./x`, `/x`, `a//b`) and then the
        # leading `..`/`~` ones. What is left is a suffix of the real path, and
        # the loop below tries each suffix from longest to shortest — so an
        # absolute path lands on its repo-relative tail rather than on nothing.
        parts = [part for part in literal.split("/") if part not in ("", ".")]
        while parts and parts[0] in ("..", "~"):
            parts.pop(0)
        hit = next(
            (
                suffix
                for start in range(len(parts))
                if (root / (suffix := "/".join(parts[start:]))).is_file()
            ),
            None,
        )
        if hit:
            resolved.add(hit)
        elif parts:
            # The TRUE last component, so a dotted name like `.hidden.py` is not
            # truncated by the separator strip above.
            resolved.update(basenames.get(parts[-1], ()))
    return tuple(sorted(resolved))


@dataclass(frozen=True)
class _Scan:
    """A directory scan: the files a file READS with no name to point at.

    `(ROOT / "scripts").glob("*.py")` reads every Python file in `scripts/` and
    names none of them, so an edge built only from names cannot see it — measured
    on this tree: `tests/unit/tui/test_visual_gallery.py` reads all 197 covered
    `scripts/*.py` that way and asserts an ordering invariant on each (#1322 QA
    round 2, Q-1). One covered `scripts/**` file changed by a token and reached
    only through that scan selected NOTHING, so the local gate went green while
    CI's `test` job was red: the round-1 blocker shape, by a third road.

    `receiver` is the scanned directory's EXPRESSION, evaluated later by
    `_scan_directory` (path literals, `__file__`, `.parent`/`.parents[N]`,
    `.resolve()`, and ONE assignment hop, so `SCRIPTS = ROOT / "scripts"` places
    too). A receiver the graph cannot evaluate is UNPLACED, and an unplaced scan of
    ANY pattern is then RECORDED in `ImportGraph.unplaced_scans`: it is a reference
    the resolver saw and could not place, so `_test_barriers` refuses to narrow the
    two pytest jobs for that diff and prints the site. The earlier policy — arm the
    `.py`-spelling subset as a reader of every covered file, print the rest — was
    withdrawn after rounds 3 and 4 each found a false green through the cases it
    left out; what survives is the refusal, not a cheaper guess.
    """

    receiver: ast.expr | None
    pattern: str
    #: Literal directory components taken from the call's ARGUMENTS or from the
    #: leading components of the pattern — `glob.glob("scripts/*.py")` and
    #: `glob.glob(os.path.join("scripts", "*.py"))` both spell their directory
    #: there, and reading hints off the receiver alone made the `os.*`/`glob.*`
    #: forms resolve to nothing at all (run 4, QA Q-3).
    dir_literals: tuple[str, ...] = ()
    #: True when the receiver is the module itself (`glob.glob`, `os.walk`), which
    #: is what makes `dir_literals` repo-relative rather than relative to a
    #: receiver directory the evaluator could not place.
    module_form: bool = False
    lineno: int = 0


#: Directory-scan APIs: name -> (takes a glob pattern, prefix that makes it depth-aware).
#: The prefix matters: `rglob(p)` and `os.walk(d)` read a SUBTREE while `glob(p)`
#: reads one level, so the pattern carries the depth and the matcher honours it.
_SCAN_APIS: Mapping[str, tuple[bool, str]] = {
    "glob": (True, ""),
    "iglob": (True, ""),
    "rglob": (True, "**/"),
    "iterdir": (False, "*"),
    "listdir": (False, "*"),
    "scandir": (False, "*"),
    "walk": (False, "**/*"),
}

#: How many `name = <expr>` hops the scan resolver will follow. Four is the
#: measured need (the live blocker is one; a `A = B / "x"` chain is two) plus room,
#: and the bound is what keeps a cyclic assignment from looping.
_MAX_ASSIGNMENT_HOPS = 4

#: A receiver that is one of these is the MODULE, not a directory: `glob.glob(p)`
#: and `os.walk(p)` spell their path in the arguments, and treating `glob`/`os` as
#: a directory expression is how both forms went unplaced (run 4, QA Q-3).
_MODULE_RECEIVERS = frozenset({"os", "glob"})

#: These take their directory as the FIRST ARGUMENT, whether they are spelled
#: `os.walk(p)` or bare `walk(p)`; reading hints off the receiver (`os`, or the
#: `os` in `os.walk`) is how a scan of a covered tree went unplaced and unprinted
#: (round 3, MINOR 1).
_PATH_ARG_SCANS = frozenset({"iterdir", "listdir", "scandir", "walk"})


def _directory_scan(node: ast.Call, assignments: Mapping[str, ast.expr]) -> _Scan | None:
    """A directory scan this call performs, or None when it is not one.

    Only the shape is read here; whether the directory is a COVERED tree is
    decided at resolve time against the real filesystem, which is also where an
    unplaceable scan becomes an armed read or a printed limit rather than a guess.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        receiver: ast.expr | None = func.value
    elif isinstance(func, ast.Name):
        name = func.id
        receiver = None
    else:
        return None
    if name not in _SCAN_APIS:
        return None
    takes_pattern, prefix = _SCAN_APIS[name]
    module_form = False
    if name in _PATH_ARG_SCANS:
        # `os.walk(p)`/`os.listdir(p)` take their directory as the FIRST argument,
        # but `SCRIPTS.iterdir()` takes none — the RECEIVER is the directory. The
        # call-argument rule alone dropped every argument-less method call, which is
        # how round 4's `iterdir()` regression made 158 scans invisible: not placed,
        # not recorded, not printed.
        if node.args:
            receiver = node.args[0]
        elif isinstance(func, ast.Attribute):
            receiver = func.value
        else:
            receiver = None
    elif isinstance(receiver, ast.Name) and receiver.id in _MODULE_RECEIVERS:
        # `glob.glob(p)` / `os.walk(p)`: the RECEIVER is the module, so both the
        # directory and the pattern live in the arguments.
        module_form = True
        receiver = None
    if receiver is None and not module_form:
        return None
    if isinstance(receiver, ast.Name) and receiver.id in assignments:
        # ONE hop, and only for a plain name: `SCRIPTS = ROOT / "scripts"` is how
        # the reader that drove round 3's blocker spelled its directory, and one
        # hop places it plus six more of the nineteen `.py` sites on this tree.
        receiver = assignments[receiver.id]
    # A scan that takes NO pattern still reads a SHAPE, and the shape differs by
    # API: `iterdir()`/`listdir()`/`scandir()` read the directory's DIRECT children
    # while `walk()` recurses (its prefix carries that). Defaulting the pattern to
    # `*` gave the non-recursive three `"**"` — "everything at any depth" — which
    # over-selected (measured: 816 targets for a `.iterdir()` whose truth is ~25
    # direct children) and put a wrong number in the note a developer reads.
    raw_pattern = "*" if takes_pattern else ""
    pattern_hints: tuple[str, ...] = ()
    if takes_pattern and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            raw_pattern = first.value
        else:
            # A computed pattern is not a reason to skip the scan, and the pieces
            # are often literals beside each other: `os.path.join("scripts",
            # "*.py")` names both the directory and the extension (QA Q-3).
            parts = [
                sub.value
                for sub in ast.walk(first)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            ]
            python_parts = [part for part in parts if _looks_like_a_python_scan(part)]
            if python_parts:
                raw_pattern = python_parts[-1]
                pattern_hints = tuple(parts[: parts.index(python_parts[-1])])
            else:
                pattern_hints = tuple(parts)
    # A pattern may spell its own directory: `glob.glob("scripts/*.py")` names
    # `scripts` in the component the matcher would otherwise treat as a wildcard
    # search. Literal leading components are consumed into the directory and the
    # rest stays the pattern.
    path_parts = [part for part in raw_pattern.split("/") if part not in ("", ".")]
    literal_leading: list[str] = []
    while path_parts and not any(char in path_parts[0] for char in "*?["):
        literal_leading.append(path_parts.pop(0))
    return _Scan(
        receiver=receiver,
        # The `or "*"` fallback belongs to the APIs that TAKE a pattern: for the
        # non-recursive three the prefix already IS the shape, and re-adding the
        # wildcard here is what turned `iterdir()` into `"**"` (F.3).
        pattern=f"{prefix}{'/'.join(path_parts) or ('*' if takes_pattern else '')}",
        dir_literals=tuple(pattern_hints) + tuple(literal_leading),
        module_form=module_form,
        lineno=node.lineno,
    )


def _path_value(
    node: ast.expr,
    rel: str,
    root: Path,
    assignments: Mapping[str, ast.expr],
    *,
    hops: int,
) -> Path | None:
    """The `Path` an expression denotes, or None when the graph cannot say.

    Deliberately tiny and total: a string literal, `__file__`, `Path(...)`,
    `/`-joins, `.parent`, `.parents[N]`, `.resolve()`/`.absolute()`, and ONE
    assignment hop. Anything else — an environment lookup, a helper call, an
    f-string, a `chdir`-relative guess — returns None, which the caller turns into
    an armed read or a printed limit rather than a guess about which directory a
    scan touches. A WRONG directory is the one outcome worse than either, because
    it is silent (round 3, MAJOR 1).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return root / rel
        if hops and node.id in assignments:
            return _path_value(assignments[node.id], rel, root, assignments, hops=hops - 1)
        return None
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            base = _path_value(node.value, rel, root, assignments, hops=hops)
            return None if base is None else base.parent
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        # `.parents[N]` is a SUBSCRIPT, not a call — and it is how the reader that
        # drove round 3's blocker spelled the repository root
        # (`Path(__file__).resolve().parents[3] / "scripts"`). Missing this form
        # left that scan armed rather than placed, which is safe but costs the
        # precision the one-hop exists for.
        if node.value.attr == "parents":
            index = node.slice
            base = _path_value(node.value.value, rel, root, assignments, hops=hops)
            if (
                base is not None
                and isinstance(index, ast.Constant)
                and isinstance(index.value, int)
                and 0 <= index.value < len(base.parents)
            ):
                return base.parents[index.value]
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _path_value(node.left, rel, root, assignments, hops=hops)
        right = _path_value(node.right, rel, root, assignments, hops=hops)
        return None if left is None or right is None else left / right
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("Path", "PosixPath"):
            # EVERY argument is a path component, so they compose: `Path(ROOT,
            # "scripts")` is `ROOT / "scripts"`. Reading only `args[0]` (round 4's
            # QA Q-4) resolved that call to the REPOSITORY ROOT, which is a WRONG
            # placement rather than an unplaced one — the one outcome the resolver
            # must never produce, because it is silent.
            parts = [
                _path_value(argument, rel, root, assignments, hops=hops) for argument in node.args
            ]
            if not parts or any(part is None for part in parts):
                return None
            joined = parts[0]
            for extra in parts[1:]:
                joined = joined / extra  # type: ignore[operator]
            return joined
        if isinstance(func, ast.Attribute) and func.attr == "join" and node.args:
            # `os.path.join("scripts", "*.py")`: joining is how a path is built
            # without an f-string, and it is evaluable when every part is.
            parts = [
                _path_value(argument, rel, root, assignments, hops=hops) for argument in node.args
            ]
            if any(part is None for part in parts):
                return None
            joined = parts[0]
            for extra in parts[1:]:
                joined = joined / extra  # type: ignore[operator]
            return joined
        if isinstance(func, ast.Attribute):
            if func.attr in ("resolve", "absolute") and not node.args:
                return _path_value(func.value, rel, root, assignments, hops=hops)
            if func.attr == "parents" and node.args:
                index = node.args[0]
                if isinstance(index, ast.Constant) and isinstance(index.value, int):
                    base = _path_value(func.value, rel, root, assignments, hops=hops)
                    return None if base is None else base.parents[index.value]
        return None
    return None


def _scan_directory(
    expression: ast.expr | None,
    rel: str,
    root: Path,
    assignments: Mapping[str, ast.expr],
) -> str | None:
    """The covered directory a scan reads, or None when it cannot be placed.

    A relative literal is read as repo-relative, which can only OVER-include: the
    files it then names are real covered files, and a scan that really read a
    runtime directory reads none of them.
    """
    if expression is None:
        return None
    # A BUDGET of hops, not one: `DIAG = SCRIPTS / "diag"` over
    # `SCRIPTS = ROOT / "scripts"` is ordinary code, and the live blocker's own
    # receiver (`SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"`) sits
    # one hop down. The budget bounds the work and any `A = B; B = A` cycle; a
    # chain longer than this is unplaced, which arms rather than guesses.
    value = _path_value(expression, rel, root, assignments, hops=_MAX_ASSIGNMENT_HOPS)
    if value is None:
        return None
    if not value.is_absolute():
        value = root / value
    try:
        relative = value.relative_to(root).as_posix()
    except ValueError:
        return None
    if relative in ("", "."):
        relative = "."
    elif not (root / relative).is_dir():
        return None
    return relative


def _segment_regex(segment: str) -> str:
    """One glob segment as a regex, where `*`/`?` do NOT cross a separator."""
    out: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            close = segment.find("]", index + 1)
            if close == -1:
                out.append(re.escape(char))
            else:
                body = segment[index + 1 : close]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                index = close
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """A glob pattern as a regex over a repo-relative path.

    Directory components in the pattern are honoured — `*/*.py` is exactly one
    level, `**/*.py` is any depth — because dropping them (and matching on the
    basename alone) made a scan resolve to the wrong set silently (round 3,
    MAJOR 1). Depth is the reason this is not `fnmatch`, whose `*` crosses `/`.
    """
    segments = [segment for segment in pattern.split("/") if segment not in ("", ".")]
    pieces: list[str] = []
    separate = False
    for index, segment in enumerate(segments):
        if segment == "**":
            # `**` matches whole directory NAMES, so it carries its own trailing
            # separator when something follows it and needs no separator before it.
            if index == len(segments) - 1:
                pieces.append("/" if separate else "")
                pieces.append(".*")
            else:
                pieces.append("(?:[^/]+/)*")
            separate = False
            continue
        if separate:
            pieces.append("/")
        pieces.append(_segment_regex(segment))
        separate = True
    return re.compile("".join(pieces) + r"\Z")


def _scan_targets(
    scan: _Scan,
    rel: str,
    root: Path,
    files: Sequence[str],
    assignments: Mapping[str, ast.expr],
) -> tuple[tuple[str, ...], bool]:
    """(covered files the scan reads, whether the scan could not be placed).

    Placement is all-or-nothing: the receiver's WHOLE literal chain has to be the
    directory (`scripts/diag`, never the `scripts` ancestor of it), and a pattern
    with directory components is matched against the path, not the basename.
    """
    directory = _scan_directory(scan.receiver, rel, root, assignments)
    if directory is not None and scan.dir_literals:
        # Components the PATTERN spelled compose onto the receiver directory:
        # `(ROOT / "scripts").glob("sub/*.py")` is `scripts/sub`, and it is
        # unplaced rather than silently widened if that is not a directory.
        candidate = "/".join((directory, *scan.dir_literals))
        directory = candidate if (root / candidate).is_dir() else None
    elif directory is None and scan.module_form and scan.dir_literals:
        # `glob.glob("scripts/*.py")`: no receiver directory exists, so the
        # literal components are repository-relative.
        candidate = "/".join(scan.dir_literals)
        directory = candidate if (root / candidate).is_dir() else None
    if directory is None:
        return (), True
    base = "" if directory == "." else f"{directory}/"
    matcher = _glob_regex(scan.pattern)
    return (
        tuple(
            candidate
            for candidate in files
            if candidate != rel
            and candidate.startswith(base)
            and matcher.match(candidate[len(base) :]) is not None
        ),
        False,
    )


def _looks_like_a_python_scan(pattern: str) -> bool:
    """Whether a scan's pattern asks for Python files.

    It no longer decides anything about arming — an unplaceable scan of any pattern
    is recorded and stops the selection (`_test_barriers`). What it still does is
    pick the pattern out of a COMPUTED one (`os.path.join("scripts", "*.py")`): the
    last `.py`-spelling piece is the pattern, and the pieces before it are the
    directory hints.
    """
    return pattern.split("/")[-1].endswith(".py")


def _dynamic_call_target(node: ast.Call) -> tuple[str, str] | None:
    """Classify `import_module(...)`/`__import__(...)` by what it can load.

    Returns `("module", name)` for a literal name, `("prefix", head)` when the
    argument is an f-string whose literal head names a package (so anything
    under that head may be loaded), or `("computed", text)` when the target is
    a name the parser cannot see. An empty literal head (`f"{pkg}.x"`) is
    computed, not a prefix: nothing about it is known.
    """
    target = ast.unparse(node.func)
    if not target.endswith(_DYNAMIC_IMPORT_NAMES):
        return None
    if not node.args:
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return ("module", argument.value)
    if isinstance(argument, ast.JoinedStr):
        head = "".join(
            value.value
            for value in argument.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
        if head:
            return ("prefix", head)
        return ("computed", ast.unparse(node))
    return ("computed", ast.unparse(node))


@dataclass(frozen=True)
class _References:
    """What one parsed file names, in the forms the graph can use."""

    modules: frozenset[str]
    #: `(base, name)` pairs from `from base import name`. Whether `base.name` is a
    #: module or just an attribute of `base` is decided at RESOLVE time: only a
    #: `base` that resolves to a package can hold submodules, and treating a
    #: plain attribute as a module invents edges (on a case-insensitive
    #: filesystem `from .alpha import ALPHA` resolves `local_operator.ALPHA` to
    #: `local_operator/alpha.py`).
    submodules: tuple[tuple[str, str], ...]
    prefixes: frozenset[str]
    unnamed: tuple[str, ...]
    literals: frozenset[str]
    scans: tuple[_Scan, ...]
    #: `name = <expr>` for the resolver's ONE assignment hop: a scan receiver that
    #: is a plain name is replaced by its assignment, and the assigned expression
    #: (`ROOT / "scripts"`) still has to resolve, which needs this map again.
    assignments: Mapping[str, ast.expr]


def _assignments(tree: ast.AST) -> Mapping[str, ast.expr]:
    """Simple `name = <expr>` targets, for the ONE hop a scan receiver may take."""
    found: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                found.setdefault(target.id, node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            found.setdefault(node.target.id, node.value)
    return found


def _references(rel: str, source: str) -> _References:
    """Extract a file's module references WITHOUT importing it.

    Every `Import`/`ImportFrom` counts, at module level or inside a function:
    a lazy import is still a real dependency, and it is how this codebase loads
    most of its heavy modules. String constants are collected too — both dotted
    module names and file paths (`PY_PATH_RE`) — because a name a file carries is
    evidence of a dependency even when no import statement exists: a registry
    naming `"local_operator.providers.oauth.anthropic"` in a table, a test
    naming `"scripts/visual_gallery.py"`, a `mock.patch("a.b.c")` target.

    What differs is which of those become edges. `imports` takes a dotted name
    only from a file that imports by name (`if refs.prefixes or refs.unnamed`),
    so prose in a file that imports nothing statically cannot invent one.
    `referrers` takes EVERY covered file's literals, deliberately: the whole point
    of that edge is the file that imports nothing and still runs the named file
    (`sys.executable` + a path, `-m`, a data table). The cost is over-inclusion —
    a name in a table can pull a run over an arm — and over-inclusion is the
    fail-closed direction here.
    """
    tree = ast.parse(source, filename=rel)
    assignments = _assignments(tree)
    package = _package_parts(rel)
    modules: set[str] = set()
    submodules: set[tuple[str, str]] = set()
    prefixes: set[str] = set()
    unnamed: list[str] = []
    literals: set[str] = set()
    scans: list[_Scan] = []
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from . import x` inside package P means P.x; level 2 means the
                # parent package, and `from .alpha import y` inside P means
                # P.alpha. An over-deep level is invalid Python; the slice below
                # clamps it to the repository root rather than raising, so a
                # weird file narrows the graph instead of breaking it.
                keep = package[: max(len(package) - (node.level - 1), 0)]
                if node.module:
                    keep = keep + node.module.split(".")
                base = ".".join(keep)
            else:
                base = node.module or ""
            if base:
                modules.add(base)
            for alias in node.names:
                if alias.name != "*":
                    submodules.add((base, alias.name))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            if DOTTED_NAME_RE.match(node.value) or PY_PATH_RE.match(node.value):
                literals.add(node.value)
        elif isinstance(node, ast.Call):
            scan = _directory_scan(node, assignments)
            if scan is not None:
                scans.append(scan)
            dynamic = _dynamic_call_target(node)
            if dynamic is None:
                continue
            kind, value = dynamic
            if kind == "module":
                modules.add(value)
            elif kind == "prefix":
                prefixes.add(value)
            else:
                unnamed.append(f"{rel}:{node.lineno}: {value}")
    return _References(
        modules=frozenset(modules),
        submodules=tuple(sorted(submodules)),
        prefixes=frozenset(prefixes),
        unnamed=tuple(sorted(unnamed)),
        literals=frozenset(literals),
        scans=tuple(scans),
        assignments=assignments,
    )


@dataclass(frozen=True)
class ImportGraph:
    """Parsed imports between the files the graph covers.

    `importers` is the reverse edge, which is what a change needs: the tests
    that can observe a module are the ones that transitively IMPORT it.

    `referrers` is the same question asked of a NAME instead of an import: a file
    that carries another file's path or dotted module name in a string constant
    can reach it at run time (`sys.executable` + a path, `python -m <name>`, a
    registry table, a `mock.patch("a.b.c")` target). That edge is reverse-only on
    purpose — it is an obligation to select the namer when the named file
    changes, not a claim that the file is imported, so `forward_closure` must not
    let it inflate the pyright cost estimate.

    The graph deliberately does NOT add an edge from an importer to a module's
    parent `__init__.py`. Importing `local_operator.tools.foo` does execute
    `local_operator/tools/__init__.py`, but every `__init__.py` is a
    `scope_barriers` trigger, so such a change never reaches the graph.
    """

    files: frozenset[str]
    imports: Mapping[str, frozenset[str]]
    importers: Mapping[str, frozenset[str]]
    prefixes: Mapping[str, tuple[str, ...]]
    referrers: Mapping[str, frozenset[str]]
    literals: Mapping[str, frozenset[str]]
    #: Files reachable by importing the application (`BOOT_ROOTS`), roots included.
    boot: frozenset[str]
    #: Scans whose directory the evaluator could not place — ANY pattern, not just
    #: the ones that spell `.py`. Recorded, never guessed at: they are what stops
    #: the test selection from being provably complete.
    unplaced_scans: tuple[str, ...]
    #: Dynamically loaded modules whose name the parser cannot see.
    unnamed: tuple[str, ...]
    unreadable: tuple[str, ...]

    def unresolved_sites(self) -> tuple[str, ...]:
        """Every reference this graph carries and cannot resolve.

        Two classes: a module loaded by a name the parser cannot see
        (`importlib.import_module(name)` with no literal head), and a directory scan
        whose directory could not be placed. BOTH stop the narrowing — the rule is
        unconditional, because deciding whether such a site *could* reach a
        particular changed file is the shape enumeration that produced four rounds
        of false greens (#1322 rounds 1-4; rounds 3 and 4 both came through the
        `.py`-only subset of exactly this set).
        """
        return tuple(sorted((*self.unnamed, *self.unplaced_scans)))

    def dependents(self, seeds: Iterable[str]) -> frozenset[str]:
        """Every file that transitively imports or NAMES a seed (seeds excluded).

        Breadth-first over the reverse edges, so a chain through an
        intermediate module counts: a test does not have to import the changed
        module directly to exercise it. This is also exactly the file list a
        narrowed `type-check` must name, because a change can only break the
        types of a file that depends on it.
        """
        seen: set[str] = set()
        queue = [seed for seed in seeds if seed in self.files]
        while queue:
            rel = queue.pop()
            for dependent in (*self.importers.get(rel, ()), *self.referrers.get(rel, ())):
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)
        return frozenset(seen - set(seeds))

    def name_referrers(self, rel: str) -> tuple[str, ...]:
        """Files whose string constants name `rel` — a path or a module name."""
        return tuple(sorted(self.referrers.get(rel, ())))

    def literals_of(self, rel: str) -> frozenset[str]:
        """The name-shaped string constants a file carries.

        Exposed for the graph's own property test, which asserts that every repo
        `.py` path a test names makes that test an input.
        """
        return self.literals.get(rel, frozenset())

    def hubs_for(self, rel: str) -> tuple[str, ...]:
        """Files that may dynamically load `rel` through a literal head.

        `importlib.import_module(f"local_operator.providers.oauth.{name}")`
        loads one of the modules under that head at run time, so the tests that
        import the HUB can exercise `rel` without importing it.
        """
        name = _module_name(rel)
        return tuple(
            sorted(
                hub
                for prefix, hubs in self.prefixes.items()
                if name.startswith(prefix)
                for hub in hubs
                if hub != rel
            )
        )

    def forward_closure(self, files: Iterable[str]) -> frozenset[str]:
        """Every file those files transitively IMPORT (the files themselves excluded).

        This is what a file-list pyright will actually analyze — measured: it
        reports diagnostics only for the files it is GIVEN — so it is the honest
        cost estimate for naming a file list instead of the whole tree.
        """
        given = {rel for rel in files if rel in self.files}
        seen: set[str] = set()
        queue = list(given)
        while queue:
            for imported in self.imports.get(queue.pop(), ()):
                if imported not in seen:
                    seen.add(imported)
                    queue.append(imported)
        return frozenset(seen - given)


#: Package heads a dotted name must start with to count as an unresolved reference
#: when it resolves to no covered module. Anything else is a name that cannot point
#: at a file this graph covers (a third-party module, a `mock.patch` target in
#: another library, prose that looks dotted), so recording it would only make the
#: barrier noisy in the fail-closed direction.
_REPO_PACKAGE_HEADS: tuple[str, ...] = ("local_operator", "tests", "scripts")

#: A constant that LOOKS like a repository path: it carries a separator or a Python
#: suffix, and it is not a glob pattern (patterns belong to the scan rules) nor a
#: URL (a scheme's `://` is not a path separator, whatever the string looks like).
_PATHLIKE_RE = re.compile(r"^(?!.*://)(?:.*/.*|[\w.\-]+\.(?:py|pyi|json|ya?ml|toml|txt|svg|tcss))$")


def _dotted_prefixes(name: str) -> tuple[str, ...]:
    """Every proper dotted prefix of a name, longest first, for module resolution."""
    parts = name.split(".")
    return tuple(".".join(parts[:index]) for index in range(len(parts) - 1, 0, -1))


def _docstring_nodes(tree: ast.AST) -> frozenset[int]:
    """The `id()` of every string constant that is a DOCSTRING.

    A docstring is prose: `'/path/to/file.py'` in an example is not a reference to
    anything, and a rule that fails closed would otherwise stop on it. Skipped by
    node identity, not by value, so a real usage of the same string still counts.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return frozenset(found)


def _looks_like_a_repo_name(literal: str) -> bool:
    """Whether an unresolved constant is a name that could have pointed at a file.

    Deliberately blunt, and blunt in the fail-closed direction: the two classes it
    admits are the ones this feature has been wrong about (`#1322` decision item 2
    — "a computed path", "a module named by a config value"), and the cost of a
    false positive is a whole-tree test run, never a green scoped one.
    """
    if any(character in literal for character in "*?["):
        return False
    if literal.startswith(_REPO_PACKAGE_HEADS):
        return True
    # A path-shaped name that claims a DIRECTORY STRUCTURE — `dir/x.py`, `/x.py`,
    # `./x.py`, `../x.py`, or the `scripts/` fragment an f-string leaves behind. A
    # bare filename is not recorded: a runtime filename with no directory component
    # ("config.yml", "state.json") is indistinguishable from ordinary data, and any
    # that DOES name a covered file is already resolved by basename above.
    return "/" in literal and bool(_PATHLIKE_RE.match(literal))


def _closure_over(imports: Mapping[str, frozenset[str]], seeds: Iterable[str]) -> frozenset[str]:
    """Everything `seeds` transitively import, the seeds themselves excluded."""
    seen: set[str] = set()
    queue = [rel for rel in seeds]
    while queue:
        for imported in imports.get(queue.pop(), ()):
            if imported not in seen:
                seen.add(imported)
                queue.append(imported)
    return frozenset(seen - set(seeds))


def build_import_graph(root: Path) -> ImportGraph:
    """Parse the covered trees into an `ImportGraph`. Never imports them."""
    files, unreadable = _graph_files(root)
    basenames = _repo_basenames(files)
    imports: dict[str, set[str]] = {}
    prefixes: dict[str, set[str]] = {}
    literals_by_file: dict[str, frozenset[str]] = {}
    scans_by_file: dict[str, tuple[_Scan, ...]] = {}
    assignments_by_file: dict[str, Mapping[str, ast.expr]] = {}
    unnamed: list[str] = []
    for rel in files:
        try:
            source = (root / rel).read_text(encoding="utf-8")
            refs = _references(rel, source)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            # Fail open: a file the graph cannot read could import anything, so
            # the selection stops being sound and the caller runs the full gate.
            unreadable.append(f"{rel} ({type(exc).__name__})")
            continue
        named = set(refs.modules)
        for base, alias in refs.submodules:
            # `from scripts import tool` names a SUBMODULE when `scripts` is a
            # package — including a namespace package, which is how this repo's
            # `scripts/` and `tests/` trees work (no `__init__.py`, importable
            # because they are on sys.path). Only a package can hold a submodule,
            # though: `from x import name` where `x` is a MODULE names an
            # attribute of it, and treating that as a module invents edges (on a
            # case-insensitive filesystem `from .alpha import ALPHA` resolves
            # `local_operator.ALPHA` onto `local_operator/alpha.py`).
            base_dir = root.joinpath(*base.split(".")) if base else root
            if base_dir.is_dir():
                named.add(f"{base}.{alias}" if base else alias)
        if refs.prefixes or refs.unnamed:
            # A file that imports by name is the one case where a dotted string
            # constant is evidence of a dependency rather than prose or data.
            named |= set(refs.literals)
        targets = {
            target
            for name in named
            if (target := _resolve_module(root, name)) is not None and target != rel
        }
        imports[rel] = targets
        literals_by_file[rel] = refs.literals
        scans_by_file[rel] = refs.scans
        assignments_by_file[rel] = refs.assignments
        for prefix in refs.prefixes:
            prefixes.setdefault(prefix, set()).add(rel)
        unnamed.extend(refs.unnamed)
    importers: dict[str, set[str]] = {}
    for rel, targets in imports.items():
        for target in targets:
            importers.setdefault(target, set()).add(rel)
    # Name edges (see `ImportGraph.referrers`): a file whose string constants
    # name another covered file can reach it at run time with no import to find.
    # Every covered file's literals are resolved, not just the importing ones —
    # a test naming `scripts/visual_gallery.py` imports nothing, so an
    # imports-only graph has no edge at all for the file it executes.
    referrers: dict[str, set[str]] = {}
    #: A name the parser SAW and could not place is recorded, not ignored. Price it
    #: first and it is the four rounds of #1322 all over again; record it and the
    #: test jobs run whole-tree while it exists. Two shapes count here, both of them
    #: "computed path" classes from that decision: a path-shaped constant that
    #: resolves to no covered file (`ROOT / name`, `f"{dir}/x.py"`, a spelling built
    #: at run time), and a dotted name that looks like it points INTO this repository
    #: (`local_operator.…`, `tests.…`, `scripts.…` — hence `_REPO_PACKAGE_HEADS`)
    #: yet resolves to no covered module. A dotted name outside those heads is not
    #: recorded: it cannot name a file this graph covers, so it cannot make a
    #: selection incomplete.
    unresolved_names: list[str] = []
    for rel, literals in literals_by_file.items():
        for literal in literals:
            targets = _name_target(root, literal, basenames)
            for target in targets:
                if target != rel:
                    referrers.setdefault(target, set()).add(rel)
            if targets or not _looks_like_a_repo_name(literal):
                continue
            if any(_name_target(root, head, basenames) for head in _dotted_prefixes(literal)):
                # `mock.patch("local_operator.api.Client.method")`: the name is a
                # module plus attributes, and the MODULE resolves. Not a hole.
                continue
            unresolved_names.append(f"{rel}: the name {literal!r} resolves to no covered file")
    unnamed.extend(unresolved_names)
    # Directory scans are the same obligation by a third road (see `_Scan`): a file
    # that globs a covered directory READS every covered file it matches and names
    # none of them, so when one of those files changes, the scanner is a referrer
    # of it. A scan this graph cannot place is NOT dropped quietly — if it spells
    # `.py` it joins the printed limit list, because the alternative (treating every
    # unplaceable scan as a reader of the whole program) was measured to select 654
    # of 724 tests for ANY change and take the scoping win to zero.
    unplaced_scans: list[str] = []
    for rel, scans in scans_by_file.items():
        assignments = assignments_by_file.get(rel, {})
        for scan in scans:
            targets, unplaced = _scan_targets(scan, rel, root, files, assignments)
            for target in targets:
                referrers.setdefault(target, set()).add(rel)
            if not unplaced:
                continue
            # NOT an edge, and NOT "armed if it spells `.py`": a scan whose
            # directory the graph cannot place is a reference the resolver does not
            # resolve, so it is RECORDED and the test jobs run whole-tree while it
            # exists (`_test_barriers`). Arming only the `.py` subset was the
            # previous policy, and rounds 3 and 4 each found a false green through
            # the cases it left out — a variable-held receiver without the hop, then
            # `iterdir()` and multi-arg `Path()`.
            unplaced_scans.append(f"{rel}:{scan.lineno}: a `{scan.pattern}` directory scan")
    return ImportGraph(
        files=frozenset(files),
        imports={rel: frozenset(targets) for rel, targets in sorted(imports.items())},
        importers={rel: frozenset(sources) for rel, sources in sorted(importers.items())},
        prefixes={prefix: tuple(sorted(hubs)) for prefix, hubs in sorted(prefixes.items())},
        referrers={rel: frozenset(sources) for rel, sources in sorted(referrers.items())},
        literals={rel: literals for rel, literals in sorted(literals_by_file.items())},
        boot=frozenset(BOOT_ROOTS)
        | _closure_over(
            {rel: frozenset(targets) for rel, targets in imports.items()},
            [rel for rel in BOOT_ROOTS if rel in files],
        ),
        unplaced_scans=tuple(sorted(unplaced_scans)),
        unnamed=tuple(sorted(unnamed)),
        unreadable=tuple(sorted(unreadable)),
    )


def test_weights(root: Path) -> tuple[dict[str, float], float] | None:
    """Per-file measured seconds from `tests/durations.json`, or None.

    The manifest is relative cost, not a contract: its own `_comment` says so.
    Unreadable or reshaped, the caller falls back to the file arm measured at
    the WEIGHT fraction, which is the conservative direction.
    """
    try:
        data = json.loads((root / DURATIONS_PATH).read_text(encoding="utf-8"))
        durations = data["durations"]
        fallback = float(data["fallback_seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(durations, dict):
        return None
    return ({str(k): float(v) for k, v in durations.items()}, fallback)


def _barrier_reason(rel: str, root: Path) -> str | None:
    """Why one changed path stops the narrowing, or None when it does not.

    The shape is a whitelist, not a denylist: a path narrows the gates only if
    it is a `.py` file the import graph covers, or documentation no gate reads.
    Everything else is a barrier, so a tree nobody thought about (package data,
    a fixture, a tree the suite reads by PATH such as `extension/`) fails CLOSED
    and runs the full command. Inverting this is how a narrowed run goes green
    on the file that mattered.
    """
    name = Path(rel).name
    for prefix, reason in STRUCTURAL_PREFIXES:
        if rel.startswith(prefix):
            return reason
    if rel in STRUCTURAL_PATHS:
        return "gate or manifest config — it changes what the gates do"
    if name == "conftest.py":
        return "a conftest decides collection and fixtures for a subtree"
    if name == "__init__.py":
        return "a package __init__ — module surface, and no import edge names it"
    if rel in ENTRY_POINT_PATHS:
        return "an entry point — every test that boots the app inherits it"
    if rel.endswith(".pyi"):
        return "a stub file — the graph parses .py, so nothing maps to it"
    if rel.endswith(LINT_SUFFIXES):
        if not (root / rel).is_file():
            return "a deleted module — its importers no longer resolve"
        if not rel.startswith(tuple(f"{tree}/" for tree in PARSED_TREES)):
            return "Python outside the trees the import graph covers"
        return None
    if rel.endswith(".md") and not rel.startswith(("local_operator/", "tests/")):
        # Documentation, and the classifier already treats it as inert for the
        # same reason (no gate reads it). A `.md` UNDER those two trees is
        # package or test data instead, and falls through to the barrier below.
        return None
    return "outside the import graph, and not documentation — read by path, not by import"


def scope_barriers(paths: Sequence[str], root: Path) -> list[str]:
    """Why this diff cannot be scoped, as printable reasons (one per path).

    Every reason names the path that produced it: "a conftest changed" is not
    actionable, `tests/unit/tui/conftest.py: a conftest…` is.
    """
    reasons: list[str] = []
    for rel in sorted({_norm(path) for path in paths}):
        reason = _barrier_reason(rel, root)
        if reason is not None:
            reasons.append(f"{rel}: {reason}")
    return reasons


def _narrow(command: str, marker: str, targets: Sequence[str]) -> str:
    """Replace one scripted input of `command` with a file list.

    The command comes from `JOB_COMMANDS`, so there is no second spelling of a
    gate here — only its scripted input (`.` for the lint tools, `tests/unit`
    for pytest) is narrowed. A command that no longer contains the marker means
    `JOB_COMMANDS` moved and this module would narrow the wrong thing, so it
    raises instead of guessing; the caller turns that into a full run.
    """
    tokens = shlex.split(command)
    if marker not in tokens:
        raise ScopeError(f"{command!r} has no {marker!r} input to narrow")
    # EVERY occurrence, not just the first: the scripted input is what a job runs
    # the tool over, and replacing one of two would leave a stray whole-tree input
    # inside a command that reports itself as scoped. The marker is matched as a
    # whole token, so `.` inside `--pythonpath .` is untouched.
    narrowed: list[str] = []
    for token in tokens:
        narrowed.extend(targets if token == marker else (token,))
    return shlex.join(narrowed)


def _narrow_where_possible(
    commands: Sequence[str], marker: str, targets: Sequence[str]
) -> tuple[tuple[str, ...], list[str]]:
    """Narrow every command of a job that HAS `marker`, keep the rest.

    A job can carry a step with no scripted input to narrow: `type-check`'s
    browser-extension protocol sync reads the whole extension surface and takes
    no file list, and it is cheap, so it runs unchanged — and the returned note
    says so, because a step that quietly did not narrow is the same class of
    surprise as a gate that quietly did not run. A job where NO command has the
    marker means `JOB_COMMANDS` moved and raises, so the caller falls back to the
    whole-tree command rather than narrowing something else.
    """
    narrowed: list[str] = []
    kept: list[str] = []
    for command in commands:
        if marker in shlex.split(command):
            narrowed.append(_narrow(command, marker, targets))
        else:
            narrowed.append(command)
            kept.append(command)
    if not kept and len(narrowed) == len(commands):
        return tuple(narrowed), []
    if len(narrowed) == len(kept):
        raise ScopeError(f"no command of {list(commands)!r} has a {marker!r} input to narrow")
    notes = [
        "this job also runs "
        + f"{len(kept)} step(s) with no file input to narrow, unchanged: "
        + "; ".join(kept)
    ]
    return tuple(narrowed), notes


@dataclass(frozen=True)
class ScopeDecision:
    """What one job will actually run, and why it is what it is."""

    job: str
    commands: tuple[str, ...]
    targets: tuple[str, ...]
    whole_tree: bool
    notes: tuple[str, ...]

    def report(self) -> str:
        """One legible line, so a narrow local run is never a quiet one."""
        if not self.whole_tree and not self.commands:
            return f"- `{self.job}`: nothing to run — {'; '.join(self.notes)}"
        if self.whole_tree:
            # "armed out" is the word for this: the job RAN its whole-tree command
            # because a barrier did not clear, which is a different fact from the
            # classifier never selecting the job (those are absent from this list,
            # and the flag row above says why). One word, two different reader
            # questions answered (#1322 QA round 1, Q-4).
            return f"- `{self.job}`: whole tree (armed out) — {'; '.join(self.notes)}"
        detail = "; ".join(self.notes) if self.notes else ""
        line = f"- `{self.job}`: scoped to {len(self.targets)} file(s)"
        return f"{line} — {detail}" if detail else line


def _fraction_reasons(
    selected: Sequence[str],
    universe: Sequence[str],
    weights: tuple[dict[str, float], float] | None,
) -> list[str]:
    """The fraction arms, each named when it fires.

    Weight first: this suite's cost is not spread evenly over its files
    (`tests/unit/tui` is 82.3% of the measured weight and 25% of the files), so
    a file arm alone waves through selections as expensive as the whole run.
    With no readable manifest the WEIGHT fraction is applied to the file arm,
    which is the conservative direction — a scoped `make check-changed` may not
    become the expensive thing it was scoping away from.
    """
    reasons: list[str] = []
    if not universe:
        return ["no test file is present in that tree in this checkout"]
    file_fraction = len(selected) / len(universe)
    file_arm = SCOPE_MAX_FILE_FRACTION if weights else SCOPE_MAX_WEIGHT_FRACTION
    if file_fraction > file_arm:
        reasons.append(
            f"the selection is {len(selected)} of {len(universe)} files "
            f"({file_fraction:.0%}), above the {file_arm:.0%} file arm"
        )
    if weights is None:
        reasons.append(
            f"{DURATIONS_PATH} is unreadable, so the weight arm could not be "
            "evaluated and the file arm was held to the weight fraction"
        )
        return reasons
    durations, fallback = weights
    total = sum(durations.get(path, fallback) for path in universe)
    chosen = sum(durations.get(path, fallback) for path in selected)
    if total > 0:
        weight_fraction = chosen / total
        if weight_fraction > SCOPE_MAX_WEIGHT_FRACTION:
            reasons.append(
                f"the selection is {chosen / 60:.0f} of {total / 60:.0f} measured "
                f"test-minutes ({weight_fraction:.0%}), above the "
                f"{SCOPE_MAX_WEIGHT_FRACTION:.0%} weight arm"
            )
    return reasons


#: pytest's own default for `python_files`, used when the config says nothing.
_PYTEST_DEFAULT_PYTHON_FILES: tuple[str, ...] = ("test_*.py", "*_test.py")


def _pytest_python_files(root: Path) -> tuple[str, ...]:
    """The `python_files` patterns the gate actually collects with.

    Read from `pyproject.toml` rather than assumed: the universe a selection is
    taken FROM has to be the set the gate collects, or a diff whose test file the
    assumed pattern does not match selects nothing and runs green. pytest's own
    defaults are used when the config is absent or unreadable, which is what pytest
    itself falls back to and the WIDER of the two readings — a superset selects
    more, never less, so the failure is a larger run and not a smaller one.
    """
    try:
        import tomllib

        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError, ImportError):
        return _PYTEST_DEFAULT_PYTHON_FILES
    configured = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("python_files")
    if isinstance(configured, str):
        return (configured,)
    if isinstance(configured, list) and all(isinstance(item, str) for item in configured):
        return tuple(configured)
    return _PYTEST_DEFAULT_PYTHON_FILES


def _is_collected(rel: str, patterns: Sequence[str]) -> bool:
    """Whether pytest collects `rel`, matching the way pytest matches.

    A configured pattern may carry directory components, so the basename and the
    repo-relative path are both tried.
    """
    name = Path(rel).name
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, f"*/{pattern}")
        for pattern in patterns
    )


def _test_universe(
    graph: ImportGraph, tree: str, patterns: Sequence[str] | None = None
) -> tuple[str, ...]:
    """The test files a job could run: what pytest's `python_files` collects."""
    wanted = tuple(patterns) if patterns else _PYTEST_DEFAULT_PYTHON_FILES
    prefix = f"{tree}/"
    return tuple(
        rel for rel in sorted(graph.files) if rel.startswith(prefix) and _is_collected(rel, wanted)
    )


def _sample(items: Sequence[str], limit: int = 3) -> str:
    """A few names plus a count, for a printed reason that stays one line."""
    listed = ", ".join(items[:limit])
    return f"{listed}, +{len(items) - limit} more" if len(items) > limit else listed


def _test_barriers(
    graph: ImportGraph, changed: Sequence[str], universe: Sequence[str]
) -> list[str]:
    """Why a TEST selection cannot be proven complete, or [] when it can.

    Two barriers, both unconditional. Neither decides whether the thing it names
    could actually reach a changed file: that decision is the shape enumeration
    that produced four rounds of false greens here, and the tool cannot make it
    soundly (#1322, decision 2026-09-19). A barrier that cannot be proven
    unnecessary runs the whole-tree command instead.

    The third barrier — a changed path the whitelist or the graph cannot account
    for — is `_barrier_reason`/`_graph_decision`, which run before this one.
    """
    reasons: list[str] = []
    boot = sorted(rel for rel in changed if rel in graph.boot)
    if boot:
        reasons.append(
            f"{len(boot)} changed file(s) are inside the app boot closure "
            f"({_sample(boot)}): every test that imports the app reaches them, and no "
            "static reference analysis can bound that"
        )
    unresolved = graph.unresolved_sites()
    if unresolved:
        reasons.append(
            f"the graph carries {len(unresolved)} reference(s) it cannot place — "
            f"{len(graph.unplaced_scans)} directory scans + {len(graph.unnamed)} "
            f"computed names, at {_sample([site.split(': ', 1)[0] for site in unresolved])} — "
            f"so no selection of {len(universe)} test file(s) can be proven complete"
        )
    return reasons


def _select_tests(
    graph: ImportGraph, seeds: Iterable[str], universe: Sequence[str]
) -> tuple[tuple[str, ...], list[str]]:
    """Tests a change can reach, plus reasons the seed set may be incomplete.

    `seeds` are changed files the graph covers — `scope_barriers` has already
    sent every path it cannot place down the full-run path, so a seed is never
    something the graph failed to see.
    """
    notes: list[str] = []
    reachable_seeds = set(seeds)
    in_universe = set(universe)
    conftest_namers: set[str] = set()
    for rel in sorted(reachable_seeds):
        hubs = graph.hubs_for(rel)
        if hubs:
            reachable_seeds.update(hubs)
            notes.append(
                f"{rel} is also loadable by name from {', '.join(hubs)}, so "
                "everything exercising those is selected"
            )
        named_by = graph.name_referrers(rel)
        if named_by:
            reachable_seeds.update(named_by)
            listed = ", ".join(named_by[:3]) + (" …" if len(named_by) > 3 else "")
            notes.append(
                f"{rel} is also named by path or module string, or read by a "
                f"scan, in {len(named_by)} file(s) ({listed}), so those are "
                "selected too"
            )
            # A conftest that names the file is a namer pytest will RUN, but it is
            # not in the test universe and nothing imports it, so seeding on it
            # selected NOTHING — the same silent false green the name edge exists
            # to close, one level up (#1322 round 2, MINOR 2). pytest's own scope
            # says what the right answer is: a conftest applies to every test under
            # its directory, so that subtree is selected.
            conftest_namers.update(
                referrer
                for referrer in named_by
                if Path(referrer).name == "conftest.py" and referrer not in in_universe
            )
    # The same obligation by the other road: a conftest that IMPORTS the changed
    # module (or one that imports something which does) does reach `dependents()`,
    # but the tests it GOVERNS do not — they are not pytest-visible dependents of
    # anything, because a conftest is not in the test universe and pytest applies
    # it by DIRECTORY. Left alone that selection is empty and the run is green while
    # the conftest's own fixtures fail (#1322 review round 1, F.1: a MAJOR for
    # exactly that reason — it is reachable the moment no unresolved site is left).
    conftest_importers = {
        rel
        for rel in graph.dependents(reachable_seeds)
        if Path(rel).name == "conftest.py" and rel not in in_universe
    }
    conftest_tests: set[str] = set()
    for conftest in sorted(conftest_namers | conftest_importers):
        parent = Path(conftest).parent.as_posix()
        # The repository-root conftest has parent `.`, and applies to the lot.
        scoped = (
            set(universe)
            if parent == "."
            else {rel for rel in universe if rel.startswith(f"{parent}/")}
        )
        conftest_tests |= scoped
        notes.append(
            f"{conftest} {'names' if conftest in conftest_namers else 'imports'} it, "
            "and pytest runs that conftest for every test "
            f"under {parent}, so those {len(scoped)} test file(s) are selected"
        )
    reachable = set(graph.dependents(reachable_seeds)) | reachable_seeds | conftest_tests
    selected = tuple(rel for rel in universe if rel in reachable)
    return selected, notes


@dataclass(frozen=True)
class ScopePlan:
    """The scoped plan for one local run: a decision per job, plus its limits."""

    decisions: Mapping[str, ScopeDecision]
    unresolved: tuple[str, ...] = ()
    graph_files: int = 0
    graph_seconds: float = 0.0

    def commands(self) -> dict[str, tuple[str, ...]]:
        """What `run_jobs` should execute, job -> commands."""
        return {job: decision.commands for job, decision in self.decisions.items()}

    def report(self) -> list[str]:
        """The printable scope section, in a deterministic order."""
        lines = ["", "### Local scope (file-level; CI still runs the full gate)"]
        if self.graph_files:
            # The graph is a real cost — every covered file is read and parsed —
            # so it is printed rather than hidden. It is the one number a reader
            # needs to see WHY a scoped run was not instant.
            lines.append(
                f"- graph: {self.graph_files} file(s) read and parsed in "
                f"{self.graph_seconds:.1f}s (nothing was imported)"
            )
        for job in sorted(self.decisions):
            lines.append(self.decisions[job].report())
        if self.decisions and not any(
            not decision.whole_tree for decision in self.decisions.values()
        ):
            lines.append(
                "- every job here runs its whole-tree command (each line says why); a job "
                "the classifier did not select is not in this list at all"
            )
        test_jobs = ("test", "tui-e2e")
        narrowed_tests = any(
            decision.job in test_jobs and not decision.whole_tree
            for decision in self.decisions.values()
        )
        if narrowed_tests:
            # A narrowed TEST selection is the one place this section still has a
            # limit to state: the reference model reads imports, literal path and
            # module names, directory scans and their arguments — nothing else. It
            # cannot see `getattr`, `eval`/`exec`, or a path assembled from data at
            # run time, so a scoped run is evidence about the files it ran and never
            # about the tree. Said here rather than implied, because the four rounds
            # that preceded this rule were all the same mistake in different shapes.
            lines.append(
                "- narrowed by a reference model that reads imports, literal path and "
                "module names, and directory scans; it cannot see dynamic attribute "
                "access, `eval`/`exec`, or a path built from data at run time — CI "
                "remains the authoritative run"
            )
        return lines


def scope_plan(jobs: Sequence[str], paths: Sequence[str], root: Path) -> ScopePlan:
    """Decide, per job, whether the local command narrows and what it runs.

    Never raises: every path out of this function ends in a runnable plan, and
    "I could not be confident" is expressed as the full command plus a note
    naming the trigger (the module's fail-open rule, one level down).

    Lint is decided BEFORE the import graph is built, because it needs no graph:
    a diff that breaks the graph should still get its changed files linted
    rather than fall back to a whole-tree lint for no reason.
    """
    normalized = [_norm(path) for path in paths]
    barriers = scope_barriers(paths, root)
    # Read once, from the config the gate itself uses (F.2): the universe a
    # selection is taken from has to be the set pytest will collect.
    collected = _pytest_python_files(root)
    weights = test_weights(root)
    decisions: dict[str, ScopeDecision] = {}
    graph: ImportGraph | None = None
    graph_reasons: list[str] = []
    graph_seconds = 0.0

    def _graph_decision(job: str) -> ScopeDecision | None:
        """Whole-tree decision for a graph-dependent job, or None to continue."""
        nonlocal graph, graph_seconds
        if graph is None:
            started = time.monotonic()
            graph = build_import_graph(root)
            graph_seconds = time.monotonic() - started
            if graph.unreadable:
                graph_reasons.append(
                    "the import graph could not be built: " + ", ".join(graph.unreadable)
                )
        if graph_reasons:
            return ScopeDecision(job, JOB_COMMANDS[job], (), True, tuple(graph_reasons))
        return None

    for job in jobs:
        commands = JOB_COMMANDS[job]
        if job in UNSCOPED_JOBS:
            decisions[job] = ScopeDecision(job, commands, (), True, (UNSCOPED_JOBS[job],))
            continue
        if job not in SCOPED_JOBS:
            decisions[job] = ScopeDecision(
                job, commands, (), True, ("not a job this module narrows",)
            )
            continue
        if barriers:
            decisions[job] = ScopeDecision(job, commands, (), True, tuple(barriers))
            continue
        if job == "lint":
            # From the diff, not from the graph: a changed `.pyi` is a lint input
            # the graph does not parse, and `scope_barriers` has already sent
            # anything it cannot place down the full-run path.
            targets = tuple(
                sorted(
                    rel
                    for rel in normalized
                    if rel.endswith(LINT_SUFFIXES) and (root / rel).is_file()
                )
            )
            if not targets:
                decisions[job] = ScopeDecision(
                    job, (), (), False, ("no changed file is one of the tools' inputs",)
                )
                continue
            try:
                marker = SCOPE_MARKERS[job]
                narrowed = tuple(_narrow(command, marker, targets) for command in commands)
            except ScopeError as exc:
                decisions[job] = ScopeDecision(job, commands, (), True, (str(exc),))
                continue
            decisions[job] = ScopeDecision(job, narrowed, targets, False, ())
            continue
        if job == "type-check":
            # pyright is named the changed files plus their transitive reverse
            # dependents. Measured: a file-list pyright reports diagnostics ONLY
            # for the files it is given (an error in an imported but unlisted
            # module is not reported), while a signature change in a listed
            # file's dependency IS reported in the listed file — so this list is
            # complete for "what the change can break", and it is also what the
            # run costs, which is what the closure arm bounds.
            whole_tree = _graph_decision(job)
            if whole_tree is not None:
                decisions[job] = whole_tree
                continue
            assert graph is not None
            seeds = [rel for rel in normalized if rel in graph.files]
            boot = sorted(rel for rel in seeds if rel in graph.boot)
            if boot:
                # A boot-closure file's dependents are the application, so naming
                # them is the whole-tree command with extra steps — and the closure
                # arm below would fire on the size alone. It is stated as its own
                # barrier because that is WHAT it is: the class is unbounded by
                # construction, not merely large (#1322, decision item 1).
                decisions[job] = ScopeDecision(
                    job,
                    commands,
                    (),
                    True,
                    (
                        f"{len(boot)} changed file(s) are inside the app boot closure "
                        f"({_sample(boot)}): every file that imports the app depends on "
                        "them, and no static reference analysis can bound that",
                    ),
                )
                continue
            targets = sorted(set(seeds) | set(graph.dependents(seeds)))
            if not targets:
                decisions[job] = ScopeDecision(
                    job, (), (), False, ("no changed Python file to type-check",)
                )
                continue
            closure = set(targets) | set(graph.forward_closure(targets))
            fraction = len(closure) / max(len(graph.files), 1)
            if fraction > SCOPE_MAX_CLOSURE_FRACTION:
                decisions[job] = ScopeDecision(
                    job,
                    commands,
                    (),
                    True,
                    (
                        f"naming {len(targets)} file(s) would pull {len(closure)} of "
                        f"{len(graph.files)} files into the analysis ({fraction:.0%}), "
                        f"above the {SCOPE_MAX_CLOSURE_FRACTION:.0%} closure arm — the "
                        "whole-tree command with extra steps",
                    ),
                )
                continue
            try:
                narrowed, notes = _narrow_where_possible(commands, SCOPE_MARKERS[job], targets)
            except ScopeError as exc:
                decisions[job] = ScopeDecision(job, commands, (), True, (str(exc),))
                continue
            decisions[job] = ScopeDecision(job, narrowed, tuple(targets), False, tuple(notes))
            continue
        whole_tree = _graph_decision(job)
        if whole_tree is not None:
            decisions[job] = whole_tree
            continue
        assert graph is not None
        seeds = [rel for rel in normalized if rel in graph.files]
        tree = SCOPE_MARKERS[job]
        universe = _test_universe(graph, tree, collected)
        barriers_here = _test_barriers(graph, seeds, universe)
        if barriers_here:
            decisions[job] = ScopeDecision(job, commands, (), True, tuple(barriers_here))
            continue
        selected, notes = _select_tests(graph, seeds, universe)
        reasons = _fraction_reasons(selected, universe, weights)
        if reasons:
            decisions[job] = ScopeDecision(job, commands, (), True, tuple(reasons))
            continue
        if not selected:
            decisions[job] = ScopeDecision(
                job,
                (),
                (),
                False,
                (
                    f"no file under {tree}/ transitively imports anything this diff "
                    f"changed ({len(seeds)} changed file(s) in the graph)",
                ),
            )
            continue
        try:
            narrowed = tuple(_narrow(command, tree, selected) for command in commands)
        except ScopeError as exc:
            decisions[job] = ScopeDecision(job, commands, (), True, (str(exc),))
            continue
        durations, fallback = weights if weights else ({}, 0.0)
        total = sum(durations.get(path, fallback) for path in universe)
        chosen = sum(durations.get(path, fallback) for path in selected)
        share = f", {chosen / total:.0%} of the measured weight" if total > 0 else ""
        decisions[job] = ScopeDecision(
            job,
            narrowed,
            selected,
            False,
            (f"{len(selected)} of {len(universe)} test files{share}", *notes),
        )
    unresolved = graph.unresolved_sites() if graph is not None and not graph_reasons else ()
    return ScopePlan(
        decisions=decisions,
        unresolved=unresolved,
        graph_files=len(graph.files) if graph is not None else 0,
        graph_seconds=graph_seconds,
    )


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


def run_jobs(
    jobs: Sequence[str], root: Path, commands: Mapping[str, Sequence[str]] | None = None
) -> int:
    """Run each selected job's local commands in order; return a shell status.

    `commands` is the scoped plan when the caller has one (see `scope_plan`);
    without it every job runs its whole-tree `JOB_COMMANDS` entry, which is what CI
    does and what `--no-scope` asks for. The scoping decision and the run are
    separate arguments on purpose: `--dry-run` prints the plan and never builds one,
    and a reviewer reading a narrow run needs the plan that produced it.

    A command that two selected jobs SHARE runs once. The `xplat-probe-*` legs
    are the case that made this explicit: they differ only by the OS leg CI
    supplies, so they carry the same local spelling, and a local run has exactly
    one host — running the battery twice would double a ~4-minute gate to answer
    the question it already answered. The skipped repeat is printed with the job
    that ran it, because a selected job that prints nothing reads as a job that
    ran nothing.
    """
    failures: list[str] = []
    already_run: dict[str, str] = {}
    for job in jobs:
        for command in (commands or JOB_COMMANDS)[job]:
            if command in already_run:
                print(
                    f"\n=== {job}: {command}\n"
                    f"    (not re-run: {already_run[command]} already ran this exact "
                    "command in this invocation)"
                )
                continue
            already_run[command] = job
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
    if not jobs:
        # Distinct from the branch below: nothing was SELECTED here, which is
        # what a clean tree produces, and saying "narrowed to no file" for it
        # tells the reader a decision was made that never was (QA Q-1 on #1322).
        print("no job was selected for this diff, so nothing ran")
        return 0
    if not any((commands or JOB_COMMANDS)[job] for job in jobs):
        print("nothing to run: every selected job was narrowed to no file")
        return 0
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
        "--dry-run",
        action="store_true",
        help=(
            "print the scope report and the job list without executing anything. "
            "The same code path as `--run`, stopped one step earlier."
        ),
    )
    parser.add_argument(
        "--no-scope",
        action="store_true",
        help=(
            "run every selected job's whole-tree command, exactly as CI spells "
            "it: the escape hatch when a narrowed selection is not trusted, and "
            "the A/B control for measuring what scoping costs."
        ),
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
    # Bound only when the classification branch below actually asks git for a
    # diff, so `--all`, a missing base and a failed `git diff` stay distinguishable
    # from "the diff is empty".
    collected: list[str] | None = None

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

    if args.run or args.dry_run:
        plan = job_plan(flags)
        selected = [
            job for job, verdict in plan.items() if verdict == "run" and job in JOB_COMMANDS
        ]
        scope = None
        if args.no_scope:
            print(
                "\n### Local scope\n- `--no-scope`: every selected job runs its "
                "whole-tree command"
            )
        elif paths:
            scope = scope_plan(selected, paths, root)
            for line in scope.report():
                print(line)
        elif collected is None:
            # No diff was available to scope by: `--all`, a base that could not be
            # resolved, or a `git diff` that failed. Classification already failed
            # open to "run everything" for each of those, and the file-level layer
            # must not turn that into "scope to nothing".
            print(
                "\n### Local scope\n- no diff is available to scope by (`--all`, an "
                "unclassifiable base, or a failed `git diff`), so every selected job "
                "runs its whole-tree command"
            )
        else:
            # An EMPTY diff — a clean tree, or a base equal to HEAD. Distinguishing
            # this from `collected is None` is the point: the base revision told the
            # reader that whole-tree commands would run, then ran none, and reported
            # `nothing to run` about a decision it never made (QA Q-1 on #1322).
            print(
                f"\n### Local scope\n- the diff is empty (no change against "
                f"{base_label}), so no job has anything to run"
            )
        for job, reason in sorted(LOCAL_EXCLUSIONS.items()):
            print(f"skipped locally: {job} — {reason}")
        if args.dry_run:
            print("dry run: no gate was executed")
            return 0
        return run_jobs(selected, root, scope.commands() if scope else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
