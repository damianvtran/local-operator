"""Guards for CI topology and local-gate invocation.

These exist because three defects all produced a *silent* green:

- #428: ``tui-e2e`` needed ``test``, so a flaky unit suite skipped the freeze
  guard (observed on PR #426, which *fixed* a deadlock while the deadlock
  job reported ``skipping``).
- #423: a stale ``.venv/bin/black`` shebang exits 126, and ``cmd | tail``
  reports ``tail``'s 0, so a lint gate that never ran looks passing.
- #381: ``[tool.pyright] exclude`` *replaces* pyright's built-in defaults.
  Dropping ``**/.*`` makes a local run type-check all of site-packages.
- #1238: the whole job set ran on every diff, so a one-line docs edit paid for
  the six-shard matrix, a Windows run, a macOS TUI run, a live-LLM run and a
  dependency audit — 16 checks for a file no gate reads. The gate is a
  classifier (`scripts/ci_scope.py`) shared with `make check-changed`; the
  assertions below pin its predicates, its fail-open direction, and the wiring
  in `ci.yml` that reads it.
- #1245 round 1: the classifier shipped with a default repository root derived
  from `__file__`, and CI runs a COPY of it from `$RUNNER_TEMP` — so the root was
  the checkout's PARENT, `git diff` exited 128, and the fail-open branch ran the
  full job set on every pull request behind a `::warning::`. Every assertion
  passed an explicit `--root`, so none of them saw it: correct code, dead gate,
  green suite. The tests here now drive the real invocation shape (module copy in
  a temp dir, a checkout as the working directory, no `--root`) and execute the
  step's own shell to check WHICH copy of the module runs.

A comment in ci.yml is not a test. Each assertion below is mutation-tested
against the defect it claims to catch.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import (
    check_serial_controls,
    ci_scope,
    merge_gate_advisory,
    run_bounded,
    shard_tests,
)

REPO = Path(__file__).resolve().parents[2]
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
#: The unit suites (`test`, `coverage-report`, `tui-e2e`) moved here from
#: ci.yml; it runs on `main` plus the merge gate's `labeled` pull request. See
#: `_ci_jobs` for how the pins follow them.
FULL_SUITE_YML = REPO / ".github" / "workflows" / "full-suite.yml"
#: Jobs that live in full-suite.yml, not ci.yml. `changes` is in both.
FULL_SUITE_JOBS = frozenset({"test", "coverage-report", "tui-e2e", "report", "merge-gate"})
PYPROJECT = REPO / "pyproject.toml"
MAKEFILE = REPO / "Makefile"

#: The classifier both CI and `make check-changed` call. Named here so the
#: drift assertions below can say what they mean without re-deriving it.
CI_SCOPE_REL = "scripts/ci_scope.py"

#: A version-only `pyproject.toml` diff, in the shape `git diff` produces it.
#: Used to pin the `release_bump` refinement (A8): the rule is the LINE PAIR,
#: not the filename.
VERSION_BUMP_DIFF = (
    "diff --git a/pyproject.toml b/pyproject.toml\n"
    "--- a/pyproject.toml\n"
    "+++ b/pyproject.toml\n"
    "@@ -4 +4 @@\n"
    '-version = "0.54.33"\n'
    '+version = "0.54.34"\n'
)

#: The negative case for the same rule: a `pyproject.toml` diff that is NOT a
#: bump, because it changes what gets installed.
DEPENDENCY_DIFF = (
    "diff --git a/pyproject.toml b/pyproject.toml\n"
    "--- a/pyproject.toml\n"
    "+++ b/pyproject.toml\n"
    "@@ -20 +20 @@\n"
    '-    "httpx>=0.27",\n'
    '+    "httpx>=0.28",\n'
)


def _ci_jobs() -> dict[str, Any]:
    """Every job the classifier models: ci.yml's plus full-suite.yml's.

    The classifier (`scripts/ci_scope.py`) still models the moved unit jobs —
    `make check-changed` runs them locally and full-suite.yml gates them with
    the same `changes` job — so every pin on them keeps reading the real
    definition, now from the file it lives in. `changes` exists in both files
    and must be identical (`test_full_suite_changes_job_matches_ci`), so
    taking ci.yml's copy loses nothing. Any OTHER id in both files is an
    ambiguity this union would hide, so it is refused.
    """
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    moved = yaml.safe_load(FULL_SUITE_YML.read_text())["jobs"]
    assert isinstance(jobs, dict) and isinstance(moved, dict)
    clash = (set(jobs) & set(moved)) - {"changes"}
    assert not clash, f"job ids defined in both ci.yml and full-suite.yml: {sorted(clash)}"
    return {**moved, **jobs}


def _pr_jobs() -> dict[str, Any]:
    """ci.yml's jobs alone: what a pull request actually runs."""
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _needs(job: str) -> set[str]:
    """`needs` of a CI job, as a set. A missing key is an empty set, which
    is a legitimate topology (no dependencies), not an error.

    `needs: test` (the scalar form GitHub also accepts) is normalised to
    `{"test"}`: `coverage-report` deliberately uses it, and a helper that
    asserted a list would turn a legitimate spelling into a test error.
    """
    declared = _ci_jobs()[job].get("needs") or []
    if isinstance(declared, str):
        return {declared}
    assert isinstance(declared, list)
    return set(declared)


def _steps(job: str) -> list[dict[str, Any]]:
    steps = _ci_jobs()[job]["steps"]
    assert isinstance(steps, list)
    return steps


def _if(job: str) -> str:
    """A job's `if:` expression, with the folded whitespace collapsed.

    `yaml.safe_load` folds a `>-` scalar onto one line, but collapsing here too
    means a future block-scalar style cannot break a substring assertion.
    """
    return " ".join(str(_ci_jobs()[job].get("if") or "").split())


def _job_evidence(job: str) -> str:
    """Everything a job invokes: its `run:` bodies, `uses:` actions and `with:`.

    `uses:` is included because one gated job (`pip-audit`) IS an action rather
    than a shell step, and the drift assertion is about *the same tool*, not
    about how the step happens to be spelled.
    """
    blobs: list[str] = []
    for step in _steps(job):
        blobs.append(str(step.get("run") or ""))
        blobs.append(str(step.get("uses") or ""))
        options = step.get("with")
        if isinstance(options, dict):
            blobs.append(" ".join(str(value) for value in options.values()))
    return "\n".join(blobs)


def _tool_key(text: str) -> str:
    """Normalise a tool name for comparison (`pip_audit` vs `pip-audit`).

    Applied to BOTH sides — the module's local command and the job's ci.yml
    evidence — so the normalisation cannot make a mismatch disappear.
    """
    return text.replace("_", "-")


def _makefile_recipes() -> list[str]:
    """The recipe lines of the Makefile (tab-indented, comments excluded)."""
    return [
        line.lstrip("\t")
        for line in MAKEFILE.read_text().splitlines()
        if line.startswith("\t") and not line.lstrip().startswith("#")
    ]


def _executed_program(command: str) -> str:
    """The first token a shell would exec for `command`, after any `env` prefix."""
    tokens = shlex.split(command)
    index = 0
    if tokens[:1] == ["env"]:
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in ("-u", "--unset"):
                # `env -u NO_COLOR`: the value is not the program.
                index += 2
            elif token.startswith("-") or "=" in token:
                index += 1
            else:
                break
    return tokens[index] if index < len(tokens) else ""


def _makefile_recipe_blocks() -> dict[str, str]:
    """target -> its recipe text (the tab-indented lines that follow it).

    Distinct from `_makefile_recipes`, which loses the association: an assertion
    that needs two facts of the SAME target (`--since` and `merge-base`, say)
    cannot get them from a flat list, and the flat-list form of exactly that
    assertion is what made A12's named mutation survive review (R4).
    """
    blocks: dict[str, str] = {}
    current: str | None = None
    for line in MAKEFILE.read_text().splitlines():
        if line.startswith("\t"):
            if current:
                blocks[current] += "\n" + line.lstrip("\t")
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+):", line)
        current = match.group(1) if match else None
        if current:
            blocks.setdefault(current, "")
    return blocks


def _changes_run() -> str:
    """The `run:` body of the `changes` job's classify step."""
    steps = [step for step in _steps("changes") if step.get("id") == "classify"]
    assert len(steps) == 1, "expected exactly one `classify` step in `changes`"
    return str(steps[0].get("run") or "")


def _git_run(repo: Path, *args: str) -> str:
    """Run git in a throwaway repository; identity/signing are pinned per call so
    a developer's global config cannot change the result."""
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.email=ci-scope@example.invalid",
            "-c",
            "user.name=ci-scope",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _make_repo(root: Path) -> Path:
    """A throwaway git repository, so a test can exercise the module in a real
    checkout without touching this one."""
    repo = root / "checkout"
    repo.mkdir(parents=True)
    _git_run(repo, "init", "-q")
    return repo


def _commit_all(repo: Path, message: str) -> str:
    _git_run(repo, "add", "-A")
    _git_run(repo, "commit", "-q", "-m", message)
    return _git_run(repo, "rev-parse", "HEAD").strip()


def _read_flags(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def test_tui_e2e_does_not_need_the_unit_suite_or_pip_audit() -> None:
    """The freeze guard must still run when `test` is red.

    `needs: test` is how GitHub Actions *skips* a job, not how it waits
    politely. A skipped freeze guard is indistinguishable from a passing
    one on the PR checks list. `pip-audit` has the same shape: a CVE
    published today would skip the macOS resume-liveness assertion on an
    unmodified tree. lint/type-check stay — they are cheap syntax gates
    and a broken install is not a freeze.

    `changes` was added by the scope-gating change (A15) and does not weaken
    either claim: it is the classifier, it always runs, and a job that `needs:`
    it is skipped only through the `tui` flag — whose predicate is deliberately
    equal to `unit`, so the guard cannot be narrowed away from the unit suite
    by this route. The re-stated defect is unchanged: `test` and `pip-audit`
    are absent, and adding either back fails here.
    """
    needs = _needs("tui-e2e")
    assert "test" not in needs, (
        "tui-e2e needs `test`, so a flaky unit suite skips the freeze "
        "guard (the #428 defect, observed on PR #426)"
    )
    assert "pip-audit" not in needs, (
        "tui-e2e needs `pip-audit`, so a newly-published CVE skips the "
        "freeze guard — the same latent disarm as `needs: test`"
    )
    # Only the classifier since the move to full-suite.yml: lint/type-check
    # are PR-only jobs, and every commit on `main` already passed them.
    assert needs == {"changes"}, (
        f"tui-e2e.needs={sorted(needs)!r}; expected only the classifier "
        "(full-suite.yml has no lint/type-check jobs to depend on)"
    )


def test_cli_and_server_sanity_still_wait_on_the_cheap_gates() -> None:
    """Live-LLM jobs are cost, not freeze guards; they keep the full needs.

    Dropping `test` from those too would be a different change. Pin the
    current contract so a drive-by edit of every `needs:` block at once
    cannot silently re-couple tui-e2e by copying the sanity list.

    `changes` is part of that contract after the scope-gating change (A15) and
    is the one addition that CANNOT silently disarm these jobs: it always runs,
    and the `cli`/`server` flags that gate them are equal to `audit`, which is
    what `pip-audit` is gated on — so a skipped audit cannot skip them (A4 is
    the assertion that keeps that implication true).
    """
    # `guards` took `test`'s place when the unit matrix moved to
    # full-suite.yml: same `unit` flag, still a failed-gate stop before spend.
    expected = {"changes", "lint", "type-check", "guards", "pip-audit"}
    for name in ("cli-sanity", "server-sanity"):
        assert _needs(name) == expected, (
            f"{name}.needs drifted; live-LLM jobs are supposed to keep "
            "the full cheap-gate list, unlike tui-e2e"
        )


def test_type_check_does_not_force_latest_pyright() -> None:
    """FORCE_VERSION=latest ignores the bundled analyzer.

    That is how CI drifted onto npm 1.1.413 while pip resolved 1.1.411
    and the lint job (which never ran pyright) pinned 1.1.408. The pin
    lives in the dev extra; bump that, not an env override.
    """
    job = _ci_jobs()["type-check"]
    assert isinstance(job, dict)
    env = job.get("env") or {}
    assert "PYRIGHT_PYTHON_FORCE_VERSION" not in env, (
        "type-check sets PYRIGHT_PYTHON_FORCE_VERSION, which downloads "
        "whatever npm currently calls latest instead of the pinned "
        f"analyzer (env={env!r})"
    )
    # The lint job used to install pyright==1.1.408 and never invoke it.
    # A third unused pin is how the versions diverged in the first place.
    lint_install = "\n".join(step.get("run", "") for step in _steps("lint"))
    assert "pyright" not in lint_install, (
        "lint job still installs pyright — it does not run it, and a "
        "second pin is how 1.1.408 / 1.1.411 / latest coexisted"
    )


def test_dev_extra_pins_the_same_pyright_the_type_check_job_installs() -> None:
    """One version across the extra, the local gate, and CI.

    An unpinned `pyright` in the extra is what let FORCE_VERSION=latest
    and a leftover lint-job pin silently disagree.
    """
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]["dev"]
    pins = [dep for dep in extras if dep.startswith("pyright")]
    assert pins == ["pyright==1.1.414"], (
        f"dev extra pyright pin drifted: {pins!r}. Bump this together "
        "with the exclude-superset guard, not by setting FORCE_VERSION."
    )


def test_configured_pyright_excludes_are_a_superset_of_the_running_defaults() -> None:
    """`exclude` replaces pyright's built-in defaults, it does not extend them.

    The running analyzer reports those defaults on `--verbose` as
    `Auto-excluding <pattern>` — that is a behavioural source, not a
    restatement of an upstream constant we would then have to keep in
    sync by hand. If a future pyright adds a fourth default, this test
    goes red until we restate it.

    The probe runs against an empty config so our restated `exclude`
    cannot mask a missing default: with `exclude` set, pyright does not
    auto-exclude anything (the replacement semantics this test exists
    to catch).
    """
    configured = set(tomllib.loads(PYPROJECT.read_text())["tool"]["pyright"]["exclude"])
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "pyrightconfig.json").write_text("{}")
        proc = subprocess.run(
            [sys.executable, "-m", "pyright", "--verbose", "."],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
    auto = {
        line.split("Auto-excluding ", 1)[1].strip()
        for line in (proc.stdout + proc.stderr).splitlines()
        if "Auto-excluding " in line
    }
    assert auto, (
        "pyright --verbose printed no Auto-excluding lines; the "
        "defaults probe cannot see drift if the analyzer stopped "
        f"reporting them (rc={proc.returncode}, stderr={proc.stderr!r})"
    )
    missing = sorted(auto - configured)
    assert not missing, (
        f"[tool.pyright] exclude is missing pyright's built-in "
        f"defaults {missing}. exclude REPLACES the defaults, so these "
        "patterns are currently type-checked. Restate them alongside "
        f"`docs`. configured={sorted(configured)}"
    )


def test_pipeline_does_not_swallow_a_126_from_a_broken_console_script() -> None:
    """`cmd | tail` reports tail's 0 even when cmd failed.

    That is the #423 mechanism: a stale shebang makes `.venv/bin/black`
    fail (macOS bash: 126 "bad interpreter"; Linux bash: 127 "required
    file not found"), and a gate script that pipes the output looks
    green because it reports `tail`'s 0. The documented gates
    (`python -m`, `uvx`) and `make lint`/`format`/`type-check` must
    not be that pipeline.

    Pinning rc==126 is how this assertion passed on macOS 3.14, then
    failed on CI 3.12 (PATH continuation found the real flake8, rc=0)
    and CI 3.13 (absolute path, rc=127). The defect is the swallow,
    not the errno. Accept any non-zero from the fake as "the script
    failed"; assert the pipeline then reports 0.

    The fake is a unique name invoked by absolute path. Linux
    bash/dash continue searching PATH after a bad shebang, so a fake
    named `flake8` on PATH is skipped and the real flake8 runs —
    which is not the defect.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "lop-stale-script"
        # A shebang pointing at a path that does not exist is how the
        # real console scripts fail after a worktree is deleted. The
        # body is unreachable on purpose.
        fake.write_text("#!/this/interpreter/does/not/exist\n")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

        # Direct execve of a missing-interpreter shebang raises
        # FileNotFoundError; a gate script sees the *shell*
        # translation. Drive bash explicitly so we are not at the
        # mercy of `/bin/sh` being dash vs bash.
        bash = ["/bin/bash", "-c"]
        direct = subprocess.run(
            [*bash, f"{fake} --version"],
            capture_output=True,
            text=True,
        )
        assert direct.returncode != 0, (
            "the fake succeeded at its own boundary "
            f"(rc={direct.returncode}, stderr={direct.stderr!r}); "
            "the rest of this test is not exercising the defect"
        )

        piped = subprocess.run(
            [*bash, f"{fake} --version | tail -1"],
            capture_output=True,
            text=True,
        )
        assert piped.returncode == 0, (
            "the #423 mechanism itself changed: a failing console "
            "script inside a pipeline no longer reports 0 "
            f"(direct rc={direct.returncode}, piped rc={piped.returncode}). "
            "If shells started honouring pipefail by default this "
            "test would need rewriting, but the Makefile still must "
            "not pipe."
        )

        # `python -m` looks up the module on sys.path; it never execs
        # a console-script shebang. Pointing PATH at the fake must
        # therefore not change the module invocation's rc.
        env = {**os.environ, "PATH": tmp + os.pathsep + os.environ.get("PATH", "")}
        via_module = subprocess.run(
            [sys.executable, "-m", "flake8", "--version"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert via_module.returncode == 0, (
            "`python -m flake8` consulted the stale PATH shebang "
            f"(rc={via_module.returncode}, stderr={via_module.stderr!r})"
        )


def test_makefile_quality_targets_do_not_invoke_console_scripts() -> None:
    """`make lint` used to be a bare `flake8`, which is the 126 path.

    After this change the quality targets go through `python -m` / `uvx`,
    matching AGENTS.md. A regression to `black .` / `flake8` / `pyright`
    re-exposes every agent on this machine to a swallowed 126.
    """
    recipes = _makefile_recipes()
    for tool in ("black", "flake8", "isort", "pyright"):
        # A recipe whose first token is the tool name is the console
        # script. `python -m flake8` and `uvx --from black==… black`
        # have a different first token.
        bare = [r for r in recipes if r == tool or r.startswith(tool + " ")]
        assert not bare, (
            f"Makefile still invokes the `{tool}` console script "
            f"({bare!r}); that is the #423 shebang path. Use "
            "`python -m` or `uvx`."
        )
        # `check-changed` (A11) reaches its gates through the classifier, so the
        # first-token rule above cannot see a `.venv/bin/<tool>` written into its
        # recipe. Assert the path directly: an absolute console-script path is
        # the same 126 trap whichever recipe spells it.
        absolute = [r for r in recipes if f".venv/bin/{tool}" in r]
        assert not absolute, (
            f"a Makefile recipe calls `.venv/bin/{tool}` ({absolute!r}); a "
            "stale shebang there exits 126, which a pipeline swallows"
        )


def test_tui_e2e_still_runs_on_macos() -> None:
    """The freeze is a macOS/BSD property; dropping the leg disarms the guard
    more thoroughly than `needs: test` ever did."""
    matrix = _ci_jobs()["tui-e2e"]["strategy"]["matrix"]["os"]
    assert isinstance(matrix, list)
    assert "macos-latest" in matrix, (
        "tui-e2e lost its macOS leg; the freeze this stage exists to "
        f"catch cannot go red on Linux (os={matrix!r})"
    )


def test_every_e2e_shard_runs_on_both_platforms() -> None:
    """Sharding the e2e tree must not narrow which platform runs it.

    The matrix is a product: `shard` decides which slice of the tree a leg
    runs, `os` decides where. A leg that exists on only one OS is a slice of
    the tree that the macOS-only freeze guard never sees — the #401 deadlock
    is a macOS/BSD property, so an ubuntu-only shard 2 would be exactly the
    test that cannot catch it, and a macos-only shard 2 would be untested on
    the platform where the rest of the stage is exercised.
    """
    matrix = _ci_jobs()["tui-e2e"]["strategy"]["matrix"]
    oses = matrix["os"]
    shards = matrix["shard"]
    assert sorted(oses) == ["macos-latest", "ubuntu-latest"], (
        "tui-e2e must run on exactly the macOS and ubuntu legs; "
        f"os={oses!r} would leave part of the freeze guard unrunnable"
    )
    assert shards, "tui-e2e has an empty shard matrix"


#: The trees CI shards, and the job whose matrix runs each one. Which tree a
#: job shards and how many shards it makes are READ from ci.yml (see
#: `_shard_plan`); this mapping only says which job owns which tree, and is
#: asserted to be exhaustive by `test_every_shard_matrix_job_uses_a_known_tree`.
SHARD_JOBS: dict[str, str] = {"test": "unit", "tui-e2e": "e2e"}

#: The job names the parametrized shard guards run against, sorted so a
#: failure names the job it is about in a stable order.
SHARD_JOB_IDS = sorted(SHARD_JOBS)


def _shard_step_run(job: str) -> str:
    """The `run:` body of `job`'s partition step, found by what it invokes.

    Found by the script it calls rather than by step NAME: a name is prose, so
    renaming a step would silently detach every assertion below from the thing
    it guards (the same mistake `_partition_step_run` used to make).
    """
    steps = [s for s in _steps(job) if "scripts/shard_tests.py" in (s.get("run") or "")]
    assert len(steps) == 1, f"{job} must have exactly one shard partition step, got {len(steps)}"
    return steps[0]["run"]


#: The unit job's SECOND pytest invocation, by step id: the serial run of the six
#: controls (see ``SERIAL_CONTROLS`` in ci.yml). Audited by name, not ignored — a
#: third pytest step, or this one under another id, fails ``_pytest_steps``.
SERIAL_STEP_ID = "serial_controls"
ORDINARY_STEP_ID = "shard_pytest"
#: tui-e2e's own pair. The Darwin evidence step is there because the sharded unit
#: job is ubuntu-only: the macOS leg is the only place a group census's treatment
#: of an unreaped zombie leader can be measured, and an unnamed extra pytest
#: invocation is exactly what this audit exists to refuse.
E2E_STEP_ID = "e2e_shard_pytest"
DARWIN_STEP_ID = "darwin_pin"


def _pytest_steps(job: str) -> list[dict[str, Any]]:
    """Every step of `job` that invokes pytest, with the job's allowance enforced.

    One step for every job but the two audited pairs: the unit job's ordinary shard
    run (`shard_pytest`) and serial controls (`serial_controls`), and tui-e2e's
    partitioned e2e run (`e2e_shard_pytest`) and Darwin evidence step
    (`darwin_pin`). The old rule ("exactly one pytest step") would reject either
    addition, and relaxing it to "at least one" would let an arbitrary extra
    command ride along unaudited; naming the ids keeps both honest.
    """
    steps = [s for s in _steps(job) if "pytest" in (s.get("run") or "")]
    if job == "test":
        ids = [s.get("id") for s in steps]
        assert ids == [ORDINARY_STEP_ID, SERIAL_STEP_ID], (
            f"{job} must run exactly its two audited pytest steps "
            f"({ORDINARY_STEP_ID}, {SERIAL_STEP_ID}) in that order, got {ids}"
        )
    elif job == "tui-e2e":
        ids = [s.get("id") for s in steps]
        assert ids == [E2E_STEP_ID, DARWIN_STEP_ID], (
            f"{job} must run exactly its two audited pytest steps "
            f"({E2E_STEP_ID}, {DARWIN_STEP_ID}) in that order, got {ids}"
        )
    else:
        assert len(steps) == 1, f"{job} must have exactly one pytest step, got {len(steps)}"
    return steps


def _run_step(job: str) -> str:
    """The `run:` body of `job`'s step that runs its SHARD of the tests.

    Found by the `pytest` invocation rather than by step name, for the same
    reason `_shard_step_run` is: the assertions built on it are about what the
    step EXECUTES (does it run the partitioned list, does it keep `-n0`), and a
    renamed step must not be able to detach them. For the unit job this is the
    ordinary run; the serial controls have their own audit below.
    """
    return _pytest_steps(job)[0]["run"]


#: A shell command substitution: `$(cat e2e_tests.txt | tr '\n' ' ')`. Replaced
#: by `_SUBSTITUTION_TOKEN` before the argv is split, because a naive split
#: cannot see the shell text inside as separate arguments and `shlex` would
#: raise on its quotes.
_SUBSTITUTION = re.compile(r"\$\([^)]*\)")
_SUBSTITUTION_TOKEN = "__substitution__"


def _pytest_argv(run: str) -> list[str]:
    """The arguments of a step's pytest invocation, substitutions included.

    `pytest $(cat list.txt) tests/e2e -q` therefore yields
    `['pytest', '__substitution__', 'tests/e2e', '-q']` -- the directory is
    visible AS an argument. That is the whole point: an assertion that looks
    for the substring `pytest tests/e2e` is evaded by appending the tree to a
    correct command (`pytest $(cat list.txt) tests/e2e`), which collects the
    list AND the whole tree, and by any reordering or extra whitespace.
    """
    return shlex.split(_SUBSTITUTION.sub(_SUBSTITUTION_TOKEN, run))


def _shard_plan(job: str) -> tuple[str, int]:
    """`(tree, total)` as `job` declares them, read from ci.yml."""
    run = _shard_step_run(job)
    total_match = re.search(r"--total\s+(\d+)", run)
    assert total_match, f"no --total in {job}'s partition step: {run!r}"
    tree_match = re.search(r"--tree\s+(\w+)", run)
    tree = tree_match.group(1) if tree_match else "unit"
    assert tree in shard_tests.TREES, f"{job} shards unknown tree {tree!r}"
    assert tree == SHARD_JOBS[job], (
        f"{job} shards tree {tree!r}, expected {SHARD_JOBS[job]!r}: the two jobs "
        "would otherwise shard the same tree to two different degrees"
    )
    return tree, int(total_match.group(1))


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_the_committed_manifest_belongs_to_the_tree_that_reads_it(job: str) -> None:
    """Every weight in a tree's manifest must name a file that tree collects.

    Coverage is deliberately NOT the property here -- a file may legitimately
    be missing, and `test_unmeasured_test_files_are_still_scheduled` pins that
    it runs anyway. The property is that no COMMITTED weight is dead. A
    manifest written for the other tree passes every other guard in this file:
    the partition finds 0 measured files, schedules everything at the fallback
    weight, reports a perfectly balanced split, and has discarded every real
    measurement it was given. `gen_test_durations.py --tree` refuses to write
    such a file (see the next test); this catches one that is already
    committed, including the stale-path case where a test file was renamed and
    its weight was never removed.
    """
    tree = SHARD_JOBS[job]
    files = set(shard_tests.collect_test_files(tree=tree))
    weights, fallback = shard_tests.load_weights(shard_tests.TREES[tree].manifest)

    assert weights, (
        f"{tree}'s manifest is empty or unreadable; every shard would then be "
        f"balanced by the {fallback}s fallback, which measures nothing"
    )
    assert fallback > 0, (
        f"{tree}'s manifest has a non-positive fallback ({fallback}); an "
        "unmeasured file would be treated as free and pile onto the first shard"
    )
    dead = sorted(set(weights) - files)
    assert not dead, (
        f"{tree}'s manifest weighs {len(dead)} file(s) the tree never collects, " f"e.g. {dead[:5]}"
    )

    # A zero weight is WORSE than an absent one: absent files are scheduled at
    # `fallback_seconds`, while zero tells LPT the file is free. JUnit reports
    # 0.0 for a file whose tests all skipped, which is how one gets in there --
    # `gen_test_durations.py` floors the value, and this pins the floor.
    nonpositive = sorted(f for f, v in weights.items() if v <= 0)
    assert not nonpositive, (
        f"{tree}'s manifest has {len(nonpositive)} non-positive weight(s), e.g. "
        f"{nonpositive[:3]}; the partitioner would treat them as free"
    )


def test_gen_refuses_a_report_from_the_other_tree(tmp_path: Path) -> None:
    """`--tree X` must refuse a JUnit report of tree Y, before writing.

    Run through the real entry point: the guard's entire value is that it fires
    BEFORE the write, and a written manifest of dead paths is indistinguishable
    from a good one afterwards (see the test above for what that costs).
    """
    from scripts import gen_test_durations

    report = tmp_path / "wrong-tree.xml"
    report.write_text(
        '<?xml version="1.0"?>\n<testsuites><testsuite name="p">'
        '<testcase classname="tests.unit.test_paths" time="1.0"/>'
        "</testsuite></testsuites>\n"
    )
    out = tmp_path / "durations-e2e.json"

    with pytest.raises(SystemExit) as excinfo:
        gen_test_durations.main(["--tree", "e2e", "--junit", str(report), "--out", str(out)])

    assert excinfo.value.code == 2, "argparse must report the mismatch as a usage error"
    assert not out.exists(), "the mismatched manifest was written anyway"


def test_a_measured_zero_never_becomes_a_free_file_or_a_free_fallback() -> None:
    """The floor applies to `fallback_seconds` too, not only to per-file weights.

    JUnit reports 0.0 for a file whose tests all skipped, so a small input -- or
    a regeneration over an unlucky one -- can put the p90 quantile AT zero. A
    zero fallback is worse than no fallback: an unmeasured file then weighs
    nothing, LPT hands it to whichever shard is next, and every unknown piles
    onto the first shard instead of being scattered across all of them. Latent
    with today's tree, wrong the moment a regeneration sees such an input.
    """
    from scripts import gen_test_durations

    per_file = {"tests/unit/test_all_skipped.py": 0.0}
    manifest = gen_test_durations.build_manifest(per_file)

    # `build_manifest` is typed loosely (`dict[str, object]`), so narrow before
    # comparing: the assertion is about the VALUES it writes, and pyright will
    # not check a comparison against `object`.
    fallback = manifest["fallback_seconds"]
    durations = manifest["durations"]
    assert isinstance(fallback, float) and fallback > 0, manifest
    assert isinstance(durations, dict), manifest
    assert durations["tests/unit/test_all_skipped.py"] > 0, manifest


def _junit_report(tmp_path: Path, *testcases: str) -> Path:
    """A minimal JUnit report holding `testcases` verbatim.

    Built as text rather than through ElementTree because the point of the
    tests below is a `classname` pytest really emits, including the empty one
    an XML writer would be within its rights to drop.
    """
    report = tmp_path / "junit.xml"
    report.write_text(
        '<?xml version="1.0"?>\n<testsuites><testsuite name="p">'
        + "".join(testcases)
        + "</testsuite></testsuites>\n"
    )
    return report


def test_a_module_level_skip_resolves_through_its_name(tmp_path: Path) -> None:
    """`classname=""` must not stop the documented CI regeneration route.

    pytest reports a MODULE-level skip (`pytest.skip(...,
    allow_module_level=True)`) with an EMPTY `classname` and the module's
    dotted path in `name` -- pytest 8, observed on CI 2026-10-19 with
    `tests/unit/tui/test_hub_links_capture.py`. That empty string used to reach
    `Path("").with_suffix(".py")` and raise `ValueError: PosixPath('.') has an
    empty name`, which took down the whole run BEFORE anything was written: the
    manifest did not refresh and went 630 files stale instead, with no CI job
    failing to say so. Driven through `main`, because the property is that the
    route completes and writes, not that one helper returns a string.
    """
    from scripts import gen_test_durations

    report = _junit_report(
        tmp_path, '<testcase classname="" name="tests.unit.test_ci_hygiene" time="2.5"/>'
    )
    out = tmp_path / "durations.json"

    assert gen_test_durations.main(["--junit", str(report), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["durations"] == {"tests/unit/test_ci_hygiene.py": 2.5}, written


@pytest.mark.parametrize("classname", ["", ".", "..", "tests."])
def test_a_classname_of_only_separators_is_rejected(classname: str, tmp_path: Path) -> None:
    """Unresolvable inputs fail with attribution context, not a Path crash.

    Even a positive duration with a malformed classname cannot silently vanish
    from the regenerated weights. Valid empty-classname skips resolve by name
    in the separate real-pytest-shape regression above.
    """
    from scripts import gen_test_durations

    report = _junit_report(tmp_path, f'<testcase classname="{classname}" time="1.0"/>')

    with pytest.raises(ValueError, match="did not map"):
        gen_test_durations.aggregate([report])


def test_the_unresolved_error_names_the_string_that_was_tried(
    tmp_path: Path,
) -> None:
    """A rejected mapping must be actionable: name the string actually tried.

    On the empty-classname path the string tried is `name`, and the warning
    recorded the raw `classname` instead -- so the message printed `['']`,
    which names no testcase the reader could go and look at. A silently
    dropped mapping understates a file's cost, which is the failure the
    warning exists to make loud.
    """
    from scripts import gen_test_durations

    report = _junit_report(
        tmp_path, '<testcase classname="" name="tests.unit.test_never_written" time="1.0"/>'
    )

    with pytest.raises(ValueError, match="tests.unit.test_never_written"):
        gen_test_durations.aggregate([report])


def test_unit_shard_capacity_changes_without_relaxing_timeout() -> None:
    """The fleet repair buys parallel capacity, never fewer tests or a higher cap."""
    assert _shard_plan("test") == ("unit", 6)
    assert shard_tests.TREES["unit"].default_shards == 6
    assert _ci_jobs()["test"]["timeout-minutes"] == 38


def test_every_shard_matrix_job_uses_a_known_tree() -> None:
    """Every job whose matrix has a `shard` axis must name one known tree.

    The failure this catches is a THIRD sharded job added later with a matrix
    of its own and no entry here: it would run whatever its inline command
    said, unguarded, and the guards below would keep passing while covering
    only the two trees they know about.
    """
    sharded = [
        name
        for name, job in _ci_jobs().items()
        if isinstance(job.get("strategy", {}).get("matrix"), dict)
        and "shard" in job["strategy"]["matrix"]
    ]
    assert sorted(sharded) == sorted(SHARD_JOBS), (
        "a CI job has a `shard` matrix axis that this module does not know "
        f"about (sharded={sorted(sharded)}, known={sorted(SHARD_JOBS)})"
    )
    for job, tree in SHARD_JOBS.items():
        _shard_plan(job)


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_every_test_file_lands_in_exactly_one_shard(job: str) -> None:
    """Each tree's partition must never drop or duplicate a test file.

    This is the load-bearing invariant of the duration-balanced split, and it
    is asserted PER TREE, because the two trees fail differently. For the unit
    tree a dropped file is a unit test that stops running while CI stays
    green. For `tests/e2e` it is worse: that tree is the only thing that drives
    the assembled application, so a dropped file is a slice of the #401 freeze
    guard that no longer executes — and with the tree now split across runners,
    a dropped file is also invisible in a way it was not before, since no leg's
    log contains the whole suite to compare against.

    Mutation-tested: filtering the file list through the manifest
    (`[f for f in files if f in weights]`) fails this test for both trees.
    """
    tree, total = _shard_plan(job)
    files = shard_tests.collect_test_files(tree=tree)
    weights, fallback = shard_tests.load_weights(shard_tests.TREES[tree].manifest)
    assert files, f"no {tree} test files collected; the glob is wrong"

    shards = shard_tests.partition(files, weights, fallback, total)
    assigned = [f for shard in shards for f in shard]

    assert len(assigned) == len(set(assigned)), "a test file was assigned twice"
    assert set(assigned) == set(files), (
        f"the {tree} partition does not cover every collected test file; "
        f"missing={sorted(set(files) - set(assigned))[:5]}"
    )


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_unmeasured_test_files_are_still_scheduled(job: str) -> None:
    """A file absent from a tree's manifest must still run.

    The manifests are committed, so they are stale the moment anyone adds a
    test. Staleness is allowed to cost BALANCE and must never cost COVERAGE: an
    unknown file is weighted at `fallback_seconds` and scheduled like any
    other. The e2e tree makes this concrete rather than theoretical: its
    manifest did not exist before the sharding PR, so EVERY e2e file is
    unmeasured on the first run, and a partition that skipped unknowns would
    have run nothing at all while all six legs reported success.
    """
    tree, total = _shard_plan(job)
    files = shard_tests.collect_test_files(tree=tree)
    weights, fallback = shard_tests.load_weights(shard_tests.TREES[tree].manifest)
    unknown = f"{shard_tests.TREES[tree].root}/test_not_in_the_manifest_at_all.py"
    assert unknown not in weights

    shards = shard_tests.partition(files + [unknown], weights, fallback, total)
    holders = [i for i, shard in enumerate(shards) if unknown in shard]
    assert (
        len(holders) == 1
    ), f"an unmeasured test file must land in exactly one shard, got {holders}"
    assert fallback > 0, "the fallback weight must be positive"


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_shard_partition_is_deterministic(job: str) -> None:
    """The same commit must always produce the same split, per tree.

    If the partition varied between the shard jobs of one run, a file could be
    run twice or not at all; if it varied between runs, a shard failure would
    be unreproducible. Determinism comes from sorting on (-weight, path) and
    breaking load ties on the lowest shard index. The e2e tree has a second
    reason to require it: the same `--total` runs on both OS legs, so a
    partition that varied would have macOS and ubuntu running DIFFERENT slices
    of the same commit, and no leg's pass would mean the whole tree passed.
    """
    tree, total = _shard_plan(job)
    files = shard_tests.collect_test_files(tree=tree)
    weights, fallback = shard_tests.load_weights(shard_tests.TREES[tree].manifest)
    first = shard_tests.partition(files, weights, fallback, total)
    for _ in range(5):
        assert shard_tests.partition(files, weights, fallback, total) == first


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_ci_partitions_each_tree_by_measured_duration(job: str) -> None:
    """CI must invoke the balanced partitioner for every sharded job.

    For the unit tree, the `i % 5` split left the heaviest shard ~47 seconds
    under a 20-minute cap and migrated that load between shards as files were
    added, so a PR was failed by a cap while its log read `3693 passed`.
    Reverting to an inline split silently restores that. For the e2e tree the
    inline split is worse than unbalanced — the tree is lumpy enough that a
    file-count split puts several of the slowest pilot files in one leg and
    leaves another leg nearly idle, which is the opposite of the wall-time cut
    the matrix exists for.
    """
    run = _shard_step_run(job)
    assert (
        "scripts/shard_tests.py" in run
    ), f"{job} no longer calls the duration-balanced partitioner"
    assert "i % total" not in run and "i % 5" not in run, "the inline positional split is back"


#: Measured overhead per shard job: everything that is not the tests —
#: checkout, `pip install -e`, collection, the coverage/artifact steps.
#:
#: Calibrated from run 35416005688 (the 02:33 main run), job wall minus the
#: pytest-reported duration in the same log: the 3.12 unit shard 0 was 688 s
#: against 650 s of tests, and the tui-e2e ubuntu leg was 1164 s against
#: 1112 s. Observed overhead is therefore 38-52 s; 60 s is that rounded up,
#: not a guess with room in it.
#:
#: The unit job's later `Install ripgrep` step postdates that run and is
#: overhead too. As an apt install it cost 7-18 s per shard (#2997 round 1,
#: review R2), enough to push the top of the range past 60 s. It now fetches
#: one ~2.2 MB pinned release tarball, about 1-2 s, which keeps the measured
#: range inside 60 s. Re-measure before you add another setup step here.
SHARD_JOB_OVERHEAD_SECONDS = 60

#: Workers the unit shard job actually gets, and the reason the two trees
#: cannot share a bound. `conftest.py`'s hook takes EVERY core when `CI` is set
#: (the 0.5 share is deliberately skipped on a dedicated runner — applying it
#: measurably halved CI parallelism), and a GitHub `ubuntu-latest` runner has 4
#: vCPUs, so a unit shard's serial weight divides by 4. The e2e tree divides by
#: 1 by design: it runs `-n0` because a hang cannot be interrupted in-process
#: and `faulthandler`'s timer is process-global, which is exactly why its
#: sharding axis had to be runners rather than workers.
CI_UNIT_WORKERS = 4


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_each_shard_job_timeout_exceeds_its_projected_wall(job: str) -> None:
    """The job ceiling must fit the work the partition gives that job.

    A timeout shorter than the shard is not a faster job, it is a cancelled
    one: GitHub reports it as a failure with a partial log, and (for the unit
    matrix) the shard's coverage artifact never uploads. Retuning the partition
    without revisiting the ceiling is the mistake this guards, and it matters
    more than it did with one serial e2e leg, because the e2e matrix now
    multiplies whatever ceiling that job carries by six jobs.

    The bound is expressed against the SERIAL weight of the heaviest shard,
    divided by the workers that job really gets, rather than against a wall
    time recorded in this file. A recorded wall is a fact about yesterday's
    tree; this is a fact about the tree the partition just read, so a suite
    that grows by a factor of two fails here before it fails CI. What validates
    the divisor: run 35416005688's heaviest 3.12 unit shard reported 1031 s of
    pytest time for 2084 s of serial weight (ratio 2.0 against the modelled
    4 workers, i.e. the model is CONSERVATIVE by 2x on that shard), and the
    unsharded e2e leg reported 1112 s for 1112 s (ratio 1.0).
    """
    tree, total = _shard_plan(job)
    files = shard_tests.collect_test_files(tree=tree)
    weights, fallback = shard_tests.load_weights(shard_tests.TREES[tree].manifest)
    shards = shard_tests.partition(files, weights, fallback, total)
    heaviest = max(sum(weights.get(f, fallback) for f in shard) for shard in shards)

    workers = 1 if tree == "e2e" else CI_UNIT_WORKERS
    projected_wall = heaviest / workers
    if job == "test":
        # SHARD 0 ALSO RUNS THE SERIAL CONTROLS, after its ordinary run, and that
        # step's whole declared ceiling is added undivided — not spread across the
        # four workers it does not use, and with nothing subtracted for the six
        # tests the ordinary run no longer executes (their weight is not itemised
        # in the manifest, so subtracting would be invention). The ceiling is a
        # declared bound, not a measurement; the first CI run reports the real time.
        serial_s = _serial_step_ceiling_seconds()
        projected_wall = max(
            heaviest / workers,
            sum(weights.get(f, fallback) for f in shards[0]) / workers + serial_s,
        )

    timeout = _ci_jobs()[job]["timeout-minutes"]
    assert isinstance(timeout, int)
    budget = timeout * 60 - SHARD_JOB_OVERHEAD_SECONDS
    assert budget >= projected_wall, (
        f"{job}'s {timeout}-minute ceiling leaves {budget:.0f}s for tests after "
        f"{SHARD_JOB_OVERHEAD_SECONDS}s of job overhead, but its heaviest {tree} "
        f"shard is {heaviest:.0f}s of serial weight — {projected_wall:.0f}s on "
        f"{workers} worker(s)"
    )


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_the_shard_job_runs_the_partitioned_list_not_the_whole_tree(job: str) -> None:
    """A partition nothing consumes is a matrix of identical jobs.

    This is the failure with the worst cost-to-noise ratio in the file: revert
    the run step to the tree root (`pytest tests/e2e -m e2e -n0 -q`) while the
    matrix still has three shards, and every leg runs the FULL tree, reports
    success, and multiplies the critical path instead of cutting it — six jobs
    each doing 19 minutes of work. The unit job has the same shape (`pytest
    tests/unit` instead of the file list from `shard_tests.txt`). Nothing else
    in CI would notice: the shards would still be balanced, deterministic and
    fully covered; they would each just run everything.

    `-m e2e`, `-n0` and the colour env are asserted alongside the file list
    because they are the properties AGENTS.md calls load-bearing for this
    stage, and a run step rewritten to consume the list is exactly when they
    get dropped.

    The tree-root check is PATH-AWARE (it parses the pytest argv) rather than a
    substring search, because a substring search asserts the absence of one
    spelling rather than the absence of the behaviour: `pytest $(cat
    e2e_tests.txt | tr '\n' ' ') tests/e2e -m e2e -n0 -q` contains the file
    list, does not contain the literal `pytest tests/e2e`, and collects the
    whole tree in addition to the shard. Mutation-tested both ways: appending
    the root to a correct command fails this test, as does replacing the list
    with the root.
    """
    run = _run_step(job)
    tree = SHARD_JOBS[job]
    root = shard_tests.TREES[tree].root
    argv = _pytest_argv(run)

    assert (
        _SUBSTITUTION_TOKEN in argv
    ), f"{job}'s run step does not pass the partition's file list: {run!r}"
    # The filename lives INSIDE the substitution, so it is checked there rather
    # than against the parsed argv (`pytest $(cat wrong_list.txt)` would
    # otherwise pass as "the list is passed").
    substitutions = _SUBSTITUTION.findall(run)
    assert any(
        "_tests.txt" in sub for sub in substitutions
    ), f"{job}'s run step reads something other than the partition's file list: {substitutions!r}"

    # The tree root must not appear as an ARGUMENT at all. A directory argument
    # (`pytest tests/e2e ...`) collects the whole tree and makes the partition
    # pointless, and it can be appended to a correct command, so this compares
    # parsed arguments rather than searching the raw text.
    roots = [a for a in argv if a.rstrip("/") == root]
    assert not roots, (
        f"{job} passes the whole {root} tree as a pytest argument ({roots}); "
        "the shard matrix would then multiply the run instead of cutting it"
    )
    if tree == "e2e":
        assert re.search(r"-m\s*e2e\b", run), "the e2e shard run lost `-m e2e`"
        assert re.search(r"(?<!\w)-n\s*0(?!\d)", run), (
            "the e2e shard run lost serial execution (`-n0`): a hang cannot be "
            "ended in-process and the timer is process-global, so under xdist a "
            "worker would strand the run and lose its stacks (see AGENTS.md)"
        )
        assert (
            "NO_COLOR" in run and "xterm-256color" in run
        ), "the e2e shard run lost the environment the TUI suite composes a frame with"


def _serial_ids(env: dict[str, Any] | None = None) -> list[str]:
    """The six serial control ids, from the unit job's single source of truth."""
    source = env if env is not None else _ci_jobs()["test"]["env"]
    raw = source.get("SERIAL_CONTROLS")
    assert isinstance(raw, str), "the unit job lost its SERIAL_CONTROLS list"
    return raw.split()


def _serial_step_ceiling_seconds() -> int:
    step = next(s for s in _steps("test") if s.get("id") == SERIAL_STEP_ID)
    minutes = step["timeout-minutes"]
    assert isinstance(minutes, int)
    return minutes * 60


def _top_level_names(path: Path) -> list[str]:
    """Names a pytest node id can START with after ``file::`` (functions and classes)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _flag_value(argv: list[str], flag: str) -> str | None:
    """The token after ``flag`` in a parsed command line, or None when absent."""
    if flag not in argv:
        return None
    index = argv.index(flag) + 1
    return argv[index] if index < len(argv) else None


def _audit_unit_serial_wiring(steps: list[dict[str, Any]], env: dict[str, Any]) -> list[str]:
    """Every way the serial-control wiring can be wrong, as sentences ([] when right).

    A function of plain data, so the NEGATIVE tests below can mutate a copy of the
    real steps and prove each defect is caught — an audit that has only ever been run
    on the correct input has never shown it can fail.
    """
    problems: list[str] = []
    ids = _serial_ids(env)
    if len(ids) != 6 or len(set(ids)) != 6:
        problems.append(f"expected six unique serial ids, got {ids}")
    for node_id in ids:
        file_part, _, name = node_id.partition("::")
        path = REPO / file_part
        if not path.is_file():
            problems.append(f"{node_id}: no such file")
            continue
        if "::" in name or name not in _top_level_names(path):
            problems.append(f"{node_id}: not a top-level test function of that file")
            continue
        # `--deselect` is a PREFIX match on the node id, so a longer name that merely
        # starts with this one would be deselected with it.
        clashes = [n for n in _top_level_names(path) if n != name and n.startswith(name)]
        if clashes:
            problems.append(f"{node_id}: prefix-collides with {clashes}")
    by_id = {s.get("id"): s for s in steps if s.get("id")}
    ordinary, serial = by_id.get(ORDINARY_STEP_ID), by_id.get(SERIAL_STEP_ID)
    if ordinary is None or serial is None:
        return problems + ["the ordinary or the serial pytest step is missing"]
    ordinary_run = ordinary.get("run") or ""
    serial_run = serial.get("run") or ""
    if "--deselect" not in ordinary_run or "$SERIAL_CONTROLS" not in ordinary_run:
        problems.append("the ordinary run does not deselect $SERIAL_CONTROLS")
    if "--cov-append" in ordinary_run or ".coverage." in ordinary_run:
        problems.append("the ordinary run must not append or rename coverage data")
    if serial.get("if") != "matrix.shard == 0":
        problems.append(f"the serial step must run on shard 0 only, has if={serial.get('if')!r}")
    if serial.get("continue-on-error"):
        problems.append("the serial step must not continue-on-error")
    argv = shlex.split(serial_run.replace("\\\n", " "))
    if "-n0" not in argv:
        problems.append("the serial run lost -n0")
    if "$SERIAL_CONTROLS" not in argv:
        problems.append("the serial run does not select $SERIAL_CONTROLS")
    for required in ("--cov=local_operator", "--cov-append", "--cov-fail-under=0", "--cov-report="):
        if required not in argv:
            problems.append(f"the serial run lost {required}")
    if "--cov" in argv:
        problems.append("a bare --cov resets the coverage selection")
    if "scripts/run_bounded.py" not in serial_run:
        problems.append("the serial run is not bounded by run_bounded")
    minutes = serial.get("timeout-minutes")
    if minutes != 7:
        problems.append(f"the serial step ceiling must be the declared 7 minutes, is {minutes!r}")
    # THE WRAPPER'S OWN BOUND MUST FIRE INSIDE THE STEP'S: SIGTERM at --timeout, SIGKILL
    # --grace later, both before the step's timeout-minutes cancels it from outside (an
    # outside cancel is the path the in-child orphan guard exists for). Parsed from the
    # real command, so an edit to either number is judged, not assumed.
    try:
        bound = float(_flag_value(argv, "--timeout") or "nan")
        grace = float(_flag_value(argv, "--grace") or "nan")
    except ValueError:
        bound = grace = float("nan")
    if isinstance(minutes, int) and not bound + grace < minutes * 60:
        problems.append(
            f"the wrapper's bound plus grace ({bound}+{grace}s) must be shorter than the "
            f"step's {minutes} minutes, and both must be given"
        )
    # NO INVENTED MEMORY CAP: this job has no memory policy today, a guessed ceiling
    # could kill a healthy test, and the wrapper refuses to start with no resolvable
    # auto budget. Zero is passed EXPLICITLY (it reports "disabled (explicit)") so the
    # absence is visible and a future cap is a deliberate decision.
    if _flag_value(argv, "--memory-mb") != "0":
        problems.append("the serial run must pass an explicit --memory-mb 0 (no invented cap)")
    order = [ORDINARY_STEP_ID, "serial_absent", SERIAL_STEP_ID, "serial_proof"]
    positions = [next((i for i, s in enumerate(steps) if s.get("id") == k), -1) for k in order]
    if -1 in positions or positions != sorted(positions):
        problems.append(f"serial steps missing or out of order: {positions}")
    final = [i for i, s in enumerate(steps) if "mv .coverage" in (s.get("run") or "")]
    if len(final) != 1 or final[0] <= max(positions):
        problems.append("the coverage rename must be one step, AFTER every pytest step")
    proof = by_id.get("serial_proof")
    if proof is not None and "steps.serial_controls.outcome" not in str(proof.get("if")):
        problems.append("the proof step is not gated on the serial step being attempted")
    if proof is not None and "always()" not in str(proof.get("if")):
        # Without it the implicit success() skips the proof exactly when the serial
        # step FAILED or hit its bound, and a failed run would carry no verdict.
        problems.append("the proof step must run under always(), not only on success")
    absent = by_id.get("serial_absent")
    if absent is not None and "always()" not in str(absent.get("if")):
        problems.append("the absence proof must run even when the ordinary run failed")
    uploads = [s for s in steps if "upload-artifact" in str(s.get("uses"))]
    junit = next(
        (s for s in uploads if "junit-timings" in str((s.get("with") or {}).get("name"))), None
    )
    if junit is None or "junit-serial-controls-" not in str((junit.get("with") or {}).get("path")):
        problems.append("the junit-timings artifact does not carry the serial XML")
    if junit is not None and str(junit.get("if")).strip() != "always()":
        problems.append("the junit upload must stay `if: always()`")
    return problems


def _lint_format_problems(steps: list[dict[str, Any]]) -> list[str]:
    """How the lint job's formatter steps can fail to report the whole answer.

    Pure data in, sentences out, so the cases below can corrupt a copy of the real
    steps. Covers only what a failed first push taught: the formatters must print their
    diff, and must run after a successful tool install even when an earlier check failed.
    """
    problems: list[str] = []
    install = next((s for s in steps if "pip install" in (s.get("run") or "")), None)
    install_id = (install or {}).get("id")
    if not install_id:
        problems.append("the lint tool-install step has no id to gate the formatters on")
    for tool in ("black", "isort"):
        step = next((s for s in steps if (s.get("run") or "").startswith(f"{tool} --check")), None)
        if step is None:
            problems.append(f"no {tool} --check step")
            continue
        if "--diff" not in step["run"]:
            problems.append(f"{tool} --check does not print its diff")
        gate = " ".join(str(step.get("if") or "").split())
        if "!cancelled()" not in gate or f"steps.{install_id}.outcome == 'success'" not in gate:
            problems.append(f"{tool} is not gated on !cancelled() and a successful install")
        if step.get("continue-on-error"):
            problems.append(f"{tool} masks its failure with continue-on-error")
    return problems


def test_the_lint_formatters_print_their_diff_and_run_after_a_successful_install() -> None:
    """Black and isort both report in one failed run, and neither can mask a failure."""
    import copy

    steps = _steps("lint")
    assert _lint_format_problems(steps) == []

    def gated(mutate: Any) -> list[str]:
        changed = copy.deepcopy(steps)
        mutate(changed)
        return _lint_format_problems(changed)

    def black(changed: list[dict[str, Any]]) -> dict[str, Any]:
        return next(s for s in changed if (s.get("run") or "").startswith("black --check"))

    assert any("diff" in p for p in gated(lambda c: black(c).update(run="black --check .")))
    assert any("gated" in p for p in gated(lambda c: black(c).pop("if")))
    assert any("masks" in p for p in gated(lambda c: black(c).update({"continue-on-error": True})))
    assert any("no id" in p for p in gated(lambda c: c[2].pop("id")))


def test_the_unit_job_runs_exactly_its_two_audited_pytest_steps() -> None:
    """The ordinary run and the serial controls are the ONLY pytest invocations.

    Mutation-tested below: a third pytest step, the serial step under another id, or
    a missing one all fail ``_pytest_steps``.
    """
    steps = _pytest_steps("test")
    assert [s["id"] for s in steps] == [ORDINARY_STEP_ID, SERIAL_STEP_ID]


def test_the_serial_controls_are_wired_once_deselected_everywhere_and_run_on_shard_zero() -> None:
    """The deselect/serial/proof/coverage wiring of the unit job is exactly as designed."""
    job = _ci_jobs()["test"]
    assert _audit_unit_serial_wiring(job["steps"], job["env"]) == []
    # The ordinary run's deselect list is the SAME list the serial run selects: both
    # read $SERIAL_CONTROLS, so they cannot name different tests.
    assert "--deselect" in _run_step("test")


def _in_run(step_id: str, old: str, new: str) -> Any:
    """A mutation: replace ``old`` by ``new`` inside one step's ``run:`` body."""

    def mutate(steps: list[dict[str, Any]]) -> None:
        step = next(s for s in steps if s.get("id") == step_id)
        assert old in step["run"], f"mutation target {old!r} is not in step {step_id}"
        step["run"] = step["run"].replace(old, new)

    return mutate


def _set_field(step_id: str, **fields: Any) -> Any:
    """A mutation: set top-level keys of one step (``if``, ``timeout-minutes``, ...)."""

    def mutate(steps: list[dict[str, Any]]) -> None:
        next(s for s in steps if s.get("id") == step_id).update(fields)

    return mutate


def _serial_xml_not_uploaded(steps: list[dict[str, Any]]) -> None:
    upload = next(
        s
        for s in steps
        if "upload-artifact" in str(s.get("uses")) and "junit" in str(s["with"]["name"])
    )
    upload["with"].update(path="junit-3.12-0.xml")


def _five_ids(env: dict[str, Any]) -> None:
    env.update(SERIAL_CONTROLS="\n".join(env["SERIAL_CONTROLS"].split()[:5]))


def _duplicate_id(env: dict[str, Any]) -> None:
    ids = env["SERIAL_CONTROLS"].split()
    env.update(SERIAL_CONTROLS="\n".join(ids[:5] + ids[:1]))


def _missing_selector(env: dict[str, Any]) -> None:
    gone = "test_cell_a_control_the_old_observer_setup_fails_its_precondition"
    env.update(SERIAL_CONTROLS=env["SERIAL_CONTROLS"].replace(gone, "test_nope"))


#: (label, the problem text that corruption MUST produce, step mutation, env mutation).
#: Keyed to the EXPECTED sentence, not merely "some problem": a corruption that trips a
#: different guard would otherwise count as caught while its own guard is dead.
_WIRING_MUTATIONS: list[tuple[str, str, Any, Any]] = [
    (
        "no deselect",
        "does not deselect",
        _in_run(ORDINARY_STEP_ID, "--deselect", "--ignore"),
        None,
    ),
    (
        "serial on every shard",
        "shard 0 only",
        _set_field(SERIAL_STEP_ID, **{"if": "true"}),
        None,
    ),
    (
        "serial under xdist",
        "lost -n0",
        _in_run(SERIAL_STEP_ID, "-n0", "-n4"),
        None,
    ),
    (
        "bare --cov",
        "a bare --cov resets",
        _in_run(SERIAL_STEP_ID, "--cov=local_operator", "--cov"),
        None,
    ),
    (
        "no append",
        "lost --cov-append",
        _in_run(SERIAL_STEP_ID, "--cov-append", ""),
        None,
    ),
    (
        "masked failure",
        "continue-on-error",
        _set_field(SERIAL_STEP_ID, **{"continue-on-error": True}),
        None,
    ),
    (
        "unbounded",
        "is not bounded by run_bounded",
        _in_run(SERIAL_STEP_ID, "scripts/run_bounded.py", ""),
        None,
    ),
    (
        "ceiling bumped",
        "must be the declared 7 minutes",
        _set_field(SERIAL_STEP_ID, **{"timeout-minutes": 20}),
        None,
    ),
    (
        "wrapper bound overflows the step",
        "must be shorter than the step's 7 minutes",
        _in_run(SERIAL_STEP_ID, "--timeout 400", "--timeout 420"),
        None,
    ),
    (
        "no explicit memory policy",
        "explicit --memory-mb 0",
        _in_run(SERIAL_STEP_ID, "--memory-mb 0", ""),
        None,
    ),
    (
        "proof only always()",
        "not gated on the serial step being attempted",
        _set_field("serial_proof", **{"if": "always()"}),
        None,
    ),
    (
        "proof loses always()",
        "must run under always()",
        _set_field("serial_proof", **{"if": "steps.serial_controls.outcome != 'skipped'"}),
        None,
    ),
    (
        "absence not always",
        "the absence proof must run even when",
        _set_field("serial_absent", **{"if": None}),
        None,
    ),
    (
        "rename inside the ordinary run",
        "must not append or rename coverage data",
        _in_run(
            ORDINARY_STEP_ID,
            "--cov-fail-under=0",
            "--cov-fail-under=0; mv .coverage .coverage.x",
        ),
        None,
    ),
    (
        "serial XML not uploaded",
        "does not carry the serial XML",
        _serial_xml_not_uploaded,
        None,
    ),
    ("five ids", "expected six unique serial ids", None, _five_ids),
    ("duplicate id", "expected six unique serial ids", None, _duplicate_id),
    ("missing selector", "not a top-level test function", None, _missing_selector),
]


def _mutated_wiring() -> list[tuple[str, str, list[str]]]:
    """(label, expected problem text, problems found) per deliberate corruption."""
    import copy

    job = _ci_jobs()["test"]
    results: list[tuple[str, str, list[str]]] = []
    for label, expected, mutate_steps, mutate_env in _WIRING_MUTATIONS:
        steps = copy.deepcopy(job["steps"])
        env = copy.deepcopy(job["env"])
        if mutate_steps is not None:
            mutate_steps(steps)
        if mutate_env is not None:
            mutate_env(env)
        results.append((label, expected, _audit_unit_serial_wiring(steps, env)))
    return results


def test_every_wiring_defect_is_caught_by_the_audit() -> None:
    """Each corruption is reported, and by the guard written for it (not just any guard)."""
    missed = [
        (label, expected, problems)
        for label, expected, problems in _mutated_wiring()
        if not any(expected in problem for problem in problems)
    ]
    assert not missed, f"the audit did not report these corruptions as expected: {missed}"
    assert (
        _audit_unit_serial_wiring(_ci_jobs()["test"]["steps"], _ci_jobs()["test"]["env"]) == []
    ), "the real wiring must be clean"


def test_a_third_or_misnamed_pytest_step_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_pytest_steps`` refuses an extra pytest command and a renamed or missing serial step."""
    import copy

    steps = copy.deepcopy(_ci_jobs()["test"]["steps"])
    third = steps + [{"name": "sneak", "run": "python -m pytest tests/unit -q"}]
    renamed = [dict(s, id="other") if s.get("id") == SERIAL_STEP_ID else s for s in steps]
    missing = [s for s in steps if s.get("id") != SERIAL_STEP_ID]
    for broken in (third, renamed, missing):
        monkeypatch.setattr(sys.modules[__name__], "_steps", lambda _job, _s=broken: _s)
        with pytest.raises(AssertionError, match="two audited pytest steps"):
            _pytest_steps("test")


def test_the_serial_budget_is_accounted_in_full_and_fits_the_unchanged_job_ceiling() -> None:
    """Shard 0's modelled wall is its weight/4 PLUS the whole 420 s, within today's 32 min.

    The serial step is not divided by four (it runs ``-n0``) and nothing is subtracted
    for the six deselected tests. The job ceiling is NOT raised to make this pass: if
    the merge ref outgrows it, the answer is to rebalance or propose a change, so this
    reads the ceiling that exists. Every other shard is judged by the existing rule.
    """
    assert _serial_step_ceiling_seconds() == 420
    tree, total = _shard_plan("test")
    files = shard_tests.collect_test_files(tree=tree)
    weights, fallback = shard_tests.load_weights(shard_tests.TREES[tree].manifest)
    shards = shard_tests.partition(files, weights, fallback, total)
    zero = sum(weights.get(f, fallback) for f in shards[0]) / CI_UNIT_WORKERS + 420
    timeout = _ci_jobs()["test"]["timeout-minutes"]
    assert zero <= timeout * 60 - SHARD_JOB_OVERHEAD_SECONDS, (
        f"shard 0 projects {zero:.0f}s with the serial step against a "
        f"{timeout * 60 - SHARD_JOB_OVERHEAD_SECONDS}s budget"
    )


# ---------------------------------------------------------------------------
# scripts/check_serial_controls.py — exercised on SYNTHETIC XML only
# ---------------------------------------------------------------------------


def _serial_pairs() -> list[tuple[str, str]]:
    """(JUnit classname, function) for each id of the real SERIAL_CONTROLS list."""
    return [check_serial_controls.split_id(node_id) for node_id in _serial_ids()]


_SERIAL = _serial_pairs()


def _ids() -> str:
    return " ".join(_serial_ids())


def _case(module: str, name: str, *, verdict: str = "", drop: str = "", oracle: str = "") -> str:
    required = check_serial_controls.REQUIRED_PROPERTIES[name]
    props = []
    for prop in required:
        if prop == drop:
            continue
        value = "x"
        if prop in ("walk_oracle", "status_oracle"):
            value = oracle or (
                "passed"
                if name in check_serial_controls._POSITIVE
                else "precondition: the observer saw 60 of 60"
            )
        props.append(f'<property name="{prop}" value="{value}"/>')
    inner = f"<properties>{''.join(props)}</properties>{verdict}"
    return f'<testcase classname="{module}" name="{name}" time="0.1">{inner}</testcase>'


def _xml(tmp_path: Path, *cases: str, name: str = "j.xml") -> Path:
    path = tmp_path / name
    path.write_text(f"<testsuites><testsuite>{''.join(cases)}</testsuite></testsuites>")
    return path


def _run_checker(mode: str, junit: Path, tmp_path: Path) -> tuple[int, str]:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = check_serial_controls.main(
            [mode, "--junit", str(junit), "--ids", _ids(), "--summary", str(tmp_path / "s.md")]
        )
    return code, buffer.getvalue()


def _all_cases(**override: Any) -> list[str]:
    return [_case(m, n, **override.get(n, {})) for m, n in _SERIAL]


def test_the_checker_proves_six_present_once_with_receipts(tmp_path: Path) -> None:
    code, out = _run_checker("present", _xml(tmp_path, *_all_cases()), tmp_path)
    assert code == 0, out
    assert out.startswith("PASS (present)")
    assert (tmp_path / "s.md").read_text().startswith("### Serial controls — PASS")


@pytest.mark.parametrize(
    "label",
    [
        "missing",
        "skipped",
        "failure",
        "error",
        "duplicate",
        "extra",
        "no_property",
        "positive_failed",
        "control_passed",
    ],
)
def test_the_checker_fails_every_violation_of_the_present_contract(
    tmp_path: Path, label: str
) -> None:
    cases = _all_cases()
    first_module, first_name = _SERIAL[0]
    control = _SERIAL[1][1]
    if label == "missing":
        cases = cases[1:]
    elif label == "skipped":
        cases[0] = _case(first_module, first_name, verdict="<skipped/>")
    elif label == "failure":
        cases[0] = _case(first_module, first_name, verdict="<failure>x</failure>")
    elif label == "error":
        cases[0] = _case(first_module, first_name, verdict="<error>x</error>")
    elif label == "duplicate":
        cases.append(cases[0])
    elif label == "extra":
        cases.append('<testcase classname="tests.unit.test_other" name="test_z" time="0"/>')
    elif label == "no_property":
        cases[0] = _case(first_module, first_name, drop="walk_report")
    elif label == "positive_failed":
        cases[0] = _case(first_module, first_name, oracle="failed: precondition")
    elif label == "control_passed":
        cases[1] = _case(_SERIAL[1][0], control, oracle="passed")
    code, out = _run_checker("present", _xml(tmp_path, *cases), tmp_path)
    assert code == 1, out
    assert out.startswith("FAIL (present)")


def test_the_checker_never_reads_a_missing_empty_or_garbled_xml_as_proof(
    tmp_path: Path,
) -> None:
    assert _run_checker("present", tmp_path / "nope.xml", tmp_path)[0] == 2
    assert _run_checker("absent", tmp_path / "nope.xml", tmp_path)[0] == 2
    empty = _xml(tmp_path, name="empty.xml")
    assert _run_checker("present", empty, tmp_path)[0] == 2
    garbled = tmp_path / "bad.xml"
    garbled.write_text("<testsuites><oops")
    assert _run_checker("absent", garbled, tmp_path)[0] == 2


def test_the_checker_proves_absence_and_fails_when_a_control_ran_in_the_shard(
    tmp_path: Path,
) -> None:
    other = '<testcase classname="tests.unit.test_other" name="test_z" time="0"/>'
    assert _run_checker("absent", _xml(tmp_path, other), tmp_path)[0] == 0
    code, out = _run_checker("absent", _xml(tmp_path, other, _all_cases()[0]), tmp_path)
    assert code == 1, out
    assert "ran in the ordinary shard run" in out


def test_the_checker_help_survives_stripped_docstrings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -OO`` strips docstrings, so ``__doc__`` is None and ``--help`` must still work.

    The description used to be ``__doc__.split(...)``, which pyright rejects
    (``reportOptionalMemberAccess``) and which would raise ``AttributeError`` here. The
    module attribute is set to None in-process, so no child is started.
    """
    monkeypatch.setattr(check_serial_controls, "__doc__", None)
    with pytest.raises(SystemExit) as stop:
        check_serial_controls.main(["--help"])
    assert stop.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_the_checker_blocks_an_id_it_has_no_receipt_contract_for(tmp_path: Path) -> None:
    junit = _xml(tmp_path, *_all_cases())
    code = check_serial_controls.main(
        ["present", "--junit", str(junit), "--ids", "tests/unit/test_x.py::test_unknown"]
    )
    assert code == 2


@pytest.mark.parametrize("job", SHARD_JOB_IDS)
def test_ci_shard_matrix_covers_every_shard_the_partitioner_is_told_to_make(
    job: str, tmp_path: Path
) -> None:
    """The matrix must run every shard `--total` splits that tree into, and
    `main()` must emit all of them.

    This asserts the coverage invariant on the REAL entry point. The other
    guards call `partition()` directly, which leaves the layer CI actually
    invokes -- argument parsing, `--tree`/`--shard` selection,
    `shards[args.shard]` lookup, the `--out` write -- untested. Review round 1
    (MAJOR-2) demonstrated three mutations that silently stop test files from
    running while all guards stayed green:

      - `selected = shards[args.shard][:1]` in `main()`  -> 91 of 92 files skipped
      - matrix `[0, 1, 2, 3]` while `--total` stays 5    -> ~92 files never run
      - `--total 6` while the matrix stays 5             -> shard 5 orphaned

    The same three shapes exist for the e2e tree, where the matrix is a
    product: a `shard` axis of `[0, 1]` against `--total 3` orphans a third of
    the freeze guard on BOTH operating systems, with every leg reporting a
    pass. `--tree` is part of the plan read here, so a job pointed at the
    wrong tree (or at none, defaulting to `unit`) fails rather than sharding
    the wrong suite.

    None of those is caught by asserting the step merely *calls* the script,
    because nothing coupled the matrix length to `--total`. Both numbers are
    read from ci.yml here rather than from constants, so changing one without
    the other fails; and the union of what `main()` actually WRITES is
    compared against the collected suite, so a truncated or misselected
    write fails too.
    """
    run = _shard_step_run(job)
    tree, total = _shard_plan(job)

    matrix = _ci_jobs()[job]["strategy"]["matrix"]["shard"]
    assert isinstance(matrix, list)
    assert sorted(matrix) == list(range(total)), (
        f"the shard matrix {sorted(matrix)} does not cover every shard of "
        f"--total {total}; files in the uncovered shards would never run"
    )

    # The step passes `--shard ${{ matrix.shard }}`, so the matrix value is
    # the argument. Drive main() the same way CI does, once per shard.
    assert "--shard" in run, f"{job}'s partition step no longer passes --shard"

    written: list[str] = []
    for shard in matrix:
        out = tmp_path / f"shard_{shard}.txt"
        rc = shard_tests.main(
            ["--tree", tree, "--shard", str(shard), "--total", str(total), "--out", str(out)]
        )
        assert rc == 0, f"main() failed for shard {shard}"
        written.extend(out.read_text().split())

    expected = shard_tests.collect_test_files(tree=tree)
    assert len(written) == len(set(written)), "a test file was written to two shards"
    assert set(written) == set(expected), (
        f"the files main() writes for the {tree} tree do not cover it; "
        f"missing={sorted(set(expected) - set(written))[:5]}"
    )


# ===========================================================================
# Change-scope gating: the classifier in scripts/ci_scope.py, and the wiring in
# ci.yml's `changes` job that reads it. Assertions A1-A14 and A16-A17 of the
# design review — A15 is cited by name in two comments below but is an assertion
# of the scope-gating change, not one in this file.
# ===========================================================================


def _scope() -> Any:
    """The classifier module under test.

    Imported rather than re-implemented: every predicate assertion below drives
    the module's own `classify()`, so a change to a predicate fails a test
    instead of silently re-defining the contract in this file.
    """
    assert ci_scope is not None
    return ci_scope


def test_every_ci_job_is_either_flag_gated_or_explicitly_ungated() -> None:
    """A1. A job in neither map pays full price on every diff, unnoticed.

    `JOB_FLAGS` is the set of jobs the classifier gates; `UNGATED_JOBS` is the
    set that must not be gated. Every job in ci.yml has to be in exactly one of
    them — both directions, so a stale name in the module cannot sit there
    pretending to gate a job that no longer exists — and a gated job that
    references a flag the `changes` job does not export is gated on an output
    that is always empty (''), which reads as "always run".

    Mutations that must fail this: delete one job's `if:`; or add a job to
    ci.yml and to neither map.
    """
    scope = _scope()
    jobs = set(_ci_jobs())

    assert set(scope.JOB_FLAGS) <= jobs, (
        "JOB_FLAGS names jobs that are not in ci.yml: " f"{sorted(set(scope.JOB_FLAGS) - jobs)}"
    )
    assert set(scope.UNGATED_JOBS) <= jobs, (
        "UNGATED_JOBS names jobs that are not in ci.yml: "
        f"{sorted(set(scope.UNGATED_JOBS) - jobs)}"
    )
    unclassified = jobs - set(scope.JOB_FLAGS) - set(scope.UNGATED_JOBS)
    assert not unclassified, (
        f"{sorted(unclassified)} are in ci.yml but in neither JOB_FLAGS nor "
        "UNGATED_JOBS: they run on every diff with nothing saying why"
    )

    for job, flags in scope.JOB_FLAGS.items():
        assert flags, f"{job} is in JOB_FLAGS with no flag — nothing gates it"
        unknown = sorted(set(flags) - set(scope.FLAGS))
        assert not unknown, (
            f"{job} is gated on {unknown}, which the `changes` job does not "
            "export: an absent output is '' and the gate would never skip"
        )
        assert "changes" in _needs(job), (
            f"{job} reads a scope flag but does not `need: changes`, so the flag "
            "is undefined and the gate is not a decision"
        )
        for flag in flags:
            assert f"needs.changes.outputs.{flag}" in _if(job), (
                f"{job}'s `if:` does not read `needs.changes.outputs.{flag}` " f"(if: {_if(job)!r})"
            )

    for job, reason in scope.UNGATED_JOBS.items():
        assert reason.strip(), f"{job} is declared ungated with no reason"


def test_a_gated_job_escapes_the_implicit_success_without_hiding_a_failure() -> None:
    """A2. `needs:` skips a job when a needed job was SKIPPED.

    GitHub applies an implicit `success()` to any `if:` that contains no
    status-check function, so a job that reads a scope flag but omits
    `!cancelled()` is skipped whenever ANY dependency was skipped. That is the
    collateral skip: `lint` deliberately skipping an inert diff would silently
    skip `tui-e2e` — the #426 shape in a new costume. The `.result` clauses are
    the other half: escaping `success()` must not also escape a FAILED cheaper
    gate, or the escape is a hole.

    Mutations that must fail this: remove `!cancelled()` from a gated job's
    `if:`; then remove `needs.lint.result == 'success'` from tui-e2e's.
    """
    checked = 0
    for job in _ci_jobs():
        expression = _if(job)
        if "needs.changes.outputs" not in expression:
            continue
        checked += 1
        assert "!cancelled()" in expression, (
            f"{job} reads a scope flag but has no status-check function, so an "
            "implicit `success()` is applied and a SKIPPED dependency skips it "
            f"(if: {expression!r})"
        )
        for dependency in sorted(_needs(job) - {"changes"}):
            assert f"needs.{dependency}.result" in expression, (
                f"{job} needs `{dependency}` without a `.result` clause, so a "
                f"failed {dependency} no longer blocks it (if: {expression!r})"
            )
    assert checked >= 2, "no job reads a scope flag; the gating wiring is gone from ci.yml"


def test_scope_gates_fail_open_and_never_compare_against_true() -> None:
    """A3. `== 'true'` is a fail-CLOSED trap in the dangerous direction.

    An unset, misnamed or never-written output renders as EMPTY, and
    `'' == 'true'` is false — so a classifier that crashed would produce a fully
    green PR that ran nothing. Every gate therefore reads `!= 'false'`, and the
    writer is the second, independent layer: it RAISES rather than emitting ''
    for a value it cannot render.

    Mutations that must fail this: flip one gate to `== 'true'`; or make
    `_flag_value` return '' instead of raising.
    """
    scope = _scope()
    for job in _ci_jobs():
        expression = _if(job)
        assert "== 'true'" not in expression and '== "true"' not in expression, (
            f"{job} compares a scope output against 'true' (if: {expression!r}); "
            "an output that was never written is EMPTY, so this skips the job "
            "whenever the classifier fails"
        )
    for job, flags in scope.JOB_FLAGS.items():
        for flag in flags:
            assert f"needs.changes.outputs.{flag} != 'false'" in _if(
                job
            ), f"{job} does not gate `{flag}` on != 'false'"

    with pytest.raises(scope.ScopeError):
        scope._flag_value("lint", {"lint": ""})
    with pytest.raises(scope.ScopeError):
        scope._flag_value("lint", {})
    assert scope._flag_value("lint", {"lint": True}) == "true"
    assert scope._flag_value("lint", {"lint": False}) == "false"


#: One representative path per category, used to sample change sets for the
#: D8a implication check (A4). Every category is covered, asserted below, so a
#: new category cannot be added to the module and left unexercised here.
_SAMPLE_PATHS: dict[str, str] = {
    "ci": ".github/workflows/ci.yml",
    "python": "local_operator/cli.py",
    "web": "local_operator/mobile/web/src/app.tsx",
    "tests": "tests/unit/test_ci_hygiene.py",
    "scripts": "scripts/shard_tests.py",
    "aux_python": "benchmarks/osworld_v2_adapter/src/adapter.py",
    "extension": "extension/src/protocol.gen.ts",
    "manifest": "pyproject.toml",
    "deps_lock": "uv.lock",
    "gate_config": "Makefile",
    "docs": "docs/store/release-record.md",
    "i18n": "i18n/ledger.json",
    "other": "newdir/unrecognised.bin",
}


def _sample_change_sets() -> list[list[str]]:
    """Change sets spanning every category, plus the path-keyed special cases.

    Pairs as well as singletons: several predicates (`budget`, `windows`) turn on
    a path that is neither the most- nor the least-live thing in the diff, and a
    singleton-only sample would never exercise those combinations.
    """
    paths = list(_SAMPLE_PATHS.values())
    samples = [[path] for path in paths]
    samples.extend(list(pair) for pair in itertools.combinations(paths, 2))
    samples.extend(
        [
            ["tests/unit/test_agent_import_boundary.py"],
            ["tests/unit/server/test_edit_workspace_boundary.py"],
            ["tests/conftest.py"],
            ["scripts/bench_context_budget.py"],
            ["scripts/real_tool_surface.py"],
        ]
    )
    return samples


def test_every_dependency_clause_is_justified_by_the_flag_implication() -> None:
    """A4 (D8a). A needed job that was SKIPPED skips its dependent silently.

    For each `(job, dependency)` pair: either the module guarantees
    `types(job) => types(dependency)` for every sampled change set — so the
    dependency can only have been skipped on a diff where the dependent is also
    skipped — or the pair must be in `PERMISSIVE_DEPS` with a non-empty reason
    AND the workflow must use the permissive clause. The implication is measured
    on the module's own `classify()`, so a narrowed predicate fails here rather
    than in production.

    This is the assertion that catches the design's sharpest trap: a flag
    narrower than its dependent's flag (or a dependency whose own flag is
    narrower) disarms an expensive job on exactly the diffs it exists for.

    Mutations that must fail this: narrow `tui` to the docs-only inert set (it
    stops implying `lint`); or delete a `PERMISSIVE_DEPS` reason string.
    """
    scope = _scope()
    samples = _sample_change_sets()
    covered = {scope.category_of(sample[0]) for sample in samples if len(sample) == 1}
    assert covered == set(scope.CATEGORIES), (
        "the sample change sets no longer cover every category: "
        f"missing={sorted(set(scope.CATEGORIES) - covered)}"
    )

    def runs(flags: dict[str, bool], job_flags: tuple[str, ...]) -> bool:
        return all(flags[flag] for flag in job_flags)

    for job, job_flags in scope.JOB_FLAGS.items():
        for dependency in sorted(_needs(job) - {"changes"}):
            assert dependency in scope.JOB_FLAGS, (
                f"{job} needs `{dependency}`, which is not a gated job; if it is "
                "genuinely ungated say so in UNGATED_JOBS"
            )
            dependency_flags = scope.JOB_FLAGS[dependency]
            holds = all(
                not runs(scope.classify(paths), job_flags)
                or runs(scope.classify(paths), dependency_flags)
                for paths in samples
            )
            permissive = scope.is_permissive_dependency(job, dependency)
            expression = _if(job)
            clause = f"needs.{dependency}.result"
            if holds:
                assert not permissive, (
                    f"PERMISSIVE_DEPS declares ({job}, {dependency}) but the "
                    "module already guarantees the implication; the permissive "
                    "clause would accept a skip that cannot happen"
                )
                assert f"{clause} == 'success'" in expression, (
                    f"{job} does not require `{dependency}` to have succeeded "
                    f"(if: {expression!r})"
                )
                assert f"{clause} == 'skipped'" not in expression, (
                    f"{job} accepts a SKIPPED `{dependency}` although the module "
                    "guarantees it cannot be skipped when this job runs"
                )
            else:
                assert permissive, (
                    f"({job}, {dependency}) does not satisfy the implication "
                    "`types(" + job + ") => types(" + dependency + ")`, so a "
                    "deliberately skipped dependency silently skips this job. "
                    "Declare the pair in PERMISSIVE_DEPS with a reason, or widen "
                    f"the `{dependency}` predicate."
                )
                reason = scope.PERMISSIVE_DEPS[(job, dependency)]
                assert reason.strip(), f"PERMISSIVE_DEPS has no reason for ({job}, {dependency})"
                assert f"({clause} == 'success' || {clause} == 'skipped')" in (expression), (
                    f"{job} declares ({job}, {dependency}) permissive but does "
                    f"not use the permissive clause (if: {expression!r})"
                )


def test_a_docs_only_diff_runs_two_cheap_checks_and_code_still_runs_everything() -> None:
    """A5. PR #1238's payload is the case: `docs/store/release-record.md` only.

    Docs-only must set every heavy flag false — that is the whole point of the
    change, and 16 checks for a file no gate reads was the measured pain. The
    same list with ONE `local_operator/**` path appended must set them all true,
    so a docs-heavy PR that also touches code cannot slip past a gate.

    §7's per-class minima are pinned here too, because "everything always runs"
    and "nothing runs" are the two ways this classification can regress and the
    second is invisible without naming a class that must keep its gates.

    Mutations that must fail this: invert one predicate's default; drop
    `local_operator/**` from the `python` category.
    """
    scope = _scope()
    docs_only = scope.classify(["docs/store/release-record.md"])
    for flag in scope.FLAGS:
        assert docs_only[flag] is False, f"a docs-only diff set {flag}=True"
    assert docs_only["release_bump"] is False
    pr_jobs = set(_pr_jobs())
    ran = {
        job
        for job, verdict in scope.job_plan(docs_only).items()
        if verdict == "run" and job in pr_jobs
    }
    assert ran == {"changes", "version-bump-guard", "ci-gate"}, (
        "a docs-only PR must cost three checks — the classifier, the version "
        "guard and the CI Gate aggregate — not a matrix. "
        f"This run would report {sorted(ran)}"
    )

    with_code = scope.classify(["docs/store/release-record.md", "local_operator/cli.py"])
    skipped = [flag for flag in scope.FLAGS if with_code[flag] is not True]
    assert not skipped, (
        f"a docs file next to `local_operator/cli.py` left {skipped} false; a "
        "docs-heavy PR that also changes code must still run every gate"
    )

    # §7: `tests/**`-only may safely skip the audit, both sanity jobs, the
    # Windows leg and the context budget (none of them reads tests/), while a
    # `scripts/**`-only diff keeps the context budget only when it touches the
    # two scripts that job is built from.
    tests_only = scope.classify(["tests/unit/test_ci_hygiene.py"])
    for flag in ("lint", "types", "unit", "tui"):
        assert tests_only[flag] is True, f"a tests-only diff must still run {flag}"
    for flag in ("budget", "windows", "audit", "cli", "server", "packaging"):
        assert tests_only[flag] is False, (
            f"a tests-only diff set {flag}=True, but nothing in that job reads "
            "tests/ (see the design's per-class cost analysis)"
        )
    scripts_only = scope.classify(["scripts/shard_tests.py"])
    assert scripts_only["budget"] is False
    assert scope.classify(["scripts/bench_context_budget.py"])["budget"] is True


def test_a_python_file_under_docs_is_live_because_flake8_walks_the_whole_tree() -> None:
    """A6. Six tracked `.py` files live under `docs/` (`git ls-files docs/**/*.py`).

    `flake8 .` and `isort --check .` walk the whole tree — `.flake8` excludes
    only `.venv` — and `[tool.pyright] exclude` is the only reason pyright spares
    them. A `docs/**` prefix allowlist would therefore skip `lint` on a diff the
    linter reads, which is the correction this assertion exists to make.

    The named file is asserted to exist so the case cannot rot into a path
    nobody has.

    Mutation that must fail this: gate `lint` on `not path.startswith("docs/")`.
    """
    scope = _scope()
    sample = "docs/store/assets/build_assets.py"
    assert (REPO / sample).is_file(), f"{sample} is gone; this case is stale"
    flags = scope.classify([sample])
    assert (
        flags["lint"] is True
    ), "a `docs/**/*.py` file left lint=false; `flake8 .` reads that file"
    assert flags["types"] is True
    assert scope.category_of(sample) != scope.CAT_DOCS, (
        "a Python file under docs/ was classified inert; only non-.py/.pyi docs " "paths are inert"
    )


def test_extension_and_web_only_diffs_route_to_the_right_jobs() -> None:
    """A7. Two inertness claims that point in opposite directions.

    `extension/**` is NOT inert: `tests/unit/browser_bridge/test_bridge_wedge.py`
    asserts `gen_ts.main(["--check"]) == 0` against the committed
    `extension/src/protocol.gen.ts`, so a UNIT test reads that tree — the unit
    matrix gates on it, not just the type-check job.
    `local_operator/mobile/web/**` IS inert for every Python job: it ships as
    package data, but no test reads it, its `dist/` is gitignored so a
    source-only change alters no committed artifact, and `mobile-web.yml` owns it
    behind its own paths filter.

    Mutations that must fail this: add `extension` to the `unit` inert set; or
    drop `web` from it.
    """
    scope = _scope()
    protocol = "extension/src/protocol.gen.ts"
    assert (REPO / protocol).is_file(), f"{protocol} is gone; this case is stale"
    extension = scope.classify([protocol])
    assert extension["unit"] is True, (
        "an `extension/**` change skipped the unit matrix, but "
        "tests/unit/browser_bridge/test_bridge_wedge.py reads "
        "extension/src/protocol.gen.ts"
    )

    web = scope.classify(["local_operator/mobile/web/src/app.tsx"])
    for flag in ("lint", "types", "budget", "unit", "tui"):
        assert web[flag] is False, (
            f"a web-only diff set {flag}=True; `local_operator/mobile/web/**` is "
            "inert for every Python job"
        )


def test_packaging_is_a_separate_question_from_unit() -> None:
    """A19. `unit` asks whether the suite must run; `packaging` whether the wheel
    must be rebuilt. They differ on exactly one class, and that class is the
    reason the flag exists.

    Sharing `unit` was safe but expensive: the jobs build and install the
    distribution the release ships, and `tests/` is in no distribution this
    repository builds (`[tool.setuptools.packages.find] include =
    ["local_operator*"]`, and `MANIFEST.in` grafts nothing test-shaped). So a
    tests-only PR paid for a wheel build plus three install-smoke legs whose
    inputs it could not touch -- four jobs per run, measured on run
    41233849362 (`test/fallback-sweep`: 19 jobs, of which packaging contributed
    4).

    The two adversarial cases Aida's rule names are here as assertions, not as
    prose: a file under `docs/` that the linter or a test actually reads is NOT
    the `docs` category (A6 pins the same rule from the other side), and a
    mobile-web change that also touches a shared i18n catalogue is a live diff
    (the catalogue is read by the i18n job, and this predicate is a SUBSET test
    on a closed set, so any unenumerated category arms the flag).

    Mutations that must fail this: derive `packaging` from `live`/`unit`; drop
    the `PACKAGING_SCRIPTS` arm; widen `PACKAGING_INERT_CATEGORIES` with any
    live category (e.g. `scripts` or `i18n`).
    """
    scope = _scope()

    tests_only = scope.classify(["tests/unit/test_ci_hygiene.py"])
    assert tests_only["unit"] is True, "a changed test must still run the suite"
    assert tests_only["packaging"] is False, (
        "a tests-only diff would rebuild the wheel, but no distribution this repo "
        "builds contains tests/"
    )
    plan = scope.job_plan(tests_only)
    assert plan["packaging-smoke-build"] == "skip", f"packaging still runs: {plan}"
    assert (
        scope.classify(["docs/a.md", "tests/unit/test_x.py"])["packaging"] is False
    ), "docs next to tests must not arm a wheel rebuild either"

    # The packaging jobs' own inputs always arm it, so a change to the slimmer or
    # the size budget still exercises what it changed.
    for script in sorted(scope.PACKAGING_SCRIPTS):
        assert (REPO / script).is_file(), f"{script} is gone; this case is stale"
        assert scope.classify([script])["packaging"] is True, f"{script} did not arm packaging"

    # Adversarial 1: a `docs/**` path that is really code. `category_of` sends
    # `.py`/`.pyi` under docs/ to `other` (live), so it can never be the inert
    # class this flag keys on.
    docs_code = "docs/store/assets/build_assets.py"
    assert scope.category_of(docs_code) != scope.CAT_DOCS
    assert (
        scope.classify([docs_code])["packaging"] is True
    ), "a Python file under docs/ skipped the wheel rebuild"

    # Adversarial 2: a mobile-web change that also edits a shared i18n catalogue.
    # The catalogue is a live category, so the subset test fails and packaging
    # runs -- and i18n does too.
    mixed = scope.classify(["local_operator/mobile/web/src/app.tsx", "i18n/ledger.json"])
    assert mixed["packaging"] is True, "an i18n catalogue edit skipped the wheel rebuild"
    assert mixed["i18n"] is True, "an i18n catalogue edit skipped the i18n job"
    for audit_path in ("i18n/audit-history.json", "i18n/audit-history/fr/wire.settings.json"):
        assert scope.classify([audit_path])["i18n"] is True

    # Everything else -- manifest, an unenumerated path, a package file -- arms it.
    for paths in (
        ["pyproject.toml"],
        ["newdir/unknown.bin"],
        ["local_operator/cli.py"],
        ["scripts/shard_tests.py"],
    ):
        assert scope.classify(paths)["packaging"] is True, (
            f"{paths} skipped the wheel rebuild; the predicate must be a subset of "
            "a closed inert set, not a denylist"
        )


def test_a_version_only_manifest_edit_is_a_release_bump_and_a_dependency_edit_is_not() -> None:
    """A8. `release_bump` is decided by the LINE PAIR, not by the filename.

    A version-only `pyproject.toml` diff is the release procedure's own diff, and
    it runs `version-bump-guard` and nothing else — the release owner should not
    pay for a matrix on a one-line number change. A dependency or metadata edit
    in the SAME file must NOT be read as a bump: it changes what gets installed,
    so it keeps the whole matrix and the audit.

    Mutations that must fail this: match the filename instead of the `version =`
    line pair; or accept any `-`/`+` pair.
    """
    scope = _scope()
    bump = scope.classify(["pyproject.toml"], VERSION_BUMP_DIFF)
    for flag in scope.FLAGS:
        assert bump[flag] is False, f"a version-only bump set {flag}=True"
    assert bump["release_bump"] is True

    dependency = scope.classify(["pyproject.toml"], DEPENDENCY_DIFF)
    assert dependency["release_bump"] is False, (
        "a dependency edit was read as a release bump; that would drop the audit "
        "and the whole matrix from the PR that changed the install"
    )
    assert all(
        dependency[flag] is True for flag in scope.FLAGS
    ), "a dependency edit must keep every gate: it changes what gets installed"

    # No diff at all cannot be a bump (the refinement needs the line pair).
    assert scope.classify(["pyproject.toml"])["release_bump"] is False


def test_a_diff_the_classifier_cannot_read_fails_open(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A9. Fail-open paths must run EVERY job, warn, and exit 0.

    Three ways the classifier can be unable to decide: an unrecognised path, a
    base commit it cannot resolve, and a `git diff` that failed. All three must
    produce every flag true, a `::warning::` naming why, and exit code 0 — the
    run must proceed rather than refuse, because this module's verdict is only
    "how much to run" and refusing would red a PR for an infrastructure reason
    and teach people to bypass the gate. The warning is what keeps a full run
    legible as "could not decide" rather than as "decided everything matters".

    Mutations that must fail this: return all-false in the failure branch; drop
    the warning; or exit non-zero.
    """
    scope = _scope()

    unknown = scope.classify(["foo.bar", "newdir/x"])
    assert all(unknown[flag] is True for flag in scope.FLAGS), (
        "an unrecognised path did not fail open; an unenumerated path is exactly "
        "the case a denylist gets wrong"
    )

    def output_flags(path: Path) -> dict[str, str]:
        return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)

    # (a) An unresolvable base.
    unresolvable = tmp_path / "out-a"
    rc = scope.main(
        [
            "--event",
            "pull_request",
            "--base",
            "0000000000000000000000000000000000000000",
            "--github-output",
            str(unresolvable),
            "--root",
            str(REPO),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, "an unresolvable base must not fail the run"
    assert "::warning" in captured.out, "an unresolvable base must warn"
    assert set(output_flags(unresolvable).values()) == {
        "true"
    }, "an unresolvable base must set every flag true"

    # (b) A `git diff` failure, with the base itself resolvable.
    monkeypatch.setattr(
        scope,
        "collect_paths",
        lambda *args, **kwargs: None,
    )
    diff_failed = tmp_path / "out-b"
    rc = scope.main(
        [
            "--event",
            "pull_request",
            "--base",
            "HEAD",
            "--github-output",
            str(diff_failed),
            "--root",
            str(REPO),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, "a failed `git diff` must not fail the run"
    assert "::warning" in captured.out, "a failed `git diff` must warn"
    assert set(output_flags(diff_failed).values()) == {"true"}

    # (c) A real unrecognised path reaching the writer, warning included.
    monkeypatch.setattr(
        scope,
        "collect_paths",
        lambda *args, **kwargs: ["newdir/unrecognised.bin"],
    )
    monkeypatch.setattr(scope, "manifest_diff", lambda *args, **kwargs: "")
    unrecognised = tmp_path / "out-c"
    rc = scope.main(
        [
            "--event",
            "pull_request",
            "--base",
            "HEAD",
            "--github-output",
            str(unrecognised),
            "--root",
            str(REPO),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "::warning" in captured.out and "Unrecognised" in captured.out, (
        "an unrecognised path must warn with the path named: " f"{captured.out!r}"
    )
    assert set(output_flags(unrecognised).values()) == {"true"}
    # Every written value is exactly 'true'/'false': the empty-string case is
    # what `!= 'false'` would read as "run", and an Actions UI cannot tell it
    # apart from a step that never ran.
    for path in (unresolvable, diff_failed, unrecognised):
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            assert value in {"true", "false"}, f"{name} was written as {value!r}"


def test_a_rename_out_of_the_inert_set_classifies_both_sides() -> None:
    """A10. `--name-status -M` emits `R100<TAB>old<TAB>new`.

    A rename out of `docs/` is a Python change, and a rename INTO `docs/` is
    only inert if nothing live moved. Classifying the new path alone gets the
    second case wrong in the dangerous direction: `local_operator/a.py ->
    docs/a.md` would read as a docs-only diff and skip every Python gate even
    though a module left the tree.

    Mutation that must fail this: classify only the new path.
    """
    scope = _scope()
    into_docs = scope._parse_name_status("R100\tlocal_operator/a.py\tdocs/a.md\n")
    assert set(into_docs) == {
        "local_operator/a.py",
        "docs/a.md",
    }, f"a rename's old side was dropped: {into_docs!r}"
    assert (
        scope.classify(into_docs)["lint"] is True
    ), "a rename out of `local_operator/` into `docs/` classified as inert"
    assert (
        scope.classify(["docs/a.md"])["lint"] is False
    ), "the new side alone is inert, which is exactly why both sides are read"
    out_of_docs = scope._parse_name_status("R100\tdocs/a.md\tlocal_operator/a.py\n")
    assert scope.classify(out_of_docs)["lint"] is True


def test_local_commands_are_safe_and_track_the_ci_steps_they_mirror() -> None:
    """A11. `make check-changed` must run the same tools CI runs, safely.

    Four claims: every gated job that can run locally has commands; no job is
    excluded locally without a reason; every command's TOOL appears in that
    job's `run:`/`uses:` evidence in ci.yml; and no command execs a console
    script directly. The first-or-two-token comparison the design sketched is
    deliberately widened to the tool name, because the local spelling differs
    from CI's BY DESIGN — `.venv/bin/python -m flake8 .` versus `flake8 .` — and
    a literal token comparison would force the bare console script back in, the
    #423 path this repo has already been bitten by once.

    Mutations that must fail this: change the module's flake8 invocation to a
    tool the job does not run; drop a command from a local job; or spell a gate
    as `.venv/bin/flake8`.
    """
    scope = _scope()

    assert not (
        set(scope.JOB_COMMANDS) & set(scope.LOCAL_EXCLUSIONS)
    ), "a job is both given local commands and declared locally excluded"
    assert set(scope.JOB_COMMANDS) | set(scope.LOCAL_EXCLUSIONS) == set(scope.JOB_FLAGS), (
        "every gated job must either have local commands or a stated reason it "
        "cannot: missing="
        f"{sorted(set(scope.JOB_FLAGS) - set(scope.JOB_COMMANDS) - set(scope.LOCAL_EXCLUSIONS))}"
    )

    for job, reason in scope.LOCAL_EXCLUSIONS.items():
        assert reason.strip(), f"{job} is excluded from the local run with no reason"

    for job, commands in scope.JOB_COMMANDS.items():
        assert commands, f"{job} has no local commands"
        evidence = _tool_key(_job_evidence(job))
        for command in commands:
            tool = _tool_key(scope._invoked_tool(command))
            assert tool, f"{job}: {command!r} invokes nothing recognisable"
            assert tool in evidence, (
                f"{job}: the local command {command!r} runs `{tool}`, which does "
                f"not appear in that job's run:/uses: steps in ci.yml — the "
                "local gate has drifted from CI"
            )
            program = _executed_program(command)
            name = Path(program).name
            # Only the interpreter and the tool runners are acceptable, and the
            # test is on the BASENAME deliberately: `_executed_program` returns
            # the whole token, so `.venv/bin/flake8` and `/abs/path/black` used
            # to pass the old `"/" in program` form — i.e. the one spelling that
            # IS the #423 hazard was the one the guard could not see (QA Q3).
            assert name == "uvx" or name == "env" or name.startswith("python"), (
                f"{job}: {command!r} execs `{program}`. Only the interpreter "
                "(`python -m …`) and `uvx` are safe spellings; a path-qualified "
                "console script (`.venv/bin/flake8`, an absolute `black`) is the "
                "#423 path — a stale shebang exits 126 and a pipeline reports 0."
            )


def test_the_local_typed_gate_is_bounded_and_reaped_by_the_wrapper() -> None:
    """A16. The typed gate is bounded and reaped, and never narrowed.

    `pyright` is a Python wrapper around an npm/node analyzer that runs as a
    SEPARATE process, and the fleet measured what a group nobody reaps leaves
    behind: node children re-parented to `ppid 1` holding 2.28 GB and 1.50 GB,
    one alive 81 minutes after its parent died. The tidy "`timeout` kills only the
    wrapper" story is NOT what was measured — GNU `timeout(1)` signals the group
    here, and no `timeout` spelling left an orphan on demand; what IS measured,
    and what the wrapper's own tests drive, is a leader that exits while a
    descendant lives on. So the local command must go through
    `scripts/run_bounded.py`, the bound must be at least ci.yml's own provision
    for the job AND at least the measured whole-tree range — a local bound tighter
    than either reds a loaded host, which is the worse failure — the wrapped
    command must still be the WHOLE-TREE one, and the wrapper must exist: a typo
    in the path would otherwise surface only on a real run.

    Mutations that must fail this: drop the wrapper from the command; drop the
    bound below ci.yml's own provision; narrow the wrapped command; or point the
    command at a wrapper path that is not there.
    """
    scope = _scope()
    command = scope.JOB_COMMANDS["type-check"][0]
    assert f"scripts/{scope.BOUNDED_WRAPPER_NAME}" in command, command
    wrapper = REPO / "scripts" / scope.BOUNDED_WRAPPER_NAME
    assert wrapper.is_file(), "the local gate names a wrapper script that does not exist"

    minutes = _ci_jobs()["type-check"].get("timeout-minutes")
    assert isinstance(minutes, int), "type-check has no timeout-minutes to mirror"
    tokens = shlex.split(command)
    assert "--timeout" in tokens, tokens
    bound = float(tokens[tokens.index("--timeout") + 1])
    assert bound >= minutes * 60, (
        f"the local bound {bound!r} is TIGHTER than ci.yml's provision for this job "
        f"({minutes} min, which also covers checkout, install and the protocol-sync "
        "step) — host load alone would then red a gate CI passes, which is a worse "
        "failure than a slow gate"
    )
    assert bound >= 1800.0, (
        f"the local bound ({bound}s) leaves no headroom over the measured whole-tree "
        "range (508 s quiet, 1170 s under load) on this fleet"
    )
    tool = scope._invoked_tool(command)
    assert tool == "pyright", "the drift assertion must see the wrapped tool, not the wrapper"
    assert "pyright" in tokens, "the wrapper must still wrap pyright itself"
    wrapper_at = next(
        index
        for index, token in enumerate(tokens)
        if Path(token).name == scope.BOUNDED_WRAPPER_NAME
    )
    inner = ci_scope._unwrap_bounded(tokens[wrapper_at + 1 :])
    # The reader splits at the SEPARATOR. The flag-counting body it replaced came
    # out right for flags BEFORE the separator and wrong once the PAYLOAD carried
    # one of its own — the two shapes asserted below — which is how a directory
    # (`tests`) once got compared against ci.yml.
    assert scope._unwrap_bounded(
        ["--timeout", "10", "--grace", "2", "--", ".venv/bin/python", "-m", "pyright", "tests"]
    ) == [".venv/bin/python", "-m", "pyright", "tests"]
    assert scope._unwrap_bounded(["--timeout", "10", "--grace", "2"]) == []
    # The shapes that actually mis-sliced: a wrapper invoked with NO flags (the
    # payload's own `--timeout` used to be treated as the wrapper's), and a payload
    # carrying its own `--grace`. Both returned `[]`/`tests` through the flag-counting
    # version, so these are the assertions that fail if it ever comes back.
    assert scope._unwrap_bounded(["--", ".venv/bin/python", "-m", "pyright", "--timeout", "5"]) == [
        ".venv/bin/python",
        "-m",
        "pyright",
        "--timeout",
        "5",
    ]
    assert scope._unwrap_bounded(["--timeout", "10", "--", "pyright", "tests", "--grace", "2"]) == [
        "pyright",
        "tests",
        "--grace",
        "2",
    ]
    # rc=124 is only attributable to the bound when the command really is wrapped:
    # anything else returning 124 must NOT be told to raise this bound (review
    # round 2, N3), so the remedy is gated on this answer rather than on the code.
    assert scope._command_bound("/bin/sh -c 'exit 124'") is None
    assert scope._command_bound(".venv/bin/python scripts/run_bounded.py -- cmd") is None
    assert str(scope._command_bound(command)).startswith("--timeout ")
    assert inner[-1] == ".", f"the bound wraps a whole-tree pyright, not a file list: {inner!r}"
    assert not any(token.endswith(".py") and "/" in token for token in inner), inner


def test_status_only_gates_request_the_command_status_contract() -> None:
    """The local status-only gates name `--contract=command-status` explicitly.

    The default `cleanup-required` contract refuses before spawn wherever no
    authoritative census backend exists (Darwin), which turned `make type-check`
    and `make check-changed` into 125 on every Mac (PR #3274, review R3/Q1).
    These gates consume pyright's status, not a cleanup certificate, so they opt
    into command-status -- which still enforces the memory grant and sweeps the
    group. CI's Linux serial-control wrapper deliberately stays strict.

    Mutations that must fail this: drop the flag from the Makefile, from
    JOB_COMMANDS, from either AGENTS.md gate spelling or from the bench script.
    """
    flag = "--contract=command-status"
    scope = _scope()
    assert flag in shlex.split(scope.JOB_COMMANDS["type-check"][0])
    recipe = _makefile_recipe_blocks()["type-check"].replace("\\\n", " ")
    assert flag in shlex.split(recipe), recipe
    agents = (REPO / "AGENTS.md").read_text()
    spellings = re.findall(r"scripts/run_bounded\.py[^\n`]*-m pyright", agents)
    assert len(spellings) >= 2, spellings
    assert all(flag in spelling for spelling in spellings), spellings
    bench = (REPO / "scripts" / "bench_send_admission_ab.sh").read_text()
    assert re.search(r"run_bounded\.py\" " + re.escape(flag), bench), "bench wrapper"
    serial = FULL_SUITE_YML.read_text()  # the serial controls moved with `test`
    assert "run_bounded.py --timeout 400" in serial, "CI serial wrapper stays strict"


def test_the_makefile_typed_gate_is_the_same_command_as_the_job() -> None:
    """A17. The Makefile target is a SECOND copy of the command, so it is guarded.

    `make type-check` spells the bounded pyright out itself; `JOB_COMMANDS` spells
    it for `make check-changed`. Two spellings of one gate is exactly the drift
    this file exists to prevent, and de-wrapping the recipe alone used to leave
    both A16 and A11 green (review round 1, MINOR 2). The forms are compared with
    the bound value abstracted, so a change to either the wrapper, the flags, the
    payload or the bound fails here, and the two DEFAULTS are asserted equal so the
    Makefile cannot quietly bound itself differently from the runner.

    Mutations that must fail this: textually unwrap the recipe; change its
    `BOUND_TIMEOUT` default; add a SECOND `BOUND_TIMEOUT` assignment (which wins
    silently, since the token comparison abstracts the value); narrow its
    payload to a file list; or raise the wrapper's node heap floor to a value
    ci.yml does not set.
    """
    scope = _scope()
    job_tokens = shlex.split(scope.JOB_COMMANDS["type-check"][0])
    job_tokens[job_tokens.index(str(scope.BOUNDED_GATE_TIMEOUT))] = "BOUND"

    # Read from the file rather than through `make -n`: `make` would resolve the
    # override and print whatever it was given, not the default a developer gets.
    # `_makefile_recipe_blocks` keeps the target association a flat list loses.
    recipe = _makefile_recipe_blocks()["type-check"].replace("\\\n", " ")
    make_tokens = shlex.split(recipe)
    assert "$(BOUND_TIMEOUT)" in make_tokens, make_tokens

    # Heap sizing lives in the shared wrapper, so compare the entire recipe:
    # neither spelling may quietly add flags or bypass the wrapper's environment.
    make_tokens[make_tokens.index("$(BOUND_TIMEOUT)")] = "BOUND"
    assert make_tokens == job_tokens, (
        "the Makefile target and JOB_COMMANDS have drifted apart: "
        f"make={' '.join(make_tokens)!r} job={' '.join(job_tokens)!r}"
    )
    text = MAKEFILE.read_text()
    assignments = re.findall(r"^BOUND_TIMEOUT\s*[:?+]?=", text, re.MULTILINE)
    assert len(assignments) == 1, (
        f"the Makefile assigns BOUND_TIMEOUT {len(assignments)} times: a second "
        "assignment silently wins, and the token comparison above abstracts the "
        "value away (review round 2, M2)"
    )
    default = re.search(r"^BOUND_TIMEOUT \?= (\d+)$", text, re.MULTILINE)
    assert default, "the Makefile has no BOUND_TIMEOUT default"
    assert int(default.group(1)) == scope.BOUNDED_GATE_TIMEOUT, (
        f"the Makefile bounds itself at {default.group(1)}s while the runner uses "
        f"{scope.BOUNDED_GATE_TIMEOUT}s"
    )
    # The shared wrapper owns local heap sizing; guard its floor against CI so
    # the scoped fallback cannot regress to node's OOM-prone default cap.
    local_heap = run_bounded.NODE_HEAP_MB
    ci_heaps = []
    for step in _steps("type-check"):
        node_options = str((step.get("env") or {}).get("NODE_OPTIONS"))
        match = re.search(r"--max-old-space-size=(\d+)", node_options)
        if match:
            ci_heaps.append(match)
    assert ci_heaps, "ci.yml's type-check job no longer sets a node old-space NODE_OPTIONS"
    # CI may exceed the local cap (local runs are deliberately memory-bounded;
    # CI was raised to 8192 after a 6 GB OOM, run 42991603157) but must never
    # drop below it, or the scoped fallback would be stricter than the gate.
    assert int(ci_heaps[0].group(1)) >= local_heap, (
        f"the wrapper raises node's heap to {local_heap} MB while ci.yml "
        f"sets only {ci_heaps[0].group(1)} MB — CI must not fall below the local floor"
    )
    # The pyright step's recorded cap must match the NODE_OPTIONS it measures against.
    pyright_step = next(s for s in _steps("type-check") if s.get("name") == "Run pyright")
    assert f"heap_mb={ci_heaps[0].group(1)}\n" in pyright_step["run"], (
        "the pyright step's heap_mb warning threshold drifted from its NODE_OPTIONS"
    )
    assert job_tokens[-1] == ".", "both spellings stay whole-tree"


@pytest.mark.parametrize(
    ("node_options", "raise_heap"),
    [
        (None, True),
        ("", True),
        ("--trace-warnings", True),
        ("--trace-warnings --max-old-space-size=2048", True),
        (f"--max-old-space-size={run_bounded.NODE_HEAP_MB}", False),
        ("--max-old-space-size=8192", False),
        ("--max-old-space-size=8192 --max-old-space-size=2048", True),
        ("--max-old-space-size=2048 --max-old-space-size=8192", False),
        ("--max-old-space-size 8192", True),
        ('"--max-old-space-size=8192', True),
        ("--max_old_space_size=8192", False),
    ],
)
def test_the_bounded_wrapper_raises_the_node_heap_only_when_needed(
    node_options: str | None, raise_heap: bool
) -> None:
    """A18. Exercise the child environment, not just A17's constant/command text.

    An always-raise mutant would lower larger operator values; a missing-value-
    only mutant would leave smaller values exposed. Probe a real wrapped child
    so deleting Popen's env argument also fails, without allocating node's heap.
    """
    env = dict(os.environ)
    env.pop("NODE_OPTIONS", None)
    if node_options is not None:
        env["NODE_OPTIONS"] = node_options
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_bounded.py"),
            # command-status: both contracts build the child env through the one
            # `_child_environment`, and the default refuses before spawn on Darwin.
            "--contract=command-status",
            "--timeout",
            "15",
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ.get('NODE_OPTIONS', ''))",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    expected = node_options or ""
    if raise_heap:
        expected = (f"{expected} " if expected else "") + (
            f"--max-old-space-size={run_bounded.NODE_HEAP_MB}"
        )
    assert proc.stdout.rstrip("\n") == expected


@pytest.mark.skipif(shutil.which("node") is None, reason="real heap probe requires node")
@pytest.mark.parametrize(
    ("node_options", "expected_cap"),
    [
        ('"--max-old-space-size=8192"', 8192),
        ('--max-old-space-size="8192"', 8192),
        ('--trace-warnings "--max_old_space_size=8192" --no-warnings', 8192),
        ("--max-old-space-size=+8192", 8192),
        ('--title=" --max-old-space-size=8192 "', run_bounded.NODE_HEAP_MB),
        ('--max-old-space-size=8192 --max-old-space-size="2048"', run_bounded.NODE_HEAP_MB),
        ('--max-old-space-size=2048 "--max-old-space-size=8192"', 8192),
    ],
)
def test_the_bounded_wrapper_preserves_real_node_heap_semantics(
    node_options: str, expected_cap: int
) -> None:
    """R1: quoting belongs to option tokens, not substrings inside other values.

    Compare real V8 limits with a same-node control rather than hard-coding its
    non-old-space overhead. These are limit probes, not multi-GB allocations.
    The old raw regex lowers quoted/positive caps and mistakes --title's value
    for an option; each of those mutants must fail on an actual node child.
    """
    node = shutil.which("node")
    assert node is not None
    probe = [node, "-e", 'console.log(require("v8").getHeapStatistics().heap_size_limit/1048576)']
    env = dict(os.environ)
    env["NODE_OPTIONS"] = f"--max-old-space-size={expected_cap}"
    control = subprocess.run(probe, env=env, capture_output=True, text=True, timeout=30)
    assert control.returncode == 0, control.stderr
    env["NODE_OPTIONS"] = node_options
    wrapped = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_bounded.py"),
            "--contract=command-status",  # same env builder; strict refuses on Darwin
            "--timeout",
            "15",
            "--",
            *probe,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert wrapped.returncode == 0, wrapped.stderr
    assert float(wrapped.stdout.strip()) == float(control.stdout.strip()), (
        node_options,
        wrapped.stdout,
        control.stdout,
    )


def test_ci_and_make_share_one_classifier_module() -> None:
    """A12. Two hand-written mappings are how CI and the local gate drift apart.

    The `changes` job and `make check-changed` must name the SAME module, and
    the outputs that job exports must be exactly the module's `FLAGS`: a flag
    added in one place and not the other is either dead YAML or a gate reading
    an output that does not exist.

    Mutations that must fail this: inline the mapping in the Makefile; add a
    flag to the module and not to the `outputs:` block; replace the merge base
    with `git rev-parse origin/main` (the assertion reads the target's whole
    recipe, so that still has to fail — a bare `"--since" in recipe` check
    could not see it).
    """
    scope = _scope()
    classify_run = "\n".join(str(step.get("run") or "") for step in _steps("changes"))
    assert classify_run.count(f"{CI_SCOPE_REL}") >= 2, (
        "the `changes` job must invoke the classifier from the base revision AND "
        f"have the checked-in fallback; run: {classify_run!r}"
    )
    assert set(_ci_jobs()["changes"]["outputs"]) == set(scope.FLAGS), (
        "the `changes` job's outputs and the module's FLAGS disagree: "
        f"outputs-only={sorted(set(_ci_jobs()['changes']['outputs']) - set(scope.FLAGS))}, "
        f"module-only={sorted(set(scope.FLAGS) - set(_ci_jobs()['changes']['outputs']))}"
    )
    blocks = {
        target: body for target, body in _makefile_recipe_blocks().items() if CI_SCOPE_REL in body
    }
    assert list(blocks) == ["check-changed"], (
        "expected exactly one Makefile target invoking the shared classifier, "
        f"found {sorted(blocks)}"
    )
    recipe = blocks["check-changed"]
    assert "merge-base" in recipe, (
        "the classifier target must pass the MERGE BASE with `--since` rather "
        "than `origin/main` itself: a two-dot diff against a moved origin/main "
        "sweeps in every commit main landed since this branch was cut. "
        f"recipe: {recipe!r}"
    )
    assert "--since" in recipe, f"no `--since` in the target: {recipe!r}"
    assert (
        "--run" in recipe
    ), f"the classifier target must actually run the selected gates: {recipe!r}"


def test_the_classifier_runs_from_the_base_revision_not_the_pull_request_copy(
    tmp_path: Path,
) -> None:
    """A13. A PR that edits the classifier must not disarm the gates it classifies.

    The `changes` step reads the BASE commit's copy (`git show
    "$base:scripts/ci_scope.py"` into `$RUNNER_TEMP`) and runs that; the only
    reason to run the checked-in copy is the window in which the base has none,
    and that path is `--all` (every flag true) with a warning. The whole scheme
    is self-consistent because any `.github/**` diff also sets every flag true,
    so this PR itself runs the full job set.

    Mutations that must fail this: run the working-tree copy; make `--all`
    return false for one flag.
    """
    classify_run = "\n".join(str(step.get("run") or "") for step in _steps("changes"))
    assert (
        "git rev-parse HEAD^1" in classify_run
    ), "the classifier no longer pins the base to the merge commit's first parent"
    assert (
        "git cat-file -e" in classify_run
    ), "the classifier does not check that the base revision HAS a classifier"
    assert f'git show "$base:{CI_SCOPE_REL}"' in classify_run, (
        "the classifier is not read from the base revision, so a PR could edit "
        "the code that decides whether its own gates run"
    )
    assert (
        "$RUNNER_TEMP/ci_scope.py" in classify_run
    ), "the base copy must be run from a path outside the PR's tree"
    assert "--all" in classify_run, "the no-classifier-at-base fallback must run everything"
    assert "::warning" in classify_run, (
        "the fallback must warn, or a full run is indistinguishable from a "
        "classification that decided everything matters"
    )

    scope = _scope()
    output = tmp_path / "all-output"
    rc = scope.main(["--all", "--github-output", str(output)])
    assert rc == 0
    values = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    assert set(values) == set(scope.FLAGS), f"`--all` wrote {sorted(values)}, not every flag"
    assert set(values.values()) == {"true"}, f"`--all` must set every flag true, got {values!r}"

    # An EMPTY change set is a different thing from an unreadable one: it is a
    # local no-op and selects nothing. `collect_paths` returns None (not []) when
    # git failed, which is the fail-open case asserted above.
    empty = scope.classify([])
    assert all(empty[flag] is False for flag in scope.FLAGS), (
        "an empty change set is not a full run; the fail-open case is an "
        "UNRESOLVABLE diff, which is asserted separately"
    )


def test_a_local_run_classifies_the_diff_instead_of_running_everything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A gap the design did not name: `make check-changed` must CLASSIFY.

    "Every flag true" is the answer for a real CI event that is not a
    `pull_request` (D5/D7). It is NOT the answer for a local run, which has no
    event at all — and a module that resolved a missing event to `local` and
    then compared `event != 'pull_request'` would run the entire job set on
    every `make check-changed`, making the local command a slower spelling of
    "run everything" and silently dropping the diff it was handed.

    Mutation that must fail this: resolve a missing event to a non-
    `pull_request` name, or short-circuit on the resolved event rather than on
    the presence of an actual CI event.
    """
    scope = _scope()
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setattr(
        scope, "collect_paths", lambda *args, **kwargs: ["docs/store/release-record.md"]
    )
    monkeypatch.setattr(scope, "manifest_diff", lambda *args, **kwargs: "")

    local_output = tmp_path / "local-output"
    rc = scope.main(["--since", "HEAD", "--github-output", str(local_output), "--root", str(REPO)])
    assert rc == 0
    local_flags = dict(
        line.split("=", 1) for line in local_output.read_text().splitlines() if "=" in line
    )
    assert set(local_flags) == set(scope.FLAGS)
    assert set(local_flags.values()) == {"false"}, (
        "a local run with no CI event ran every job instead of classifying the "
        f"docs-only diff it was handed: {local_flags!r}"
    )

    # The CI events still short-circuit: `main` must stay unconditional.
    for event in ("push", "workflow_dispatch"):
        event_output = tmp_path / f"event-{event}"
        rc = scope.main(
            [
                "--event",
                event,
                "--base",
                "HEAD",
                "--github-output",
                str(event_output),
                "--root",
                str(REPO),
            ]
        )
        assert rc == 0
        values = dict(
            line.split("=", 1) for line in event_output.read_text().splitlines() if "=" in line
        )
        assert set(values.values()) == {
            "true"
        }, f"a `{event}` run must not be gated (D5): {values!r}"


def test_coverage_report_inherits_a_skipped_test_job_and_has_no_always() -> None:
    """A14. `coverage-report` must not get an `always()`.

    It combines the six shard artifacts. If `test` was skipped there are zero
    artifacts, and `coverage combine` over nothing goes red on a PR that changed
    nothing the unit suite reads. GitHub's default — a skipped need skips the
    dependent — is exactly the wanted behaviour here, so this job keeps
    `needs: test` and acquires no status-check function. That is the deliberate
    "do nothing" answer to the whole gating change, which is why it is pinned.

    Mutation that must fail this: add `if: always()`.
    """
    expression = _if("coverage-report")
    scope = _scope()
    assert "always()" not in expression, (
        f"coverage-report grew `always()` (if: {expression!r}); a combine over "
        "zero shard artifacts goes red on a docs-only PR"
    )
    assert expression == "", (
        f"coverage-report grew an `if:` ({expression!r}); it is supposed to "
        "inherit the skip from `needs: test`"
    )
    assert _needs("coverage-report") == {
        "test"
    }, "coverage-report's only dependency must stay `test`"
    assert scope.INHERITS_SKIP == {"coverage-report": ("test",)}, (
        "the module's job plan must model the inherited skip: the `--summary` "
        "job list is the only place the real per-run job set is written down, so "
        "listing a job as running when GitHub will skip it is a false report"
    )
    plan = scope.job_plan(scope.classify(["docs/store/release-record.md"]))
    assert plan["coverage-report"] == "skip", (
        "on a docs-only diff the coverage job must be reported as skipped, "
        "because a skipped `test` skips it"
    )


def test_the_classifier_resolves_its_repo_from_the_invocation_not_its_own_path(
    tmp_path: Path,
) -> None:
    """R1/Q2. The classifier must classify in CI, from a copy run out of tree.

    This is the assertion the first round did not have, and its absence is why
    the gate shipped dead: the `changes` step runs the BASE revision's copy from
    `$RUNNER_TEMP`, so a module that derived its repository from `__file__`'s
    grandparent looked at `/home/runner/work` — one level ABOVE the checkout and
    inside no repository. `root` is also the `cwd` of every git call, so
    `git diff` exited 128, the fail-open branch fired, and EVERY pull request ran
    the whole job set behind a `::warning::` annotation. Correct classifier,
    never engaged.

    The shape here is the CI one exactly: a real checkout as the invocation
    directory, the module copied outside it, no `--root`. Both directions are
    covered, because "always all true" would also pass a one-sided test: a code
    diff must classify as every flag true, and a docs-only diff as every flag
    false.

    Mutations that must fail this: restore the `Path(__file__)…parent.parent`
    default; or drop `--root` from the `changes` step while breaking the
    default (the step passes it explicitly, and this test is what keeps the
    default honest on its own).
    """
    repo = _make_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "probe.md").write_text("base\n")
    docs_base = _commit_all(repo, "base")
    (repo / "local_operator").mkdir()
    (repo / "local_operator" / "probe.py").write_text("x = 1\n")
    code_head = _commit_all(repo, "code")
    (repo / "docs" / "notes.md").write_text("more\n")
    _commit_all(repo, "docs on top of code")

    # The module, copied OUT of the repository exactly as the step does it.
    runner_temp = tmp_path / "_temp"
    runner_temp.mkdir()
    copied = runner_temp / Path(CI_SCOPE_REL).name
    shutil.copyfile(REPO / CI_SCOPE_REL, copied)

    def classify_out_of_tree(base: str, output: Path) -> dict[str, str]:
        proc = subprocess.run(
            [
                sys.executable,
                str(copied),
                "--event",
                "pull_request",
                "--base",
                base,
                "--github-output",
                str(output),
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            "the classifier failed on a copy run from outside the checkout:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
        assert "::warning" not in proc.stdout, (
            "the classifier fell back to failing open, so it did not resolve the "
            f"checkout as its repository:\n{proc.stdout}"
        )
        return _read_flags(output)

    code = classify_out_of_tree(docs_base, tmp_path / "code-flags")
    assert set(code) == set(_scope().FLAGS)
    assert all(value == "true" for value in code.values()), (
        f"a `local_operator/**` diff from a copy in $RUNNER_TEMP left {code!r} "
        "false; this is the CI shape, where every flag must still be true"
    )

    docs = classify_out_of_tree(code_head, tmp_path / "docs-flags")
    assert all(value == "false" for value in docs.values()), (
        "a docs-only diff from a copy in $RUNNER_TEMP did not classify as inert "
        f"({docs!r}); the whole point of the gate is this case"
    )


def test_the_step_executes_the_base_revision_copy(tmp_path: Path) -> None:
    """Q4. WHICH copy of the classifier runs, tested by running the step.

    A13's substring assertions could not see the realistic defect: swap the
    invocation for the checked-in copy, keep the `git show … >
    $RUNNER_TEMP/ci_scope.py` line, and every substring still matches while the
    PR's own code decides its own gates. So this executes the step's shell body
    in a throwaway repository where the two copies write different markers, and
    asserts the BASE one ran.

    Mutation that must fail this: change the invocation from
    `python "$RUNNER_TEMP/ci_scope.py"` to the checked-in
    `python scripts/ci_scope.py` (leaving the `git show` line alone).
    """
    repo = _make_repo(tmp_path)
    marker = tmp_path / "marker.txt"
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()

    def stub(tag: str) -> str:
        # The step passes real flags; the stub ignores them and records which
        # copy of the module the shell actually executed.
        return (
            "import os\n"
            "with open(os.environ['CI_SCOPE_MARKER'], 'a') as handle:\n"
            f"    handle.write({tag!r} + '\\n')\n"
        )

    (scripts_dir / "ci_scope.py").write_text(stub("base-copy"))
    _commit_all(repo, "base")
    (scripts_dir / "ci_scope.py").write_text(stub("head-copy"))
    _commit_all(repo, "head")

    shim = tmp_path / "bin"
    shim.mkdir()
    # The step calls `python`, which on a runner is the interpreter
    # setup-python just put on PATH.
    (shim / "python").symlink_to(sys.executable)
    runner_temp = tmp_path / "_temp"
    runner_temp.mkdir()
    env = {
        **os.environ,
        "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '')}",
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_OUTPUT": str(tmp_path / "github_output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "github_summary"),
        "GITHUB_WORKSPACE": str(repo),
        "CI_SCOPE_MARKER": str(marker),
    }
    proc = subprocess.run(
        ["/bin/bash", "-c", _changes_run()],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"the step failed: {proc.stdout}\n{proc.stderr}"
    assert marker.read_text().split() == ["base-copy"], (
        "the step did not execute the BASE revision's classifier copy; a pull "
        "request that edits the classifier could therefore disarm its own gates "
        f"(marker: {marker.read_text()!r})"
    )


def test_the_cli_sanity_job_keeps_its_named_script_input() -> None:
    """R2. `cli-sanity` is the only CI surface for the streaming-contract script.

    `ci.yml` runs `python scripts/check_streaming_contract.py run.jsonl` inside
    that job, and nothing else in the repository imports the script — so without
    a named input the `cli` flag covers `{python, manifest, deps_lock, ci,
    other}` and a scripts-only diff skips the only place the script is
    exercised: the diff that changes a guard is the diff that skips it.

    The named set rather than `scripts/**` in the predicate is deliberate (see
    the module comment): widening would run both live-LLM jobs, with real spend,
    on every scripts-only diff, which the per-class cost analysis rates as safe
    to skip. The last assertion pins that the widening did NOT happen.

    Mutation that must fail this: drop the script from `CLI_SANITY_SCRIPTS`.
    """
    scope = _scope()
    script = "scripts/check_streaming_contract.py"
    assert (REPO / script).is_file(), f"{script} is gone; this case is stale"
    assert any(
        script in str(step.get("run") or "") for step in _steps("cli-sanity")
    ), f"cli-sanity no longer runs {script}; this assertion's premise is stale"

    flags = scope.classify([script])
    for flag in ("cli", "server", "audit", "lint", "types", "unit", "tui"):
        assert flags[flag] is True, (
            f"a {script}-only diff left {flag}=False; cli-sanity (and the pip-audit "
            "it depends on) would skip their only surface for that script"
        )
    plan = scope.job_plan(flags)
    assert plan["cli-sanity"] == "run" and plan["server-sanity"] == "run"

    # The deliberate non-widening: some other script keeps the old behaviour.
    other = scope.classify(["scripts/shard_tests.py"])
    assert other["cli"] is False and other["audit"] is False, (
        "the `cli`/`audit` predicate was widened to all of scripts/**; that runs "
        "the live-LLM jobs on every scripts-only diff, which is the cost this "
        "change exists to remove"
    )


def test_the_windows_input_set_covers_what_that_job_loads() -> None:
    """R3. The Windows job loads more than the test files it names.

    Its pytest run imports the `conftest.py` chain, and the root conftest
    imports the shard-side test helpers at module scope; pytest also imports
    the `__init__.py` markers for the selected modules. A change to any of
    them can change that job's behaviour without touching a file it lists, which
    is the "flag narrower than the guard's real input" pattern this classifier
    exists to avoid.

    Mutation that must fail this: drop either shard-side helper from
    `WINDOWS_LOADED_PATHS`, or drop its import from the root conftest without
    revisiting this case.
    """
    scope = _scope()
    loaded = (
        "conftest.py",
        "tests/conftest.py",
        "tests/shard_memory_dump.py",
        "tests/shard_stall_watchdog.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/unit/server/__init__.py",
        "tests/unit/test_agent_import_boundary.py",
        "tests/unit/server/test_edit_workspace_boundary.py",
        "tests/unit/network/test_credentials_private_paths.py",
        "tests/unit/conftest.py",
        "tests/unit/network/conftest.py",
        "tests/unit/network/test_credential_inherit.py",
        "tests/unit/network/test_credentials_consent.py",
        "tests/unit/network/test_approvals_store.py",
        "tests/unit/network/test_credentials_real_link.py",
        "tests/unit/network/test_credentials_owner.py",
        "tests/unit/network/test_relay_e2e.py",
        "tests/unit/team_bridge/test_connector_platforms.py",
        "tests/unit/team_bridge/__init__.py",
    )
    conftest = (REPO / "conftest.py").read_text()
    for helper in ("shard_memory_dump", "shard_stall_watchdog"):
        assert helper in conftest, (
            f"the root conftest no longer imports {helper}; WINDOWS_LOADED_PATHS "
            "can drop that name, and this assertion is why it must be revisited"
        )
    for path in loaded:
        assert (REPO / path).is_file(), f"{path} is gone; this case is stale"
        assert scope.classify([path])["windows"] is True, (
            f"a {path} change would skip filesystem-boundaries-windows, which " "loads it"
        )

    # …and a test file with no relationship to that job still skips it: the
    # point is a curated input set, not "anything under tests/".
    assert scope.classify(["tests/unit/test_ci_hygiene.py"])["windows"] is False


@pytest.mark.parametrize(
    ("path", "selected"),
    (
        ("tests/unit/network/test_credentials_private_paths.py", True),
        ("tests/unit/conftest.py", True),
        ("tests/unit/network/conftest.py", True),
        ("tests/unit/network/test_credential_inherit.py", True),
        ("tests/unit/network/test_credentials_consent.py", True),
        ("tests/unit/network/test_approvals_store.py", True),
        ("tests/unit/network/test_credentials_real_link.py", True),
        ("tests/unit/network/test_credentials_owner.py", True),
        ("tests/unit/network/test_relay_e2e.py", True),
        ("tests/unit/test_ci_hygiene.py", False),
    ),
)
def test_windows_root_input_only_changes_select_the_native_job(path: str, selected: bool) -> None:
    """A test-only diff must reach its native lane, but not widen all tests/**.

    The helpers are module-level imports of the newly selected module, not
    optional fixture bodies. Removing one curated input must fail its own row;
    broadening CAT_TESTS must fail the unrelated-test negative control.
    """
    entrypoint = "tests/unit/network/test_credentials_private_paths.py::"
    assert any(
        entrypoint in str(step.get("run") or "") for step in _steps("filesystem-boundaries-windows")
    ), "the ROOT nodes left the Windows command; revisit its curated inputs"
    scope = _scope()
    flags = scope.classify([path])
    assert flags["windows"] is selected
    assert scope.job_plan(flags)["filesystem-boundaries-windows"] == ("run" if selected else "skip")


def test_the_xplat_input_set_covers_what_the_battery_reads() -> None:
    """R4. The two probe legs read more than the battery script itself.

    `scripts/xplat_probe.py` is only half of the job's input: the `tui.boot`
    probe's `--driver tui` child imports `scripts.probe_isolation`,
    `scripts.visual_capture` and — this is the one a path-shaped guess gets
    wrong — `tests.unit.tui.test_app_pilot`, the fake session the TUI's own
    tests are written against. So a change to any of the three changes what the
    probe measures, and a predicate that names only `scripts/xplat*.py` would
    skip both legs on exactly the PR that changed the probe's own machinery.

    Both directions are pinned, because the failure modes are opposite: the
    package and the app double must SET the flag (a flag narrower than the
    job's real inputs), and an unrelated script or test file must NOT (a flag
    widened to `scripts/**` or `tests/**` would put a Windows runner and a full
    battery on every one of those diffs).

    Mutations that must fail this: drop `scripts/visual_capture.py` or
    `tests/unit/tui/test_app_pilot.py` from the input sets; add `scripts/**` or
    `tests/**` to the predicate.
    """
    scope = _scope()

    # The premise, read from the probe rather than assumed: if the driver stops
    # importing either script or the app double, this set can shrink.
    probe = (REPO / "scripts" / "xplat_probe.py").read_text()
    for imported in ("scripts.probe_isolation", "scripts.visual_capture"):
        assert imported in probe, (
            f"scripts/xplat_probe.py no longer imports {imported}; that name can "
            "leave XPLAT_SCRIPT_INPUTS, and this assertion is why it must be revisited"
        )
    loaded = (
        "local_operator/tui/app.py",
        "scripts/xplat_probe.py",
        "scripts/xplat_report.py",
        "scripts/xplat/Dockerfile.probe",
        "scripts/probe_isolation.py",
        "scripts/visual_capture.py",
        "pyproject.toml",
        "tests/unit/tui/test_app_pilot.py",
        "tests/unit/tui/__init__.py",
    )
    for path in loaded:
        assert (REPO / path).is_file(), f"{path} is gone; this case is stale"
        assert scope.classify([path])["xplat"] is True, (
            f"a {path} change would skip the xplat probe legs, which read it "
            "(the battery walks and imports the package, and its TUI driver "
            "imports the test suite's app double)"
        )

    plan = scope.job_plan(scope.classify(["scripts/xplat_probe.py"]))
    assert plan["xplat-probe-linux"] == "run" and plan["xplat-probe-windows"] == "run", (
        "a change to the battery itself did not select both probe legs: " f"{plan!r}"
    )

    # …and the deliberate non-widening, in both directions.
    other_script = scope.classify(["scripts/shard_tests.py"])
    assert other_script["xplat"] is False, (
        "the `xplat` predicate was widened to all of scripts/**; a script no "
        "probe reads now pays for a Windows runner and a full battery"
    )
    assert scope.classify(["tests/unit/test_ci_hygiene.py"])["xplat"] is False, (
        "the `xplat` predicate was widened to all of tests/**; the apps double "
        "is a curated input, not the whole suite"
    )
    assert scope.classify(["docs/store/release-record.md"])["xplat"] is False


def test_a_local_run_does_not_run_a_shared_command_twice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two probe legs share ONE local spelling, so it runs once.

    Both legs run the battery against whatever host they are handed, and a
    developer's machine is a single host — so the pair's local command is
    identical, and the battery takes minutes. Running it once per leg would
    double the slowest gate in a local run to produce the same reading twice.

    Printing nothing for the second job would be the worse fix: a selected job
    with no output reads as a job that ran nothing, which is exactly the
    "a skipped job is a claim" failure this module exists to prevent. So the
    suppression line is part of the contract, not an implementation detail.

    Mutation that must fail this: drop the dedup (two calls), or drop the line
    that names the job whose command was reused (only one banner printed).
    """
    scope = _scope()
    assert scope.JOB_COMMANDS["xplat-probe-linux"] == scope.JOB_COMMANDS["xplat-probe-windows"], (
        "the two probe legs no longer share a local spelling, which is the fact "
        "the dedup below is pinned to"
    )
    calls: list[str] = []

    class _Done:
        returncode = 0

    def fake_run(command: str, **kwargs: object) -> _Done:
        calls.append(command)
        return _Done()

    monkeypatch.setattr(scope.subprocess, "run", fake_run)
    assert scope.run_jobs(["xplat-probe-linux", "xplat-probe-windows"], REPO) == 0
    assert calls == [
        scope.JOB_COMMANDS["xplat-probe-linux"][0]
    ], f"a shared local command ran {len(calls)} times: {calls!r}"
    printed = capsys.readouterr().out
    for job in ("xplat-probe-linux", "xplat-probe-windows"):
        assert f"=== {job}:" in printed, f"{job} was selected but never mentioned: {printed!r}"
    assert "not re-run" in printed, printed


def test_an_explicit_root_is_authoritative_for_base_resolution(tmp_path: Path) -> None:
    """F2. `--root` must decide the repository for BASE resolution too.

    `resolve_base` was the only pair of git calls in the module with no `cwd`, so
    it read the PROCESS cwd while everything downstream read the resolved root.
    With the process started outside any repository — the case `default_root`'s
    middle branch exists for — a valid `--root` naming a real checkout and a
    valid base SHA still failed to resolve, the fail-open branch fired, and that
    middle branch could never turn the situation into a classification: the
    resolution order the docstring described was unreachable in a successful run.

    The discriminating direction is the DOCS-ONLY case: a fail-open run sets
    every flag true, so asserting "all true" would pass on the defect.

    Mutation that must fail this: drop `cwd` from `resolve_base`'s git calls.
    """
    repo = _make_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "probe.md").write_text("base\n")
    base = _commit_all(repo, "base")
    (repo / "docs" / "notes.md").write_text("more\n")
    _commit_all(repo, "docs on top")

    outside = tmp_path / "not-a-repository"
    outside.mkdir()
    output = tmp_path / "flags"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / CI_SCOPE_REL),
            "--event",
            "pull_request",
            "--base",
            base,
            "--root",
            str(repo),
            "--github-output",
            str(output),
        ],
        cwd=str(outside),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "::warning" not in proc.stdout, (
        "the base did not resolve even though `--root` named a repository that "
        f"contains it, so `--root` is not authoritative:\n{proc.stdout}"
    )
    flags = _read_flags(output)
    assert set(flags) == set(_scope().FLAGS)
    assert all(value == "false" for value in flags.values()), (
        "a docs-only diff classified as live from a cwd outside any repository "
        f"({flags!r}); this is the fail-open answer, not a classification"
    )


# Through twelve so a correct doc update past seven reads as a match, not as an
# unknown word that fails with "says 'eight'; the workflows give 8".
_COUNT_WORDS = {
    word: n
    for n, word in enumerate(
        "one two three four five six seven eight nine ten eleven twelve".split(), start=1
    )
}


def test_agents_doc_names_every_job_a_local_run_skips() -> None:
    """F1. The local-exclusions paragraph must not undercount or omit a job.

    That paragraph exists so a local green is not over-read, and the job it is
    most dangerous to omit is the one a reader would assume their green covers:
    `pip-audit` was added to `LOCAL_EXCLUSIONS` in the same commit that left the
    prose saying "three gated jobs", which is exactly the drift this asserts
    against. The count AND every name are checked, so bumping one without the
    other fails.

    Mutation that must fail this: add a job to `LOCAL_EXCLUSIONS` (or remove the
    `pip-audit` sentence from `AGENTS.md`) without touching the other side.
    """
    scope = _scope()
    doc = (REPO / "AGENTS.md").read_text()
    marker = "gated jobs are never"
    assert marker in doc, "the local-exclusions sentence is gone from AGENTS.md"
    sentence = re.search(r"\*\*(\w+) gated jobs are never", doc)
    assert sentence, (
        "the local-exclusions sentence changed shape; re-point this guard at it "
        "rather than deleting it"
    )
    word = sentence.group(1).lower()
    assert word in _COUNT_WORDS, f"unknown count word {word!r} in AGENTS.md"
    assert _COUNT_WORDS[word] == len(scope.LOCAL_EXCLUSIONS), (
        f"AGENTS.md says {word!r} of the {len(scope.LOCAL_EXCLUSIONS)} jobs in "
        f"LOCAL_EXCLUSIONS never run locally: {sorted(scope.LOCAL_EXCLUSIONS)}"
    )
    start = doc.index(marker)
    paragraph = doc[start - 60 : start + 1200]
    for job in scope.LOCAL_EXCLUSIONS:
        assert f"`{job}`" in paragraph, (
            f"AGENTS.md does not name `{job}` among the jobs a local run never "
            "covers, so a reader can take a local green as covering it"
        )


def test_agents_doc_states_the_check_count_a_release_bump_pr_runs() -> None:
    """M1 (#2990). The release sections' check count must match the workflows.

    `AGENTS.md` said a bump PR runs "two checks" for weeks after
    `release-gate.yml` added four more, because nothing tied the prose number
    to its source. The count is derived here, not pinned: the `ci.yml` jobs
    `job_plan` runs on a real version-only diff, plus every `release-gate.yml`
    job times its Python matrix (that file has no `if:`, so each leg reports
    on a push to the bump branch).

    NOT covered, deliberately: only the `python-version` matrix axis is
    counted, and trigger `paths:` filters are not checked. A second axis,
    `include:` legs or a path filter that stops the gate firing on a bump
    would pass unnoticed; neither exists today, so guarding them is
    speculative.

    Mutation that must fail this: add a job or interpreter to the release
    gate, or ungate a `ci.yml` job, without updating every number word.
    """
    scope = _scope()
    bump = scope.classify(["pyproject.toml"], VERSION_BUMP_DIFF)
    # ci.yml's rows only: the PR check list is `CI Gate`; the full suite runs
    # at the merge gate (`Merge gate (full suite)`), which a version-only bump
    # PR never labels.
    pr_jobs = set(_pr_jobs())
    ci_rows = [
        job for job, verdict in scope.job_plan(bump).items() if verdict == "run" and job in pr_jobs
    ]

    gate = yaml.safe_load((CI_YML.parent / "release-gate.yml").read_text())
    # PyYAML reads the bare `on:` key as the boolean True (YAML 1.1).
    branches = gate.get("on", gate.get(True))["push"]["branches"]
    assert "release-next" in branches, "the bump branch no longer triggers the release gate"
    gate_rows = 0
    for job_id, job in gate["jobs"].items():
        assert "if" not in job, f"release-gate `{job_id}` is now conditional; recount by hand"
        gate_rows += len(job.get("strategy", {}).get("matrix", {}).get("python-version", [None]))

    expected = len(ci_rows) + gate_rows
    doc = (REPO / "AGENTS.md").read_text()
    # Every place the release sections state the count, including the
    # round-1 remediation's own sentences: a stale one is the same M1 drift.
    for pattern in (
        r"release\s+bump runs (\w+) checks",
        r"\*\*Expect (\w+) checks on this PR",
        r"\*\*(\w+)\s+checks\s+appear\s+only\s+when\s+the\s+bump",
        r"read\s+the\s+(\w+)-check\s+list",
    ):
        found = re.search(pattern, doc)
        assert found, f"the sentence matching {pattern!r} changed shape; re-point this guard"
        word = found.group(1).lower()
        assert word in _COUNT_WORDS, f"teach _COUNT_WORDS the word {word!r}"
        assert _COUNT_WORDS[word] == expected, (
            f"AGENTS.md says a release-bump PR runs {word!r} checks; the workflows "
            f"give {expected}: ci.yml {sorted(ci_rows)} + {gate_rows} release-gate rows"
        )


def test_shard_plugin_bootstrap_does_not_export_a_cwd_pythonpath():
    """Child -P isolation must survive loading the checkout's ordering plugin.

    Exporting PYTHONPATH=. makes every spawned interpreter search its own cwd,
    including children deliberately protected with -P. Module launch resolves
    the early plugin without changing the environment inherited by user code.
    """
    run = _run_step("test")
    assert "python -m pytest -p tests.ci_order_owned_first" in run
    assert "PYTHONPATH=" not in run


# --- slim PR CI: the unit suites run at the merge gate ----------------------
#
# The PR path dropped `test`, `coverage-report` and `tui-e2e` (89% of its
# runner time, measured 2026-10-24..11-07) in favour of targeted local
# evidence (`scripts/affected_tests.py`) plus a `guards` job. The full suite
# then runs ONCE, at the merge gate (`pull_request: [labeled]`), and
# post-merge/nightly on `main` (which files a `ci-main-red` issue). These pins
# keep each half of that bargain from eroding silently.


def test_pull_requests_do_not_run_the_unit_suites() -> None:
    """The moved jobs exist ONLY in full-suite.yml, and run on a PR only when
    the merge gate asks for them.

    The bargain has two halves: review rounds stay cheap (no unit matrix on
    `opened`/`synchronize`), and `main` cannot go without the full suite. Both
    are pinned here — the trigger list is what keeps the first, and the
    aggregate (`test_merge_gate_aggregates_every_full_suite_job`) is what keeps
    the second from being a red pipeline reaching `main` through the label.
    """
    pr_jobs = _pr_jobs()
    assert not ({"test", "coverage-report", "tui-e2e"} & set(pr_jobs)), sorted(pr_jobs)
    full = yaml.safe_load(FULL_SUITE_YML.read_text())
    triggers = full.get("on", full.get(True))
    # `labeled` ONLY: `opened`, `synchronize` and `reopened` are the three that
    # would put the matrix back on every push, so they are refused by name.
    assert triggers["pull_request"] == {"types": ["labeled"]}, (
        "full-suite.yml's pull_request trigger widened: the full suite is a "
        "merge-gate run, not a per-push one "
        f"(pull_request={triggers.get('pull_request')!r})"
    )
    assert "pull_request_target" not in triggers
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["schedule"] and "workflow_dispatch" in triggers
    assert set(full["jobs"]) == FULL_SUITE_JOBS | {"changes"}
    for job in pr_jobs.values():
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        assert not (set(needs) & FULL_SUITE_JOBS), f"a PR job still needs a moved job: {needs}"


# --- the merge gate: the full suite runs ONCE, for a labeled PR ------------
#
# Review rounds must not pay for the full suite, and a red pipeline must never
# reach `main`. Merge queue is not available to this repository (user-owned,
# so `enableMergeQueue` does not exist and `repository.mergeQueue` is null), so
# the gate is a `labeled`-triggered run plus one required aggregate check. The
# pins below keep the trigger narrow, the aggregate unskippable, and the re-arm
# ritual present: each is a way the gate could be made to *look* green.


def test_merge_gate_aggregates_every_full_suite_job() -> None:
    """One stable required name, `always()`, unconditional, over every suite job.

    `needs` is derived, not listed: a new suite job that the aggregate does not
    wait on is a job that can go red while the required check stays green —
    the whole defect this gate exists to prevent. `report` is the one
    exclusion, and it is structural: it is the `ci-main-red` issue filer, it
    acts only on `refs/heads/main`, and making the PR gate wait on it would put
    an issue-writing job on the merge path.
    """
    jobs = yaml.safe_load(FULL_SUITE_YML.read_text())["jobs"]
    gate = jobs["merge-gate"]
    assert gate["name"] == "Merge gate (full suite)", (
        f"the required check name drifted to {gate['name']!r}: branch protection "
        "names this exact string, and a rename stops gating silently"
    )
    assert gate["if"] == "always()", (
        "the aggregate must report on EVERY run of this workflow, whatever event "
        "started it — a condition that can be false publishes `skipped`, which "
        f"satisfies a required check (if: {gate['if']!r})"
    )
    assert set(gate["needs"]) == set(jobs) - {"merge-gate", "report"}, (
        "merge-gate must need every job that can go red, or a red suite passes "
        f"it (needs={gate['needs']!r}, jobs={sorted(jobs)!r})"
    )
    assert not gate.get(
        "continue-on-error"
    ), "continue-on-error makes a required check report success for a red run"
    run = gate["steps"][-1]["run"]
    assert (
        '"failure"' in run and '"cancelled"' in run and "exit 1" in run
    ), "the aggregate must fail on a failed or cancelled need"
    assert '"skipped"' not in run, (
        "a classifier skip must pass the gate: a docs-only head legitimately "
        "skips the matrix and must still be mergeable"
    )


def test_no_merge_gate_job_is_conditional_on_the_label_name() -> None:
    """Hostile case: GitHub counts `skipped` as a passed required check.

    A job whose `if:` read the label would report `skipped` on a run started by
    ANY OTHER label, and a red PR would then merge on a check that never ran.
    The invariant is about the label CONTEXT, so both spellings are refused:
    `github.event.label.name` (singular, one label) is as good a hole as
    `github.event.pull_request.labels.*.name`, and so is any `contains(` written
    over either — a name test can be built from any of them. Asserted over
    EVERY job, not just the aggregate: a conditional suite job skips the
    coverage while the gate goes on reporting `success` for it.
    """
    jobs = yaml.safe_load(FULL_SUITE_YML.read_text())["jobs"]
    for job_id, job in jobs.items():
        condition = str(job.get("if") or "")
        assert not re.search(r"\blabels?\b", condition), (
            f"full-suite `{job_id}` tests the pull-request label in its `if:` "
            f"({condition!r}). A label-gated job reports `skipped` on a run "
            "started by another label, and `skipped` satisfies a required "
            "status check — a red PR would merge"
        )
        assert not re.search(r"contains\([^()]*\blabels?\b", condition), (
            f"full-suite `{job_id}` builds a label-name test with `contains(` "
            f"({condition!r}); the same `skipped` hole, written the long way"
        )
        # The ritual's own label, named directly: the context rules above are the
        # invariant, this is the instance the repository actually uses.
        assert "ready-to-merge" not in condition, (
            f"full-suite `{job_id}` names the re-arm label in its `if:` " f"({condition!r})"
        )


def test_no_full_suite_job_is_conditional_on_the_triggering_event() -> None:
    """A job `if:` must never depend on WHICH event started the run.

    This workflow starts on four events (`labeled` PRs, `push`, `schedule`,
    `workflow_dispatch`) and every run publishes its checks on its OWN SHA, so
    an event condition is a way to publish `skipped` on a head whose suite ran
    red — and GitHub counts `skipped` as a satisfied required check. The
    aggregate carried exactly that hole until PR #3631: `if: always() &&
    github.event_name == 'pull_request'` left `Merge gate (full suite)` =
    `skipped` on a PR head after the documented `WIDE:` `workflow_dispatch`,
    while that dispatch's suite was red (run 42501880282).

    What stays legitimate is data about the DIFF, not about the event:
    `always()`, `!cancelled()` and the `needs.changes.outputs.*` classifier
    flags. Those may skip a job — the aggregate passes a classifier skip on
    purpose, because a docs-only head must stay mergeable.
    """
    jobs = yaml.safe_load(FULL_SUITE_YML.read_text())["jobs"]
    for job_id, job in jobs.items():
        condition = str(job.get("if") or "")
        assert "github." not in condition, (
            f"full-suite `{job_id}` conditions its `if:` on the triggering event "
            f"or the repo context ({condition!r}). Every run lands its checks on "
            "the SHA it ran on and a `skipped` job satisfies a required status "
            "check, so a job must be scheduled by the classifier "
            "(`needs.changes.outputs.*`), never by which event ran"
        )
    assert (
        jobs["merge-gate"]["if"] == "always()"
    ), "the aggregate is the required check: it must be unconditional"


def test_the_makefile_rearms_the_merge_gate_with_the_label_ritual() -> None:
    """Hostile cases: a push after the gate, and a ritual that arms nothing.

    A push AFTER the gate leaves the check on the old head, where it blocks the
    merge — the intended block, but it has to be exitable by an agent with no
    UI access and no way to re-point a check at a SHA. Removing and re-adding
    the label is that exit, so the target is pinned like any other gate.

    It is also pinned for the opposite failure (QA round 1, Q-3): a ritual that
    PRINTS success without arming anything. `.[0]` on an empty run list renders
    `null`, `--limit 1` returns the PREVIOUS run on any re-arm (so `gh run
    watch` exits on an old verdict immediately), and an `|| true` removal leaves
    the label on, which makes the add a no-op, fires no `labeled` event and
    strands the head in "Expected".
    """
    makefile = MAKEFILE.read_text()
    assert re.search(r"^merge-gate: ##", makefile, re.M), (
        "`make merge-gate PR=<n>` is gone: after a push the required check "
        "sits in Expected forever with no documented way out"
    )
    recipe = _makefile_recipe_blocks()["merge-gate"]
    for needle in (
        "--remove-label ready-to-merge",
        "--add-label ready-to-merge",
        "gh label create ready-to-merge",
        "gh run watch",
        "gh pr checks",
        "PR=<number>",
    ):
        assert needle in recipe, f"the merge-gate ritual lost {needle!r}"
    # Remove-THEN-add is load-bearing, not cosmetic: `--add-label` on a label
    # that is already present is a no-op, so no `labeled` event fires and the
    # head stays in "Expected" while the ritual reports success.
    assert recipe.index("--remove-label ready-to-merge") < recipe.index(
        "--add-label ready-to-merge"
    ), "the ritual must remove the label before adding it, or no event fires"
    # It must be able to TELL whether it armed anything: the PR is validated
    # open first, the new run is resolved by head SHA + created-after (never by
    # row order), and a missing run is a non-zero exit with the next commands,
    # not a `gh run watch null`.
    assert '!= "OPEN"' in recipe, "a closed or nonexistent PR must be refused"
    assert "headRefOid" in recipe, "the ritual must know the PR's head SHA"
    assert (
        "headSha" in recipe and "createdAt" in recipe
    ), "the new run must be resolved by head SHA + created-after, not row order"
    assert (
        'jq -r --arg sha "$$head_sha"' in recipe
    ), "the run list must be filtered by the system `jq`, which does accept `--arg`"
    assert '--arg prev "$$prev"' in recipe, "the previous run's id must not match"
    # R6, measured on gh 2.100.0: `gh` has NO `--arg` flag and its `--jq` takes
    # exactly one token, so `--jq --arg sha … --arg since … --arg prev …` exits 1
    # with `unknown flag: --arg`. The recipe discarded that stderr, so `prev`
    # was always empty and the poll — which passed the same flags — never
    # matched: the success branch was unreachable and every arming reported
    # failure. The values must travel to `jq` by pipe, never to `gh`.
    for segment in re.split(r"[|;\n]", recipe):
        if re.search(r"\bgh\s+\w", segment):
            assert "--arg" not in segment, (
                "`gh` has no `--arg` flag (`unknown flag: --arg` on 2.100.0); a "
                f"jq argument must go to `jq`, not to gh: {segment.strip()!r}"
            )
    assert "--jq --" not in recipe, (
        "`--jq` takes exactly one token, so `--jq --arg` hands gh a flag it does "
        "not have and the query never runs"
    )
    # The bound is arithmetic the failure message promises out loud: 12 polls
    # 5 s apart is the "60 s" an operator reads when it gives up.
    attempts = int(re.search(r"-lt (\d+)", recipe).group(1))
    poll_pause = int(re.findall(r"sleep (\d+)", recipe)[-1])
    assert attempts * poll_pause == 60, (
        f"the poll is {attempts} attempts x {poll_pause} s = {attempts * poll_pause} s "
        "but its failure message promises 60 s"
    )
    assert "within 60 s" in recipe, "the bound must be stated where it fails"
    assert (
        "--limit 1\n" not in recipe and "--limit 1 " not in recipe
    ), "`--limit 1` cannot tell the new run from the previous one"
    assert "exit 1" in recipe, "arming nothing must exit non-zero, not print null"
    # The removal is the one call that may NOT be softened: `|| true` there is
    # what let a failed removal pass as a successful arm (QA round 1, Q-3).
    assert (
        "could not remove ready-to-merge" in recipe
    ), "a failed label removal must fail loudly, not be swallowed"
    assert recipe.count("|| true") == 1, (
        "the only tolerated soft failure is the idempotent `gh label create`; a "
        "swallowed `--remove-label` leaves the label on, so the add is a no-op, "
        "no event fires and the ritual still reports success"
    )


#: The head SHA the stubbed `gh` reports. Hex, as GitHub's `headRefOid` is.
_MERGE_GATE_SHA = "0123456789abcdef0123456789abcdef01234567"
#: Format-identical to the recipe's `date -u +%Y-%m-%dT%H:%M:%SZ` (which is what
#: it compares `createdAt` against): one run comfortably after that instant and
#: one comfortably before it, so both of the poll's predicates are exercised
#: rather than the test racing the wall clock.
_MERGE_GATE_AFTER = "2999-01-01T00:00:00Z"
_MERGE_GATE_BEFORE = "2000-01-01T00:00:00Z"
#: A stubbed `gh`: records every call, answers with the OUTPUT of the queries the
#: ritual reads, and reproduces the one flag real `gh` does not have (R6). On gh
#: 2.100.0 `gh run list … --jq --arg sha …` exits 1 with `unknown flag: --arg`;
#: the recipe discarded that stderr, so `prev` stayed empty, the poll matched
#: nothing and the ritual reported failure for every run it armed. The stub
#: rejects `--arg` the same way, so a regression fails here instead of only on a
#: real arming. Historical `run list` answers raw JSON for the real `jq`;
#: the active-run existence query also returns raw JSON for response validation.
_MERGE_GATE_GH_BODY = """
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["STUB_LOG"]).open("a") as handle:
    handle.write(" ".join(args) + "\\n")


def answer(message, code):
    print(message, file=sys.stderr if code else sys.stdout)
    raise SystemExit(code)


if "--arg" in args:
    answer("unknown flag: --arg", 1)

if args[:2] == ["pr", "view"]:
    if "state,headRefOid,headRefName" in args:
        if os.environ.get("STUB_PR_MISSING") == "1":
            answer("graphql: Could not resolve to a PullRequest", 1)
        print(os.environ["STUB_STATE"], os.environ["STUB_SHA"], os.environ["STUB_HEAD"])
        raise SystemExit(0)
    if "labels" in args:
        for label in json.loads(Path(os.environ["STUB_LABEL_STATE"]).read_text()):
            print(label)
        raise SystemExit(0)
    answer("unexpected pr view: " + " ".join(args), 3)

if args[:2] == ["label", "create"]:
    answer("Label already exists", 1)

if args[:2] == ["pr", "edit"]:
    if "--remove-label" in args and os.environ.get("STUB_REMOVE_FAIL") == "1":
        answer("could not update label", 1)
    label_state = Path(os.environ["STUB_LABEL_STATE"])
    labels = json.loads(label_state.read_text())
    if "--remove-label" in args:
        labels.remove(args[args.index("--remove-label") + 1])
    if "--add-label" in args:
        label = args[args.index("--add-label") + 1]
        if label not in labels:
            labels.append(label)
    label_state.write_text(json.dumps(labels))
    with Path(os.environ["STUB_LABEL_TRANSITIONS"]).open("a") as handle:
        handle.write(json.dumps(labels) + "\\n")
    raise SystemExit(0)

if args[:2] == ["run", "list"]:
    if "--status" in args:
        if "--commit" not in args or "--event=pull_request" not in args:
            answer("active lookup must filter head and eligible event", 3)
        status = args[args.index("--status") + 1]
        sha = args[args.index("--commit") + 1]
        failure = os.environ.get("STUB_ACTIVE_FAIL", "0")
        if failure != "0":
            counter = Path(os.environ["STUB_ACTIVE_COUNT"])
            failures = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(failures + 1))
            if failures < int(os.environ.get("STUB_ACTIVE_FAILURES", "999")):
                answer(failure, 1)
        if os.environ.get("STUB_ACTIVE_RESPONSE"):
            answer(os.environ["STUB_ACTIVE_RESPONSE"], 0)
        print(json.dumps([
            {"databaseId": run["databaseId"], "url": run["url"]}
            for run in json.loads(os.environ.get("STUB_ACTIVE_RUNS", "[]"))
            if run["headSha"] == sha and run["status"] == status
        ][:1]))
        raise SystemExit(0)
    counter = Path(os.environ["STUB_RUN_LIST_COUNT"])
    calls = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(calls + 1))
    # Call 1 is the pre-arm lookup for the run to exclude; every later call is
    # one attempt of the bounded poll.
    print(os.environ["STUB_PREV_RUNS" if calls == 0 else "STUB_POLL_RUNS"])
    raise SystemExit(0)

answer("unexpected gh call: " + " ".join(args), 3)
"""


def _merge_gate_runs(*runs: tuple[int, str]) -> str:
    """A `gh run list --json databaseId,url,headSha,createdAt` payload."""
    return json.dumps(
        [
            {
                "databaseId": run_id,
                "url": "https://github.com/damianvtran/local-operator/actions/runs/" f"{run_id}",
                "headSha": _MERGE_GATE_SHA,
                "createdAt": created_at,
            }
            for run_id, created_at in runs
        ]
    )


def _arm_merge_gate(
    tmp_path: Path,
    *,
    state: str = "OPEN",
    labels: str = "",
    remove_fail: str = "0",
    pr_missing: str = "0",
    prev_runs: str = "[]",
    poll_runs: str = "[]",
    active_runs: str = "[]",
    active_fail: str = "0",
    active_failures: int = 999,
    active_response: str = "",
    know_run_exists: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the REAL `make merge-gate` target with a stubbed `gh` on PATH.

    `make` runs it, not the extracted recipe text: `$$` is make's escape for a
    literal `$`, so a shell handed that text directly would read every variable
    as its own process id. `sleep` is stubbed as well — the real bound is 12
    attempts 5 s apart plus a 3 s pause after a removal, which is 63 s per case;
    the bound's ARITHMETIC is pinned in the test above instead.

    Returns the process and every `gh` invocation, in order, as argv joined by
    spaces.
    """
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "gh").write_text(f"#!{sys.executable}\n{_MERGE_GATE_GH_BODY}")
    (shim / "gh").chmod(0o755)
    (shim / "sleep").write_text(f"#!{sys.executable}\n")
    (shim / "sleep").chmod(0o755)
    (tmp_path / "home").mkdir(exist_ok=True)
    log = tmp_path / "gh.log"
    # Persist boundary state: a re-arm must really remove/add the existing label,
    # not merely record an add intention that would emit no labeled event.
    (tmp_path / "labels.json").write_text(json.dumps(labels.split()))
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("CMUX_")},
        "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path / "home"),
        "STUB_LOG": str(log),
        "STUB_RUN_LIST_COUNT": str(tmp_path / "runlist.count"),
        "STUB_SHA": _MERGE_GATE_SHA,
        "STUB_HEAD": "ci/merge-gate-merge-gate",
        "STUB_STATE": state,
        "STUB_LABEL_STATE": str(tmp_path / "labels.json"),
        "STUB_LABEL_TRANSITIONS": str(tmp_path / "label-transitions.jsonl"),
        "STUB_REMOVE_FAIL": remove_fail,
        "STUB_PR_MISSING": pr_missing,
        "STUB_PREV_RUNS": prev_runs,
        "STUB_POLL_RUNS": poll_runs,
        "STUB_ACTIVE_RUNS": active_runs,
        "STUB_ACTIVE_FAIL": active_fail,
        "STUB_ACTIVE_FAILURES": str(active_failures),
        "STUB_ACTIVE_COUNT": str(tmp_path / "active.count"),
        "STUB_ACTIVE_RESPONSE": active_response,
    }
    proc = subprocess.run(
        [
            "make",
            "-f",
            str(MAKEFILE),
            "merge-gate",
            "PR=3631",
            *(["I_KNOW_A_GATE_RUN_EXISTS=1"] if know_run_exists else []),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return proc, (log.read_text().splitlines() if log.exists() else [])


def test_the_merge_gate_ritual_prints_the_run_it_armed(tmp_path: Path) -> None:
    """The success branch must be reachable: it prints the NEW run's URL.

    R6, as a behaviour. The poll used to hand jq's own flags to `gh`
    (`--jq --arg sha …`), which exits 1 with `unknown flag: --arg`; the recipe
    assigned empty on failure, so no attempt of the bounded poll could ever
    match and the ritual reported "armed … but no NEW run" for an arming it had
    just performed. Mutation that must fail this: put `--arg` back on the `gh`
    call, or drop the `!= prev` / `createdAt >= since` predicates.
    """
    previous, armed = 1111, 2222
    proc, calls = _arm_merge_gate(
        tmp_path,
        # Already labelled, so the remove-then-add order is exercised too: a
        # `--add-label` over a present label is a no-op, fires no `labeled`
        # event and strands the head in "Expected" while printing success.
        labels="ready-to-merge",
        prev_runs=_merge_gate_runs((previous, _MERGE_GATE_BEFORE)),
        poll_runs=_merge_gate_runs((armed, _MERGE_GATE_AFTER), (previous, _MERGE_GATE_BEFORE)),
    )
    assert proc.returncode == 0, f"the ritual failed: {proc.stdout}\n{proc.stderr}"
    url = f"https://github.com/damianvtran/local-operator/actions/runs/{armed}"
    assert url in proc.stdout, (
        f"the ritual did not print the URL of the run it armed ({url}); it said: " f"{proc.stdout}"
    )
    assert (
        f"gh run watch {armed} --exit-status" in proc.stdout
    ), f"the follow command must name the armed run, not the previous one: {proc.stdout}"
    assert (
        f"runs/{previous}" not in proc.stdout
    ), f"the ritual printed the PREVIOUS run on this head: {proc.stdout}"
    edits = [call for call in calls if call.startswith("pr edit")]
    assert (
        edits[:1] and "--remove-label" in edits[0]
    ), f"a present label must be removed first, or no event fires: {calls}"
    assert len(edits) > 1 and "--add-label" in edits[1], f"the add must follow the removal: {calls}"


@pytest.mark.parametrize(
    ("scenario", "stub", "expected", "arms"),
    [
        (
            "no-new-run",
            {
                "prev_runs": _merge_gate_runs((1111, _MERGE_GATE_BEFORE)),
                # The only run for this head is the OLD one: created before the
                # arm time and equal to `prev`, so neither predicate matches.
                "poll_runs": _merge_gate_runs((1111, _MERGE_GATE_BEFORE)),
            },
            "no NEW full-suite",
            # This one DOES arm — the add runs — and then cannot confirm the
            # run; the error must then name what to check, not claim success.
            True,
        ),
        (
            "removal-fails",
            {"labels": "ready-to-merge", "remove_fail": "1"},
            "could not remove ready-to-merge",
            False,
        ),
        ("pr-missing", {"pr_missing": "1"}, "not found (gh pr view failed)", False),
        ("pr-closed", {"state": "CLOSED"}, "not OPEN", False),
    ],
)
def test_the_merge_gate_ritual_refuses_without_arming(
    tmp_path: Path, scenario: str, stub: dict[str, Any], expected: str, arms: bool
) -> None:
    """Every refusal is non-zero, actionable, and happens BEFORE the add.

    A ritual that prints success without arming anything is the failure mode
    this target exists to prevent (QA round 1, Q-3), and a failed `--remove-label`
    must abort: with the label still on, the add is a no-op. Mutation that must
    fail this: swallow the removal (`|| true`), drop the bounded poll's non-zero
    exit, or skip the PR state check.
    """
    proc, calls = _arm_merge_gate(tmp_path, **stub)
    assert (
        proc.returncode != 0
    ), f"{scenario}: the ritual reported success after arming nothing: {proc.stdout}"
    assert (
        expected in proc.stderr
    ), f"{scenario}: expected {expected!r} on stderr, got {proc.stderr!r}"
    if not arms:
        assert not [
            call for call in calls if "--add-label" in call
        ], f"{scenario}: a refusal before the add must leave the label alone: {calls}"
        assert (
            "re-armed the merge gate" not in proc.stdout
        ), f"{scenario}: a failed ritual must not print the success line: {proc.stdout}"
    if scenario.startswith("pr-"):
        assert calls == [
            call for call in calls if call.startswith("pr view")
        ], f"{scenario}: a PR that cannot be re-armed must not be mutated: {calls}"


@pytest.mark.parametrize("status", ["queued", "in_progress", "pending", "waiting", "requested"])
def test_merge_gate_refuses_an_existing_live_gate(tmp_path: Path, status: str) -> None:
    """Exercise the real target: a duplicate refusal precedes ALL label writes."""
    active = [
        {
            "databaseId": 4242,
            "url": "https://example.test/runs/4242",
            "headSha": _MERGE_GATE_SHA,
            "status": status,
        }
    ]
    proc, calls = _arm_merge_gate(
        tmp_path, labels="ready-to-merge synthetic", active_runs=json.dumps(active)
    )
    assert proc.returncode == 2
    assert json.loads((tmp_path / "labels.json").read_text()) == ["ready-to-merge", "synthetic"]
    assert "refusing to arm" in proc.stderr and "4242" in proc.stderr
    assert "I_KNOW_A_GATE_RUN_EXISTS=1" in proc.stderr
    assert not any(call.startswith(("pr edit", "label create")) for call in calls)


@pytest.mark.parametrize("existing_head", [_MERGE_GATE_SHA, "f" * 40])
def test_merge_gate_explicit_duplicate_override_and_other_head_control(
    tmp_path: Path, existing_head: str
) -> None:
    """The override is visible; a queued gate on another head is not a match."""
    same_head = existing_head == _MERGE_GATE_SHA
    active = [
        {
            "databaseId": 4242,
            "url": "https://example.test/runs/4242",
            "headSha": existing_head,
            "status": "queued",
        }
    ]
    proc, calls = _arm_merge_gate(
        tmp_path,
        active_runs=json.dumps(active),
        know_run_exists=same_head,
        poll_runs=_merge_gate_runs((5678, _MERGE_GATE_AFTER)),
    )
    assert proc.returncode == 0, proc.stderr
    assert "--add-label" in "\n".join(calls)
    assert ("WARNING: I_KNOW_A_GATE_RUN_EXISTS=1" in proc.stderr) is same_head
    if same_head:
        assert "lookup did not run (skipped)" in proc.stderr
        assert not any("--status" in call for call in calls)


@pytest.mark.parametrize(
    ("failure", "arms"),
    [
        ("HTTP 500: Internal Server Error", True),
        ("HTTP 503: Service Unavailable", True),
        ("request timed out", True),
        ("HTTP 429: rate limit exceeded", True),
        ("HTTP 403: API rate limit exceeded", True),
        ("error connecting to api.github.com", True),
        ("", True),
        ("HTTP 401: Bad credentials", False),
        ("HTTP 401: Bad credentials (after connection timeout)", False),
        ("HTTP 403: Resource not accessible by integration", False),
        ("unexpected unauthorised response", False),
    ],
)
def test_merge_gate_lookup_availability_is_advisory(
    tmp_path: Path, failure: str, arms: bool
) -> None:
    """Outages must not become a fleet-wide gate; configuration errors stay loud."""
    proc, calls = _arm_merge_gate(
        tmp_path,
        labels="ready-to-merge synthetic",
        active_fail=failure,
        poll_runs=_merge_gate_runs((5678, _MERGE_GATE_AFTER)),
    )
    assert proc.returncode == (0 if arms else 2), proc.stderr
    labels = json.loads((tmp_path / "labels.json").read_text())
    if arms:
        assert "WARNING: existing-gate lookup unavailable" in proc.stderr
        assert (failure or "no response") in proc.stderr
        assert "proceeding to arm" in proc.stderr
        assert "I_KNOW_A_GATE_RUN_EXISTS=1" in proc.stderr
        assert sum("--status" in call for call in calls) == 3
        assert "retry 1/2" in proc.stderr and "retry 2/2" in proc.stderr
        assert labels == ["synthetic", "ready-to-merge"]
        assert (tmp_path / "label-transitions.jsonl").read_text().splitlines() == [
            '["synthetic"]',
            '["synthetic", "ready-to-merge"]',
        ]
    else:
        assert sum("--status" in call for call in calls) == 1
        assert "retry" not in proc.stderr
        assert "no label changed" in proc.stderr
        assert labels == ["ready-to-merge", "synthetic"]
        assert not any(call.startswith(("pr edit", "label create")) for call in calls)
        assert not (tmp_path / "label-transitions.jsonl").exists()


def test_merge_gate_retry_recovers_before_matching_active_run(tmp_path: Path) -> None:
    """Retry is not permission to arm: a recovered queued match still wins."""
    proc, calls = _arm_merge_gate(
        tmp_path,
        labels="ready-to-merge synthetic",
        active_fail="HTTP 503 Service Unavailable",
        active_failures=1,
        active_runs=json.dumps(
            [
                {
                    "databaseId": 4242,
                    "url": "https://example.test/runs/4242",
                    "headSha": _MERGE_GATE_SHA,
                    "status": "queued",
                }
            ]
        ),
    )
    assert proc.returncode == 2, proc.stderr
    assert "retry 1/2" in proc.stderr and "retry 2/2" not in proc.stderr
    assert "queued gate run 4242" in proc.stderr
    assert "proceeding to arm" not in proc.stderr
    assert sum("--status" in call for call in calls) == 2
    assert json.loads((tmp_path / "labels.json").read_text()) == ["ready-to-merge", "synthetic"]
    assert not any(call.startswith(("pr edit", "label create")) for call in calls)


@pytest.mark.parametrize("response", ["not json", '{"message":"Unauthorized"}', "null", "[{}]"])
def test_merge_gate_malformed_lookup_refuses(tmp_path: Path, response: str) -> None:
    proc, calls = _arm_merge_gate(
        tmp_path, labels="ready-to-merge synthetic", active_response=response
    )
    assert proc.returncode == 2, proc.stderr
    assert "malformed existing-gate response" in proc.stderr
    assert "lane-owner manual recovery only:" in proc.stderr
    assert "I_KNOW_A_GATE_RUN_EXISTS=1" in proc.stderr
    assert json.loads((tmp_path / "labels.json").read_text()) == ["ready-to-merge", "synthetic"]
    assert not any(call.startswith(("pr edit", "label create")) for call in calls)


def test_merge_gate_completed_success_history_arms(tmp_path: Path) -> None:
    """A historical green is not an active duplicate or a reason to skip arming."""
    proc, calls = _arm_merge_gate(
        tmp_path,
        labels="ready-to-merge synthetic",
        active_runs=json.dumps(
            [
                {
                    "databaseId": 4242,
                    "url": "https://example.test/runs/4242",
                    "headSha": _MERGE_GATE_SHA,
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        ),
        prev_runs=_merge_gate_runs((4242, _MERGE_GATE_BEFORE)),
        poll_runs=_merge_gate_runs((5678, _MERGE_GATE_AFTER)),
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" not in proc.stderr
    assert sum("--status" in call for call in calls) == 5
    assert json.loads((tmp_path / "labels.json").read_text()) == ["synthetic", "ready-to-merge"]


@pytest.mark.parametrize("failure", ["request timed out", "lookup unusable"])
def test_merge_gate_override_skips_unreachable_advisory_lookup(
    tmp_path: Path, failure: str
) -> None:
    proc, calls = _arm_merge_gate(
        tmp_path,
        labels="ready-to-merge synthetic",
        active_fail=failure,
        know_run_exists=True,
        poll_runs=_merge_gate_runs((5678, _MERGE_GATE_AFTER)),
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARNING: I_KNOW_A_GATE_RUN_EXISTS=1" in proc.stderr
    assert "existing-gate lookup did not run (skipped)" in proc.stderr
    assert not any("--status" in call for call in calls)
    assert json.loads((tmp_path / "labels.json").read_text()) == ["synthetic", "ready-to-merge"]


@pytest.mark.parametrize("scenario", ["recovers", "exhausted", "deadline", "auth", "unusable"])
def test_merge_gate_advisory_retry_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], scenario: str
) -> None:
    """A synthetic clock proves the budget even when a request consumes its cap.

    This is a policy budget, not a latency benchmark. The deadline is shared by
    the Makefile across statuses; an already-spent budget must issue no request.
    """
    clock = [0.0]
    calls: list[float] = []
    sleeps: list[float] = []
    deadline = 1 if scenario == "deadline" else 17
    monkeypatch.setattr(merge_gate_advisory.time, "monotonic", lambda: clock[0])

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command == ["gh", "run", "list"]
        calls.append(kwargs["timeout"])
        if scenario == "recovers" and len(calls) == 2:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        if scenario in {"auth", "unusable"}:
            return subprocess.CompletedProcess(
                command, 1, "", "HTTP 401" if scenario == "auth" else "unknown failure"
            )
        clock[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(merge_gate_advisory.time, "sleep", sleep)
    monkeypatch.setattr(merge_gate_advisory.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["advisory", str(deadline), "gh", "run", "list"])
    assert merge_gate_advisory.BUDGET_SECONDS == 3 * 5 + 2 * 1 == 17
    assert "time.monotonic() + 17" in MAKEFILE.read_text()
    assert merge_gate_advisory.main() == (0 if scenario == "recovers" else 1)
    output = capsys.readouterr()
    if scenario == "recovers":
        assert calls == [5, 5] and sleeps == [1]
        assert output.out == "[]"
    elif scenario == "exhausted":
        assert calls == [5, 5, 5] and sleeps == [1, 1]
        assert clock[0] == 17
    elif scenario == "deadline":
        assert calls == [1] and clock[0] == 1
        assert "timed out" in output.out
        assert merge_gate_advisory.main() == 1
        assert calls == [1], "a spent shared budget must not start another status request"
    else:
        assert calls == [5] and sleeps == []
        assert "retry" not in output.err


def test_full_suite_run_name_identifies_the_owning_lane_and_trigger() -> None:
    """Ownership must be visible in gh run list, before any runner starts."""
    full = yaml.safe_load(FULL_SUITE_YML.read_text())
    name = full["run-name"]
    for token in (
        "owner=",
        "github.actor",
        "lane-",
        "github.event.pull_request.number",
        "github.ref_name",
        "arm/re-arm:",
        "github.event.label.name",
        "github.event_name",
        "github.event.pull_request.head.sha",
        "github.sha",
        "github.run_attempt",
    ):
        assert token in name, f"run-name lost its ownership/trigger field: {token}"


def test_full_suite_keeps_both_interpreters_and_bounded_concurrency() -> None:
    """3.13 is where version-class bugs surface; one running + newest pending."""
    full = yaml.safe_load(FULL_SUITE_YML.read_text())
    for job in ("test", "coverage-report"):
        assert full["jobs"][job]["strategy"]["matrix"]["python-version"] == ["3.12", "3.13"]
    # main keeps one fixed group; a branch dispatch (WIDE infra PRs) gets its
    # own, so it can never evict main's newest pending run.
    group = full["concurrency"]["group"]
    assert full["concurrency"]["cancel-in-progress"] is False
    assert "'full-suite-main'" in group and "github.ref" in group, group
    assert "workflow_dispatch" in (full.get("on", full.get(True)) or {})
    assert full["permissions"] == {"contents": "read"}, "test jobs must keep a read-only token"
    report = full["jobs"]["report"]
    assert report["if"] == "always()"
    assert report["permissions"] == {"issues": "write"}
    # `merge-gate` is excluded deliberately: on `main` it mirrors this job's
    # verdict, so the issue filer stays the aggregate that narrates main's
    # health (and the gate must never wait on an issue-writing job).
    assert set(report["needs"]) == set(full["jobs"]) - {"report", "merge-gate"}
    run = report["steps"][0]["run"]
    assert (
        'if [ "$GITHUB_REF" != "refs/heads/main" ]' in run
    ), "branch runs must not touch the issue"
    for needle in ("--label ci-main-red", "gh issue create", "gh issue comment", "gh issue close"):
        assert needle in run, needle


def test_full_suite_changes_job_matches_ci() -> None:
    """`_ci_jobs` takes ci.yml's `changes`; that is only honest if they agree."""
    full = copy.deepcopy(yaml.safe_load(FULL_SUITE_YML.read_text())["jobs"]["changes"])
    # The ONE allowed difference: full-suite's merge-base resolve/apply steps
    # and their `gate_base` output. A labeled-event gate checks out a cached
    # `refs/pull/<n>/merge` that can be dozens of commits behind main (see
    # scripts/gate_merge_base.py); ci.yml runs on the push that recomputes
    # that ref, so it needs neither. Strip exactly those and demand the rest
    # stays byte-for-byte ci.yml's, classifier step included.
    assert full["outputs"].pop("gate_base") == "${{ steps.gate_base.outputs.gate_base }}"
    gate_steps = [s for s in full["steps"] if "scripts/gate_merge_base.py" in str(s.get("run"))]
    assert [s["run"] for s in gate_steps] == [
        "python3 scripts/gate_merge_base.py resolve",
        "python3 scripts/gate_merge_base.py apply",
    ]
    full["steps"] = [s for s in full["steps"] if s not in gate_steps]
    assert full == _pr_jobs()["changes"]


#: Every full-suite job that checks out code. A new one must join this set,
#: and so inherit the fresh-merge step below, or the pin goes red.
GATE_MERGE_JOBS = {
    "changes": "steps.gate_base.outputs.gate_base",
    "test": "needs.changes.outputs.gate_base",
    "coverage-report": "needs.test.outputs.gate_base",
    "tui-e2e": "needs.changes.outputs.gate_base",
}


def test_every_full_suite_checkout_is_rebuilt_on_the_run_main_tip() -> None:
    """A labeled-event gate must test the PR against TODAY's main, once per run.

    WHY: `actions/checkout` on `pull_request: labeled` checks out the cached
    `refs/pull/<n>/merge`, which GitHub recomputes only on a push to the PR.
    On 2026-11-10 three re-armed gates (#3723, #3720, #3732) tested merges
    22-38 commits behind main and one re-failed a test main had fixed. The
    fix resolves the tip ONCE in `changes` and every checking-out job merges
    the PR head into exactly that SHA right after its checkout. This pins:
    every checkout job runs `apply` immediately after the checkout (step 2,
    so the `test` checkout stays the bare first step gen_test_durations.py
    binds to), each reads the SAME run-wide base through job outputs (never
    its own `ls-remote`, which would let shards drift), and the head is the
    event's PR head SHA.
    """
    jobs = yaml.safe_load(FULL_SUITE_YML.read_text())["jobs"]
    checkout_jobs = {
        job_id
        for job_id, job in jobs.items()
        if any("actions/checkout" in str(step.get("uses")) for step in job.get("steps", []))
    }
    assert checkout_jobs == set(GATE_MERGE_JOBS), checkout_jobs
    for job_id, base_expr in GATE_MERGE_JOBS.items():
        steps = jobs[job_id]["steps"]
        assert steps[0]["uses"] == "actions/checkout@v7", job_id
        apply = [s for s in steps if s.get("run") == "python3 scripts/gate_merge_base.py apply"]
        assert len(apply) == 1, f"{job_id}: expected exactly one apply step"
        index = steps.index(apply[0])
        # `changes` resolves first (step 2) then applies (step 3); every other
        # job applies straight after its checkout.
        assert index == (2 if job_id == "changes" else 1), (job_id, index)
        assert apply[0]["env"] == {
            "GATE_BASE": "${{ " + base_expr + " }}",
            "GATE_HEAD": "${{ github.event.pull_request.head.sha }}",
        }, job_id
        assert "if" not in apply[0], f"{job_id}: the apply step must not be skippable"
    assert jobs["changes"]["steps"][1] == {
        "name": "Resolve the main tip this run tests against",
        "id": "gate_base",
        "run": "python3 scripts/gate_merge_base.py resolve",
    }
    assert jobs["test"]["outputs"] == {"gate_base": "${{ needs.changes.outputs.gate_base }}"}
    # The greppable per-job line the requester asked for, and the comment
    # that tells a log reader how to check it.
    from scripts import gate_merge_base

    assert gate_merge_base.LOG_PREFIX == "full-suite tested-base="
    assert "full-suite tested-base=<base> head=<head> tree=<tree>" in FULL_SUITE_YML.read_text()


def test_guards_job_reads_the_one_guard_list() -> None:
    """The guard list is spelled once (GUARD_TESTS), and every entry exists."""
    from scripts import affected_tests

    assert affected_tests.GUARD_TESTS, "an empty guard list runs nothing on PRs"
    for node in affected_tests.GUARD_TESTS:
        rel, _, name = node.partition("::")
        assert (REPO / rel).is_file(), f"GUARD_TESTS names a missing file: {rel}"
        # A renamed test function would make pytest error "not found" in CI;
        # catch it here first, where the fix is obvious.
        assert not name or f"def {name}(" in (REPO / rel).read_text(), node
    job = _pr_jobs()["guards"]
    assert "needs.changes.outputs.unit != 'false'" in _if("guards")
    run = "\n".join(step.get("run", "") for step in job["steps"])
    assert "scripts/affected_tests.py --guards" in run
    assert "tests/unit/" not in run, "the guard list must not be re-spelled in ci.yml"


def test_ci_gate_aggregates_every_pr_job() -> None:
    """`CI Gate` is the single required check: it must see every job."""
    jobs = _pr_jobs()
    gate = jobs["ci-gate"]
    assert gate["name"] == "CI Gate"
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == set(jobs) - {"ci-gate"}
    run = gate["steps"][-1]["run"]
    assert '"failure"' in run and '"cancelled"' in run and "exit 1" in run
    assert '"skipped"' not in run, "a classifier skip must pass the gate"
