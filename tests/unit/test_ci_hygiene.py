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
  the five-shard matrix, a Windows run, a macOS TUI run, a live-LLM run and a
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

import itertools
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

from scripts import ci_scope, shard_tests

REPO = Path(__file__).resolve().parents[2]
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
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
    assert needs == {"changes", "lint", "type-check"}, (
        f"tui-e2e.needs={sorted(needs)!r}; expected only the classifier and "
        "the two cheap syntax gates (the job does not install through them)"
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
    expected = {"changes", "lint", "type-check", "test", "pip-audit"}
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


def _run_step(job: str) -> str:
    """The `run:` body of `job`'s step that actually runs the tests.

    Found by the `pytest` invocation rather than by step name, for the same
    reason `_shard_step_run` is: the assertions built on it are about what the
    step EXECUTES (does it run the partitioned list, does it keep `-n0`), and a
    renamed step must not be able to detach them.
    """
    steps = [s for s in _steps(job) if "pytest" in (s.get("run") or "")]
    assert len(steps) == 1, f"{job} must have exactly one pytest step, got {len(steps)}"
    return steps[0]["run"]


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
    reason to require it: the same `--total 3` runs on both OS legs, so a
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
SHARD_JOB_OVERHEAD_SECONDS = 60

#: Workers the unit shard job actually gets, and the reason the two trees
#: cannot share a bound. `conftest.py`'s hook takes EVERY core when `CI` is set
#: (the 0.5 share is deliberately skipped on a dedicated runner — applying it
#: measurably halved CI parallelism), and a GitHub `ubuntu-latest` runner has 4
#: vCPUs, so a unit shard's serial weight divides by 4. The e2e tree divides by
#: 1 by design: it runs `-n0` because a fired watchdog exits the process, which
#: is exactly why its sharding axis had to be runners rather than workers.
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
    """
    run = _run_step(job)
    tree = SHARD_JOBS[job]
    root = shard_tests.TREES[tree].root

    assert (
        "_tests.txt" in run
    ), f"{job}'s run step does not consume the partition's file list: {run!r}"
    # `pytest <root>` collects the whole tree and ignores the list entirely.
    assert (
        f"pytest {root}" not in run
    ), f"{job} runs the whole {root} tree; the shard matrix is doing nothing"
    if tree == "e2e":
        assert "-m e2e" in run, "the e2e shard run lost `-m e2e`"
        assert "-n0" in run, (
            "the e2e shard run lost `-n0`: under xdist a fired watchdog kills a "
            "worker carrying unrelated tests (see AGENTS.md)"
        )
        assert (
            "NO_COLOR" in run and "xterm-256color" in run
        ), "the e2e shard run lost the environment the TUI suite composes a frame with"


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
# ci.yml's `changes` job that reads it. Assertions A1-A15 of the design review.
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
    ran = {job for job, verdict in scope.job_plan(docs_only).items() if verdict == "run"}
    assert ran == {"changes", "version-bump-guard"}, (
        "a docs-only PR must cost two checks — the classifier and the version "
        "guard — not a matrix; `coverage-report` inherits the skip from `test`. "
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
    for flag in ("budget", "windows", "audit", "cli", "server"):
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

    It combines the five shard artifacts. If `test` was skipped there are zero
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
    """R3. The Windows job loads more than the two test files it names.

    Its pytest run imports the `conftest.py` chain, and the root conftest does
    `from tests import shard_stall_watchdog` at module scope; pytest also imports
    the `__init__.py` markers for both protected modules. A change to any of
    them can change that job's behaviour without touching a file it lists, which
    is the "flag narrower than the guard's real input" pattern this classifier
    exists to avoid.

    Mutation that must fail this: drop `tests/shard_stall_watchdog.py` from
    `WINDOWS_LOADED_PATHS`.
    """
    scope = _scope()
    loaded = (
        "conftest.py",
        "tests/conftest.py",
        "tests/shard_stall_watchdog.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/unit/server/__init__.py",
        "tests/unit/test_agent_import_boundary.py",
        "tests/unit/server/test_edit_workspace_boundary.py",
    )
    conftest = (REPO / "conftest.py").read_text()
    assert "from tests import shard_stall_watchdog" in conftest, (
        "the root conftest no longer imports the watchdog; WINDOWS_LOADED_PATHS "
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


_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


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
