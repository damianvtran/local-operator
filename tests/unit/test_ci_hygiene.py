"""Guards for CI topology and local-gate invocation.

These exist because three defects all produced a *silent* green:

- #428: ``tui-e2e`` needed ``test``, so a flaky unit suite skipped the freeze
  guard (observed on PR #426, which *fixed* a deadlock while the deadlock
  job reported ``skipping``).
- #423: a stale ``.venv/bin/black`` shebang exits 126, and ``cmd | tail``
  reports ``tail``'s 0, so a lint gate that never ran looks passing.
- #381: ``[tool.pyright] exclude`` *replaces* pyright's built-in defaults.
  Dropping ``**/.*`` makes a local run type-check all of site-packages.

A comment in ci.yml is not a test. Each assertion below is mutation-tested
against the defect it claims to catch.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO / "pyproject.toml"
MAKEFILE = REPO / "Makefile"


def _ci_jobs() -> dict[str, Any]:
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _needs(job: str) -> set[str]:
    """`needs` of a CI job, as a set. A missing key is an empty set, which
    is a legitimate topology (no dependencies), not an error."""
    declared = _ci_jobs()[job].get("needs") or []
    assert isinstance(declared, list)
    return set(declared)


def _steps(job: str) -> list[dict[str, Any]]:
    steps = _ci_jobs()[job]["steps"]
    assert isinstance(steps, list)
    return steps


def test_tui_e2e_does_not_need_the_unit_suite_or_pip_audit() -> None:
    """The freeze guard must still run when `test` is red.

    `needs: test` is how GitHub Actions *skips* a job, not how it waits
    politely. A skipped freeze guard is indistinguishable from a passing
    one on the PR checks list. `pip-audit` has the same shape: a CVE
    published today would skip the macOS resume-liveness assertion on an
    unmodified tree. lint/type-check stay — they are cheap syntax gates
    and a broken install is not a freeze.
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
    assert needs == {"lint", "type-check"}, (
        f"tui-e2e.needs={sorted(needs)!r}; expected only lint and "
        "type-check (cheap syntax gates the job does not install itself)"
    )


def test_cli_and_server_sanity_still_wait_on_the_cheap_gates() -> None:
    """Live-LLM jobs are cost, not freeze guards; they keep the full needs.

    Dropping `test` from those too would be a different change. Pin the
    current contract so a drive-by edit of every `needs:` block at once
    cannot silently re-couple tui-e2e by copying the sanity list.
    """
    expected = {"lint", "type-check", "test", "pip-audit"}
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
    assert pins == ["pyright==1.1.411"], (
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
    """`cmd | tail` reports tail's 0 even when cmd exited 126.

    That is the #423 mechanism: a stale shebang makes `.venv/bin/black`
    fail with `bad interpreter` / rc=126, and a gate script that pipes
    the output looks green. The documented gates (`python -m`, `uvx`)
    and `make lint`/`format`/`type-check` must not be that pipeline.

    This test does not invoke the real `.venv/bin/black` — that would
    pass on a healthy venv and fail only on a stale one, which is the
    opposite of a guard. It builds a 126-exiting fake and shows that
    (a) a pipeline swallows it and (b) `python -m` does not consult
    that shebang at all.

    The fake is invoked by absolute path, not via PATH lookup. Linux
    bash/dash continue searching PATH after a bad shebang, so a fake
    named `flake8` on PATH is skipped and the real flake8 in the venv
    runs (rc=0) — which is how this assertion passed locally on macOS
    (no PATH continuation) and then failed on CI 3.12 by detecting it
    was not exercising the defect. An absolute path cannot be skipped.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Unique name, not `flake8`: even with an absolute-path
        # invocation we must not share a name with a real console
        # script, because a future edit that switches back to PATH
        # lookup would silently start testing the real flake8 again.
        fake = Path(tmp) / "lop-stale-script"
        # A shebang pointing at a path that does not exist is how the
        # real console scripts fail after a worktree is deleted. The
        # kernel returns 126 ("bad interpreter") before the script body
        # runs, so the body here is unreachable on purpose.
        fake.write_text("#!/this/interpreter/does/not/exist\n")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

        # Direct execve of a missing-interpreter shebang raises
        # FileNotFoundError; a gate script sees the *shell*
        # translation, which is 126. Drive bash explicitly so we are
        # not at the mercy of `/bin/sh` being dash vs bash.
        bash = ["/bin/bash", "-c"]
        direct = subprocess.run(
            [*bash, f"{fake} --version"],
            capture_output=True,
            text=True,
        )
        assert direct.returncode == 126, (
            "the fake did not fail with 126 at its own boundary "
            f"(rc={direct.returncode}, stderr={direct.stderr!r}); "
            "the rest of this test is not exercising the defect"
        )

        piped = subprocess.run(
            [*bash, f"{fake} --version | tail -1"],
            capture_output=True,
            text=True,
        )
        assert piped.returncode == 0, (
            "the #423 mechanism itself changed: a 126 inside a pipeline "
            f"no longer reports 0 (rc={piped.returncode}). If shells "
            "started honouring pipefail by default this test would need "
            "rewriting, but the Makefile still must not pipe."
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
    recipes = [
        line.lstrip("\t")
        for line in MAKEFILE.read_text().splitlines()
        if line.startswith("\t") and not line.lstrip().startswith("#")
    ]
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


def test_tui_e2e_still_runs_on_macos() -> None:
    """The freeze is a macOS/BSD property; dropping the leg disarms the guard
    more thoroughly than `needs: test` ever did."""
    matrix = _ci_jobs()["tui-e2e"]["strategy"]["matrix"]["os"]
    assert isinstance(matrix, list)
    assert "macos-latest" in matrix, (
        "tui-e2e lost its macOS leg; the freeze this stage exists to "
        f"catch cannot go red on Linux (os={matrix!r})"
    )
