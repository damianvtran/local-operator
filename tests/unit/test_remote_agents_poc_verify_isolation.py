"""`verify` must answer about the recorded run, not about the host that runs it.

WHY THIS FILE EXISTS. The real-key run's agent committed
`__pycache__/calc.cpython-312.pyc` beside its fix, so the BRANCH that run produced tracks a
bytecode file the fixture runner regenerates the moment it imports `calc` (its merge base tracks
none — what collides is the checkout of the branch, against the untracked path the fixture-SHA
run has just written). `verify` handed that runner its own inherited environment, which made
acceptance 2 depend on an ambient `PYTHONDONTWRITEBYTECODE`: unset, the fixture-SHA run drops the
untracked `__pycache__/`, the branch checkout refuses to overwrite it, and
`acceptance2.checkout_branch` FAILs on a run that is itself correct — observed on the recorded
`ct_92161251` artifact as 20 PASS / 1 FAIL / 0 BLOCKED over the 21 rows printed, one row short of
the passing table because the failing row returns before `acceptance2.test_passes_on_branch`.
The verdict is the product here, so that is a correctness defect in `verify` rather than a
caveat about a host, and the driver now builds the child's environment itself
(`_fixture_runner_environment`).

The cases are the pair the repository asks for: one asserts the environment the child
RECEIVES (including the credential families that must not cross into it), and two run
the real thing end to end on a synthetic fixture whose agent branch tracks a pyc — the
green arm, and the red arm that proves the guard can still fail.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "scripts/remote_agents_poc.py"

#: The fixture's own runner in miniature: it imports `calc`, so importing it is what
#: regenerates the tracked bytecode beside the source. The recorded fixture's
#: `run_tests.py` is the same shape (`_fixture_test_command` prefers it when present).
RUNNER = """\
import sys

from calc import add, sub

TESTS = (("test_add", lambda: add(2, 3), 5), ("test_sub", lambda: sub(5, 3), 2))


def main() -> int:
    failures = 0
    for name, call, expected in TESTS:
        actual = call()
        if actual == expected:
            print(f"PASS {name}: {actual} == {expected}")
        else:
            failures += 1
            print(f"FAIL {name}: {actual} != {expected}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
"""

BROKEN_CALC = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
FIXED_CALC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    """`remote_agents_poc.py` loaded by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("poc_driver_verify", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# What the child is handed
# ---------------------------------------------------------------------------


def test_the_fixture_runner_environment_carries_the_switch_and_little_else(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built, not inherited: the switch to a KNOWN value, and no credentials.

    The allowlist is the assertion, not an observation about this host: PATH and the
    locale family are what the runner needs to run and to be read, and everything else
    — the assumed controller session's AWS material, the `lop` store's key, and any
    bytecode redirect that would only hide this defect elsewhere — must not cross.
    """
    for name in ("LC_ALL", "LANGUAGE", "LC_CTYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-credential")
    monkeypatch.setenv("AWS_PROFILE", "minerva_sandbox")
    monkeypatch.setenv("LOP_POC_MODEL_KEY", "not-a-real-credential")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/somewhere/else")
    # The host that reported the defect had the switch on; the driver's own value is
    # what the child must be given either way, so start from the hostile state.
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)

    environment = driver._fixture_runner_environment()

    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["LANG"] == "en_US.UTF-8"
    assert sorted(environment) == ["LANG", "PATH", "PYTHONDONTWRITEBYTECODE"]


def test_both_fixture_runs_are_handed_that_environment(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BOTH runs, because the FIRST one lays the bytecode the checkout collides with.

    A rig that only fixed the second run would still fail the artifact: the fixture-SHA
    run is where the untracked `__pycache__/` appears. The recorded child environment is
    what is asserted here, so this case does not depend on an interpreter writing
    anything.
    """
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-credential")
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "run_tests.py").write_text(RUNNER, encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_run(
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        stdin: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == sys.executable:
            calls.append({"command": command, "env": env})
            return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0, "out", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(driver, "_run", fake_run)
    verifier = driver.Verifier()
    driver._verify_fixture_tests(verifier, clone, "fixture-sha", "refs/heads/lop/ct_test")

    assert len(calls) == 2, calls
    for call in calls:
        environment = call["env"]
        assert isinstance(environment, dict)
        assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert [name for name, ok, _ in verifier.rows if not ok] == []


# ---------------------------------------------------------------------------
# The recorded shape, end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fixture:
    """A synthetic stand-in for the recorded fixture, bytecode and all."""

    origin: Path
    bundle: Path
    fixture_sha: str
    branch: str
    pyc_relpath: str
    pyc_bytes: bytes


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, f"git {' '.join(args)}: {completed.stdout}{completed.stderr}"
    return completed.stdout


def _build_fixture(root: Path) -> _Fixture:
    """The fixture's real shape: the fix's commit carries a tracked `.pyc`.

    The bytecode's NAME is derived from the interpreter running these tests, because
    that is the path a runner would regenerate and therefore the one the checkout
    collides with; its BYTES are a placeholder, so a regenerated file could never be
    mistaken for the committed one.
    """
    origin = root / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "poc@example.invalid")
    _git(origin, "config", "user.name", "POC")
    (origin / "calc.py").write_text(BROKEN_CALC, encoding="utf-8")
    (origin / "run_tests.py").write_text(RUNNER, encoding="utf-8")
    (origin / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "fixture")
    fixture_sha = _git(origin, "rev-parse", "HEAD").strip()

    branch = "lop/ct_test"
    _git(origin, "checkout", "-q", "-b", branch)
    (origin / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
    # `cache_from_source` is the interpreter's own spelling of where `calc`'s bytecode goes,
    # and it reads the PROCESS's `sys.pycache_prefix` — so a host that exports
    # `PYTHONPYCACHEPREFIX` makes it an ABSOLUTE path outside the fixture, which would plant
    # the placeholder in the operator's shared cache, leave the branch tracking nothing, and
    # quietly cost both arms their teeth (review MAJOR). `_unset_bytecode_variables` clears
    # the attribute; the assertion makes a regression of that loud HERE, where the damage
    # would be done, rather than as a silently toothless arm later.
    pyc_relpath = importlib.util.cache_from_source("calc.py")
    redirect_active = Path(pyc_relpath).is_absolute()
    assert not redirect_active, "bytecode redirect active: .pyc would land outside the fixture"
    pyc_bytes = b"\x00" * 442
    pyc = origin / pyc_relpath
    pyc.parent.mkdir(parents=True, exist_ok=True)
    pyc.write_bytes(pyc_bytes)
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "fix add and carry the container's bytecode")
    _git(origin, "checkout", "-q", "main")

    bundle = root / "repo.bundle"
    _git(origin, "bundle", "create", str(bundle), "--branches")
    # The full ref is what `_verify_bundle` reads back out of `git bundle list-heads`.
    return _Fixture(origin, bundle, fixture_sha, f"refs/heads/{branch}", pyc_relpath, pyc_bytes)


def _clone(fixture: _Fixture, root: Path) -> Path:
    clone = root / "clone"
    _git(root, "clone", "-q", str(fixture.origin), str(clone))
    return clone


def _unset_bytecode_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ambient state that exposed the defect, cleared on BOTH sides of the switch.

    `PYTHONDONTWRITEBYTECODE` goes so that a child inheriting the environment writes
    bytecode, and `PYTHONPYCACHEPREFIX` goes so that it writes it BESIDE THE SOURCE, where
    the collision is. Clearing only `os.environ` is not the same thing: the interpreter set
    `sys.pycache_prefix` from that variable at start-up, `importlib.util.cache_from_source`
    reads the ATTRIBUTE rather than the variable, and a host that exports the redirect
    leaves this file answering about the host — the fixture's `.pyc` planted outside the
    fixture, the branch tracking nothing, the red arm failing outright and the green one
    asserting about a path outside the clone. `tests/unit/test_bytecode_cache.py` clears the
    same attribute for the same reason.
    """
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.setattr(sys, "pycache_prefix", None)


def test_verify_fails_the_normal_acceptance_rows_but_never_dirties_the_clone(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The green arm, with the ambient switch UNSET.

    Acceptance 2's own verdicts are the point (`test_fails_at_fixture_sha` must still be
    a FAIL exit, `test_passes_on_branch` a PASS one), and the last assertion is the
    defect's whole shape: `git status` in the clone is empty, so the run left neither the
    untracked `__pycache__/` that blocked the checkout nor a rewritten tracked `.pyc`.
    """
    _unset_bytecode_variables(monkeypatch)
    fixture = _build_fixture(tmp_path)
    clone = _clone(fixture, tmp_path)
    verifier = driver.Verifier()

    branch = driver._verify_bundle(verifier, clone, fixture.bundle, fixture.fixture_sha)
    assert branch == fixture.branch
    driver._verify_fixture_tests(verifier, clone, fixture.fixture_sha, branch)

    names = [name for name, _, _ in verifier.rows]
    assert [name for name, ok, _ in verifier.rows if not ok] == [], verifier.rows
    assert "acceptance2.checkout_branch" in names
    assert "acceptance2.test_fails_at_fixture_sha" in names
    assert "acceptance2.test_passes_on_branch" in names

    # The tracked bytecode is the commit's, byte for byte...
    assert (clone / fixture.pyc_relpath).read_bytes() == fixture.pyc_bytes
    # ...and the clone is exactly as the checkout left it.
    assert _git(clone, "status", "--porcelain").strip() == ""


def test_with_bytecode_writing_on_the_branch_checkout_is_the_row_that_fails(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The red arm: the guard is a guard only if its defect still reproduces.

    The same two runs with the environment INHERITED — what the driver did before the
    fix — must leave the untracked file and make the branch checkout fail, so a future
    revert of `_fixture_runner_environment` lands on a red test rather than on a green
    one. An interpreter that writes no bytecode whatever the switch says cannot produce
    the case; that is named as a skip rather than hidden.
    """
    _unset_bytecode_variables(monkeypatch)
    fixture = _build_fixture(tmp_path)
    clone = _clone(fixture, tmp_path)
    branch = driver._verify_bundle(driver.Verifier(), clone, fixture.bundle, fixture.fixture_sha)
    assert branch == fixture.branch

    command = driver._fixture_test_command(clone)
    at_fixture = driver._run(["git", "checkout", "--quiet", fixture.fixture_sha], cwd=clone)
    assert at_fixture.returncode == 0
    assert driver._run(command, cwd=clone).returncode != 0
    if not (clone / fixture.pyc_relpath).exists():
        pytest.skip("this interpreter writes no bytecode even with the switch unset")

    checkout = driver._run(["git", "checkout", "--quiet", branch], cwd=clone)
    assert checkout.returncode != 0, "bytecode writing on must still break the branch checkout"
    # The property, not git's wording: the untracked file is still standing and the branch
    # (whose commit tracks it) is still not checked out. Git localises its refusal text.
    assert (clone / fixture.pyc_relpath).exists()
    assert _git(clone, "rev-parse", "HEAD").strip() == fixture.fixture_sha
