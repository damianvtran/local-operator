"""The arm builder's guard rails, and dependency admission's refusal cases.

Two subjects, one file, because they are one apparatus: the builder decides what
an arm installs, and the admission gate decides whether that installed set
conforms to the reviewed override policy. Every refusal case below is a control
that was exercised by hand during the main17 preflight; the point of the tests
is that they stay exercised rather than being re-derived by hand next time.

``RAW_CHECK`` is real output from a correctly built arm (main17, 2026-10-25):
rc 1 with exactly the two overridden conflicts. It is the POSITIVE control --
if admission ever starts demanding a green checker, or stops accepting exactly
this report, these tests break.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from scripts import build_osworld_arm as arm
from scripts import osworld_dependency_admission as admission

ROOT = Path(__file__).resolve().parents[5]

RAW_CHECK = """Using Python 3.12.13 environment at: /arms/main17/venv
Checked 406 packages in 14ms
Found 2 incompatibilities
The package `zhipuai` requires `pyjwt>=2.8.0,<2.9.0`, but `2.13.0` is installed
The package `osworld` requires `requests~=2.31.0`, but `2.34.2` is installed
"""

OVERRIDES = ["requests>=2.32.0", "pyjwt[crypto]>=2.10,<3"]
DECLARED = {"osworld": ["requests~=2.31.0"], "zhipuai": ["pyjwt>=2.8.0,<2.9.0"]}
INSTALLED = {
    "requests": "2.34.2",
    "pyjwt": "2.13.0",
    "miniaudio": "1.71.0",
    "watchdog": "6.0.0",
}
CORE = [
    "requests>=2.32.0",
    "pyjwt[crypto]>=2.10,<3",
    "miniaudio>=1.71",
    "watchdog>=6.0.0",
]


def _admit(
    text: str = RAW_CHECK,
    returncode: int = 1,
    *,
    overrides: list[str] | None = None,
    installed: dict[str, str] | None = None,
    declared: dict[str, list[str]] | None = None,
    core: list[str] | None = None,
) -> admission.Admission:
    return admission.admit(
        text,
        returncode,
        overrides=OVERRIDES if overrides is None else overrides,
        installed=INSTALLED if installed is None else installed,
        declared=DECLARED if declared is None else declared,
        core_requirements=CORE if core is None else core,
    )


def test_the_reviewed_policy_matches_the_committed_adapter_metadata() -> None:
    """The reviewed set must agree with the repository it admits arms for.

    Three files have to stay in step, and a test is cheaper than noticing in a
    build: the adapter's committed overrides, the frozen pins those overrides
    are allowed to move, and the reviewed conflict list this module enforces.
    """

    pyproject = tomllib.loads((ROOT / "benchmarks/osworld_v2_adapter/pyproject.toml").read_text())
    overrides = [
        admission.Requirement(text) for text in pyproject["tool"]["uv"]["override-dependencies"]
    ]
    # The override list covers exactly the dependencies the reviewed bounds name.
    assert {admission.canonicalize_name(item.name) for item in overrides} == {
        bound.dependency for bound in admission.SUPERSEDED_BOUNDS
    }
    # And the committed adapter lock's own pin for each of them satisfies the
    # override: the policy may not forbid the version the lock installs.
    lock = tomllib.loads((ROOT / "benchmarks/osworld_v2_adapter/uv.lock").read_text())
    locked = {row["name"]: row["version"] for row in lock["package"]}
    for item in overrides:
        name = admission.canonicalize_name(item.name)
        assert admission.Version(locked[name]) in item.specifier, (name, locked[name])


def test_a_correct_arm_is_admitted_without_calling_its_checker_green() -> None:
    verdict = _admit()
    assert verdict.admitted, verdict.refusals
    receipt = verdict.as_receipt()
    # The honest report: rc 1 is preserved and admission is a separate verdict.
    assert receipt["raw_pip_check_rc"] == 1
    assert receipt["plain_pip_check_green"] is False
    assert receipt["policy_conformant_dependency_admission"] is True
    assert [row[0] for row in receipt["admitted_superseded_bounds"]] == ["osworld", "zhipuai"]


def test_a_third_conflict_is_refused() -> None:
    """The guard this apparatus exists for: no new conflict rides along.

    Main16 had exactly such a third finding (``local-operator`` required
    ``miniaudio``, not installed). Admitting it would have meant a paid arm
    whose own harness requirement was unmet.
    """

    text = RAW_CHECK.replace("Found 2 incompatibilities", "Found 3 incompatibilities")
    text += "The package `core` requires `somepkg>=1`, but `0.9` is installed\n"
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("unadmitted conflict" in reason for reason in verdict.refusals)


def test_a_missing_dependency_is_refused() -> None:
    text = RAW_CHECK.replace("Found 2 incompatibilities", "Found 3 incompatibilities")
    text += "The package `local-operator` requires `miniaudio>=1.71`, but it's not installed\n"
    verdict = _admit(text)
    assert not verdict.admitted
    assert any(reason.startswith("missing dependency") for reason in verdict.refusals)


def test_a_changed_upstream_bound_is_refused() -> None:
    """A moved bound invalidates the review that approved the override."""

    text = RAW_CHECK.replace("~=2.31.0", "~=2.30.0")
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("unadmitted conflict" in reason for reason in verdict.refusals)
    assert any("superseded bound absent" in reason for reason in verdict.refusals)


def test_installed_metadata_disagreeing_with_the_report_is_refused() -> None:
    verdict = _admit(declared={"osworld": ["requests~=2.31.0"]})
    assert not verdict.admitted
    assert any("does not declare" in reason for reason in verdict.refusals)


def test_a_fabricated_green_report_is_refused() -> None:
    """Green is not a better outcome here; it means the input is not this arm."""

    green = "Checked 406 packages in 14ms\nAll installed packages are compatible\n"
    for verdict in (_admit(green, 0), _admit(RAW_CHECK, 0), _admit(green, 1)):
        assert not verdict.admitted
    assert any("green" in reason for reason in _admit(green, 0).refusals)


def test_unrecognised_checker_output_is_refused() -> None:
    verdict = _admit(RAW_CHECK + "error: cannot resolve environment\n")
    assert not verdict.admitted
    assert any("unrecognised checker output" in reason for reason in verdict.refusals)


def test_a_finding_count_that_disagrees_with_its_lines_is_refused() -> None:
    verdict = _admit(RAW_CHECK.replace("Found 2 incompatibilities", "Found 4 incompatibilities"))
    assert not verdict.admitted
    assert any("counted 4 findings but listed 2" in reason for reason in verdict.refusals)


def test_policy_drift_in_the_override_list_is_refused() -> None:
    drifted = [*OVERRIDES, "urllib3>=2.0"]
    verdict = _admit(overrides=drifted)
    assert not verdict.admitted
    assert any("a policy change needs review" in reason for reason in verdict.refusals)


def test_a_weakened_security_floor_is_refused() -> None:
    """Same two package names, one of them no longer a floor -- still drift.

    A name-only comparison would admit this, and the arm would install an
    insecure requests without any admission error.
    """

    weakened = ["requests>=1", "pyjwt[crypto]>=2.10,<3"]
    verdict = _admit(overrides=weakened)
    assert not verdict.admitted
    assert any("requests>=2.32.0" in reason for reason in verdict.refusals)


def test_dropping_the_crypto_extra_from_the_jwt_override_is_refused() -> None:
    # The harness floor carries the crypto extra; ``pyjwt>=2.10,<3`` alone
    # installs a JWT implementation without the crypto backend it needs.
    refused = admission.check_policy(["requests>=2.32.0", "pyjwt>=2.10,<3"])
    assert refused and "pyjwt[crypto]" in refused[0]


def test_the_exact_committed_override_list_passes_the_policy_check() -> None:
    pyproject = tomllib.loads((ROOT / "benchmarks/osworld_v2_adapter/pyproject.toml").read_text())
    assert admission.check_policy(list(pyproject["tool"]["uv"]["override-dependencies"])) == []


def test_an_installed_version_outside_the_override_floor_is_refused() -> None:
    """A downgraded security library is not a way to satisfy a stale SDK."""

    verdict = _admit(installed={**INSTALLED, "pyjwt": "2.8.0"})
    assert not verdict.admitted
    assert any("outside override" in reason for reason in verdict.refusals)
    assert any("core requirement" in reason for reason in verdict.refusals)


def test_a_core_requirement_that_is_absent_is_refused() -> None:
    installed = {name: value for name, value in INSTALLED.items() if name != "watchdog"}
    verdict = _admit(installed=installed)
    assert not verdict.admitted
    assert any("watchdog>=6.0.0 is not installed" in reason for reason in verdict.refusals)


def test_the_cli_writes_its_receipt_before_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal has to leave the raw output behind, or it cannot be argued with."""

    adapter = tmp_path / "adapter.toml"
    adapter.write_text(
        '[tool.uv]\noverride-dependencies = ["requests>=2.32.0", "pyjwt[crypto]>=2.10,<3"]\n'
    )
    core = tmp_path / "core.toml"
    core.write_text('[project]\ndependencies = ["watchdog>=6.0.0"]\n')
    out = tmp_path / "receipts"
    out.mkdir()

    def fake_run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        text = RAW_CHECK.replace("Found 2 incompatibilities", "Found 3 incompatibilities")
        text += "The package `core` requires `missing>=1`, but it's not installed\n"
        return subprocess.CompletedProcess(argv, 1, text, "")

    monkeypatch.setattr(admission.subprocess, "run", fake_run)
    rc = admission.main(
        ["--adapter-pyproject", str(adapter), "--core-pyproject", str(core), "--out", str(out)]
    )
    assert rc == 1
    assert (out / "raw-pip-check.log").read_text().startswith("Using Python")
    assert (out / "raw-pip-check.rc").read_text().strip() == "1"
    receipt = json.loads((out / "policy-admission.json").read_text())
    assert receipt["policy_conformant_dependency_admission"] is False
    assert receipt["plain_pip_check_green"] is False
    assert any("missing dependency" in reason for reason in receipt["refusals"])


def test_the_frozen_export_keeps_its_rows_and_records_the_dropped_harness_row() -> None:
    rows, dropped = arm.split_frozen_requirements(
        "# autogenerated\n"
        "aiohttp==3.14.3\n"
        "    # via osworld\n"
        "local-operator==0.56.4\n"
        "osworld @ git+https://github.com/xlang-ai/OSWorld-V2@325ab352\n"
    )
    # The harness row is dropped (the arm installs the wheel built from the same
    # commit) and NAMED, so the substitution is in the receipts, not implied.
    assert dropped == ["local-operator==0.56.4"]
    assert rows == [
        "aiohttp==3.14.3",
        "osworld @ git+https://github.com/xlang-ai/OSWorld-V2@325ab352",
    ]


def test_a_frozen_export_carrying_a_pip_option_is_refused() -> None:
    """An export is data; an option in it would change what gets installed."""

    with pytest.raises(arm.BuildRefused):
        arm.split_frozen_requirements("--no-deps\nrequests==2.34.2\n")


def test_the_resolution_project_carries_the_policy_and_the_frozen_constraints() -> None:
    text = arm.resolution_project(
        adapter_rows=["requests==2.34.2", "osworld @ git+https://example.invalid/osworld@abc"],
        core_requirements=["miniaudio>=1.71"],
        overrides=OVERRIDES,
        constraint_versions={"requests": "2.34.2", "pyjwt": "2.13.0"},
    )
    assert 'override-dependencies = ["requests>=2.32.0", "pyjwt[crypto]>=2.10,<3"]' in text
    assert 'constraint-dependencies = ["pyjwt==2.13.0", "requests==2.34.2"]' in text
    assert '"miniaudio>=1.71"' in text
    # package = false: the directory is a resolution input, not a distribution.
    assert "package = false" in text


def test_the_resolution_project_refuses_a_drifted_override_list() -> None:
    with pytest.raises(arm.BuildRefused):
        arm.resolution_project(
            adapter_rows=["requests==2.34.2"],
            core_requirements=[],
            overrides=["requests>=2.32.0"],
            constraint_versions={"requests": "2.34.2", "pyjwt": "2.13.0"},
        )


def test_install_overrides_are_exact_pins_preserving_the_crypto_extra() -> None:
    lock = (
        '[[package]]\nname = "requests"\nversion = "2.34.2"\n'
        '[[package]]\nname = "pyjwt"\nversion = "2.13.0"\n'
    )
    assert arm.installed_overrides(lock) == ["requests==2.34.2", "pyjwt[crypto]==2.13.0"]
    assert arm.lock_versions(lock) == {"requests": "2.34.2", "pyjwt": "2.13.0"}


def test_the_installed_set_must_be_the_planned_set() -> None:
    """Drift at install time is exactly what a reproducibility claim cannot absorb."""

    planned = {"requests": "2.34.2", "pyjwt": "2.13.0"}
    assert arm.compare_closure(pinned=planned, installed=dict(planned), allowed_extra=[]) == []
    assert (
        arm.compare_closure(
            pinned=planned,
            installed={**planned, "local-operator": "0.90.4", "lop-osworld-v2-adapter": "0.1.3"},
            allowed_extra=["local-operator", "lop-osworld-v2-adapter"],
        )
        == []
    )
    extra = arm.compare_closure(
        pinned=planned, installed={**planned, "surprise": "1.0"}, allowed_extra=[]
    )
    assert extra == ["installed surprise 1.0 is not in the resolution plan"]
    moved = arm.compare_closure(
        pinned=planned, installed={**planned, "requests": "2.35.0"}, allowed_extra=[]
    )
    assert moved == ["installed requests 2.35.0 differs from planned 2.34.2"]


def test_a_marker_excluded_row_may_be_absent_but_is_never_invented() -> None:
    # Absence is tolerated (a pinned row whose marker excludes this platform is
    # legitimately not installed); only an installed-but-unplanned package fails.
    assert arm.compare_closure(pinned={"pywin32": "306"}, installed={}, allowed_extra=[]) == []


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_a_short_or_unmerged_sha_is_refused(git_repo: Path) -> None:
    """Only one exact, MERGED commit may be built.

    A floating ref would let a later build pick up commits nobody reviewed, and
    an arm built from unmerged source is one nobody can reproduce from ``main``.
    """

    with pytest.raises(arm.BuildRefused, match="full 40-character"):
        arm.guard_source(git_repo, "abc123")
    merged = _git(git_repo, "rev-parse", "HEAD")
    arm.guard_source(git_repo, merged)  # the merged commit is accepted
    _git(git_repo, "checkout", "-q", "-b", "unmerged")
    (git_repo / "later.txt").write_text("unmerged work\n")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-q", "-m", "unmerged")
    unmerged = _git(git_repo, "rev-parse", "HEAD")
    with pytest.raises(arm.BuildRefused, match="not merged into origin/main"):
        arm.guard_source(git_repo, unmerged)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway repository whose ``origin/main`` is its first commit.

    Built rather than borrowed from the real checkout: ``guard_source`` asserts
    ancestry against ``origin/main``, and a test that depends on the real
    repository's branch state fails for reasons unrelated to this code. The
    identity is set as LOCAL config so every later git call in the test needs no
    environment, and the system/global git config is bypassed so an operator's
    credential helper is never consulted.
    """

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("fixture\n")
    git("add", ".")
    git("commit", "-q", "-m", "initial")
    git("update-ref", "refs/remotes/origin/main", git("rev-parse", "HEAD"))
    return repo
