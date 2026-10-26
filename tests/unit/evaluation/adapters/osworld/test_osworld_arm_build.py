"""The arm builder's guard rails, and dependency admission's refusal cases.

One apparatus, hence one file: the builder decides what an arm installs and the
admission gate decides whether that installed set conforms to the reviewed
override policy. Every refusal case here is a control that must stay exercised,
because each of them corresponds to a real way this could go wrong:

* ``RAW_CHECK`` is a real ``uv pip check`` report from a correctly built arm
  (main17, 2026-10-25): rc 1, exactly the two overridden conflicts. If admission
  ever demands a green checker, or stops accepting this exact report, the
  positive control breaks.
* The extras / marker / duplicate rows are the three real lines ``uv 0.10.8``
  prints for a package that declares ``requests~=2.31.0``,
  ``requests[socks]~=2.31.0`` and ``requests~=2.31.0 ; python_full_version >=
  '3.8'``. They look like the admitted bound and are not, so they are the
  regression guard for keying a finding on its whole requirement.
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
from scripts import osworld_arm_verify as verify
from scripts import osworld_dependency_admission as admission

ROOT = Path(__file__).resolve().parents[5]

# Captured before any test patches ``subprocess.run`` on the shared module: the
# launcher checks below must run the REAL command, or a stub answers them.
_REAL_RUN = subprocess.run

RAW_CHECK = """Using Python 3.12.13 environment at: {prefix}
Checked 406 packages in 14ms
Found 2 incompatibilities
The package `zhipuai` requires `pyjwt>=2.8.0,<2.9.0`, but `2.13.0` is installed
The package `osworld` requires `requests~=2.31.0`, but `2.34.2` is installed
"""

OVERRIDES = ["requests>=2.32.0", "pyjwt[crypto]>=2.10,<3"]
# The installed Requires-Dist rows of the two parents, in their real spelling.
DECLARED = {"osworld": ["requests~=2.31.0"], "zhipuai": ["pyjwt (>=2.8.0,<2.9.0)"]}
INSTALLED = {"requests": "2.34.2", "pyjwt": "2.13.0", "miniaudio": "1.71.0", "watchdog": "6.0.0"}
CORE = ["requests>=2.32.0", "pyjwt[crypto]>=2.10,<3", "miniaudio>=1.71", "watchdog>=6.0.0"]


def _admit(
    text: str,
    returncode: int = 1,
    *,
    overrides: list[str] | None = None,
    installed: dict[str, str] | None = None,
    declared: dict[str, list[str]] | None = None,
    core: list[str] | None = None,
    prefix: str | None = None,
) -> admission.Admission:
    return admission.admit(
        text,
        returncode,
        overrides=OVERRIDES if overrides is None else overrides,
        installed=INSTALLED if installed is None else installed,
        declared=DECLARED if declared is None else declared,
        core_requirements=CORE if core is None else core,
        prefix=prefix,
    )


def _check(prefix: str = "/arms/main17/venv") -> str:
    return RAW_CHECK.format(prefix=prefix)


# --- admission: the positive control ---------------------------------------


def test_a_correct_arm_is_admitted_without_calling_its_checker_green() -> None:
    verdict = _admit(_check())
    assert verdict.admitted, verdict.refusals
    receipt = verdict.as_receipt()
    # The honest report: rc 1 is preserved and admission is a separate verdict.
    assert receipt["raw_pip_check_rc"] == 1
    assert receipt["plain_pip_check_green"] is False
    assert receipt["policy_conformant_dependency_admission"] is True
    assert [row[0] for row in receipt["admitted_superseded_bounds"]] == ["osworld", "zhipuai"]


def test_the_reviewed_policy_matches_the_committed_adapter_metadata() -> None:
    """The reviewed set must agree with the repository it admits arms for.

    A policy edit that leaves this module behind would let an arm be admitted
    against a rule nobody reviewed, so the two are compared rather than trusted.
    """

    pyproject = tomllib.loads((ROOT / "benchmarks/osworld_v2_adapter/pyproject.toml").read_text())
    overrides = [
        admission.Requirement(text) for text in pyproject["tool"]["uv"]["override-dependencies"]
    ]
    assert admission.check_policy(list(pyproject["tool"]["uv"]["override-dependencies"])) == []
    # The override list covers exactly the dependencies the reviewed bounds name.
    assert {admission.canonicalize_name(item.name) for item in overrides} == {
        bound.dependency for bound in admission.SUPERSEDED_BOUNDS
    }
    # And the committed adapter lock's own pin for each satisfies the override:
    # the policy may not forbid the version the lock installs.
    lock = tomllib.loads((ROOT / "benchmarks/osworld_v2_adapter/uv.lock").read_text())
    locked = {row["name"]: row["version"] for row in lock["package"]}
    for item in overrides:
        name = admission.canonicalize_name(item.name)
        assert admission.Version(locked[name]) in item.specifier, (name, locked[name])


# --- admission: third conflicts that merely LOOK admitted -------------------


def test_a_third_conflict_is_refused() -> None:
    """No new conflict rides along.

    Main16 had exactly such a third finding (``local-operator`` required
    ``miniaudio``, not installed). Admitting it would have meant a paid arm
    whose own harness requirement was unmet.
    """

    text = _check().replace(
        "Found 2 incompatibilities",
        "Found 3 incompatibilities\n"
        "The package `gymnasium` requires `numpy<2`, but `2.2.6` is installed",
    )
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("unadmitted conflict: gymnasium requires numpy<2" in r for r in verdict.refusals)


def test_an_extras_qualified_bound_is_not_the_admitted_bound() -> None:
    """``requests[socks]~=2.31.0`` is a different requirement from ``requests~=2.31.0``.

    uv reports it as its own finding, and the override replaces it with plain
    ``requests>=2.32.0`` -- so the extra's package may never be installed, which
    uv's checker does not report when the version constraint is satisfied. A key
    that ignored extras would admit it silently.
    """

    report = _report_with(
        extra_line="The package `osworld` requires `requests[socks]~=2.31.0`,"
        " but `2.34.2` is installed"
    )
    declared = {
        "osworld": ["requests~=2.31.0", "requests[socks]~=2.31.0"],
        "zhipuai": list(DECLARED["zhipuai"]),
    }
    verdict = _admit(report, declared=declared)
    assert not verdict.admitted
    assert any("requests[socks]" in reason for reason in verdict.refusals)


def _report_with(*, extra_line: str, found: int = 3) -> str:
    """The positive-control report with one extra finding and a matching count."""

    return _check().replace(
        "Found 2 incompatibilities",
        f"Found {found} incompatibilities\n{extra_line}",
    )


def test_a_marker_qualified_bound_is_not_the_admitted_bound() -> None:
    report = _report_with(
        extra_line=(
            "The package `osworld` requires `requests~=2.31.0 ; python_full_version >= '3.8'`,"
            " but `2.34.2` is installed"
        )
    )
    verdict = _admit(report)
    assert not verdict.admitted
    assert any("unadmitted conflict" in r and "python_full_version" in r for r in verdict.refusals)


def test_a_duplicated_admitted_line_is_refused() -> None:
    """Two identical admitted lines are still one finding too many.

    A count check cannot catch this on its own -- the duplicate is counted --
    so the admitted bound must match EXACTLY ONCE.
    """

    report = _report_with(
        extra_line="The package `zhipuai` requires `pyjwt>=2.8.0,<2.9.0`, but `2.13.0` is installed"
    )
    verdict = _admit(report)
    assert not verdict.admitted
    assert any("appears 2 times" in r for r in verdict.refusals)


def test_an_extras_qualified_metadata_row_is_refused() -> None:
    """The installed metadata is audited separately, with the same strictness."""

    declared = {
        "osworld": ["requests[socks]~=2.31.0"],
        "zhipuai": list(DECLARED["zhipuai"]),
    }
    verdict = _admit(_check(), declared=declared)
    assert not verdict.admitted
    assert any("does not declare requests~=2.31.0 exactly" in r for r in verdict.refusals)


# --- admission: the other refusals ----------------------------------------


def test_a_missing_dependency_is_refused() -> None:
    text = _check().replace(
        "Found 2 incompatibilities",
        "Found 3 incompatibilities\n"
        "The package `local-operator` requires `miniaudio>=1.71`, but it's not installed",
    )
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("missing dependency" in r and "miniaudio" in r for r in verdict.refusals)


def test_a_changed_upstream_bound_is_refused() -> None:
    text = _check().replace("`requests~=2.31.0`", "`requests~=2.32.0`")
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("superseded bound absent" in r for r in verdict.refusals)


def test_installed_metadata_disagreeing_with_the_report_is_refused() -> None:
    verdict = _admit(_check(), declared={"osworld": ["requests~=2.30"], "zhipuai": []})
    assert not verdict.admitted
    assert any("metadata does not declare" in r for r in verdict.refusals)


def test_a_fabricated_green_report_is_refused() -> None:
    verdict = _admit("Checked 406 packages in 9ms\nAll installed packages are compatible\n", 0)
    assert not verdict.admitted
    assert any("reports green" in r for r in verdict.refusals)


def test_a_report_about_another_environment_is_refused() -> None:
    verdict = _admit(_check("/arms/some-other/venv"), prefix="/arms/main17/venv")
    assert not verdict.admitted
    assert any("not this arm's" in r for r in verdict.refusals)


def test_a_report_claiming_the_wrong_installed_version_is_refused() -> None:
    text = _check().replace("but `2.13.0` is installed", "but `1.0.0` is installed")
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("raw report says pyjwt 1.0.0" in r for r in verdict.refusals)


def test_unrecognised_checker_output_is_refused() -> None:
    verdict = _admit(_check() + "warning: something new\n")
    assert not verdict.admitted
    assert any("unrecognised checker output" in r for r in verdict.refusals)


def test_a_finding_count_that_disagrees_with_its_lines_is_refused() -> None:
    text = _check().replace("Found 2 incompatibilities", "Found 3 incompatibilities")
    verdict = _admit(text)
    assert not verdict.admitted
    assert any("counted 3 findings but listed 2" in r for r in verdict.refusals)


# --- admission: the override policy itself ---------------------------------


def test_policy_drift_in_the_override_list_is_refused() -> None:
    verdict = _admit(_check(), overrides=[*OVERRIDES, "urllib3>=2.0"])
    assert not verdict.admitted
    assert any("a policy change needs review" in r for r in verdict.refusals)


def test_a_weakened_security_floor_is_refused() -> None:
    verdict = _admit(_check(), overrides=["requests>=1", "pyjwt[crypto]>=2.10,<3"])
    assert not verdict.admitted
    assert any("requests>=2.32.0" in r for r in verdict.refusals)


def test_dropping_the_crypto_extra_from_the_override_is_refused() -> None:
    found = admission.check_policy(["requests>=2.32.0", "pyjwt>=2.10,<3"])
    assert found and "pyjwt[crypto]" in found[0]


def test_an_override_disabled_by_a_marker_is_refused() -> None:
    """A marker turns the override off on the platforms it excludes.

    ``requests>=2.32.0 ; python_version < '3'`` is never applied where the arm
    runs, so the stale upstream bound would win -- with no visible difference in
    the package name, the specifier or the extras.
    """

    found = admission.check_policy(
        ["requests>=2.32.0 ; python_version < '3'", "pyjwt[crypto]>=2.10,<3"]
    )
    assert found and "python_version" in found[0]


def test_an_installed_version_outside_the_override_floor_is_refused() -> None:
    verdict = _admit(_check(), installed={**INSTALLED, "requests": "2.31.0"})
    assert not verdict.admitted
    assert any("outside override" in r for r in verdict.refusals)


def test_a_core_requirement_that_is_absent_is_refused() -> None:
    verdict = _admit(_check(), core=[*CORE, "qa-nonexistent-dist>=1.0"])
    assert not verdict.admitted
    assert any("qa-nonexistent-dist" in r for r in verdict.refusals)


# --- admission: the CLI's environment handling -----------------------------


def test_the_checker_runs_with_colour_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--color never`` is load-bearing, not cosmetic.

    With ``FORCE_COLOR``/``CLICOLOR_FORCE`` in the caller's environment uv wraps
    even its non-tty output in ANSI escapes; every line then fails to parse and a
    correct arm is refused.
    """

    seen: list[list[str]] = []

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "Found 0 incompatibilities\n"

    def fake_run(argv: list[str], **kwargs: Any) -> _Result:
        seen.append(list(argv))
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    text, returncode = admission.run_check(tmp_path)
    assert returncode == 1
    assert seen and seen[0][:4] == ["uv", "pip", "check", "--color"]
    assert "--color" in seen[0] and "never" in seen[0]
    assert (tmp_path / "raw-pip-check.log").read_text() == text


def test_a_missing_uv_refuses_cleanly_and_keeps_a_raw_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing binary must leave evidence, not a traceback."""

    def fake_run(argv: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "uv")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text, returncode = admission.run_check(tmp_path)
    assert returncode == 127
    assert "uv could not be executed" in text
    assert (tmp_path / "raw-pip-check.rc").read_text().strip() == "127"


# --- the builder's pure pieces ---------------------------------------------


def test_the_frozen_export_keeps_its_rows_and_records_the_dropped_harness_row() -> None:
    rows, dropped = arm.split_frozen_requirements(
        "# header\nrequests==2.34.2\nlocal-operator==0.56.4\n\npyjwt==2.13.0\n"
    )
    assert rows == ["requests==2.34.2", "pyjwt==2.13.0"]
    assert dropped == ["local-operator==0.56.4"]


def test_a_frozen_export_carrying_a_pip_option_is_refused() -> None:
    with pytest.raises(arm.BuildRefused, match="pip option"):
        arm.split_frozen_requirements("--index-url https://example.invalid\nrequests==2.34.2\n")


def test_the_resolution_project_carries_the_policy_and_the_frozen_constraints() -> None:
    text = arm.resolution_project(
        adapter_rows=["requests==2.34.2"],
        core_requirements=["miniaudio>=1.71"],
        overrides=OVERRIDES,
        constraint_versions={"requests": "2.34.2"},
    )
    assert 'override-dependencies = ["requests>=2.32.0", "pyjwt[crypto]>=2.10,<3"]' in text
    assert 'constraint-dependencies = ["requests==2.34.2"]' in text
    # The harness's own requirement is a normal dependency: resolution may ADD a
    # genuinely new mandatory one instead of hiding it behind --no-deps.
    assert '"miniaudio>=1.71"' in text


def test_the_resolution_project_refuses_a_drifted_override_list() -> None:
    with pytest.raises(arm.BuildRefused, match="policy change needs review"):
        arm.resolution_project(
            adapter_rows=["requests==2.34.2"],
            core_requirements=[],
            overrides=["requests>=1", "pyjwt[crypto]>=2.10,<3"],
            constraint_versions={"requests": "2.34.2"},
        )


def test_install_overrides_are_exact_pins_derived_from_the_committed_policy() -> None:
    """Names and extras come from the override list, never from this module.

    A hard-coded ``pyjwt`` would silently drop the ``[crypto]`` extra that the
    harness floor needs, and a hard-coded pair would miss a third override.
    """

    lock_text = (
        '[[package]]\nname = "requests"\nversion = "2.34.2"\n'
        '[[package]]\nname = "pyjwt"\nversion = "2.13.0"\n'
    )
    assert arm.installed_overrides(lock_text, OVERRIDES) == [
        "requests==2.34.2",
        "pyjwt[crypto]==2.13.0",
    ]
    with pytest.raises(arm.BuildRefused, match="no numpy row"):
        arm.installed_overrides(lock_text, [*OVERRIDES, "numpy>=2"])


def test_constraints_follow_the_committed_lock_and_name_every_override() -> None:
    assert arm._constraint_versions({"requests": "2.34.2", "pyjwt": "2.13.0"}, OVERRIDES) == {
        "requests": "2.34.2",
        "pyjwt": "2.13.0",
    }
    with pytest.raises(arm.BuildRefused, match="no row"):
        arm._constraint_versions({"requests": "2.34.2"}, OVERRIDES)


def test_a_plan_digest_covers_the_installed_set_not_the_serialisation() -> None:
    rows = ["requests==2.34.2", "pyjwt==2.13.0"]
    assert arm.plan_digest(rows, OVERRIDES) == arm.plan_digest(list(reversed(rows)), OVERRIDES)
    assert arm.plan_digest(rows, OVERRIDES) != arm.plan_digest(
        ["requests==2.35.0", "pyjwt==2.13.0"], OVERRIDES
    )
    assert arm.plan_digest(rows, OVERRIDES) != arm.plan_digest(rows, ["requests>=2.33"])


def test_the_installed_set_must_be_the_planned_set() -> None:
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
    assert arm.compare_closure(
        pinned=planned, installed={**planned, "surprise": "1.0"}, allowed_extra=[]
    ) == ["installed surprise 1.0 is not in the resolution plan"]
    assert arm.compare_closure(
        pinned=planned, installed={**planned, "requests": "2.35.0"}, allowed_extra=[]
    ) == ["installed requests 2.35.0 differs from planned 2.34.2"]


def test_a_marker_excluded_row_may_be_absent_but_is_never_invented() -> None:
    # Absence is tolerated (a pinned row whose marker excludes this platform is
    # legitimately not installed); only an installed-but-unplanned package fails.
    assert arm.compare_closure(pinned={"pywin32": "306"}, installed={}, allowed_extra=[]) == []


# --- the builder's guard rails --------------------------------------------


def test_an_arm_directory_is_created_atomically_and_never_reused(tmp_path: Path) -> None:
    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    assert layout.receipts.is_dir()
    with pytest.raises(arm.BuildRefused, match="immutable"):
        layout.create_arm()


def test_a_short_unmerged_or_branch_only_sha_is_refused(git_repo: Path) -> None:
    """Only one exact commit on ``origin/main``'s FIRST-PARENT line may be built.

    Mere ancestry is not enough: this repository merges with merge commits, so
    every intermediate commit of a merged branch is an ancestor of main --
    including pre-remediation states review rejected.
    """

    with pytest.raises(arm.BuildRefused, match="full 40-character"):
        arm.guard_source(git_repo, "abc123")
    merged = _git(git_repo, "rev-parse", "origin/main")
    arm.guard_source(git_repo, merged)

    _git(git_repo, "checkout", "-q", "-b", "feature")
    (git_repo / "feature.txt").write_text("feature work\n")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-q", "-m", "feature")
    feature = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "checkout", "-q", "main")
    _git(git_repo, "merge", "-q", "--no-ff", "-m", "merge feature", "feature")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", _git(git_repo, "rev-parse", "HEAD"))

    # The merge commit is on the first-parent line; the branch commit is an
    # ancestor but NOT a commit main was ever at.
    arm.guard_source(git_repo, _git(git_repo, "rev-parse", "origin/main"))
    with pytest.raises(arm.BuildRefused, match="first-parent"):
        arm.guard_source(git_repo, feature)

    _git(git_repo, "checkout", "-q", "-b", "unmerged", _git(git_repo, "rev-parse", "origin/main"))
    (git_repo / "later.txt").write_text("unmerged work\n")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-q", "-m", "unmerged")
    with pytest.raises(arm.BuildRefused, match="first-parent"):
        arm.guard_source(git_repo, _git(git_repo, "rev-parse", "HEAD"))


FACTS = {"plan_digest": "a" * 64, "resolution_package_count": 420, "security_pins": {}}


def _stub_preinstall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Everything up to the install, stubbed. Returns the list of steps taken.

    A unit test must never build the 420-package closure or install it, so the
    SEQUENCE is what is under test here: an existence refusal, a wrong plan or a
    re-resolution that drifted are the failures this guards.
    """

    calls: list[str] = []

    def resolve(**kwargs: Any) -> tuple[list[str], Path, dict[str, Any]]:
        receipts: Path = kwargs["receipts"]
        project = receipts / "resolution-project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "uv.lock").write_text(
            '[[package]]\nname = "requests"\nversion = "2.34.2"\n'
            '[[package]]\nname = "pyjwt"\nversion = "2.13.0"\n'
        )
        pins = receipts / "policy-pins.txt"
        pins.write_text("pyjwt[crypto]==2.13.0\nrequests==2.34.2\n")
        calls.append("resolve")
        return ["requests==2.34.2", "pyjwt[crypto]==2.13.0"], pins, dict(FACTS)

    monkeypatch.setattr(arm, "guard_source", lambda repo, sha: None)
    # tmp_path IS volatile, and refusing it is correct behaviour -- it is covered
    # by its own test below. These tests are about the sequence after it.
    monkeypatch.setattr(arm, "refuse_volatile", lambda path, label: None)
    monkeypatch.setattr(
        arm, "materialise_source", lambda repo, sha, dest: dest.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(arm, "metadata_from_source", lambda source: _metadata(tmp_path))
    monkeypatch.setattr(arm, "metadata_from_git", lambda repo, sha, dest: _metadata(tmp_path))
    monkeypatch.setattr(arm, "policy", lambda metadata: (OVERRIDES, CORE))
    monkeypatch.setattr(
        arm, "export_adapter_requirements", lambda metadata, receipts: (["requests==2.34.2"], [])
    )
    monkeypatch.setattr(arm, "create_arm_venv", lambda venv, receipts: {"interpreter": "stub"})
    monkeypatch.setattr(
        arm, "lock_versions", lambda text: {"requests": "2.34.2", "pyjwt": "2.13.0"}
    )
    monkeypatch.setattr(arm, "resolve_policy_project", resolve)
    monkeypatch.setattr(
        arm,
        "build_wheels",
        lambda layout: calls.append("wheels")
        or (_whl(tmp_path, "core"), _whl(tmp_path, "adapter")),
    )
    monkeypatch.setattr(arm, "install_arm", lambda **kwargs: calls.append("install"))
    monkeypatch.setattr(arm, "uv_version", lambda: "uv 0.10.8")
    return calls


def _stub_verifiers(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Stand in for the two child interpreters, writing the report they would."""

    def fake_run(argv: list[str], **kwargs: Any) -> Any:
        joined = " ".join(argv)
        if "osworld_arm_verify.py" in joined:
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            arm_root = out.parent.parent
            out.write_text(
                json.dumps(
                    {
                        # A verification report names the build it verified; the
                        # builder refuses one that describes another source or
                        # another interpreter (round-3 stale-identity concern).
                        "commit": "a" * 40,
                        "prefix": str(arm_root / "venv"),
                        "verified": True,
                        "harness": {"version": "0.90.4"},
                        # The real shape: the builder reads this block, so a fake
                        # report missing a key must break the test, not silently
                        # record null (review round 2, R2-1).
                        "core_module_digest": "f" * 64,
                        "core_source": {
                            "sdk": {"module": "local_operator.sdk", "hash": "1" * 64},
                            "loop": {"module": "local_operator.harness.loop", "hash": "2" * 64},
                        },
                        "adapter": {
                            "version": "0.1.3",
                            "package_digest": "d" * 64,
                            "source_record_digest": "e" * 64,
                        },
                        "installed_distributions": {
                            "requests": "2.34.2",
                            "pyjwt": "2.13.0",
                            "local-operator": "0.90.4",
                            "lop-osworld-v2-adapter": "0.1.3",
                        },
                        "verified": True,
                    }
                )
            )
            calls.append("verify")
        elif "osworld_dependency_admission.py" in joined:
            calls.append("admission")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(arm.subprocess, "run", fake_run)


def test_plan_mode_writes_nothing_under_out_and_is_refused_by_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan mode used to create the arm dir (~220 MB), making execute impossible.

    The documented plan → execute pair must work as written, so a plan creates
    NOTHING at ``--out`` and a plan run that is handed one is refused outright
    rather than silently ignoring it -- the caller would otherwise believe an arm
    had been prepared.
    """

    plan_path = tmp_path / "plan.json"
    out = tmp_path / "arm"
    _stub_preinstall(monkeypatch, tmp_path)

    assert arm.main(["--source-sha", "a" * 40, "--plan", str(plan_path)]) == 0
    assert plan_path.is_file(), "the plan receipt is the whole product of a plan run"
    assert not out.exists(), "plan mode must not create the arm directory"
    receipt = json.loads(plan_path.read_text())
    assert receipt["plan_digest"] == "a" * 64
    assert receipt["source_commit"] == "a" * 40
    # R2-1/Q2-1: the plan directory is temporary, so a path to its pins file
    # would dangle; the content digest is what survives the run.
    assert "requirements_installed" not in receipt
    assert len(receipt["requirements_digest"]) == 64

    assert arm.main(["--source-sha", "a" * 40, "--out", str(out)]) == 2
    assert not out.exists()


def test_execute_binds_to_the_plan_and_is_not_refused_for_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second documented command must get past the immutability guard.

    Both directions of the binding are asserted: a matching plan proceeds, a
    drifted resolution is refused instead of installed.
    """

    out = tmp_path / "arm"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"source_commit": "a" * 40, "plan_digest": FACTS["plan_digest"]})
    )
    calls = _stub_preinstall(monkeypatch, tmp_path)
    _stub_verifiers(monkeypatch, calls)

    rc = arm.main(
        ["--source-sha", "a" * 40, "--out", str(out), "--execute", "--plan", str(plan_path)]
    )
    assert rc == 0, "the documented plan-then-execute sequence must succeed"
    assert calls == ["resolve", "wheels", "install", "admission", "verify"], calls
    canonical = out / "build-record.json"
    receipts_copy = out / "receipts" / "build-record.json"
    assert canonical.is_file() and receipts_copy.is_file()
    assert canonical.read_bytes() == receipts_copy.read_bytes()
    record = json.loads(canonical.read_text())
    assert record["record_contract"] == "lop-osworld-arm-build-record/v1"
    assert record["verified_commit"] == "a" * 40
    assert record["verified_venv_prefix"] == str(out / "venv")
    assert record["plan_binding"]["verified"] is True
    assert record["refusals"] == []
    assert record["package_digest_scope"].startswith("this install only")
    assert record["cross_build_identity"]["source_record_digest"] == "e" * 64
    # R2-1: these were read from keys the producer never wrote, so they were null.
    assert record["cross_build_identity"]["core_module_digest"] == "f" * 64
    assert record["cross_build_identity"]["core_module_hashes"] == {
        "sdk": "1" * 64,
        "loop": "2" * 64,
    }
    assert record["uv_version"] == "uv 0.10.8"

    # A drifted resolution must not install: the plan was the reviewed state.
    before = list(calls)
    drifted = tmp_path / "arm-2"
    monkeypatch.setattr(arm, "resolve_policy_project", lambda **kwargs: _drifted(kwargs, tmp_path))
    assert (
        arm.main(
            ["--source-sha", "a" * 40, "--out", str(drifted), "--execute", "--plan", str(plan_path)]
        )
        == 2
    )
    assert calls == before, "a drifted plan must refuse BEFORE building or installing"
    assert not (drifted / "receipts" / "installed-verification.json").exists()


def _drifted(kwargs: dict[str, Any], tmp_path: Path) -> tuple[list[str], Path, dict[str, Any]]:
    """A resolution whose installed set differs from the planned one."""

    receipts: Path = kwargs["receipts"]
    project = receipts / "resolution-project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "uv.lock").write_text('[[package]]\nname = "requests"\nversion = "2.34.2"\n')
    pins = receipts / "policy-pins.txt"
    pins.write_text("requests==2.34.2\n")
    return ["requests==2.34.2"], pins, {**FACTS, "plan_digest": "b" * 64}


def test_the_build_record_is_written_at_the_caller_contract_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher reads ``<arm>/build-record.json``; the tool used to write only receipts.

    ``scripts/materialize_workspace_21.sh`` takes ``package_digest`` from the arm
    root, so an arm built by this tool needed a hand-made hard link before the
    launcher would accept it. One document, both paths, byte-identical -- and the
    writer says so rather than leaving two copies to drift.
    """

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    written = arm.write_build_record(
        layout, {"record_contract": "x", "arm_source_commit": "a" * 40}, publish=True
    )

    canonical = layout.root / "build-record.json"
    receipts = layout.receipts / "build-record.json"
    assert written["record_contract"] == "lop-osworld-arm-build-record/v1"
    assert written["published"] is True
    assert written["canonical"] == str(canonical)
    assert canonical.read_bytes() == receipts.read_bytes()
    assert json.loads(canonical.read_text())["arm_source_commit"] == "a" * 40


def test_an_existing_build_record_is_never_overwritten(tmp_path: Path) -> None:
    """An adopted or hand-repaired arm must not be relabelled by a later run."""

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    sentinel = '{"arm_source_commit": "old"}\n'
    canonical = layout.root / "build-record.json"
    canonical.write_text(sentinel)
    with pytest.raises(arm.BuildRefused, match="immutable"):
        arm.write_build_record(layout, {"arm_source_commit": "a" * 40}, publish=True)
    assert canonical.read_text() == sentinel, "the existing record must be untouched"
    assert not (layout.receipts / "build-record.json").exists()


def test_a_refused_build_publishes_no_caller_record(tmp_path: Path) -> None:
    """Publication is conditional: a refusal leaves evidence, never a digest file.

    The launcher reads ``json.load(...)['package_digest']`` and then tests the
    result as a shell string. A refused record's ``package_digest`` is JSON null,
    which prints as the NON-EMPTY string ``None`` and passes ``[ -n "$PKG" ]`` --
    so the caller-facing path must not exist at all for a refused build.
    """

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    written = arm.write_build_record(
        layout, {"package_digest": None, "refusals": ["verification refused"]}, publish=False
    )
    assert written["published"] is False
    assert written["canonical"] is None
    assert not (layout.root / "build-record.json").exists()
    evidence = layout.receipts / "build-record.json"
    assert json.loads(evidence.read_text())["refusals"] == ["verification refused"]


def test_a_reverse_order_sentinel_leaves_no_canonical_behind(tmp_path: Path) -> None:
    """receipts exists, canonical does not: nothing may be created at either path.

    The pre-fix writer created the canonical file FIRST and then refused on the
    receipts path, leaving two records that disagreed -- the divergence this API
    claims cannot happen.
    """

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    sentinel = '{"old": true}\n'
    receipts = layout.receipts / "build-record.json"
    receipts.write_text(sentinel)
    canonical = layout.root / "build-record.json"
    with pytest.raises(arm.BuildRefused, match="immutable"):
        arm.write_build_record(layout, {"arm_source_commit": "a" * 40}, publish=True)
    assert not canonical.exists(), "a refusal must not leave a new canonical record"
    assert receipts.read_text() == sentinel


def test_a_failed_second_create_rolls_back_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything this call created is removed if the other path cannot be created."""

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    calls: list[Path] = []
    real = arm._create_exclusive

    def flaky(path: Path, data: bytes) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated disk trouble on the second destination")
        real(path, data)

    monkeypatch.setattr(arm, "_create_exclusive", flaky)
    with pytest.raises(OSError, match="simulated"):
        arm.write_build_record(layout, {"arm_source_commit": "a" * 40}, publish=True)
    assert calls, "the test must have attempted both creates"
    assert not (layout.root / "build-record.json").exists()
    assert not (layout.receipts / "build-record.json").exists()


def test_a_crlf_rewrite_is_not_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """read_text() normalises newlines; the comparison must not (R1-4)."""

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    canonical = layout.root / "build-record.json"
    real_read_bytes = Path.read_bytes

    def crlf_reader(self: Path) -> bytes:
        data = real_read_bytes(self)
        return data.replace(b"\n", b"\r\n") if self == canonical else data

    monkeypatch.setattr(Path, "read_bytes", crlf_reader)
    with pytest.raises(arm.BuildRefused, match="byte-identical"):
        arm.write_build_record(layout, {"arm_source_commit": "a" * 40}, publish=True)
    monkeypatch.undo()
    assert not canonical.exists(), "a diverged record must be rolled back"
    assert not (layout.receipts / "build-record.json").exists()


def test_a_verification_report_for_another_source_is_refused(tmp_path: Path) -> None:
    """The stale-identity case: a record may not cite a build it did not verify.

    Round 3 asked for this explicitly. The report is the ONLY source of the
    recorded identity, so a report that names a different commit or a different
    interpreter must not be able to hand the caller a previous arm's receipt.
    """

    layout = arm.ArmLayout(root=tmp_path / "arm")
    layout.create_arm()
    fresh = {"commit": "a" * 40, "prefix": str(layout.venv)}
    assert arm.verify_recorded_identity(fresh, source_sha="a" * 40, layout=layout) == []

    stale = {**fresh, "commit": "b" * 40}
    assert arm.verify_recorded_identity(stale, source_sha="a" * 40, layout=layout) == [
        f"the verification report covers {'b' * 40}, not this build's {'a' * 40}"
    ]

    other_arm = {**fresh, "prefix": str(tmp_path / "somewhere-else" / "venv")}
    refused = arm.verify_recorded_identity(other_arm, source_sha="a" * 40, layout=layout)
    assert refused and "not " in refused[0]

    assert arm.verify_recorded_identity({}, source_sha="a" * 40, layout=layout) == [
        "no installed-verification report was produced, so nothing is verified"
    ]


def test_a_stale_verification_report_cannot_reach_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the execute path: refusal recorded, identity not inherited."""

    out = tmp_path / "arm"
    calls = _stub_preinstall(monkeypatch, tmp_path)

    def stale_verifiers(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            if "osworld_arm_verify.py" in " ".join(argv):
                target = Path(argv[argv.index("--out") + 1])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            # A PREVIOUS source: exactly what must never be recorded.
                            "commit": "b" * 40,
                            "prefix": str(out / "venv"),
                            "verified": True,
                            "harness": {"version": "0.90.4"},
                            "adapter": {"version": "0.1.3", "package_digest": "d" * 64},
                            "installed_distributions": {
                                "requests": "2.34.2",
                                "local-operator": "0.90.4",
                                "lop-osworld-v2-adapter": "0.1.3",
                            },
                        }
                    )
                )
                calls.append("verify")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        monkeypatch.setattr(arm.subprocess, "run", fake_run)

    stale_verifiers(monkeypatch, calls)
    rc = arm.main(["--source-sha", "a" * 40, "--out", str(out), "--execute"])
    assert rc == 4, "a stale identity must exit as UNVERIFIED, not BUILT"
    assert not (out / "build-record.json").exists(), "a refused build publishes nothing"
    record = json.loads((out / "receipts" / "build-record.json").read_text())
    assert record["published"] is False
    assert record["refusals"], record
    assert any("not this build's" in reason for reason in record["refusals"])
    assert record["verified_commit"] == "b" * 40, "the record states what was verified"
    assert record["cross_build_identity"]["source_commit"] == "a" * 40


def _launcher_code(record: Path) -> str:
    """The launcher's own expression, quoting included.

    ``scripts/materialize_workspace_21.sh:61`` builds exactly this: a Python
    one-liner with the record path quoted inside, run in a ``$(...)`` capture.
    """

    return f"import json;print(json.load(open('{record}'))['package_digest'])"


def _launcher_read(record: Path) -> tuple[int, str]:
    done = _REAL_RUN(
        ["python3", "-c", _launcher_code(record)], capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout.strip()


def _launcher_guard(record: Path) -> int:
    """``PKG=$(...); [ -n "$PKG" ]`` -- the shell test a hand link used to satisfy."""

    script = f'PKG="$(python3 -c "{_launcher_code(record)}" 2>/dev/null)"; [ -n "$PKG" ]'
    return _REAL_RUN(["sh", "-c", script], check=False).returncode


def test_a_refused_build_cannot_be_read_by_the_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-1 on the real caller pattern: no digest file, so the guard fails.

    The prior round published the canonical record even when verification
    refused, and ``package_digest: null`` printed as the non-empty string
    ``None`` -- which satisfied ``[ -n "$PKG" ]``. The refusal now keeps its
    evidence under receipts and publishes nothing the launcher can consume.
    """

    out = tmp_path / "arm"
    calls = _stub_preinstall(monkeypatch, tmp_path)

    def refusing_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            if "osworld_arm_verify.py" in " ".join(argv):
                target = Path(argv[argv.index("--out") + 1])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            "commit": "a" * 40,
                            "prefix": str(out / "venv"),
                            "verified": False,
                            "refusals": ["core module loop hash differs"],
                            "adapter": {"version": "0.1.3", "package_digest": None},
                            "installed_distributions": {},
                        }
                    )
                )
                calls.append("verify")
                return subprocess.CompletedProcess(argv, 1, "", "REFUSED")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        monkeypatch.setattr(arm.subprocess, "run", fake_run)

    refusing_verifier(monkeypatch)
    rc = arm.main(["--source-sha", "a" * 40, "--out", str(out), "--execute"])
    assert rc == 4

    canonical = out / "build-record.json"
    assert not canonical.exists(), "a refused build must publish no canonical record"
    # The launcher's read must FAIL (which a shell turns into an empty PKG).
    read_rc, stdout = _launcher_read(canonical)
    assert read_rc != 0 and stdout == "", (read_rc, stdout)
    assert _launcher_guard(canonical) != 0, "the launcher's non-empty check must not pass"
    evidence = json.loads((out / "receipts" / "build-record.json").read_text())
    assert evidence["published"] is False
    assert evidence["refusals"]


def test_the_launcher_reads_a_published_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive half of the same pattern, so the guard above is discriminating."""

    out = tmp_path / "arm"
    calls = _stub_preinstall(monkeypatch, tmp_path)
    _stub_verifiers(monkeypatch, calls)
    assert arm.main(["--source-sha", "a" * 40, "--out", str(out), "--execute"]) == 0

    read_rc, stdout = _launcher_read(out / "build-record.json")
    assert read_rc == 0
    assert stdout == "d" * 64
    assert _launcher_guard(out / "build-record.json") == 0


def test_a_malformed_verification_report_is_a_refusal_with_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-3: invalid JSON is unverified, not a traceback out of ``_execute``."""

    out = tmp_path / "arm"
    calls = _stub_preinstall(monkeypatch, tmp_path)

    def broken_json(monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            if "osworld_arm_verify.py" in " ".join(argv):
                target = Path(argv[argv.index("--out") + 1])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{not json")
                calls.append("verify")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        monkeypatch.setattr(arm.subprocess, "run", fake_run)

    broken_json(monkeypatch)
    rc = arm.main(["--source-sha", "a" * 40, "--out", str(out), "--execute"])
    assert rc == 4, "an unparseable report must exit as UNVERIFIED"
    assert (out / "receipts" / "installed-verification.json").read_text() == "{not json"
    assert not (out / "build-record.json").exists()
    evidence = json.loads((out / "receipts" / "build-record.json").read_text())
    assert any("not readable JSON" in reason for reason in evidence["refusals"])


def test_a_volatile_arm_root_is_refused(tmp_path: Path) -> None:
    """An arm under a purgeable root is one OS sweep away from a half receipt.

    The check itself is shared with ``scripts/build_osworld_adapter.py`` through
    ``runner.durable_root``, so the arm and the episode agree on "durable".
    """

    # Explicitly under /private/tmp rather than tmp_path: pytest's basetemp is not
    # always there (a custom --basetemp made this pass for the wrong reason), and
    # the refusal is a property of the path, not of the runner.
    with pytest.raises(arm.BuildRefused, match="purge"):
        arm.refuse_volatile(Path("/private/tmp/lop-arm-volatile-probe"), "arm root")


# --- the verifier ----------------------------------------------------------


def test_the_verifier_writes_its_report_when_it_refuses(tmp_path: Path) -> None:
    """A refusal without its evidence is not actionable.

    The report is written BEFORE the exit code is decided, so an exit-1 refusal
    still leaves the module hashes, the installed set and the refusal reasons.
    """

    source = tmp_path / "source"
    (source / "local_operator").mkdir(parents=True)
    out = tmp_path / "installed-verification.json"
    rc = verify.main(["--source", str(source), "--commit", "a" * 40, "--out", str(out)])
    assert rc == 1
    report = json.loads(out.read_text())
    assert report["verified"] is False
    assert report["refusals"]
    # The repo venv has no adapter distribution, so that is a refusal too -- and
    # the report still carries the core-source rows it managed to gather.
    assert report["core_source"]
    assert any("adapter is not installed" in reason for reason in report["refusals"])


def test_the_cross_build_digest_ignores_installer_written_rows() -> None:
    """``package_digest`` moves between installs of one commit; this must not.

    ``direct_url.json`` records the wheel path and ``uv_cache.json`` a timestamp,
    and both live in the installed RECORD -- which is why the selector digest is a
    this-install identity and needs a companion for drift questions.
    """

    def record(extra: str) -> list[str]:
        return [
            "lop_osworld_v2_adapter/__init__.py,sha256=aaa,1",
            f"lop_osworld_v2_adapter-0.1.3.dist-info/{extra},sha256=bbb,2",
            "lop_osworld_v2_adapter-0.1.3.dist-info/RECORD,,",
        ]

    first = verify._source_record_digest(record("direct_url.json"))
    second = verify._source_record_digest(record("uv_cache.json"))
    assert first == second
    changed = verify._source_record_digest(
        [
            "lop_osworld_v2_adapter/__init__.py,sha256=ccc,1",
            "lop_osworld_v2_adapter-0.1.3.dist-info/direct_url.json,sha256=bbb,2",
            "lop_osworld_v2_adapter-0.1.3.dist-info/RECORD,,",
        ]
    )
    assert changed != first


def test_the_record_self_row_is_excluded_by_its_path_field() -> None:
    """A real RECORD self-row ends in ``,,``; a suffix test on the whole row missed it.

    Excluding it matters because ``source_record_digest`` is meant to cover the
    wheel's OWN files: RECORD cannot hash itself, and its row carries the
    dist-info directory name.
    """

    without_self = ["lop_osworld_v2_adapter/__init__.py,sha256=aaa,1"]
    with_self = [*without_self, "lop_osworld_v2_adapter-0.1.3.dist-info/RECORD,,"]
    assert verify._source_record_digest(with_self) == verify._source_record_digest(without_self)


def test_a_source_file_whose_name_contains_an_installer_row_is_kept() -> None:
    """The installer rows are matched by exact path component, not by substring.

    A prefix match would drop ``REQUESTED_handlers.py`` — real code — from the
    identity, and would do it silently.
    """

    base = ["lop_osworld_v2_adapter/__init__.py,sha256=aaa,1"]
    decoy = [
        *base,
        "lop_osworld_v2_adapter/REQUESTED_handlers.py,sha256=ccc,3",
    ]
    assert verify._source_record_digest(decoy) != verify._source_record_digest(base)
    # ... and the installer's own exact-name row is still dropped.
    with_installer = [
        *base,
        "lop_osworld_v2_adapter-0.1.3.dist-info/REQUESTED,sha256=ddd,0",
    ]
    assert verify._source_record_digest(with_installer) == verify._source_record_digest(base)


# --- fixtures and helpers --------------------------------------------------


def _metadata(root: Path) -> arm.Metadata:
    adapter = root / "metadata" / "benchmarks" / "osworld_v2_adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "pyproject.toml").write_text("[tool.uv]\noverride-dependencies = []\n")
    (adapter / "uv.lock").write_text('[[package]]\nname = "requests"\nversion = "2.34.2"\n')
    (root / "metadata" / "pyproject.toml").write_text("[project]\ndependencies = []\n")
    return arm.Metadata(adapter, root / "metadata" / "pyproject.toml")


def _whl(root: Path, name: str) -> Path:
    path = root / "dist" / f"{name}.whl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a wheel")
    return path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway repository whose ``origin/main`` is its ``main``.

    Built rather than borrowed from the real checkout: ``guard_source`` asserts
    ancestry against ``origin/main``, and a test that depends on the real
    repository's branch state fails for reasons unrelated to this code. The
    identity is local config, and the system/global git config is bypassed so an
    operator's credential helper is never consulted.
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
