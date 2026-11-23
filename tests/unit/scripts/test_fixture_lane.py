"""Decision table for scripts/fixture_lane.py (the #4074 affected lane).

WHY: the lane may skip most of the merge gate, so every case that must NOT
skip is pinned here against a real throwaway git repository (the planner reads
`git diff --raw`, `git show` and `git grep`, so a mock would test nothing).
Anything outside the proven shape must plan `full`.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from scripts import fixture_lane

BODY = (
    "import pytest\n\n\n@pytest.mark.parametrize('x', [1, 2])\n"
    "def test_a(x):\n    assert x {op} 0\n"
)
ENV = {
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "PATH": os.environ["PATH"],
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


# Mapping, not dict: dict is invariant in its value type, so a caller's
# dict[str, str] literal is rejected; these helpers only read the mapping.
def _commit(repo: Path, files: Mapping[str, str | None]) -> str:
    for rel, text in files.items():
        path = repo / rel
        if text is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c", "--allow-empty")
    return _git(repo, "rev-parse", "HEAD")


# Synthetic stand-ins for the whole-tree reader catalog, so a throwaway repo can
# be `affected` at all (a missing catalog entry is FULL, by design).
READER_SRC = {
    node.partition("::")[0]: f"def {node.partition('::')[2]}():\n    pass\n"
    for node in fixture_lane.GLOBAL_READERS
}


def _plan(
    repo: Path,
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
    event: str = "pull_request",
    readers: bool = True,
):
    base = _commit(repo, {**(READER_SRC if readers else {}), **before})
    head = _commit(repo, after)
    return fixture_lane.plan(repo, event=event, base=base, head=head)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


T = "tests/unit/test_alpha.py"


def test_body_only_edit_of_two_modules_is_affected(repo: Path) -> None:
    seed = {T: BODY.format(op=">"), "tests/unit/sub/test_beta.py": BODY.format(op=">")}
    lane, targets, _ = _plan(
        repo, seed, {T: BODY.format(op=">="), "tests/unit/sub/test_beta.py": BODY.format(op=">=")}
    )
    assert (lane, sorted(targets)) == (
        "affected",
        sorted([*seed, *fixture_lane.GLOBAL_READERS]),
    )


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch", ""])
def test_every_non_pr_event_is_full(repo: Path, event: str) -> None:
    assert _plan(repo, {T: BODY.format(op=">")}, {T: BODY.format(op=">=")}, event)[0] == "full"


def test_new_import_is_full(repo: Path) -> None:
    after = "import os\n" + BODY.format(op=">")
    assert _plan(repo, {T: BODY.format(op=">")}, {T: after})[0] == "full"


@pytest.mark.parametrize(
    "extra",
    [
        "VALUE = compute()\n",  # a call at import
        "pytest_plugins = ['x']\n",  # plugin registration
        "import sys\nsys.path.append('x')\n",  # global mutation
        "class M(type):\n    pass\n\n\nclass K(metaclass=M):\n    pass\n",  # metaclass
        "class K(Base):\n    pass\n",  # unknown inheritance
        "@pytest.fixture(scope='session')\ndef fx():\n    return 1\n",  # session fixture
        "@dynamic\ndef f():\n    pass\n",  # unknown decorator
        "def f(a=make()):\n    pass\n",  # default executes at import
        "if True:\n    pass\n",  # control flow
    ],
)
def test_import_time_hazards_are_full(repo: Path, extra: str) -> None:
    # The hazard is in BOTH versions' module level, so only the head/base check
    # (not the import diff) can refuse it.
    seed = BODY.format(op=">") + extra
    assert _plan(repo, {T: seed}, {T: seed.replace(">", ">=")})[0] == "full"


def test_module_scope_fixture_is_still_affected(repo: Path) -> None:
    seed = BODY.format(op=">") + "\n\n@pytest.fixture(scope='module')\ndef fx():\n    return 1\n"
    assert _plan(repo, {T: seed}, {T: seed.replace(">", ">=")})[0] == "affected"


@pytest.mark.parametrize(
    "other",
    [
        "from tests.unit import test_alpha\n",
        "from .test_alpha import test_a\n",
        "PATH = 'tests/unit/test_alpha.py'\n",
    ],
)
def test_another_file_naming_the_module_is_full(repo: Path, other: str) -> None:
    seed = {T: BODY.format(op=">"), "tests/unit/test_user.py": other}
    assert _plan(repo, seed, {T: BODY.format(op=">=")})[0] == "full"


@pytest.mark.parametrize(
    "path",
    [
        "tests/unit/conftest.py",
        "tests/unit/sub/conftest.py",
        "conftest.py",
        "tests/unit/__init__.py",
        "tests/unit/helpers.py",
        "tests/unit/tui/test_x.py",
        "tests/e2e/test_x.py",
        "tests/unit/test_perf_x.py",
        "local_operator/x.py",
        ".github/workflows/ci.yml",
    ],
)
def test_ineligible_paths_are_full(repo: Path, path: str) -> None:
    assert _plan(repo, {path: "A = 1\n"}, {path: "A = 2\n"})[0] == "full"


def test_added_deleted_and_mixed_changes_are_full(repo: Path) -> None:
    body = BODY.format(op=">")
    assert _plan(repo, {T: body}, {"tests/unit/test_new.py": body})[0] == "full"  # add
    assert _plan(repo, {T: body}, {T: None})[0] == "full"  # delete


def test_rename_mode_and_symlink_are_full(repo: Path) -> None:
    body = BODY.format(op=">")
    _commit(repo, {T: body})
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "mv", T, "tests/unit/test_gamma.py")
    _git(repo, "commit", "-qm", "r")
    assert (
        fixture_lane.plan(
            repo, event="pull_request", base=base, head=_git(repo, "rev-parse", "HEAD")
        )[0]
        == "full"
    )
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "tests/unit/test_gamma.py").chmod(0o755)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "m")
    assert (
        fixture_lane.plan(
            repo, event="pull_request", base=base, head=_git(repo, "rev-parse", "HEAD")
        )[0]
        == "full"
    )


def test_zero_too_many_unparsable_and_unreadable_are_full(repo: Path) -> None:
    body = BODY.format(op=">")
    assert _plan(repo, {T: body}, {})[0] == "full"  # empty diff: zero targets
    many = {f"tests/unit/test_m{i}.py": body for i in range(11)}
    assert _plan(repo, many, {k: v.replace(">", ">=") for k, v in many.items()})[0] == "full"
    assert _plan(repo, {T: body}, {T: "def ("})[0] == "full"
    lane, _, why = fixture_lane.plan(repo, event="pull_request", base="0" * 40, head="HEAD")
    assert lane == "full" and why


def test_cli_prints_the_outputs_the_workflow_reads(repo: Path, tmp_path: Path) -> None:
    base = _commit(repo, {**READER_SRC, T: BODY.format(op=">")})
    head = _commit(repo, {T: BODY.format(op=">=")})
    out = tmp_path / "out.txt"
    assert (
        fixture_lane.main(
            [
                "--event",
                "pull_request",
                "--base",
                base,
                "--head",
                head,
                "--root",
                str(repo),
                "--github-output",
                str(out),
            ]
        )
        == 0
    )
    assert out.read_text().splitlines()[:2] == [
        "lane=affected",
        f"targets={json.dumps([T, *fixture_lane.GLOBAL_READERS])}",
    ]


def test_workflow_runs_only_shard_zero_of_an_affected_lane_and_skips_coverage() -> None:
    """Static wiring: idle legs, unchanged six-shard matrix, honest coverage and gate."""
    wf = yaml.safe_load(Path(".github/workflows/full-suite.yml").read_text())["jobs"]
    test = wf["test"]
    assert test["strategy"]["matrix"]["shard"] == [0, 1, 2, 3, 4, 5]
    # PR merge gates run Python 3.13 only; push/schedule/workflow_dispatch keep
    # both ["3.12", "3.13"] while shard stays a literal six-element YAML list.
    assert test["strategy"]["matrix"]["python-version"] == (
        "${{ fromJSON(github.event_name == 'pull_request' "
        '&& \'["3.13"]\' || \'["3.12", "3.13"]\') }}'
    )
    assert (
        test["env"]["LEG_IDLE"]
        == "${{ needs.changes.outputs.lane == 'affected' && matrix.shard != 0 }}"
    )
    # Every step but the bare checkout is skipped on an idle leg.
    for step in test["steps"][2:]:  # checkout + the unskippable merge rebuild
        # Pinned-by-hygiene exceptions: the serial steps follow a skipped pytest step, and
        # the junit upload stays `always()` (`if-no-files-found: ignore` makes it a no-op).
        exempt = step.get("id", "").startswith("serial_") or step["name"] == "Upload junit timings"
        assert "env.LEG_IDLE != 'true'" in step["if"] or exempt, step["name"]
    assert wf["coverage-report"]["if"] == "needs.test.outputs.lane != 'affected'"
    gate = "\n".join(s.get("run", "") for s in wf["merge-gate"]["steps"])
    assert (
        "lane=affected but test=" in gate and "!= affected" in gate
    )  # unknown lane is judged FULL


# --- #4074 round-2 remediation -------------------------------------------------


@pytest.mark.parametrize(
    "ann",
    [
        "x: os.system('echo hazard-literal') = 1\n",  # module-level annotation call
        "x: make_type()\n",  # bare annotation, no value
        "class C:\n    y: make_type() = 1\n",  # class-level annotation
    ],
)
def test_import_time_annotation_hazards_are_full(repo: Path, ann: str) -> None:
    """A1: an AnnAssign annotation is evaluated at import, so a call there is FULL.

    Source only: the strings are parsed, never executed.
    """
    seed = BODY.format(op=">") + "\n" + ann
    assert _plan(repo, {T: seed}, {T: seed.replace(">", ">=")})[0] == "full"


def test_a_changed_annotation_call_is_full_and_plain_types_stay_affected(repo: Path) -> None:
    ok = BODY.format(op=">") + "\nx: list[int] = []\ny: int | None = None\n"
    assert _plan(repo, {T: ok}, {T: ok.replace(">", ">=")})[0] == "affected"
    # Even under the future import (annotations are only strings there) the
    # planner stays conservative: a hazard is FULL.
    future = "from __future__ import annotations\n" + BODY.format(op=">")
    bad = future + "\nz: make_type() = 1\n"
    assert _plan(repo, {T: bad}, {T: bad.replace(">", ">=")})[0] == "full"


def test_a_scanner_that_names_no_target_is_still_in_the_plan(repo: Path) -> None:
    """R1-M1: a whole-tree reader never names the edited module, yet must run."""
    scanner = "tests/unit/test_census.py"
    seed = {
        T: BODY.format(op=">"),
        scanner: "from pathlib import Path\n\n\ndef test_census():\n"
        "    assert not [p for p in Path('tests').rglob('test_*.py')\n"
        "                if 'harmless-needle' in p.read_text()]\n",
    }
    lane, targets, why = _plan(repo, seed, {T: BODY.format(op=">=")})
    assert lane == "affected", why
    assert targets == [T, *fixture_lane.GLOBAL_READERS]  # the edited file + readers only
    assert scanner not in targets  # the synthetic scanner is NOT named; the catalog is
    assert "2 whole-tree guard node(s)" in why


def test_missing_renamed_or_edited_reader_is_full(repo: Path) -> None:
    node = fixture_lane.GLOBAL_READERS[0]
    rel, _, func = node.partition("::")
    renamed = {rel: READER_SRC[rel].replace(func, func + "_renamed")}
    lane, targets, why = _plan(repo, {T: BODY.format(op=">"), **renamed}, {T: BODY.format(op=">=")})
    assert (lane, targets) == ("full", []) and "catalog entry not found" in why


def test_catalog_file_absent_is_full(repo: Path) -> None:
    lane, _, why = _plan(repo, {T: BODY.format(op=">")}, {T: BODY.format(op=">=")}, readers=False)
    assert lane == "full" and "unreadable" in why


def test_editing_a_reader_file_itself_is_full(repo: Path) -> None:
    rel = fixture_lane.GLOBAL_READERS[0].partition("::")[0]
    src = READER_SRC[rel]
    lane, _, why = _plan(repo, {T: BODY.format(op=">")}, {rel: src + "\n# edited\n"})
    assert lane == "full"


def test_the_real_catalog_resolves_in_this_checkout() -> None:
    """Rot pin: every catalog node is a real function in the real tree."""
    import ast

    for node in fixture_lane.GLOBAL_READERS:
        rel, _, func = node.partition("::")
        tree = ast.parse(Path(rel).read_text())
        assert any(isinstance(n, ast.FunctionDef) and n.name == func for n in ast.walk(tree)), node


def test_reason_cr_and_lf_cannot_forge_an_output_line(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "o.txt"
    base = _commit(repo, {T: BODY.format(op=">")})
    head = _commit(repo, {T: BODY.format(op=">="), "tests/unit/test_new.py": "x\n"})
    fixture_lane.main(
        ["--event", "pull_request", "--base", base, "--head", head, "--root", str(repo),
         "--github-output", str(out)]
    )  # fmt: skip
    for line in out.read_text().splitlines():
        assert line.split("=", 1)[0] in {
            "lane", "targets", "reason", "planner_base", "planner_blob"
        }  # fmt: skip
    assert "\r" not in out.read_text()


def _extract_run() -> str:
    wf = yaml.safe_load(Path(".github/workflows/full-suite.yml").read_text())["jobs"]
    step = next(s for s in wf["test"]["steps"] if "LANE_TARGETS" in (s.get("run") or "")
                or "LANE_TARGETS" in str(s.get("env")))  # fmt: skip
    return step["run"]


@pytest.mark.parametrize("targets", ["[]", "{}", '[""]', '["../etc/passwd"]', "[1]", "null"])
def test_the_affected_extract_step_refuses_an_empty_or_malformed_list(
    tmp_path: Path, targets: str
) -> None:
    """A3: an empty list would make `pytest` collect EVERYTHING; the step must fail."""
    done = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _extract_run()],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "LANE": "affected", "LANE_TARGETS": targets},
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0, done.stdout


def test_the_affected_extract_step_writes_a_valid_list_and_full_is_untouched(
    tmp_path: Path,
) -> None:
    nodes = [T, *fixture_lane.GLOBAL_READERS]
    done = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _extract_run()],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "LANE": "affected", "LANE_TARGETS": json.dumps(nodes)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "shard_tests.txt").read_text().split() == nodes


def test_serial_controls_do_not_run_on_the_affected_lane_but_do_on_full() -> None:
    """A2: the six serial controls (up to 7 min) are unrelated to the targets."""
    wf = yaml.safe_load(Path(".github/workflows/full-suite.yml").read_text())["jobs"]
    serial = next(s for s in wf["test"]["steps"] if s.get("id") == "serial_controls")
    assert serial["if"] == "matrix.shard == 0 && needs.changes.outputs.lane != 'affected'"
    proof = next(s for s in wf["test"]["steps"] if s.get("id") == "serial_proof")
    assert "steps.serial_controls.outcome != 'skipped'" in proof["if"]
    gate = "\n".join(s.get("run", "") for s in wf["merge-gate"]["steps"])
    assert "3.12, 3.13" not in gate  # no hard-coded Python versions in the summary
