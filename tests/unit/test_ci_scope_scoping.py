"""The local file-level selector in `scripts/ci_scope.py`.

WHY THESE EXIST
---------------
The selector narrows what a LOCAL `make check-changed` runs. CI is untouched,
but a local green is now a claim about a SUBSET of the tree, so the way that
claim can be wrong needs a test that fails:

* the narrowing runs too little — a file was missed, or the graph was believed
  where it cannot see (a computed-name import, an unparsable module, a
  structural path). Each of those has an assertion here, driven through the
  public `scope_plan`, because the fallback IS the feature;
* the narrowing runs something CI does not — `_narrow` rewrites a command out
  of `JOB_COMMANDS`, so the drift answer is asserted in
  `tests/unit/test_ci_hygiene.py`, which already owns the CI-vs-local contract.

A synthetic repository is used throughout: the selection rules are about graph
SHAPE (a transitive edge, a converging dependency, a barrier, an arm that
fires), and a ten-file fixture states the shape without depending on this
repository's layout staying put. The fixture's WEIGHTS are deliberately
lopsided, mirroring the real suite (`tests/durations.json` puts 82.3% of the
weight in a quarter of the files), so "cheap selection" means cheap by weight
and not merely by file count.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts import ci_scope

#: Filler tests that import `gamma` and nothing else. They exist to keep the
#: fixture's file count above the fraction arms' reach while carrying a weight
#: the arms can see: without them, selecting "the three tests that reach alpha"
#: would be 3 of 4 files and the file arm would (correctly) fire.
FILLERS = 6

#: The same, for the e2e tree — whose universe is 49 files in the real repo, so a
#: one-file selection there is 2% of it, not 100%.
E2E_FILLERS = 11

#: The fixture's default weights: every filler is heavy, everything a change
#: under test reaches is light — the same asymmetry the real manifest has.
_FILLER_SECONDS = 10.0
_LIGHT_SECONDS = 0.1


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _fixture_repo(
    root: Path,
    *,
    durations: dict[str, float] | None = None,
    fallback_seconds: float = _LIGHT_SECONDS,
) -> Path:
    """A small package with eleven unit tests, one e2e test and one script.

    Shape (the arrow is "imports"):

        tests/unit/test_alpha.py       -> alpha
        tests/unit/test_beta.py        -> beta -> alpha      (transitive edge)
        tests/unit/tui/test_widget.py  -> tui/widget -> alpha
        tests/unit/test_gamma.py       -> gamma
        tests/unit/test_filler_N.py    -> gamma              (N = FILLERS)
        tests/e2e/test_boot.py         -> alpha
        scripts/tool.py                -> nothing in the tree imports it
    """
    _write(root, "local_operator/__init__.py", "")
    _write(root, "local_operator/alpha.py", "ALPHA = 1\n")
    _write(root, "local_operator/beta.py", "from local_operator.alpha import ALPHA\n")
    _write(root, "local_operator/gamma.py", "GAMMA = 3\n")
    _write(root, "local_operator/tui/__init__.py", "")
    _write(root, "local_operator/tui/widget.py", "from local_operator.alpha import ALPHA\n")
    _write(root, "scripts/tool.py", "TOOL = True\n")
    _write(root, "tests/unit/__init__.py", "")
    _write(root, "tests/unit/test_alpha.py", "from local_operator.alpha import ALPHA\n")
    _write(root, "tests/unit/test_beta.py", "from local_operator.beta import ALPHA\n")
    _write(root, "tests/unit/test_gamma.py", "from local_operator.gamma import GAMMA\n")
    _write(root, "tests/unit/tui/test_widget.py", "from local_operator.tui.widget import ALPHA\n")
    for index in range(FILLERS):
        _write(
            root,
            f"tests/unit/test_filler_{index}.py",
            "from local_operator.gamma import GAMMA\n",
        )
    _write(root, "tests/e2e/test_boot.py", "from local_operator.alpha import ALPHA\n")
    for index in range(E2E_FILLERS):
        _write(
            root,
            f"tests/e2e/test_e2e_{index}.py",
            "from local_operator.gamma import GAMMA\n",
        )
    _write(root, "conftest.py", "")
    if durations is None:
        durations = {
            f"tests/unit/test_filler_{index}.py": _FILLER_SECONDS for index in range(FILLERS)
        }
        for light in (
            "tests/unit/test_alpha.py",
            "tests/unit/test_beta.py",
            "tests/unit/test_gamma.py",
            "tests/unit/tui/test_widget.py",
        ):
            durations[light] = _LIGHT_SECONDS
    _write(
        root,
        "tests/durations.json",
        json.dumps(
            {
                "_comment": "written by the test fixture",
                "durations": durations,
                "fallback_seconds": fallback_seconds,
            }
        ),
    )
    return root


def _write_durations(root: Path, durations: dict[str, float], fallback: float) -> None:
    _write(
        root,
        "tests/durations.json",
        json.dumps({"durations": durations, "fallback_seconds": fallback}),
    )


def _plan(
    root: Path, paths: list[str], jobs: tuple[str, ...] = ("lint", "test", "tui-e2e")
) -> Mapping[str, ci_scope.ScopeDecision]:
    return ci_scope.scope_plan(list(jobs), paths, root).decisions


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_the_graph_resolves_imports_by_parsing_and_never_imports(tmp_path):
    """The module under test raises on import, so an import would fail here."""
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/alpha.py", "raise RuntimeError('the graph imported me')\n")
    _write(root, "local_operator/beta.py", "from .alpha import ALPHA\n")
    graph = ci_scope.build_import_graph(root)

    assert graph.unreadable == ()
    assert graph.imports["local_operator/beta.py"] == frozenset({"local_operator/alpha.py"})
    assert graph.imports["local_operator/tui/widget.py"] == frozenset({"local_operator/alpha.py"})


def test_dependents_are_transitive_and_run_the_edge_backwards(tmp_path):
    root = _fixture_repo(tmp_path)
    graph = ci_scope.build_import_graph(root)

    dependents = graph.dependents({"local_operator/alpha.py"})
    assert "local_operator/beta.py" in dependents, "a direct importer is a dependent"
    assert "tests/unit/test_beta.py" in dependents, "a dependent through beta is transitive"
    assert "tests/unit/tui/test_widget.py" in dependents
    assert "tests/unit/test_gamma.py" not in dependents, "gamma does not reach alpha"
    assert "local_operator/alpha.py" not in dependents, "a seed is not its own dependent"


def test_an_unparsable_module_makes_every_graph_job_whole_tree(tmp_path):
    """Fail OPEN: a file the graph cannot read could import anything."""
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/broken.py", "def broken(:\n")

    decisions = _plan(root, ["local_operator/alpha.py"])
    for job in ("test", "tui-e2e"):
        decision = decisions[job]
        assert decision.whole_tree, f"{job} was narrowed on an unbuildable graph"
        assert decision.commands == ci_scope.JOB_COMMANDS[job]
    assert any("could not be built" in note for note in decisions["test"].notes)
    assert any("local_operator/broken.py" in note for note in decisions["test"].notes)


def test_lint_still_narrows_when_the_graph_is_unbuildable(tmp_path):
    """Lint takes no graph input, so a broken graph must not cost it its scope."""
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/broken.py", "def broken(:\n")

    decisions = _plan(root, ["local_operator/alpha.py"])
    assert not decisions["lint"].whole_tree
    assert "-m flake8 local_operator/alpha.py" in decisions["lint"].commands[0]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_a_changed_module_selects_the_tests_that_reach_it(tmp_path):
    root = _fixture_repo(tmp_path)
    decisions = _plan(root, ["local_operator/alpha.py"])

    assert decisions["test"].targets == (
        "tests/unit/test_alpha.py",
        "tests/unit/test_beta.py",
        "tests/unit/tui/test_widget.py",
    )
    assert "tests/unit/test_filler_0.py" not in decisions["test"].commands[0]
    assert decisions["tui-e2e"].targets == ("tests/e2e/test_boot.py",)
    assert "tests/e2e/test_e2e_0.py" not in decisions["tui-e2e"].commands[0]


def test_a_changed_test_file_is_selected_even_with_no_dependents(tmp_path):
    """The common case: an author iterating on the test they are writing."""
    root = _fixture_repo(tmp_path)
    decisions = _plan(root, ["tests/unit/test_gamma.py"])

    assert decisions["test"].targets == ("tests/unit/test_gamma.py",)
    assert not decisions["tui-e2e"].commands, "no e2e file reaches a unit test"


def test_a_change_no_test_reaches_runs_no_test_command(tmp_path):
    root = _fixture_repo(tmp_path)
    decisions = _plan(root, ["scripts/tool.py"])

    assert decisions["test"].commands == ()
    assert not decisions["test"].whole_tree
    assert "transitively imports" in decisions["test"].notes[0]
    assert decisions["lint"].targets == ("scripts/tool.py",)


def test_a_computed_name_import_in_a_test_selects_what_it_can_load(tmp_path):
    """`import_module(f"…oauth.{name}")` is an edge the AST cannot name.

    The literal head is what the graph has, so a change anywhere under it pulls
    in the hub — the test that drives it — instead of silently missing it.
    """
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/oauth/__init__.py", "")
    _write(root, "local_operator/oauth/anthropic.py", "OAUTH = True\n")
    _write(
        root,
        "tests/unit/test_oauth.py",
        """
        import importlib

        MODULES = ("anthropic",)

        def test_login():
            for name in MODULES:
                importlib.import_module(f"local_operator.oauth.{name}")
        """,
    )
    graph = ci_scope.build_import_graph(root)
    assert graph.prefixes == {"local_operator.oauth.": ("tests/unit/test_oauth.py",)}

    decisions = _plan(root, ["local_operator/oauth/anthropic.py"])
    assert "tests/unit/test_oauth.py" in decisions["test"].targets


def test_a_namespace_package_import_keeps_its_edge(tmp_path):
    """`from scripts import tool` — `scripts/` has no `__init__.py` here.

    That is how this repo's `scripts/` and `tests/` trees are importable, and
    `tests/unit/test_ci_hygiene.py` reaches `scripts/ci_scope.py` exactly this
    way. Missing the edge fails OPEN: the script's own test would not be
    selected, and a narrowed run would report "nothing to run" for a diff that
    has a test.
    """
    root = _fixture_repo(tmp_path)
    assert not (root / "scripts" / "__init__.py").exists()
    _write(root, "tests/unit/test_tool.py", "from scripts import tool\n")

    graph = ci_scope.build_import_graph(root)
    assert graph.imports["tests/unit/test_tool.py"] == frozenset({"scripts/tool.py"})

    decisions = _plan(root, ["scripts/tool.py"])
    assert decisions["test"].targets == ("tests/unit/test_tool.py",)


def test_a_module_attribute_import_does_not_invent_a_module(tmp_path):
    """`from x import name` where `x` is a module is not a submodule edge.

    On a case-insensitive filesystem the naive form resolves
    `from .alpha import ALPHA` onto `local_operator/alpha.py` and calls it a
    dependency, which is how this was found.
    """
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/beta.py", "from .alpha import ALPHA\n")
    graph = ci_scope.build_import_graph(root)

    assert graph.imports["local_operator/beta.py"] == frozenset({"local_operator/alpha.py"})


def test_a_literal_name_import_in_a_hub_selects_the_module_it_names(tmp_path):
    """A registry that names a module in a literal is a real dependency."""
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/providers/oauth_deep.py", "DEEP = True\n")
    _write(
        root,
        "local_operator/registry.py",
        """
        import importlib

        LAZY = {"deep": "local_operator.providers.oauth_deep"}

        def load(key):
            return importlib.import_module(LAZY[key])
        """,
    )
    _write(root, "tests/unit/test_registry.py", "from local_operator.registry import load\n")

    decisions = _plan(root, ["local_operator/providers/oauth_deep.py"])
    assert "tests/unit/test_registry.py" in decisions["test"].targets


# ---------------------------------------------------------------------------
# The fraction arms
# ---------------------------------------------------------------------------


def test_a_selection_that_is_light_by_file_count_but_heavy_by_weight_falls_back(tmp_path):
    """The measured shape of this suite: weight is not spread over files.

    One selected file of ten, carrying 98% of the weight. A file-count arm alone
    would wave this through — and the scoped run would cost what it was scoping
    away from.
    """
    root = _fixture_repo(tmp_path)
    durations = {f"tests/unit/test_filler_{index}.py": 1.0 for index in range(FILLERS)}
    durations.update(
        {
            "tests/unit/test_alpha.py": 1.0,
            "tests/unit/test_beta.py": 1.0,
            "tests/unit/test_gamma.py": 1.0,
            "tests/unit/tui/test_widget.py": 500.0,
        }
    )
    _write_durations(root, durations, 1.0)

    decision = _plan(root, ["local_operator/tui/widget.py"])["test"]
    assert decision.whole_tree, "1 of 10 files but 98% of the weight must not be scoped"
    assert any("weight arm" in note for note in decision.notes), decision.notes
    assert decision.commands == ci_scope.JOB_COMMANDS["test"]


def test_a_selection_over_half_the_files_falls_back_even_when_it_is_cheap(tmp_path):
    root = _fixture_repo(tmp_path)
    # Seven of ten files selected; only those seven are light, so ONLY the file
    # arm can fire here and the assertion says which one did.
    durations = {f"tests/unit/test_filler_{index}.py": 0.01 for index in range(FILLERS)}
    durations.update(
        {
            "tests/unit/test_alpha.py": 50.0,
            "tests/unit/test_beta.py": 50.0,
            "tests/unit/tui/test_widget.py": 50.0,
            "tests/unit/test_gamma.py": 0.01,
        }
    )
    _write_durations(root, durations, 0.01)

    decision = _plan(root, ["local_operator/gamma.py"])["test"]
    assert decision.whole_tree
    assert any("file arm" in note for note in decision.notes), decision.notes
    assert not any("weight arm" in note for note in decision.notes), decision.notes


def test_without_a_duration_manifest_the_file_arm_uses_the_weight_fraction(tmp_path):
    """An unreadable manifest must not buy a LOOSER arm."""
    root = _fixture_repo(tmp_path)
    (root / "tests/durations.json").write_text("{ not json", encoding="utf-8")

    decision = _plan(root, ["local_operator/gamma.py"])["test"]
    assert decision.whole_tree, "7 of 10 files is above the weight fraction too"
    assert any("unreadable" in note for note in decision.notes), decision.notes


# ---------------------------------------------------------------------------
# Barriers — the fail-open list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changed",
    [
        "conftest.py",
        "tests/unit/conftest.py",
        "tests/unit/tui/conftest.py",
        "pyproject.toml",
        "Makefile",
        ".flake8",
        "uv.lock",
        ".github/workflows/ci.yml",
        "local_operator/__init__.py",
        "tests/unit/__init__.py",
        "local_operator/cli.py",
        "local_operator/__main__.py",
        "tests/helpers/support.py",
        "local_operator/tui/theme.tcss",
        "tests/unit/golden.json",
        "docs/tooling.py",
        "local_operator/types.pyi",
        "benchmarks/osworld_v2_adapter/src/evaluation_examples/adapter.py",
        "extension/src/boot.ts",
        "static/logo.png",
        "docs/figure.png",
    ],
)
def test_a_barrier_path_runs_the_whole_command_and_names_itself(tmp_path, changed):
    root = _fixture_repo(tmp_path)
    for rel in (
        "tests/helpers/support.py",
        "local_operator/tui/theme.tcss",
        "tests/unit/golden.json",
        "docs/tooling.py",
        "local_operator/types.pyi",
        "benchmarks/osworld_v2_adapter/src/evaluation_examples/adapter.py",
        "extension/src/boot.ts",
        "static/logo.png",
        "docs/figure.png",
    ):
        _write(root, rel, "TOOL = 1\n")

    decisions = _plan(root, [changed])
    for job in ("lint", "test", "tui-e2e"):
        decision = decisions[job]
        assert decision.whole_tree, f"{job} narrowed past the barrier {changed}"
        assert decision.commands == ci_scope.JOB_COMMANDS[job]
        assert any(changed in note for note in decision.notes), (changed, decision.notes)


def test_documentation_does_not_force_the_whole_tree(tmp_path):
    """The whitelist's other half: a path no gate reads must stay narrowable.

    Without this, the whitelist would send every PR that touches a `.md` back to
    the whole-tree commands — and the classifier already calls `docs/**` inert
    for the same reason (no gate reads it).
    """
    root = _fixture_repo(tmp_path)
    _write(root, "README.md", "# Read me\n")
    _write(root, "docs/guide.md", "# Guide\n")

    decisions = _plan(root, ["README.md", "local_operator/alpha.py", "docs/guide.md"])
    assert not decisions["test"].whole_tree
    assert decisions["test"].targets == (
        "tests/unit/test_alpha.py",
        "tests/unit/test_beta.py",
        "tests/unit/tui/test_widget.py",
    )
    assert decisions["lint"].targets == ("local_operator/alpha.py",)


def test_a_markdown_file_under_a_covered_tree_is_package_data_not_docs(tmp_path):
    """`local_operator/prompts_md/x.md` is read at run time; `docs/x.md` is not."""
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/prompts_md/system.md", "# Prompt\n")

    decisions = _plan(root, ["local_operator/prompts_md/system.md"])
    assert decisions["test"].whole_tree
    assert any("read by path" in note for note in decisions["test"].notes)


def test_a_deleted_module_is_a_barrier(tmp_path):
    """Its importers can no longer be resolved, so the graph cannot select them."""
    root = _fixture_repo(tmp_path)
    (root / "local_operator/gamma.py").unlink()

    decisions = _plan(root, ["local_operator/gamma.py"])
    assert decisions["test"].whole_tree
    assert any("deleted module" in note for note in decisions["test"].notes)


def test_a_gate_command_that_lost_its_input_falls_back_instead_of_guessing(tmp_path, monkeypatch):
    """Drift safety: `_narrow` must refuse to guess which token to replace."""
    root = _fixture_repo(tmp_path)
    monkeypatch.setitem(ci_scope.JOB_COMMANDS, "lint", (".venv/bin/python -m flake8 --version",))

    decisions = _plan(root, ["local_operator/alpha.py"])
    assert decisions["lint"].whole_tree
    assert any("input to narrow" in note for note in decisions["lint"].notes)


# ---------------------------------------------------------------------------
# Plan shape, limits, determinism
# ---------------------------------------------------------------------------


def test_type_check_names_the_changed_files_and_their_dependents(tmp_path):
    """A file-list pyright reports only for the files it is GIVEN.

    So the list has to be the changed files plus their transitive reverse
    dependents — measured: with only the DEPENDENT listed, a changed signature in
    its dependency is still reported, while an error in an unlisted module is not.
    The bound and the wrapper survive narrowing, and the protocol-sync step (no
    file input to narrow) is kept and REPORTED rather than silently narrowed.
    """
    root = _fixture_repo(tmp_path)
    decision = _plan(root, ["local_operator/alpha.py"], jobs=("type-check",))["type-check"]

    assert not decision.whole_tree
    pyright_command = decision.commands[0]
    assert ci_scope.BOUNDED_WRAPPER_NAME in pyright_command
    assert "--timeout" in pyright_command
    assert "local_operator/alpha.py" in pyright_command
    assert "tests/unit/test_beta.py" in pyright_command, "a transitive dependent must be named"
    assert "tests/unit/test_filler_0.py" not in pyright_command, "an unrelated test must not be"
    assert (
        decision.commands[1] == ci_scope.JOB_COMMANDS["type-check"][1]
    ), "the protocol-sync step has no file input and must run unchanged"
    assert any("no file input to narrow" in note for note in decision.notes), decision.notes


def test_type_check_falls_back_when_the_list_would_pull_in_the_program(tmp_path):
    """Naming most of the program is the whole-tree command with extra steps."""
    root = _fixture_repo(tmp_path)
    monkey_target = ci_scope.SCOPE_MAX_CLOSURE_FRACTION
    try:
        ci_scope.SCOPE_MAX_CLOSURE_FRACTION = 0.0
        decision = _plan(root, ["local_operator/alpha.py"], jobs=("type-check",))["type-check"]
    finally:
        ci_scope.SCOPE_MAX_CLOSURE_FRACTION = monkey_target

    assert decision.whole_tree
    assert decision.commands == ci_scope.JOB_COMMANDS["type-check"]
    assert any("closure arm" in note for note in decision.notes), decision.notes


def test_the_local_typed_gate_keeps_its_bound_and_its_reaper():
    command = ci_scope.JOB_COMMANDS["type-check"][0]
    assert ci_scope.BOUNDED_WRAPPER_NAME in command
    assert "--timeout" in command


def test_the_forward_closure_is_what_a_file_list_run_would_analyze(tmp_path):
    root = _fixture_repo(tmp_path)
    graph = ci_scope.build_import_graph(root)

    closure = graph.forward_closure({"tests/unit/test_beta.py"})
    assert "local_operator/beta.py" in closure
    assert "local_operator/alpha.py" in closure, "the closure is transitive"
    assert "tests/unit/test_beta.py" not in closure, "the given files are not their own closure"


def test_every_scoped_job_is_a_job_with_local_commands():
    assert set(ci_scope.SCOPED_JOBS) <= set(ci_scope.JOB_COMMANDS)
    assert not set(ci_scope.SCOPED_JOBS) & set(ci_scope.UNSCOPED_JOBS)
    assert not set(ci_scope.SCOPED_JOBS) & set(ci_scope.LOCAL_EXCLUSIONS)
    assert set(ci_scope.SCOPE_MARKERS) == set(ci_scope.SCOPED_JOBS)
    for job, reason in ci_scope.UNSCOPED_JOBS.items():
        assert job in ci_scope.JOB_COMMANDS, f"{job} has no local command to leave whole-tree"
        assert reason.strip()


def test_the_plan_is_deterministic(tmp_path):
    root = _fixture_repo(tmp_path)
    first = ci_scope.scope_plan(["lint", "test", "tui-e2e"], ["local_operator/alpha.py"], root)
    second = ci_scope.scope_plan(["lint", "test", "tui-e2e"], ["local_operator/alpha.py"], root)

    assert first.commands() == second.commands()
    assert first.unnamed == second.unnamed
    assert [line for line in first.report() if "graph:" not in line] == [
        line for line in second.report() if "graph:" not in line
    ]


def test_the_report_states_what_a_narrowed_run_cannot_see(tmp_path):
    """A limit that is not printed is the quiet narrowing this design forbids."""
    root = _fixture_repo(tmp_path)
    _write(
        root,
        "local_operator/agents.py",
        """
        import importlib

        def load(name):
            return importlib.import_module(name)
        """,
    )
    plan = ci_scope.scope_plan(["test"], ["local_operator/alpha.py"], root)

    assert plan.unnamed and "local_operator/agents.py" in plan.unnamed[0]
    report = "\n".join(plan.report())
    assert "selection limit" in report
    # The label covers names AND directory scans, because both are sites the graph
    # cannot see (#1322 QA round 2 added the second).
    assert "load or read their target at run time" in report
    assert "scoped to" in report


def test_the_report_says_nothing_about_limits_when_no_test_selection_was_narrowed(tmp_path):
    root = _fixture_repo(tmp_path)
    _write(
        root,
        "local_operator/agents.py",
        """
        import importlib

        def load(name):
            return importlib.import_module(name)
        """,
    )
    # A selection the weight arm refuses: the graph IS built (so `unnamed` is
    # populated) but no test job ends up narrowed.
    durations = {f"tests/unit/test_filler_{index}.py": 1.0 for index in range(FILLERS)}
    durations.update(
        {
            "tests/unit/test_alpha.py": 1.0,
            "tests/unit/test_beta.py": 1.0,
            "tests/unit/test_gamma.py": 1.0,
            "tests/unit/tui/test_widget.py": 500.0,
        }
    )
    _write_durations(root, durations, 1.0)

    plan = ci_scope.scope_plan(["test"], ["local_operator/tui/widget.py"], root)
    assert plan.unnamed
    assert plan.decisions["test"].whole_tree
    assert "selection limit" not in "\n".join(plan.report())


def test_a_whole_tree_decision_reports_itself_as_whole_tree(tmp_path):
    root = _fixture_repo(tmp_path)
    line = _plan(root, ["pyproject.toml"])["test"].report()
    assert line.startswith("- `test`: whole tree")
    assert "pyproject.toml" in line


def test_run_jobs_with_no_commands_says_nothing_ran(tmp_path, capsys):
    """A narrowed-to-empty plan must not read as a passing gate."""
    rc = ci_scope.run_jobs(["lint"], tmp_path, {"lint": ()})
    assert rc == 0
    assert "nothing to run" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Files a gate reaches by PATH or by `-m`, not by an import
# ---------------------------------------------------------------------------
#
# `scripts/` is executed by the suite through `sys.executable` + a path, and
# several modules are spawned with `-m`, so an imports-only graph has no edge to
# the file a test actually runs. On this head that made `make check-changed`
# print `all selected gates passed` for a one-token change to
# `scripts/visual_gallery.py` whose CI `test` job was red: the narrowed `test`
# job selected NOTHING (review BLOCKER 1 on #1322). These tests are the shape
# that has to fail if the name edge is ever lost again — the first two synthetic,
# the third over the real tree so a rename cannot hide the instance.


def test_a_script_a_test_executes_by_path_is_an_input(tmp_path):
    """The executed target must be an input, spelled exactly as the suite does."""
    root = _fixture_repo(tmp_path)
    _write(root, "scripts/visual_gallery.py", "GALLERY = True\n")
    _write(
        root,
        "tests/unit/test_gallery.py",
        """
        import subprocess
        import sys
        from pathlib import Path

        # The real shape: `tests/unit/tui/test_visual_gallery.py` builds this path
        # from `__file__` and runs the script as a child process.
        REPO = Path(__file__).resolve().parents[2]

        def test_the_gallery_prints_its_inventory():
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "visual_gallery.py")],
                check=True,
            )
        """,
    )

    decision = _plan(root, ["scripts/visual_gallery.py"])["test"]

    assert not decision.whole_tree, decision.notes
    assert decision.targets == ("tests/unit/test_gallery.py",)
    assert any("named by path" in note for note in decision.notes), decision.notes


def test_a_module_a_test_runs_with_dash_m_is_an_input(tmp_path):
    """`python -m local_operator.x` records no import of `x` either."""
    root = _fixture_repo(tmp_path)
    _write(root, "local_operator/probe_worker.py", "WORKER = True\n")
    _write(
        root,
        "tests/unit/test_worker_spawn.py",
        """
        import subprocess
        import sys

        def test_it_runs_the_worker():
            subprocess.run([sys.executable, "-m", "local_operator.probe_worker"], check=True)
        """,
    )

    decision = _plan(root, ["local_operator/probe_worker.py"])["test"]

    assert not decision.whole_tree, decision.notes
    assert decision.targets == ("tests/unit/test_worker_spawn.py",)


@pytest.mark.slow
def test_a_repo_python_file_a_test_names_is_always_selected():
    """The same property over the REAL tree, so a rename cannot hide the case.

    `scripts/visual_gallery.py` → `tests/unit/tui/test_visual_gallery.py` is the
    instance review found; this asserts the class — every repo file a test names
    in a string constant, by path or by dotted name, selects that test when it
    changes. White-box (`build_import_graph` + `_select_tests`) because building
    the real graph once is the only affordable way to ask it ~100 times.
    """
    root = Path(__file__).resolve().parents[2]
    graph = ci_scope.build_import_graph(root)
    universe = ci_scope._test_universe(graph, "tests/unit")
    basenames = ci_scope._repo_basenames(sorted(graph.files))
    # Driven from the LITERALS each test carries, not from `graph.referrers`: the
    # referrer map is resolution OUTPUT, so a spelling the resolver cannot see is
    # invisible to it — which is exactly how the round-2 MAJOR survived a test
    # written to guard that class.
    by_target: dict[str, list[str]] = {}
    for test in universe:
        for literal in graph.literals_of(test):
            for target in ci_scope._name_target(root, literal, basenames):
                if target != test:
                    by_target.setdefault(target, []).append(test)

    assert by_target, "no test names any repo file, so this property proves nothing"

    offenders: list[tuple[str, str]] = []
    for target, tests in sorted(by_target.items()):
        selected = set(ci_scope._select_tests(graph, [target], universe)[0])
        offenders.extend((target, test) for test in sorted(tests) if test not in selected)

    assert not offenders, f"a test names this file and is not selected on change: {offenders[:5]}"

    # …and the same question for a file reached by a DIRECTORY SCAN rather than a
    # name. `tests/unit/tui/test_visual_gallery.py` globs `scripts/*.py` and asserts
    # an ordering invariant on each, so a one-token change to any of them must
    # select it — that is QA round 2's Q-1, and the fix it asks for.
    # BOTH live readers, because they spell the same directory differently: the
    # gallery builds it inline, the capture test holds it in a variable — and the
    # second was round 3's BLOCKER, printed but not armed, so CI went red while the
    # local run printed green.
    scanners = (
        "tests/unit/tui/test_visual_gallery.py",
        "tests/unit/tui/test_visual_capture.py",
    )
    # Directly under `scripts/`: the scans are `.glob("*.py")`, which is NOT recursive,
    # so a nested script like `scripts/cold_engage_site/sitecustomize.py` is
    # correctly NOT read by them.
    scanned = sorted(
        f
        for f in graph.files
        if f.startswith("scripts/") and f.endswith(".py") and "/" not in f[len("scripts/") :]
    )
    assert scanned, "no covered scripts/*.py to scan"
    for scanner in scanners:
        assert scanner in universe, f"{scanner} moved; update this guard"
        for target in scanned:
            selected = set(ci_scope._select_tests(graph, [target], universe)[0])
            assert scanner in selected, f"{scanner} reads {target} by glob and must be selected"


# ---------------------------------------------------------------------------
# The reporting contract of a local run
# ---------------------------------------------------------------------------


def test_narrow_replaces_every_scripted_input():
    """A command with the marker twice must not keep one whole-tree input.

    Every command in `JOB_COMMANDS` carries its scripted input once, so nothing
    is wrong today; the assertion is that a future one cannot leave a stray
    whole-tree input inside a command that reports itself as scoped (review NIT 1
    on #1322). The marker is matched as a whole token, so `.` inside
    `--pythonpath .` is not a marker.
    """
    narrowed = ci_scope._narrow(
        ".venv/bin/python -m pytest tests/unit tests/unit/x.py --pythonpath .",
        "tests/unit",
        ["tests/unit/alpha.py"],
    )

    assert narrowed == (
        ".venv/bin/python -m pytest tests/unit/alpha.py tests/unit/x.py --pythonpath ."
    )


def test_run_jobs_with_no_jobs_says_nothing_was_selected(tmp_path, capsys):
    """An empty selection is not the same sentence as a narrowed-to-empty job.

    They shared one branch, so a clean tree read `every selected job was narrowed
    to no file` about a decision that was never made (QA Q-1 on #1322).
    """
    rc = ci_scope.run_jobs([], tmp_path, {})

    assert rc == 0
    out = capsys.readouterr().out
    assert "no job was selected for this diff, so nothing ran" in out
    assert "nothing to run" not in out


def test_an_empty_diff_is_reported_as_empty_not_as_uncollected(tmp_path, capsys):
    """A clean tree used to be told BOTH that whole-tree commands would run and
    that nothing would: an empty diff and an uncollectable one shared a branch.

    The distinction is observable behaviour — the two cases print different
    things, and only the failure case promises a whole-tree run — so it is
    asserted against a real (empty) repository rather than a mock.
    """
    if shutil.which("git") is None:  # pragma: no cover - every gate host has git
        pytest.skip("git is not on PATH")
    root = _git_repo(tmp_path)

    assert ci_scope.main(["--root", str(root), "--since", "HEAD", "--run"]) == 0

    out = capsys.readouterr().out
    scope = out.split("### Local scope")[1]
    assert "the diff is empty" in scope
    assert "whole-tree command" not in scope.split("skipped locally")[0]
    assert "no job was selected for this diff, so nothing ran" in out


def _git_repo(root: Path) -> Path:
    """A committed, clean repository: what an EMPTY diff has to be observed in."""
    root.mkdir(parents=True, exist_ok=True)
    # A scratch HOME and no system/global gitconfig: this only needs a commit
    # identity, and the host's global config can carry a credential helper that
    # reaches the login keychain (`credential.helper = osxkeychain`).
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for args in (
        ["init", "-q", "-b", "main", "."],
        [
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=fixture",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=str(root), env=env, check=True, capture_output=True)
    return root


@pytest.mark.parametrize(
    "template",
    [
        "scripts/tool.py",  # as written against the repo root
        "./scripts/tool.py",  # what a `cd`-relative command line looks like
        "../scripts/tool.py",  # a test that walks up from its own directory
        "../../scripts/tool.py",
        "~/scripts/tool.py",  # a home-relative path
        "/scripts/tool.py",  # the segment an f-string leaves behind
        "{abs}/scripts/tool.py",  # the full absolute path
        "tool.py",  # a bare basename, resolved through the index
    ],
)
def test_every_spelling_of_a_repo_path_is_collected_and_resolved(tmp_path, template):
    """The SPELLINGS the rule admits, asserted as inputs rather than as output.

    `test_a_repo_python_file_a_test_names_is_always_selected` walks the graph's
    literals and the referrer map; a literal the COLLECTOR's regex rejects never
    reaches either, so the round-1 blocker could come back as
    `f"{ROOT}/scripts/target.py"` (whose constant is `/scripts/target.py`) with
    every test still green. This is the guard for that: the collector and the
    resolver must admit the same spellings, and each must land on the same file
    (#1322 round 2, MAJOR 1).
    """
    root = _fixture_repo(tmp_path)
    literal = template.format(abs=tmp_path.as_posix())
    files, _ = ci_scope._graph_files(root)
    basenames = ci_scope._repo_basenames(files)

    assert ci_scope._name_target(root, literal, basenames) == ("scripts/tool.py",)
    assert literal in ci_scope._references("probe.py", f"X = {literal!r}").literals


def test_a_path_built_by_an_f_string_is_an_input(tmp_path):
    """The live shape review round 2 reproduced.

    An f-string leaves its literal segments as separate constants, so
    `f"{ROOT}/scripts/target.py"` contributes `/scripts/target.py`. With that
    spelling dropped, a one-token change to the script planned `nothing to run`
    and the local gate went green with nothing disclosed.
    """
    root = _fixture_repo(tmp_path)
    _write(root, "scripts/target.py", "TARGET = True\n")
    _write(
        root,
        "tests/unit/test_runs_target.py",
        """
        import subprocess
        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[2]

        def test_it_runs_the_target():
            subprocess.run([sys.executable, f"{ROOT}/scripts/target.py"], check=True)
        """,
    )

    decision = _plan(root, ["scripts/target.py"])["test"]

    assert not decision.whole_tree, decision.notes
    assert decision.targets == ("tests/unit/test_runs_target.py",)


def test_a_conftest_that_names_a_file_selects_the_tests_it_governs(tmp_path):
    """A conftest is a namer pytest RUNS but not a universe member.

    Nothing imports it either (pytest loads it by path), so seeding on it selected
    NOTHING and a conftest that executes the named file broke its whole subtree
    silently. pytest's own scope is the answer: that conftest's subtree.
    """
    root = _fixture_repo(tmp_path)
    _write(root, "scripts/target.py", "TARGET = True\n")
    _write(
        root,
        "tests/unit/tui/conftest.py",
        """
        import subprocess
        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[3]

        def pytest_configure():
            subprocess.run([sys.executable, f"{ROOT}/scripts/target.py"], check=True)
        """,
    )

    decision = _plan(root, ["scripts/target.py"])["test"]

    assert not decision.whole_tree, decision.notes
    assert decision.targets == ("tests/unit/tui/test_widget.py",)
    assert any("conftest" in note for note in decision.notes), decision.notes


# ---------------------------------------------------------------------------
# Directory scans: files READ with no name to point at
# ---------------------------------------------------------------------------


def test_a_test_that_globs_a_covered_directory_reads_every_file_it_matches(tmp_path):
    """The QA round-2 Q-1 shape: a glob is a reader, not an absence of an edge.

    `tests/unit/tui/test_visual_gallery.py` iterates `(ROOT / "scripts").glob("*.py")`
    and asserts an ordering invariant on each file, so a one-token change to a
    script it reads must select it. Before the scan edge, that change selected
    NOTHING, the local run printed `all selected gates passed`, and CI's `test`
    job failed.
    """
    root = _fixture_repo(tmp_path)
    _write(
        root,
        "tests/unit/test_scan.py",
        """
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[2]

        def test_every_script_has_an_isolation_step():
            for path in (ROOT / "scripts").glob("*.py"):
                assert "TOOL" in path.read_text() or True
        """,
    )

    decision = _plan(root, ["scripts/tool.py"])["test"]

    assert not decision.whole_tree, decision.notes
    assert decision.targets == ("tests/unit/test_scan.py",)


def test_a_scan_this_graph_cannot_place_is_printed_not_silently_dropped(tmp_path):
    """A scan over a directory the graph cannot resolve is a printed limit.

    The directory here is a `tmp_path` value: no literal, no `__file__`, and a
    pattern that asks for `.py`. Treating every such scan as a reader of the whole
    program was measured at 654 of 724 tests selected for ANY change — the scoping
    win gone — so the honest answer is to say it out loud instead.
    """
    root = _fixture_repo(tmp_path)
    _write(
        root,
        "tests/unit/test_unplaced_scan.py",
        """
        from pathlib import Path

        def test_it_scans_a_runtime_directory(tmp_path):
            assert list(tmp_path.glob("*.py")) == []
        """,
    )

    plan = ci_scope.scope_plan(["test"], ["local_operator/alpha.py"], root)

    # ARMED, not merely printed: round 3 measured that the print-only policy was a
    # live false green, because the limit line's ellipsis hid the reader that
    # mattered. `unnamed` is now the LOOSE set; a `.py` scan lands here.
    assert any("directory scan" in site for site in plan.armed_scans), plan.armed_scans
    assert "armed reads" in "\n".join(plan.report())


# ---------------------------------------------------------------------------
# Scan placement: the SHAPES a receiver can take
# ---------------------------------------------------------------------------
#
# Round 3 measured ten shapes and found three that resolved to the WRONG
# directory silently (a nested literal receiver, a `__file__` + subdirectory, and
# a pattern with its own directory components), plus `os.walk(path)` never
# placing at all, plus a variable-held receiver that was printed but not armed —
# a live false green. One table, so the next narrowing fails here.

_SCAN_PROLOGUE = """
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
"""

#: id -> (lines after the prologue, extra files, changed path, must-select, must-not)
_SCAN_SHAPES: dict[str, tuple[str, dict[str, str], str, tuple[str, ...], tuple[str, ...]]] = {
    "literal": (
        'for path in (ROOT / "scripts").glob("*.py"):\n    path.read_text()\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "variable-held": (
        'SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"\n'
        'for path in SCRIPTS.glob("*.py"):\n    path.read_text()\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "nested-literal": (
        'for path in (ROOT / "scripts" / "diag").glob("*.py"):\n    path.read_text()\n',
        {"scripts/diag/probe.py": "PROBE = 1\n"},
        "scripts/diag/probe.py",
        ("tests/unit/test_scan_shape.py",),
        # The `scripts` ANCESTOR must not be the directory: that is the wrong edge.
        ("scripts/tool.py",),
    ),
    "pattern-directories": (
        'for path in (ROOT / "scripts").glob("*/*.py"):\n    path.read_text()\n',
        {"scripts/diag/probe.py": "PROBE = 1\n"},
        "scripts/diag/probe.py",
        ("tests/unit/test_scan_shape.py",),
        ("scripts/tool.py",),
    ),
    "rglob": (
        'for path in (ROOT / "scripts").rglob("*.py"):\n    path.read_text()\n',
        {"scripts/diag/probe.py": "PROBE = 1\n"},
        "scripts/diag/probe.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "file-parent": (
        'target = ROOT / "scripts" / "tool.py"\n'
        'for path in target.parent.glob("*.py"):\n    path.read_text()\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "own-directory": (
        'for path in Path(__file__).parent.glob("*.py"):\n    path.read_text()\n',
        {"tests/unit/helpers.py": "HELPER = 1\n"},
        "tests/unit/helpers.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "file-plus-subdir": (
        'for path in (Path(__file__).parent / "fixtures").glob("*.py"):\n    path.read_text()\n',
        {"tests/unit/fixtures/data.py": "DATA = 1\n"},
        "tests/unit/fixtures/data.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "walk": (
        'for _dirpath, _dirs, names in os.walk(ROOT / "scripts"):\n    list(names)\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    # The CALL forms QA round 3 measured: the `os.*`/`glob.*` spellings put their
    # directory in the arguments, and one of them puts it inside the pattern.
    "os-listdir": (
        'for name in os.listdir("scripts"):\n    (ROOT / "scripts" / name).read_text()\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "os-scandir": (
        'for entry in os.scandir("scripts"):\n    entry.name\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "os-walk-argument": (
        'for _dirpath, _dirs, names in os.walk("scripts"):\n    list(names)\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "glob-module": (
        'for name in glob.glob("scripts/*.py"):\n    Path(name).read_text()\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "glob-join": (
        'for name in glob.glob(os.path.join("scripts", "*.py")):\n    Path(name).read_text()\n',
        {},
        "scripts/tool.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
    "unplaceable-py": (
        'for path in Path(os.environ.get("SCAN_DIR", "tmp")).glob("*.py"):\n    path.read_text()\n',
        {},
        # Nothing places this directory, so it must be ARMED: any covered Python
        # file is treated as read by it (round 3 asked for exactly this).
        "local_operator/alpha.py",
        ("tests/unit/test_scan_shape.py",),
        (),
    ),
}


@pytest.mark.parametrize("shape", sorted(_SCAN_SHAPES))
def test_a_scan_shape_places_the_directory_it_reads(tmp_path, shape):
    """Each receiver shape the graph must place — and the two it must arm."""
    lines, extra, changed, must, must_not = _SCAN_SHAPES[shape]
    root = _fixture_repo(tmp_path)
    for rel, source in extra.items():
        _write(root, rel, source)

    _write(root, "tests/unit/test_scan_shape.py", _SCAN_PROLOGUE + lines + _CANNED_TEST)
    plan = ci_scope.scope_plan(["test"], [changed], root)
    decision = plan.decisions["test"]

    assert not decision.whole_tree, decision.notes
    selected = set(decision.targets)
    for expected in must:
        assert expected in selected, f"{shape}: {expected} not selected ({decision.notes})"
    for forbidden in must_not:
        # The ancestor-directory edge is a WRONG edge, not a conservative one: it
        # says this reader observes a file it never opens. Arming must not be what
        # makes this pass either, so it is checked through the reader's own plan.
        wrong = ci_scope.scope_plan(["test"], [forbidden], root).decisions["test"]
        assert (
            "tests/unit/test_scan_shape.py" not in wrong.targets
        ), f"{shape}: {forbidden} must not be an input of the reader"
    if shape == "unplaceable-py":
        assert plan.armed_scans, "an unplaceable .py scan must be armed"
    else:
        # …and a scan that IS placed must not also be armed: arming every shape
        # would hide a placement regression behind a program-wide read.
        assert not plan.armed_scans, f"{shape}: placed, so nothing may be armed"


_CANNED_TEST = """\

def test_the_scan_runs():
    assert True
"""


def test_an_unplaceable_python_scan_is_armed_and_says_so(tmp_path):
    """Armed is not the same as disclosed, and the report distinguishes them."""
    root = _fixture_repo(tmp_path)
    _write(
        root,
        "tests/unit/test_armed.py",
        _SCAN_PROLOGUE
        + 'for path in Path(os.environ.get("SCAN_DIR", "tmp")).glob("*.py"):\n'
        + "    path.read_text()\n"
        + _CANNED_TEST,
    )

    plan = ci_scope.scope_plan(["test"], ["local_operator/alpha.py"], root)
    report = "\n".join(plan.report())

    assert any("armed" in site for site in plan.armed_scans), plan.armed_scans
    assert "armed reads" in report
    # The fixture's `alpha.py` has real importers too, so the assertion is that the
    # armed reader is among the selected — not that it is alone.
    assert "tests/unit/test_armed.py" in plan.decisions["test"].targets


def test_the_limit_line_says_how_many_files_it_is_not_showing(tmp_path):
    """Round 3: the reader that mattered hid behind a bare ellipsis."""
    root = _fixture_repo(tmp_path)
    for index in range(8):
        _write(
            root,
            f"tests/unit/test_loose_{index}.py",
            "import os\nfrom pathlib import Path\n\n"
            f"def test_loose_{index}(tmp_path):\n"
            '    assert list(tmp_path.glob("*")) == []\n',
        )

    plan = ci_scope.scope_plan(["test"], ["local_operator/alpha.py"], root)
    report = "\n".join(plan.report())

    assert len(plan.unnamed) >= 8
    assert "more not shown" in report, report
