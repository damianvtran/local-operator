"""``scripts/affected_tests.py``: diff -> targeted tests, never the whole tree.

The helper replaced PR CI's full unit matrix with targeted local evidence, so the
properties that matter are the ones that keep that evidence honest: it selects
what a change names or directly imports, it reports what it cannot place
instead of silently widening to ``tests/unit``, it always carries the guard
list, and a large selection is kept and warned about rather than replaced.

Every case builds a small synthetic tree under ``tmp_path``. Git runs with the
system/global config disabled and prompts off (AGENTS.md keychain rule), so no
credential helper or ambient config can reach the result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import affected_tests

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "affected_tests.py"


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(affected_tests, "GUARD_TESTS", ("tests/unit/test_guard.py",))
    return _tree(
        tmp_path,
        {
            "local_operator/__init__.py": "",
            "local_operator/alpha.py": "X = 1\n",
            "local_operator/beta.py": "from local_operator import alpha\n",
            "local_operator/tui/__init__.py": "",
            "local_operator/tui/app.tcss": "Screen {}\n",
            "tests/unit/test_guard.py": "def test_g(): pass\n",
            "tests/unit/test_alpha.py": "def test_a(): pass\n",
            "tests/unit/test_alpha_more.py": "def test_a2(): pass\n",
            "tests/unit/test_alphabet.py": "def test_not_alpha(): pass\n",
            "tests/unit/pkg/conftest.py": "",
            "tests/unit/pkg/test_direct.py": "from local_operator.alpha import X\n",
            "tests/unit/pkg/test_transitive.py": "import local_operator.beta\n",
            "tests/unit/pkg/test_relative.py": "from .helpers import thing\n",
            "tests/unit/pkg/helpers.py": "thing = 1\n",
            "tests/unit/tui/test_widget.py": "def test_w(): pass\n",
        },
    )


def test_name_and_direct_import_select_but_transitive_does_not(tree: Path) -> None:
    sel = affected_tests.select(tree, ["local_operator/alpha.py"])
    assert "tests/unit/test_alpha.py" in sel.targets
    assert "tests/unit/test_alpha_more.py" in sel.targets
    assert (
        "tests/unit/test_alphabet.py" not in sel.targets
    ), "stem match must stop at a word boundary"
    assert "tests/unit/pkg/test_direct.py" in sel.targets
    # Depth 1 by design: the transitive closure through hub modules is the
    # whole-tree run this helper replaces.
    assert "tests/unit/pkg/test_transitive.py" not in sel.targets
    assert sel.reasons["tests/unit/pkg/test_direct.py"] == "imports local_operator.alpha"


def test_relative_import_in_a_test_resolves(tree: Path) -> None:
    sel = affected_tests.select(tree, ["tests/unit/pkg/helpers.py"])
    assert "tests/unit/pkg/test_relative.py" in sel.targets


def test_changed_and_deleted_test_files(tree: Path) -> None:
    sel = affected_tests.select(tree, ["tests/unit/tui/test_widget.py", "tests/unit/test_gone.py"])
    assert "tests/unit/tui/test_widget.py" in sel.targets
    assert "tests/unit/test_gone.py" not in sel.targets
    assert not sel.unplaced, "a deleted test is a placed change, not an unknown one"


def test_unplaced_path_takes_nearest_directory_never_the_whole_tree(tree: Path) -> None:
    sel = affected_tests.select(
        tree, ["local_operator/tui/app.tcss", "docs/notes.md", "tests/unit/conftest.py"]
    )
    assert "tests/unit/tui" in sel.targets
    assert "tests/unit" not in sel.targets and "tests" not in sel.targets
    assert sel.unplaced == ("docs/notes.md",)
    assert sel.wide == ("tests/unit/conftest.py",), "a root conftest is WIDE, not unplaced"


def test_conftest_selects_its_directory(tree: Path) -> None:
    sel = affected_tests.select(tree, ["tests/unit/pkg/conftest.py"])
    assert "tests/unit/pkg" in sel.targets


def test_guards_always_ride_along_unless_opted_out(tree: Path) -> None:
    assert "tests/unit/test_guard.py" in affected_tests.select(tree, ["docs/notes.md"]).targets
    assert affected_tests.select(tree, ["docs/notes.md"], include_guards=False).targets == ()


def test_large_selection_is_kept_and_warned(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(affected_tests, "WARN_SELECTION_FILES", 1)
    sel = affected_tests.select(tree, ["local_operator/alpha.py"])
    assert len(sel.targets) > 1, "the selection must be kept, not replaced"
    assert sel.warnings and "narrow it by" in sel.warnings[0]


def test_parse_counts_reads_the_final_summary() -> None:
    out = "..F\n=== 1 failed, 2 passed, 1 error in 3.20s ===\n"
    assert affected_tests.parse_counts(out) == {"failed": 1, "passed": 2, "error": 1}
    assert affected_tests.parse_counts("4 passed in 0.1s\n") == {"passed": 4}
    assert affected_tests.parse_counts("no summary\n") == {}


def test_evidence_markdown_shape(tree: Path) -> None:
    sel = affected_tests.select(tree, ["local_operator/alpha.py", "docs/notes.md"])
    md = affected_tests.evidence_markdown(
        base="a" * 40,
        head="b" * 40 + "+worktree",
        selection=sel,
        command=["pytest", "-q", "tests/unit/test_alpha.py"],
        rc=0,
        counts={"passed": 3},
        wall=1.25,
    )
    assert md.startswith("## Targeted test evidence\n")
    assert f"`{'a' * 12}..{'b' * 12}+worktree`" in md
    assert "**PASS** — 3 passed" in md and "1.2 s" in md
    assert "1 unplaced: `docs/notes.md`" in md
    assert "pytest -q tests/unit/test_alpha.py" in md
    # Q3 on #3592: explicit paths still name the commit they were selected at.
    explicit = affected_tests.evidence_markdown(
        base=affected_tests.EXPLICIT,
        head="c" * 40,
        selection=sel,
        command=["pytest"],
        rc=0,
        counts={},
        wall=0.0,
    )
    assert f"- range: explicit paths at `{'c' * 12}`" in explicit


def _env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "CMUX_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return env


def test_cli_since_runs_and_prints_evidence(tmp_path: Path) -> None:
    """The real path: git diff since a base, run through run_bounded, evidence."""
    root = _tree(
        tmp_path / "repo",
        {
            "local_operator/__init__.py": "",
            "local_operator/alpha.py": "X = 1\n",
            "tests/unit/test_alpha.py": (
                "from local_operator.alpha import X\n\ndef test_x():\n    assert X == 2\n"
            ),
            "tests/unit/test_other.py": "def test_never_selected():\n    raise AssertionError\n",
        },
    )
    env = _env(tmp_path)

    def git(*argv: str) -> None:
        subprocess.run(["git", *argv], cwd=root, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-qm", "base")
    (root / "local_operator/alpha.py").write_text("X = 2\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--since",
            "HEAD",
            "--no-guards",
            "--evidence",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "## Targeted test evidence" in result.stdout
    assert "**PASS** — 1 passed" in result.stdout
    assert "tests/unit/test_alpha.py" in result.stdout
    assert "test_other.py" not in result.stdout + result.stderr
    assert "- changed paths: 1" in result.stdout


def test_cli_guards_prints_the_list() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--guards"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert tuple(result.stdout.split()) == affected_tests.GUARD_TESTS


def test_data_as_text_selects_tests_that_name_the_file(tree: Path) -> None:
    """Escape class DATA-AS-TEXT: a test reads a seed/guide as data, imports nothing."""
    _tree(
        tree,
        {
            "local_operator/agent_seeds/coder.md": "# coder\n",
            "local_operator/guides/delivery/GUIDE.md": "# guide\n",
            "tests/unit/test_seeds.py": 'SEEDS = ROOT / "agent_seeds"\n',
            "tests/unit/test_guides.py": 'paths = ROOT.glob("guides/*/GUIDE.md")\n',
            "tests/unit/test_by_path.py": 'P = "local_operator/agent_seeds/coder.md"\n',
            "tests/unit/test_unrelated.py": 'X = "GUIDE.mdx agent_seedsx"\n',
            # Cited in a docstring/comment only: not a reader of the file.
            "tests/unit/test_cites.py": (
                '"""See local_operator/agent_seeds/coder.md and GUIDE.md."""\n'
                "# agent_seeds/coder.md\n"
            ),
        },
    )
    seeds = affected_tests.select(tree, ["local_operator/agent_seeds/coder.md"])
    assert "tests/unit/test_seeds.py" in seeds.targets
    assert "tests/unit/test_by_path.py" in seeds.targets
    guide = affected_tests.select(tree, ["local_operator/guides/delivery/GUIDE.md"])
    assert "tests/unit/test_guides.py" in guide.targets
    assert "tests/unit/test_unrelated.py" not in seeds.targets + guide.targets
    assert "tests/unit/test_cites.py" not in seeds.targets + guide.targets


def test_name_match_is_scoped_to_the_mirroring_directory(tree: Path) -> None:
    """R1/Q1 on #3592: a generic stem must not pull same-named tests tree-wide."""
    _tree(
        tree,
        {
            "local_operator/tunnels/__init__.py": "",
            "local_operator/tunnels/cli.py": "",
            "local_operator/team_bridge/__init__.py": "",
            "local_operator/team_bridge/store.py": "",
            "tests/unit/tunnels/test_cli.py": "",
            "tests/unit/tunnels/sub/test_cli_more.py": "",
            "tests/unit/test_tunnels_cli.py": "",
            "tests/unit/i18n/test_cli_catalogue.py": "",
            "tests/unit/i18n/test_tunnels_cli_copy.py": "",
            "tests/unit/test_cli.py": "",
            # Names the TOP-LEVEL cli.py by path: not a reader of tunnels/cli.py.
            "tests/unit/asks/test_wake_row.py": 'P = "local_operator/cli.py"\n',
            "tests/unit/team_bridge/test_store.py": "",
            "tests/unit/analytics/test_store.py": "",
            "tests/unit/test_tunnels.py": "",
            "tests/unit/tunnels/test_tunnels_init.py": "",
        },
    )
    cli = affected_tests.select(tree, ["local_operator/tunnels/cli.py"], include_guards=False)
    assert set(cli.targets) == {
        "tests/unit/tunnels/test_cli.py",
        "tests/unit/tunnels/sub/test_cli_more.py",
        "tests/unit/test_tunnels_cli.py",
        "tests/unit/i18n/test_tunnels_cli_copy.py",
    }, cli.reasons
    store = affected_tests.select(
        tree, ["local_operator/team_bridge/store.py"], include_guards=False
    )
    assert set(store.targets) == {"tests/unit/team_bridge/test_store.py"}, store.reasons
    # A top-level module mirrors tests/unit itself: only that directory's files.
    top = affected_tests.select(tree, ["local_operator/alpha.py"], include_guards=False)
    assert "tests/unit/test_alpha.py" in top.targets
    root_cli = affected_tests.select(tree, ["local_operator/cli.py"], include_guards=False)
    assert "tests/unit/test_cli.py" in root_cli.targets
    assert "tests/unit/i18n/test_cli_catalogue.py" not in root_cli.targets
    assert "tests/unit/asks/test_wake_row.py" in root_cli.targets, "full path still matches"
    # A package __init__ is tested beside the package and inside it.
    init = affected_tests.select(tree, ["local_operator/tunnels/__init__.py"], include_guards=False)
    assert {"tests/unit/test_tunnels.py", "tests/unit/tunnels/test_tunnels_init.py"} <= set(
        init.targets
    )


def test_dotted_module_name_in_a_literal_selects(tree: Path) -> None:
    _tree(tree, {"tests/unit/pkg/test_patch.py": 'T = "local_operator.alpha.X"\n'})
    sel = affected_tests.select(tree, ["local_operator/alpha.py"], include_guards=False)
    assert (
        sel.reasons["tests/unit/pkg/test_patch.py"] == "names local_operator/alpha.py in its source"
    )
    near = affected_tests.select(tree, ["local_operator/beta.py"], include_guards=False)
    assert "tests/unit/pkg/test_patch.py" not in near.targets


def test_parent_folder_needs_a_path_shaped_literal(tree: Path) -> None:
    """Q2 on #3592: the parent of `i18n/audit-history/hi/x.json` is `hi`."""
    _tree(
        tree,
        {
            "i18n/audit-history/hi/wire.settings.json": "{}\n",
            "tests/unit/i18n/test_audit_history.py": 'P = "i18n/audit-history/hi/"\n',
            "tests/unit/test_greets.py": 'MSG = "hi"\nOTHER = "say hi/bye"\n',
            "tests/unit/test_hi_dir.py": 'D = ROOT / "locale/hi"\n',
            # A slash command, not a path.
            "tests/unit/test_slash.py": 'CMD = "/hi"\n',
            # Same basename, another catalogue: not a reader of the ledger shard.
            "tests/unit/i18n/test_messages.py": 'P = root / "en" / "wire.settings.json"\n',
            "local_operator/guides/team-bridge/GUIDE.md": "# guide\n",
            "tests/unit/test_setup.py": 'G = ROOT / "guides" / "team-bridge"\nT = G / "GUIDE.md"\n',
        },
    )
    sel = affected_tests.select(
        tree, ["i18n/audit-history/hi/wire.settings.json"], include_guards=False
    )
    for spurious in (
        "tests/unit/test_greets.py",
        "tests/unit/test_slash.py",
        "tests/unit/i18n/test_messages.py",
    ):
        assert spurious not in sel.targets, sel.reasons
    guide = affected_tests.select(
        tree, ["local_operator/guides/team-bridge/GUIDE.md"], include_guards=False
    )
    assert "tests/unit/test_setup.py" in guide.targets, "a compound parent as a whole literal"
    _tree(tree, {"tests/unit/test_sibling.py": 'P = "/guides/browser/GUIDE.md"\n'})
    other = affected_tests.select(
        tree, ["local_operator/guides/delivery/GUIDE.md"], include_guards=False
    )
    assert "tests/unit/test_sibling.py" not in other.targets, "a sibling guide's path"
    # A whole `"GUIDE.md"` literal reads every guide, inside the mirror scope only.
    _tree(
        tree,
        {
            "tests/unit/guides/test_all_guides.py": 'BODY = base / "GUIDE.md"\n',
            "tests/unit/other/test_one_guide.py": 'BODY = base / "GUIDE.md"\n',
        },
    )
    every = affected_tests.select(
        tree, ["local_operator/guides/delivery/GUIDE.md"], include_guards=False
    )
    assert "tests/unit/guides/test_all_guides.py" in every.targets
    assert "tests/unit/other/test_one_guide.py" not in every.targets
    assert "tests/unit/i18n/test_audit_history.py" in sel.targets
    # A `/`-bounded segment inside a path literal is still a reader.
    assert "tests/unit/test_hi_dir.py" in sel.targets


def test_infra_paths_print_wide_and_never_select_the_tree(tree: Path) -> None:
    """Escape class INFRA: conftest/run_bounded/ci_scope/plugins reach every test."""
    sel = affected_tests.select(
        tree,
        [
            "conftest.py",
            "tests/unit/conftest.py",
            "scripts/run_bounded.py",
            "tests/unit/pkg/conftest.py",
        ],
    )
    assert set(sel.wide) == {
        "conftest.py",
        "tests/unit/conftest.py",
        "scripts/run_bounded.py",
        "tests/unit/pkg/conftest.py",
    }
    assert "tests/unit" not in sel.targets and "tests" not in sel.targets
    assert "tests/unit/pkg" in sel.targets, "a nested conftest still selects its own directory"
    line = affected_tests.wide_directive(sel.wide)
    assert line.startswith("WIDE: infra change")
    assert "gh workflow run full-suite.yml --ref" in line
    assert affected_tests.select(tree, ["local_operator/alpha.py"]).wide == ()
