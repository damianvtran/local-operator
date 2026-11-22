"""In-memory remove-all mutant; never edits the worktree."""

import ast
from pathlib import Path

import pytest

import tests.unit.tui.test_agent_facts as tests
from local_operator.tui.widgets import settings_view as view

source = (
    Path(view.__file__)
    .read_text()
    .replace(
        "head[:scope_index] + head[scope_index + 1 :]", "[fact for fact in head if fact != scope]"
    )
    .replace(
        "head[:marker_index] + head[marker_index + 1 :]",
        "[fact for fact in head if fact != marker]",
    )
)
node = next(
    n
    for n in ast.parse(source).body
    if isinstance(n, ast.FunctionDef) and n.name == "_paint_facts_lines"
)
node.name = "_remove_all_mutant"
exec(
    compile(ast.Module(body=[node], type_ignores=[]), "remove-all-mutant.py", "exec"), view.__dict__
)
tests._paint_facts_lines = view.__dict__[node.name]
raise SystemExit(
    pytest.main(
        [
            "-n",
            "0",
            "-q",
            "tests/unit/tui/test_agent_facts.py::test_spilling_duplicate_facts_projects_only_first_occurrences",
        ]
    )
)
