"""The `bridge` tool must be offered whenever a session actually has a bridge.

WHY THIS FILE EXISTS. Until 0.84.2 `session_factory._prepare` resolved the
bridge runtime ~170 lines AFTER building the tool table. `build_bridge_tool` is
createIf: it returns ``None`` unless ``context.bridge_runtime`` is present, so
the tool was dropped from every session's table while the runtime itself was
still attached to the Session. Nothing observed the difference from outside —
enrollment committed, the gates passed, the executor would have worked — and the
model simply had no ``bridge`` tool to call. The fix moved the resolution above
``create_tools`` and handed the same value to the ToolContext.

A test that only asserted "``build_bridge_tool`` returns a tool for a context
that has a runtime" would have passed throughout the broken period, because it
never builds a real context. These tests therefore pin the ORDER in the factory
as well as the createIf semantics.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from local_operator.team_bridge.tool import build_bridge_tool

REPO_ROOT = Path(__file__).resolve().parents[3]
FACTORY = REPO_ROOT / "local_operator" / "session_factory.py"


def _prepare_body() -> list[ast.stmt]:
    tree = ast.parse(FACTORY.read_text(encoding="utf-8"), filename=str(FACTORY))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_prepare":
            return node.body
    raise AssertionError("session_factory._prepare not found")


def _first_index(nodes: list[ast.stmt], predicate) -> int:
    for index, node in enumerate(nodes):
        if predicate(node):
            return index
    return -1


def _assigns_bridge_runtime(node: ast.stmt) -> bool:
    """`bridge_runtime = bridge_runtime_for_session(...)` as a direct statement."""

    if not isinstance(node, ast.Assign):
        return False
    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
    value = node.value
    return (
        "bridge_runtime" in targets
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "bridge_runtime_for_session"
    )


def _builds_tool_table(node: ast.stmt) -> bool:
    """`tools = create_tools(tool_context)` as a direct statement."""

    if not isinstance(node, ast.Assign):
        return False
    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
    value = node.value
    return (
        "tools" in targets
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "create_tools"
    )


def test_runtime_is_resolved_before_the_tool_table_is_built() -> None:
    """The regression itself: createIf reads presence, so order is the contract."""

    body = _prepare_body()
    runtime_at = _first_index(body, _assigns_bridge_runtime)
    tools_at = _first_index(body, _builds_tool_table)

    assert runtime_at != -1, "the bridge runtime is no longer resolved in _prepare"
    assert tools_at != -1, "the tool table is no longer built in _prepare"
    assert runtime_at < tools_at, (
        "bridge_runtime must be resolved BEFORE tools = create_tools(tool_context); "
        "building the table first drops the createIf `bridge` tool from every "
        "session that has a bridge (the 0.84.1 defect)"
    )


def test_the_same_runtime_is_installed_on_the_tool_context() -> None:
    """One resolution, so the createIf inspection and the executor agree."""

    body = _prepare_body()
    assignments = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "bridge_runtime"
            and isinstance(t.value, ast.Name)
            and t.value.id == "tool_context"
            for t in node.targets
        )
    ]
    assert assignments, (
        "tool_context must carry bridge_runtime before create_tools runs: that "
        "context is what decides whether the `bridge` tool exists"
    )


def test_create_if_needs_presence_only() -> None:
    """The semantics the ordering depends on: presence, not truthiness of a gate."""

    assert build_bridge_tool(SimpleNamespace(bridge_runtime=None)) is None

    offered = build_bridge_tool(SimpleNamespace(bridge_runtime=object()))
    assert offered is not None
    assert offered.name == "bridge"
