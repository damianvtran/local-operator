"""Tests for the managed-variables surface (list_variables / read_variable).

Verifies the token-conservation contract: listing returns NAMES only (never
values), reading returns one value live, unknown names error cleanly, and
oversize values are elided rather than dumped into context.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools.builtin import (
    build_list_variables_tool,
    build_read_variable_tool,
)
from local_operator.variables import VariableStore


def _ctx(config: dict[str, str] | None = None) -> ToolContext:
    return ToolContext(
        cwd="/tmp",
        variables=VariableStore(
            cwd="/tmp",
            config_values=config or {},
            env={"PROJECT_KEY": "project-value", "SECRET": "shh", "EMPTY": ""},
        ),
    )


def _result(executor, tool_call_id: str, args: dict, ctx: ToolContext):
    return asyncio.run(executor(tool_call_id, args, context=ctx))


def test_list_variables_returns_names_only() -> None:
    tool = build_list_variables_tool()
    result = _result(tool.execute, "t1", {}, _ctx())
    assert not result.is_error
    assert "PROJECT_KEY" in result.text
    assert "SECRET" in result.text
    # Names, never values.
    assert "project-value" not in result.text
    assert "shh" not in result.text


def test_list_variables_includes_config_override() -> None:
    tool = build_list_variables_tool()
    result = _result(tool.execute, "t1", {}, _ctx({"MY_CONFIG": "cfg"}))
    assert "MY_CONFIG" in result.text


def test_read_variable_env_value() -> None:
    tool = build_read_variable_tool()
    result = _result(tool.execute, "t1", {"name": "PROJECT_KEY"}, _ctx())
    assert not result.is_error
    assert result.text == "project-value"


def test_read_variable_config_overrides_env() -> None:
    # Config wins over env for the same name.
    tool = build_read_variable_tool()
    result = _result(
        tool.execute, "t1", {"name": "PROJECT_KEY"}, _ctx({"PROJECT_KEY": "from-config"})
    )
    assert result.text == "from-config"


def test_read_variable_unknown_is_error() -> None:
    tool = build_read_variable_tool()
    result = _result(tool.execute, "t1", {"name": "NO_SUCH_KEY"}, _ctx())
    assert result.is_error
    assert "unknown variable" in result.text


def test_read_variable_requires_name() -> None:
    tool = build_read_variable_tool()
    result = _result(tool.execute, "t1", {"name": "  "}, _ctx())
    assert result.is_error


def test_oversize_value_is_elided() -> None:
    tool = build_read_variable_tool()
    ctx = _ctx({"BIG": "x" * 10_000})
    result = _result(tool.execute, "t1", {"name": "BIG"}, ctx)
    assert not result.is_error
    # Not the full value: capped and marked.
    assert "x" * 10_000 not in result.text
    assert "elided" in result.text


def test_variable_store_falls_back_to_process_env() -> None:
    os.environ["LO_TOOL_TEST_VAR"] = "proc"
    store = VariableStore(cwd="/tmp")
    try:
        assert "LO_TOOL_TEST_VAR" in store.names()
        assert store.read("LO_TOOL_TEST_VAR") == "proc"
    finally:
        os.environ.pop("LO_TOOL_TEST_VAR", None)


def test_variable_store_project_file(tmp_path) -> None:
    (tmp_path / ".local-operator.env").write_text('# comment\nPROJ=alpha\nQUOTED="beta"\n')
    store = VariableStore(cwd=str(tmp_path), env={"OTHER": "x"})
    assert store.read("PROJ") == "alpha"
    assert store.read("QUOTED") == "beta"
    # Environment still resolves for non-project keys.
    assert store.read("OTHER") == "x"


def test_variable_store_missing_raises_keyerror() -> None:
    store = VariableStore(cwd="/tmp", env={})
    with pytest.raises(KeyError):
        store.read("MISSING")
