"""Tests for the managed-variables surface (list_variables / read_variable).

Verifies the token-conservation contract: listing returns NAMES only (never
values), reading returns one value live, unknown names error cleanly, and
oversize values are elided rather than dumped into context.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

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
            env={
                "LOCAL_OPERATOR_ENV_COLOR": "project-value",
                "SECRET": "shh",  # must be denied even though env-shaped
                "OPENROUTER_API_KEY": "not-exposed",  # non-prefixed env secret
                "EMPTY": "",
            },
        ),
    )


def _result(executor, tool_call_id: str, args: dict[str, Any], ctx: ToolContext):
    return asyncio.run(executor(tool_call_id, args, context=ctx))


def test_list_variables_returns_names_only() -> None:
    tool = build_list_variables_tool()
    result = _result(tool.execute, "t1", {}, _ctx())
    assert not result.is_error
    assert "LOCAL_OPERATOR_ENV_COLOR" in result.text
    # Names, never values.
    assert "project-value" not in result.text


def test_secret_shaped_and_non_prefixed_env_are_denied() -> None:
    """Security contract: secret-pattern names and non-opted-in env vars are
    never surfaced, so a model cannot enumerate or exfiltrate credentials."""
    tool = build_list_variables_tool()
    result = _result(tool.execute, "t1", {}, _ctx())
    # Secret-shaped name from the project/env source is not listed.
    assert "SECRET" not in result.text
    # Non-preferred env var (incl. a real-looking API key) is not listed.
    assert "OPENROUTER_API_KEY" not in result.text
    # Reading one is an error, never the value.
    denied = _result(tool.execute, "t2", {"name": "SECRET"}, _ctx())
    assert denied.is_error
    assert "shh" not in denied.text


def test_non_prefixed_env_value_not_readable() -> None:
    tool = build_read_variable_tool()
    result = _result(tool.execute, "t1", {"name": "OPENROUTER_API_KEY"}, _ctx())
    assert result.is_error  # unknown: not opted in


def test_list_variables_includes_config_override() -> None:
    tool = build_list_variables_tool()
    result = _result(tool.execute, "t1", {}, _ctx({"MY_CONFIG": "cfg"}))
    assert "MY_CONFIG" in result.text


def test_read_variable_env_value() -> None:
    tool = build_read_variable_tool()
    result = _result(tool.execute, "t1", {"name": "LOCAL_OPERATOR_ENV_COLOR"}, _ctx())
    assert not result.is_error
    assert result.text == "project-value"


def test_read_variable_config_overrides_env() -> None:
    # Config wins over env for the same opted-in name.
    tool = build_read_variable_tool()
    result = _result(
        tool.execute,
        "t1",
        {"name": "LOCAL_OPERATOR_ENV_COLOR"},
        _ctx({"LOCAL_OPERATOR_ENV_COLOR": "from-config"}),
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
    assert "10000 chars total" in result.text


def test_variable_store_falls_back_to_process_env() -> None:
    os.environ["LO_TOOL_TEST_VAR"] = "proc"
    os.environ["LOCAL_OPERATOR_TOOL_TEST_VAR"] = "proc"
    store = VariableStore(cwd="/tmp")
    try:
        assert "LOCAL_OPERATOR_TOOL_TEST_VAR" in store.names()
        assert store.read("LOCAL_OPERATOR_TOOL_TEST_VAR") == "proc"
    finally:
        os.environ.pop("LOCAL_OPERATOR_TOOL_TEST_VAR", None)


def test_variable_store_project_file(tmp_path) -> None:
    (tmp_path / ".local-operator.env").write_text('# comment\nPROJ=alpha\nQUOTED="beta"\n')
    store = VariableStore(cwd=str(tmp_path), env={"LOCAL_OPERATOR_OTHER": "x"})
    assert store.read("PROJ") == "alpha"
    assert store.read("QUOTED") == "beta"
    # Environment resolves for opted-in keys.
    assert store.read("LOCAL_OPERATOR_OTHER") == "x"


def test_variable_store_missing_raises_keyerror() -> None:
    store = VariableStore(cwd="/tmp", env={})
    with pytest.raises(KeyError):
        store.read("MISSING")


def test_session_credential_is_listed_by_name_and_never_readable() -> None:
    """The operator hands the process a secret the agent must USE and must
    never READ: the name is advertised, ``read`` still refuses it."""
    store = VariableStore(cwd="/tmp", env={})
    result = store.store_credential("github token", "ghp_secret_value_1", "command")
    assert result.ok is True
    assert result.credential is not None
    assert result.credential.key == "GITHUB_TOKEN"
    assert store.credential_names() == ["GITHUB_TOKEN"]
    assert "GITHUB_TOKEN" not in store.names()
    assert store.get("GITHUB_TOKEN") is None
    with pytest.raises(KeyError):
        store.read("GITHUB_TOKEN")
    assert store.credential_env() == {"GITHUB_TOKEN": "ghp_secret_value_1"}


def test_session_credential_replaces_and_forgets() -> None:
    store = VariableStore(cwd="/tmp", env={})
    first = store.store_credential("API_KEY", "first-value-xx", "command")
    second = store.store_credential("API_KEY", "second-value-yy", "ask")
    assert first.replaced is False
    assert second.replaced is True
    assert store.credential_env()["API_KEY"] == "second-value-yy"
    assert store.list_credentials()[0].source == "ask"
    assert store.forget_credential("api-key") is True
    assert store.credential_names() == []
    assert store.forget_credential("API_KEY") is False


def test_session_credential_empty_value_is_refused() -> None:
    store = VariableStore(cwd="/tmp", env={})
    result = store.store_credential("API_KEY", "   \n", "command")
    assert result.ok is False
    assert result.reason == "empty-value"
    assert store.credential_names() == []


def test_parse_credential_command_verbs_cannot_collide_with_a_key() -> None:
    from local_operator.variables import parse_credential_command

    assert parse_credential_command("").action == "list"
    stored = parse_credential_command("github token")
    assert stored.action == "store" and stored.key == "GITHUB_TOKEN"
    forgotten = parse_credential_command("--forget github-token")
    assert forgotten.action == "forget" and forgotten.key == "GITHUB_TOKEN"
    assert parse_credential_command("--forget-all").action == "forget-all"
    assert parse_credential_command("--forget").action == "error"
    assert parse_credential_command("--nope").action == "error"
    assert parse_credential_command("!!!").action == "error"


def test_redact_replaces_longest_secret_first() -> None:
    from local_operator.variables import redact_secret_values

    text = redact_secret_values("xxxyyy and xxx", {"A": "xxx", "B": "xxxyyy"})
    assert text == "[redacted] and [redacted]"
    assert "xxx" not in text


@pytest.mark.asyncio
async def test_bash_injects_session_credentials_and_redacts_them_from_output() -> None:
    """The agent uses a secret by letting bash inherit it; echoing it back
    must not put the bytes in the tool result."""
    from local_operator.tools.builtin import execute_bash

    store = VariableStore(cwd="/tmp", env={})
    store.store_credential("LO_TEST_SECRET", "super-secret-value-xyz", "command")
    ctx = ToolContext(cwd="/tmp", variables=store)
    result = await execute_bash(
        "bash-1",
        {"command": "printf '%s' \"$LO_TEST_SECRET\""},
        None,
        None,
        ctx,
    )
    assert not result.is_error, result.text
    assert "super-secret-value-xyz" not in result.text
    assert "[redacted]" in result.text
