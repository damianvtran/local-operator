"""The ``agent`` tool's org ops: absent from the wire unless the Radient Team gate is open."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.harness.types import ToolContext
from local_operator.hub_org import eligibility
from local_operator.providers.radient_credentials import configured_radient_base_url
from local_operator.tools.agent_tool import build_agent_tool, execute_agent
from local_operator.tools.tool_docs import render_tool_doc

TENANT = "tenant-minerva"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".local-operator"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


def _seed(root: Path, orgs: list[dict[str, str]]) -> None:
    base = configured_radient_base_url(ConfigManager(root))

    def write(data: dict[str, Any]) -> None:
        data.update(base_url=base, checked_at=time.time(), orgs=orgs, account_id="acct")

    eligibility.update_record(root, write)


def _context(root: Path) -> ToolContext:
    return ToolContext(cwd=".", agent_registry=AgentRegistry(root))


def _tool(root: Path) -> Any:
    tool = build_agent_tool(_context(root))
    assert tool is not None
    return tool


def _run(root: Path, args: dict[str, Any]) -> str:
    async def go() -> Any:
        return await execute_agent("call", args, None, None, _context(root))

    result = asyncio.run(go())
    return " ".join(getattr(block, "text", "") for block in result.content)


def _ops(tool: Any) -> list[str]:
    return tool.parameters["properties"]["op"]["enum"]


def test_schema_omits_org_ops_when_signed_out(_isolated: Path) -> None:
    tool = _tool(_isolated)
    assert "org_list" not in _ops(tool) and "org_pull" not in _ops(tool)
    for prop in ("org", "kinds", "on_conflict"):
        assert prop not in tool.parameters["properties"]
    doc = render_tool_doc(tool)
    assert "org_list" not in doc and "Radient Team" not in doc


def test_schema_omits_org_ops_with_no_team_plan(_isolated: Path) -> None:
    _seed(_isolated, [])
    assert "org_pull" not in _ops(_tool(_isolated))


def test_schema_carries_org_ops_for_a_member(_isolated: Path) -> None:
    _seed(_isolated, [{"tenant_id": TENANT, "tenant_name": "Minerva", "role": "member"}])
    tool = _tool(_isolated)
    assert {"org_list", "org_pull"} <= set(_ops(tool))
    assert "on_conflict" in tool.parameters["properties"]
    assert "org_pull" in render_tool_doc(tool)


def test_executor_refuses_org_ops_when_ineligible(_isolated: Path) -> None:
    assert "Radient Team feature" in _run(_isolated, {"op": "org_list"})


def test_hub_wins_needs_confirmation(_isolated: Path) -> None:
    _seed(_isolated, [{"tenant_id": TENANT, "tenant_name": "Minerva", "role": "member"}])
    text = _run(_isolated, {"op": "org_pull", "on_conflict": "adopt"})
    assert "confirm_replace=true" in text
