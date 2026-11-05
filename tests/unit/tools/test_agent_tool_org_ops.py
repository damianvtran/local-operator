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


# -- U2/Q1: org_pull resolves name= exactly as the CLI does --------------------------------


@pytest.fixture
def member_hub(_isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from tests.unit.hub_org.test_pull import FakeOrgHub

    _seed(_isolated, [{"tenant_id": TENANT, "tenant_name": "Minerva", "role": "member"}])
    hub = FakeOrgHub(configured_radient_base_url(ConfigManager(_isolated)))
    monkeypatch.setattr(
        "local_operator.hub_org.client.build_org_client",
        lambda _m: type("B", (), {"client": hub})(),
    )
    monkeypatch.setattr(
        "local_operator.hub_sync.service.build_clients_sync",
        lambda _m: ((lambda _t: hub), "ok"),
    )
    monkeypatch.setattr(
        "local_operator.hub_org.eligibility.eligible_orgs",
        lambda _cfg: [eligibility.OrgRef(TENANT, "Minerva", "member")],
    )
    return hub


SKILL_A = "33333333-3333-4333-8333-333333333333"
SKILL_B = "55555555-5555-4555-8555-555555555555"


def test_org_pull_resolves_a_unique_prefix_and_names_the_id(_isolated: Path, member_hub) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "x\n")
    member_hub.skills[SKILL_B] = _skill_doc("support-triage", "y\n")
    text = _run(_isolated, {"op": "org_pull", "kinds": "skill", "name": "acme-sty"})
    assert f"resolved 'acme-sty' → skill acme-style (id {SKILL_A})" in text
    assert f"skill acme-style id={SKILL_A}: pulled" in text
    assert "support-triage" not in text
    assert (_isolated / "skills" / "acme-style" / "SKILL.md").is_file()


def test_org_pull_refuses_an_ambiguous_prefix_with_candidates(_isolated: Path, member_hub) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("support-triage", "x\n")
    member_hub.skills[SKILL_B] = _skill_doc("support-writer", "y\n")
    text = _run(_isolated, {"op": "org_pull", "name": "support"})
    assert "ambiguous" in text
    assert f"skill support-triage (id {SKILL_A})" in text
    assert f"skill support-writer (id {SKILL_B})" in text
    assert not (_isolated / "skills").exists()


def test_org_pull_exact_name_and_id_still_resolve(_isolated: Path, member_hub) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("support", "x\n")
    member_hub.skills[SKILL_B] = _skill_doc("support-writer", "y\n")
    # Exact name beats the prefix it also is.
    assert f"skill support id={SKILL_A}: pulled" in _run(
        _isolated, {"op": "org_pull", "name": "support"}
    )
    assert f"skill support-writer id={SKILL_B}: pulled" in _run(
        _isolated, {"op": "org_pull", "name": SKILL_B}
    )
    assert "matches 'zzz'" in _run(_isolated, {"op": "org_pull", "name": "zzz"})
