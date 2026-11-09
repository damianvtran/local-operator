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


# -- agent-performable pulls/updates (hub_pull consent, 2026-11-09) -------------------
#
# The operator's request: agents must be able to pull from the org hub and update
# already-pulled items without sending non-technical users to a CLI. The linked
# arm applies available updates through the hub-sync merge (no LLM, conflicts
# reported, never forced); the safe class asks consent as ``hub_pull`` with a
# HOST-built plan; hub-wins adopt/reconcile stay ``sync`` + confirm_replace.

TEAM_ID = "22222222-2222-4222-8222-222222222222"


def _skill_body(root: Path, name: str) -> str:
    return (root / "skills" / name / "SKILL.md").read_text()


def test_org_pull_updates_linked_items_and_org_list_says_whether_current(
    _isolated: Path, member_hub
) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "v1\n")
    first = _run(_isolated, {"op": "org_pull", "kinds": "skill"})
    assert f"skill acme-style id={SKILL_A}: pulled" in first
    assert "[linked as acme-style, current]" in _run(
        _isolated, {"op": "org_list", "kinds": "skill"}
    )
    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "v2\n")
    listed = _run(_isolated, {"op": "org_list", "kinds": "skill"})
    assert "[linked as acme-style, update-available]" in listed
    assert "op='org_pull' updates them" in listed
    text = _run(_isolated, {"op": "org_pull", "kinds": "skill"})
    # Previously: "already-linked" and nothing applied (the Victor incident).
    assert "already-linked" not in text
    assert "skill acme-style: updated to revision skill-rev-1" in text, text
    assert "v2" in _skill_body(_isolated, "acme-style")
    assert "[linked as acme-style, current]" in _run(
        _isolated, {"op": "org_list", "kinds": "skill"}
    )


def test_org_pull_update_keeps_local_edits_and_never_forces(
    _isolated: Path, member_hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.hub_sync import service as svc
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "line one\n")
    _run(_isolated, {"op": "org_pull", "kinds": "skill"})
    path = _isolated / "skills" / "acme-style" / "SKILL.md"
    path.write_text(path.read_text().replace("line one", "my own edit"))
    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "hub rewrite\n")
    calls: list[dict[str, Any]] = []
    real = svc.apply_items

    def spy(ctx: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(ctx, **kwargs)

    monkeypatch.setattr(svc, "apply_items", spy)
    listed = _run(_isolated, {"op": "org_list", "kinds": "skill"})
    assert "update-available" in listed
    _run(_isolated, {"op": "org_pull", "kinds": "skill"})
    # The update-all door: prefer/replace are ignored there, and no LLM merge.
    (call,) = calls
    assert call["all_available"] is True and call["allow_llm"] is False
    assert call.get("replace") is None and call.get("prefer", "none") == "none"
    assert "my own edit" in path.read_text()


def test_needs_review_is_reported_not_forced() -> None:
    from local_operator.hub_org.consent import PullPlan, apply_linked
    from local_operator.hub_sync.report import ApplyReport, ItemMergeReport

    plan = PullPlan(tenant=TENANT, org_name="Minerva", kinds=("agent",))
    plan.checks = [type("C", (), {"kind": "agent", "hub_id": "h", "remote_revision": '"r2"'})()]
    report = ApplyReport(
        (ItemMergeReport("agent", "id", "writer", "h", "needs-review", message="instructions"),)
    )
    import local_operator.hub_sync.service as svc

    original = svc.apply_items, svc.sync_context
    svc.apply_items = lambda _ctx, **_k: report  # type: ignore[assignment]
    svc.sync_context = lambda _m: type("Ctx", (), {})()  # type: ignore[assignment]
    try:
        lines = apply_linked(object(), plan, client=None)
    finally:
        svc.apply_items, svc.sync_context = original
    assert lines == [
        "agent writer: needs-review (local edits conflict with the hub; kept, not forced)"
        " (instructions)"
    ]


class _Scoped:
    """A conversation-scoped registry: the shape that makes account writes need consent."""

    def __init__(self, account: AgentRegistry) -> None:
        self.global_registry = account
        self.session_registry = object()
        self.config_dir = account.config_dir

    def __getattr__(self, name: str) -> Any:
        return getattr(self.global_registry, name)


def _scoped_run(root: Path, args: dict[str, Any], decide) -> tuple[Any, list[Any]]:
    seen: list[Any] = []

    async def authorize(intent: Any) -> Any:
        seen.append(intent)
        return decide(intent)

    context = ToolContext(
        cwd=".",
        agent_registry=_Scoped(AgentRegistry(root)),
        authorize_durable_write=authorize,
    )
    async def go() -> Any:
        return await execute_agent("call", args, None, None, context)

    return asyncio.run(go()), seen


def test_safe_pull_asks_as_hub_pull_with_a_host_plan_and_queues(
    _isolated: Path, member_hub
) -> None:
    from local_operator.harness.types import QueuedHubPull
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "x\n")
    member_hub.teams[TEAM_ID] = {
        "id": TEAM_ID,
        "tenant_id": TENANT,
        "name": "lopdev",
        "description": "org",
        "manager": "manager",
        "members": [],
        "instructions": "Org brief.",
        "project": "",
    }
    result, seen = _scoped_run(
        _isolated,
        {"op": "org_pull"},
        lambda intent: QueuedHubPull(ask_id="a-9", tenant=TENANT),
    )
    text = " ".join(block.text for block in result.content)
    assert not result.is_error and result.details["status"] == "ask_queued"
    assert "ask a-9" in text and "End your turn" in text and "`lop " not in text
    (intent,) = seen
    assert intent.operation == "hub_pull" and intent.plan["tenant"] == TENANT
    assert {(r["kind"], r["name"]) for r in intent.plan["new"]} == {
        ("skill", "acme-style"),
        ("team", "lopdev"),
    }
    # Nothing was written before the answer.
    assert not (_isolated / "skills" / "acme-style").exists()


@pytest.mark.parametrize("reason", [None, "declined"])
def test_unconsented_safe_pull_writes_nothing_and_names_no_cli(
    _isolated: Path, member_hub, reason
) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "x\n")
    result, _ = _scoped_run(_isolated, {"op": "org_pull", "kinds": "skill"}, lambda _i: None)
    text = " ".join(block.text for block in result.content)
    assert result.is_error and "`lop " not in text and "lop agents sync" not in text
    assert not (_isolated / "skills" / "acme-style").exists()


def test_granted_safe_pull_applies(_isolated: Path, member_hub) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "x\n")
    result, seen = _scoped_run(
        _isolated, {"op": "org_pull", "kinds": "skill"}, lambda _i: "ask:a-1"
    )
    assert not result.is_error and seen[0].operation == "hub_pull"
    assert (_isolated / "skills" / "acme-style" / "SKILL.md").is_file()


def test_hub_wins_adopt_stays_a_sync_intent_not_hub_pull(_isolated: Path, member_hub) -> None:
    from tests.unit.hub_org.test_pull import _skill_doc

    member_hub.skills[SKILL_A] = _skill_doc("acme-style", "x\n")
    result, seen = _scoped_run(
        _isolated,
        {"op": "org_pull", "kinds": "skill", "on_conflict": "adopt", "confirm_replace": True},
        lambda _i: None,
    )
    assert result.is_error and [i.operation for i in seen] == ["sync"]
    assert seen[0].plan is None


def test_cross_org_pull_is_refused(_isolated: Path, member_hub) -> None:
    result, seen = _scoped_run(
        _isolated, {"op": "org_pull", "org": "tenant-elsewhere"}, lambda _i: "ask:a-1"
    )
    assert result.is_error and seen == []
    assert "no eligible organization" in " ".join(b.text for b in result.content)
