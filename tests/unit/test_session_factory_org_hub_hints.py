"""The org Agent Hub is advertised ONLY to Radient Team members.

Operator constraint: the private hub "shouldn't be advertised if they aren't
logged in with Radient or their Radient account is not part of a team". So the
three states are pinned through the REAL boot path (``create_session`` ->
knowledge hooks -> rendered block) with the gate record written by the real
``refresh`` (memberships stubbed — no network):

- signed out          -> no ``org-hub`` guide, no recommendation, no resolver;
- signed in, no team  -> the same;
- team member         -> guide present and the trigger fires on org asks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from local_operator.compaction.tokens import count_text_tokens
from local_operator.hub_org import eligibility as el
from local_operator.session.session import Session
from local_operator.session_factory import (
    _ORG_HUB_RULE,
    _KnowledgeHooks,
    _org_hub_recommendation,
    _select_knowledge_block,
)

TEAM = [
    {
        "tenant_id": "tnt_synthetic",
        "tenant_name": "Acme",
        "role": "member",
        "status": "active",
        "plan": {"status": "active", "seats": 3},
    }
]
NO_PLAN = [{**TEAM[0], "plan": {"status": "canceled"}}]

ORG_ASKS = [
    "pull our team's skills",
    "update skills",
    "pull the latest",
    "get our org's teams",
    "install the company's agents",
    "can you update all agents",
    "pull our shared skills from the hub",
]
UNRELATED = [
    "how do agents work",
    "update the README",
    "summarise this CSV",
    "pull request status on CI",
    "write a skill for parsing PDFs",
]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("an org-hub gate or trigger made a network call")

    monkeypatch.setattr(httpx.Client, "send", _refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", _refuse)
    el.reset_memo()


def _hooks(available: bool) -> _KnowledgeHooks:
    hooks = _KnowledgeHooks()
    hooks.org_hub_available = available
    return hooks


@pytest.mark.parametrize("query", ORG_ASKS)
def test_trigger_fires_for_a_member(query: str) -> None:
    hit = _org_hub_recommendation(query, _hooks(True))
    assert hit is not None and hit.resource_url == "guide://org-hub", query


@pytest.mark.parametrize("query", ORG_ASKS)
def test_trigger_never_fires_for_a_non_member(query: str) -> None:
    assert _org_hub_recommendation(query, _hooks(False)) is None


@pytest.mark.parametrize("query", UNRELATED)
def test_trigger_stays_quiet_on_unrelated_turns(query: str) -> None:
    assert _org_hub_recommendation(query, _hooks(True)) is None, query


def test_rule_line_is_one_short_line() -> None:
    assert "\n" not in _ORG_HUB_RULE
    assert count_text_tokens(_ORG_HUB_RULE) <= 30


async def _boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memberships: list[dict[str, Any]] | None
) -> tuple[Session, Any]:
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.session_factory import create_session

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_RUNTIME_ADOPT_SESSION", raising=False)
    monkeypatch.delenv("LOP_RUNTIME_DEFER_MATERIALISE", raising=False)
    config_dir = tmp_path / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    config = ConfigManager(config_dir)
    if memberships is not None:
        # Signed in: the real refresh writes the gate record from the stubbed
        # ``/me/memberships`` answer (``None`` = signed out, no record at all).
        monkeypatch.setattr(el, "_fetch_memberships", lambda *_: ("acc_syn", memberships))
        el.refresh(config_dir, boundary="test", force=True)
    args = argparse.Namespace(
        hosting="test", model="test", agent_name=None, agent_id=None, yolo=True, train=False
    )
    session = cast(Session, await create_session(args, config, AgentRegistry(config_dir)))
    hooks = session._knowledge_hooks
    assert hooks is not None
    return session, hooks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "memberships"), [("signed-out", None), ("no-team", NO_PLAN), ("no-orgs", [])]
)
async def test_ineligible_sessions_carry_no_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, memberships: Any
) -> None:
    session, hooks = await _boot(tmp_path, monkeypatch, memberships)
    try:
        assert "org-hub" not in hooks.guides_by_name, state
        assert hooks.org_hub_available is False
        for query in ORG_ASKS:
            block = await _select_knowledge_block(hooks, query, task_id=f"{state}-{query}")
            assert "org-hub" not in block, (state, query, block)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_team_member_gets_guide_and_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, hooks = await _boot(tmp_path, monkeypatch, TEAM)
    try:
        assert "org-hub" in hooks.guides_by_name
        assert hooks.org_hub_available is True
        for query in ("pull our team's skills", "update skills", "pull the latest"):
            block = await _select_knowledge_block(hooks, query, task_id=query)
            # Reached exactly once: either the semantic index listed the guide
            # (``- org-hub:``) and the trigger's line deduped against it, or
            # the trigger supplied the ``guide://org-hub`` line itself.
            listed = block.count("- org-hub:")
            recommended = block.count("- guide://org-hub")
            assert listed + recommended == 1, (query, block)
        quiet = await _select_knowledge_block(hooks, "summarise this CSV", task_id="q")
        assert "org-hub" not in quiet
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_trigger_line_reaches_the_block_when_the_index_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no semantic pick, the deterministic line alone carries the guide."""

    session, hooks = await _boot(tmp_path, monkeypatch, TEAM)
    try:
        hooks.index = None
        block = await _select_knowledge_block(hooks, "update skills", task_id="miss")
        assert block.count("- guide://org-hub") == 1, block
    finally:
        await session.dispose()
