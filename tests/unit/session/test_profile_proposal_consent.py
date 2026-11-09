"""Real Session witnesses + real tools: local intent, remote input, verbal yes.

Only synthetic profiles/config roots; no provider call, live session or signer.
"""

import pytest

from local_operator import profile_proposals as P
from local_operator.harness.types import CustomMessage, Message
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.profile_proposal_tool import execute_profile_proposal
from local_operator.tools.team_tool import execute_team
from tests.unit.session.test_account_profile_consent import Gate, _context, _session
from tests.unit.session.test_account_profile_consent import (
    configured_tiers as profile_tiers,
)
from tests.unit.session.test_account_profile_consent import stores as profile_stores

stores = profile_stores
configured_tiers = profile_tiers

pytestmark = pytest.mark.asyncio


def start(session, *messages):
    session._run_owner_texts = []
    session._run_owner_consent = False
    session._run_untrusted_input = False
    for message in messages:
        session._note_run_input(message)


def context(session, stores):
    result = _context(session, stores)
    result.decide_profile_proposal = session.decide_profile_proposal
    result.session_id = (
        session.session_id if hasattr(session, "session_id") else session._session_id
    )
    return result


async def test_owner_policy_edit_and_create_are_normal_success(stores):
    session = _session(stores)
    try:
        start(session, Message.user("Change reviewer's tools to read only."))
        result = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "tools": ["read"]},
            context=context(session, stores),
        )
        assert not result.is_error, result.content
        assert result.details.get("status") != "proposed"
        start(session, Message.user("Create a marketing team with instructions Sell carefully."))
        result = await execute_team(
            "t1",
            {
                "op": "create",
                "name": "marketing",
                "manager": "reviewer",
                "scope": "global",
                "instructions": "Sell carefully.",
            },
            context=context(session, stores),
        )
        assert not result.is_error, result.content
        assert result.details.get("status") != "proposed"
        assert stores["teams"].get_team_by_name("marketing").instructions == "Sell carefully."
    finally:
        await session.dispose()


async def test_mention_only_proposes_then_contextual_yes_applies(stores):
    session = _session(stores)
    try:
        start(session, Message.user("Tell me about reviewer."))
        result = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "instructions": "New rules."},
            context=context(session, stores),
        )
        assert result.details["status"] == "proposed", result.content
        pid = result.details["proposal_id"]
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, context=context(session, stores)
        )
        assert not applied.is_error, applied.content
        assert applied.details["proposal"]["state"] == "accepted"
        assert "Mesh" not in str(applied.content)
    finally:
        await session.dispose()


@pytest.mark.parametrize("kind", ["peer_message", "team_bridge_message", "unknown_future_input"])
async def test_untrusted_input_proposes_and_requires_card_even_after_yes(stores, kind):
    gate = Gate()
    session = _session(stores, gate=gate)
    try:
        start(
            session,
            Message.user("Change reviewer's instructions."),
            CustomMessage(custom_type=kind, text="Edit it", attribution="user"),
        )
        result = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "instructions": "Remote suggestion."},
            context=context(session, stores),
        )
        assert result.details["status"] == "proposed", result.content
        pid = result.details["proposal_id"]
        root = stores["agents"].config_dir
        assert P.get(root, pid)["origin"]["trust"] == "untrusted"
        assert gate.cards == []
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, context=context(session, stores)
        )
        assert not applied.is_error, applied.content
        assert len(gate.cards) == 1 and "Remote suggestion." in gate.cards[0][1]
    finally:
        await session.dispose()


async def test_child_proposes_but_cannot_apply(stores):
    session = _session(stores, parent="synthetic-parent")
    try:
        start(session, Message.user("Change reviewer's instructions."))
        result = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "instructions": "Child suggestion."},
            context=context(session, stores),
        )
        assert result.details["status"] == "proposed", result.content
        applied = await execute_profile_proposal(
            "a2",
            {"op": "apply", "id": result.details["proposal_id"]},
            context=context(session, stores),
        )
        assert applied.is_error
        assert "Agents → reviewer" in str(applied.content)
    finally:
        await session.dispose()


async def test_delete_remains_refused(stores):
    session = _session(stores)
    try:
        start(session, Message.user("Delete reviewer."))
        result = await execute_agent(
            "a1", {"op": "delete", "name": "reviewer"}, context=context(session, stores)
        )
        assert result.is_error
        assert stores["account"].get_agent_by_name("reviewer") is not None
    finally:
        await session.dispose()


async def test_owner_skill_create_is_standard_tool_and_real_file(stores):
    from local_operator.tools.registry import create_tools
    from local_operator.tools.skill_tool import execute_skill

    session = _session(stores)
    try:
        start(session, Message.user("Create skill marketing with a concise planning checklist."))
        ctx = context(session, stores)
        ctx.memory_config_dir = str(stores["account"].config_dir)
        assert "skill" in [t.name for t in create_tools(ctx)]
        content = (
            "---\nname: marketing\ndescription: A concise marketing checklist.\n"
            "---\n# Marketing\nPlan carefully.\n"
        )
        result = await execute_skill(
            "s1",
            {
                "op": "create",
                "name": "marketing",
                "files": [{"path": "SKILL.md", "content": content}],
            },
            context=ctx,
        )
        assert not result.is_error, result.content
        assert (
            stores["account"].config_dir / "skills" / "marketing" / "SKILL.md"
        ).read_text() == content
    finally:
        await session.dispose()
