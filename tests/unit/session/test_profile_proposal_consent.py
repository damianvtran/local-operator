"""Real Session witnesses + real tools: local intent, remote input, verbal yes.

Only synthetic profiles/config roots; no provider call, live session or signer.
"""

from typing import Any

import pytest

from local_operator import profile_proposals as P
from local_operator.harness.types import CustomMessage, Message, ToolContext
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


def _details(result) -> dict[str, Any]:
    """A tool result's details; every result asserted on here carries them."""
    assert result.details is not None, result.content
    return result.details


def _get(root, proposal_id) -> dict[str, Any]:
    """``P.get`` for a proposal the test just filed: absent is a failure."""
    record = P.get(root, proposal_id)
    assert record is not None, proposal_id
    return record


def start(session, *messages):
    session._run_owner_texts = []
    session._run_owner_text_bindings = []
    session._run_owner_consent = False
    session._run_untrusted_input = False
    for message in messages:
        session._note_run_input(message)


def _ctx(session, stores) -> ToolContext:
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
            None,
            None,
            _ctx(session, stores),
        )
        assert not result.is_error, result.content
        assert _details(result).get("status") != "proposed"
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
            None,
            None,
            _ctx(session, stores),
        )
        assert not result.is_error, result.content
        assert _details(result).get("status") != "proposed"
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
            None,
            None,
            _ctx(session, stores),
        )
        assert _details(result)["status"] == "proposed", result.content
        pid = _details(result)["proposal_id"]
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        assert _details(applied)["proposal"]["state"] == "accepted"
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
            CustomMessage(custom_type=kind, attribution="user", details={"text": "Edit it"}),
        )
        result = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "instructions": "Remote suggestion."},
            None,
            None,
            _ctx(session, stores),
        )
        assert _details(result)["status"] == "proposed", result.content
        pid = _details(result)["proposal_id"]
        root = stores["agents"].config_dir
        assert _get(root, pid)["origin"]["trust"] == "untrusted"
        assert gate.cards == []
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
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
            None,
            None,
            _ctx(session, stores),
        )
        assert _details(result)["status"] == "proposed", result.content
        applied = await execute_profile_proposal(
            "a2",
            {"op": "apply", "id": _details(result)["proposal_id"]},
            None,
            None,
            _ctx(session, stores),
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
            "a1", {"op": "delete", "name": "reviewer"}, None, None, _ctx(session, stores)
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
        ctx = _ctx(session, stores)
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
            None,
            None,
            ctx,
        )
        assert not result.is_error, result.content
        assert (
            stores["account"].config_dir / "skills" / "marketing" / "SKILL.md"
        ).read_text() == content
    finally:
        await session.dispose()


# --- round 1 remediation regressions (F2/F7, U1, U2, F3, F8) -----------------


async def _propose_agent(session, stores, text="Tell me about reviewer.", body="New rules."):
    start(session, Message.user(text))
    result = await execute_agent(
        "a1",
        {"op": "update", "name": "reviewer", "instructions": body},
        None,
        None,
        _ctx(session, stores),
    )
    assert _details(result)["status"] == "proposed", result.content
    return _details(result)["proposal_id"]


async def _propose_team(session, stores, body):
    start(session, Message.user("How is lopdev doing?"))
    result = await execute_team(
        "t1",
        {"op": "update", "name": "lopdev", "instructions": body},
        None,
        None,
        _ctx(session, stores),
    )
    assert _details(result)["status"] == "proposed", result.content
    return _details(result)["proposal_id"]


async def test_yes_cannot_be_rebound_by_a_show_in_the_same_run(stores):
    """F2: the owner's "yes" answered something else; the model then shows a
    DIFFERENT proposal and applies it. The yes is bound to what was displayed
    before it arrived, so the apply is refused. Next turn, after the proposal
    was genuinely shown, the same "yes" applies it."""
    session = _session(stores)
    try:
        agent_pid = await _propose_agent(session, stores)
        team_pid = await _propose_team(session, stores, "Team draft.")  # displayed last
        start(session, Message.user("yes"))
        ctx = _ctx(session, stores)
        shown = await execute_profile_proposal(
            "s1", {"op": "show", "id": agent_pid}, None, None, ctx
        )
        assert not shown.is_error, shown.content
        refused = await execute_profile_proposal(
            "a2", {"op": "apply", "id": agent_pid}, None, None, ctx
        )
        assert refused.is_error, refused.content
        root = stores["agents"].config_dir
        assert _get(root, agent_pid)["state"] == "pending"
        assert _get(stores["teams"].config_dir, team_pid)["state"] == "pending"
        # The show above displayed it to the person; their NEXT yes refers to it.
        start(session, Message.user("yes"))
        applied = await execute_profile_proposal(
            "a3", {"op": "apply", "id": agent_pid}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        assert _get(root, agent_pid)["state"] == "accepted"
    finally:
        await session.dispose()


async def test_ask_answer_alone_never_grants_a_proposal(stores):
    """F7: a person answering some ask is not bound to any proposal."""
    session = _session(stores)
    try:
        pid = await _propose_agent(session, stores)
        start(session)
        session._run_owner_consent = True  # what a tui/desktop ask answer sets
        result = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert result.is_error
        assert _get(stores["agents"].config_dir, pid)["state"] == "pending"
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    "reply",
    [
        "ok",
        "OK go ahead",
        "looks good, apply it",
        "Yes please apply the change",
        "yes go ahead and apply",
        "sounds good",
        "lgtm",
    ],
)
async def test_contextual_assent_applies_displayed_proposal_without_card(stores, reply):
    """U2: common assent applies the proposal shown before it, with no second card."""
    gate = Gate()
    session = _session(stores, gate=gate)
    try:
        pid = await _propose_agent(session, stores)
        start(session, Message.user(reply))
        applied = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        assert gate.cards == []
    finally:
        await session.dispose()


@pytest.mark.parametrize("reply", ["no, don't apply it", "yes but change the tone", "ok?"])
async def test_hedged_or_negative_reply_is_not_assent(stores, reply):
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        pid = await _propose_agent(session, stores)
        start(session, Message.user(reply))
        result = await execute_profile_proposal(
            "a2", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert result.is_error
        assert len(gate.cards) == 1  # only the exact-proposal card could apply it
        assert _get(stores["agents"].config_dir, pid)["state"] == "pending"
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    "request_text",
    [
        "Could you please create a new team marketing",
        "Set up a team marketing",
        "I need a new team marketing",
        "Let\u2019s create a team marketing",
        "Hey, make me a team marketing",
    ],
)
async def test_natural_owner_create_applies_directly(stores, request_text):
    """U1: polite or framed creation requests are the owner's explicit instruction."""
    session = _session(stores)
    try:
        start(session, Message.user(request_text))
        result = await execute_team(
            "t1",
            {"op": "create", "name": "marketing", "manager": "reviewer", "scope": "global"},
            None,
            None,
            _ctx(session, stores),
        )
        assert not result.is_error, result.content
        assert _details(result).get("status") != "proposed", result.content
        assert stores["teams"].get_team_by_name("marketing") is not None
    finally:
        await session.dispose()


async def test_natural_owner_edit_applies_directly(stores):
    session = _session(stores)
    try:
        start(session, Message.user("Can you please make reviewer more concise?"))
        result = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "instructions": "Be concise."},
            None,
            None,
            _ctx(session, stores),
        )
        assert not result.is_error, result.content
        assert _details(result).get("status") != "proposed", result.content
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    "request_text",
    [
        'He wrote "create a team marketing" in the doc.',
        "Tell me how to create a team marketing",
        "What would a marketing team look like?",
    ],
)
async def test_mention_or_quote_of_creation_is_not_consent(stores, request_text):
    session = _session(stores)
    try:
        start(session, Message.user(request_text))
        result = await execute_team(
            "t1",
            {"op": "create", "name": "marketing", "manager": "reviewer", "scope": "global"},
            None,
            None,
            _ctx(session, stores),
        )
        assert _details(result).get("status") == "proposed", result.content
        assert stores["teams"].get_team_by_name("marketing") is None
    finally:
        await session.dispose()


async def test_team_proposal_after_verbal_apply_is_not_stale_at_birth(stores):
    """F3/Q1: the apply writes through a fresh registry; the session's team
    registry must not hand the next proposal a cached before-image."""
    session = _session(stores)
    try:
        root = stores["teams"].config_dir
        stores["teams"].get_team_by_name("lopdev")  # warm the interval cache
        first = await _propose_team(session, stores, "Applied text.")
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "a2", {"op": "apply", "id": first}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        second = await _propose_team(session, stores, "Next idea.")
        record = _get(root, second)
        assert record is not None
        assert record["before"]["instructions"] == "Applied text."
        assert record["stale"] is False
    finally:
        await session.dispose()


async def test_identical_refile_of_rejected_proposal_is_not_revived(stores):
    """F8: the receipt reports the real state; nothing pending is resurrected."""
    session = _session(stores)
    try:
        pid = await _propose_agent(session, stores)
        root = stores["agents"].config_dir
        P.reject(root, pid, actor="desktop")
        start(session, Message.user("Tell me about reviewer."))
        again = await execute_agent(
            "a1",
            {"op": "update", "name": "reviewer", "instructions": "New rules."},
            None,
            None,
            _ctx(session, stores),
        )
        assert not again.is_error, again.content
        assert _details(again)["proposal_id"] == pid
        assert _details(again)["status"] == "rejected"
        assert "is rejected" in str(again.content)
        assert _get(root, pid)["state"] == "rejected"
        assert P.list_proposals(root) == []
    finally:
        await session.dispose()
