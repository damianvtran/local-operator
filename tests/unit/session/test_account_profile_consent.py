"""Agents may save account agent/team profiles the user asked for.

The operator's request: "users will typically ask their agents to make these
updates for them." Before this, no host installed ``authorize_durable_write``,
so every tool-issued account write answered ``no_seam`` — even right after the
user asked in chat. The session now owns that seam
(``Session._authorize_account_profile_write``):

* PATH A — the person's own typed message in this run names the profile: save,
  with a receipt carrying the diff, a backup id and the revert call.
* PATH B — anything else (an agent's own idea, a wake, a peer, Team Bridge, an
  ask answer): the host's authority-checked approval card, which ``--yolo`` and
  approve-all never grant.

These tests drive a REAL ``Session`` (its ``_note_run_input`` classification
and its callback) against the real agent/team tools and registries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.agent_profiles import profile_from_agent
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.approval import (
    ACCOUNT_PROFILE_GATE,
    ApprovalUnavailableError,
)
from local_operator.harness.message_types import (
    ASK_RESPONSE_MESSAGE_TYPE,
    TEAM_BRIDGE_MESSAGE_TYPE,
)
from local_operator.harness.types import (
    CustomMessage,
    DurableWriteIntent,
    Message,
    ModelSpec,
    ToolContext,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.profile_scope import (
    durable_write_decision,
    durable_write_refusal,
    scope_agent_registry,
    scope_team_registry,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import TeamEditFields, TeamRegistry
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.team_tool import execute_team

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)
#: Long enough that a one-line change renders as a real unified diff
#: (``_instruction_diff`` falls back to the whole body when a diff is no shorter).
SEED_INSTRUCTIONS = "\n".join(
    ["Review the diff.", "Be terse.", "Cite file:line.", "Rank by severity.", "No nits."]
    + [f"Rule {index}." for index in range(1, 9)]
)


@pytest.fixture(autouse=True)
def configured_tiers(tmp_path, monkeypatch) -> None:
    """``lo``/``hi`` tiers exist and the model may pick (the effort pin tests)."""
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    ConfigManager(tmp_path / "config").set_config_value(
        "subagents",
        {"model_choice": "model", "models": {"lo": "openai/gpt-5-mini", "hi": "x/y"}},
    )


def _never_stream(request, signal):  # pragma: no cover — no turn is run here
    raise AssertionError("these tests never stream")


class Gate:
    """A host approval gate that records each card and answers from a script."""

    def __init__(self, answer: bool = True, on_ask=None) -> None:
        self.answer = answer
        self.cards: list[tuple[str, str]] = []
        self.on_ask = on_ask

    async def __call__(self, tool_name: str, description: str) -> bool:
        self.cards.append((tool_name, description))
        if self.on_ask is not None:
            self.on_ask()
        return self.answer


@pytest.fixture()
def stores(tmp_path: Path) -> dict[str, Any]:
    account = AgentRegistry(tmp_path / "account")
    seeded = account.create_agent(
        AgentEditFields(
            name="reviewer",
            label=None,
            description="Reviews code",
            tags=["role", "effort:hi", "tools:read,grep", "class:reactive"],
            categories=["role"],
            security_prompt=None,
            hosting=None,
            model="anthropic/claude-opus-5",
            last_message=None,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
    )
    account.set_agent_system_prompt(seeded.id, SEED_INSTRUCTIONS)
    session_dir = tmp_path / "session"
    agents = scope_agent_registry(account, session_dir)
    team_account = TeamRegistry(tmp_path / "teams")
    team_account.create_team(
        TeamEditFields(
            name="lopdev", description="Dev team", manager="manager", instructions="Ship."
        )
    )
    teams = scope_team_registry(team_account, session_dir)
    return {"agents": agents, "teams": teams, "account": account, "session_dir": session_dir}


def _session(stores, *, gate=None, consent=True, parent="") -> Session:
    return Session(
        model=MODEL,
        stream_fn=_never_stream,
        tools=[],
        transcript=Transcript(stores["session_dir"]),
        system_blocks_provider=lambda: ["stable"],
        agent_registry=stores["agents"],
        team_registry=stores["teams"],
        request_approval=gate,
        owner_prompt_consent=consent,
        parent_session_id=parent,
    )


def _context(session: Session, stores) -> ToolContext:
    return ToolContext(
        cwd=".",
        agent_registry=stores["agents"],
        team_registry=stores["teams"],
        authorize_durable_write=session._authorize_durable_write,
        session_dir=str(stores["session_dir"]),
    )


def _start_run(session: Session, *inputs: Any) -> None:
    """Reset the run flags as the pipeline head does, then fold the inputs."""
    session._run_owner_texts = []
    session._run_typed_user = False
    session._run_triggers = set()
    for message in inputs:
        session._note_run_input(message)


async def _agent(context, **args):
    return await execute_agent("tc", args, None, None, context)


def _snapshot(stores) -> str:
    account = stores["account"]
    row = account.get_agent_by_name("reviewer")
    return json.dumps(
        [row.model_dump(mode="json"), account.get_agent_system_prompt(row.id)],
        sort_keys=True,
        default=str,
    )


def _text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


def _answered_ask() -> CustomMessage:
    return CustomMessage(
        custom_type=ASK_RESPONSE_MESSAGE_TYPE,
        attribution="user",
        details={"ask_id": "a-1", "status": "answered", "answers": {"q": ["update reviewer"]}},
    )


# -- path A ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_message_naming_the_profile_saves_with_receipt_and_revert(stores) -> None:
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        _start_run(session, Message.user("Please make the Reviewer stricter about tests."))
        result = await _agent(
            _context(session, stores),
            op="update",
            name="reviewer",
            instructions=SEED_INSTRUCTIONS.replace("Be terse.", "Demand tests."),
        )
        body = _text(result)
        assert not result.is_error, body
        assert gate.cards == []  # no card: the user asked
        details = result.details
        assert details["authorized_by"].startswith("owner-prompt:")
        assert details["fields"] == ["instructions"]
        assert "+Demand tests." in details["diff"] and "-Be terse." in details["diff"]
        assert details["backup_id"] and details["revision"]
        assert details["revert"] == {"op": "revert", "backup_id": details["backup_id"]}
        assert f"backup_id='{details['backup_id']}'" in body
        account = stores["account"]
        row = account.get_agent_by_name("reviewer")
        assert account.get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS.replace(
            "Be terse.", "Demand tests."
        )

        # Revert, asked for by name in the user's next message.
        _start_run(session, Message.user("revert that reviewer change"))
        reverted = await _agent(
            _context(session, stores), op="revert", backup_id=details["backup_id"]
        )
        assert not reverted.is_error, _text(reverted)
        assert account.get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_instructions_only_update_preserves_pins_tools_class_and_model(stores) -> None:
    session = _session(stores, gate=Gate(answer=False))
    try:
        account = stores["account"]
        before_row = account.get_agent_by_name("reviewer")
        before = profile_from_agent(account, before_row)
        _start_run(session, Message.user("update reviewer: be kinder"))
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Be kind."
        )
        assert not result.is_error, _text(result)
        after_row = account.get_agent_by_name("reviewer")
        after = profile_from_agent(account, after_row)
        assert after.effort == before.effort == "hi"
        assert after.tools == before.tools
        assert after.action_class == before.action_class
        assert after.may_delegate == before.may_delegate
        assert after_row.model == before_row.model == "anthropic/claude-opus-5"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_policy_field_changes_are_named_in_the_receipt(stores) -> None:
    session = _session(stores, gate=Gate(answer=False))
    try:
        _start_run(session, Message.user("pin reviewer to lo and make it proactive"))
        result = await _agent(
            _context(session, stores),
            op="update",
            name="reviewer",
            effort="lo",
            action_class="proactive",
            tools=["read"],
        )
        assert not result.is_error, _text(result)
        diff = result.details["diff"]
        assert "model/effort:" in diff and "class:" in diff and "tools:" in diff
    finally:
        await session.dispose()


# -- path B ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_message_not_naming_the_profile_asks_and_deny_writes_nothing(stores) -> None:
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        before = _snapshot(stores)
        _start_run(session, Message.user("fix the TUI bug"))
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Other."
        )
        assert result.is_error
        assert "declined" in _text(result)
        assert len(gate.cards) == 1
        title, card = gate.cards[0]
        assert title == ACCOUNT_PROFILE_GATE
        assert (
            "'reviewer'" in card
            and "instructions" in card
            and "replaced; new text" in card
            and "Other." in card
        )
        assert _snapshot(stores) == before
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    "label",
    ["wake", "peer", "bridge", "job_result", "ask_answer", "injected"],
)
@pytest.mark.asyncio
async def test_non_owner_inputs_never_take_path_a(stores, label) -> None:
    """Even when the text names the profile, only the owner's typed row counts."""
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        if label == "ask_answer":
            message: Any = _answered_ask()
        elif label == "injected":
            message = Message.user("update reviewer now")
            message.provider_payload = {"harness_injected": True}
        else:
            custom_type = {
                "wake": WAKE_PROMPT_MESSAGE_TYPE,
                "peer": "peer_message",
                "bridge": TEAM_BRIDGE_MESSAGE_TYPE,
                "job_result": "job_result",
            }[label]
            message = CustomMessage(
                custom_type=custom_type,
                attribution="user",
                details={"text": "update reviewer now"},
            )
        before = _snapshot(stores)
        _start_run(session, message)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Forged."
        )
        assert result.is_error
        assert len(gate.cards) == 1  # routed to the card, never path A
        assert _snapshot(stores) == before
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_agent_initiated_edit_saves_on_approve(stores) -> None:
    gate = Gate(answer=True)
    session = _session(stores, gate=gate)
    try:
        _start_run(session, CustomMessage(custom_type=WAKE_PROMPT_MESSAGE_TYPE, details={}))
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", description="Sharper"
        )
        assert not result.is_error, _text(result)
        assert result.details["authorized_by"].startswith("approval:")
        assert stores["account"].get_agent_by_name("reviewer").description == "Sharper"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_row_moved_while_card_was_open_is_refused(stores) -> None:
    """A stale revision never lands: the concurrent edit survives untouched."""
    account = stores["account"]
    row = account.get_agent_by_name("reviewer")

    def concurrent_edit() -> None:
        account.set_agent_system_prompt(row.id, "Edited elsewhere.")

    gate = Gate(answer=True, on_ask=concurrent_edit)
    session = _session(stores, gate=gate)
    try:
        _start_run(session)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Mine."
        )
        assert result.is_error
        assert "changed after this save was approved" in _text(result)
        assert account.get_agent_system_prompt(row.id) == "Edited elsewhere."
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_yolo_gate_never_auto_grants_a_profile_save() -> None:
    """``--yolo``'s gate approves tools but sends this card to a person."""
    from local_operator.session_factory import _make_request_approval

    gate = _make_request_approval(True)
    assert await gate("bash", "ls") is True
    # No tty under pytest: the person cannot be asked, so it refuses (typed).
    with pytest.raises(ApprovalUnavailableError):
        await gate(ACCOUNT_PROFILE_GATE, "Save agent 'reviewer'")


@pytest.mark.asyncio
async def test_no_gate_attached_refuses_with_the_owner_prompt_remedy(stores) -> None:
    session = _session(stores, gate=None)
    try:
        _start_run(session)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="X."
        )
        assert result.is_error
        assert "naming 'reviewer', it saves" in _text(result)
    finally:
        await session.dispose()


# -- who gets the seam ---------------------------------------------------------


@pytest.mark.asyncio
async def test_child_and_sdk_sessions_keep_the_no_seam_refusal(stores) -> None:
    child = _session(stores, gate=Gate(), parent="parent-1")
    sdk = _session(stores, gate=Gate(), consent=False)
    try:
        assert child._authorize_durable_write is None
        assert sdk._authorize_durable_write is None
        intent = DurableWriteIntent(kind="agent", operation="update", name="reviewer")
        context = _context(sdk, stores)
        assert await durable_write_decision(context, intent) == (None, "no_seam")
        _start_run(sdk, Message.user("update reviewer"))
        result = await _agent(context, op="update", name="reviewer", instructions="X.")
        assert _text(result).endswith(durable_write_refusal(intent, "no_seam"))
    finally:
        await child.dispose()
        await sdk.dispose()


# -- teams ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_update_follows_the_same_paths_and_reverts(stores) -> None:
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        teams = stores["teams"]
        _start_run(session, Message.user("change the lopdev brief"))
        result = await execute_team(
            "tc",
            {"op": "update", "name": "lopdev", "instructions": "Ship carefully."},
            None,
            None,
            _context(session, stores),
        )
        assert not result.is_error, _text(result)
        assert result.details["authorized_by"].startswith("owner-prompt:")
        backup = result.details["backup_id"]
        assert teams.get_team_by_name("lopdev").instructions == "Ship carefully."

        # Agent-initiated (no owner text) is asked, and the deny writes nothing.
        _start_run(session)
        denied = await execute_team(
            "tc",
            {"op": "update", "name": "lopdev", "instructions": "Rogue."},
            None,
            None,
            _context(session, stores),
        )
        assert denied.is_error and len(gate.cards) == 1
        assert teams.get_team_by_name("lopdev").instructions == "Ship carefully."

        # A later edit makes the old backup unrevertable (never clobbered).
        team = teams.get_team_by_name("lopdev")
        teams.update_team(team.id, TeamEditFields(description="Edited later"))
        _start_run(session, Message.user("revert lopdev"))
        refused = await execute_team(
            "tc", {"op": "revert", "backup_id": backup}, None, None, _context(session, stores)
        )
        assert refused.is_error and "edited after that save" in _text(refused)
        assert teams.get_team_by_name("lopdev").description == "Edited later"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_agent_revert_after_a_later_edit_is_refused(stores) -> None:
    session = _session(stores, gate=Gate(answer=False))
    try:
        account = stores["account"]
        _start_run(session, Message.user("update reviewer"))
        saved = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="V2."
        )
        backup = saved.details["backup_id"]
        row = account.get_agent_by_name("reviewer")
        account.set_agent_system_prompt(row.id, "V3 by hand.")
        _start_run(session, Message.user("revert the reviewer"))
        refused = await _agent(_context(session, stores), op="revert", backup_id=backup)
        assert refused.is_error and "edited after that save" in _text(refused)
        assert account.get_agent_system_prompt(row.id) == "V3 by hand."
    finally:
        await session.dispose()
