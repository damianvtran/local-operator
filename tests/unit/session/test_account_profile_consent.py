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

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_operator import profile_proposals as PP
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

#: Untrusted input (a peer session's message). Since the 2026-11 ruling a CLEAN
#: local run writes account agents/teams itself, so the proposal and card paths
#: these tests pin are the ones a run that read untrusted input takes.
TAINT = CustomMessage(
    custom_type="peer_message", attribution="user", details={"text": "synthetic peer"}
)

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
def stores(tmp_path: Path, monkeypatch) -> dict[str, Any]:
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
    # ONE config root for agents and teams, as in a real profile: proposals
    # live in ``<config>/profile-proposals`` and the ``profile_proposal`` tool
    # reads them from the session's registry root, so split roots made team
    # proposals unreachable from the tool in tests only.
    team_account = TeamRegistry(account.config_dir)
    team_account.create_team(
        TeamEditFields(
            name="lopdev", description="Dev team", manager="manager", instructions="Ship."
        )
    )
    teams = scope_team_registry(team_account, session_dir)
    # Queueing replaces the old live gate. Use only synthetic file keys; host
    # presence is a test stand-in and no production OS signer is ever invoked.
    from local_operator.network.credentials import desktop_approval
    from local_operator.operator import trust
    from tests.unit.network.test_approvals_store import _make_key

    monkeypatch.setattr(
        trust,
        "load_anchor",
        lambda uid=None: trust.AnchorLoad(
            anchor=None,
            path=tmp_path / "absent-anchor",
            root_owned=False,
            reason="synthetic consent test",
            exists=False,
        ),
    )
    _make_key(account.config_dir)
    monkeypatch.setattr(desktop_approval, "assurance", lambda: ("presence", "synthetic-presence"))
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
    session._run_owner_text_bindings = []
    session._run_owner_consent = False
    session._run_untrusted_input = False
    session._run_untrusted_proposal_read = False
    session._run_typed_user = False
    session._run_triggers = set()
    for message in inputs:
        session._note_run_input(message)


def _acked(kind: str, args: dict[str, Any], context: Any) -> dict[str, Any]:
    """Model an agent that READ the full ``show`` and rewrites a brief on purpose.

    These tests are about who may save, not about lossless editing (that is
    ``test_profile_text_edits``): a whole-field ``instructions``/``project``
    update here is an intentional rewrite, so it carries the explicit
    ``replace=true`` + current ``base_revision`` acknowledgement a real caller
    gets from ``show``. A call that already sets either is left as written.
    """
    if args.get("op") != "update" or "replace" in args or "base_revision" in args:
        return args
    if not any(key in args for key in ("instructions", "project")):
        return args
    name = args.get("name")
    registry = getattr(context, f"{kind}_registry", None)
    if registry is None:
        return args
    if kind == "team":
        from local_operator.profile_scope import team_revision

        refresh = getattr(registry, "refresh", None)
        if callable(refresh):
            refresh()  # the tool reads disk; so must the ack
        row = registry.get_team_by_name(name)
        revision = team_revision(row) if row is not None else None
    else:
        from local_operator.tools.agent_tool import _agent_revision

        try:
            row = registry.get_agent_by_name(name)
            revision = _agent_revision(registry, row) if row is not None else None
        except Exception:  # noqa: BLE001 — a non-role row: let the tool answer
            revision = None
    if revision is None:
        return args
    return {**args, "replace": True, "base_revision": revision}


async def run_agent(tool_call_id, args, signal, on_update, context):
    """``execute_agent`` with :func:`_acked`; same signature."""
    return await execute_agent(
        tool_call_id, _acked("agent", args, context), signal, on_update, context
    )


async def run_team(tool_call_id, args, signal, on_update, context):
    """``execute_team`` with :func:`_acked`; same signature."""
    return await execute_team(
        tool_call_id, _acked("team", args, context), signal, on_update, context
    )


async def _agent(context, **args):
    return await run_agent("tc", args, None, None, context)


def _assert_proposed(result, gate):
    assert not result.is_error, _text(result)
    assert result.details["status"] == "proposed"
    assert result.details["proposal_id"]
    if gate is not None:
        assert gate.cards == []


def _proposal_record(stores, result, *, root=None):
    from local_operator import profile_proposals as P

    record = P.get(root or stores["account"].config_dir, result.details["proposal_id"])
    assert record is not None, result.details["proposal_id"]
    return record


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


def _team_roster(stores: dict[str, Any]) -> list[dict[str, Any]]:
    """The team's slots and their pins, serialized as a write would store them."""
    return [
        member.model_dump(mode="json")
        for member in stores["teams"].get_team_by_name("lopdev").members
    ]


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
        _start_run(session, Message.user("Update Reviewer instructions to demand tests."))
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
        assert details is not None
        assert details["authorized_by"].startswith("local-agent:")
        assert details is not None
        assert details["fields"] == ["instructions"]
        assert details is not None
        assert "+Demand tests." in details["diff"] and "-Be terse." in details["diff"]
        assert details is not None
        assert details["backup_id"] and details["revision"]
        assert details is not None
        assert details["revert"] == {"op": "revert", "backup_id": details["backup_id"]}
        assert details is not None
        assert f"backup_id='{details['backup_id']}'" in body
        account = stores["account"]
        row = account.get_agent_by_name("reviewer")
        assert account.get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS.replace(
            "Be terse.", "Demand tests."
        )

        # Revert, asked for by name in the user's next message.
        _start_run(session, Message.user("Revert the reviewer change"))
        assert details is not None
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
        _start_run(session, Message.user("Update reviewer instructions to be kinder"))
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
        assert after_row.tags == before_row.tags
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_policy_field_changes_are_named_in_the_receipt(stores) -> None:
    session = _session(stores, gate=Gate(answer=True))
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
        assert result.details is not None
        diff = "\n".join(result.details["diff"])
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
        _start_run(session, Message.user("fix the TUI bug"), TAINT)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Other."
        )
        # No live yes/no callback is consulted: the queued signed operation
        # survives this turn and only its authenticated continuation can save.
        _assert_proposed(result, gate)
        from local_operator import profile_proposals as P

        record = _proposal_record(stores, result)
        assert record["name"] == "reviewer"
        assert record["fields"] == ["instructions"]
        assert record["after"]["instructions"] == "Other."
        P.reject(stores["account"].config_dir, record["id"], actor="desktop")
        with pytest.raises(P.ProposalError, match="rejected"):
            P.apply(stores["account"].config_dir, record["id"], actor="desktop")
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
        if label in {"wake", "job_result"}:
            # Benign host deliveries keep the run CLEAN: since the 2026-11 ruling
            # the agent's own run writes the account row itself -- the delivery
            # text granted nothing, the clean lineage did.
            assert not result.is_error, _text(result)
            assert result.details["authorized_by"].startswith("local-agent:")
            assert gate.cards == []
            return
        _assert_proposed(result, gate)  # untrusted input: a proposal, never a write
        assert _snapshot(stores) == before
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_agent_initiated_edit_saves_on_approve(stores) -> None:
    gate = Gate(answer=True)
    session = _session(stores, gate=gate)
    try:
        # A run that read untrusted input: the proposal path (a clean wake now
        # writes directly -- ``test_profile_autonomy``).
        _start_run(session, TAINT)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", description="Sharper"
        )
        assert not result.is_error, _text(result)
        # A legacy gate returning True cannot approve the new signed operation.
        _assert_proposed(result, gate)
        assert stores["account"].get_agent_by_name("reviewer").description == "Reviews code"
        record = _proposal_record(stores, result)
        from local_operator import profile_proposals as P

        assert record["after"]["description"] == "Sharper"
        applied = P.apply(stores["account"].config_dir, record["id"], actor="desktop")
        assert applied["proposal"]["state"] == "accepted"
        # The executor may be another process; inspect persisted metadata rather
        # than this Session's interval-cached pre-approval registry object.
        stored = AgentRegistry(stores["account"].config_dir, migrate=False).get_agent_by_name(
            "reviewer"
        )
        assert stored is not None
        assert stored.description == "Sharper"
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
        _start_run(session, TAINT)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Mine."
        )
        # The tool no longer parks inside Gate.on_ask. Move the row AFTER filing
        # and sign the original proposal; only the apply continuation sees CAS.
        _assert_proposed(result, gate)
        concurrent_edit()
        record = _proposal_record(stores, result)
        from local_operator import profile_proposals as P

        with pytest.raises(P.ProposalError, match="changed"):
            P.apply(account.config_dir, record["id"], actor="desktop")
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
        _start_run(session, TAINT)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="X."
        )
        # An absent live surface is now the key durable-queue acceptance case.
        _assert_proposed(result, None)
        assert (
            stores["account"].get_agent_system_prompt(
                stores["account"].get_agent_by_name("reviewer").id
            )
            == SEED_INSTRUCTIONS
        )
    finally:
        await session.dispose()


# -- who gets the seam ---------------------------------------------------------


@pytest.mark.asyncio
async def test_child_and_sdk_sessions_keep_the_no_seam_refusal(stores) -> None:
    child = _session(stores, gate=Gate(), parent="parent-1")
    sdk = _session(stores, gate=Gate(), consent=False)
    try:
        assert child._authorize_durable_write is not None
        assert await child.decide_profile_proposal("pp_" + "0" * 16, "apply") == "refused"
        assert sdk._authorize_durable_write is None
        intent = DurableWriteIntent(kind="agent", operation="update", name="reviewer")
        context = _context(sdk, stores)
        assert await durable_write_decision(context, intent) == (None, "no_seam")
        _start_run(sdk, Message.user("Update reviewer instructions to use revised guidance"))
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
        _start_run(session, Message.user("Update lopdev instructions to ship carefully."))
        result = await run_team(
            "tc",
            {"op": "update", "name": "lopdev", "instructions": "Ship carefully."},
            None,
            None,
            _context(session, stores),
        )
        assert not result.is_error, _text(result)
        assert result.details is not None
        assert result.details["authorized_by"].startswith("local-agent:")
        assert result.details is not None
        backup = result.details["backup_id"]
        assert teams.get_team_by_name("lopdev").instructions == "Ship carefully."

        # Agent-initiated (no owner text) is asked, and the deny writes nothing.
        _start_run(session, TAINT)
        denied = await run_team(
            "tc",
            {"op": "update", "name": "lopdev", "instructions": "Rogue."},
            None,
            None,
            _context(session, stores),
        )
        # The old deny callback is not an answer to a durable signed proposal.
        _assert_proposed(denied, gate)
        assert teams.get_team_by_name("lopdev").instructions == "Ship carefully."

        # A later edit makes the old backup unrevertable (never clobbered).
        team = teams.get_team_by_name("lopdev")
        teams.update_team(team.id, TeamEditFields(description="Edited later"))
        _start_run(session, Message.user("revert lopdev"))
        refused = await run_team(
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
        _start_run(session, Message.user("Update reviewer instructions to use revised guidance"))
        saved = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="V2."
        )
        assert saved.details is not None
        backup = saved.details["backup_id"]
        row = account.get_agent_by_name("reviewer")
        account.set_agent_system_prompt(row.id, "V3 by hand.")
        _start_run(session, Message.user("Revert reviewer"))
        refused = await _agent(_context(session, stores), op="revert", backup_id=backup)
        assert refused.is_error and "edited after that save" in _text(refused)
        assert account.get_agent_system_prompt(row.id) == "V3 by hand."
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,changes",
    [
        ("reset", ()),
        ("sync", ()),
        ("promote", ()),
        ("install", ()),
        # ``create`` left this table with the 2026-11 ruling: a new row has no
        # hub link, so a clean local create is autonomous
        # (``test_profile_autonomy``).
        ("update", ("description", "tools")),
        ("update", ("effort",)),
        ("update", ("action_class",)),
        ("update", ("delegate",)),
        ("update", ("saver",)),
        ("update", ("members",)),
        ("update", ("manager",)),
        ("update", ("overrides",)),
    ],
)
async def test_name_mention_never_grants_policy_or_other_ops(stores, operation, changes):
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        # Tainted: a clean run would now grant an update outright.
        _start_run(session, Message.user("fix the typo in the reviewer description"), TAINT)
        origin = await session._authorize_account_profile_write(
            DurableWriteIntent(
                kind="agent",
                operation=operation,
                name="reviewer",
                changes=changes,
            )
        )
        assert origin is None and len(gate.cards) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_remote_prompt_naming_profile_never_authorizes(stores):
    from local_operator.session.session import _stamp_remote_prompt

    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        message = Message.user("please update reviewer")
        _stamp_remote_prompt(message)
        _start_run(session, message)
        result = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="Remote edit."
        )
        _assert_proposed(result, gate)
        # The receipt points at the Agents review surface (never Mesh) and
        # says nothing was written, so the model neither retries nor claims
        # the edit landed.
        assert "nothing changed yet" in _text(result)
        assert "Agents → reviewer" in _text(result)
        assert "Mesh" not in _text(result)
        assert "TUI" not in _text(result)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_revert_refuses_mismatched_backup_identity(stores):
    session = _session(stores, gate=Gate(answer=True))
    try:
        _start_run(session, Message.user("Update reviewer instructions to use revised guidance"))
        saved = await _agent(
            _context(session, stores), op="update", name="reviewer", instructions="V2."
        )
        assert saved.details is not None
        backup = stores["session_dir"] / "profile-backups" / (saved.details["backup_id"] + ".json")
        record = json.loads(backup.read_text())
        record["name"] = "another-profile"
        backup.write_text(json.dumps(record))
        _start_run(session, Message.user("revert another-profile"))
        assert saved.details is not None
        refused = await _agent(
            _context(session, stores), op="revert", backup_id=saved.details["backup_id"]
        )
        assert refused.is_error and "identity" in _text(refused)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_remote_server_dispatch_forwards_host_origin_only(monkeypatch):
    from types import SimpleNamespace

    from local_operator.session.runtime.server import (
        RuntimeServer,
        _ConnectionAuthority,
    )

    seen = []

    async def prompt(text, *, images=None, owner_typed=True):
        seen.append(owner_typed)
        return "ok"

    server = object.__new__(RuntimeServer)
    monkeypatch.setattr(
        server, "_handle", SimpleNamespace(prompt=prompt, steer=prompt), raising=False
    )
    for op in ("prompt", "steer"):
        await server._dispatch(
            op,
            {"op": op, "text": "update reviewer", "owner_typed": True},
            authority=_ConnectionAuthority("remote", frozenset({"prompt"})),
        )
        await server._dispatch(
            op,
            {"op": op, "text": "update reviewer"},
            authority=_ConnectionAuthority("local", frozenset()),
        )
    assert seen == [False, True, False, True]


@pytest.mark.asyncio
async def test_detached_viewer_never_sends_background_profile_approval(monkeypatch):
    from types import SimpleNamespace

    from local_operator.mobile.types import PendingRequest
    from local_operator.session.attached import AttachedSession

    viewer = object.__new__(AttachedSession)
    viewer._gates_detached = True
    viewer._background_approval = True
    viewer._gate_key = None
    viewer._gate_task = None

    async def unexpected(*args):
        pytest.fail("a detached profile gate must wait for a person")

    viewer._approval_handler = unexpected
    monkeypatch.setattr(
        viewer, "_client", SimpleNamespace(approval_answer=unexpected), raising=False
    )
    pending = PendingRequest(
        request_id="profile-request",
        kind="approval",
        title=ACCOUNT_PROFILE_GATE,
        detail="Change reviewer instructions",
    )
    await viewer._run_approval(pending)
    assert viewer._gate_task is None


@pytest.mark.asyncio
async def test_receipt_is_the_toolcard_diff_contract(stores):
    from local_operator.tui.widgets.tool_card import ToolCard

    session = _session(stores, gate=Gate(answer=False))
    try:
        _start_run(session, Message.user("Update reviewer instructions to use revised guidance"))
        result = await _agent(
            _context(session, stores),
            op="update",
            name="reviewer",
            instructions=SEED_INSTRUCTIONS.replace("Be terse.", "Demand tests."),
        )
        card = ToolCard("tc", "agent", {"op": "update", "name": "reviewer"})
        card._absorb_result(_text(result), result.details)
        assert card._diff and "+Demand tests." in card._diff
        assert "instructions: +1/-1 lines" in _text(result)
        assert any(line.startswith("Revert with agent(") for line in card._output)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_serving_preserves_remote_origin_through_prompt_queue_and_steer(tmp_path):
    from tests.unit.session.runtime.test_serving import make_handle

    directory = tmp_path / "sessions" / "remote-origin"
    directory.mkdir(parents=True)
    handle, session = make_handle(directory=directory)
    seen = []

    async def prompt(text, images=None, *, message_id=None, admitted=None, owner_typed=True):
        seen.append(("prompt", owner_typed))
        if admitted is not None:
            admitted.set_result(None)

    def steer(text, images=None, *, message_id=None, owner_typed=True):
        seen.append(("steer", owner_typed))

    session.prompt = prompt
    session.steer = steer
    try:
        await handle.prompt("update reviewer", owner_typed=False, wait_complete=True)
        await handle.steer("update reviewer", owner_typed=False)
        assert seen == [("prompt", False), ("steer", False)]
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_text",
    [
        "Ask the reviewer to look at PR12",
        "What does reviewer do?",
        "Please do not change reviewer",
        "Tell me how to change reviewer instructions",
    ],
)
async def test_profile_mentions_are_not_definition_edit_consent(stores, owner_text):
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        _start_run(session, Message.user(owner_text), TAINT)
        before = _snapshot(stores)
        result = await _agent(
            _context(session, stores),
            op="update",
            name="reviewer",
            instructions="Must not land on mention alone.",
        )
        _assert_proposed(result, gate)
        assert _snapshot(stores) == before
    finally:
        await session.dispose()


# -- the seam's own failure classes (incident, 2026-11-08) --------------------


def _intent(**overrides: Any) -> DurableWriteIntent:
    """The minimal intent the seam is asked about, for the reason tests."""
    fields: dict[str, Any] = dict(
        kind="team", operation="update", name="lopdev", changes=("instructions",), summary=""
    )
    fields.update(overrides)
    return DurableWriteIntent(**fields)


class _SeamContext:
    """A ``ToolContext`` whose consent seam is exactly this behavior."""

    def __init__(self, behavior) -> None:
        if behavior is not None:
            self.authorize_durable_write = behavior


@pytest.mark.asyncio
async def test_nested_eval_tool_call_id_files_an_approval_instead_of_failing(
    stores, caplog
) -> None:
    """A nested ``eval`` call id contains ``:``; filing must survive that.

    THE INCIDENT: ``harness/loop.py``'s ``dispatch_tool`` bridge mints a nested
    call id ``"<outer>:<uuid4 hex>"``, so ``tool("team", op="update", ...)``
    from a Python cell handed ``profile_approval.build`` an id ``Token`` cannot
    carry. ``model_validate`` raised ``ValidationError``, the seam's blanket
    ``except`` turned it into ``error``, and the run was told "the confirmation
    channel failed before the user could answer" — a save that was impossible
    from inside ``eval`` and said nothing about it.
    """
    import logging
    import uuid

    caplog.set_level(logging.INFO, logger="local_operator.profile_approval")
    session = _session(stores, gate=None)
    try:
        _start_run(session, TAINT)
        nested_id = f"call_00_ET_eBN6EjMAI7AMruqfWNYv2438:{uuid.uuid4().hex}"
        result = await run_team(
            nested_id,
            {"op": "update", "name": "lopdev", "instructions": "Ship carefully."},
            None,
            None,
            _context(session, stores),
        )
        assert not result.is_error, _text(result)
        assert result.details is not None and result.details["status"] == "proposed"
        record = _proposal_record(stores, result, root=stores["teams"].config_dir)
        # Proposals retain the host call ID verbatim: no mesh Token grammar
        # and no operator anchor are needed to file a non-authorizing suggestion.
        assert record["origin"]["tool_call_id"] == nested_id
        assert stores["teams"].get_team_by_name("lopdev").instructions == "Ship."
    finally:
        await session.dispose()


def test_request_token_shape_matches_its_documented_guarantee() -> None:
    """The docstring's claim is MEASURED here, not decorative (round 1, nit).

    The docs promised a wider guarantee than the code delivers, so the numbers
    they now state are pinned: a ``Token``-valid id is BYTE-IDENTICAL, anything
    else is a sanitised prefix of at most 143 bytes plus a separator plus 16 hex
    of the raw value's sha256 (160 bytes at the truncation boundary, shorter
    below it and never padded), every output is itself ``Token``-valid and
    idempotent, and inputs differing only in a forbidden character stay apart.
    """
    from local_operator.profile_approval import request_token

    # Token-valid input: byte-identical, so no receipt, request id or journal
    # row of an existing filing changes.
    assert request_token("call_00_ET_abc-def") == "call_00_ET_abc-def"

    # At the truncation boundary: exactly 143 prefix bytes + "-" + 16 hex.
    long_token = request_token("x" * 400)
    assert len(long_token) == 160
    assert long_token[:143] == "x" * 143
    assert long_token[143] == "-"
    assert long_token[144:] == hashlib.sha256(b"x" * 400).hexdigest()[:16]

    # Below it: prefix + separator + digest, never padded to 160.
    short_token = request_token("abc:def")
    assert short_token == "abc-def-" + hashlib.sha256(b"abc:def").hexdigest()[:16]

    # Differing only in a character the pattern forbids still separates them.
    assert request_token("outer:abc") != request_token("outer;abc")

    # Every output is Token-valid (so it can never fail the record's field) and
    # the map is idempotent, so re-filing the same id is stable.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    for raw in ("", "-", "x" * 161, "\u00e9:\u00fc", "abc.def", "outer:abc"):
        token = request_token(raw)
        assert token and len(token) <= 160
        assert set(token) <= allowed
        assert request_token(token) == token


@pytest.mark.asyncio
async def test_team_wording_edit_from_the_owner_message_saves_with_no_approval_surface(
    stores,
) -> None:
    """PATH A FOR TEAMS, in the incident's condition: NO approval surface at all.

    The owner's own typed message names the team and asks for the definition
    field, so the save needs no card — and it records who asked plus a backup,
    never a silent success.
    """
    session = _session(stores, gate=None)
    try:
        _start_run(session, Message.user("Update lopdev instructions to ship carefully."))
        result = await run_team(
            "tc-team-a",
            {"op": "update", "name": "lopdev", "instructions": "Ship carefully."},
            None,
            None,
            _context(session, stores),
        )
        assert not result.is_error, _text(result)
        assert result.details is not None
        authorized_by = result.details["authorized_by"]
        assert authorized_by.startswith("local-agent:")
        assert authorized_by.endswith(f"@{int(session._run_started_at * 1000)}")
        backup_id = result.details["backup_id"]
        assert backup_id
        assert stores["teams"].get_team_by_name("lopdev").instructions == "Ship carefully."
        backups = list((Path(stores["session_dir"])).rglob(f"*{backup_id}*"))
        assert backups, "a granted save must leave a backup on disk"
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"members": ["coder", "qa-tester"]},
        {"members": ["coder:2"]},
    ],
)
async def test_owner_team_roster_and_pin_changes_apply_directly(stores, args) -> None:
    """A name mention is not permission to change the roster or its pins.

    The owner message names the team and asks for a roster change; only WORDING
    fields may take the no-card path, so this one is filed for the card and the
    roster is untouched.
    """
    session = _session(stores, gate=None)
    try:
        _start_run(session, Message.user("Update lopdev to add a coder and a qa-tester."))
        before = _team_roster(stores)
        result = await run_team(
            "tc-team-roster",
            {"op": "update", "name": "lopdev", **args},
            None,
            None,
            _context(session, stores),
        )
        assert not result.is_error, _text(result)
        assert result.details is not None and result.details.get("status") != "proposed"
        assert _team_roster(stores) != before
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_seam_reason_classes_are_named_and_keep_their_sentences(caplog) -> None:
    """All four reasons, their exact sentences, and the fault an ``error`` names.

    ``unavailable`` (nobody can be asked) must never read as ``error`` (the run
    broke) and vice versa: the incident's confusion was a fault wearing the
    "nobody to ask" costume.
    """
    import logging

    from local_operator.network.types import MeshRefusal

    caplog.set_level(logging.WARNING, logger="local_operator.profile_scope")
    intent = _intent()

    origin, reason = await durable_write_decision(_SeamContext(None), intent)
    assert origin is None and reason == "no_seam"
    assert durable_write_refusal(intent, reason) == durable_write_refusal(intent, "no_seam")
    assert "no way to ask" in durable_write_refusal(intent, reason)

    async def unavailable(_intent):
        raise ApprovalUnavailableError(ACCOUNT_PROFILE_GATE, "no attached surface")

    origin, reason = await durable_write_decision(_SeamContext(unavailable), intent)
    assert origin is None and reason == "unavailable"
    assert "no one is attached to approve it" in durable_write_refusal(intent, reason)

    async def anchorless(_intent):
        raise MeshRefusal("approval_anchor_unavailable", "set up local operator authority")

    origin, reason = await durable_write_decision(_SeamContext(anchorless), intent)
    assert origin is None and reason == "error"
    assert "approval_anchor_unavailable" in durable_write_refusal(intent, reason)

    async def no(_intent):
        return None

    origin, reason = await durable_write_decision(_SeamContext(no), intent)
    assert origin is None and reason == "declined"
    assert durable_write_refusal(intent, reason).startswith(
        "the user declined to save this team for every conversation"
    )

    async def boom(_intent):
        raise RuntimeError("seam exploded")

    origin, reason = await durable_write_decision(_SeamContext(boom), intent)
    assert origin is None and reason == "error"
    sentence = durable_write_refusal(intent, reason)
    assert sentence.startswith("the confirmation channel failed before the user could answer")
    assert "RuntimeError" in sentence, "the fault type must be legible in the refusal"
    assert any(
        "durable-write consent seam failed: RuntimeError" in record.getMessage()
        for record in caplog.records
    ), "the fault type must also be in the log line"
    assert any(record.exc_info for record in caplog.records), "the traceback must stay logged"


@pytest.mark.asyncio
async def test_direct_save_card_counts_changes_then_shows_clipped_content(stores):
    """L1/R2-2/R3-1 (PR #3777): the non-proposal save card appended
    ``intent.summary`` verbatim, so a direct save of a long brief or skill file
    filled the ask card. It keeps its first line, then the names of what
    changes (no counts: the intent has no structured before/after), then —
    because a direct save has no review page — the content clipped to ~4 KB
    with a ``… N more lines not shown`` tail."""
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    body = "\n".join(f"+brief line {i} MARKER-{i}" for i in range(400))
    try:
        _start_run(session, Message.user("look at the reviewer"))
        origin = await session._authorize_account_profile_write(
            DurableWriteIntent(
                kind="agent",
                operation="update",
                name="reviewer",
                changes=("instructions", "effort"),
                summary=f"instructions:\n--- before\n+++ after\n{body}\nmodel/effort: a -> b",
            )
        )
        assert origin is None and len(gate.cards) == 1
        card = gate.cards[0][1]
        lines = card.splitlines()
        assert lines[:3] == [
            "Save agent 'reviewer' in your account (update).",
            "Changes: instructions, model/effort.",
            "",
        ], card
        assert lines[3:6] == ["instructions:", "--- before", "+++ after"]
        assert "+brief line 0 MARKER-0" in lines
        assert "MARKER-399" not in card, "the tail past the cap is not shown"
        assert re.fullmatch(r"… \d+ more lines not shown", lines[-1]), lines[-1]
        shown = len(lines) - 4  # head, changes, blank, tail
        assert int(lines[-1].split()[1]) == 404 - shown, "tail counts the hidden lines"
        assert len(card) <= 200 + PP.SAVE_DETAIL_MAX_CHARS
    finally:
        await session.dispose()


async def _direct_save_card(stores, intent: DurableWriteIntent) -> list[str]:
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        _start_run(session, Message.user("save it"))
        assert await session._authorize_account_profile_write(intent) is None
        assert len(gate.cards) == 1
        return gate.cards[0][1].splitlines()
    finally:
        await session.dispose()


#: R3-1: no ``+N −M`` anywhere on line 2 — prose-derived counts were fabricated.
_COUNTS = re.compile(r"[+−-]\d+")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "line2", "excerpt"),
    [
        pytest.param(
            # A whole new skill file: YAML front matter, ``- `` bullets and a
            # ``→`` that the R2-3 arrow split read as old/new halves.
            DurableWriteIntent(
                kind="skill",
                operation="create",
                name="triage",
                changes=("SKILL.md", "refs/steps.md"),
                summary="SKILL.md:\n---\nname: triage\n---\n- one\n- two\n"
                "Open Agents → Reviewer.\nrefs/steps.md:\n- a",
            ),
            "Changes: SKILL.md, refs/steps.md.",
            ["SKILL.md:", "- one", "Open Agents → Reviewer.", "refs/steps.md:", "- a"],
            id="bulleted-new-skill-file",
        ),
        pytest.param(
            DurableWriteIntent(
                kind="agent",
                operation="update",
                name="reviewer",
                changes=("instructions",),
                summary="instructions (replaced; new text):\nSee Settings → Models.\n- a\n- b",
            ),
            "Changes: instructions.",
            ["instructions (replaced; new text):", "See Settings → Models.", "- a", "- b"],
            id="replaced-instructions-with-arrow-and-bullets",
        ),
        pytest.param(
            DurableWriteIntent(
                kind="agent",
                operation="update",
                name="reviewer",
                changes=("instructions", "effort"),
                summary="instructions:\n--- before\n+++ after\n@@ -1,2 +1,2 @@\n keep\n"
                "-old\n+new\nmodel/effort: a -> b",
            ),
            "Changes: instructions, model/effort.",
            ["--- before", "+++ after", "@@ -1,2 +1,2 @@", "-old", "+new", "model/effort: a -> b"],
            id="unified-diff",
        ),
    ],
)
async def test_direct_save_card_lists_changes_without_fabricated_counts(
    stores, intent, line2, excerpt
):
    """R3-1 (PR #3777): line 2 names the fields/paths only — never counts parsed
    from the prose summary — and the content excerpt follows a blank line."""
    lines = await _direct_save_card(stores, intent)
    assert lines[1] == line2, lines
    assert not _COUNTS.search(lines[1]), lines[1]
    assert lines[2] == ""
    assert all(line in lines[3:] for line in excerpt), lines
