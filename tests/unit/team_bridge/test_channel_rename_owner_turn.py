"""The bridge tool's ``channel_rename``: the owner's rename, on the owner's own word.

Operator decision 2026-11-21 (option a): a chief of staff may rename a channel or
group when its operator asks, AS THE HUMAN OWNER through
``TeamBridgeClient.conversation_rename`` with NO ``actor_endpoint_id`` (the hub v1
refuses agent renames), and ONLY when the operator's own typed turn in the current
session asked for it. Every attempt is audited locally.

These tests drive a REAL ``Session`` (its ``_note_run_input`` classification and
``owner_turn_witness``) and the REAL ``TeamBridgeClient`` against an
``httpx.MockTransport`` stub hub, so "exactly one rename POST with no
``actor_endpoint_id``" is read off the wire, not off a mock's call record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY
from local_operator.harness import owner_turn
from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE
from local_operator.harness.message_types import (
    PEER_MESSAGE_MESSAGE_TYPE,
    TEAM_BRIDGE_MESSAGE_TYPE,
)
from local_operator.harness.types import CustomMessage, Message, ModelSpec, ToolContext
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.session.session import REMOTE_PROMPT_KEY, Session
from local_operator.session.transcript import Transcript
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.rename_audit import rename_audit_path
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.tool import (
    OWNER_CONFIRMATION_REQUIRED,
    BridgeRuntime,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import FakeStore, _err, _ok

TENANT = "ten_1"
CONV = "bcv_launch001"
NEW_NAME = "Launch Room"
MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Scratch config root: session_lineage reads <config>/sessions and
    # <config>/guardrails, never the operator's real state.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    # The fence and admission machines have their own suites; opened here so
    # these tests isolate the owner-turn gate (as test_channel_tool.py does).
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _never_stream(request, signal):  # pragma: no cover — no turn is run here
    raise AssertionError("these tests never stream")


def _session(tmp_path: Path, *, sid: str = "s_operator", parent: str = "") -> Session:
    directory = tmp_path / "config" / "sessions" / sid
    directory.mkdir(parents=True, exist_ok=True)
    return Session(
        model=MODEL,
        stream_fn=_never_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        owner_prompt_consent=True,
        parent_session_id=parent,
    )


def _start_run(session: Session, *inputs: Any) -> None:
    """Reset the run flags as the pipeline head does, then fold the inputs."""
    session._run_owner_texts = []
    session._run_owner_text_bindings = []
    session._run_owner_entry_ids = []
    session._run_owner_consent = False
    session._run_untrusted_input = False
    session._run_typed_user = False
    session._run_last_trigger = None
    session._run_triggers = set()
    for message in inputs:
        session._note_run_input(message)


class Hub:
    """The stub hub: records every request, answers from a script."""

    def __init__(self, answer: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.answer = answer or _ok({"id": CONV, "kind": "channel", "name": NEW_NAME})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.answer


def _runtime(tmp_path: Path, hub: Hub, *, session_id: str = "s_operator") -> BridgeRuntime:
    client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(hub),
    )
    return BridgeRuntime(
        client=client,
        config_dir=tmp_path / "config",
        tenant_id=TENANT,
        owner=OwnerSnapshot(
            credential_id=1,
            account_id="acc_1",
            tenant_id=TENANT,
            endpoint_id="bep_1",
            canonical_base_url="https://hub.example",
            enrollment_generation=1,
        ),
        session_id=session_id,
        admission="admitted",
    )


def _audit_rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = rename_audit_path(tmp_path / "config")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _args(**extra: Any) -> dict[str, Any]:
    return {"op": "channel_rename", "channel_id": CONV, "name": NEW_NAME, **extra}


# -- 1. the operator's own typed turn renames ----------------------------------


@pytest.mark.asyncio
async def test_operator_typed_request_renames_once_as_the_owner(tmp_path: Path) -> None:
    session = _session(tmp_path)
    typed = Message.user(f"please rename {CONV} to {NEW_NAME}")
    _start_run(session, typed)
    hub = Hub()

    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(), owner_turn=session.owner_turn_witness
    )

    assert outcome["ok"] is True, outcome
    assert outcome["state"] == "renamed"
    assert outcome["name"] == NEW_NAME
    # Exactly ONE hub call: the rename POST, on the human path.
    assert len(hub.requests) == 1
    request = hub.requests[0]
    assert request.method == "POST"
    assert request.url.path == f"/v1/messaging/conversations/{CONV}/rename"
    body = json.loads(request.content)
    assert body == {"name": NEW_NAME, "tenant_id": TENANT}
    assert "actor_endpoint_id" not in body
    assert "actor_endpoint_id" not in str(request.url)
    # Audited: authorized (before the POST) then renamed, both citing the turn.
    rows = _audit_rows(tmp_path)
    assert [row["outcome"] for row in rows] == ["authorized", "renamed"]
    for row in rows:
        assert row["conversation_id"] == CONV
        assert row["name"] == NEW_NAME
        assert row["authorized_by_entry_id"] == typed.id
        assert row["acted_as"] == "owner"
        assert row["session_id"] == "s_operator"
    assert rows[1]["canonical_name"] == NEW_NAME
    # Never the operator's words, never a token.
    raw = rename_audit_path(tmp_path / "config").read_text()
    assert "please rename" not in raw and "token-1" not in raw
    assert oct(rename_audit_path(tmp_path / "config").stat().st_mode & 0o777) == "0o600"


@pytest.mark.asyncio
async def test_the_registered_tool_reads_the_witness_from_its_context(tmp_path: Path) -> None:
    """The AgentTool path: ToolContext.owner_turn_witness reaches the executor."""
    session = _session(tmp_path)
    _start_run(session, Message.user(ASK))
    hub = Hub()
    runtime = _runtime(tmp_path, hub)
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    context = ToolContext(bridge_runtime=runtime, owner_turn_witness=session.owner_turn_witness)
    result = await tool.execute("call_1", _args(), None, None, context)
    assert result.is_error is False, result
    assert len(hub.requests) == 1

    # The same call with a context that carries no witness refuses, sends nothing.
    bare = await tool.execute("call_2", _args(), None, None, ToolContext(bridge_runtime=runtime))
    assert bare.is_error is True
    assert bare.details["code"] == OWNER_CONFIRMATION_REQUIRED
    assert len(hub.requests) == 1


def test_the_session_hands_its_witness_to_every_tool_context(tmp_path: Path) -> None:
    session = _session(tmp_path)
    context = session._build_tool_context()
    assert context.owner_turn_witness == session.owner_turn_witness


# -- 2. anything not typed by the operator in this session refuses -------------


def _peer(text: str) -> CustomMessage:
    return CustomMessage(
        custom_type=PEER_MESSAGE_MESSAGE_TYPE, attribution="user", details={"text": text}
    )


def _bridge(text: str) -> CustomMessage:
    return CustomMessage(
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE, attribution="user", details={"text": text}
    )


def _wake(text: str) -> CustomMessage:
    return CustomMessage(
        custom_type=WAKE_PROMPT_MESSAGE_TYPE, attribution="user", details={"text": text}
    )


def _job_result(text: str) -> CustomMessage:
    return CustomMessage(
        custom_type=JOB_RESULT_MESSAGE_TYPE, attribution="user", details={"text": text}
    )


def _remote(text: str) -> Message:
    message = Message.user(text)
    message.provider_payload = {REMOTE_PROMPT_KEY: True}
    return message


def _injected(text: str) -> Message:
    message = Message.user(text)
    message.provider_payload = {RENDERED_INJECTION_KEY: True}
    return message


ASK = f"rename {CONV} to {NEW_NAME}"


@pytest.mark.parametrize(
    ("label", "inputs", "reason"),
    [
        (
            "different name capitalization",
            lambda: [Message.user(ASK.lower())],
            owner_turn.REASON_NOT_NAMED,
        ),
        ("peer send message", lambda: [_peer(ASK)], owner_turn.REASON_NO_OWNER_TURN),
        ("team bridge inbound", lambda: [_bridge(ASK)], owner_turn.REASON_NO_OWNER_TURN),
        (
            "operator turn + peer folded in",
            lambda: [Message.user("hi, catch up on the bridge"), _peer(ASK)],
            owner_turn.REASON_UNTRUSTED_INPUT,
        ),
        (
            "subagent output (job result)",
            lambda: [_job_result(ASK)],
            owner_turn.REASON_NO_OWNER_TURN,
        ),
        ("scheduled wake", lambda: [_wake(ASK)], owner_turn.REASON_NO_OWNER_TURN),
        (
            "monitor",
            lambda: [
                CustomMessage(
                    custom_type="monitor_prompt", attribution="user", details={"text": ASK}
                )
            ],
            owner_turn.REASON_NO_OWNER_TURN,
        ),
        (
            "tool result",
            lambda: [Message(role="tool", content=Message.user(ASK).content)],
            owner_turn.REASON_NO_OWNER_TURN,
        ),
        (
            "wake after the operator's turn",
            lambda: [Message.user(ASK), _wake("check in")],
            owner_turn.REASON_LATER_TRIGGER,
        ),
        ("remote/relayed prompt", lambda: [_remote(ASK)], owner_turn.REASON_NO_OWNER_TURN),
        ("harness-injected row", lambda: [_injected(ASK)], owner_turn.REASON_NO_OWNER_TURN),
        (
            "operator never named the name",
            lambda: [Message.user("tidy up the channel names however you like")],
            owner_turn.REASON_NOT_NAMED,
        ),
        ("negated request", lambda: [Message.user(f"do not {ASK}")], owner_turn.REASON_NOT_NAMED),
        (
            "mere mention",
            lambda: [Message.user(f"we talked about {NEW_NAME}")],
            owner_turn.REASON_NOT_NAMED,
        ),
        (
            "quoted peer request",
            lambda: [Message.user(f"a peer wrote: {ASK}")],
            owner_turn.REASON_NOT_NAMED,
        ),
        ("question", lambda: [Message.user(f"should we {ASK}?")], owner_turn.REASON_NOT_NAMED),
        (
            "different room",
            lambda: [Message.user(f"rename bcv_different1 to {NEW_NAME}")],
            owner_turn.REASON_NOT_NAMED,
        ),
        (
            "different name",
            lambda: [Message.user(f"rename {CONV} to Other Room")],
            owner_turn.REASON_NOT_NAMED,
        ),
    ],
)
@pytest.mark.asyncio
async def test_non_operator_requests_refuse_with_zero_hub_calls(
    tmp_path: Path, label: str, inputs: Any, reason: str
) -> None:
    session = _session(tmp_path)
    _start_run(session, *inputs())
    hub = Hub()

    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(), owner_turn=session.owner_turn_witness
    )

    assert outcome["ok"] is False, label
    assert outcome["code"] == OWNER_CONFIRMATION_REQUIRED, label
    assert f"reason: {reason}" in outcome["error"], (label, outcome["error"])
    assert hub.requests == [], label
    rows = _audit_rows(tmp_path)
    assert len(rows) == 1, label
    assert rows[0]["outcome"] == OWNER_CONFIRMATION_REQUIRED
    assert rows[0]["reason"] == reason
    assert rows[0]["authorized_by_entry_id"] is None
    assert rows[0]["conversation_id"] == CONV


@pytest.mark.asyncio
async def test_a_subagent_refuses_even_on_its_parents_typed_words(tmp_path: Path) -> None:
    # A child's prompt is its parent's text: the typed row never makes it an owner.
    child = _session(tmp_path, sid="s_child", parent="s_operator")
    _start_run(child, Message.user(ASK))
    hub = Hub()
    outcome = await execute_bridge(
        _runtime(tmp_path, hub, session_id="s_child"), _args(), owner_turn=child.owner_turn_witness
    )
    assert outcome["code"] == OWNER_CONFIRMATION_REQUIRED
    assert "reason: subagent" in outcome["error"]
    assert hub.requests == []
    assert [row["reason"] for row in _audit_rows(tmp_path)] == ["subagent"]


@pytest.mark.asyncio
async def test_an_agent_opened_session_refuses_its_openers_brief(tmp_path: Path) -> None:
    # A workstream another agent opened: its first "user" row is that agent's
    # brief. The guard's own provenance (origin.json opened_by) says so.
    session = _session(tmp_path, sid="s_opened")
    origin = tmp_path / "config" / "sessions" / "s_opened" / "origin.json"
    origin.write_text(json.dumps({"opened_by": {"session": "s_manager"}}))
    _start_run(session, Message.user(ASK))
    hub = Hub()
    outcome = await execute_bridge(
        _runtime(tmp_path, hub, session_id="s_opened"),
        _args(),
        owner_turn=session.owner_turn_witness,
    )
    assert outcome["code"] == OWNER_CONFIRMATION_REQUIRED
    assert "reason: agent_opened" in outcome["error"]
    assert hub.requests == []


# -- 3. the hub's refusals surface unchanged -----------------------------------


@pytest.mark.parametrize(
    ("status", "code", "details"),
    [
        (409, "conversation_name_taken", {"conversation_id": "bcv_other123", "name": NEW_NAME}),
        (403, "conversation_not_permitted", None),
        (400, "invalid_request", {"field": "name", "rule": "too_long"}),
        (400, "invalid_request", {"field": "name", "rule": "control_character"}),
    ],
)
@pytest.mark.asyncio
async def test_hub_refusals_surface_unchanged(
    tmp_path: Path, status: int, code: str, details: dict[str, Any] | None
) -> None:
    session = _session(tmp_path)
    typed = Message.user(ASK)
    _start_run(session, typed)
    hub = Hub(_err(status, code, details=details))

    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(), owner_turn=session.owner_turn_witness
    )

    assert outcome["ok"] is False
    assert outcome["code"] == code
    assert outcome["http_status"] == status
    assert outcome.get("details") == (details or {})
    if details and "rule" in details:
        assert outcome["rule"] == details["rule"]
        assert f"rule={details['rule']}" in outcome["error"] or details["rule"] in outcome["error"]
    assert len(hub.requests) == 1  # one POST; a refusal is not retried
    rows = _audit_rows(tmp_path)
    assert [row["outcome"] for row in rows] == ["authorized", code]
    assert rows[1]["http_status"] == status
    assert rows[1]["authorized_by_entry_id"] == typed.id


# -- 4. a missing or invalid conversation id is refused locally ----------------


@pytest.mark.parametrize(
    ("channel_id", "reason"),
    [
        (None, "missing_conversation_id"),
        ("", "missing_conversation_id"),
        ("general", "invalid_conversation_id"),
        ("#launch", "invalid_conversation_id"),
        ("bcv_UPPER!!", "invalid_conversation_id"),
    ],
)
@pytest.mark.asyncio
async def test_a_missing_or_invalid_conversation_id_is_refused_locally(
    tmp_path: Path, channel_id: str | None, reason: str
) -> None:
    session = _session(tmp_path)
    _start_run(session, Message.user(ASK))
    hub = Hub()
    args = _args()
    if channel_id is None:
        args.pop("channel_id")
    else:
        args["channel_id"] = channel_id

    outcome = await execute_bridge(
        _runtime(tmp_path, hub), args, owner_turn=session.owner_turn_witness
    )

    assert outcome["ok"] is False
    assert outcome["code"] == "invalid_request"
    assert hub.requests == []
    rows = _audit_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["reason"] == reason
    # A malformed id is free model text: never written to the audit.
    assert rows[0]["conversation_id"] is None


@pytest.mark.asyncio
async def test_a_credential_shaped_name_is_refused_and_withheld_from_the_audit(
    tmp_path: Path,
) -> None:
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    session = _session(tmp_path)
    _start_run(session, Message.user(f"rename {CONV} to {token}"))
    hub = Hub()
    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(name=token), owner_turn=session.owner_turn_witness
    )
    assert outcome["ok"] is False
    assert hub.requests == []
    raw = rename_audit_path(tmp_path / "config").read_text()
    assert token not in raw
    assert _audit_rows(tmp_path)[0]["reason"] == "credential_shaped_name"


@pytest.mark.asyncio
async def test_an_unwritable_audit_refuses_before_any_hub_call(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start_run(session, Message.user(ASK))
    # A FILE where the team_bridge directory must be: the audit cannot be written.
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "team_bridge").write_text("not a directory")
    hub = Hub()
    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(), owner_turn=session.owner_turn_witness
    )
    assert outcome["ok"] is False
    assert outcome["code"] == "audit_unavailable"
    assert hub.requests == []


@pytest.mark.asyncio
async def test_an_earlier_turn_never_authorizes_a_later_wake(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start_run(session, Message.user(ASK))
    assert session.owner_turn_witness(ASK).ok
    _start_run(session, _wake(ASK))
    hub = Hub()
    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(), owner_turn=session.owner_turn_witness
    )
    assert outcome["code"] == OWNER_CONFIRMATION_REQUIRED
    assert hub.requests == []
    assert _audit_rows(tmp_path)[0]["authorized_by_entry_id"] is None


@pytest.mark.asyncio
async def test_unknown_model_confirmation_fields_cannot_grant_authority(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start_run(session, Message.user(ASK))
    hub = Hub()
    outcome = await execute_bridge(
        _runtime(tmp_path, hub), _args(owner_confirmed=True), owner_turn=session.owner_turn_witness
    )
    assert outcome["code"] == "invalid_request"
    assert hub.requests == []


@pytest.mark.asyncio
async def test_a_fence_refusal_is_audited_and_never_calls_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        host_mod,
        "owner_fence_refusal",
        lambda runtime: {"code": "not_designated", "error": "not designated"},
    )
    hub = Hub()
    outcome = await execute_bridge(_runtime(tmp_path, hub), _args())
    assert outcome["code"] == "not_designated"
    assert hub.requests == []
    assert _audit_rows(tmp_path)[0]["outcome"] == "not_designated"
