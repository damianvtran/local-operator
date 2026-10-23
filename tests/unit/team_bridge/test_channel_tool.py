"""The bridge tool's channel ops: identity from the runtime, never arguments.

The wire names themselves are pinned in ``test_client.py``; these tests pin the
TOOL's semantics — the creator member the server requires, the per-call
``client_ref`` a 401 retry reuses, the exclusive history bounds, and the refusal
to act as a human when the runtime has no agent endpoint.
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.tool import (
    BRIDGE_OPS,
    CHANNEL_OPS,
    BridgeRuntime,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import FakeStore, _ok

TENANT = "ten_1"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about channel semantics, not the fences.

    The fence and admission state machines have their own files; patching them
    open keeps a store-less runtime readable here (the same shape
    ``test_h5d_tool.py`` uses).
    """

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _owner(endpoint_id: str = "bep_1") -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id=TENANT,
        endpoint_id=endpoint_id,
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


class RecordingClient:
    """Captures the exact call the tool makes; returns scripted answers."""

    def __init__(
        self,
        *,
        conversations: dict[str, Any] | None = None,
        created: dict[str, Any] | None = None,
        sent: dict[str, Any] | None = None,
        messages: dict[str, Any] | None = None,
    ) -> None:
        self.conversations_result = conversations or {"conversations": [], "next_cursor": None}
        self.created_result = created or {
            "id": "bcv_newchat1",
            "kind": "channel",
            "member_count": 1,
        }
        self.sent_result = sent or {"message": {"id": "cmsg_1", "seq": 4}}
        self.messages_result = messages or {"messages": []}
        self.calls: list[dict[str, Any]] = []

    async def conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "conversations", **kwargs})
        return self.conversations_result

    async def create_conversation(
        self,
        body: Any,
        *,
        actor_endpoint_id: str | None = None,
        origin_session_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "create_conversation",
                "body": dict(body),
                "actor_endpoint_id": actor_endpoint_id,
                "origin_session_id": origin_session_id,
            }
        )
        return self.created_result

    async def conversation_send(
        self,
        conversation_id: str,
        *,
        client_ref: str,
        text: str,
        tenant_id: str | None = None,
        agent_label: str | None = None,
        actor_endpoint_id: str | None = None,
        origin_session_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "conversation_send",
                "conversation_id": conversation_id,
                "client_ref": client_ref,
                "text": text,
                "tenant_id": tenant_id,
                "agent_label": agent_label,
                "actor_endpoint_id": actor_endpoint_id,
                "origin_session_id": origin_session_id,
            }
        )
        return self.sent_result

    async def conversation_react(self, conversation_id, message_id, emoji, **kwargs):
        self.calls.append(
            {
                "method": "conversation_react",
                "conversation_id": conversation_id,
                "message_id": message_id,
                "emoji": emoji,
                **kwargs,
            }
        )
        return {"message_id": message_id, "reactions": []}

    async def conversation_messages(
        self,
        conversation_id: str,
        *,
        tenant_id: str | None = None,
        limit: int | None = None,
        before_seq: int | None = None,
        after_seq: int | None = None,
        endpoint_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "conversation_messages",
                "conversation_id": conversation_id,
                "tenant_id": tenant_id,
                "limit": limit,
                "before_seq": before_seq,
                "after_seq": after_seq,
                "endpoint_id": endpoint_id,
            }
        )
        return self.messages_result


def _runtime(client: Any, *, endpoint_id: str = "bep_1") -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, client),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(endpoint_id),
        session_id="sess_1",
    )


def test_the_schema_names_the_channel_ops_and_never_decide() -> None:
    runtime = _runtime(RecordingClient())
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None

    schema = json.dumps(tool.parameters)
    op_enum = tool.parameters["properties"]["op"]["enum"]
    assert set(CHANNEL_OPS) <= set(op_enum)
    assert "decide" not in op_enum
    assert set(CHANNEL_OPS) <= set(BRIDGE_OPS)
    assert tool.parameters["additionalProperties"] is False
    # The model cannot name an actor: identity is runtime state only.
    assert "actor_endpoint_id" not in schema
    assert "origin_session_id" not in schema


@pytest.mark.asyncio
async def test_channels_lists_with_the_tenant_and_returns_discovery_rows() -> None:
    row = {
        "id": "conv_2",
        "kind": "channel",
        "discoverable": True,
        "member": False,
        "member_count": 3,
    }
    client = RecordingClient(conversations={"conversations": [row], "next_cursor": "bcur_2"})
    outcome = await execute_bridge(_runtime(client), {"op": "channels", "limit": 10})

    assert outcome["ok"] is True
    page = json.loads(outcome["content"])
    assert page["conversations"] == [row]
    assert page["next_cursor"] == "bcur_2"
    assert client.calls == [
        {"method": "conversations", "tenant_id": TENANT, "limit": 10, "cursor": None}
    ]


@pytest.mark.asyncio
async def test_channel_create_adds_the_agent_and_not_the_human_owner() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(
        _runtime(client),
        {
            "op": "channel_create",
            "kind": "channel",
            "name": "build",
            "topic": "CI signals",
            "visibility": "tenant",
            "members": [
                {"account_id": "acc_2", "kind": "human", "display_label": "Dana"},
            ],
        },
    )

    assert outcome["ok"] is True
    call = client.calls[0]
    assert call["method"] == "create_conversation"
    assert call["actor_endpoint_id"] == "bep_1"
    assert call["origin_session_id"] == "sess_1"
    assert call["body"] == {
        "tenant_id": TENANT,
        "kind": "channel",
        "name": "build",
        "topic": "CI signals",
        "visibility": "tenant",
        "members": [
            {"account_id": "acc_2", "kind": "human", "display_label": "Dana"},
            {
                "account_id": "acc_1",
                "kind": "agent",
                "endpoint_id": "bep_1",
                "display_label": "agent",
            },
        ],
    }
    # The human owner is NOT a member unless explicitly listed.
    assert all(
        not (m["account_id"] == "acc_1" and m["kind"] == "human") for m in call["body"]["members"]
    )


@pytest.mark.asyncio
async def test_channel_create_keeps_an_explicit_creator_entry_once() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(
        _runtime(client),
        {
            "op": "channel_create",
            "kind": "group",
            "members": [
                {
                    "account_id": "acc_1",
                    "kind": "agent",
                    "endpoint_id": "bep_1",
                    "display_label": "Builder",
                },
                {"account_id": "acc_2", "kind": "human", "display_label": "Dana"},
            ],
        },
    )

    assert outcome["ok"] is True
    members = client.calls[0]["body"]["members"]
    creators = [
        m for m in members if m["account_id"] == "acc_1" and m.get("endpoint_id") == "bep_1"
    ]
    assert creators == [
        {"account_id": "acc_1", "kind": "agent", "endpoint_id": "bep_1", "display_label": "Builder"}
    ]
    assert len(members) == 2


@pytest.mark.asyncio
async def test_channel_post_mints_one_client_ref_and_a_retry_reuses_it() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(401, json={"error": "unauthorized"})
        return _ok({"message": {"id": f"cmsg_{len(requests)}", "seq": len(requests)}})

    runtime = _runtime(
        TeamBridgeClient(
            None,
            base_url="https://hub.example/v1",
            store=FakeStore(refresh={"access": "token-2"}),
            transport=httpx.MockTransport(handler),
        )
    )
    first = await execute_bridge(
        runtime, {"op": "channel_post", "channel_id": "bcv_abcd1234", "body": "hello"}
    )
    assert first["ok"] is True

    # The 401 refresh retried ONCE with the same body: one client_ref.
    assert len(requests) == 2
    first_body = json.loads(requests[0].content)
    retried_body = json.loads(requests[1].content)
    assert first_body["client_ref"]
    assert retried_body["client_ref"] == first_body["client_ref"]
    assert retried_body["actor_endpoint_id"] == "bep_1"
    assert retried_body["origin_session_id"] == "sess_1"
    assert retried_body["tenant_id"] == TENANT

    second = await execute_bridge(
        runtime, {"op": "channel_post", "channel_id": "bcv_abcd1234", "body": "hello again"}
    )
    assert second["ok"] is True
    # A second CALL is a second message, so it gets a fresh idempotency key.
    assert json.loads(requests[2].content)["client_ref"] != first_body["client_ref"]


@pytest.mark.asyncio
async def test_channel_history_refuses_both_bounds_together() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(
        _runtime(client),
        {
            "op": "channel_history",
            "channel_id": "bcv_abcd1234",
            "before_seq": 10,
            "after_seq": 5,
        },
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert client.calls == []


@pytest.mark.asyncio
async def test_channel_history_reads_as_the_agent_endpoint() -> None:
    client = RecordingClient(
        messages={
            "messages": [
                {"id": "cmsg_1", "seq": 6, "sender": {"kind": "agent"}, "body": None},
            ]
        }
    )
    outcome = await execute_bridge(
        _runtime(client),
        {"op": "channel_history", "channel_id": "bcv_abcd1234", "limit": 20, "after_seq": 5},
    )

    assert outcome["ok"] is True
    assert outcome["content"].startswith(BRIDGE_UNTRUSTED_CLAUSE)
    assert client.calls == [
        {
            "method": "conversation_messages",
            "conversation_id": "bcv_abcd1234",
            "tenant_id": TENANT,
            "limit": 20,
            "before_seq": None,
            "after_seq": 5,
            "endpoint_id": "bep_1",
        }
    ]
    page = json.loads(outcome["content"][len(BRIDGE_UNTRUSTED_CLAUSE) + 2 :])
    assert page["messages"][0]["body"] is None


@pytest.mark.asyncio
async def test_channel_history_limit_is_bounded() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(
        _runtime(client),
        {"op": "channel_history", "channel_id": "bcv_abcd1234", "limit": 101},
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op", CHANNEL_OPS)
async def test_every_channel_op_refuses_without_an_owner_endpoint(op: str) -> None:
    client = RecordingClient()
    outcome = await execute_bridge(_runtime(client, endpoint_id=""), {"op": op})
    assert outcome["ok"] is False
    assert outcome["code"] == "no_agent_endpoint"
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "extra"),
    [
        ("channel_post", {"body": "hello"}),
        ("channel_history", {"limit": 20}),
    ],
)
@pytest.mark.parametrize(
    "channel_id",
    [
        "",
        "conv_1",
        "../../status",
        "bcv_abcd1234#x",
        "bcv_abcd1234/messages",
    ],
)
async def test_a_malformed_channel_id_never_reaches_the_client(
    op: str, extra: dict[str, Any], channel_id: str
) -> None:
    """The id is interpolated into the upstream path; the bcv_ grammar is the gate."""

    client = RecordingClient()
    outcome = await execute_bridge(_runtime(client), {"op": op, "channel_id": channel_id, **extra})
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "member",
    [
        {"account_id": "acc_2", "kind": "human", "endpoint_id": "bep_2"},
        {"account_id": "acc_2", "kind": "agent"},
        {"account_id": "acc_2", "kind": "agent", "endpoint_id": "   "},
    ],
)
async def test_member_shapes_that_could_collide_with_the_actor_are_refused(
    member: dict[str, str],
) -> None:
    """A human endpoint would win the creator key; an agent without one is dead."""

    client = RecordingClient()
    outcome = await execute_bridge(
        _runtime(client), {"op": "channel_create", "kind": "channel", "members": [member]}
    )
    assert outcome["ok"] is False and outcome["code"] == "invalid_request"
    assert client.calls == []


@pytest.mark.asyncio
async def test_decide_keeps_its_own_sentence_before_the_endpoint_gate() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(_runtime(client, endpoint_id=""), {"op": "decide"})
    assert outcome["ok"] is False and outcome["code"] == "not_permitted"
    assert "ask queue" in outcome["error"]
    assert client.calls == []

    unknown = await execute_bridge(_runtime(client, endpoint_id=""), {"op": "invented"})
    assert unknown["ok"] is False and unknown["code"] == "invalid_request"
    assert "unknown bridge op" in unknown["error"]
    assert client.calls == []


@pytest.mark.asyncio
async def test_a_whitespace_human_endpoint_is_absent_not_wired() -> None:
    """The validators strip; the serializer must strip too, or "   " rides."""

    client = RecordingClient()
    outcome = await execute_bridge(
        _runtime(client),
        {
            "op": "channel_create",
            "kind": "channel",
            "members": [
                {
                    "account_id": "acc_2",
                    "kind": "human",
                    "endpoint_id": "   ",
                    "display_label": "Dana",
                }
            ],
        },
    )
    assert outcome["ok"] is True
    members = client.calls[0]["body"]["members"]
    human = next(member for member in members if member["account_id"] == "acc_2")
    assert "endpoint_id" not in human


@pytest.mark.asyncio
async def test_react_is_an_agent_write_with_runtime_identity():
    client = RecordingClient()
    runtime = _runtime(client)
    result = await execute_bridge(
        runtime,
        {
            "op": "react",
            "channel_id": "bcv_abcd1234",
            "message_id": "bcmsg_abcd1234",
            "emoji": "✅",
        },
    )
    assert result["ok"] is True
    assert BRIDGE_UNTRUSTED_CLAUSE in result["content"]
    assert client.calls == [
        {
            "method": "conversation_react",
            "conversation_id": "bcv_abcd1234",
            "message_id": "bcmsg_abcd1234",
            "emoji": "✅",
            "active": True,
            "tenant_id": TENANT,
            "actor_endpoint_id": "bep_1",
        }
    ]
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"op": "react"}) == "write"
    result = await execute_bridge(
        runtime,
        {
            "op": "react",
            "channel_id": "bcv_abcd1234",
            "message_id": "bcmsg_abcd1234",
            "emoji": "✅",
            "remove": True,
        },
    )
    assert result["ok"] is True
    assert client.calls[-1]["active"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id,emoji", [("../other", "✅"), ("bcmsg_abcd1234", "😀" * 33)])
async def test_react_bad_args_never_dispatch(message_id, emoji):
    client = RecordingClient()
    result = await execute_bridge(
        _runtime(client),
        {"op": "react", "channel_id": "bcv_abcd1234", "message_id": message_id, "emoji": emoji},
    )
    assert result["ok"] is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_react_paused_admission_refuses_before_write(monkeypatch):
    client = RecordingClient()
    reads = []

    def refuse(runtime, *, is_read=False):
        reads.append(is_read)
        return {"code": "bridge_paused", "error": "paused"}

    monkeypatch.setattr(host_mod, "admission_refusal", refuse)
    monkeypatch.setattr(host_mod, "admit_soon", lambda runtime: None)
    result = await execute_bridge(
        _runtime(client),
        {
            "op": "react",
            "channel_id": "bcv_abcd1234",
            "message_id": "bcmsg_abcd1234",
            "emoji": "✅",
        },
    )
    assert result["code"] == "bridge_paused"
    assert reads == [False], "react never uses the read-shaped admission exception"
    assert client.calls == []
