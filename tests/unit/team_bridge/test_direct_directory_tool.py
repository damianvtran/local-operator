"""Directory metadata and direct-message receipts must not imply agent execution.

The local owner fence has independent coverage; keep admission real here so a
read-only directory cannot accidentally make DM a read-shaped mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_operator.aida import state as aida_state
from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import authority as authority_mod
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import BridgeRefusal
from local_operator.team_bridge.tool import (
    DIRECTORY_MAX,
    BridgeRuntime,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _runtime


class DirectoryClient(RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.directory_result: dict[str, Any] = {
            "members": [
                {
                    "account_id": "acc_1",
                    "display_label": "Owner",
                    "role": "admin",
                    "bridge_enabled": True,
                    "bridge_state": "active",
                    "agent": {"endpoint_id": "bep_1", "name": "My CoS", "state": "active"},
                },
                {
                    "account_id": "acc_2",
                    "display_label": "Dana",
                    "role": "member",
                    "bridge_enabled": True,
                    "bridge_state": "active",
                    "agent": {"endpoint_id": "bep_2", "name": "Dana's CoS", "state": "active"},
                },
            ]
        }
        self.created_result = {
            "id": "bcv_direct01",
            "kind": "dm",
            "members": [
                {"account_id": "acc_1", "kind": "agent", "endpoint_id": "bep_1", "state": "active"},
                {"account_id": "acc_2", "kind": "agent", "endpoint_id": "bep_2", "state": "active"},
            ],
        }

    async def directory(
        self, tenant_id: str, *, expected_account_id: str | None = None
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "directory",
                "tenant_id": tenant_id,
                "expected_account_id": expected_account_id,
            }
        )
        return self.directory_result


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BridgeRuntime:
    # Outbound ops take the relaxed fence (designation excluded); shared reads
    # take the strict one. Both are fenced open here — the fence has its own
    # coverage in test_outbound_session_scope.py / test_h5d_tool.py.
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    # A denied call may schedule admission, but this unit must never start a
    # background status request against a real account or keep a task alive.
    monkeypatch.setattr(host_mod, "admit_soon", lambda runtime: None)
    value = _runtime(DirectoryClient())
    value.config_dir = tmp_path
    value.admission = "admitted"
    return value


def _page(outcome: dict[str, Any]) -> dict[str, Any]:
    assert outcome["ok"] is True
    assert outcome["content"].startswith(BRIDGE_UNTRUSTED_CLAUSE + "\n\n")
    return json.loads(outcome["content"][len(BRIDGE_UNTRUSTED_CLAUSE) + 2 :])


@pytest.mark.asyncio
async def test_directory_is_a_whitelist_not_a_profile_or_body_dump(runtime):
    client = runtime.client
    client.directory_result["body"] = "TOP_LEVEL_BODY_CANARY"
    for member in client.directory_result["members"]:
        member.update(
            body="MEMBER_BODY_CANARY",
            profile={"email": "PROFILE_CANARY", "bio": "BIO_CANARY"},
            avatar_url="https://private.example/AVATAR_CANARY",
            future_field="FUTURE_CANARY",
        )
        member["agent"].update(body="AGENT_BODY_CANARY", profile="AGENT_PROFILE_CANARY")
    outcome = await execute_bridge(runtime, {"op": "directory"})
    expected = [
        {
            "account_id": "acc_1",
            "name": "Owner",
            "role": "admin",
            "bridge_enabled": True,
            "bridge_state": "active",
            "you": True,
            "chief_of_staff": {
                "endpoint_id": "bep_1",
                "name": "My CoS",
                "state": "active",
                "reachable": True,
                "you": True,
            },
        },
        {
            "account_id": "acc_2",
            "name": "Dana",
            "role": "member",
            "bridge_enabled": True,
            "bridge_state": "active",
            "chief_of_staff": {
                "endpoint_id": "bep_2",
                "name": "Dana's CoS",
                "state": "active",
                "reachable": True,
            },
        },
    ]
    assert outcome["members"] == expected
    assert _page(outcome) == {
        "members": expected,
        "count": 2,
        "hub_limit": DIRECTORY_MAX,
        "possibly_truncated": False,
    }
    assert "CANARY" not in json.dumps(outcome)
    assert client.calls == [
        {"method": "directory", "tenant_id": "ten_1", "expected_account_id": "acc_1"}
    ]
    assert list(Path(runtime.config_dir).iterdir()) == []


@pytest.mark.asyncio
async def test_directory_refetches_current_names_and_endpoints(runtime):
    first = await execute_bridge(runtime, {"op": "directory"})
    member = runtime.client.directory_result["members"][1]
    member["display_label"] = "Renamed person"
    member["agent"].update(name="Renamed CoS", endpoint_id="bep_reenrolled")
    second = await execute_bridge(runtime, {"op": "directory"})
    assert first["members"][1]["chief_of_staff"]["name"] == "Dana's CoS"
    assert second["members"][1]["name"] == "Renamed person"
    assert second["members"][1]["chief_of_staff"]["name"] == "Renamed CoS"
    assert second["members"][1]["chief_of_staff"]["endpoint_id"] == "bep_reenrolled"
    assert len(runtime.client.calls) == 2


@pytest.mark.asyncio
async def test_directory_pages_and_caps_the_server_answer(runtime):
    runtime.client.directory_result = {
        "members": [{"account_id": f"acc_{i}"} for i in range(DIRECTORY_MAX + 5)]
    }
    first = await execute_bridge(runtime, {"op": "directory", "limit": 100})
    second = await execute_bridge(
        runtime, {"op": "directory", "limit": 100, "cursor": first["next_cursor"]}
    )
    assert _page(first)["count"] == DIRECTORY_MAX
    assert _page(first)["hub_limit"] == DIRECTORY_MAX
    assert _page(first)["possibly_truncated"] is True
    assert [r["account_id"] for r in first["members"]] == [f"acc_{i}" for i in range(100)]
    assert first["next_cursor"] == "100"
    assert [r["account_id"] for r in second["members"]] == [f"acc_{i}" for i in range(100, 200)]
    assert second["next_cursor"] is None
    empty = await execute_bridge(runtime, {"op": "directory", "cursor": "200"})
    assert empty["members"] == [] and empty["next_cursor"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args", [{"limit": 0}, {"limit": 101}, {"cursor": "-1"}, {"cursor": "oops"}]
)
async def test_bad_directory_bounds_never_fetch(runtime, args):
    result = await execute_bridge(runtime, {"op": "directory", **args})
    assert result["ok"] is False and result["code"] == "invalid_request"
    assert runtime.client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("address", [{"account_id": "acc_2"}, {"endpoint_id": "bep_2"}])
async def test_dm_resolves_a_current_agent_and_posts_as_runtime_identity(runtime, address):
    result = await execute_bridge(runtime, {"op": "dm", **address, "body": "hello"})
    assert result["ok"] is True
    assert result["destination_type"] == "conversation_dm"
    assert result["conversation_id"] == "bcv_direct01"
    assert result["message_id"] == "bcmsg_post0001" and result["seq"] == 4
    assert result["recipient_state"] == "active"
    assert "Dana's CoS" in result["content"]
    directory, create, post = runtime.client.calls
    assert directory == {
        "method": "directory",
        "tenant_id": "ten_1",
        "expected_account_id": "acc_1",
    }
    assert create == {
        "method": "create_conversation",
        "expected_account_id": "acc_1",
        "body": {
            "tenant_id": "ten_1",
            "kind": "dm",
            "members": [
                {"account_id": "acc_1", "kind": "agent", "endpoint_id": "bep_1"},
                {"account_id": "acc_2", "kind": "agent", "endpoint_id": "bep_2"},
            ],
        },
        "actor_endpoint_id": "bep_1",
        "origin_session_id": "sess_1",
    }
    assert post["conversation_id"] == "bcv_direct01"
    assert post["tenant_id"] == "ten_1" and post["text"] == "hello"
    assert post["actor_endpoint_id"] == "bep_1" and post["origin_session_id"] == "sess_1"
    assert post["client_ref"] and post["thread_root_id"] is None
    assert post["expected_account_id"] == "acc_1"
    assert list(Path(runtime.config_dir).iterdir()) == []


@pytest.mark.asyncio
async def test_dm_human_target_does_not_silently_invite_their_agent(runtime):
    del runtime.client.directory_result["members"][1]["agent"]
    result = await execute_bridge(
        runtime, {"op": "dm", "account_id": "acc_2", "target": "human", "text": "for Dana"}
    )
    assert result["ok"] is True
    assert runtime.client.calls[1]["body"]["members"][1] == {
        "account_id": "acc_2",
        "kind": "human",
    }
    assert runtime.client.calls[2]["text"] == "for Dana"
    assert "recipient_state" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["paused", "disabled"])
async def test_unreachable_agent_never_creates_or_posts(runtime, state):
    runtime.client.directory_result["members"][1]["agent"]["state"] = state
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == "not_reachable"
    assert result["reason"] == f"agent_{state}"
    assert [c["method"] for c in runtime.client.calls] == ["directory"]


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["agent", "human"])
async def test_disabled_member_without_agent_never_writes(runtime, target):
    member = runtime.client.directory_result["members"][1]
    member.update(bridge_enabled=False, bridge_state="disabled")
    del member["agent"]
    result = await execute_bridge(
        runtime, {"op": "dm", "account_id": "acc_2", "target": target, "body": "hi"}
    )
    assert result["ok"] is False and result["code"] == "not_reachable"
    assert result["reason"] == ("bridge_disabled" if target == "human" else "no_agent_disabled")
    assert [c["method"] for c in runtime.client.calls] == ["directory"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address,code,reason",
    [
        ({"account_id": "acc_absent"}, "not_reachable", "not_a_member"),
        ({"endpoint_id": "bep_stale"}, "not_reachable", "unknown_endpoint"),
        ({"account_id": "acc_1"}, "self_target", None),
        ({"account_id": "acc_1", "target": "human"}, "self_target", None),
        ({"endpoint_id": "bep_1"}, "self_target", None),
    ],
)
async def test_unknown_or_self_target_has_no_write(runtime, address, code, reason):
    result = await execute_bridge(runtime, {"op": "dm", **address, "body": "hi"})
    assert result["ok"] is False and result["code"] == code
    assert result.get("reason") == reason
    assert [c["method"] for c in runtime.client.calls] == ["directory"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"body": "hi"},
        {"account_id": "acc_2", "endpoint_id": "bep_2", "body": "hi"},
        {"account_id": "acc_2", "body": "  "},
        {"account_id": "acc_2", "body": "hi", "thread_root_id": "../../bad"},
        {"account_id": "acc_2", "body": "hi", "actor_endpoint_id": "bep_forged"},
        {"account_id": "acc_2", "body": "hi", "origin_session_id": "forged"},
        {"account_id": "acc_2", "body": "hi", "target": "owner"},
    ],
)
async def test_malformed_dm_or_forged_identity_never_reaches_client(runtime, args):
    result = await execute_bridge(runtime, {"op": "dm", **args})
    assert result["ok"] is False and result["code"] == "invalid_request"
    assert runtime.client.calls == []


@pytest.mark.asyncio
async def test_invited_agent_receipt_says_acceptance_is_needed_before_wake(runtime):
    runtime.client.created_result["members"][1]["state"] = "invited"
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is True and result["state"] == "sent"
    assert result["message_id"] == "bcmsg_post0001" and result["seq"] == 4
    assert result["recipient_state"] == "invited"
    assert "owner must accept before it is woken" in result["content"]
    assert [c["method"] for c in runtime.client.calls] == [
        "directory",
        "create_conversation",
        "conversation_send",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["dm", "shared_conversations", "shared_history"])
async def test_agent_ops_never_fall_back_to_human_identity(runtime, op):
    runtime.owner = _runtime(runtime.client, endpoint_id="").owner
    result = await execute_bridge(runtime, {"op": op})
    assert result["ok"] is False and result["code"] == "no_agent_endpoint"
    assert runtime.client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,code", [("paused", "bridge_paused"), ("disabled", "bridge_disabled")]
)
async def test_local_admission_denies_dm_before_directory_or_writes(runtime, state, code):
    runtime.admission = state
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == code
    assert runtime.client.calls == []


@pytest.mark.asyncio
async def test_directory_is_readable_while_paused_but_not_disabled(runtime):
    runtime.admission = "paused"
    assert (await execute_bridge(runtime, {"op": "directory"}))["ok"] is True
    runtime.client.calls.clear()
    runtime.admission = "disabled"
    result = await execute_bridge(runtime, {"op": "directory"})
    assert result["ok"] is False and result["code"] == "bridge_disabled"
    assert runtime.client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["directory", "dm", "shared_conversations", "shared_history"])
async def test_owner_fence_precedes_every_new_operation(runtime, monkeypatch, op):
    def refusal(runtime):
        return {"code": "owner_changed", "error": "owner changed"}

    # Both fences: shared_* keep the strict one, the rest take the outbound one.
    monkeypatch.setattr(host_mod, "owner_fence_refusal", refusal)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", refusal)
    result = await execute_bridge(runtime, {"op": op, "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == "owner_changed"
    assert runtime.client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op,code", [("directory", "directory_failed"), ("dm", "dm_failed")])
async def test_directory_failure_never_falls_back_to_a_guessed_recipient(runtime, op, code):
    runtime.client.directory = AsyncMock(side_effect=RuntimeError("directory unavailable"))
    result = await execute_bridge(runtime, {"op": op, "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == code
    runtime.client.directory.assert_awaited_once_with("ten_1", expected_account_id="acc_1")
    assert runtime.client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["agent_not_permitted", "member_not_found"])
async def test_hub_refused_recipient_never_posts_even_after_directory_says_active(runtime, code):
    runtime.client.create_conversation = AsyncMock(side_effect=BridgeRefusal(code, "refused"))
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == "not_reachable"
    assert result["reason"] == code
    runtime.client.create_conversation.assert_awaited_once()
    assert [c["method"] for c in runtime.client.calls] == ["directory"]


@pytest.mark.asyncio
async def test_opened_dm_with_failed_post_does_not_claim_sent(runtime):
    runtime.client.conversation_send = AsyncMock(side_effect=RuntimeError("transport unavailable"))
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == "post_failed"
    assert "bcv_direct01" in result["error"]
    assert "state" not in result and "message_id" not in result
    runtime.client.conversation_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_opened_dm_id_never_becomes_a_post_path(runtime):
    runtime.client.created_result = {"id": "../../other"}
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == "dm_failed"
    assert [c["method"] for c in runtime.client.calls] == ["directory", "create_conversation"]


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["shared_conversations", "shared_history"])
async def test_shared_reads_reject_task_session_even_with_matching_endpoint(
    runtime, monkeypatch, op
):
    monkeypatch.setattr(authority_mod, "authority_fence", lambda runtime: None)
    aida_state.write_state(runtime.config_dir, {"session_id": "different-chief-session"})
    result = await execute_bridge(runtime, {"op": op, "channel_id": "bcv_direct01"})
    assert result["ok"] is False and result["code"] == "not_chief_of_staff"
    assert runtime.client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["shared_conversations", "shared_history"])
async def test_shared_reads_recheck_authority_even_for_durable_cos(runtime, monkeypatch, op):
    aida_state.write_state(runtime.config_dir, {"session_id": runtime.session_id})
    monkeypatch.setattr(
        authority_mod,
        "authority_fence",
        lambda runtime: {"code": "owner_changed", "error": "owner changed"},
    )
    result = await execute_bridge(runtime, {"op": op, "channel_id": "bcv_direct01"})
    assert result["ok"] is False and result["code"] == "owner_changed"
    assert runtime.client.calls == []


def test_new_ops_publish_read_write_tiers_and_no_actor_override(runtime):
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None and tool.call_approval_tier is not None
    for op in ("directory", "shared_conversations", "shared_history"):
        assert tool.call_approval_tier({"op": op}) == "read"
    assert tool.call_approval_tier({"op": "dm"}) == "write"
    props = tool.parameters["properties"]
    assert props["target"]["enum"] == ["agent", "human"]
    assert "account_id" in props and "endpoint_id" in props
    assert "actor_endpoint_id" not in props and "origin_session_id" not in props
