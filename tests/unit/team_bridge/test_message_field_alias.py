"""A message op reads its payload from ``body`` OR ``text``, never neither.

THE DEFECT (reproduced twice in production, 2026-11-04): the bridge params model
is shared by every op and carries both ``text`` (the pair-inbox name) and
``body`` (the room name). A model that wrote ``channel_post`` with ``text`` got
"channel_post needs non-empty body" and had to retry; ``send``/``reply`` with
``body`` failed the same way in the other direction. For the four message ops
the two words can only mean the same payload, so the executor folds the alias
onto the op's own field — and refuses, sending nothing, when both are given with
different content, because choosing one would send a message the caller may not
have meant.

The other half is pinned too: no op that does NOT take a message changes
meaning. ``react`` takes ``emoji``, ``channel_create`` takes ``name``/``topic``,
``take_up`` returns ``text`` rather than taking it, and the reads take neither.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.tool import (
    BRIDGE_OPS,
    MESSAGE_FIELD_BY_OP,
    BridgeRuntime,
    _normalize_message_field,
    execute_bridge,
)
from tests.unit.team_bridge.test_channel_tool import RecordingClient
from tests.unit.team_bridge.test_channel_tool import _runtime as _room_runtime
from tests.unit.team_bridge.test_client import _ok
from tests.unit.team_bridge.test_dm_addressing_and_refusals import (
    PEER_ENDPOINT,
    DmClient,
    _member,
)
from tests.unit.team_bridge.test_dm_addressing_and_refusals import (
    _runtime as _dm_runtime,
)
from tests.unit.team_bridge.test_h5d_tool import TENANT, _client, _owner

CHANNEL = "bcv_abcd1234"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gates open and a fresh cwd: the subject is argument handling, not the
    fences (their own files), and the reply guard writes under ``config_dir``."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _post(**fields: Any) -> dict[str, Any]:
    return {"op": "channel_post", "channel_id": CHANNEL, "new_post": True, **fields}


def _sent_texts(client: RecordingClient | DmClient) -> list[str]:
    return [call["text"] for call in client.calls if call["method"] == "conversation_send"]


# -- channel_post ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"text": "claiming QA-1426"},  # the production repro
        {"body": "claiming QA-1426"},
        {"body": "claiming QA-1426", "text": "claiming QA-1426"},  # one message spelt twice
        {"body": "   ", "text": "claiming QA-1426"},  # blank canonical is empty
    ],
)
async def test_channel_post_takes_its_message_from_either_field(fields: dict[str, str]) -> None:
    client = RecordingClient()
    outcome = await execute_bridge(_room_runtime(client), _post(**fields))
    assert outcome["ok"] is True, outcome
    assert _sent_texts(client) == ["claiming QA-1426"]


@pytest.mark.asyncio
async def test_channel_post_refuses_two_different_messages_and_sends_nothing() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(_room_runtime(client), _post(body="first", text="second"))
    assert outcome["ok"] is False
    assert outcome["code"] == "invalid_request"
    assert "both body and text" in outcome["error"]
    assert "as body" in outcome["error"]
    assert client.calls == []


@pytest.mark.asyncio
async def test_channel_post_with_neither_field_still_refuses_empty() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(_room_runtime(client), _post(text="  "))
    assert outcome == {
        "ok": False,
        "code": "invalid_request",
        "error": "channel_post needs non-empty body",
    }
    assert client.calls == []


# -- dm ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [{"text": "ping"}, {"body": "ping"}, {"body": "ping", "text": "ping"}],
)
async def test_dm_takes_its_message_from_either_field(fields: dict[str, str]) -> None:
    client = DmClient([_member()])
    outcome = await execute_bridge(
        _dm_runtime(client), {"op": "dm", "endpoint_id": PEER_ENDPOINT, **fields}
    )
    assert outcome["ok"] is True, outcome
    assert _sent_texts(client) == ["ping"]


@pytest.mark.asyncio
async def test_dm_refuses_two_different_messages_before_any_hub_call() -> None:
    client = DmClient([_member()])
    outcome = await execute_bridge(
        _dm_runtime(client),
        {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": "ping", "text": "pong"},
    )
    assert outcome["ok"] is False and "both body and text" in outcome["error"]
    # Not even the directory read: the conflict is refused before resolution.
    assert client.calls == []


# -- send / reply (pair inbox; canonical field is ``text``) -----------------------


def _pair_runtime(tmp_path: Any, requests: list[httpx.Request]) -> BridgeRuntime:
    return BridgeRuntime(
        client=_client(
            lambda request: (
                requests.append(request),
                _ok({"message": {"id": "bmsg_9"}}, status=201),
            )[1]
        ),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )


def _pair_args(op: str, **fields: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"op": op, "to": {"account_id": "acc_2"}, **fields}
    if op == "reply":
        args["thread_id"] = "bth_1"
    return args


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
@pytest.mark.parametrize(
    "fields",
    [{"body": "hi"}, {"text": "hi"}, {"text": "hi", "body": "hi"}, {"text": "", "body": "hi"}],
)
async def test_send_and_reply_take_their_message_from_either_field(
    tmp_path: Any, op: str, fields: dict[str, str]
) -> None:
    requests: list[httpx.Request] = []
    outcome = await execute_bridge(_pair_runtime(tmp_path, requests), _pair_args(op, **fields))
    assert outcome["ok"] is True, outcome
    (request,) = requests
    assert json.loads(request.content)["body"] == {"text": "hi"}


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
async def test_send_and_reply_refuse_two_different_messages_and_record_nothing(
    tmp_path: Any, op: str
) -> None:
    from local_operator.team_bridge.outbox import Outbox

    requests: list[httpx.Request] = []
    runtime = _pair_runtime(tmp_path, requests)
    outcome = await execute_bridge(runtime, _pair_args(op, text="yes", body="no"))
    assert outcome["ok"] is False
    assert "both text and body" in outcome["error"] and "as text" in outcome["error"]
    assert requests == []
    assert Outbox(tmp_path).pending(runtime.key()) == []


# -- the map, and every op outside it ---------------------------------------------


def test_the_map_names_exactly_the_ops_that_send_a_message() -> None:
    assert MESSAGE_FIELD_BY_OP == {
        "channel_post": "body",
        "dm": "body",
        "send": "text",
        "reply": "text",
    }
    assert set(MESSAGE_FIELD_BY_OP) <= set(BRIDGE_OPS)


@pytest.mark.parametrize("op", [op for op in BRIDGE_OPS if op not in MESSAGE_FIELD_BY_OP])
def test_an_op_without_a_message_keeps_its_arguments_untouched(op: str) -> None:
    """``react``/``channel_create``/``take_up``/the reads: neither field is moved,
    dropped or refused, even when both are present and differ."""

    args = {"op": op, "text": "a", "body": "b", "emoji": "✅", "topic": "t"}
    folded, refused = _normalize_message_field(op, args)
    assert refused is None
    assert folded is args
    assert folded == {"op": op, "text": "a", "body": "b", "emoji": "✅", "topic": "t"}


@pytest.mark.asyncio
async def test_react_still_reads_only_its_emoji_when_text_is_present() -> None:
    client = RecordingClient()
    outcome = await execute_bridge(
        _room_runtime(client),
        {
            "op": "react",
            "channel_id": CHANNEL,
            "message_id": "bcmsg_abcd1234",
            "emoji": "✅",
            "text": "not a reaction",
        },
    )
    assert outcome["ok"] is True, outcome
    (call,) = client.calls
    assert call["method"] == "conversation_react" and "not a reaction" not in json.dumps(call)
