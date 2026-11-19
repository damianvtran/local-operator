"""Default thread placement for bound channel_post replies.

Operator, 2026-11-17 (Team Bridge #general seq 94): in a channel or group an
agent's answer to a known message goes into that message's thread; top-level
is for new topics (``new_post: true``), and DMs stay inline. The root is the
hub's own feed fact for a message this session PRESENTED — never a guess — so
a reply-to-a-reply lands under the same root, and an unknown message falls
back to the pre-threading channel-level post with a warning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.tool import (
    _PRESENTED_PLACEMENT_MAX,
    BridgeRuntime,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_body_turn_e2e import (  # noqa: F401
    CONVERSATION_ID,
    MESSAGE_ID,
    SESSION_ID,
    TENANT,
    CapturingStream,
    FakeBridgeClient,
    _carrying,
    _owner,
    _reserve,
    _row,
    _runtime,
    _serve_body_turn,
    _session,
    _store,
    telemetry_off,
)
from tests.unit.team_bridge.test_exact_reply_destination import MA, A

from tests.unit.team_bridge.test_exact_reply_destination import (  # noqa: F401  # isort: skip
    bound_runtime as _exact_reply_runtime,
)

ROOT = "bcmsg_rootroot"


@pytest.fixture
def bound_runtime(request):
    """The exact-reply suite's real enrolled store + admitted reservations.

    Re-exposed under its own name: a bare import would collide with the test
    parameters of that name (flake8 F811), and the aliased import registers
    under the alias.
    """
    return request.getfixturevalue("_exact_reply_runtime")


def _present(runtime: BridgeRuntime, message_id: str, thread_root_id: str | None = None) -> None:
    runtime.note_presented(
        SimpleNamespace(conversation_id=A, message_id=message_id, thread_root_id=thread_root_id)
    )


def _sends(client: Any) -> list[dict[str, Any]]:
    return [call for call in client.calls if call["method"] == "conversation_send"]


async def _post(runtime: BridgeRuntime, **extra: Any) -> dict[str, Any]:
    return await execute_bridge(
        runtime, {"op": "channel_post", "channel_id": A, "body": "answer", **extra}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "presented_root", "expected_root"),
    [
        ("channel", None, MA),  # a channel-level message: its own thread
        ("group", None, MA),  # groups thread exactly like channels
        ("dm", None, None),  # a DM stays inline
        ("channel", ROOT, ROOT),  # reply-to-reply: the SAME root, never MA
        ("group", ROOT, ROOT),
    ],
)
async def test_bound_reply_default_placement(bound_runtime, kind, presented_root, expected_root):
    runtime, client, _ = bound_runtime
    client.conversation_result = {"id": A, "kind": kind}
    _present(runtime, MA, presented_root)

    out = await _post(runtime, in_reply_to_message_id=MA)

    assert out["ok"], out
    (send,) = _sends(client)
    assert send["thread_root_id"] == expected_root
    assert "warning" not in out
    if expected_root:
        assert f"(reply in thread {expected_root})" in out["content"]


@pytest.mark.asyncio
async def test_new_post_is_the_top_level_opt_out(bound_runtime):
    runtime, client, _ = bound_runtime
    client.conversation_result = {"id": A, "kind": "channel"}
    _present(runtime, MA)
    runtime.bridge_input_presented = True

    out = await _post(runtime, new_post=True)

    assert out["ok"], out
    (send,) = _sends(client)
    assert send["thread_root_id"] is None
    # No placement read at all: an opt-out spends no hub call.
    assert [c["method"] for c in client.calls] == ["conversation_send"]


@pytest.mark.asyncio
async def test_explicit_thread_root_wins_without_a_kind_read(bound_runtime):
    runtime, client, _ = bound_runtime
    _present(runtime, MA)

    out = await _post(runtime, in_reply_to_message_id=MA, thread_root_id=ROOT)

    assert out["ok"], out
    assert [c["method"] for c in client.calls] == ["conversation_send"]
    assert client.calls[0]["thread_root_id"] == ROOT


@pytest.mark.asyncio
async def test_unpresented_message_posts_channel_level_with_a_warning(bound_runtime):
    runtime, client, _ = bound_runtime
    client.conversation_result = {"id": A, "kind": "channel"}

    out = await _post(runtime, in_reply_to_message_id=MA)

    assert out["ok"], out
    assert [c["method"] for c in client.calls] == ["conversation_send"]
    assert client.calls[0]["thread_root_id"] is None
    assert "thread is unknown here" in out["warning"]


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["raises", {"id": A}])
async def test_unreadable_kind_falls_back_to_channel_level(bound_runtime, monkeypatch, view):
    runtime, client, _ = bound_runtime
    _present(runtime, MA)
    if view == "raises":

        async def broken(*args, **kwargs):
            raise RuntimeError("hub down")

        monkeypatch.setattr(client, "conversation", broken)
    else:
        client.conversation_result = view

    out = await _post(runtime, in_reply_to_message_id=MA)

    assert out["ok"], out
    (send,) = _sends(client)
    assert send["thread_root_id"] is None
    assert "thread is unknown here" in out["warning"]


@pytest.mark.asyncio
async def test_a_named_recipient_reuses_the_member_read_for_the_kind(bound_runtime):
    runtime, client, _ = bound_runtime
    client.conversation_result = {
        "id": A,
        "kind": "channel",
        "members": [{"account_id": "acc_2", "display_label": "Recipient", "kind": "human"}],
    }
    _present(runtime, MA)

    out = await _post(runtime, in_reply_to_message_id=MA, to={"name": "Recipient"})

    assert out["ok"], out
    assert [c["method"] for c in client.calls] == ["conversation", "conversation_send"]
    assert client.calls[1]["thread_root_id"] == MA


def test_presented_placements_are_bounded_oldest_first() -> None:
    runtime = BridgeRuntime(cast(Any, None), "/nonexistent", TENANT, _owner(), SESSION_ID)
    for index in range(_PRESENTED_PLACEMENT_MAX + 2):
        _present(runtime, f"bcmsg_{index:08d}")
    assert len(runtime.presented_placements) == _PRESENTED_PLACEMENT_MAX
    assert "bcmsg_00000000" not in runtime.presented_placements
    assert "bcmsg_00000001" not in runtime.presented_placements
    assert runtime.presented_placements[f"bcmsg_{_PRESENTED_PLACEMENT_MAX + 1:08d}"] == (A, "")


class _KindHub(FakeBridgeClient):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.sends: list[tuple[str, dict[str, Any]]] = []

    async def conversation(self, conversation_id, *, tenant_id=None):
        return {"id": conversation_id, "kind": self.kind}

    async def conversation_send(self, conversation_id, **kwargs):
        guard = kwargs.pop("pre_http_guard", None)
        if guard is not None:
            guard()
        self.sends.append((conversation_id, kwargs))
        return {"id": "bcmsg_result01", "seq": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "expected_root"), [("channel", MESSAGE_ID), ("dm", None)])
async def test_real_session_dispatch_records_the_placement_the_post_uses(
    tmp_path, kind, expected_root
):
    """The session's provider seam, not a hand-fed map, is what records it."""
    config = tmp_path / "config"
    config.mkdir()
    store = _store(config)
    _reserve(store, config_dir=config)
    hub = _KindHub(kind)
    runtime = BridgeRuntime(
        cast(TeamBridgeClient, hub), config, TENANT, _owner(), SESSION_ID, admission="admitted"
    )
    args = {
        "op": "channel_post",
        "channel_id": CONVERSATION_ID,
        "body": "answer",
        "in_reply_to_message_id": MESSAGE_ID,
    }
    stream = CapturingStream(
        [
            [
                StreamToolCallDelta(
                    index=0, id="reply", name="bridge", argument_delta=json.dumps(args)
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    session = _session(tmp_path, stream, runtime, tools=[tool])
    try:
        await asyncio.wait_for(_serve_body_turn(session), 60)
    finally:
        await session.dispose()
    assert runtime.presented_placements[MESSAGE_ID] == (CONVERSATION_ID, "")
    (sent,) = hub.sends
    assert sent[0] == CONVERSATION_ID
    assert sent[1]["thread_root_id"] == expected_root


class _RaisingPlacementRuntime(SimpleNamespace):
    """A runtime whose placement record fails at the seam."""

    def note_presented(self, ref: Any) -> None:
        raise RuntimeError("placement store unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["missing", "raising"])
async def test_a_failed_placement_record_never_breaks_the_body_turn(
    tmp_path, caplog, variant
) -> None:
    """Recording a presented placement is best-effort (PR #4177).

    A miss already degrades to a channel-level post with a warning, so a
    runtime without ``note_presented`` (older doubles, other runtimes) or one
    whose record raises must still dispatch, serve and complete the body.
    """
    config = tmp_path / "config"
    config.mkdir()
    store = _store(config)
    _reserve(store, config_dir=config)
    base = _runtime(config, FakeBridgeClient())
    runtime = base if variant == "missing" else _RaisingPlacementRuntime(**vars(base))
    assert hasattr(runtime, "note_presented") is (variant == "raising")
    stream = CapturingStream([[StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, runtime)
    try:
        await asyncio.wait_for(_serve_body_turn(session), 60)
    finally:
        await session.dispose()
    assert _carrying(stream) == [0], "the body still reached the provider"
    assert _row(config)["state"] == "completed"
    # Logged at debug at most: never surfaced as a failed model stream.
    loud = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and ("note_presented" in r.getMessage() or "placement store" in r.getMessage())
    ]
    assert loud == []


# The agent-facing reads must keep ``reply_count`` (the hub's per-root thread
# rollup), or an agent cannot tell which channel messages already HAVE a thread
# to answer in. The model-facing copy goes through ``bound_inline_data_payload``;
# a data URI in a body forces its copy-on-write path, the one that rebuilds dicts.
_DATA_URI = "data:image/png;base64," + "A" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["plain", f"see {_DATA_URI}"])
async def test_reply_count_survives_into_the_model_facing_reads(bound_runtime, body) -> None:
    from local_operator.team_bridge.tool import _INLINE_DATA_KEY

    runtime, client, _ = bound_runtime
    root = {"id": MA, "seq": 3, "reply_count": 2, "body": body}
    client.messages_result = {"messages": [dict(root)]}
    client.replies_result = {"root": dict(root), "replies": [], "has_more": False}
    history = await execute_bridge(runtime, {"op": "channel_history", "channel_id": A, "limit": 10})
    thread = await execute_bridge(
        runtime, {"op": "channel_thread", "channel_id": A, "thread_root_id": MA}
    )
    for outcome in (history, thread):
        assert outcome["ok"] is True, outcome
        model_copy = outcome[_INLINE_DATA_KEY]
        model_text = model_copy if isinstance(model_copy, str) else json.dumps(model_copy)
        assert '"reply_count": 2' in model_text
        assert _DATA_URI not in model_text
