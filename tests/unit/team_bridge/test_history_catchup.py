"""Requested-page admission, with no live auth, model or bridge traffic.

These are source-only regression cases until the parent's resource hold lifts.
The shared-handler cases keep the real claim/fence/admit spine but replace the
model and #3063's authority fetch. They are not native or live-server evidence.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from local_operator.team_bridge import history_catchup as catchup
from local_operator.team_bridge.connector import conversation as conversation_mod
from local_operator.team_bridge.connector import wake as wake_mod
from local_operator.team_bridge.screening import ScreenResult
from local_operator.team_bridge.store import Store
from local_operator.team_bridge.types import ConversationFeedRef
from tests.unit.team_bridge.test_conversation_delivery import (
    CONVERSATION_ID,
    MESSAGE_ID,
    FakeServer,
    FakeWire,
    _enroll,
    _owner,
    _receipt,
)

NOW = 1_800_000_000.0


def _row(seq, status="not_screened", *, age=0):
    row = {
        "id": f"bcmsg_history{seq:08d}", "seq": seq, "body": None,
        "created_at": datetime.fromtimestamp(NOW - age, timezone.utc).isoformat(),
    }
    if status is not None:
        row["agent_delivery_status"] = status
    return row


def _runtime(tmp_path, page):
    owner = _owner("catchup", 1)
    _enroll(tmp_path, owner, "session-catchup")
    client = SimpleNamespace(
        current_access=AsyncMock(return_value=SimpleNamespace(credential_id=1)),
        _auth_store=lambda: object(),
        conversation_messages=AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(page)),
    )
    return SimpleNamespace(
        owner=owner, config_dir=tmp_path, tenant_id=owner.tenant_id,
        session_id="session-catchup", client=client, admission="admitted",
    )


async def _read(runtime, page, **kwargs):
    return await catchup.catch_up_history_page(
        runtime, CONVERSATION_ID, copy.deepcopy(page),
        limit=kwargs.get("limit", 50), before_seq=kwargs.get("before_seq"),
        after_seq=kwargs.get("after_seq"),
    )


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(catchup.time, "time", lambda: NOW)


@pytest.mark.asyncio
async def test_missing_status_prejoin_and_rejected_never_screen(tmp_path, monkeypatch):
    page = {"messages": [_row(1, None), _row(2, "rejected"), _row(3, "admitted")]}
    page["messages"][1]["body"] = "must never escape"
    runtime = _runtime(tmp_path, page)
    handler = AsyncMock(side_effect=AssertionError("no eligible rows"))
    monkeypatch.setattr(catchup, "handle_conversation_available", handler)
    answer = await _read(runtime, page)
    assert answer["messages"][1]["body"] is None
    assert "catchup" not in answer
    handler.assert_not_called()
    runtime.client.current_access.assert_not_called()
    runtime.client.conversation_messages.assert_not_called()


@pytest.mark.asyncio
async def test_oldest_first_bounded_and_resumed_without_repeat_rejection(tmp_path, monkeypatch):
    page = {"messages": [_row(seq) for seq in range(12, 0, -1)]}
    runtime = _runtime(tmp_path, page)
    calls = []

    async def handle(owner, ref, **kwargs):
        assert kwargs["admission_only"] is True and kwargs["wire"] is runtime.client
        assert owner == runtime.owner
        calls.append(ref.conversation_seq)
        row = next(row for row in page["messages"] if row["id"] == ref.message_id)
        row["agent_delivery_status"] = "rejected" if ref.conversation_seq == 2 else "admitted"
        row["body"] = None if ref.conversation_seq == 2 else "authorized history"
        return {"action": "rejected" if ref.conversation_seq == 2 else "admitted"}

    monkeypatch.setattr(catchup, "handle_conversation_available", handle)
    first = await _read(runtime, page, after_seq=1)
    assert calls == list(range(1, 11))
    assert first["catchup"]["code"] == "message_limit"
    assert first["catchup"]["refreshed"] is True
    assert first["messages"][-2]["agent_delivery_status"] == "rejected"
    second = await _read(runtime, page, after_seq=1)
    assert calls == list(range(1, 13))
    assert second["catchup"]["confirmed"] == 2
    assert runtime.client.conversation_messages.await_count == 2
    assert runtime.client.conversation_messages.call_args.kwargs == {
        "tenant_id": runtime.tenant_id, "limit": 50, "before_seq": None,
        "after_seq": 1, "endpoint_id": runtime.owner.endpoint_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["hold", "claimed_elsewhere", "badge_only"])
async def test_nonterminal_cursor_never_skips_held_oldest(tmp_path, monkeypatch, action):
    page = {"messages": [_row(2), _row(1)]}
    runtime = _runtime(tmp_path, page)
    calls = []

    async def handle(owner, ref, **kwargs):
        calls.append(ref.conversation_seq)
        return {"action": action}

    monkeypatch.setattr(catchup, "handle_conversation_available", handle)
    for _ in range(2):
        answer = await _read(runtime, page)
        assert answer["catchup"]["state"] == "held"
    assert calls == [1, 1]
    runtime.client.conversation_messages.assert_not_called()


@pytest.mark.asyncio
async def test_reject_alone_refreshes_authoritative_status(tmp_path, monkeypatch):
    page = {"messages": [_row(1)]}
    runtime = _runtime(tmp_path, page)

    async def reject(*args, **kwargs):
        page["messages"][0]["agent_delivery_status"] = "rejected"
        return {"action": "rejected"}

    monkeypatch.setattr(catchup, "handle_conversation_available", reject)
    answer = await _read(runtime, page)
    assert answer["messages"][0]["agent_delivery_status"] == "rejected"
    assert answer["messages"][0]["body"] is None
    runtime.client.conversation_messages.assert_awaited_once()


@pytest.mark.asyncio
async def test_age_and_unknown_timestamp_skip_without_stopping_the_page(tmp_path, monkeypatch):
    # Age is PER CANDIDATE (module docstring): an old row and a row whose timestamp
    # cannot establish age are both excluded, and the eligible row after them is
    # still attempted. A page-wide hold here is how a single malformed row used to
    # freeze every later message behind it.
    page = {"messages": [_row(3), _row(2), _row(1, age=catchup.CATCHUP_AGE_S + 1)]}
    page["messages"][1]["created_at"] = "missing timezone"
    runtime = _runtime(tmp_path, page)
    handler = AsyncMock(return_value={"action": "admitted", "message_id": "bcmsg"})
    monkeypatch.setattr(catchup, "handle_conversation_available", handler)
    answer = await _read(runtime, page)
    assert answer["catchup"]["age_skipped"] == 1
    assert answer["catchup"]["age_unknown"] == 1
    assert answer["catchup"]["confirmed"] == 1
    assert [call.args[1].conversation_seq for call in handler.await_args_list] == [3]
    assert runtime.client.conversation_messages.await_count == 1


@pytest.mark.asyncio
async def test_future_timestamp_is_skipped_not_a_page_wide_hold(tmp_path, monkeypatch):
    # Hub/host clock disagreement is not evidence that a message is new, and it
    # must not pin the page: the future-dated row is excluded, the older eligible
    # row is still attempted.
    page = {"messages": [_row(2, age=-3600), _row(1)]}
    runtime = _runtime(tmp_path, page)
    handler = AsyncMock(return_value={"action": "rejected", "message_id": "bcmsg"})
    monkeypatch.setattr(catchup, "handle_conversation_available", handler)
    answer = await _read(runtime, page)
    assert answer["catchup"]["age_unknown"] == 1
    assert answer["catchup"]["rejected"] == 1
    assert [call.args[1].conversation_seq for call in handler.await_args_list] == [1]


@pytest.mark.asyncio
async def test_unconfirmed_handler_is_resolved_by_server_status(tmp_path, monkeypatch):
    # THE LIVENESS CASE (review 2026-10-28): a retry-after-verdict hold is not a
    # verdict. The hub's own status for the message id decides, so a text-only
    # allow already recorded can never pin the cursor forever.
    page = {"messages": [_row(1)]}
    runtime = _runtime(tmp_path, page)
    handler = AsyncMock(return_value={"action": "hold", "code": "admission_unverified"})
    monkeypatch.setattr(catchup, "handle_conversation_available", handler)
    admitted = {"messages": [{**_row(1, status="admitted"), "body": "screened body"}]}
    runtime.client.conversation_messages = AsyncMock(return_value=admitted)
    answer = await _read(runtime, page)
    assert answer["catchup"]["confirmed"] == 1
    assert answer["catchup"]["refreshed"] is True
    assert answer["messages"][0]["body"] == "screened body"
    # The confirmed message is behind the cursor: the next identical read asks the
    # handler about nothing and the hub is not re-read for it.
    handler.reset_mock()
    runtime.client.conversation_messages = AsyncMock(return_value=admitted)
    again = await _read(runtime, page)
    handler.assert_not_called()
    assert again["messages"][0]["body"] is None


@pytest.mark.asyncio
async def test_server_authority_hold_keeps_the_cursor(tmp_path, monkeypatch):
    # Still not_screened at the hub: a genuine hold, and the cursor must stop BEFORE
    # the row so the next read retries it instead of skipping it.
    page = {"messages": [_row(1)]}
    runtime = _runtime(tmp_path, page)
    handler = AsyncMock(return_value={"action": "hold", "code": "admission_held"})
    monkeypatch.setattr(catchup, "handle_conversation_available", handler)
    runtime.client.conversation_messages = AsyncMock(return_value=copy.deepcopy(page))
    answer = await _read(runtime, page)
    assert answer["catchup"]["state"] == "held"
    assert answer["catchup"]["code"] == "admission_held"
    assert answer["catchup"]["confirmed"] == 0
    handler.reset_mock()
    await _read(runtime, page)
    assert [call.args[1].conversation_seq for call in handler.await_args_list] == [1]


@pytest.mark.asyncio
async def test_total_timeout_preserves_cursor_and_no_screen_bytes_escape(tmp_path, monkeypatch):
    page = {"messages": [_row(1)]}
    page["messages"][0]["body"] = "unscreened canary"
    runtime = _runtime(tmp_path, page)
    calls = []

    async def held(owner, ref, **kwargs):
        calls.append(ref.conversation_seq)
        await asyncio.Event().wait()

    monkeypatch.setattr(catchup, "handle_conversation_available", held)
    monkeypatch.setattr(catchup, "CATCHUP_TIMEOUT_S", 0.01)
    for _ in range(2):
        answer = await _read(runtime, page)
        assert answer["catchup"]["code"] == "time_limit"
        assert answer["messages"][0]["body"] is None
    assert calls == [1, 1]
    runtime.client.conversation_messages.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["paused", "disabled", "pending"])
async def test_retained_history_read_does_not_mutate_when_not_admitted(tmp_path, monkeypatch, admission):
    page = {"messages": [_row(1)]}
    runtime = _runtime(tmp_path, page)
    runtime.admission = admission
    handler = AsyncMock()
    monkeypatch.setattr(catchup, "handle_conversation_available", handler)
    answer = await _read(runtime, page)
    assert answer["catchup"]["code"] == "not_admitted"
    handler.assert_not_called()


def test_cursor_is_durable_and_bound_to_window_owner_and_generation(tmp_path):
    owner = _owner("catchup", 1)
    store = _enroll(tmp_path, owner, "session-catchup")
    store.advance_history_catchup_cursor(owner, CONVERSATION_ID, "window-a", 10)
    store = Store(tmp_path)
    assert store.history_catchup_cursor(owner, CONVERSATION_ID, "window-a") == 10
    store.advance_history_catchup_cursor(owner, CONVERSATION_ID, "window-a", 5)
    assert store.history_catchup_cursor(owner, CONVERSATION_ID, "window-a") == 10
    assert store.history_catchup_cursor(owner, CONVERSATION_ID, "window-b") == 0
    assert store.history_catchup_cursor(owner, "bcv_another12", "window-a") == 0
    other = _owner("other", 2)
    _enroll(tmp_path, other, "session-other")
    assert store.history_catchup_cursor(other, CONVERSATION_ID, "window-a") == 0
    newer = replace(owner, enrollment_generation=2)
    _enroll(tmp_path, newer, "session-catchup")
    assert store.history_catchup_cursor(newer, CONVERSATION_ID, "window-a") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["allow", "reject"])
async def test_shared_handler_admits_without_wake_or_reservation(tmp_path, monkeypatch, verdict):
    owner = _owner("catchup", 1)
    _enroll(tmp_path, owner, "session-catchup")
    wire = FakeWire(FakeServer(), owner.endpoint_id)
    gate = AsyncMock(return_value="")
    monkeypatch.setattr(wake_mod, "pre_handoff_gate", gate, raising=False)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda *a: "")
    monkeypatch.setattr(conversation_mod, "screen_item", AsyncMock(return_value=ScreenResult(
        disposition="deliver" if verdict == "allow" else "bounce", receipt=_receipt(verdict),
    )))

    def forbidden(*args, **kwargs):
        raise AssertionError("an explicit read must not wake or reserve another turn")

    monkeypatch.setattr(conversation_mod, "_record_body_reservation", forbidden)
    monkeypatch.setattr(conversation_mod, "_deliver_card", forbidden)
    result = await conversation_mod.handle_conversation_available(
        owner, ConversationFeedRef(CONVERSATION_ID, MESSAGE_ID, 1, False),
        config_dir=tmp_path, auth_store=object(), access=SimpleNamespace(credential_id=1),
        wire=wire, admission_only=True,
    )
    assert result["action"] == ("admitted" if verdict == "allow" else "rejected")
    assert wire.server.admits[0]["verdict"] == verdict
    gate.assert_awaited_once()
    # The finally released the same claim even though no wake was scheduled.
    Store(tmp_path).claim(owner, MESSAGE_ID, "session-catchup", "next-claim", expiry=NOW + 60)


@pytest.mark.asyncio
async def test_shared_gate_refusal_blocks_admit_and_keeps_claim_releasable(tmp_path, monkeypatch):
    owner = _owner("catchup", 1)
    _enroll(tmp_path, owner, "session-catchup")
    wire = FakeWire(FakeServer(), owner.endpoint_id)
    monkeypatch.setattr(wake_mod, "pre_handoff_gate", AsyncMock(return_value="policy_stale"), raising=False)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda *a: "")
    monkeypatch.setattr(conversation_mod, "screen_item", AsyncMock(return_value=ScreenResult(
        disposition="deliver", receipt=_receipt("allow"),
    )))
    result = await conversation_mod.handle_conversation_available(
        owner, ConversationFeedRef(CONVERSATION_ID, MESSAGE_ID, 1, False),
        config_dir=tmp_path, auth_store=object(), access=SimpleNamespace(credential_id=1),
        wire=wire, admission_only=True,
    )
    assert result["code"] == "policy_stale" and wire.server.admits == []
    Store(tmp_path).claim(owner, MESSAGE_ID, "session-catchup", "next-claim", expiry=NOW + 60)


@pytest.mark.asyncio
async def test_missing_3063_gate_fails_closed_before_store_or_wire(tmp_path, monkeypatch):
    monkeypatch.delattr(wake_mod, "pre_handoff_gate", raising=False)
    result = await conversation_mod.handle_conversation_available(
        _owner("catchup", 1), ConversationFeedRef(CONVERSATION_ID, MESSAGE_ID, 1, False),
        config_dir=tmp_path, auth_store=object(), access=None,
        wire=SimpleNamespace(), admission_only=True,
    )
    assert result["code"] == "handoff_gate_unavailable"
