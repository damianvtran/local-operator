"""``bridge(op="ack")`` posts working/done to the hub (Team Bridge requirements #2/#4).

Before this, ``ack working|done`` only took and released a local SQLite claim:
the hub's delivery row stayed at ``read`` and its append-only ``state_events``
audit never showed the request being worked or finished. These tests drive the
REAL tool and the REAL ``TeamBridgeClient`` against a stub hub that mirrors
agent-server's single-step machine (``bridge_ack_service.go``
``evaluateBridgeAck``: read→working→done, an agreeing same-state repeat is a
200 no-op, anything else a 409 ``invalid_transition``), and count the POSTs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge import pull
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import (
    ACK_CONFIRMED,
    ACK_CONFLICT,
    ACK_PENDING,
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
)
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from local_operator.team_bridge.types import ScreeningReceipt
from tests.unit.team_bridge.test_client import FakeStore, _err, _ok

TENANT = "ten_1"
MSG = "bmsg_1"


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fence and admission gate have their own suites; these tests are
    # about what the ack op sends once admitted (same seam as test_h5d_tool).
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


class StubHub:
    """The hub's ack machine for ONE delivery, recording every ack POST."""

    _MOVES = {("unread", "read"), ("read", "working"), ("working", "done")}

    def __init__(self, state: str = "read") -> None:
        self.state = state
        self.state_events: list[str] = [state]
        self.posts: list[dict[str, Any]] = []
        #: Scripted answers consumed before the machine runs: a status code to
        #: answer with, or an exception to raise from the transport.
        self.script: list[Any] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path.endswith(f"/{MSG}/ack")
        body = json.loads(request.content)
        self.posts.append(body)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, int):
                return _err(step, "unavailable" if step >= 500 else "invalid_transition")
        target = body["state"]
        if target == self.state:
            return _ok({"delivery": {"state": target}})
        if (self.state, target) not in self._MOVES:
            return _err(409, "invalid_transition")
        self.state = target
        self.state_events.append(target)
        return _ok({"delivery": {"state": target}})


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id=TENANT,
        endpoint_id="bep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _runtime(tmp_path, hub: StubHub, session_id: str = "s1") -> BridgeRuntime:
    owner = _owner()
    store = Store(tmp_path)
    if (
        store.resolve_owner(
            credential_id=1,
            account_id="acc_1",
            tenant_id=TENANT,
            endpoint_id="bep_1",
            canonical_base_url=owner.canonical_base_url,
        )
        is None
    ):
        store.commit_enrollment(owner, OwnerReceipt(consent_version="cv-1", consent_ref="cr-1"))
    client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(hub),
    )
    return BridgeRuntime(
        client=client, config_dir=tmp_path, tenant_id=TENANT, owner=owner, session_id=session_id
    )


async def _ack(runtime: BridgeRuntime, state: str, message_id: str = MSG) -> dict[str, Any]:
    return await execute_bridge(runtime, {"op": "ack", "message_id": message_id, "state": state})


@pytest.mark.asyncio
async def test_read_working_done_posts_once_each_in_order(tmp_path) -> None:
    hub = StubHub("unread")
    runtime = _runtime(tmp_path, hub)
    receipt = ScreeningReceipt("allow", "m", "bridge-screen-v1", "a" * 64, "c" * 64, ())
    read = await pull.ack(
        runtime.owner,
        MSG,
        "read",
        receipt,
        config_dir=tmp_path,
        access=await runtime.client.current_access(),
        client=cast(pull.WireClient, runtime.client),
    )
    assert read.state == ACK_CONFIRMED

    working = await _ack(runtime, "working")
    assert working["ok"] is True and working["hub_ack"] == "confirmed"
    assert MSG in runtime.claims
    done = await _ack(runtime, "done")
    assert done["ok"] is True and done["state"] == "done" and done["hub_ack"] == "confirmed"
    assert MSG not in runtime.claims

    assert [p["state"] for p in hub.posts] == ["read", "working", "done"]
    assert hub.state_events == ["unread", "read", "working", "done"]
    # No screening tuple rides a progress ack: the hub checks its stored one.
    assert "screening" in hub.posts[0]
    assert all("screening" not in p for p in hub.posts[1:])
    assert len({p["client_ref"] for p in hub.posts}) == 3


@pytest.mark.asyncio
async def test_a_repeated_working_reuses_its_ref_and_is_idempotent(tmp_path) -> None:
    hub = StubHub("read")
    hub.script = [503]  # the first POST is ambiguous: the intent stays pending
    runtime = _runtime(tmp_path, hub)

    first = await _ack(runtime, "working")
    assert first["ok"] is True and first["hub_ack"] == "pending"
    second = await _ack(runtime, "working")
    assert second["ok"] is True and second["hub_ack"] == "confirmed"
    third = await _ack(runtime, "working")  # confirmed: replay sends nothing
    assert third["ok"] is True and third["hub_ack"] == "confirmed"

    assert [p["state"] for p in hub.posts] == ["working", "working"]
    assert hub.posts[0]["client_ref"] == hub.posts[1]["client_ref"]
    assert hub.state_events == ["read", "working"]


@pytest.mark.asyncio
async def test_a_hub_409_on_done_surfaces_typed_and_releases_the_claim(tmp_path) -> None:
    hub = StubHub("read")
    runtime = _runtime(tmp_path, hub)
    assert (await _ack(runtime, "working"))["hub_ack"] == "confirmed"
    hub.state = "read"  # the hub row is NOT at working: done would be an illegal jump

    done = await _ack(runtime, "done")
    assert done["ok"] is False and done["code"] == "invalid_transition"
    assert MSG not in runtime.claims
    store = Store(tmp_path)
    assert store.find_intent_state(_owner(), MSG, "done") == ACK_CONFLICT
    assert [p["state"] for p in hub.posts] == ["working", "done"]


@pytest.mark.asyncio
async def test_working_on_an_unread_delivery_is_refused_typed_and_unclaimed(tmp_path) -> None:
    hub = StubHub("unread")
    runtime = _runtime(tmp_path, hub)

    working = await _ack(runtime, "working")
    assert working["ok"] is False and working["code"] == "invalid_transition"
    assert MSG not in runtime.claims
    # The claim was released, so another session is not blocked for an hour.
    other = _runtime(tmp_path, hub, session_id="s2")
    assert Store(tmp_path).claim(_owner(), MSG, "s2", "tok-2", 9e12) is None
    assert other.claims == {}


@pytest.mark.asyncio
async def test_done_replays_a_pending_working_first(tmp_path) -> None:
    hub = StubHub("read")
    hub.script = [503]
    runtime = _runtime(tmp_path, hub)
    assert (await _ack(runtime, "working"))["hub_ack"] == "pending"

    done = await _ack(runtime, "done")
    assert done["ok"] is True and done["hub_ack"] == "confirmed"
    assert [p["state"] for p in hub.posts] == ["working", "working", "done"]
    assert hub.posts[0]["client_ref"] == hub.posts[1]["client_ref"]
    assert hub.state_events == ["read", "working", "done"]


@pytest.mark.asyncio
async def test_done_holds_the_claim_while_working_cannot_reach_the_hub(tmp_path) -> None:
    hub = StubHub("read")
    hub.script = [503, 503]
    runtime = _runtime(tmp_path, hub)
    await _ack(runtime, "working")

    done = await _ack(runtime, "done")
    assert done["ok"] is False and MSG in runtime.claims
    assert [p["state"] for p in hub.posts] == ["working", "working"]  # done never sent


@pytest.mark.asyncio
async def test_a_crash_between_intent_and_post_replays_the_same_ref(tmp_path, monkeypatch) -> None:
    hub = StubHub("read")
    runtime = _runtime(tmp_path, hub)
    store = Store(tmp_path)
    original_ack = runtime.client.ack
    seen_refs = []

    async def die_before_post(message_id, **kwargs):
        # The intent has committed, but the wire request has not begun.
        assert store.find_intent_state(_owner(), MSG, "working") == ACK_PENDING
        seen_refs.append(kwargs["client_ref"])
        raise asyncio.CancelledError()

    monkeypatch.setattr(runtime.client, "ack", die_before_post)
    with pytest.raises(asyncio.CancelledError):
        await _ack(runtime, "working")
    assert hub.posts == []
    stored_ref = store.intent_client_ref(_owner(), MSG, "working")
    monkeypatch.setattr(runtime.client, "ack", original_ack)

    again = await _ack(runtime, "working")
    assert again["ok"] is True and again["hub_ack"] == "confirmed"
    assert seen_refs == [stored_ref]
    assert [p["client_ref"] for p in hub.posts] == [stored_ref]
    assert store.find_intent_state(_owner(), MSG, "working") == ACK_CONFIRMED


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["working", "done"])
async def test_conversation_ids_are_refused_with_zero_posts(tmp_path, state) -> None:
    hub = StubHub("read")
    runtime = _runtime(tmp_path, hub)

    out = await _ack(runtime, state, message_id="bcmsg_abc123")
    assert out["ok"] is False and out["code"] == "conversation_message"
    assert hub.posts == [] and runtime.claims == {}
    assert Store(tmp_path).intent_client_ref(_owner(), "bcmsg_abc123", state) is None


def test_progress_intents_refuse_a_receipt_and_read_still_requires_one(tmp_path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt(consent_version="cv-1", consent_ref="cr-1"))
    with pytest.raises(StoreRefused) as no_receipt:
        store.record_ack_intent(owner, MSG, "read", None, "ref-r")
    assert no_receipt.value.code == "receipt_required"

    receipt = ScreeningReceipt("allow", "m", "bridge-screen-v1", "a" * 64, "c" * 64, ())
    with pytest.raises(StoreRefused) as extra_receipt:
        store.record_ack_intent(owner, MSG, "working", receipt, "ref-w")
    assert extra_receipt.value.code == "receipt_not_allowed"

    first = store.record_ack_intent(owner, MSG, "working", None, "ref-w")
    assert store.record_ack_intent(owner, MSG, "working", None, "ref-w") == first
    with pytest.raises(StoreRefused) as other_ref:
        store.record_ack_intent(owner, MSG, "working", None, "ref-other")
    assert other_ref.value.code == "intent_conflict"
    intent = store.ack_intent(owner, first)
    assert intent is not None and intent["receipt"] is None and intent["target"] == "working"


@pytest.mark.asyncio
async def test_pending_done_keeps_claim_and_retries_same_ref(tmp_path) -> None:
    hub = StubHub("read")
    runtime = _runtime(tmp_path, hub)
    await _ack(runtime, "working")
    hub.script = [503]
    first = await _ack(runtime, "done")
    assert first["ok"] is False and MSG in runtime.claims
    second = await _ack(runtime, "done")
    assert second["ok"] is True and MSG not in runtime.claims
    assert [p["state"] for p in hub.posts] == ["working", "done", "done"]
    assert hub.posts[1]["client_ref"] == hub.posts[2]["client_ref"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["working", "done"])
async def test_a_stale_claim_token_cannot_post_progress(tmp_path, state) -> None:
    hub = StubHub("read")
    runtime = _runtime(tmp_path, hub)
    Store(tmp_path).claim(_owner(), MSG, "s2", "successor-token", 9e12)
    runtime.claims[MSG] = "stale-token"
    outcome = await _ack(runtime, state)
    assert outcome["ok"] is False and outcome["code"] == "not_holder"
    assert hub.posts == []


@pytest.mark.asyncio
async def test_changed_signed_in_account_cannot_post_for_enrolled_owner(tmp_path) -> None:
    from tests.unit.team_bridge.test_client import FakeRow

    hub = StubHub("read")
    runtime = _runtime(tmp_path, hub)
    runtime.client = TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore([FakeRow(2, data={"access": "fixture-token", "account_id": "acc_other"})]),
        transport=httpx.MockTransport(hub),
    )
    result = await _ack(runtime, "working")
    assert result["ok"] is False and result["code"] == "owner_mismatch"
    assert hub.posts == [] and runtime.claims == {}
