"""D4 — ``reply`` answers a message the agent RECEIVED, resolved from the local row.

Diagnosis (post-0.90.17 defect D4): ``reply`` demanded ``to`` before it ever
looked at ``message_id``, although the owner-fenced ``deliveries`` row already
holds the sender and thread for every inbound pair message; and a ``bcmsg_``
(conversation) id was accepted and silently pair-sent, so a DM answer landed in
the peer inbox and never in the DM. These pin the repaired shape:

* ``reply`` with a received ``bmsg_`` id and no ``to`` reaches the original
  sender in the message's thread;
* an explicit ``to`` (or ``thread_id``) that disagrees with that row is refused;
* a ``bcmsg_`` id is refused with the ``channel_post`` call that answers in the
  conversation, and nothing is posted anywhere;
* an unknown id, or one received by a DIFFERENT owner, is refused — reply stays
  bound to the untrusted-inbox rule "only what this agent actually received".

Every refusal is local: no outbox intent, no HTTP request, no admission read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from local_operator.team_bridge.types import MessageRef
from tests.unit.team_bridge.test_client import FakeStore, _ok

HUB = "https://hub.example/v1"
CONVERSATION_ID = "bcv_aaefeb388bc842d4"
CONVERSATION_MESSAGE_ID = "bcmsg_aaefeb388bc842d4"

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url=HUB,
    enrollment_generation=1,
)


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture
def admissions(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Open both gates, recording every admission check.

    The fence and admission states are covered elsewhere; here an admission
    call is the observable "this destination got past local validation", so a
    refusal that reaches it would have scheduled a status read.
    """

    seen: list[object] = []

    def admission(runtime, *, is_read=False):
        seen.append(runtime)
        return None

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", admission)
    return seen


def _runtime(
    tmp_path: Path, requests: list[httpx.Request], owner: OwnerSnapshot = OWNER
) -> BridgeRuntime:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok({"message": {"id": "bmsg_reply"}}, status=201)

    return BridgeRuntime(
        client=TeamBridgeClient(
            None, base_url=HUB, store=FakeStore(), transport=httpx.MockTransport(handler)
        ),
        config_dir=tmp_path,
        tenant_id=owner.tenant_id,
        owner=owner,
        session_id="s1",
    )


def _received(tmp_path: Path, **sender: str | None) -> Store:
    """A committed owner holding one inbound delivery ``bmsg_in1`` in ``bth_1``."""

    store = Store(tmp_path)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "bep_1"), cos_session_id="s1")
    ids = {"sender_account_id": "acc_2", "sender_endpoint_id": "bep_2", **sender}
    ref = MessageRef(
        message_id="bmsg_in1",
        thread_id="bth_1",
        tenant_id="tnt_1",
        message_type="note",
        body_hash=None,
        seq=3,
        origin="agent",
        delivery_state="unread",
        state_at=None,
        sender_account_id=ids["sender_account_id"],
        sender_endpoint_id=ids["sender_endpoint_id"],
        recipient_account_id="acc_1",
        recipient_endpoint_id="bep_1",
    )
    store.merge_page(OWNER, [ref], None)
    return store


def _pending(tmp_path: Path, runtime: BridgeRuntime) -> list[Any]:
    from local_operator.team_bridge.outbox import Outbox

    return Outbox(tmp_path).pending(runtime.key())


@pytest.mark.asyncio
@pytest.mark.parametrize("with_thread", [False, True])
async def test_reply_without_to_reaches_the_original_sender_in_its_thread(
    tmp_path: Path, admissions, with_thread
):
    _received(tmp_path)
    requests: list[httpx.Request] = []
    args = {"op": "reply", "message_id": "bmsg_in1", "text": "ok"}
    if with_thread:
        args["thread_id"] = "bth_1"
    out = await execute_bridge(_runtime(tmp_path, requests), args)
    assert out["ok"] is True, out
    assert [r.url.path for r in requests] == ["/v1/messaging/messages"]
    body = json.loads(requests[0].content)
    assert body["to"] == {"endpoint_id": "bep_2"} and body["thread_id"] == "bth_1"


@pytest.mark.asyncio
async def test_reply_falls_back_to_the_sender_account_when_no_endpoint_is_stored(
    tmp_path: Path, admissions
):
    _received(tmp_path, sender_endpoint_id=None)
    requests: list[httpx.Request] = []
    out = await execute_bridge(
        _runtime(tmp_path, requests), {"op": "reply", "message_id": "bmsg_in1", "text": "ok"}
    )
    assert out["ok"] is True, out
    assert json.loads(requests[0].content)["to"] == {"account_id": "acc_2"}


@pytest.mark.asyncio
@pytest.mark.parametrize("to", [{"endpoint_id": "bep_2"}, {"account_id": "acc_2"}])
async def test_an_explicit_to_naming_the_sender_is_still_accepted(tmp_path: Path, admissions, to):
    _received(tmp_path)
    requests: list[httpx.Request] = []
    out = await execute_bridge(
        _runtime(tmp_path, requests),
        {"op": "reply", "message_id": "bmsg_in1", "to": to, "text": "ok"},
    )
    assert out["ok"] is True, out
    body = json.loads(requests[0].content)
    assert body["to"] == to and body["thread_id"] == "bth_1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"to": {"endpoint_id": "bep_9"}},
        {"to": {"account_id": "acc_9"}},
        {"thread_id": "bth_other"},
    ],
)
async def test_a_reply_that_conflicts_with_the_received_message_is_refused(
    tmp_path: Path, admissions, extra
):
    _received(tmp_path)
    requests: list[httpx.Request] = []
    runtime = _runtime(tmp_path, requests)
    out = await execute_bridge(
        runtime, {"op": "reply", "message_id": "bmsg_in1", "text": "ok", **extra}
    )
    assert out["ok"] is False and out["code"] == "reply_conflict", out
    assert requests == [] and admissions == [] and _pending(tmp_path, runtime) == []


@pytest.mark.asyncio
async def test_a_to_the_row_cannot_check_is_refused_not_trusted(tmp_path: Path, admissions):
    """The row knows the sender only by account: an endpoint ``to`` is unverifiable."""

    _received(tmp_path, sender_endpoint_id=None)
    requests: list[httpx.Request] = []
    out = await execute_bridge(
        _runtime(tmp_path, requests),
        {"op": "reply", "message_id": "bmsg_in1", "to": {"endpoint_id": "bep_2"}, "text": "ok"},
    )
    assert out["ok"] is False and out["code"] == "reply_conflict", out
    assert requests == []


@pytest.mark.asyncio
async def test_a_conversation_message_is_refused_with_the_channel_post_call(
    tmp_path: Path, admissions
):
    store = _received(tmp_path)
    store.record_body_reservation(
        OWNER,
        conversation_id=CONVERSATION_ID,
        message_id=CONVERSATION_MESSAGE_ID,
        conversation_seq=4,
        mention=True,
        body_sha256="a" * 64,
        screening_fingerprint="b" * 64,
        policy_epoch=1,
        consent_version="bridge_consent_v1",
        designated_session="s1",
        now=1.0,
    )
    requests: list[httpx.Request] = []
    runtime = _runtime(tmp_path, requests)
    out = await execute_bridge(
        runtime,
        {
            "op": "reply",
            "message_id": CONVERSATION_MESSAGE_ID,
            "to": {"account_id": "acc_2"},
            "thread_id": "bth_x",
            "text": "ok",
        },
    )
    assert out["ok"] is False and out["code"] == "conversation_message", out
    assert f"channel_post channel_id={CONVERSATION_ID}" in out["error"]
    assert "thread_root_id" in out["error"]
    assert len(out["error"]) < 200, "the refusal is billed into the agent's context"
    assert requests == [] and admissions == [] and _pending(tmp_path, runtime) == []


@pytest.mark.asyncio
async def test_an_unreserved_conversation_message_is_refused_with_the_generic_route(
    tmp_path: Path, admissions
):
    _received(tmp_path)
    requests: list[httpx.Request] = []
    out = await execute_bridge(
        _runtime(tmp_path, requests),
        {"op": "reply", "message_id": CONVERSATION_MESSAGE_ID, "text": "ok"},
    )
    assert out["ok"] is False and out["code"] == "conversation_message", out
    assert "channel_post" in out["error"] and "conversation_id" in out["error"]
    assert requests == [] and admissions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("to", [None, {"endpoint_id": "bep_2"}])
async def test_a_reply_to_an_unknown_message_is_refused(tmp_path: Path, admissions, to):
    _received(tmp_path)
    requests: list[httpx.Request] = []
    args: dict[str, Any] = {
        "op": "reply",
        "message_id": "bmsg_never",
        "thread_id": "bth_1",
        "text": "ok",
    }
    if to is not None:
        args["to"] = to
    out = await execute_bridge(_runtime(tmp_path, requests), args)
    assert out["ok"] is False and out["code"] == "unknown_message", out
    assert requests == [] and admissions == []


@pytest.mark.asyncio
async def test_a_message_received_by_another_owner_does_not_resolve(tmp_path: Path, admissions):
    """The row is owner-keyed and generation-fenced: a runtime bound to a
    different endpoint cannot borrow this owner's sender as a destination."""

    _received(tmp_path)
    other = OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id="tnt_1",
        endpoint_id="bep_other",
        canonical_base_url=HUB,
        enrollment_generation=1,
    )
    requests: list[httpx.Request] = []
    out = await execute_bridge(
        _runtime(tmp_path, requests, owner=other),
        {"op": "reply", "message_id": "bmsg_in1", "text": "ok"},
    )
    assert out["ok"] is False and out["code"] == "unknown_message", out
    assert requests == [] and admissions == []


@pytest.mark.asyncio
async def test_an_absent_store_resolves_nothing_and_stays_absent(tmp_path: Path, admissions):
    requests: list[httpx.Request] = []
    out = await execute_bridge(
        _runtime(tmp_path, requests), {"op": "reply", "message_id": "bmsg_in1", "text": "ok"}
    )
    assert out["ok"] is False and out["code"] == "unknown_message", out
    assert list(tmp_path.iterdir()) == [], "a read must never create the store"
