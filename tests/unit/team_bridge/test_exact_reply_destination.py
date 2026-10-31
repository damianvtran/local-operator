"""Exact reply binding is independent of threading and the loop/cost guard."""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator.team_bridge import host, reply_guard
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _owner

A = "bcv_aaaaaaaa"
B = "bcv_bbbbbbbb"
MA = "bcmsg_aaaaaaaa"
MB = "bcmsg_bbbbbbbb"


@pytest.fixture
def bound_runtime(tmp_path, monkeypatch):
    owner = _owner()
    (tmp_path / "config.yml").write_text("values:\n  team_bridge:\n    enabled: true\n")
    store = Store(tmp_path)
    store.commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", owner.endpoint_id), cos_session_id="s1"
    )
    for conversation_id, message_id in ((A, MA), (B, MB)):
        store.record_body_reservation(
            owner,
            conversation_id=conversation_id,
            message_id=message_id,
            conversation_seq=1,
            mention=False,
            body_sha256="a" * 64,
            screening_fingerprint="b" * 64,
            policy_epoch=1,
            consent_version="bridge_consent_v1",
            designated_session="s1",
            now=1,
        )
    # Only the authenticated remote admission is stubbed; the local owner and
    # designation fences and both real admitted reservations remain in use.
    monkeypatch.setattr(host, "admission_refusal", lambda runtime, **kw: None)
    client = RecordingClient()
    runtime = BridgeRuntime(cast(Any, client), tmp_path, owner.tenant_id, owner, "s1")
    return runtime, client, store


@pytest.mark.asyncio
async def test_bridge_input_cannot_omit_exact_binding(bound_runtime):
    runtime, client, _ = bound_runtime
    runtime.bridge_input_presented = True
    out = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": B,
            "body": "reply for A",
        },
    )
    assert out.get("code") == "reply_binding_required", (out, client.calls)
    assert client.calls == []


@pytest.mark.asyncio
async def test_reply_a_cannot_be_sent_to_authorized_conversation_b(bound_runtime, monkeypatch):
    runtime, client, _ = bound_runtime
    budgets = []
    original = reply_guard.reserve_post

    def reserve(*args, **kwargs):
        budgets.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(reply_guard, "reserve_post", reserve)
    out = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": B,
            "body": "reply for A",
            "in_reply_to_message_id": MA,
        },
    )
    assert out.get("code") == "reply_destination_mismatch", out
    assert client.calls == []
    assert budgets == []


@pytest.mark.asyncio
@pytest.mark.parametrize("deliveries", [((A, MA), (B, MB)), ((B, MB), (A, MA))])
async def test_each_delivery_and_multiple_replies_bind_independently(bound_runtime, deliveries):
    runtime, client, _ = bound_runtime
    runtime.bridge_input_presented = True
    for conversation, message in (*deliveries, deliveries[0]):
        out = await execute_bridge(
            runtime,
            {
                "op": "channel_post",
                "channel_id": conversation,
                "body": "correct reply",
                "in_reply_to_message_id": message,
            },
        )
        assert out["ok"], out
    assert [c["conversation_id"] for c in client.calls] == [
        c for c, _ in (*deliveries, deliveries[0])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{}, {"thread_root_id": MA}])
async def test_thread_root_is_not_reply_authority(bound_runtime, extra):
    runtime, client, _ = bound_runtime
    runtime.bridge_input_presented = True
    out = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": B,
            "body": "reply",
            **extra,
        },
    )
    assert out.get("code") == "reply_binding_required", out
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge_input,new_post", [(False, False), (True, True)])
async def test_direct_user_and_explicit_new_post_control(bound_runtime, bridge_input, new_post):
    runtime, client, _ = bound_runtime
    runtime.bridge_input_presented = bridge_input
    args = {"op": "channel_post", "channel_id": B, "body": "new topic"}
    if new_post:
        args["new_post"] = True
    out = await execute_bridge(runtime, args)
    assert out["ok"], out
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_new_post_cannot_contradict_exact_reply(bound_runtime):
    runtime, client, _ = bound_runtime
    out = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": A,
            "body": "reply",
            "in_reply_to_message_id": MA,
            "new_post": True,
        },
    )
    assert out.get("code") == "reply_binding_conflict", out
    assert client.calls == []


def test_admitted_lookup_matches_exact_metadata_only(bound_runtime):
    runtime, _, store = bound_runtime
    row = store.admitted_inbound_reservation_readonly(
        runtime.owner,
        in_reply_to_message_id=MA,
        designated_session="s1",
    )
    assert row == {"conversation_id": A, "message_id": MA}


@pytest.mark.asyncio
async def test_unreadable_binding_refuses_before_admission_members_and_budget(
    bound_runtime, monkeypatch
):
    from local_operator.team_bridge.store import StoreRefused

    runtime, client, _ = bound_runtime

    def broken(*args, **kwargs):
        raise StoreRefused("state_unreadable")

    def unexpected(*args, **kwargs):
        pytest.fail("binding refusal must precede admission and budget")

    monkeypatch.setattr(Store, "admitted_inbound_reservation_readonly", broken)
    monkeypatch.setattr(host, "admission_refusal", unexpected)
    monkeypatch.setattr(reply_guard, "reserve_post", unexpected)
    out = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": A,
            "body": "reply",
            "in_reply_to_message_id": MA,
            "to": {"name": "someone"},
        },
    )
    assert out.get("code") == "reply_binding_unavailable", out
    assert client.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("designated_session", "s2"),
        ("enrollment_generation", 2),
        ("endpoint_id", "bep_other"),
        ("tenant_id", "other"),
        ("consent_ref", None),
        ("state", "retired"),
    ],
)
def test_admitted_lookup_refuses_changed_live_enrollment(bound_runtime, field, value):
    runtime, _, store = bound_runtime
    with store._connect() as conn:
        conn.execute(f"UPDATE enrollments SET {field} = ?", (value,))
    assert (
        store.admitted_inbound_reservation_readonly(
            runtime.owner,
            in_reply_to_message_id=MA,
            designated_session="s1",
        )
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", "other"),
        ("endpoint_id", "bep_other"),
        ("account_id", "acc_other"),
        ("credential_id", 2),
        ("canonical_base_url", "https://other.example"),
        ("enrollment_generation", 2),
        ("designated_session", "s2"),
        ("state", "retired"),
    ],
)
def test_admitted_lookup_refuses_drifted_reservation(bound_runtime, field, value):
    runtime, _, store = bound_runtime
    # Synthetic DB corruption tests the redundant full fence, not a permissive
    # insertion helper; the real writer correctly refuses several of these.
    with store._connect() as conn:
        conn.execute(f"UPDATE body_reservations SET {field} = ? WHERE message_id = ?", (value, MA))
    assert (
        store.admitted_inbound_reservation_readonly(
            runtime.owner,
            in_reply_to_message_id=MA,
            designated_session="s1",
        )
        is None
    )


def test_admitted_lookup_refuses_duplicate_message_across_conversations(bound_runtime):
    runtime, _, store = bound_runtime
    with store._connect() as conn:
        conn.execute(
            "UPDATE body_reservations SET message_id = ? WHERE conversation_id = ?", (MA, B)
        )
    assert (
        store.admitted_inbound_reservation_readonly(
            runtime.owner,
            in_reply_to_message_id=MA,
            designated_session="s1",
        )
        is None
    )


@pytest.mark.asyncio
async def test_completed_reservation_allows_later_reply_under_current_fence(bound_runtime):
    runtime, client, store = bound_runtime
    with store._connect() as conn:
        conn.execute("UPDATE body_reservations SET state = 'completed' WHERE message_id = ?", (MA,))
    out = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": A,
            "body": "later clarification",
            "in_reply_to_message_id": MA,
        },
    )
    assert out["ok"], out
    assert len(client.calls) == 1
