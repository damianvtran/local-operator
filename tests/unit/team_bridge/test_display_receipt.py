"""Display metadata never grants admission or recovers content from an envelope."""

from types import SimpleNamespace

import pytest

from local_operator.team_bridge.display_receipt import validated_display_receipt


def carriage():
    return {
        "bridge_receipt": {
            "v": 1,
            "sender": {"name": "Example member", "kind": "human"},
            "conversation": {"kind": "dm"},
            "attachment_count": 1,
        },
        "bridge_receipt_source": {
            "account_id": "account",
            "tenant_id": "tenant",
            "endpoint_id": "endpoint",
            "conversation_id": "conversation",
            "message_ids": ["message"],
            "conversation_seq": 4,
        },
    }


def owner():
    # A REAL OwnerSnapshot, not a SimpleNamespace stand-in: since PR #4348,
    # ``_deliver_card`` asks the bridge Store for the card's hand-off binding
    # BEFORE dispatch (``Store.card_handoff_bindings``), and the store keys
    # that lookup on the full owner identity (``canonical_base_url`` +
    # account/tenant/endpoint, live ``enrollment_generation``). A partial
    # owner would no longer reach the display path at all. The ids still
    # match ``bridge_receipt_source`` so the source-binding checks are real.
    from local_operator.team_bridge.types import OwnerSnapshot

    return OwnerSnapshot(
        credential_id=1,
        account_id="account",
        tenant_id="tenant",
        endpoint_id="endpoint",
        canonical_base_url="https://api.radienthq.com/v1",
        enrollment_generation=1,
    )


def enrolled_store(config_dir):
    """A disposable Store under ``config_dir`` with ``owner()`` committed.

    The card hand-off gate reads the binding only for the LIVE owner (a fenced
    owner reads every id as unreserved without looking). Enrolling makes the
    tests drive that real lookup: these cards have no body reservation row, so
    the store answers UNRESERVED and the card takes its historical minted-id
    path — the precondition the display assertions rely on, not a bypass.
    """

    from local_operator.team_bridge.store import OwnerReceipt, Store

    store = Store(config_dir)
    store.commit_enrollment(
        owner(), OwnerReceipt("bridge_consent_v1", "ref-display"), cos_session_id=None
    )
    assert store.active_owner() == owner()
    return store


def test_display_receipt_is_metadata_only_and_source_bound():
    result = validated_display_receipt(carriage(), owner=owner())
    assert result == carriage()
    assert "body" not in result["bridge_receipt"]
    assert validated_display_receipt(carriage(), owner=None) == {}
    for field in ("account_id", "tenant_id", "endpoint_id"):
        bad = carriage()
        bad["bridge_receipt_source"][field] = "other"
        assert validated_display_receipt(bad, owner=owner()) == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("body", "private"),
        ("admission", {"state": "admitted"}),
        ("images", [{"attachment": "private"}]),
        ("received_at", "2026-01-01"),
    ],
)
def test_unknown_content_or_authority_fields_refuse_the_optional_receipt(field, value):
    value_carriage = carriage()
    value_carriage["bridge_receipt"][field] = value
    assert validated_display_receipt(value_carriage, owner=owner()) == {}


def test_counts_and_identity_are_bounded_without_role_inference():
    value = carriage()
    value["bridge_receipt"]["sender"] = {"name": "Example (human)"}
    assert validated_display_receipt(value, owner=owner())["bridge_receipt"]["sender"] == {
        "name": "Example (human)"
    }
    for count in (-1, True, 1.2, "1"):
        value["bridge_receipt"]["attachment_count"] = count
        assert validated_display_receipt(value, owner=owner()) == {}


@pytest.mark.asyncio
async def test_receiver_strips_reserved_sender_carriage_and_preserves_model_context(
    tmp_path, monkeypatch
):
    from local_operator.harness.types import StreamEndEvent
    from local_operator.team_bridge import authority
    from local_operator.team_bridge.store import Store
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    session.bridge_runtime = SimpleNamespace(config_dir=tmp_path)
    monkeypatch.setattr(authority, "authority_fence", lambda runtime: None)
    monkeypatch.setattr(Store, "active_owner", lambda store: owner())
    supplied = {"name": "Example member", **carriage()}
    try:
        message = session._peer_custom_message(
            "metadata card, not a body", supplied, custom_type="team_bridge_message"
        )
        assert message.details["bridge_receipt"] == carriage()["bridge_receipt"]
        assert message.details["bridge_receipt_source"] == carriage()["bridge_receipt_source"]
        assert message.details["sender"] == {"name": "Example member"}
        assert "untrusted" in message.details["text"]
        control = session._peer_custom_message(
            "metadata card, not a body",
            {"name": "Example member"},
            custom_type="team_bridge_message",
        )
        assert message.details["text"] == control.details["text"]
        assert message.details["body"] == control.details["body"]
        assert "bridge_receipt" in supplied, "never mutate the caller/spooled dictionary"
        peer = session._peer_custom_message("ordinary peer", supplied, custom_type="peer_message")
        assert "bridge_receipt" not in peer.details
        assert peer.details["sender"] == {"name": "Example member"}
        monkeypatch.setattr(
            authority, "authority_fence", lambda runtime: {"code": "consent_revoked"}
        )
        stale = session._peer_custom_message(
            "still a delivered card", supplied, custom_type="team_bridge_message"
        )
        assert "bridge_receipt" not in stale.details
        assert stale.details["sender"] == {"name": "Example member"}
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_real_metadata_handoff_live_and_replay_keep_the_same_display_without_body(
    tmp_path, monkeypatch
):
    from local_operator.harness.types import StreamEndEvent
    from local_operator.mobile import peer_send
    from local_operator.team_bridge import authority
    from local_operator.team_bridge.connector.conversation import (
        ConversationCardMetadata,
        _deliver_card,
    )
    from local_operator.team_bridge.store import Store
    from tests.unit.mobile.test_projection_condense_fields import make_fold
    from tests.unit.session.test_session import ScriptedStream, make_session

    sent = []

    async def primitive(_context, **kwargs):
        sent.append(kwargs)
        return SimpleNamespace(detail="delivered", state="delivered")

    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda session_id: True)
    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", primitive)
    enrolled_store(tmp_path)
    metadata = ConversationCardMetadata(
        conversation_id="conversation",
        tenant_id="tenant",
        conversation_label="Project notes",
        sender_label="Example member",
        sender_kind="human",
        mention=False,
        message_id="message",
        conversation_seq=4,
        conversation_kind="dm",
        attachments=1,
    )
    assert (
        await _deliver_card(owner(), metadata, session_id="synthetic-session", config_dir=tmp_path)
        == "delivered"
    )
    sender = sent[0]["sender"]
    assert sender["bridge_receipt"]["attachment_count"] == 1
    assert sender["bridge_receipt"]["sender"]["kind"] == "human"
    assert "body" not in sender["bridge_receipt"]
    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    session.bridge_runtime = SimpleNamespace(config_dir=tmp_path)
    monkeypatch.setattr(authority, "authority_fence", lambda runtime: None)
    monkeypatch.setattr(Store, "active_owner", lambda store: owner())
    events = []

    async def emit(event):
        events.append(event)

    monkeypatch.setattr(session, "_emit", emit)
    try:
        message = session._peer_custom_message(
            sent[0]["text"], sender, custom_type="team_bridge_message", entry_id="synthetic-message"
        )
        await session._emit_peer_receipt(message, sender)
        assert "bridge_receipt" not in events[0].sender
        assert "bridge_receipt_source" not in events[0].sender
        live = make_fold()
        # The daemon enables the delivery-aware fold with the negotiated peer
        # certificate protocol. Its legacy default intentionally ignores it.
        live.projection.peer_cycles = 1
        live.fold_event(events[0])
        replay = make_fold()
        replay.fold_history([message])
        for field in ("bridge_receipt", "bridge_receipt_source"):
            assert (
                live.projection.transcript[0].details[field]
                == replay.projection.transcript[0].details[field]
                == message.details[field]
            )
        assert "body" not in live.projection.transcript[0].details["bridge_receipt"]
        wrong = {
            **sender,
            "bridge_receipt_source": {
                **sender["bridge_receipt_source"],
                "account_id": "other-account",
            },
        }
        rejected = session._peer_custom_message(
            "still delivered", wrong, custom_type="team_bridge_message"
        )
        await session._emit_peer_receipt(rejected, wrong)
        assert "bridge_receipt" not in rejected.details
        assert "bridge_receipt_source" not in rejected.details
        assert "bridge_receipt" not in events[-1].sender
        assert "bridge_receipt_source" not in events[-1].sender
        ordinary = session._peer_custom_message("still a peer", sender, custom_type="peer_message")
        await session._emit_peer_receipt(ordinary, sender)
        assert "bridge_receipt" not in events[-1].sender
        assert "bridge_receipt_source" not in events[-1].sender
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_unadmitted_coalesced_card_has_counts_and_ordered_sources_but_no_inherited_identity(
    tmp_path, monkeypatch
):
    from local_operator.team_bridge.connector import conversation
    from local_operator.team_bridge.connector.conversation import (
        ConversationCardMetadata,
    )

    sent = []

    async def deliver(_owner, payload, **kwargs):
        sent.append(kwargs["sender_display"])
        return "delivered"

    monkeypatch.setattr(conversation, "deliver_wake", deliver)
    from local_operator.team_bridge.store import CARD_HANDOFF_UNRESERVED

    store = enrolled_store(tmp_path)
    # The unadmitted recovery card never had a reservation row: the live
    # owner's real binding lookup must say so, or this is not that card.
    assert store.card_handoff_bindings(owner(), "conversation", ["older", "newer"]) == {
        "older": CARD_HANDOFF_UNRESERVED,
        "newer": CARD_HANDOFF_UNRESERVED,
    }
    metadata = ConversationCardMetadata(
        conversation_id="conversation",
        tenant_id="tenant",
        conversation_label="Project notes",
        sender_label="several members",
        sender_kind="mixed",
        mention=False,
        message_id="newer",
        conversation_seq=8,
        attachments=2,
        attachments_note="Images are not admitted",
        coalesced_message_ids=("older", "newer"),
    )
    await conversation._deliver_card(
        owner(), metadata, session_id="synthetic-session", config_dir=tmp_path
    )
    receipt = validated_display_receipt(sent[0], owner=owner())
    assert receipt["bridge_receipt"]["sender"] == {}
    assert receipt["bridge_receipt"]["attachment_count"] == 2
    assert receipt["bridge_receipt_source"]["message_ids"] == ["older", "newer"]
    assert receipt["bridge_receipt_source"]["conversation_seq"] == 8
    assert "admission" not in receipt["bridge_receipt"]
    assert "body" not in receipt["bridge_receipt"]
    assert "images" not in receipt["bridge_receipt"]


def test_session_display_receipt_closed_roundtrip_and_projection():
    from local_operator.team_bridge.display_receipt import projected_display_receipt

    value = carriage()
    value["bridge_receipt"]["sender"] = {
        "kind": "agent",
        "name": "Endpoint (Owner)",
        "agent_session": {"display_label": "Task"},
    }
    checked = validated_display_receipt(value, owner=owner())
    assert checked == value
    assert projected_display_receipt(checked) == checked
    value["bridge_receipt"]["sender"]["agent_session"]["display_label"] = "mutated"
    assert checked["bridge_receipt"]["sender"]["agent_session"]["display_label"] == "Task"
    checked["bridge_receipt"]["sender"]["agent_session"]["authority"] = "spoof"
    assert validated_display_receipt(checked, owner=owner()) == {}
