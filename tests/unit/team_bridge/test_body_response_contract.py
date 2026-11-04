"""The conversation-body response the connector reads, pinned to the server's bytes.

WHY THIS FILE EXISTS. The Q-1 fix reads `thread_follower` and `conversation_kind`
off the body response, and the first version of it believed a FAKE: `Wire` in
`test_wake_every_message.py` hand-added both fields to a payload the server did
not send, so the tests proved a shape that never reached production (round 6,
M1). A fake is only worth its green when something pins it to the real wire.

WHAT IS PINNED HERE. `fixtures/team_bridge/conversation-body-response.json` is
built from `BridgeConversationBodyResponse`'s json tags at agent-server
`2501f5b1` (origin/feat/bridge-wake-all-members), including the two fields #255
adds. These tests hold the connector and the test fake to that file:

* every key the connector reads exists in the fixture and vice versa — an
  invented or renamed key fails here rather than in production;
* `thread_follower` is a POINTER on the wire: `false` is emitted, `true` is
  emitted, and a root carries NO key at all, which the connector must read as
  "the server did not say" (not as "false");
* the thread ROOT is not part of this response at all: it rides the feed ref, so
  the card binds a body to the ref's root. A fixture that invented `thread_root_id`
  would stop being the server's marshalled output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge.connector.conversation import (
    ConversationCardMetadata,
    _card_metadata,
    build_conversation_card,
    card_directed,
    wake_is_addressed,
)
from local_operator.team_bridge.types import ConversationFeedRef, OwnerSnapshot

CONV = "bcv_bodyresp01"
ROOT = "bcmsg_root0001"

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "team_bridge"
    / "conversation-body-response.json"
)


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _allowed_keys() -> set[str]:
    data = _fixture()
    return set(data["payload_required"]) | set(data["payload_optional"])


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_a",
        tenant_id="tnt_1",
        endpoint_id="bep_a",
        canonical_base_url="https://api.example/v1",
        enrollment_generation=1,
    )


def _ref(*, thread_root_id: str | None) -> ConversationFeedRef:
    return ConversationFeedRef(
        conversation_id=CONV,
        message_id="bcmsg_replyfollow1",
        conversation_seq=7,
        mention=False,
        sender_kind="human",
        thread_root_id=thread_root_id,
    ).validated()


def test_every_fixture_case_is_the_shape_the_server_marshals() -> None:
    """No case may carry a key outside the struct's tags, and none may miss one."""

    data = _fixture()
    allowed = _allowed_keys()
    for name, case in data["cases"].items():
        assert set(case) <= allowed, (name, set(case) - allowed)
        for required in data["payload_required"]:
            assert required in case, (name, required)


def test_a_root_carries_no_follower_key_but_a_reply_carries_both_states() -> None:
    """The pointer rule: absent is not false, and both are on the wire."""

    cases = _fixture()["cases"]
    assert "thread_follower" not in cases["root_message"]
    assert cases["followed_thread_reply"]["thread_follower"] is True
    assert cases["unfollowed_thread_reply"]["thread_follower"] is False
    # The thread root is NOT on this response: it reaches the card from the ref.
    assert not any("thread_root_id" in case for case in cases.values())


def test_the_connector_reads_exactly_the_keys_the_fixture_declares() -> None:
    """The reader's own key set is the fixture's, so neither can drift alone."""

    data = _fixture()
    for name, case in data["cases"].items():
        assert set(case) <= _allowed_keys(), name
    followed = data["cases"]["followed_thread_reply"]
    metadata = _card_metadata(_owner(), _ref(thread_root_id=ROOT), followed)
    # The two facts the card needs survive the read; nothing else is invented.
    assert metadata.thread_follower is True
    assert metadata.conversation_kind == "channel"
    assert metadata.conversation_label == "#agents"
    assert metadata.sender_kind == "human"


@pytest.mark.parametrize(
    ("case", "directed", "addressed", "expect_line"),
    [
        ("followed_thread_reply", "thread", True, "expect: reply"),
        # Its sender is an AGENT, so the ambient line carries the agent caution.
        (
            "unfollowed_thread_reply",
            "none",
            False,
            "expect: weigh relevance; no_reply unless it asks you directly",
        ),
        (
            "root_message",
            "none",
            False,
            "expect: weigh relevance; no_reply if nothing is needed",
        ),
        (
            "dm_root_message",
            "none",
            False,
            "expect: weigh relevance; no_reply if nothing is needed",
        ),
    ],
)
def test_the_fixtures_drive_the_card_and_the_brake(
    case: str, directed: str, addressed: bool, expect_line: str
) -> None:
    """One response -> the card's directedness AND the brake's decision.

    The two predicates differ on purpose: the card may derive `thread` from a
    thread root when the server says nothing, but the brake only lifts on an
    explicit mention or an explicit follower bit (round 6, M2).
    """

    body = _fixture()["cases"][case]
    # The root comes from the feed ref, exactly as in production.
    thread_root = ROOT if case.endswith("thread_reply") else None
    metadata = _card_metadata(_owner(), _ref(thread_root_id=thread_root), body)
    assert card_directed(metadata) == directed
    assert wake_is_addressed(metadata) is addressed
    card = build_conversation_card(metadata)
    assert expect_line in card.splitlines()


def test_an_older_server_that_sends_no_bit_keeps_the_card_derivation_only() -> None:
    """The asymmetry the review asked for, stated as a test.

    A body response without `thread_follower` (every server before #255) still
    renders `directed: thread` for a thread reply — that is the reader's best
    guess and costs a paragraph at worst. It must NOT lift the reply brake: with
    no bit, a human reply in a thread the agent never joined is indistinguishable
    from a followed one.
    """

    root = _ref(thread_root_id=ROOT)
    legacy = {
        "message_id": "bcmsg_replyfollow1",
        "body": "an older server, no follower bit",
        "body_hash": "e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4",
        "conversation_name": "#agents",
        "sender_label": "Ada",
        "sender_kind": "human",
    }
    metadata = _card_metadata(_owner(), root, legacy)
    assert metadata.thread_follower is None
    assert card_directed(metadata) == "thread"
    assert wake_is_addressed(metadata) is False
    assert isinstance(metadata, ConversationCardMetadata)


def test_the_fake_connector_body_matches_the_fixture() -> None:
    """`Wire`'s fake second source of truth is this fixture, key for key."""

    import tests.unit.team_bridge.test_wake_every_message as wake_tests

    hub = wake_tests.Hub(members=("bep_a",))
    root_msg = hub.post(None, "a root message", kind="channel")
    reply_msg = hub.post(None, "a followed reply", kind="channel")
    reply_msg.thread_follower = True
    wire = wake_tests.Wire(hub, "bep_a")

    import asyncio

    root_body = asyncio.run(wire.conversation_body(CONV, root_msg.id))
    reply_body = asyncio.run(wire.conversation_body(CONV, reply_msg.id))
    allowed = _allowed_keys()
    assert set(root_body) <= allowed, set(root_body) - allowed
    assert set(reply_body) <= allowed, set(reply_body) - allowed
    # The fake must express the POINTER rule too, or it hides the absent case.
    assert "thread_follower" not in root_body
    assert reply_body["thread_follower"] is True
