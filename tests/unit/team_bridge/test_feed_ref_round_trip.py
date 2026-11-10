"""A threaded feed ref may carry routing tags past the old 7-part cap.

The regression: ``ConversationFeedRef.encode`` emits up to eight
``/``-separated parts (4 fixed + sender kind + ``thread_root`` +
``addressed`` + ``relation``), but ``parse_conversation_ref_value``
refused any ref with more than seven parts. ``Store.record_feed_batch``
canonicalises every ref through that parser, so the one hub batch
carrying such a ref rolled back as ``ref_migration_malformed``, the
cursor never advanced, and the hub re-sent the same batch forever.

These cases pin the fix at the grammar: encode and parse must agree
on every shape the encoder can emit, the real ``record_feed_batch``
path must accept it, and the genuinely malformed shapes must stay
refused.
"""

import pytest

from local_operator.team_bridge.store import OwnerReceipt, Store, StoreRefused
from local_operator.team_bridge.types import (
    CONVERSATION_FEED_KIND,
    CONVERSATION_FEED_TYPE,
    FEED_ADDRESSED_VALUES,
    FEED_RELATION_VALUES,
    ConversationFeedRef,
    OwnerSnapshot,
    feed_seq,
    parse_conversation_ref_value,
    project_events_batch,
)

CONVERSATION_ID = "bcv_abc12345"
MESSAGE_ID = "bcmsg_abc"
THREAD_ROOT = "bcmsg_r1"

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example",
    enrollment_generation=1,
).validated()

PREFIX = f"{CONVERSATION_FEED_TYPE}:{CONVERSATION_FEED_KIND}="


def _ref_value(**kwargs) -> str:
    ref = ConversationFeedRef(
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=1,
        mention=True,
        **kwargs,
    ).validated()
    full = ref.encode()
    return full[len(PREFIX) :]


@pytest.mark.parametrize("sender_kind", [None, "human", "agent"])
@pytest.mark.parametrize("thread_root_id", [None, THREAD_ROOT])
@pytest.mark.parametrize("addressed", [None, *sorted(FEED_ADDRESSED_VALUES)])
@pytest.mark.parametrize("sender_relation", [None, *sorted(FEED_RELATION_VALUES)])
def test_encode_parse_round_trip(sender_kind, thread_root_id, addressed, sender_relation):
    """Every encoder shape parses back byte-identically.

    ``project_events_batch`` drops a ``thread_root_id`` without a
    sender kind, so that combination never reaches the encoder from
    the wire; a directly-constructed ref carrying it is not validated
    away, but the parser's own rule (thread root needs sender kind)
    keeps refusing it, so it is excluded here.
    """
    if thread_root_id is not None and sender_kind is None:
        pytest.skip("thread root without sender kind is refused by the grammar")
    value = _ref_value(
        sender_kind=sender_kind,
        thread_root_id=thread_root_id,
        addressed=addressed,
        sender_relation=sender_relation,
    )
    parsed = parse_conversation_ref_value(value)
    assert parsed is not None, value
    assert parsed.encode() == PREFIX + value, value


def test_the_threaded_eight_part_ref_round_trips():
    """The exact shape from the production outage, pinned verbatim."""
    value = (
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human"
        f"/thread_root={THREAD_ROOT}/addressed=direct/relation=owner"
    )
    assert len(value.split("/")) == 8
    parsed = parse_conversation_ref_value(value)
    assert parsed is not None
    assert parsed.encode() == PREFIX + value


def test_project_events_batch_ref_is_storable():
    """The hub event shape that produced the outage reaches the store."""
    _cursor, refs, _resync = project_events_batch(
        {
            "cursor": "5",
            "events": [
                {
                    "type": "message_available",
                    "ids": {
                        "conversation_id": CONVERSATION_ID,
                        "message_id": MESSAGE_ID,
                        "conversation_seq": 1,
                        "mention": True,
                        "sender_kind": "human",
                        "thread_root_id": THREAD_ROOT,
                        "addressed": "direct",
                        "sender_relation": "owner",
                    },
                    "at": "2026-11-10T00:00:00Z",
                }
            ],
        }
    )
    assert len(refs) == 1
    full_ref = refs[0]
    assert full_ref.startswith(PREFIX)
    value = full_ref[len(PREFIX) :]
    assert len(value.split("/")) == 8
    parsed = parse_conversation_ref_value(value)
    assert parsed is not None and parsed.encode() == full_ref


def test_record_feed_batch_accepts_the_eight_part_ref(tmp_path):
    """The real path: the batch records, the cursor advances, the ref is stored."""
    store = Store(tmp_path)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "receipt"))
    value = (
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human"
        f"/thread_root={THREAD_ROOT}/addressed=direct/relation=owner"
    )
    full_ref = PREFIX + value
    store.record_feed_batch(OWNER, feed_seq(5), [full_ref], resync_required=False)
    assert store.sync_state(OWNER)["feed_seq"] == 5
    assert store.drain_feed_refs(OWNER, 5) == [full_ref]


@pytest.mark.parametrize(
    "value",
    [
        # Positional after a tag.
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human/thread_root={THREAD_ROOT}/agent",
        # Empty tag value.
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human/thread_root=",
        # Three positionals plus tags (the old layout never carried tags).
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/NAMES/LABELS/human/addressed=direct",
        # Thread root without a sender kind.
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/thread_root={THREAD_ROOT}",
        # Thread root with a migration alias.
        (
            f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human"
            f"/legacy_alias={'a' * 64}/thread_root={THREAD_ROOT}"
        ),
        # Unknown routing values.
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human/addressed=everybody",
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human/relation=boss",
        # More positionals than any layout has.
        f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human/extra/extra2/addressed=direct",
    ],
)
def test_genuinely_malformed_refs_are_still_refused(value):
    assert parse_conversation_ref_value(value) is None


def test_record_feed_batch_still_refuses_a_malformed_ref(tmp_path):
    store = Store(tmp_path)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "receipt"))
    with pytest.raises(StoreRefused) as excinfo:
        store.record_feed_batch(
            OWNER,
            feed_seq(5),
            [PREFIX + f"{CONVERSATION_ID}/{MESSAGE_ID}/1/1/human/thread_root="],
            resync_required=False,
        )
    assert excinfo.value.code == "ref_migration_malformed"
