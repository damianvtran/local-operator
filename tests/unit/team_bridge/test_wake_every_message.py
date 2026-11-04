"""Wake-on-every-message: card, durable ref, coalescing, reply guard, two agents.

The wake contract (2026-10-30) wakes an agent member for EVERY message in a
conversation it belongs to and leaves relevance to the agent. That is only safe
with three bounds, each pinned here at its real seam:

* the CARD says whether a reply is expected (mention / followed thread) or the
  message is ambient (weigh it, end with ``no_reply``), and cautions against
  answering an agent's ambient chatter;
* the connector COALESCES a burst into one wake per (endpoint, conversation)
  window, while every message is still screened on its own;
* ``channel_post``/``dm`` NEVER cap replies (operator, 2026-11-03): every post
  is delivered, an exact double send within 10 s is skipped, and a long
  one-sided run carries a one-line warning.

The last test drives two agents whose policy is "reply to every wake" through
the real connector iteration and the real bridge tool against a fake hub that
applies the server's run caps, and once more against an OLD hub with no caps,
and asserts the server's brake stops it while the client refuses nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge import pull as pull_mod
from local_operator.team_bridge import reply_guard
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.connector import coalesce as coalesce_mod
from local_operator.team_bridge.connector import conversation as conversation_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.connector.conversation import (
    ConversationCardMetadata,
    build_conversation_card,
    card_directed,
)
from local_operator.team_bridge.screening import ScreenResult
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from local_operator.team_bridge.types import (
    ConversationFeedRef,
    parse_conversation_ref_value,
    project_events_batch,
)
from tests.unit.team_bridge.test_conversation_delivery import (
    _enroll,
    _owner,
    _persist_fake_receipt,
    _receipt,
)

CONV = "bcv_wakeall01"
ROOT = "bcmsg_root0001"
PREFIX = "message_available:conversation_id="


def _card(**overrides: Any) -> str:
    base: dict[str, Any] = dict(
        conversation_id=CONV,
        tenant_id="tnt_1",
        conversation_label="#agents",
        sender_label="Ada",
        sender_kind="human",
        mention=False,
        message_id="bcmsg_m3",
        conversation_seq=3,
    )
    base.update(overrides)
    return build_conversation_card(ConversationCardMetadata(**base))


# -- card --------------------------------------------------------------------


def test_a_mention_card_expects_a_reply() -> None:
    lines = _card(mention=True).splitlines()
    assert "mention: yes" in lines
    # `mention` already says `directed: mention`: the same fact is not billed
    # twice on every wake (review round 1, m4).
    assert not any(line.startswith("directed:") for line in lines)
    assert "expect: reply" in lines
    assert not any("no_reply" in line for line in lines)


def test_an_unmentioned_card_says_weigh_it_and_no_reply_is_the_way_out() -> None:
    lines = _card(mention=False).splitlines()
    assert "mention: no" in lines
    assert not any(line.startswith("directed:") for line in lines)
    assert "expect: weigh relevance; no_reply if nothing is needed" in lines
    assert "expect: reply" not in lines
    # The removed conversation-kind line: nothing derives it any more, and the
    # ref no longer carries it (review round 1, m3/m4).
    assert not any(line.startswith("conversation_kind:") for line in lines)


def test_an_unmentioned_agent_card_asks_for_no_reply_unless_asked_directly() -> None:
    # Two agents answering each other's ambient chatter is the loop the whole
    # contract guards against; one line, not two (review m4).
    lines = _card(sender_kind="agent", sender_label="Bea's agent").splitlines()
    assert "expect: weigh relevance; no_reply unless it asks you directly" in lines


def test_a_mentioning_agent_card_carries_no_caution() -> None:
    # An agent that @-mentions you asked you something: answer it.
    lines = _card(sender_kind="agent", mention=True).splitlines()
    assert "expect: reply" in lines
    assert not any("no_reply" in line for line in lines)


def test_a_thread_reply_is_directed_without_a_mention() -> None:
    """A non-mentioned thread row IS directed: only followers are woken for it.

    Derived from the thread root (review round 1, m3), never carried as its own
    ref tag — the deref stays the shape the previous release already reads.
    """

    lines = _card(thread_root_id=ROOT, sender_kind="agent").splitlines()
    assert "mention: no" in lines
    assert "directed: thread" in lines
    assert "expect: reply" in lines
    assert f"thread_root_id: {ROOT} (reply to)" in lines
    # ONE line names the thread: `reply_to_message_id:` said the same id again
    # (review m4) and is folded into the `(reply to)` marker.
    assert not any(line.startswith("reply_to_message_id:") for line in lines)
    # The thread route stays directly under thread_root_id, above the footer.
    assert lines[-3].startswith("thread_root_id: ")
    assert lines[-2].startswith("reply in thread: ")


def test_a_thread_reply_without_a_mention_is_never_marked_as_one() -> None:
    assert "mention: no" in _card(thread_root_id=ROOT).splitlines()


def test_the_card_carries_no_line_that_repeats_another() -> None:
    """The card is billed per turn: no line repeats what another already says.

    Measured with the same fields through origin/main's builder and this one:
    root card 421 → 483 characters (+14.7%), thread reply 424 → 466 (+9.9%);
    before this compaction the growth was +33% (review round 1, m4). These
    assertions are the structural half of that budget, so a later line cannot
    quietly reintroduce the growth.
    """

    root = _card(sender_kind="agent", sender_label="Bea's agent").splitlines()
    reply = _card(sender_kind="agent", sender_label="Bea's agent", thread_root_id=ROOT).splitlines()
    for lines in (root, reply):
        # `directed:` appears only for `thread`; mention and none are already
        # stated by the `mention:` line, and `reply_to_message_id:` repeated the
        # thread root verbatim.
        assert sum(1 for line in lines if line.startswith("directed:")) <= 1
        assert not any(line.startswith("reply_to_message_id:") for line in lines)
        assert not any(line.startswith("conversation_kind:") for line in lines)
    assert [line for line in reply if line.startswith("thread_root_id:")] == [
        f"thread_root_id: {ROOT} (reply to)"
    ]


def test_a_coalesced_card_names_every_message_and_binds_the_newest() -> None:
    lines = _card(
        message_id="bcmsg_m3", coalesced_message_ids=("bcmsg_m1", "bcmsg_m2", "bcmsg_m3")
    ).splitlines()
    assert "message_ids: bcmsg_m1,bcmsg_m2,bcmsg_m3" in lines
    assert "message_id: bcmsg_m3" in lines


def test_a_single_message_card_has_no_message_ids_line() -> None:
    assert "message_ids:" not in _card(coalesced_message_ids=("bcmsg_m3",))


def test_merging_a_mixed_burst_never_stamps_the_agent_caution_on_a_human() -> None:
    def meta(mid: str, seq: int, kind: str, label: str, mention: bool = False):
        return ConversationCardMetadata(
            conversation_id=CONV,
            tenant_id="tnt_1",
            conversation_label="#agents",
            sender_label=label,
            sender_kind=kind,
            mention=mention,
            message_id=mid,
            conversation_seq=seq,
        )

    merged = coalesce_mod.merge_metadata(
        [meta("bcmsg_m1", 1, "human", "Ada"), meta("bcmsg_m2", 2, "agent", "Bea's agent", True)]
    )
    assert merged.message_id == "bcmsg_m2"
    assert merged.mention is True, "a mention anywhere in the burst directs the card"
    assert merged.sender_kind == "mixed"
    assert merged.sender_label == "several members"
    card = build_conversation_card(merged)
    assert "no_reply unless it asks you directly" not in card


# -- durable ref -------------------------------------------------------------


def test_the_ref_keeps_the_shape_the_previous_release_already_reads() -> None:
    """review m3: a rolled-back ``lop`` must still read these durable rows.

    The ref keys the feed queue AND the hold that re-drives it, so a shape the
    previous release's parser refuses (it accepted 4–7 parts) silently drops the
    wake after a downgrade. The two runner-up facts a card wanted are therefore
    derived at render time (a thread root with no mention IS a directed wake)
    and the conversation kind is simply not carried.
    """

    ref = ConversationFeedRef(
        conversation_id=CONV,
        message_id="bcmsg_m1",
        conversation_seq=4,
        mention=False,
        sender_kind="agent",
        thread_root_id=ROOT,
    ).validated()
    encoded = ref.encode()
    assert encoded == f"{PREFIX}{CONV}/bcmsg_m1/4/0/agent/thread_root={ROOT}"
    assert len(encoded[len(PREFIX) :].split("/")) == 6, "a reply stays within the old bound"
    assert parse_conversation_ref_value(encoded[len(PREFIX) :]) == ref


@pytest.mark.parametrize(
    "value",
    [
        f"{CONV}/bcmsg_m1/4/0",
        f"{CONV}/bcmsg_m1/4/1/human",
        f"{CONV}/bcmsg_m1/4/0/agent/thread_root={ROOT}",
        f"{CONV}/bcmsg_m1/4/0/human/legacy_alias={'a' * 64}",
    ],
)
def test_old_refs_parse_unchanged_and_re_encode_byte_identically(value: str) -> None:
    ref = parse_conversation_ref_value(value)
    assert ref is not None
    assert ref.encode() == PREFIX + value


@pytest.mark.parametrize(
    "value",
    [
        # An UNKNOWN tag is ignored, exactly as the previous release ignores it —
        # that tolerance is what lets a tag be added at all. The part bound is
        # the old one, so a shape this release never writes is still refused.
        f"{CONV}/bcmsg_m1/4/0/agent/thread_follower=1",
        f"{CONV}/bcmsg_m1/4/0/agent/agent_run=3",
    ],
)
def test_an_unknown_tag_still_decodes_to_the_pre_contract_ref(value: str) -> None:
    ref = parse_conversation_ref_value(value)
    assert ref is not None and ref.thread_root_id is None


def test_a_ref_longer_than_the_previous_releases_bound_is_refused() -> None:
    assert parse_conversation_ref_value(f"{CONV}/bcmsg_m1/4/0/agent/a=1/b=2/c=3") is None


def test_the_feed_projection_carries_the_thread_root_and_nothing_else() -> None:
    cursor, refs, _resync = project_events_batch(
        {
            "cursor": "9",
            "events": [
                {
                    "type": "message_available",
                    "ids": {
                        "conversation_id": CONV,
                        "message_id": "bcmsg_m1",
                        "conversation_seq": 4,
                        "sender_kind": "agent",
                        "thread_root_id": ROOT,
                        # A newer hub's extra wake facts. The ref carries none of
                        # them (review m3): the card derives `directed:` from the
                        # thread root, and nothing reads the rest.
                        "thread_follower": True,
                        "agent_run": 2,
                        "conversation_kind": "group",
                    },
                    "at": "2026-10-30T00:00:00Z",
                }
            ],
        }
    )
    assert cursor == 9
    ref = parse_conversation_ref_value(refs[0][len(PREFIX) :])
    assert ref is not None
    assert ref.thread_root_id == ROOT
    assert refs[0] == f"{PREFIX}{CONV}/bcmsg_m1/4/0/agent/thread_root={ROOT}"


def test_an_old_hubs_feed_row_projects_to_the_pre_contract_ref() -> None:
    _cursor, refs, _resync = project_events_batch(
        {
            "cursor": "9",
            "events": [
                {
                    "type": "message_available",
                    "ids": {
                        "conversation_id": CONV,
                        "message_id": "bcmsg_m1",
                        "conversation_seq": 4,
                        "sender_kind": "human",
                    },
                    "at": "2026-10-30T00:00:00Z",
                }
            ],
        }
    )
    assert refs == (f"{PREFIX}{CONV}/bcmsg_m1/4/0/human",)


# -- reply guard: never a cap, a warning on long one-sided runs ---------------


def _reserve(config_dir: Path, now: float, body: str | None = None, endpoint: str = "bep_a") -> Any:
    return reply_guard.reserve_post(
        config_dir, endpoint_id=endpoint, conversation_id=CONV, body=body, now=now
    )


def _trigger(
    config_dir: Path, kind: str, now: float, mention: bool = False, endpoint: str = "bep_a"
) -> None:
    reply_guard.record_trigger(
        config_dir,
        endpoint_id=endpoint,
        conversation_id=CONV,
        sender_kind=kind,
        mention=mention,
        now=now,
    )


def test_thirty_back_and_forth_agent_replies_in_a_dm_all_post(tmp_path: Path) -> None:
    """Operator, 2026-11-03: no limit on agent replies. The installed build
    refused the 4th of these with ``reply_cooldown``; the round-1 build would
    have bounded the run hourly. 30 replies each, 5 s a turn, all post, and an
    ordinary exchange never draws the warning (every reply answers a new one).
    """

    now = 1_000_000.0
    for turn in range(30):
        for me, peer in (("bep_a", "bep_b"), ("bep_b", "bep_a")):
            when = now + turn * 10 + (0 if me == "bep_a" else 5)
            _trigger(tmp_path, "agent", when, endpoint=me)
            slot = _reserve(tmp_path, when, body=f"{me} turn {turn}", endpoint=me)
            assert isinstance(slot, reply_guard.Reservation), (turn, me, slot)
            assert slot.warning is None, (turn, me, slot.warning)


def test_a_one_sided_run_warns_once_per_threshold_and_never_refuses(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "agent", now)
    warned: dict[int, str] = {}
    for index in range(1, 46):
        slot = _reserve(tmp_path, now + index, body=f"update {index}")
        assert isinstance(slot, reply_guard.Reservation), (index, slot)
        if slot.warning:
            warned[index] = slot.warning
    assert sorted(warned) == [5, 10, 20, 30, 40]
    assert warned[5] == (
        "Posted 5 messages in this conversation with no reply; "
        "be brief and avoid repeating yourself."
    )
    assert "\n" not in warned[5]


def test_a_new_inbound_message_resets_the_warning(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "agent", now)
    for index in range(1, 5):
        assert _reserve(tmp_path, now + index, body=f"a{index}").warning is None
    assert _reserve(tmp_path, now + 5, body="a5").warning is not None
    _trigger(tmp_path, "human", now + 6)
    # The run restarts at 1: the next warning is the 5th post after the reply.
    for index in range(1, 5):
        assert _reserve(tmp_path, now + 6 + index, body=f"b{index}").warning is None
    assert "Posted 5 messages" in str(_reserve(tmp_path, now + 11, body="b5").warning)


def test_a_group_with_a_person_and_three_agents_keeps_flowing(tmp_path: Path) -> None:
    """Three agents and one person, 200 messages at machine-ish speed (3 s)."""

    now = 1_000_000.0
    agents = ("bep_a", "bep_b", "bep_c")
    posted = 0
    for index in range(200):
        when = now + 3 * index
        author = "human" if index % 7 == 0 else agents[index % 3]
        for agent in agents:
            if agent != author:
                _trigger(tmp_path, "human" if author == "human" else "agent", when, endpoint=agent)
        if author == "human":
            continue
        slot = _reserve(tmp_path, when + 1, body=f"{author} says {index}", endpoint=author)
        assert isinstance(slot, reply_guard.Reservation), (index, slot)
        assert slot.warning is None
        posted += 1
    assert posted == 200 - len(range(0, 200, 7))


def test_a_person_talking_to_their_agent_is_never_throttled(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "human", now, mention=True)
    for index in range(40):
        slot = _reserve(tmp_path, now + index, body=f"progress {index}")
        assert isinstance(slot, reply_guard.Reservation) and not slot.agent_to_agent


def test_an_exact_double_send_within_ten_seconds_is_skipped(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "agent", now)
    assert isinstance(_reserve(tmp_path, now, body="Approved."), reply_guard.Reservation)
    skipped = _reserve(tmp_path, now + 3.6, body="Approved.")
    assert isinstance(skipped, reply_guard.Skipped)
    result = skipped.as_result(CONV)
    assert result["ok"] is True and result["skipped"] is True
    # n-2: "not posted" once; m-2: never claims the earlier copy was delivered.
    assert result["warning"] == (
        "Skipped: identical to a message you sent 3s ago here; not posted. "
        "If that send failed, send it again or change the text."
    )
    assert result["warning"].count("not posted") == 1
    # 11 s after the only POSTED copy (the skip records nothing): posts.
    assert isinstance(_reserve(tmp_path, now + 11, body="Approved."), reply_guard.Reservation)


def test_the_same_text_to_another_thread_or_addressee_is_not_a_double_send(
    tmp_path: Path,
) -> None:
    """Round 3, M-1, the reviewer's reproduction: two questions coalesced into
    one wake share a trigger time (100.0 and 100.2), so "Done." to thread A at
    104 and "Done." to thread B at 106 both follow it. The second was skipped;
    the thread and the addressees are now part of the double-send key."""

    _trigger(tmp_path, "agent", 100.0)
    _trigger(tmp_path, "agent", 100.2)
    thread_a = reply_guard.reserve_post(
        tmp_path,
        endpoint_id="bep_a",
        conversation_id=CONV,
        body="Done.",
        thread_root_id="bcmsg_threadA",
        now=104.0,
    )
    thread_b = reply_guard.reserve_post(
        tmp_path,
        endpoint_id="bep_a",
        conversation_id=CONV,
        body="Done.",
        thread_root_id="bcmsg_threadB",
        now=106.0,
    )
    assert isinstance(thread_a, reply_guard.Reservation)
    assert isinstance(thread_b, reply_guard.Reservation), thread_b
    # The same thread again IS the double send, and says "in this thread".
    again = reply_guard.reserve_post(
        tmp_path,
        endpoint_id="bep_a",
        conversation_id=CONV,
        body="Done.",
        thread_root_id="bcmsg_threadB",
        now=107.0,
    )
    assert isinstance(again, reply_guard.Skipped) and "in this thread" in again.warning

    # Addressees: "Approved." to Alice, then to Bob, posts; Alice again (listed
    # in another order, with stray spaces) is the double send.
    def to(*who: str, at: float) -> Any:
        return reply_guard.reserve_post(
            tmp_path,
            endpoint_id="bep_a",
            conversation_id=CONV,
            body="Approved.",
            addressees=who,
            now=at,
        )

    assert isinstance(to("agent:acct_alice", at=108.0), reply_guard.Reservation)
    assert isinstance(to("agent:acct_bob", at=108.5), reply_guard.Reservation)
    assert isinstance(to(" agent:acct_alice", at=109.0), reply_guard.Skipped)
    assert isinstance(to("agent:acct_bob", "agent:acct_alice", at=109.2), reply_guard.Reservation)
    assert isinstance(to("agent:acct_alice", "agent:acct_bob", at=109.4), reply_guard.Skipped)


def test_different_text_within_a_second_posts(tmp_path: Path) -> None:
    now = 1_000_000.0
    assert isinstance(_reserve(tmp_path, now, body="Approved."), reply_guard.Reservation)
    # Exact, not folded: any edit is deliberate.
    for offset, body in ((0.2, "approved."), (0.4, "Approved. "), (0.6, "Approved!")):
        assert isinstance(_reserve(tmp_path, now + offset, body=body), reply_guard.Reservation)


def test_the_same_answer_to_a_new_message_is_not_a_double_send(tmp_path: Path) -> None:
    """ "Yes." to two questions 3 s apart: the new inbound makes it an answer."""

    now = 1_000_000.0
    _trigger(tmp_path, "agent", now)
    assert isinstance(_reserve(tmp_path, now, body="Yes."), reply_guard.Reservation)
    _trigger(tmp_path, "agent", now + 2)
    assert isinstance(_reserve(tmp_path, now + 3, body="Yes."), reply_guard.Reservation)


def test_the_double_send_rule_is_per_endpoint_and_per_conversation(tmp_path: Path) -> None:
    now = 1_000_000.0
    assert isinstance(_reserve(tmp_path, now, body="same"), reply_guard.Reservation)
    assert isinstance(
        _reserve(tmp_path, now + 1, body="same", endpoint="bep_b"), reply_guard.Reservation
    )
    other = reply_guard.reserve_post(
        tmp_path, endpoint_id="bep_a", conversation_id="bcv_otherconv", body="same", now=now + 1
    )
    assert isinstance(other, reply_guard.Reservation)


def test_only_a_keyed_digest_of_the_body_is_stored(tmp_path: Path) -> None:
    """Round 1, S-3: a plain SHA-256 of "approved" is a dictionary lookup."""

    import sqlite3
    import stat

    _trigger(tmp_path, "agent", 1.0)
    _reserve(tmp_path, 1.0, body="approved")
    key = reply_guard.key_path(tmp_path).read_bytes()
    assert len(key) == 32
    assert stat.S_IMODE(reply_guard.key_path(tmp_path).stat().st_mode) == 0o600
    conn = sqlite3.connect(reply_guard.guard_path(tmp_path))
    try:
        rows = conn.execute("SELECT body_hash FROM posts").fetchall()
        dump = "\n".join(conn.iterdump())
    finally:
        conn.close()
    assert rows == [(reply_guard.body_hash("approved", key=key),)]
    # The thread and addressees change the digest (M-1); nothing else is stored.
    assert reply_guard.body_hash("approved", key=key, thread_root_id="bcmsg_t") != rows[0][0]
    assert hashlib.sha256(b"approved").hexdigest() not in dump
    assert "approved" not in dump
    # Each install draws its own key, so equal bodies do not match across files.
    other = tmp_path / "other"
    _trigger(other, "agent", 1.0)
    _reserve(other, 1.0, body="approved")
    assert reply_guard.key_path(other).read_bytes() != key


def test_an_old_build_cool_down_row_does_not_refuse(tmp_path: Path) -> None:
    """Upgrade: an older build's live ``runs.cooldown_until`` and its posts block nothing."""

    import sqlite3

    path = reply_guard.guard_path(tmp_path)
    path.parent.mkdir(parents=True)
    now = 1_000_000.0
    conn = sqlite3.connect(path)
    try:
        # The previous build's exact schema: no ``body_hash`` column yet.
        conn.executescript("""
            CREATE TABLE triggers (endpoint_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                at REAL NOT NULL, sender_kind TEXT NOT NULL, mention INTEGER NOT NULL,
                directed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (endpoint_id, conversation_id));
            CREATE TABLE posts (id INTEGER PRIMARY KEY AUTOINCREMENT, endpoint_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, at REAL NOT NULL, agent_to_agent INTEGER NOT NULL,
                human_prompted INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE runs (endpoint_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                consecutive INTEGER NOT NULL DEFAULT 0, cooldown_until REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (endpoint_id, conversation_id));
            """)
        conn.execute(
            "INSERT INTO triggers VALUES ('bep_a', ?, ?, 'agent', 0, 0)", (CONV, now - 600)
        )
        # Far past every count rule any earlier build had (3 per inbound, 20/min,
        # 60 and 120 per 10 min, 120/h), all AFTER the latest inbound message.
        conn.executemany(
            "INSERT INTO posts (endpoint_id, conversation_id, at, agent_to_agent) "
            "VALUES ('bep_a', ?, ?, 1)",
            [(CONV, now - 590 + 4 * i) for i in range(140)],
        )
        # A LIVE cool-down: the previous build would refuse for another 590 s.
        conn.execute("INSERT INTO runs VALUES ('bep_a', ?, 3, ?)", (CONV, now + 590))
        conn.commit()
    finally:
        conn.close()
    # No new trigger: the old rows are all the state there is.
    slot = _reserve(tmp_path, now, body="the approval")
    assert isinstance(slot, reply_guard.Reservation), slot


def test_an_older_builds_posts_never_count_towards_the_run_warning(tmp_path: Path) -> None:
    """Round 3, QA Q5: the first post after an upgrade warned "Posted 140
    messages" because the previous build's rows counted. Only rows this build
    wrote (``tracked``) count, so the run starts at 1."""

    import sqlite3

    path = reply_guard.guard_path(tmp_path)
    path.parent.mkdir(parents=True)
    now = 1_000_000.0
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE posts (id INTEGER PRIMARY KEY AUTOINCREMENT, endpoint_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, at REAL NOT NULL, agent_to_agent INTEGER NOT NULL,
                human_prompted INTEGER NOT NULL DEFAULT 0, body_hash TEXT);
            """)
        conn.executemany(
            "INSERT INTO posts (endpoint_id, conversation_id, at, agent_to_agent, body_hash) "
            "VALUES ('bep_a', ?, ?, 1, NULL)",
            [(CONV, now - 600 + 4 * i) for i in range(139)],
        )
        conn.commit()
    finally:
        conn.close()
    warned = []
    for index in range(1, 6):
        slot = _reserve(tmp_path, now + index, body=f"after upgrade {index}")
        assert isinstance(slot, reply_guard.Reservation)
        if slot.warning:
            warned.append((index, slot.warning))
    assert warned == [
        (
            5,
            "Posted 5 messages in this conversation with no reply; "
            "be brief and avoid repeating yourself.",
        )
    ]


def test_a_released_slot_is_neither_counted_nor_a_duplicate(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "agent", now)
    slots = [_reserve(tmp_path, now + i, body=f"r{i}") for i in range(4)]
    reply_guard.release(tmp_path, slots[-1], endpoint_id="bep_a", conversation_id=CONV)
    # The slot AND its body hash are gone: the retry posts, and it is post 4,
    # not 5 (no warning yet).
    retry = _reserve(tmp_path, now + 4, body="r3")
    assert isinstance(retry, reply_guard.Reservation) and retry.warning is None


def test_an_unreadable_state_dir_is_classed_broken(tmp_path: Path) -> None:
    # A FILE where the state directory belongs: the guard cannot open its db.
    (tmp_path / "team_bridge").write_text("not a directory")
    with pytest.raises(reply_guard.ReplyGuardUnavailable) as caught:
        _reserve(tmp_path, 1.0)
    assert caught.value.transient is False


def test_a_damaged_or_unreadable_key_is_classed_broken(tmp_path: Path) -> None:
    reply_guard.key_path(tmp_path).parent.mkdir(parents=True)
    reply_guard.key_path(tmp_path).write_bytes(b"short")
    with pytest.raises(reply_guard.ReplyGuardUnavailable) as caught:
        _reserve(tmp_path, 1.0, body="hello")
    assert caught.value.transient is False
    reply_guard.key_path(tmp_path).unlink()
    reply_guard.key_path(tmp_path).mkdir()  # unreadable as a file
    with pytest.raises(reply_guard.ReplyGuardUnavailable) as caught:
        _reserve(tmp_path, 1.0, body="hello")
    assert caught.value.transient is False


def test_a_locked_state_file_is_classed_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    monkeypatch.setattr(reply_guard, "_BUSY_TIMEOUT_S", 0.05)
    _reserve(tmp_path, 1.0)  # create the file and schema
    holder = sqlite3.connect(reply_guard.guard_path(tmp_path), isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        with pytest.raises(reply_guard.ReplyGuardUnavailable) as caught:
            _reserve(tmp_path, 2.0)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert caught.value.transient is True


def test_the_fail_open_warning_matches_the_cause() -> None:
    """Round 3, m-3: never a refusal. Busy asks nothing; broken asks for the operator."""

    busy = reply_guard.unavailable_warning(
        reply_guard.ReplyGuardUnavailable("reply guard state is busy", transient=True)
    )
    assert busy == "reply guard busy; posted without duplicate/run tracking"
    broken = reply_guard.unavailable_warning(
        reply_guard.ReplyGuardUnavailable("reply guard key is damaged")
    )
    assert broken == (
        "reply guard state is broken (reply guard key is damaged); posted without "
        "duplicate/run tracking — do not resend or retry; report this to the operator"
    )


def test_the_agent_run_notice_warns_once_per_threshold(tmp_path: Path) -> None:
    """Round 3, I-1: #294's ``agent_wake`` notice becomes a warning at 8, 10, 20 …

    In a back-and-forth each agent sees every OTHER run value, so thresholds
    are levels: this agent sees 8, 10, 12 … 22 and is warned at 8, 10 and 20.
    A run that falls below 8 (a person spoke) re-arms it; an older server's
    absent field changes nothing.
    """

    def notice(count: int | None) -> str | None:
        payload = (
            None
            if count is None
            else {"unanswered_messages": 1, "consecutive_agent_messages": count}
        )
        return reply_guard.agent_run_warning(
            tmp_path, endpoint_id="bep_a", conversation_id=CONV, notice=payload
        )

    seen = {count: notice(count) for count in range(2, 24, 2)}
    assert [count for count, warning in seen.items() if warning] == [8, 10, 20]
    assert seen[8] == (
        "8 consecutive agent messages in this conversation; be brief and avoid repeating yourself."
    )
    assert notice(None) is None and notice(3) is None  # old server / a person spoke
    assert notice(8) is not None, "a new run warns again"
    assert (
        reply_guard.agent_run_warning(
            tmp_path,
            endpoint_id="bep_a",
            conversation_id=CONV,
            notice={"consecutive_agent_messages": True},
        )
        is None
    )


def test_the_digest_key_survives_a_creation_race(tmp_path: Path) -> None:
    """Eight processes create the key at once: one key, every digest agrees."""

    import subprocess
    import sys

    start = time.time() + 2.0
    script = (
        "import sys, time\n"
        "from local_operator.team_bridge import reply_guard\n"
        "while time.time() < float(sys.argv[2]): pass\n"
        "print(reply_guard._digest_key(sys.argv[1]).hex())\n"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(start)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    keys = {proc.communicate(timeout=60)[0].strip() for proc in procs}
    assert all(proc.returncode == 0 for proc in procs)
    assert keys == {reply_guard.key_path(tmp_path).read_bytes().hex()}
    leftovers = [p.name for p in reply_guard.key_path(tmp_path).parent.iterdir()]
    assert leftovers == ["reply_guard.key"], "no temp file is left behind"


def test_an_upgrade_race_between_processes_never_fails_closed(tmp_path: Path) -> None:
    """Round 1, m1: concurrent first opens of an old file both tried the ALTER.

    Eight processes open ten fresh previous-build files in lockstep (one file
    per 0.4 s slot). Measured on the head under review: about 1 in 15 opens
    failed with ``duplicate column name: body_hash``. Every open must succeed.
    """

    import sqlite3
    import subprocess
    import sys

    old_schema = (
        "CREATE TABLE triggers (endpoint_id TEXT NOT NULL, conversation_id TEXT NOT NULL,"
        " at REAL NOT NULL, sender_kind TEXT NOT NULL, mention INTEGER NOT NULL,"
        " directed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (endpoint_id, conversation_id));"
        "CREATE TABLE posts (id INTEGER PRIMARY KEY AUTOINCREMENT, endpoint_id TEXT NOT NULL,"
        " conversation_id TEXT NOT NULL, at REAL NOT NULL, agent_to_agent INTEGER NOT NULL,"
        " human_prompted INTEGER NOT NULL DEFAULT 0);"
    )
    trials = 10
    for trial in range(trials):
        path = reply_guard.guard_path(tmp_path / f"t{trial}")
        path.parent.mkdir(parents=True)
        conn = sqlite3.connect(path)
        conn.executescript(old_schema)
        conn.close()
    start = time.time() + 2.0
    script = (
        "import sys, time\n"
        "from local_operator.team_bridge import reply_guard\n"
        "root, start, trials, me = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]\n"
        "failures = 0\n"
        "for trial in range(trials):\n"
        "    while time.time() < start + 0.4 * trial: pass\n"
        "    try:\n"
        "        reply_guard.reserve_post(f'{root}/t{trial}', endpoint_id=me,\n"
        "            conversation_id='bcv_race', body=None)\n"
        "    except reply_guard.ReplyGuardUnavailable:\n"
        "        failures += 1\n"
        "print(failures)\n"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(start), str(trials), f"bep_{n}"],
            stdout=subprocess.PIPE,
            text=True,
        )
        for n in range(8)
    ]
    failures = [int(proc.communicate(timeout=120)[0].strip()) for proc in procs]
    assert all(proc.returncode == 0 for proc in procs)
    assert sum(failures) == 0, failures


# -- the shared multi-message rig ------------------------------------------------


@dataclass
class Msg:
    id: str
    seq: int
    sender_endpoint: str | None  # None = the human
    body: str
    mentions: frozenset[str] = frozenset()
    agent_run: int = 0
    #: The conversation kind the row carries (``dm`` for the 1:1 lane).
    kind: str = "channel"
    #: The server's own follower bit for a thread reply; ``None`` = not stated.
    thread_follower: bool | None = None
    #: The thread root a reply answers (``/events`` ``ids.thread_root_id``).
    thread_root: str | None = None

    @property
    def sender_kind(self) -> str:
        return "human" if self.sender_endpoint is None else "agent"


@dataclass
class Hub:
    """A fake hub with the wake contract's fan-out rule.

    ``caps=True`` applies the server rule (agent-sent messages reach other
    agents only while the conversation's consecutive agent run is below the
    SOFT cap; at/after the HARD cap not even mentions do; a human message
    resets the run). ``caps=False`` is an OLD hub that fans everything out —
    the client guard alone must then bound the conversation.
    """

    members: tuple[str, ...]
    caps: bool = True
    soft_cap: int = 4
    hard_cap: int = 8
    messages: dict[str, Msg] = field(default_factory=dict)
    feeds: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    run: int = 0
    screened: list[tuple[str, str]] = field(default_factory=list)

    def post(
        self,
        sender_endpoint: str | None,
        body: str,
        mentions: frozenset[str] = frozenset(),
        *,
        kind: str = "channel",
        thread_root: str | None = None,
        thread_follower: bool | None = None,
    ) -> Msg:
        self.run = 0 if sender_endpoint is None else self.run + 1
        seq = len(self.messages) + 1
        msg = Msg(
            f"bcmsg_m{seq:04d}",
            seq,
            sender_endpoint,
            body,
            mentions,
            self.run,
            kind,
            thread_follower,
            thread_root,
        )
        self.messages[msg.id] = msg
        for endpoint in self.members:
            if endpoint == sender_endpoint or not self._wakes(endpoint, msg):
                continue
            ids: dict[str, Any] = {
                "conversation_id": CONV,
                "message_id": msg.id,
                "conversation_seq": msg.seq,
                "sender_kind": msg.sender_kind,
                "conversation_kind": msg.kind,
            }
            if endpoint in mentions:
                ids["mention"] = True
            if msg.agent_run:
                ids["agent_run"] = msg.agent_run
            if msg.thread_root is not None:
                ids["thread_root_id"] = msg.thread_root
            self.feeds.setdefault(endpoint, []).append(
                {"type": "message_available", "ids": ids, "at": "2026-10-30T00:00:00Z"}
            )
        return msg

    def _wakes(self, endpoint: str, msg: Msg) -> bool:
        if not self.caps or msg.sender_endpoint is None:
            return True
        if msg.agent_run >= self.hard_cap:
            return False
        if endpoint in msg.mentions:
            return True
        return msg.agent_run < self.soft_cap

    def take_feed(self, endpoint: str) -> list[dict[str, Any]]:
        rows, self.feeds[endpoint] = self.feeds.get(endpoint, []), []
        return rows

    def agent_posts(self) -> list[Msg]:
        return [m for m in self.messages.values() if m.sender_endpoint is not None]


class Wire:
    def __init__(self, hub: Hub, endpoint: str) -> None:
        self.hub, self.endpoint = hub, endpoint

    async def conversation_body(self, conversation_id: str, message_id: str, **kwargs: Any):
        msg = self.hub.messages[message_id]
        return {
            "message_id": message_id,
            "body": msg.body,
            "body_hash": hashlib.sha256(msg.body.encode("utf-8")).hexdigest(),
            "conversation_name": "#agents",
            "sender_label": (
                "Ada" if msg.sender_endpoint is None else f"{msg.sender_endpoint}-agent"
            ),
            "sender_kind": msg.sender_kind,
            # The live row's own routing facts (agent-server #255). Absent for an
            # older hub; the card falls back to the pre-#255 derivation then.
            "conversation_kind": msg.kind,
            **({"thread_follower": msg.thread_follower} if msg.thread_follower is not None else {}),
        }

    async def conversation_admit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


class ToolClient:
    """The bridge tool's client: a post lands on the hub as THIS agent."""

    def __init__(self, hub: Hub, endpoint: str) -> None:
        self.hub, self.endpoint = hub, endpoint

    async def conversation_send(self, conversation_id: str, *, text: str, **kwargs: Any):
        assert conversation_id == CONV
        guard = kwargs.get("pre_http_guard")
        if guard is not None:
            guard()
        msg = self.hub.post(self.endpoint, text)
        return {"id": msg.id, "seq": msg.seq}


def _rig(monkeypatch: pytest.MonkeyPatch, hub: Hub, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    """Patch the connector seams around the fake hub; return the wake log."""

    wakes: list[dict[str, Any]] = []
    batches: dict[str, list[str]] = {}

    async def fake_events(
        owner: Any, *, config_dir: Path, access: Any, after: Any = None, **kw: Any
    ):
        rows = hub.take_feed(owner.endpoint_id)
        _cursor, refs, _ = project_events_batch({"cursor": "1", "events": rows})
        batches[owner.endpoint_id] = list(refs)
        return {"cursor": 1, "refs": list(refs), "resync": False}

    def fake_drain(owner: Any, *, config_dir: Path, upto_seq: int) -> list[str]:
        return batches.pop(owner.endpoint_id, [])

    async def fake_wire_for(owner: Any, config_dir: Path, auth_store: Any) -> Wire:
        return Wire(hub, owner.endpoint_id)

    async def fake_screen(owner: Any, item: Any, *, config_dir: Path, **kwargs: Any):
        message_id = item["message"]["id"]
        hub.screened.append((owner.endpoint_id, message_id))
        receipt = _receipt("allow", hub.messages[message_id].body)
        _persist_fake_receipt(config_dir, receipt, message_id=message_id)
        return ScreenResult(disposition="deliver", receipt=receipt)

    async def fake_deliver(
        owner: Any, payload: str, *, session_id: str, config_dir: Path, **kw: Any
    ):
        wakes.append({"endpoint": owner.endpoint_id, "session": session_id, "card": payload})
        return "delivered and woke the session"

    monkeypatch.setattr(pull_mod, "events", fake_events)
    monkeypatch.setattr(pull_mod, "drain_feed_refs", fake_drain)
    monkeypatch.setattr(conversation_mod, "_wire_for", fake_wire_for)
    monkeypatch.setattr(conversation_mod, "screen_item", fake_screen)
    monkeypatch.setattr(conversation_mod, "deliver_wake", fake_deliver)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    # The tool's fences have their own suites; open them for the post path.
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)
    # A fresh coalescer per test: the module-level one is process state.
    monkeypatch.setattr(coalesce_mod, "CONNECTOR_COALESCER", coalesce_mod.WakeCoalescer())
    return wakes


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def _iterate(owner: Any, config_dir: Path) -> dict[str, Any]:
    return asyncio.run(
        loop_mod.run_iteration(
            owner,
            config_dir=config_dir,
            access=SimpleNamespace(credential_id=owner.credential_id),
            auth_store=None,
        )
    )


def _card_field(card: str, name: str) -> str | None:
    match = re.search(rf"^{name}: (.+)$", card, re.M)
    return match.group(1) if match else None


# -- coalescing --------------------------------------------------------------


def test_three_messages_in_one_window_are_one_wake_and_three_screens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    dirs = {"a": tmp_path / "a"}
    dirs["a"].mkdir()
    _enroll(dirs["a"], a, "sess_a")
    hub = Hub(members=("bep_a",))
    wakes = _rig(monkeypatch, hub, dirs)

    sent = [hub.post(None, f"line {i}") for i in range(3)]
    outcome = _iterate(a, dirs["a"])

    assert [row["action"] for row in outcome["handled"]] == ["delivered"] * 3
    assert hub.screened == [("bep_a", m.id) for m in sent], "each message screened on its own"
    assert len(wakes) == 1, "ONE wake for the burst"
    card = wakes[0]["card"]
    assert _card_field(card, "message_ids") == ",".join(m.id for m in sent)
    assert _card_field(card, "message_id") == sent[-1].id
    # The card names one thread root at most and no conversation kind: the ref
    # this came from carries neither (review round 1, m3).
    assert _card_field(card, "conversation_kind") is None
    assert _card_field(card, "thread_root_id") is None
    assert outcome["coalesced_wakes"][0]["message_ids"] == [m.id for m in sent]


def test_a_second_burst_inside_the_window_waits_then_wakes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from local_operator.team_bridge.store import state_db_path

    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    dirs = {"a": tmp_path / "a"}
    dirs["a"].mkdir()
    _enroll(dirs["a"], a, "sess_a")
    hub = Hub(members=("bep_a",))
    wakes = _rig(monkeypatch, hub, dirs)

    hub.post(None, "first")
    _iterate(a, dirs["a"])
    assert len(wakes) == 1

    clock.now += 1.0  # inside the 3 s window
    second = [hub.post(None, "again"), hub.post(None, "and again")]
    _iterate(a, dirs["a"])
    assert len(wakes) == 1, "rate-limited: no second wake inside the window"
    # The buffered group is crash-safe: a durable hold per message.
    with sqlite3.connect(state_db_path(dirs["a"])) as conn:
        codes = [row[0] for row in conn.execute("SELECT hold_code FROM screening_holds")]
    assert codes == [coalesce_mod.HOLD_WAKE_COALESCING] * 2
    # And the long poll is shortened so the card is not stuck behind a park.
    assert loop_mod._events_wait_s() <= 3

    clock.now += 3.0
    _iterate(a, dirs["a"])
    assert len(wakes) == 2
    assert _card_field(wakes[1]["card"], "message_ids") == ",".join(m.id for m in second)
    with sqlite3.connect(state_db_path(dirs["a"])) as conn:
        assert conn.execute("SELECT COUNT(*) FROM screening_holds").fetchone()[0] == 0
        posted = conn.execute(
            "SELECT COUNT(*) FROM body_reservations WHERE card_posted_at IS NOT NULL"
        ).fetchone()[0]
    assert posted == 3, "every covered reservation is marked posted at flush"


def test_without_a_connector_scope_the_card_goes_out_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct callers (re-drives outside an iteration, tests) keep the
    pre-coalescing behaviour: one card per message, at once."""

    a = _owner("a", 1)
    dirs = {"a": tmp_path / "a"}
    dirs["a"].mkdir()
    _enroll(dirs["a"], a, "sess_a")
    hub = Hub(members=("bep_a",))
    wakes = _rig(monkeypatch, hub, dirs)
    msg = hub.post(None, "hi")
    ref = parse_conversation_ref_value(f"{CONV}/{msg.id}/{msg.seq}/0/human")
    assert ref is not None
    outcome = asyncio.run(
        conversation_mod.handle_conversation_available(
            a,
            ref,
            config_dir=dirs["a"],
            auth_store=None,
            access=SimpleNamespace(credential_id=1),
        )
    )
    assert outcome["action"] == "delivered" and outcome["detail"] != "coalesced"
    assert len(wakes) == 1


def test_the_connector_records_the_trigger_the_guard_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    dirs = {"a": tmp_path / "a"}
    dirs["a"].mkdir()
    _enroll(dirs["a"], a, "sess_a")
    hub = Hub(members=("bep_a", "bep_b"), caps=False)
    _rig(monkeypatch, hub, dirs)
    hub.post("bep_b", "from an agent")
    _iterate(a, dirs["a"])
    slot = reply_guard.reserve_post(dirs["a"], endpoint_id="bep_a", conversation_id=CONV)
    assert isinstance(slot, reply_guard.Reservation) and slot.agent_to_agent


# -- channel_post guard through the real tool --------------------------------------


def _runtime(hub: Hub, owner: Any, config_dir: Path) -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, ToolClient(hub, owner.endpoint_id)),
        config_dir=config_dir,
        tenant_id=owner.tenant_id,
        owner=owner,
        session_id=f"sess_{owner.endpoint_id}",
    )


def _post(runtime: BridgeRuntime, body: str = "ok") -> dict[str, Any]:
    return asyncio.run(
        execute_bridge(runtime, {"op": "channel_post", "channel_id": CONV, "body": body})
    )


def test_channel_post_never_refuses_on_count_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real tool: 12 posts with nothing new to answer all land; the
    warning rides the 5th and 10th results only, as a field AND in ``content``
    (what the model actually reads)."""

    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    results = [_post(runtime, body=f"part {index}") for index in range(1, 13)]
    assert all(result["ok"] is True and not result.get("skipped") for result in results)
    assert len(hub.agent_posts()) == 12
    warned = [index for index, result in enumerate(results, 1) if "warning" in result]
    assert warned == [5, 10]
    assert results[4]["warning"].startswith("Posted 5 messages in this conversation")
    assert "warning" not in results[5]


def test_the_model_reads_the_warning_and_the_skip_in_the_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text block the MODEL receives (``AgentTool.execute``), not the dict:
    the 5th post's result carries the warning line, the 4th's does not, and a
    double send is a non-error result saying it was not posted."""

    from local_operator.harness.types import ToolContext
    from local_operator.team_bridge.tool import build_bridge_tool

    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    tool = build_bridge_tool(ToolContext(bridge_runtime=_runtime(hub, a, tmp_path)))
    assert tool is not None

    def call(body: str) -> Any:
        clock.now += 1
        args = {"op": "channel_post", "channel_id": CONV, "body": body}

        async def run() -> Any:
            return await tool.execute("call_w", args, None, None, ToolContext())

        return asyncio.run(run())

    from local_operator.harness.types import Message
    from local_operator.providers.clients import AnthropicClient

    def provider_text(result: Any) -> str:
        # The text the PROVIDER receives (QA round 3, Q-1): details never reach it.
        blocks = AnthropicClient._tool_result_blocks(Message.tool_result(result))
        return "\n".join(block["text"] for block in blocks if block.get("type") == "text")

    texts = [call(f"status {index}") for index in range(1, 6)]
    rendered = [provider_text(result) for result in texts]
    assert all(not result.is_error for result in texts)
    assert "warning:" not in rendered[3]
    assert rendered[4].endswith(
        "warning: Posted 5 messages in this conversation with no reply; "
        "be brief and avoid repeating yourself."
    )
    print(f"\n5th post, model sees: {rendered[4]!r}")
    skipped = call("status 5")
    assert not skipped.is_error
    text = provider_text(skipped)
    assert "not posted" in text and "1s ago" in text and "change the text" in text
    print(f"double send, model sees: {text!r}")
    assert len(hub.agent_posts()) == 5


class _NoticeClient(ToolClient):
    """A #294 hub: the send answer carries ``agent_wake`` once the run is long."""

    run = 0

    async def conversation_send(self, conversation_id: str, *, text: str, **kwargs: Any):
        answer = await super().conversation_send(conversation_id, text=text, **kwargs)
        if self.run:
            answer["agent_wake"] = {
                "unanswered_messages": 1,
                "consecutive_agent_messages": self.run,
            }
        return answer


@pytest.mark.parametrize(
    "breakage",
    ["locked", "damaged_key", "unreadable_key", "unwritable_dir", "corrupt_db"],
)
def test_an_unusable_guard_never_stops_a_post_and_the_model_is_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breakage: str
) -> None:
    """Round 3, m-3: fail OPEN. Every class of unusable guard state posts, and
    the provider text carries the one-line warning (Q-1). Busy asks nothing of
    anyone; every broken class asks the model to tell the operator."""

    import sqlite3

    from local_operator.harness.types import Message, ToolContext
    from local_operator.providers.clients import AnthropicClient
    from local_operator.team_bridge.tool import build_bridge_tool

    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    config = tmp_path / "cfg"
    state = config / "team_bridge"
    holder = None
    if breakage == "locked":
        monkeypatch.setattr(reply_guard, "_BUSY_TIMEOUT_S", 0.05)
        _reserve(config, 1.0, body="seed")
        holder = sqlite3.connect(reply_guard.guard_path(config), isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
    elif breakage == "damaged_key":
        state.mkdir(parents=True)
        reply_guard.key_path(config).write_bytes(b"short")
    elif breakage == "unreadable_key":
        state.mkdir(parents=True)
        reply_guard.key_path(config).mkdir()  # a directory where the key file belongs
    elif breakage == "unwritable_dir":
        config.mkdir()
        state.write_text("a file where the state directory belongs")
    else:
        state.mkdir(parents=True)
        reply_guard.guard_path(config).write_bytes(b"this is not a sqlite database" * 64)
    tool = build_bridge_tool(ToolContext(bridge_runtime=_runtime(hub, a, config)))
    assert tool is not None
    try:
        args = {"op": "channel_post", "channel_id": CONV, "body": "the approval"}

        async def run() -> Any:
            return await tool.execute("call_b", args, None, None, ToolContext())

        result = asyncio.run(run())
    finally:
        if holder is not None:
            holder.execute("ROLLBACK")
            holder.close()
    blocks = AnthropicClient._tool_result_blocks(Message.tool_result(result))
    text = "\n".join(block["text"] for block in blocks if block.get("type") == "text")
    assert not result.is_error and result.details["ok"] is True
    assert len(hub.agent_posts()) == 1, "posted despite the guard"
    if breakage == "locked":
        assert text.endswith("warning: reply guard busy; posted without duplicate/run tracking")
    else:
        assert "warning: reply guard state is broken (" in text
        assert text.endswith(
            "posted without duplicate/run tracking — do not resend or retry; "
            "report this to the operator"
        )
    print(f"\n{breakage}: model sees {text!r}")


def test_the_server_agent_run_notice_reaches_the_model_once_per_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 3, I-1: a back-and-forth pair never trips the one-sided warning, so
    #294's ``agent_wake`` notice is surfaced instead, once per threshold."""

    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    client = _NoticeClient(hub, a.endpoint_id)
    runtime.client = cast(TeamBridgeClient, client)
    warnings = {}
    for run in (0, 6, 8, 9, 10, 12):
        client.run = run
        _trigger(tmp_path, "agent", time.time() + run)  # alternating: a new inbound each time
        result = _post(runtime, body=f"turn {run}")
        assert result["ok"] is True
        warnings[run] = result.get("warning")
    assert {run for run, warning in warnings.items() if warning} == {8, 10}
    assert warnings[8] == (
        "8 consecutive agent messages in this conversation; be brief and avoid repeating yourself."
    )


class _Hub294Client(ToolClient):
    """agent-server #294's send answer, faithfully: ``agent_wake`` is OMITTED
    while the conversation's agent run is below 8 (an alternating pair never
    reaches the 3-in-a-row streak), and present from 8 on (QA round 4, Q-2)."""

    run = 0

    async def conversation_send(self, conversation_id: str, *, text: str, **kwargs: Any):
        answer = await super().conversation_send(conversation_id, text=text, **kwargs)
        if self.run >= 8:
            answer["agent_wake"] = {
                "unanswered_messages": 1,
                "consecutive_agent_messages": self.run,
            }
        return answer


@pytest.mark.parametrize("first_run_ends", [9, 25])
def test_a_second_agent_run_re_arms_through_an_omitted_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_run_ends: int
) -> None:
    """QA round 4, Q-2 / code m-B, QA's reproduction. Run 1 → a gap in which the
    server sends NO notice (a person spoke, the run restarted below 8) → run 2.
    Run 2 must warn at every threshold again, including the one run 1 stopped
    at (on 54f74c5 the 8 or the 20 was lost)."""

    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    client = _Hub294Client(hub, a.endpoint_id)
    runtime.client = cast(TeamBridgeClient, client)

    def turn(run: int) -> str | None:
        clock.now += 30
        client.run = run
        _trigger(tmp_path, "agent", clock.now)  # alternating: a new inbound each turn
        result = _post(runtime, body=f"turn {clock.now}")
        assert result["ok"] is True
        return result.get("warning")

    first = [run for run in range(8, first_run_ends + 1) if turn(run)]
    assert first == [r for r in (8, 10, 20) if r <= first_run_ends]
    gap = [turn(run) for run in range(1, 8)]  # run 1..7 of the next exchange: field absent
    assert gap == [None] * 7
    second = [run for run in range(8, 21) if turn(run)]
    assert second == [8, 10, 20], second


def test_a_person_speaking_clears_the_notice_level(tmp_path: Path) -> None:
    """The server resets the run on any human post; so does the client (Q-2)."""

    def notice(count: int) -> str | None:
        return reply_guard.agent_run_warning(
            tmp_path,
            endpoint_id="bep_a",
            conversation_id=CONV,
            notice={"unanswered_messages": 1, "consecutive_agent_messages": count},
        )

    assert notice(8) is not None and notice(9) is None
    _trigger(tmp_path, "human", 5.0)  # ambient, not addressed: still a reset
    assert notice(8) is not None, "a new run after a person spoke warns at 8 again"


@pytest.mark.parametrize("count", [10**30, 2**63, -1, 10**6 + 1])
def test_an_out_of_range_run_count_never_fails_a_delivered_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """Security round 4, S-11: 10**30 raised OverflowError AFTER the post had
    landed, and the harness told the model it failed. Bounded now: posted,
    ok, no warning, nothing raised."""

    from local_operator.harness.types import ToolContext
    from local_operator.team_bridge.tool import build_bridge_tool

    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    client = _NoticeClient(hub, a.endpoint_id)
    client.run = count
    runtime.client = cast(TeamBridgeClient, client)
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    args = {"op": "channel_post", "channel_id": CONV, "body": "the approval"}

    async def run() -> Any:
        return await tool.execute("call_o", args, None, None, ToolContext())

    result = asyncio.run(run())
    assert not result.is_error and result.details["ok"] is True
    assert "warning" not in result.details
    assert len(hub.agent_posts()) == 1


def test_an_advisory_fault_after_delivery_is_never_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-11, structural: whatever the notice bookkeeping raises after the hub
    accepted the post, the post is reported as posted. The DM call site has its
    own end-to-end twin through a real ``dm`` (round 5, m-C:
    ``test_an_advisory_fault_after_a_delivered_dm_is_never_a_failure``); a
    ``[dm]`` case here only called the helper."""

    from local_operator.team_bridge import tool as tool_mod

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise OverflowError("bookkeeping fault")

    monkeypatch.setattr(tool_mod.reply_guard, "agent_run_warning", explode)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    advice = tool_mod._agent_run_advice(runtime, "bep_a", CONV, {"id": "bcmsg_x"}, guard_busy=False)
    assert advice is None
    result = _post(runtime, body="landed")
    assert result["ok"] is True and len(hub.agent_posts()) == 1


def test_a_busy_guard_is_not_asked_again_for_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code round 4, n-A: one busy timeout per call, not two."""

    import sqlite3

    from local_operator.team_bridge import tool as tool_mod

    calls: list[Any] = []
    real = tool_mod.reply_guard.agent_run_warning
    monkeypatch.setattr(
        tool_mod.reply_guard,
        "agent_run_warning",
        lambda *a, **k: calls.append(k) or real(*a, **k),
    )
    monkeypatch.setattr(reply_guard, "_BUSY_TIMEOUT_S", 0.05)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    _reserve(tmp_path, 1.0, body="seed")
    holder = sqlite3.connect(reply_guard.guard_path(tmp_path), isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        busy = _post(runtime, body="while busy")
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert busy["ok"] is True and "reply guard busy" in busy["warning"]
    assert calls == [], "the notice lookup was skipped after a busy reservation"
    assert _post(runtime, body="after")["ok"] is True
    assert len(calls) == 1, "and runs normally once the guard is free"


def test_channel_post_skips_an_exact_double_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    assert _post(runtime, body="Thanks!")["ok"] is True
    clock.now += 2
    skipped = _post(runtime, body="Thanks!")
    assert skipped["ok"] is True and skipped["skipped"] is True
    assert "2s ago" in skipped["warning"] and "not posted" in skipped["warning"]
    assert len(hub.agent_posts()) == 1, "the double send never reached the hub"
    # Folded variants are deliberate edits, and post.
    assert _post(runtime, body="  thanks!  ")["ok"] is True
    clock.now += 11
    assert _post(runtime, body="Thanks!").get("skipped") is None
    assert len(hub.agent_posts()) == 3


def test_a_post_the_hub_refuses_does_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("hub down")

    monkeypatch.setattr(runtime.client, "conversation_send", refuse)
    for _ in range(6):
        assert _post(runtime)["code"] == "post_failed"
    monkeypatch.undo()
    _rig(monkeypatch, hub, {})
    # Six failed attempts left no rows: the same body is not a double send and
    # this is post 1 of the run, so no warning either.
    landed = _post(_runtime(hub, a, tmp_path))
    assert landed["ok"] is True and "warning" not in landed and "skipped" not in landed


# -- two agents that answer every wake --------------------------------------------


def _converse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    caps: bool,
    rounds: int = 60,
    step: float = 5.0,
    varied: bool = False,
    until_quiet: bool = True,
) -> tuple[Hub, list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive two "reply to every wake" agents until the conversation goes quiet.

    Each round: every agent's connector runs one real iteration (feed →
    screen → admit → coalesced card); every card it received becomes ONE
    ``channel_post`` through the real bridge tool (the guard decides whether
    it lands). A refusal is the agent ending its turn. The clock advances
    ``step`` seconds per round so coalescing windows elapse. ``varied`` gives
    every post its own text, the way model replies differ, so no duplicate
    rule can be what ends the loop. ``until_quiet=False`` runs all ``rounds``
    without demanding the conversation end (an old hub has no brake, and the
    client deliberately adds none).
    """

    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    owners = {"a": _owner("a", 1), "b": _owner("b", 2)}
    dirs = {name: tmp_path / name for name in owners}
    for name, owner in owners.items():
        dirs[name].mkdir()
        _enroll(dirs[name], owner, f"sess_{name}")
    hub = Hub(members=("bep_a", "bep_b"), caps=caps)
    wakes = _rig(monkeypatch, hub, dirs)
    runtimes = {name: _runtime(hub, owners[name], dirs[name]) for name in owners}
    refusals: list[dict[str, Any]] = []

    hub.post(None, "morning, anyone around?")  # a human's ambient message
    for _ in range(rounds):
        before = len(wakes)
        for name, owner in owners.items():
            _iterate(owner, dirs[name])
        new = wakes[before:]
        for wake in new:
            name = wake["endpoint"].removeprefix("bep_")
            body = f"{name} answering #{len(hub.agent_posts())}" if varied else f"{name} answering"
            outcome = _post(runtimes[name], body=body)
            if not outcome["ok"] or outcome.get("skipped"):
                refusals.append(outcome)
        clock.now += step
        if not new and not any(hub.feeds.values()):
            break
    else:
        if until_quiet:
            raise AssertionError("the conversation never went quiet: runaway")
    return hub, wakes, refusals


def test_a_simulated_pre_294_hub_with_agent_run_caps_ends_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins only what a hub WITH the agent-run caps did: prod before #294.

    agent-server #294 deletes those caps, so this is not the bound after it
    deploys: from then on only the S-2 room budget remains, and it covers
    unaddressed group/channel posts only — DM and mention loops are unbounded
    by operator decision (review round 3, S-8; reply_guard's docstring). The
    client refuses nothing either way.
    """

    hub, wakes, refusals = _converse(tmp_path, monkeypatch, caps=True)
    assert refusals == []
    posts = hub.agent_posts()
    # The server's soft cap stops agent→agent fan-out at a run of 4; the posts
    # that run produced are the whole conversation.
    assert len(posts) <= hub.soft_cap + 1, [m.agent_run for m in posts]
    assert max(m.agent_run for m in posts) <= hub.soft_cap + 1
    assert len(wakes) <= len(posts) + 2  # the human's 2 wakes + one per fanned-out post
    print(f"\ncaps on: {len(posts)} agent posts, {len(wakes)} wakes, {len(refusals)} refusals")


def test_on_an_old_hub_the_client_never_refuses_an_agent_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator, 2026-11-03: no client cap on agent replies. An OLD hub with no
    wake brake fans every message to every agent, so two reply-to-everything
    agents keep going; the client delivers every post (none refused, none
    skipped: each answers a new message). After #294 a DM or addressed loop
    has no server bound either; this pins the accepted trade rather than
    hiding it.
    """

    hub, wakes, refusals = _converse(
        tmp_path, monkeypatch, caps=False, rounds=40, step=5.0, until_quiet=False
    )
    posts = hub.agent_posts()
    assert refusals == []
    assert len(posts) >= 60, len(posts)  # two agents, ~one post each per round
    print(f"\ncaps off: {len(posts)} agent posts over 40 rounds, 0 refusals")


# -- the buffered card survives a stop (review round 1, M1) --------------------


def _coalescing_holds(config_dir: Path, owner: Any) -> int:
    from local_operator.team_bridge.store import Store

    return Store(config_dir).hold_count_readonly(owner, coalesce_mod.HOLD_WAKE_COALESCING)


def _one_message_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Path], Hub, list[dict[str, Any]]]:
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    dirs = {"a": tmp_path / "a"}
    dirs["a"].mkdir()
    _enroll(dirs["a"], a, "sess_a")
    hub = Hub(members=("bep_a",))
    return a, dirs, hub, _rig(monkeypatch, hub, dirs)


def test_a_stop_during_the_coalesced_flush_keeps_the_buffered_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: a BaseException at the hand-off used to lose the wake for good.

    The reviewer reproduced the state after a stop as ``feed_refs=0,
    screening_holds=0, owed reservations=1`` — nothing re-drove the message, so
    an hour of iterations delivered 0 wakes instead of 1. The ``wake_coalescing``
    hold is written when the card is BUFFERED now, which is the only moment
    before the feed ref (the message's sole durable locator) disappears.
    """

    a, dirs, hub, wakes = _one_message_rig(tmp_path, monkeypatch)

    async def stopped_at_the_handoff(owner: Any, payload: str, **kwargs: Any) -> str:
        # A connector stop's cancellation or an interrupt: both are
        # BaseException, and neither is a delivery failure to retry in place.
        raise KeyboardInterrupt

    monkeypatch.setattr(conversation_mod, "deliver_wake", stopped_at_the_handoff)
    sent = hub.post(None, "one message")
    with pytest.raises(KeyboardInterrupt):
        _iterate(a, dirs["a"])
    assert wakes == [], "the card never went out"
    assert _coalescing_holds(dirs["a"], a) == 1, "durable BEFORE the ref could go"

    # RESTART: a fresh process (fresh coalescer) over the same store, an hour on.
    monkeypatch.undo()
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    restarted_wakes = _rig(monkeypatch, hub, dirs)
    clock.now += 3_600
    _iterate(a, dirs["a"])

    assert len(restarted_wakes) == 1, "the restart delivered the buffered card"
    assert _card_field(restarted_wakes[0]["card"], "message_id") == sent.id
    assert _coalescing_holds(dirs["a"], a) == 0, "delivered, so the hold is cleared"


def test_a_kill_between_the_ref_delete_and_the_flush_leaves_a_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other way in: SIGKILL after the refs are deleted, before any flush.

    The iteration's inner half IS that process — it deletes the feed refs as its
    last act and never reaches ``run_iteration``'s flush.
    """

    a, dirs, hub, wakes = _one_message_rig(tmp_path, monkeypatch)
    sent = hub.post(None, "one message")
    # The buffering scope WITHOUT the flush: the iteration's end (where the flush
    # runs) is exactly what a kill takes away, after it has already deleted the
    # message's feed refs.
    with coalesce_mod.CONNECTOR_COALESCER.scope():
        asyncio.run(
            loop_mod._run_iteration_inner(
                a,
                config_dir=dirs["a"],
                access=SimpleNamespace(credential_id=a.credential_id),
                auth_store=None,
            )
        )
    assert wakes == []
    assert _coalescing_holds(dirs["a"], a) == 1

    monkeypatch.undo()
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    restarted_wakes = _rig(monkeypatch, hub, dirs)
    clock.now += 3_600
    _iterate(a, dirs["a"])
    assert len(restarted_wakes) == 1
    assert _card_field(restarted_wakes[0]["card"], "message_id") == sent.id


# -- the DM lane wakes too (operator-requested coverage) ----------------------


def _dm_message_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Path], Hub, list[dict[str, Any]]]:
    """The one-message rig with the REAL wake delivery (the stub is undone).

    The coalescing tests stub ``deliver_wake``; the DM lane is about what that
    function does, so it is restored to the module's own seam and only the two
    primitives it calls are doubled.
    """

    from local_operator.team_bridge.connector import wake as wake_mod

    a, dirs, hub, wakes = _one_message_rig(tmp_path, monkeypatch)
    monkeypatch.setattr(conversation_mod, "deliver_wake", wake_mod.deliver_wake)
    return a, dirs, hub, wakes


def test_a_dm_card_wakes_a_live_idle_session_through_the_mailbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's DM question, against the EXISTING primitives.

    No new transport: the card goes out through ``deliver_wake``, which mails it
    into the designated session (the same call the Send tool's live delivery
    uses) with wake on, stamped with the bridge provenance kind so the receiving
    session renders it as a bridge message rather than a generic peer one.
    """

    from local_operator.mobile import peer_send as peer_mod

    a, dirs, hub, _wakes = _dm_message_rig(tmp_path, monkeypatch)
    calls: dict[str, Any] = {}

    async def fake_outcome(
        record: Any,
        *,
        session_id: str,
        text: str,
        mode: str,
        wake: bool,
        sender: Any,
        custom_type: str | None = None,
        bridge_meta: Any = None,
    ) -> Any:
        calls["live"] = {
            "session_id": session_id,
            "text": text,
            "mode": mode,
            "wake": wake,
            "custom_type": custom_type,
        }
        return SimpleNamespace(detail="delivered (live)", state="delivered")

    monkeypatch.setattr(peer_mod, "session_has_durable_history", lambda sid: True)
    monkeypatch.setattr(peer_mod, "deliver_peer_message_outcome", fake_outcome)
    sent = hub.post(None, "hello in a dm", kind="dm")

    outcome = _iterate(a, dirs["a"])

    assert outcome["handled"][0]["action"] == "delivered"
    assert calls["live"]["session_id"] == "sess_a"
    # A mailbox INJECT with the wake flag — the live-idle path, not a bare badge.
    assert calls["live"]["mode"] == "mailbox" and calls["live"]["wake"] is True
    assert calls["live"]["custom_type"] == "team_bridge_message"
    assert _card_field(calls["live"]["text"], "message_id") == sent.id
    assert _card_field(calls["live"]["text"], "conversation_id") == CONV


def test_a_dm_card_engages_a_cold_session_and_carries_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold with no durable history: the errand CARRIES the card, never a bare wake."""

    from local_operator.mobile import peer_send as peer_mod
    from local_operator.session.runtime import launch as launch_mod

    a, dirs, hub, _wakes = _dm_message_rig(tmp_path, monkeypatch)
    monkeypatch.setattr(peer_mod, "session_has_durable_history", lambda sid: False)
    calls: dict[str, Any] = {}

    async def fake_engage(session_id: str, cwd: str, work: Any, *, config_dir: Path) -> Any:
        calls["cold"] = {
            "session_id": session_id,
            "work": type(work).__name__,
            "text": work.text,
            "mode": work.mode,
            "wake": work.wake,
        }
        return SimpleNamespace(state="started")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    sent = hub.post(None, "hello in a dm", kind="dm")

    _iterate(a, dirs["a"])

    assert calls["cold"]["session_id"] == "sess_a"
    assert calls["cold"]["work"] == "PeerMessageErrand"
    assert calls["cold"]["mode"] == "mailbox" and calls["cold"]["wake"] is True
    assert _card_field(calls["cold"]["text"], "message_id") == sent.id


# -- an arrival that wakes nobody must be visible (review round 1, 7b) --------


def test_an_arrival_that_wakes_nobody_is_logged_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Consent with no designated session: nothing wakes, and that used to be silent.

    The outcome stays ``badge_only`` (a real state, not a failure), but the owner
    can now see it: one WARNING with the typed code, and a counter the receiver
    status payload exposes.
    """

    from local_operator.team_bridge.store import Store
    from local_operator.team_bridge.wake_diagnostics import woke_nobody_counts

    a = _owner("a", 1)
    dirs = {"a": tmp_path / "a"}
    dirs["a"].mkdir()
    _enroll(dirs["a"], a, "sess_a")
    Store(dirs["a"]).update_designated_session(a, None)
    hub = Hub(members=("bep_a",))
    wakes = _rig(monkeypatch, hub, dirs)
    hub.post(None, "nobody to wake")

    with caplog.at_level("WARNING"):
        outcome = _iterate(a, dirs["a"])

    first = outcome["handled"][0]
    assert first["action"] == "badge_only"
    assert first["code"] == "no_designation"
    assert wakes == [], "a badge-only arrival wakes nothing"
    assert "code=no_designation" in caplog.text
    assert woke_nobody_counts(dirs["a"]) == {"no_designation": 1}


def test_a_redrive_killed_before_its_flush_keeps_the_hold_it_just_rewrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1′: the re-drive pass retired the very row it had just rewritten.

    Three steps, exactly as the round-2 review laid them out:

    1. buffer a card, then a hard stop at the hand-off — the durable hold is the
       only trace left (this is round 1's M1, already fixed);
    2. RESTART #1 re-drives that hold and is killed before its flush, so the
       re-driven card is buffered and unposted. The pass used to treat
       ``{"action": "delivered", "detail": "coalesced"}`` as terminal and delete
       the ``wake_coalescing`` row ``enqueue`` had just written — from there to
       the kill nothing durable said the card was owed;
    3. RESTART #2, an hour later, must deliver the card exactly once.

    The first pass and this pass now decide "finished?" through ONE predicate
    (``coalesce.card_is_buffered``), which is what stops them drifting again.
    """

    a, dirs, hub, wakes = _one_message_rig(tmp_path, monkeypatch)

    async def stopped_at_the_handoff(owner: Any, payload: str, **kwargs: Any) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(conversation_mod, "deliver_wake", stopped_at_the_handoff)
    sent = hub.post(None, "one message")
    with pytest.raises(KeyboardInterrupt):
        _iterate(a, dirs["a"])
    assert wakes == []
    assert _coalescing_holds(dirs["a"], a) == 1, "step 1: durable before the ref went"

    # ---- RESTART #1: run the re-drive pass, then die before the flush --------
    monkeypatch.undo()
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    redriven_wakes = _rig(monkeypatch, hub, dirs)
    loop_mod._RETRY_ATTEMPTS.clear()
    clock.now += 3_600
    with coalesce_mod.CONNECTOR_COALESCER.scope():
        asyncio.run(
            loop_mod._run_iteration_inner(
                a,
                config_dir=dirs["a"],
                access=SimpleNamespace(credential_id=a.credential_id),
                auth_store=None,
            )
        )
    assert redriven_wakes == [], "the re-driven card was buffered, not posted"
    assert _coalescing_holds(dirs["a"], a) == 1, "step 2: the kill must not retire it"

    # ---- RESTART #2: the buffered card is delivered, once --------------------
    monkeypatch.undo()
    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    restarted_wakes = _rig(monkeypatch, hub, dirs)
    loop_mod._RETRY_ATTEMPTS.clear()
    # An hour apart, like the review's own repro: the killed pass left a live
    # claim on the body reservation, so the first re-drive after the restart
    # parks on it and the next one (claim expired) takes it. What must NOT
    # happen is that no pass ever wakes the session.
    for _ in range(3):
        clock.now += 3_600
        _iterate(a, dirs["a"])

    assert len(restarted_wakes) == 1, "step 3: the card must not be lost"
    assert _card_field(restarted_wakes[0]["card"], "message_id") == sent.id
    assert _coalescing_holds(dirs["a"], a) == 0, "delivered, so the hold is retired"


def test_a_buffered_handoff_is_never_terminal_for_the_row() -> None:
    """The predicate both passes share, pinned on its own (review round 2, M1′)."""

    from local_operator.team_bridge.connector.coalesce import (
        CARD_COALESCED,
        card_is_buffered,
    )

    # The predicate reads the ``detail``: the spines say it as a bare string,
    # the loop's re-drive reads it off the outcome it just got.
    assert card_is_buffered(CARD_COALESCED) is True
    assert card_is_buffered({"detail": CARD_COALESCED}.get("detail")) is True
    assert card_is_buffered(None) is False
    assert card_is_buffered("delivered and woke the session") is False
    assert card_is_buffered({"action": "delivered"}.get("detail")) is False


# -- the server's follower bit decides directedness (E2E round 1, Q-1) --------


def test_a_followed_thread_reply_is_directed():
    lines = _card(thread_root_id=ROOT, thread_follower=True).splitlines()
    assert "mention: no" in lines
    assert "directed: thread" in lines
    assert "expect: reply" in lines


def test_a_thread_reply_the_agent_does_not_follow_is_ambient():
    """The server says NO: the reply fanned out to a non-follower.

    This is Q-1: the card used to call any non-mention row with a thread root
    "directed", so an agent was told to reply to a thread it never joined.
    """

    card = _card(thread_root_id=ROOT, thread_follower=False)
    lines = card.splitlines()
    assert "mention: no" in lines
    assert not any(line.startswith("directed:") for line in lines)
    assert "expect: weigh relevance; no_reply if nothing is needed" in lines
    assert "expect: reply" not in lines


def test_without_the_bit_the_thread_root_keeps_its_old_meaning():
    """An older server: non-mention thread rows reached followers only."""

    lines = _card(thread_root_id=ROOT).splitlines()
    assert "directed: thread" in lines and "expect: reply" in lines
    assert "thread_root_id: %s (reply to)" % ROOT in lines


def test_a_mention_outranks_the_follower_bit():
    """Being named is being named, whatever the thread says."""

    lines = _card(mention=True, thread_root_id=ROOT, thread_follower=False).splitlines()
    assert "mention: yes" in lines and "expect: reply" in lines


def test_the_live_row_carries_the_bit_and_the_kind_onto_the_card() -> None:
    """The facts come off the LIVE read, not the durable ref (round 1, m3)."""

    from local_operator.team_bridge.connector.conversation import _card_metadata

    owner = _owner("a", 1)
    ref = ConversationFeedRef(
        conversation_id=CONV,
        message_id="bcmsg_m1",
        conversation_seq=4,
        mention=False,
        sender_kind="agent",
        thread_root_id=ROOT,
    ).validated()
    display = {
        "conversation_name": "#agents",
        "sender_label": "Ada",
        "sender_kind": "human",
        "thread_follower": False,
        "conversation_kind": "group",
    }
    metadata = _card_metadata(owner, ref, display)
    assert metadata.thread_follower is False
    assert metadata.conversation_kind == "group"
    assert card_directed(metadata) == "none"
    # And the ref itself still carries neither (the downgrade rule).
    assert not hasattr(ref, "thread_follower")


@pytest.mark.parametrize(
    "display",
    [
        {"thread_follower": "yes"},
        {"thread_follower": 1},
        {"conversation_kind": "room"},
        {"conversation_kind": 7},
    ],
)
def test_a_malformed_live_routing_fact_is_refused(display: dict[str, Any]) -> None:
    from local_operator.team_bridge.connector.conversation import _card_metadata

    ref = ConversationFeedRef(
        conversation_id=CONV,
        message_id="bcmsg_m1",
        conversation_seq=4,
        mention=False,
        sender_kind="human",
    ).validated()
    with pytest.raises(ValueError):
        _card_metadata(_owner("a", 1), ref, {"sender_kind": "human", **display})


# -- a 1:1 DM card names its sender, not a raw id (E2E round 1, Q-2) ----------


def test_a_dm_card_reads_as_a_direct_message():
    card = _card(conversation_kind="dm", conversation_label="", conversation_id="bcv_abc12345")
    header = card.splitlines()[0]
    assert header == "Direct message · from Ada (human, same tenant)"
    # The id still appears where it is USEFUL (the route line the agent posts
    # with); what Q-2 removes is the id standing in as the conversation's NAME.
    assert "conversation_id: bcv_abc12345" in card


def test_a_dm_thread_reply_says_thread_reply_and_direct_message():
    card = _card(conversation_kind="dm", conversation_label="", thread_root_id=ROOT)
    assert card.splitlines()[0] == "Thread reply · Direct message · from Ada (human, same tenant)"


def test_a_channel_card_keeps_its_name_label():
    card = _card(conversation_kind="channel")
    assert card.splitlines()[0] == "Channel message · #agents · from Ada (human, same tenant)"


def test_without_a_kind_on_the_wire_the_label_is_unchanged():
    """An older server sends no kind: the pre-Q-2 text is kept verbatim."""

    card = _card(conversation_label="", conversation_id="bcv_wakeall01")
    assert card.splitlines()[0] == "Channel message · bcv_wakeall01 · from Ada (human, same tenant)"


def test_a_dm_conversation_reaches_the_card_as_a_direct_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the REAL iteration: the live row's kind is what the card reads."""

    a, dirs, hub, wakes = _one_message_rig(tmp_path, monkeypatch)
    hub.post(None, "hello in a dm", kind="dm")
    _iterate(a, dirs["a"])
    assert len(wakes) == 1
    assert wakes[0]["card"].splitlines()[0] == "Direct message · from Ada (human, same tenant)"


def _captured_directed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, thread_follower: bool | None
) -> Any:
    """Run one human thread reply through the REAL iteration; return the brake's bit."""

    a, dirs, hub, wakes = _one_message_rig(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def fake_record(config_dir: Path, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(reply_guard, "record_trigger", fake_record)
    hub.post(
        None,
        "a reply in a thread",
        thread_root="bcmsg_root0001",
        thread_follower=thread_follower,
    )
    _iterate(a, dirs["a"])
    assert len(wakes) == 1, "the card still goes out"
    return captured


def test_a_followed_thread_reply_lifts_the_brake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 6, M2: the predicate is wired into the trigger record."""

    captured = _captured_directed(tmp_path, monkeypatch, thread_follower=True)
    assert captured["directed"] is True
    assert captured["mention"] is False


def test_a_reply_without_the_bit_does_not_lift_the_brake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older server sends no bit: the card derives `thread`, the brake does not.

    This is the regression the review found: `directed=card_directed(...)` made
    any human reply in any thread lift an agent-to-agent cool-down.
    """

    captured = _captured_directed(tmp_path, monkeypatch, thread_follower=None)
    assert captured["directed"] is False


def test_an_unfollowed_thread_reply_does_not_lift_the_brake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _captured_directed(tmp_path, monkeypatch, thread_follower=False)
    assert captured["directed"] is False


def test_a_group_settled_before_its_flush_is_an_outcome_not_a_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CI shard 1: a card the flush declines to send must SAY it declined.

    The replay path can settle a reservation while its card is still buffered (a
    hold re-drive, a body turn). ``flush_due`` then owes no wake — correct — but
    it used to delete the group and record NOTHING, which reads exactly like a
    lost card to anything watching the iteration summary (a test asserting an
    empty transcript, an owner reading ``coalesced_wakes``). It now reports
    ``already_posted`` and logs the verdict once.

    ``_still_owed`` is stubbed to the settled verdict so the cell pins the
    OUTCOME plumbing without re-testing the store read, which has its own cells.
    """

    from local_operator.team_bridge.connector import coalesce as coalesce_mod

    config_dir = tmp_path / "a"
    config_dir.mkdir()
    owner = SimpleNamespace(
        endpoint_id="bep_a",
        tenant_id="tnt_1",
        account_id="acc_a",
        credential_id=1,
        canonical_base_url="https://api.example/v1",
        enrollment_generation=1,
    )
    metadata = ConversationCardMetadata(
        conversation_id=CONV,
        tenant_id="tnt_1",
        conversation_label="#agents",
        sender_label="Ada",
        sender_kind="human",
        mention=False,
        message_id="bcmsg_m9",
        conversation_seq=9,
    )
    monkeypatch.setattr(coalesce_mod, "_still_owed", lambda group: [])
    coalescer = coalesce_mod.WakeCoalescer(window_s=0.0)
    with coalescer.scope():
        coalescer.enqueue(
            owner, metadata, ref_key="ref-nine", session_id="sess_a", config_dir=config_dir
        )

    async def never(_owner: Any, _metadata: Any, **_kwargs: Any) -> str:
        raise AssertionError("a settled group must not be delivered")

    with caplog.at_level(logging.INFO, logger=coalesce_mod.logger.name):
        outcomes = asyncio.run(
            coalescer.flush_due(deliver=never, owner_filter=owner, config_dir=config_dir)
        )
    assert [outcome["action"] for outcome in outcomes] == ["already_posted"]
    assert outcomes[0]["conversation_id"] == CONV
    assert "code=card_already_posted" in caplog.text
    # And the group is gone, so a later pass does not report it again.
    again = asyncio.run(
        coalescer.flush_due(deliver=never, owner_filter=owner, config_dir=config_dir)
    )
    assert again == []


# -- #3377 deferred follow-ups (RG5 / RG7) ------------------------------------


@pytest.mark.parametrize("kind", ["", "mixed", "unknown"])
def test_an_unknown_sender_classifies_as_agent_chatter(tmp_path: Path, kind: str) -> None:
    """Round 2, S-5: a hub from before ``sender_kind`` records every sender as
    ``unknown``; only a positively human trigger is exempt from the agent-to-agent
    classification. Classification only: the post is never refused."""

    now = 1_000_000.0
    _trigger(tmp_path, kind, now)
    slot = _reserve(tmp_path, now + 1, body="hello")
    assert isinstance(slot, reply_guard.Reservation) and slot.agent_to_agent
    # A stale unknown trigger explains nothing, like a stale agent one.
    stale = _reserve(tmp_path, now + reply_guard.TRIGGER_TTL_S + 2, body="later")
    assert isinstance(stale, reply_guard.Reservation) and not stale.agent_to_agent


def test_a_human_trigger_is_still_the_only_exemption(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "human", now)
    slot = _reserve(tmp_path, now + 1, body="hello")
    assert isinstance(slot, reply_guard.Reservation) and not slot.agent_to_agent


@pytest.mark.parametrize(
    "reason",
    [
        "reply guard key is damaged",
        "reply guard key is unreadable or unwritable",
        "reply guard state file is unreadable or unwritable",
        "reply guard state directory is unreadable or unwritable",
    ],
)
def test_a_permanent_guard_failure_says_stop_and_report(reason: str) -> None:
    """Round 2, S-6: a permanent failure never recovers, so the line tells the
    agent the post landed, not to resend or retry, and to report it — never a
    wait-and-retry that would wake it every few seconds for ever."""

    text = reply_guard.unavailable_warning(reply_guard.ReplyGuardUnavailable(reason))
    assert text.endswith("do not resend or retry; report this to the operator")
    assert reason in text and "posted without" in text
    assert not re.search(r"\d+\s*s\b|retry in|resumes|arm a wake", text)
    # The transient class keeps asking nothing of anyone.
    busy = reply_guard.unavailable_warning(
        reply_guard.ReplyGuardUnavailable("reply guard state is busy", transient=True)
    )
    assert "report" not in busy and "retry" not in busy


def test_no_guard_result_states_a_retry_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 2, S-7: a stated delay longer than ``TRIGGER_TTL_S`` was not the
    earliest admit time it claimed. With no ceiling left, no guard outcome may
    state a retry time at all — not the skip, not a run warning, not the
    unusable-state lines, and not through the real tool."""

    clock = Clock()
    monkeypatch.setattr(time, "time", clock)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    results = [_post(runtime, body=f"part {index}") for index in range(1, 11)]
    clock.now += 1
    results.append(_post(runtime, body="part 10"))  # the exact double send
    assert results[-1]["skipped"] is True
    texts = [str(result.get("warning") or "") for result in results]
    texts += [
        reply_guard.unavailable_warning(reply_guard.ReplyGuardUnavailable("x", transient=True)),
        reply_guard.unavailable_warning(reply_guard.ReplyGuardUnavailable("x")),
    ]
    assert all("retry_after_s" not in result for result in results)
    assert sum(bool(text) for text in texts) >= 5  # 5th, 10th, the skip, both unusable
    for text in texts:
        assert not re.search(r"retry (in|after)|try again in|resumes in|\bwait\b", text), text


def test_a_dropped_advisory_fault_logs_its_class_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Security round 5, n-3: the swallowed post-delivery fault leaves one DEBUG
    line naming the exception class — never its message, which may carry
    hub-supplied values."""

    from local_operator.team_bridge import tool as tool_mod

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise OverflowError("SENTINEL-body-text 10**30")

    monkeypatch.setattr(tool_mod.reply_guard, "agent_run_warning", explode)
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    with caplog.at_level(logging.DEBUG, logger=tool_mod.__name__):
        result = _post(runtime, body="landed")
    assert result["ok"] is True and "warning" not in result
    lines = [r.getMessage() for r in caplog.records if r.name == tool_mod.__name__]
    assert "bridge: agent-run advice dropped code=OverflowError" in lines
    assert not any("SENTINEL" in line or "landed" in line for line in lines)
