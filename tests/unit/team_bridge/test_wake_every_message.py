"""Wake-on-every-message: card, durable ref, coalescing, reply guard, two agents.

The wake contract (2026-10-30) wakes an agent member for EVERY message in a
conversation it belongs to and leaves relevance to the agent. That is only safe
with three bounds, each pinned here at its real seam:

* the CARD says whether a reply is expected (mention / followed thread) or the
  message is ambient (weigh it, end with ``no_reply``), and cautions against
  answering an agent's ambient chatter;
* the connector COALESCES a burst into one wake per (endpoint, conversation)
  window, while every message is still screened on its own;
* ``channel_post`` enforces a per-conversation reply BUDGET and an
  agent-to-agent COOL-DOWN.

The last test drives two agents whose policy is "reply to every wake" through
the real connector iteration and the real bridge tool against a fake hub that
applies the server's run caps, and once more against an OLD hub with no caps,
and asserts the conversation stops on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
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
    assert "directed: mention" in lines
    assert "expect: reply" in lines
    assert not any("no_reply" in line for line in lines)
    assert not any(line.startswith("agent sender:") for line in lines)


def test_an_unmentioned_card_says_weigh_it_and_no_reply_is_the_way_out() -> None:
    lines = _card(mention=False, conversation_kind="channel").splitlines()
    assert "mention: no" in lines
    assert "directed: none" in lines
    assert "conversation_kind: channel" in lines
    assert "expect: weigh relevance; if nothing is needed end with no_reply" in lines
    assert "expect: reply" not in lines
    # A human sender gets no agent caution.
    assert not any(line.startswith("agent sender:") for line in lines)


def test_an_unmentioned_agent_card_cautions_against_answering_it() -> None:
    lines = _card(sender_kind="agent", sender_label="Bea's agent").splitlines()
    assert "agent sender: do not reply unless it asks you a direct question" in lines


def test_a_mentioning_agent_card_carries_no_caution() -> None:
    # An agent that @-mentions you asked you something: answer it.
    lines = _card(sender_kind="agent", mention=True).splitlines()
    assert "expect: reply" in lines
    assert not any(line.startswith("agent sender:") for line in lines)


def test_a_followed_thread_reply_is_directed_without_a_mention() -> None:
    lines = _card(thread_root_id=ROOT, thread_follower=True, sender_kind="agent").splitlines()
    assert "mention: no" in lines
    assert "directed: thread" in lines
    assert "expect: reply" in lines
    assert f"thread_root_id: {ROOT}" in lines
    assert f"reply_to_message_id: {ROOT}" in lines
    # Directed at you, so the agent-sender caution does not apply.
    assert not any(line.startswith("agent sender:") for line in lines)
    # The thread route stays directly under thread_root_id, above the footer.
    assert lines[-3].startswith("thread_root_id: ")
    assert lines[-2].startswith("reply in thread: ")


def test_a_thread_reply_the_agent_does_not_follow_is_ambient() -> None:
    lines = _card(thread_root_id=ROOT, thread_follower=False).splitlines()
    assert "directed: none" in lines
    assert "expect: weigh relevance; if nothing is needed end with no_reply" in lines


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
    assert "agent sender:" not in card


# -- durable ref -------------------------------------------------------------


def test_the_ref_round_trips_every_wake_tag() -> None:
    ref = ConversationFeedRef(
        conversation_id=CONV,
        message_id="bcmsg_m1",
        conversation_seq=4,
        mention=False,
        sender_kind="agent",
        thread_root_id=ROOT,
        thread_follower=True,
        agent_run=3,
        conversation_kind="channel",
    ).validated()
    encoded = ref.encode()
    assert encoded == (
        f"{PREFIX}{CONV}/bcmsg_m1/4/0/agent/thread_root={ROOT}"
        "/thread_follower=1/agent_run=3/conversation_kind=channel"
    )
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
    assert (ref.thread_follower, ref.agent_run, ref.conversation_kind) == (False, 0, None)
    assert ref.encode() == PREFIX + value


@pytest.mark.parametrize(
    "value",
    [
        f"{CONV}/bcmsg_m1/4/0/agent/agent_run=x",
        f"{CONV}/bcmsg_m1/4/0/agent/agent_run=01",
        f"{CONV}/bcmsg_m1/4/0/agent/thread_follower=yes",
        f"{CONV}/bcmsg_m1/4/0/agent/conversation_kind=room",
        # A follower bit with no thread to follow.
        f"{CONV}/bcmsg_m1/4/0/agent/thread_follower=1",
        # Wake tags need a stated sender kind, like thread_root.
        f"{CONV}/bcmsg_m1/4/0/agent_run=2",
    ],
)
def test_a_malformed_wake_tag_is_refused_not_guessed(value: str) -> None:
    assert parse_conversation_ref_value(value) is None


def test_the_feed_projection_carries_the_wake_facts_into_the_ref() -> None:
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
                        "reply_to_message_id": ROOT,
                        "thread_follower": True,
                        "agent_run": 2,
                        "conversation_kind": "group",
                    },
                    "at": "2026-10-30T00:00:00Z",
                }
            ],
        }
    )
    ref = parse_conversation_ref_value(refs[0][len(PREFIX) :])
    assert ref is not None
    assert (ref.thread_root_id, ref.thread_follower, ref.agent_run, ref.conversation_kind) == (
        ROOT,
        True,
        2,
        "group",
    )


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


# -- reply guard -------------------------------------------------------------


def _reserve(config_dir: Path, now: float) -> Any:
    return reply_guard.reserve_post(config_dir, endpoint_id="bep_a", conversation_id=CONV, now=now)


def _trigger(config_dir: Path, kind: str, now: float, mention: bool = False) -> None:
    reply_guard.record_trigger(
        config_dir,
        endpoint_id="bep_a",
        conversation_id=CONV,
        sender_kind=kind,
        mention=mention,
        now=now,
    )


def test_the_budget_allows_six_posts_per_ten_minutes(tmp_path: Path) -> None:
    now = 1_000_000.0
    _trigger(tmp_path, "human", now)
    for index in range(reply_guard.BUDGET_POSTS):
        assert isinstance(_reserve(tmp_path, now + index), reply_guard.Reservation)
    refused = _reserve(tmp_path, now + 10)
    assert isinstance(refused, reply_guard.Refusal)
    assert refused.code == reply_guard.CODE_BUDGET
    result = refused.as_result()
    assert result["ok"] is False and "no_reply" in result["next"]
    assert 0 < result["retry_after_s"] <= reply_guard.BUDGET_WINDOW_S
    # The window slides: the first post ages out.
    later = now + reply_guard.BUDGET_WINDOW_S + 1
    assert isinstance(_reserve(tmp_path, later), reply_guard.Reservation)


def test_the_budget_is_per_conversation(tmp_path: Path) -> None:
    now = 1_000_000.0
    for index in range(reply_guard.BUDGET_POSTS):
        _reserve(tmp_path, now + index)
    other = reply_guard.reserve_post(
        tmp_path, endpoint_id="bep_a", conversation_id="bcv_otherconv", now=now + 10
    )
    assert isinstance(other, reply_guard.Reservation)


def test_three_agent_to_agent_turns_start_a_cool_down(tmp_path: Path) -> None:
    now = 1_000_000.0
    for index in range(reply_guard.COOLDOWN_AFTER):
        _trigger(tmp_path, "agent", now + index)
        slot = _reserve(tmp_path, now + index)
        assert isinstance(slot, reply_guard.Reservation) and slot.agent_to_agent
    _trigger(tmp_path, "agent", now + 5)
    refused = _reserve(tmp_path, now + 5)
    assert isinstance(refused, reply_guard.Refusal) and refused.code == reply_guard.CODE_COOLDOWN
    # A human post WITHOUT a mention does not lift it...
    _trigger(tmp_path, "human", now + 6, mention=False)
    assert isinstance(_reserve(tmp_path, now + 6), reply_guard.Refusal)
    # ...it expires on its own...
    assert isinstance(_reserve(tmp_path, now + 5 + reply_guard.COOLDOWN_S), reply_guard.Reservation)


def test_a_human_mention_resets_the_run_and_lifts_the_cool_down(tmp_path: Path) -> None:
    now = 1_000_000.0
    for index in range(reply_guard.COOLDOWN_AFTER):
        _trigger(tmp_path, "agent", now + index)
        _reserve(tmp_path, now + index)
    assert isinstance(_reserve(tmp_path, now + 4), reply_guard.Refusal)
    _trigger(tmp_path, "human", now + 5, mention=True)
    slot = _reserve(tmp_path, now + 5)
    assert isinstance(slot, reply_guard.Reservation) and not slot.agent_to_agent


def test_a_human_prompted_post_breaks_the_agent_run(tmp_path: Path) -> None:
    now = 1_000_000.0
    for step, kind in enumerate(["agent", "agent", "human", "agent", "agent"]):
        _trigger(tmp_path, kind, now + step)
        assert isinstance(_reserve(tmp_path, now + step), reply_guard.Reservation)


def test_a_released_slot_returns_the_budget(tmp_path: Path) -> None:
    now = 1_000_000.0
    slots = [_reserve(tmp_path, now + i) for i in range(reply_guard.BUDGET_POSTS)]
    reply_guard.release(tmp_path, slots[-1], endpoint_id="bep_a", conversation_id=CONV)
    assert isinstance(_reserve(tmp_path, now + 10), reply_guard.Reservation)


def test_an_unreadable_guard_fails_closed(tmp_path: Path) -> None:
    # A FILE where the state directory belongs: the guard cannot open its db.
    (tmp_path / "team_bridge").write_text("not a directory")
    with pytest.raises(reply_guard.ReplyGuardUnavailable):
        _reserve(tmp_path, 1.0)


# -- the shared multi-message rig ------------------------------------------------


@dataclass
class Msg:
    id: str
    seq: int
    sender_endpoint: str | None  # None = the human
    body: str
    mentions: frozenset[str] = frozenset()
    agent_run: int = 0

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
        self, sender_endpoint: str | None, body: str, mentions: frozenset[str] = frozenset()
    ) -> Msg:
        self.run = 0 if sender_endpoint is None else self.run + 1
        seq = len(self.messages) + 1
        msg = Msg(f"bcmsg_m{seq:04d}", seq, sender_endpoint, body, mentions, self.run)
        self.messages[msg.id] = msg
        for endpoint in self.members:
            if endpoint == sender_endpoint or not self._wakes(endpoint, msg):
                continue
            ids: dict[str, Any] = {
                "conversation_id": CONV,
                "message_id": msg.id,
                "conversation_seq": msg.seq,
                "sender_kind": msg.sender_kind,
                "conversation_kind": "channel",
            }
            if endpoint in mentions:
                ids["mention"] = True
            if msg.agent_run:
                ids["agent_run"] = msg.agent_run
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
        }

    async def conversation_admit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


class ToolClient:
    """The bridge tool's client: a post lands on the hub as THIS agent."""

    def __init__(self, hub: Hub, endpoint: str) -> None:
        self.hub, self.endpoint = hub, endpoint

    async def conversation_send(self, conversation_id: str, *, text: str, **kwargs: Any):
        assert conversation_id == CONV
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
    assert _card_field(card, "conversation_kind") == "channel"
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


def test_channel_post_refuses_past_the_budget_with_a_structured_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)
    for _ in range(reply_guard.BUDGET_POSTS):
        assert _post(runtime)["ok"] is True
    refused = _post(runtime)
    assert refused["ok"] is False
    assert refused["code"] == reply_guard.CODE_BUDGET
    assert refused["retry_after_s"] > 0
    assert (
        len(hub.agent_posts()) == reply_guard.BUDGET_POSTS
    ), "the refused post never reached the hub"


def test_a_post_the_hub_refuses_does_not_spend_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _owner("a", 1)
    hub = Hub(members=("bep_a",))
    _rig(monkeypatch, hub, {})
    runtime = _runtime(hub, a, tmp_path)

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("hub down")

    monkeypatch.setattr(runtime.client, "conversation_send", refuse)
    for _ in range(reply_guard.BUDGET_POSTS + 2):
        assert _post(runtime)["code"] == "post_failed"
    monkeypatch.undo()
    _rig(monkeypatch, hub, {})
    assert _post(_runtime(hub, a, tmp_path))["ok"] is True


# -- two agents that answer every wake --------------------------------------------


def _converse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, caps: bool, rounds: int = 60
) -> tuple[Hub, list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive two "reply to every wake" agents until the conversation goes quiet.

    Each round: every agent's connector runs one real iteration (feed →
    screen → admit → coalesced card); every card it received becomes ONE
    ``channel_post`` through the real bridge tool (the guard decides whether
    it lands). A refusal is the agent ending its turn with no_reply. The
    clock advances 5 s per round so coalescing windows elapse.
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
            outcome = _post(runtimes[name], body=f"{name} answering")
            if not outcome["ok"]:
                refusals.append(outcome)
        clock.now += 5.0
        if not new and not any(hub.feeds.values()):
            break
    else:
        raise AssertionError("the conversation never went quiet: runaway")
    return hub, wakes, refusals


def test_two_reply_to_everything_agents_stop_under_the_server_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub, wakes, refusals = _converse(tmp_path, monkeypatch, caps=True)
    posts = hub.agent_posts()
    # The server's soft cap stops agent→agent fan-out at a run of 4; the posts
    # that run produced are the whole conversation.
    assert len(posts) <= hub.soft_cap + 1, [m.agent_run for m in posts]
    assert max(m.agent_run for m in posts) <= hub.soft_cap + 1
    assert len(wakes) <= len(posts) + 2  # the human's 2 wakes + one per fanned-out post
    print(f"\ncaps on: {len(posts)} agent posts, {len(wakes)} wakes, {len(refusals)} refusals")


def test_two_reply_to_everything_agents_stop_on_the_client_guard_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OLD hub fans every message out to every agent. The client guard
    must end it: each agent gets one human-prompted post plus at most
    COOLDOWN_AFTER agent-to-agent posts before its cool-down refuses."""

    hub, wakes, refusals = _converse(tmp_path, monkeypatch, caps=False)
    posts = hub.agent_posts()
    per_agent = {e: sum(1 for m in posts if m.sender_endpoint == e) for e in ("bep_a", "bep_b")}
    for endpoint, count in per_agent.items():
        assert count <= 1 + reply_guard.COOLDOWN_AFTER, per_agent
        assert count <= reply_guard.BUDGET_POSTS
    assert len(posts) <= 2 * (1 + reply_guard.COOLDOWN_AFTER)
    assert refusals, "the guard, not luck, ended it"
    assert {r["code"] for r in refusals} <= {reply_guard.CODE_COOLDOWN, reply_guard.CODE_BUDGET}
    print(f"\ncaps off: {per_agent}, {len(wakes)} wakes, refusals={[r['code'] for r in refusals]}")
