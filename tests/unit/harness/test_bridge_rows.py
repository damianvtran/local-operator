"""The Team Bridge tool row: ``bridge_row`` tokens and their keyed English copy.

The tokens (``harness/rows.py::bridge_row``) are a contract three renderers read
— the TUI card, the phone projection and the desktop — so these tests pin the
dict shape and every op's mapping; the copy tests pin what the two core hosts
paint from those tokens (``harness/bridge_copy.py``). The result ``details``
literals mirror ``team_bridge/tool.py``'s return shapes, and one test pins the
op list to the tool's own schema so a new op cannot silently fall through to
"Used Team Bridge".
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest

from local_operator.harness.bridge_copy import (
    bridge_expanded,
    bridge_row_line,
    render_bridge_row,
)
from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.rows import BRIDGE_KNOWN_OPS, bridge_row

CHANNEL = "bcv_b0044d9e1a2f"


def _untrusted(payload: Any) -> str:
    return f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(payload)}"


def _posted(seq: int = 42, **extra: Any) -> dict[str, Any]:
    receipt = {
        "ok": True,
        "state": "sent",
        "conversation_id": CHANNEL,
        "message_id": "bcmsg_01",
        "seq": seq,
        "destination_type": "conversation",
        "content": f"posted to {CHANNEL}: message bcmsg_01 at seq {seq}",
    }
    receipt.update(extra)
    return receipt


def test_known_ops_match_the_tool_schema() -> None:
    from local_operator.team_bridge.tool import BridgeToolParams

    annotation = BridgeToolParams.model_fields["op"].annotation
    assert set(get_args(annotation)) == set(BRIDGE_KNOWN_OPS)


def test_tokens_are_the_documented_json_shape() -> None:
    row = bridge_row({"op": "channel_post", "channel_id": CHANNEL, "body": "hi"}, _posted())
    assert set(row) == {
        "op",
        "action",
        "target",
        "preview",
        "count",
        "emoji",
        "outcome",
        "code",
    }
    assert set(row["target"]) == {"kind", "label"}
    assert json.loads(json.dumps(row)) == row
    # Pure: the same call yields the same dict.
    assert bridge_row({"op": "channel_post", "channel_id": CHANNEL, "body": "hi"}, _posted()) == row


@pytest.mark.parametrize(
    ("args", "details", "expected"),
    [
        (
            {"op": "channel_post", "channel_id": CHANNEL, "name": "general", "body": "Ship it"},
            _posted(),
            {"action": "channel_post", "kind": "channel", "label": "general", "preview": "Ship it"},
        ),
        (
            {"op": "dm", "account_id": "acc_9", "body": "review?"},
            {
                "ok": True,
                "state": "sent",
                "destination_type": "conversation_dm",
                "content": f"direct message to Ana Ruiz in {CHANNEL}: message bcmsg_2 at seq 3",
            },
            {"action": "dm", "kind": "dm", "label": "Ana Ruiz", "preview": "review?"},
        ),
        (
            {"op": "send", "to": {"name": "Riley"}, "text": "ping"},
            {"ok": True, "state": "sent", "destination_type": "peer_inbox"},
            {"action": "send", "kind": "person", "label": "Riley", "preview": "ping"},
        ),
        (
            {"op": "reply", "to": "acc_1", "thread_id": "t", "text": "done"},
            {"ok": True, "state": "sent", "destination_type": "peer_inbox"},
            {"action": "reply", "kind": "person", "label": None, "preview": "done"},
        ),
        (
            {"op": "channels"},
            {"ok": True, "state": "ok", "content": json.dumps({"conversations": [{}, {}]})},
            {"action": "channels", "kind": "none", "label": None, "count": 2},
        ),
        (
            {"op": "channel_create", "name": "draft"},
            {
                "ok": True,
                "state": "created",
                "conversation_id": CHANNEL,
                "content": json.dumps({"id": CHANNEL, "kind": "channel", "name": "launch"}),
            },
            {"action": "channel_create", "kind": "channel", "label": "launch"},
        ),
        (
            {"op": "channel_history", "channel_id": CHANNEL},
            {"ok": True, "state": "ok", "content": _untrusted({"messages": [{}, {}, {}]})},
            {"action": "channel_history", "kind": "channel", "label": None, "count": 3},
        ),
        (
            {"op": "channel_thread", "channel_id": CHANNEL, "thread_root_id": "bcmsg_1"},
            {"ok": True, "state": "ok", "content": _untrusted({"root": {}, "replies": [{}, {}]})},
            {"action": "channel_thread", "kind": "channel", "count": 3},
        ),
        (
            {"op": "react", "channel_id": CHANNEL, "message_id": "bcmsg_1", "emoji": "👍"},
            {"ok": True, "state": "reacted", "message_id": "bcmsg_1"},
            {"action": "react", "kind": "channel", "emoji": "👍"},
        ),
        (
            {
                "op": "react",
                "channel_id": CHANNEL,
                "message_id": "m",
                "emoji": "👍",
                "remove": True,
            },
            {"ok": True, "state": "removed", "message_id": "m"},
            {"action": "react-remove", "kind": "channel", "emoji": "👍"},
        ),
        (
            {"op": "directory"},
            {
                "ok": True,
                "state": "ok",
                "members": [{}, {}],
                "content": _untrusted({"members": [{}, {}], "count": 8}),
            },
            {"action": "directory", "kind": "directory", "label": None, "count": 8},
        ),
        (
            {"op": "shared_conversations"},
            {"ok": True, "state": "ok", "content": _untrusted([{}, {}, {}, {}])},
            {"action": "shared_conversations", "kind": "none", "count": 4},
        ),
        (
            {"op": "shared_history", "channel_id": CHANNEL},
            {"ok": True, "state": "ok", "content": _untrusted({"messages": []})},
            {"action": "shared_history", "kind": "channel", "count": 0},
        ),
        (
            {"op": "take_up", "message_id": "x"},
            {"ok": True, "message_id": "x", "text": "\n\nplease look at PR 12\nthanks"},
            {
                "action": "take_up",
                "kind": "message",
                "label": None,
                "preview": "please look at PR 12",
            },
        ),
        (
            {"op": "ack", "message_id": "x", "state": "working"},
            {"ok": True, "state": "working", "message_id": "x"},
            {"action": "ack-working", "kind": "message"},
        ),
        (
            {"op": "ack", "message_id": "x", "state": "done"},
            {"ok": True, "state": "done", "message_id": "x"},
            {"action": "ack-done", "kind": "message"},
        ),
    ],
)
def test_every_op_maps_to_its_tokens(
    args: dict[str, Any], details: dict[str, Any], expected: dict[str, Any]
) -> None:
    row = bridge_row(args, details)
    assert row["op"] == args["op"]
    for key, value in expected.items():
        actual = row["target"][key] if key in ("kind", "label") else row[key]
        assert actual == value, (key, row)


def test_unknown_op_keeps_its_word_and_guesses_nothing() -> None:
    row = bridge_row({"op": "teleport", "name": "general", "body": "x"}, {"ok": False, "code": "q"})
    assert row["action"] == "teleport"
    assert row["target"] == {"kind": "none", "label": None}
    assert row["outcome"] is None
    assert row["preview"] is None
    verb, summary, word = render_bridge_row(row, running=False)
    assert (verb, summary, word) == ("Used Team Bridge", "teleport", None)


@pytest.mark.parametrize(
    ("details", "outcome", "code"),
    [
        ({"ok": False, "code": "not_permitted", "error": "no"}, "refused", "not_permitted"),
        ({"ok": False, "code": "history_failed", "error": "down"}, "failed", "history_failed"),
        ({"ok": False, "code": "post_failed", "error": "lost"}, "unconfirmed", "post_failed"),
        (
            {"ok": False, "code": "send_unconfirmed", "error": "?"},
            "unconfirmed",
            "send_unconfirmed",
        ),
        ({"ok": True, "state": "sent", "destination_type": "peer_inbox"}, "inbox_only", None),
        (_posted(), None, None),
    ],
)
def test_outcomes(details: dict[str, Any], outcome: str | None, code: str | None) -> None:
    row = bridge_row({"op": "channel_post", "channel_id": CHANNEL, "body": "x"}, details)
    assert (row["outcome"], row["code"]) == (outcome, code)


def test_dm_to_an_agent_not_yet_accepted_is_awaiting_acceptance() -> None:
    details = _posted(destination_type="conversation_dm", recipient_state="invited")
    assert bridge_row({"op": "dm", "body": "x"}, details)["outcome"] == "awaiting_acceptance"
    active = _posted(destination_type="conversation_dm", recipient_state="active")
    assert bridge_row({"op": "dm", "body": "x"}, active)["outcome"] is None


def test_running_call_has_no_outcome_and_no_count() -> None:
    row = bridge_row({"op": "channel_history", "channel_id": CHANNEL})
    assert row["outcome"] is None and row["count"] is None


def test_conversation_label_detail_wins_over_every_other_source() -> None:
    details = _posted(conversation_label="ops-war-room")
    row = bridge_row({"op": "channel_post", "name": "general", "body": "x"}, details)
    assert row["target"]["label"] == "ops-war-room"


@pytest.mark.parametrize(
    "args",
    [
        {"op": "channel_post", "channel_id": CHANNEL, "name": CHANNEL, "body": "x"},
        {"op": "send", "to": {"name": "acc_123"}, "text": "x"},
        {"op": "send", "to": {"name": "EP_abc"}, "text": "x"},
        {"op": "dm", "name": "bcmsg_01", "body": "x"},
    ],
)
def test_an_id_never_becomes_the_label(args: dict[str, Any]) -> None:
    row = bridge_row(args, {"ok": True, "content": f"direct message to acc_9 in {CHANNEL}: m"})
    assert row["target"]["label"] is None
    line = bridge_row_line(row, running=False)
    for prefix in ("bcv_", "bcmsg_", "acc_", "ep_", "EP_"):
        assert prefix not in line


def test_control_characters_and_escapes_are_stripped_from_label_and_preview() -> None:
    args = {
        "op": "channel_post",
        "name": "gen\x1b[31meral\x07",
        "body": "\n   \n  hello\x1b]0;title\x07 \u0086world\x00\n second line",
    }
    row = bridge_row(args, _posted())
    assert row["target"]["label"] == "general"
    assert row["preview"] == "hello world"


def test_label_and_preview_whitespace_is_flattened() -> None:
    row = bridge_row({"op": "send", "to": {"name": " Ana \n  Ruiz\t"}, "text": "a\tb  c"})
    assert row["target"]["label"] == "Ana Ruiz"
    assert row["preview"] == "a b c"


# -- English copy ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "details", "line"),
    [
        (
            {"op": "channel_post", "name": "general", "body": "Ship it"},
            _posted(),
            "Posted in #general · Ship it",
        ),
        (
            {"op": "channel_post", "name": "general", "body": "ok", "thread_root_id": "bcmsg_1"},
            _posted(),
            "Posted in #general thread · ok",
        ),
        ({"op": "channel_post", "name": "#ops", "body": "x"}, _posted(), "Posted in #ops · x"),
        (
            {"op": "dm", "body": "review?"},
            {"ok": True, "content": f"direct message to Ana Ruiz in {CHANNEL}: m"},
            "Messaged Ana Ruiz · review?",
        ),
        (
            {"op": "send", "to": "acc_1", "text": "ping"},
            {"ok": True, "destination_type": "peer_inbox"},
            "Sent to a teammate · ping · not posted to a conversation",
        ),
        (
            {"op": "channels"},
            {"ok": True, "content": json.dumps([{}, {}, {}])},
            "Listed conversations (3)",
        ),
        (
            {"op": "channel_create", "name": "launch"},
            {"ok": True, "content": json.dumps({"name": "launch"})},
            "Created #launch",
        ),
        (
            {"op": "channel_history", "channel_id": CHANNEL},
            {"ok": True, "content": _untrusted({"messages": [{}]})},
            "Read a conversation (1 message)",
        ),
        (
            {"op": "channel_history", "name": "general"},
            {"ok": True, "content": _untrusted({"messages": []})},
            "Read #general (no messages)",
        ),
        (
            {"op": "channel_thread", "name": "general", "thread_root_id": "bcmsg_1"},
            {"ok": True, "content": _untrusted({"root": {}, "replies": [{}]})},
            "Read thread in #general thread (2 messages)",
        ),
        (
            {"op": "react", "name": "general", "emoji": "👍"},
            {"ok": True},
            "Reacted in #general · 👍",
        ),
        (
            {"op": "react", "name": "general", "emoji": "👍", "remove": True},
            {"ok": True},
            "Removed reaction in #general · 👍",
        ),
        (
            {"op": "directory"},
            {"ok": True, "content": _untrusted({"count": 1})},
            "Looked up the team directory (1 member)",
        ),
        ({"op": "shared_conversations"}, {"ok": True}, "Listed shared conversations"),
        (
            {"op": "shared_history", "name": "board"},
            {"ok": True, "content": _untrusted({"messages": [{}, {}]})},
            "Read shared #board (2 messages)",
        ),
        (
            {"op": "take_up", "message_id": "m"},
            {"ok": True, "text": "look at PR 12"},
            "Took up a message · look at PR 12",
        ),
        ({"op": "ack", "state": "working"}, {"ok": True}, "Claimed a message"),
        ({"op": "ack", "state": "done"}, {"ok": True}, "Released a message"),
        (
            {"op": "channel_post", "name": "general", "body": "x"},
            {"ok": False, "code": "not_permitted", "error": "no"},
            "Couldn't post in #general · x · refused · not_permitted",
        ),
        (
            {"op": "channel_history", "name": "general"},
            {"ok": False, "code": "history_failed", "error": "down"},
            "Couldn't read #general · failed · history_failed",
        ),
        (
            {"op": "channel_post", "name": "general", "body": "x"},
            {"ok": False, "code": "post_failed", "error": "lost"},
            "Posted in #general · x · delivery unconfirmed",
        ),
        (
            {"op": "dm", "body": "hi"},
            _posted(recipient_state="invited", content="direct message to Bo in bcv_x: m"),
            "Messaged Bo · hi · awaiting acceptance",
        ),
    ],
)
def test_settled_english_line(args: dict[str, Any], details: dict[str, Any], line: str) -> None:
    in_thread = bool(args.get("thread_root_id"))
    assert bridge_row_line(bridge_row(args, details), running=False, in_thread=in_thread) == line


def test_running_verbs_and_no_outcome_word() -> None:
    row = bridge_row({"op": "channel_post", "name": "general", "body": "x"})
    assert render_bridge_row(row, running=True) == ("Posting in", "#general · x", None)
    row = bridge_row({"op": "directory"})
    assert render_bridge_row(row, running=True) == ("Looking up the team directory", "", None)


def test_host_failed_verdict_without_details_uses_the_failed_verb() -> None:
    row = bridge_row({"op": "dm", "body": "x"}, {})
    assert render_bridge_row(row, running=False, failed=True)[0] == "Couldn't message"


def test_expanded_parts_order_and_internals() -> None:
    args = {"op": "channel_post", "channel_id": CHANNEL, "name": "general", "body": "**hi**"}
    view = bridge_expanded(bridge_row(args, _posted()), args, _posted())
    assert view["headline"] == "Posted in #general"
    assert view["body"] == "**hi**"
    assert view["outcome"] == "Posted at seq 42"
    fields = [name for name, _value in view["internals"]]
    assert fields[:3] == ["conversation_id", "message_id", "seq"]
    assert "content" in fields
    # Identifiers live ONLY in the internals.
    assert "bcv_" not in view["headline"] + view["outcome"]


def test_expanded_dm_headline_and_inbox_and_refusal_sentences() -> None:
    dm = {"ok": True, "state": "sent", "content": f"direct message to Ana in {CHANNEL}: m"}
    args = {"op": "dm", "body": "x"}
    assert bridge_expanded(bridge_row(args, dm), args, dm)["headline"] == "Direct message to Ana"
    inbox = {"ok": True, "state": "sent", "destination_type": "peer_inbox"}
    args = {"op": "send", "to": "acc_1", "text": "x"}
    assert bridge_expanded(bridge_row(args, inbox), args, inbox)["outcome"].startswith(
        "Accepted into their inbox — not posted to a conversation"
    )
    refused = {"ok": False, "code": "not_permitted", "error": "writes are off"}
    args = {"op": "channel_post", "body": "x"}
    assert (
        bridge_expanded(bridge_row(args, refused), args, refused)["outcome"]
        == "Refused: writes are off (not_permitted)"
    )
