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
    outcome_word,
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
            {"op": "channel_post", "channel_id": CHANNEL, "body": "Ship it"},
            _posted(conversation_label="general"),
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
    row = bridge_row(
        {"op": "teleport", "channel_id": CHANNEL, "body": "x"},
        {"ok": False, "code": "q", "conversation_label": "general"},
    )
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
    row = bridge_row({"op": "channel_post", "channel_id": CHANNEL, "body": "x"}, details)
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
        "op": "channel_create",
        "name": "gen\x1b[31meral\x07",
        "body": "\n   \n  hello\x1b]0;title\x07 \u0086world\x00\n second line",
    }
    assert bridge_row(args, {"ok": True})["target"]["label"] == "general"
    row = bridge_row({**args, "op": "channel_post", "channel_id": CHANNEL}, _posted())
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
            {"op": "channel_post", "channel_id": CHANNEL, "body": "Ship it"},
            _posted(conversation_label="general"),
            "Posted in #general · Ship it",
        ),
        (
            {
                "op": "channel_post",
                "channel_id": CHANNEL,
                "body": "ok",
                "thread_root_id": "bcmsg_1",
            },
            _posted(conversation_label="general"),
            "Posted in #general thread · ok",
        ),
        (
            {"op": "channel_post", "channel_id": CHANNEL, "body": "x"},
            _posted(conversation_label="#ops"),
            "Posted in #ops · x",
        ),
        (
            # R4: the executor ignores `name` on a post, so the row does too.
            {"op": "channel_post", "channel_id": CHANNEL, "name": "general", "body": "x"},
            _posted(),
            "Posted in a conversation · x",
        ),
        (
            # R1: `to` on a post is a MENTION, never the room.
            {
                "op": "channel_post",
                "channel_id": CHANNEL,
                "body": "Ship it",
                "to": {"name": "Alice"},
            },
            _posted(),
            "Posted in a conversation · Ship it",
        ),
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
            {"op": "channel_history", "channel_id": CHANNEL},
            {"ok": True, "conversation_label": "general", "content": _untrusted({"messages": []})},
            "Read #general (no messages)",
        ),
        (
            {"op": "channel_thread", "channel_id": CHANNEL, "thread_root_id": "bcmsg_1"},
            {
                "ok": True,
                "conversation_label": "general",
                "content": _untrusted({"root": {}, "replies": [{}]}),
            },
            # The thread suffix is a channel_post reply's alone: the verb
            # already says "thread" here.
            "Read thread in #general (2 messages)",
        ),
        (
            {"op": "react", "channel_id": CHANNEL, "emoji": "👍"},
            {"ok": True, "conversation_label": "general"},
            "Reacted in #general · 👍",
        ),
        (
            {"op": "react", "channel_id": CHANNEL, "emoji": "👍", "remove": True},
            {"ok": True, "conversation_label": "general"},
            "Removed reaction in #general · 👍",
        ),
        (
            {"op": "directory"},
            {"ok": True, "content": _untrusted({"count": 1})},
            "Looked up the team directory (1 member)",
        ),
        ({"op": "shared_conversations"}, {"ok": True}, "Listed shared conversations"),
        (
            {"op": "shared_history", "channel_id": CHANNEL},
            {
                "ok": True,
                "conversation_label": "board",
                "content": _untrusted({"messages": [{}, {}]}),
            },
            "Read shared conversation #board (2 messages)",
        ),
        (
            # R5: the verb composes with the fallback noun too.
            {"op": "shared_history", "channel_id": CHANNEL},
            {"ok": False, "code": "history_failed", "error": "down"},
            "Couldn't read shared conversation a conversation · failed · history_failed",
        ),
        (
            {"op": "take_up", "message_id": "m"},
            {"ok": True, "text": "look at PR 12"},
            "Took up a message · look at PR 12",
        ),
        ({"op": "ack", "state": "working"}, {"ok": True}, "Claimed a message"),
        ({"op": "ack", "state": "done"}, {"ok": True}, "Released a message"),
        (
            # R3: a stateless ack is refused by the executor; its row takes the
            # claim verb's refused form, never the unknown-op copy.
            {"op": "ack", "message_id": "m"},
            {"ok": False, "code": "invalid_request", "error": "ack needs state"},
            "Couldn't claim a message · refused · invalid_request",
        ),
        (
            {"op": "channel_post", "channel_id": CHANNEL, "body": "x"},
            {"ok": False, "code": "not_permitted", "error": "no", "conversation_label": "general"},
            "Couldn't post in #general · x · refused · not_permitted",
        ),
        (
            {"op": "channel_history", "channel_id": CHANNEL},
            {
                "ok": False,
                "code": "history_failed",
                "error": "down",
                "conversation_label": "general",
            },
            "Couldn't read #general · failed · history_failed",
        ),
        (
            {"op": "channel_post", "channel_id": CHANNEL, "body": "x"},
            {"ok": False, "code": "post_failed", "error": "lost", "conversation_label": "general"},
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
    row = bridge_row({"op": "channel_create", "name": "general"})
    assert render_bridge_row(row, running=True) == ("Creating", "#general", None)
    row = bridge_row({"op": "channel_post", "channel_id": CHANNEL, "body": "x"})
    assert render_bridge_row(row, running=True) == ("Posting in", "a conversation · x", None)
    row = bridge_row({"op": "directory"})
    assert render_bridge_row(row, running=True) == ("Looking up the team directory", "", None)


def test_host_failed_verdict_without_details_uses_the_failed_verb() -> None:
    row = bridge_row({"op": "dm", "body": "x"}, {})
    assert render_bridge_row(row, running=False, failed=True)[0] == "Couldn't message"


def test_expanded_parts_order_and_internals() -> None:
    args = {"op": "channel_post", "channel_id": CHANNEL, "body": "**hi**"}
    details = _posted(conversation_label="general", execution_confirmed=False)
    view = bridge_expanded(bridge_row(args, details), args, details)
    assert view["headline"] == "Posted in #general"
    # The body stays markdown for the host to render; only the preview is plain.
    assert view["body"] == "**hi**"
    assert view["outcome"] == "Posted at seq 42"
    internals = dict(view["internals"])
    fields = [name for name, _value in view["internals"]]
    assert fields[:3] == ["conversation_id", "message_id", "seq"]
    # D2: `channel_id` equal to `conversation_id` is one destination, listed
    # once; the write receipt's `content` restates the fields above it.
    assert "channel_id" not in fields
    assert "content" not in fields
    # D5: JSON spelling for booleans.
    assert internals["execution_confirmed"] == "false"
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


def test_details_keep_a_distinct_channel_id_and_an_informative_content() -> None:
    args = {"op": "channel_post", "channel_id": "bcv_other", "body": "x"}
    details = _posted(message_id="", content=f"posted to {CHANNEL}: accepted, but no id")
    fields = dict(bridge_expanded(bridge_row(args, details), args, details)["internals"])
    assert fields["channel_id"] == "bcv_other"
    assert fields["content"].endswith("accepted, but no id")


@pytest.mark.parametrize(
    ("args", "state", "sentence"),
    [
        ({"op": "react", "channel_id": CHANNEL, "emoji": "👍"}, "reacted", "Reacted."),
        (
            {"op": "react", "channel_id": CHANNEL, "emoji": "👍", "remove": True},
            "removed",
            "Removed reaction.",
        ),
    ],
)
def test_reaction_expansion_states_its_outcome(
    args: dict[str, Any], state: str, sentence: str
) -> None:
    details = {"ok": True, "state": state}
    assert bridge_expanded(bridge_row(args, details), args, details)["outcome"] == sentence


@pytest.mark.parametrize(
    "name",
    [
        "#eng\u202eDROW\u202c#ops",
        "#eng\u2066DROW\u2069#ops",
        "#eng\u200eDROW\u200f\u061c#ops",
        "#eng\u200bDROW\u2060\ufeff#ops",
    ],
)
def test_bidi_and_format_characters_are_stripped(name: str) -> None:
    row = bridge_row({"op": "channel_create", "name": name}, {"ok": True})
    assert row["target"]["label"] == "#engDROW#ops"
    row = bridge_row({"op": "send", "to": {"name": "Ana"}, "text": f"hi {name}"})
    assert row["preview"] == "hi #engDROW#ops"


def test_joiners_that_shape_text_and_emoji_survive() -> None:
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    row = bridge_row({"op": "react", "channel_id": CHANNEL, "emoji": family}, {"ok": True})
    assert row["emoji"] == family
    hindi = "\u0915\u094d\u200d\u0937"
    assert bridge_row({"op": "dm", "to": {"name": hindi}, "body": "x"})["target"]["label"] == hindi


@pytest.mark.parametrize(
    ("body", "preview"),
    [
        (
            "Release **v0.62.0** is cut — notes in the thread.",
            "Release v0.62.0 is cut — notes in the thread.",
        ),
        ("__bold__ and *em* and _em_ and ~~gone~~", "bold and em and em and gone"),
        ("run `make check` now", "run make check now"),
        ("# Heading", "Heading"),
        ("> quoted *text*", "quoted text"),
        ("- item one", "item one"),
        ("3. third", "third"),
        ("see [the PR](https://example.com/pr/1) ![alt](x.png)", "see the PR alt"),
        ("---\n\n```\nreal line", "real line"),
        ("keep snake_case and 2 * 3", "keep snake_case and 2 * 3"),
    ],
)
def test_preview_strips_markdown_markers(body: str, preview: str) -> None:
    row = bridge_row({"op": "channel_post", "channel_id": CHANNEL, "body": body})
    assert row["preview"] == preview


@pytest.mark.parametrize(
    ("outcome", "full", "short"),
    [
        (
            {"ok": True, "destination_type": "peer_inbox"},
            "not posted to a conversation",
            "not posted",
        ),
        (
            {"ok": True, "destination_type": "conversation_dm", "recipient_state": "invited"},
            "awaiting acceptance",
            "awaiting",
        ),
        ({"ok": False, "code": "post_failed"}, "delivery unconfirmed", "unconfirmed"),
        (
            {"ok": False, "code": "not_permitted"},
            "refused · not_permitted",
            "refused · not_permitted",
        ),
    ],
)
def test_short_outcome_words(outcome: dict[str, Any], full: str, short: str) -> None:
    row = bridge_row({"op": "dm", "body": "x"}, outcome)
    assert outcome_word(row) == full
    assert outcome_word(row, short=True) == short


@pytest.mark.parametrize(
    ("details", "sentence"),
    [
        ({"ok": False, "code": "not_permitted", "error": ""}, "Refused (not_permitted)"),
        ({"ok": False, "error": "writes are off"}, "Refused: writes are off"),
        ({"ok": False}, "Refused."),
        ({"ok": False, "code": "history_failed"}, "Failed (history_failed)"),
        ({"ok": False, "code": "x_failed", "error": "down"}, "Failed: down (x_failed)"),
        ({"ok": False, "code": "post_failed"}, "Delivery unconfirmed."),
    ],
)
def test_refusal_sentences_never_paint_empty_slots(details: dict[str, Any], sentence: str) -> None:
    args = {"op": "channel_post", "channel_id": CHANNEL, "body": "x"}
    view = bridge_expanded(bridge_row(args, details), args, details)
    assert view["outcome"] == sentence
    assert "()" not in view["outcome"] and ":  " not in view["outcome"]


def test_failure_with_neither_error_nor_code_reads_couldnt_complete() -> None:
    from local_operator.harness.bridge_copy import _refusal_sentence

    assert _refusal_sentence("", "", failed=True) == "Couldn't complete."


def test_refusal_sentence_scrubs_ids_but_details_keep_them() -> None:
    args = {"op": "channel_post", "channel_id": CHANNEL, "body": "x"}
    details = {
        "ok": False,
        "code": "not_permitted",
        "error": f"conversation {CHANNEL} refused bcmsg_9 from ACC_12",
        "conversation_label": "ops",
    }
    view = bridge_expanded(bridge_row(args, details), args, details)
    assert view["outcome"] == "Refused: conversation #ops refused … from … (not_permitted)"
    unlabelled = {**details, "conversation_label": None}
    view = bridge_expanded(bridge_row(args, unlabelled), args, unlabelled)
    assert "conversation a conversation" in view["outcome"]
    assert dict(view["internals"])["channel_id"] == CHANNEL


def test_unlabelled_dm_headline_names_a_teammate() -> None:
    args = {"op": "dm", "body": "x"}
    view = bridge_expanded(bridge_row(args, _posted()), args, _posted())
    assert view["headline"] == "Direct message to a teammate"


def test_the_markdown_strip_lives_in_the_shared_token() -> None:
    # The relay passthrough forwards `bridge_row`'s preview verbatim, so the
    # strip must be in the token, not in a host renderer.
    row = bridge_row({"op": "dm", "body": "**bold** and `code`"})
    assert row["preview"] == "bold and code"
