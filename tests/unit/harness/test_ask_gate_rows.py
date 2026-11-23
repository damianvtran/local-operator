"""The ask gate's shared predicates (``harness.rows``): one decision per shape.

Design docs/design/ask-gate.md §3's predicate table. Every human surface reads
the divert marker through these functions and none re-derives it, so each
predicate is pinned against ALL the shapes it must accept — a live details
mapping, a stored ``{type, payload}`` row, a rendered ``Message`` — and against
the near-misses it must refuse (a normal ask row, the patience pair, the wake
marker), so an over-broad predicate fails here rather than silently blanking a
transcript.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from local_operator.harness.rows import (
    ask_gate_diverted_call_ids,
    is_ask_gate_divert_details,
    is_ask_gate_divert_message,
    is_ask_gate_divert_row,
    is_settle_only_ask,
    queued_ask_engine_live,
    without_ask_gate_divert,
)
from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult

MARKER = {"ask_gate": {"hidden": True, "verdict": "clear", "reason": "plainly best"}}


def _result_message(*, marker: bool, text: str = "receipt") -> Message:
    return Message.tool_result(
        ToolResult(
            tool_call_id="call-ask",
            tool_name="ask",
            content=[TextContent(text=text)],
            details=dict(MARKER) if marker else {},
        )
    )


# --- is_ask_gate_divert_details ---------------------------------------------


def test_details_marker_shapes() -> None:
    assert is_ask_gate_divert_details(MARKER) is True
    assert is_ask_gate_divert_details(dict(MARKER)) is True
    # Negatives: wrong shape, absent, a non-mapping, and hidden not True.
    assert is_ask_gate_divert_details({}) is False
    assert is_ask_gate_divert_details(None) is False
    assert is_ask_gate_divert_details("ask_gate") is False
    assert is_ask_gate_divert_details({"ask_gate": {"hidden": False}}) is False
    assert is_ask_gate_divert_details({"ask_gate": "nope"}) is False
    assert is_ask_gate_divert_details({"other": {"hidden": True}}) is False


# --- is_ask_gate_divert_row --------------------------------------------------


def test_stored_row_predicate() -> None:
    row = {
        "type": "message",
        "payload": {"role": "tool", "tool_name": "ask", "provider_payload": {"details": MARKER}},
    }
    assert is_ask_gate_divert_row(row) is True

    # Negatives: a normal ask result row, a patience row, the wake marker, and
    # the malformed shapes a caller may hand a duck-typed predicate.
    assert (
        is_ask_gate_divert_row({"type": "message", "payload": {"role": "tool", "tool_name": "ask"}})
        is False
    )
    assert (
        is_ask_gate_divert_row(
            {"type": "message", "payload": {"role": "tool", "tool_name": "patience"}}
        )
        is False
    )
    assert (
        is_ask_gate_divert_row(
            {
                "type": "message",
                "payload": {"custom_type": "wake_prompt", "details": {"hidden": True}},
            }
        )
        is False
    )
    assert is_ask_gate_divert_row("row") is False
    assert is_ask_gate_divert_row({"payload": None}) is False


# --- is_ask_gate_divert_message ---------------------------------------------


def test_rendered_message_predicate() -> None:
    assert is_ask_gate_divert_message(_result_message(marker=True)) is True
    assert is_ask_gate_divert_message(_result_message(marker=False)) is False
    assert is_ask_gate_divert_message(Message.user("hello")) is False
    assert is_ask_gate_divert_message(None) is False


# --- ask_gate_diverted_call_ids ----------------------------------------------


def test_call_id_set_reads_the_marked_results() -> None:
    messages: list[Any] = [
        Message.user("go"),
        _result_message(marker=True),
        Message.tool_result(
            ToolResult(tool_call_id="call-read", tool_name="read", content=[TextContent(text="f")])
        ),
    ]
    assert ask_gate_diverted_call_ids(messages) == {"call-ask"}
    assert ask_gate_diverted_call_ids([]) == set()


# --- without_ask_gate_divert -------------------------------------------------


def test_transform_drops_the_result_and_strips_the_call() -> None:
    ids = {"call-ask"}
    assert without_ask_gate_divert(_result_message(marker=True), ids) is None

    # An assistant row with no prose and only the diverted call goes entirely.
    bare_call = Message(
        role="assistant",
        content=[],
        tool_calls=[ToolCall(id="call-ask", name="ask", arguments={})],
    )
    assert without_ask_gate_divert(bare_call, ids) is None

    # A mixed row keeps its prose and its other calls.
    mixed = Message(
        role="assistant",
        content=[TextContent(text="let me check")],
        tool_calls=[
            ToolCall(id="call-ask", name="ask", arguments={}),
            ToolCall(id="call-read", name="read", arguments={}),
        ],
    )
    stripped = without_ask_gate_divert(mixed, ids)
    assert stripped is not None
    assert stripped.text == "let me check"
    assert [call.id for call in stripped.tool_calls] == ["call-read"]

    # Passthrough when nothing matches.
    untouched = Message.user("hi")
    assert without_ask_gate_divert(untouched, ids) is untouched
    assert without_ask_gate_divert(untouched, set()) is untouched


# --- is_settle_only_ask ------------------------------------------------------


def test_settle_only_predicate_table() -> None:
    assert is_settle_only_ask("ask", queued_engine=True) is True
    assert is_settle_only_ask("ask", queued_engine=False) is False
    assert is_settle_only_ask("bash", queued_engine=True) is False
    assert is_settle_only_ask(None, queued_engine=True) is False
    assert is_settle_only_ask("ASK", queued_engine=True) is False


# --- queued_ask_engine_live --------------------------------------------------


def test_mode_probe_owner_and_viewer() -> None:
    # The owner: ``ask_queue()`` is the one authority (None on the blocking arm).
    assert queued_ask_engine_live(SimpleNamespace(ask_queue=lambda: object())) is True
    assert queued_ask_engine_live(SimpleNamespace(ask_queue=lambda: None)) is False
    assert queued_ask_engine_live(SimpleNamespace(ask_queue=object())) is False  # not callable

    def _boom() -> object:
        raise RuntimeError("no queue")

    assert queued_ask_engine_live(SimpleNamespace(ask_queue=_boom)) is False

    # The viewer: the presence of the ``asks`` OR ``asks_open`` wire field on
    # the frontend state (the WIRE FIX added the tally, which a live-but-empty
    # queue publishes with the rows absent).
    assert (
        queued_ask_engine_live(SimpleNamespace(frontend_state=SimpleNamespace(asks=[{"id": "a1"}])))
        is True
    )
    assert (
        queued_ask_engine_live(SimpleNamespace(frontend_state=SimpleNamespace(asks=None))) is False
    )
    # Live-but-empty: rows absent, the tally present — engine is live.
    assert (
        queued_ask_engine_live(
            SimpleNamespace(frontend_state=SimpleNamespace(asks=None, asks_open=0))
        )
        is True
    )
    # Both absent: a runtime that cannot say — today's blocking-arm answer.
    assert (
        queued_ask_engine_live(
            SimpleNamespace(frontend_state=SimpleNamespace(asks=None, asks_open=None))
        )
        is False
    )
    # A facade that cannot say, and a bare object: False, never a raise.
    assert queued_ask_engine_live(SimpleNamespace()) is False
    assert queued_ask_engine_live(object()) is False


def test_a_dismissal_receipt_names_who_which_surface_and_why():
    from local_operator.harness.rows import ask_response_notice

    text, kind = ask_response_notice(
        {
            "status": "dismissed",
            "ask_id": "a-1",
            "reason": "answered_in_chat",
            "by": {"surface": "remote"},
        }
    )
    assert text == "a-1 — Dismissed by you (phone), settled in chat"
    assert kind == "info"
    assert "Answered" not in text
    bare, _ = ask_response_notice({"status": "dismissed", "ask_id": "a-1"})
    assert bare == "a-1 — Dismissed by you, no longer needed"


# -- naming an ask in words (Ben, 2026-11-23) --------------------------------


def test_a_receipt_names_the_ask_in_words_and_puts_the_id_first():
    """An id alone is hard to match to an ask in the UIs, so every receipt names
    it in the ask's own few words — and the id still leads."""
    from local_operator.harness.rows import ask_response_notice, ask_timeout_notice

    questions = [{"question": "Settings layout: one column or two?"}]
    answered, kind = ask_response_notice(
        {"status": "answered", "ask_id": "a-745e", "questions": questions}
    )
    assert answered == 'a-745e · "Settings layout: one column or two?" — Answered, delivering'
    assert kind == "info"
    timed_out, severity = ask_timeout_notice(
        {"ask_id": "a-745e", "questions": questions, "waited_s": 2520}
    )
    assert timed_out == (
        'a-745e · "Settings layout: one column or two?" — Timed out after 42m, '
        "the agent goes ahead without your answer"
    )
    assert severity == "warning"
    # A row with nothing to name (an older timeout row) keeps the id, which is
    # still the addressable token.
    assert ask_timeout_notice({"ask_id": "a-745e"})[0] == (
        "a-745e — Timed out after a while, the agent goes ahead without your answer"
    )


def test_a_secret_ask_is_named_by_a_label_and_never_by_its_question():
    """Only the card that owns a secret ask may show its words: a receipt, a
    phone frame and another session's relay all read this name."""
    from local_operator.harness.rows import ask_name, ask_response_notice

    text = ask_response_notice(
        {
            "status": "answered",
            "ask_id": "a-9",
            "questions": [
                {"question": "Paste the production API key", "secret": True, "key": "PROD_KEY"}
            ],
        }
    )[0]
    assert text == 'a-9 · "secret question" — Answered, delivering'
    assert "API" not in text and "PROD_KEY" not in text
    # ONE secret question makes the WHOLE name generic: naming the first
    # question of a mixed ask would leak the prompt beside it.
    mixed = ask_name(
        {
            "questions": [
                {"question": "Which region?"},
                {"question": "Paste the key", "secret": True},
            ]
        }
    )
    assert mixed == "secret question +1"


def test_ask_name_cuts_at_a_word_boundary_and_counts_the_rest():
    from local_operator.harness.rows import ask_name

    long_question = "Which database should the migration target, staging or prod?"
    name = ask_name({"questions": [{"question": long_question}]})
    assert name is not None
    assert name.endswith("…")
    assert len(name.rstrip("…")) <= 40
    # The cut keeps whole words: the character before the ellipsis is a letter.
    assert name[-2].isalpha()
    assert long_question.startswith(name[:-1])
    assert ask_name({"questions": [{"question": "A"}, {"question": "B"}, {"question": "C"}]}) == (
        "A +2"
    )
    # Whitespace collapses, and nothing to name omits the name entirely.
    assert ask_name({"questions": [{"question": "  one   line\n\ttwo  "}]}) == "one line two"
    assert ask_name({"questions": []}) is None
    assert ask_name({}) is None
    assert ask_name({"questions": [{"question": "   "}]}) is None


def test_the_parts_are_the_one_source_the_sentence_is_composed_from():
    """A surface that lays the id out separately reads the same parts, so it
    cannot word the state differently from the phone it sits beside."""
    from local_operator.harness.rows import (
        ask_response_notice,
        ask_response_parts,
        ask_timeout_parts,
    )

    details = {
        "status": "late",
        "ask_id": "a-745e",
        "questions": [{"question": "Settings layout?"}],
        "waited_s": 60,
    }
    ask_id, name, state, ink = ask_response_parts(details)
    assert (ask_id, name, state, ink) == (
        "a-745e",
        "Settings layout?",
        "Answered late, the agent was told",
        "warning",
    )
    assert ask_response_notice(details) == (f'a-745e · "Settings layout?" — {state}', ink)
    timeout_id, _, timeout_state, timeout_ink = ask_timeout_parts(details)
    assert (timeout_id, timeout_ink) == ("a-745e", "warning")
    assert timeout_state.startswith("Timed out after ")
