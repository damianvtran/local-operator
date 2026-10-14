"""D2: the decision card's title knows who asked.

A Team Bridge carriage is a request from ANOTHER member's agent. The card title
used to say "the agent asked and moved on" for every queued ask, which tells
this user that their OWN agent asked — wrong subject, and the answer travels
back to a different person. These pins hold both branches and the structural
predicate between them.
"""

from __future__ import annotations

from local_operator.tui.app import OperatorApp as App
from local_operator.tui.widgets.ask_queue import (
    ASK_MARKER,
    STATUS_OPEN,
    STATUS_TIMED_OUT,
    AskRow,
)


def _row(questions, status: str = STATUS_OPEN) -> AskRow:
    return AskRow(
        ask_id="a-1",
        status=status,
        created_at=1,
        expires_at=2,
        urgent=False,
        questions=tuple(questions),
    )


def test_an_ordinary_ask_keeps_the_ordinary_copy() -> None:
    row = _row([{"id": "q1", "prompt": "which one?"}])
    assert App._ask_card_title(row) == (
        f"{ASK_MARKER} the agent asked and moved on — your answer still reaches it"
    )


def test_a_bridge_carriage_names_the_other_member_not_this_agents_user() -> None:
    row = _row([{"id": "q1", "question_id": "q-9", "bridge": {"delivery_ref": "dref_1"}}])
    title = App._ask_card_title(row)
    assert "another member's agent asked" in title
    assert "the agent asked and moved on" not in title


def test_the_predicate_is_structural_not_a_label() -> None:
    """Either seat proves the carriage: the locator key alone, or the
    ``question_id`` alone (the field a bridge question carries)."""
    locator_only = _row([{"id": "q1", "bridge": {"delivery_ref": "dref_1"}}])
    question_id_only = _row([{"id": "q1", "question_id": "q-9"}])
    assert App._ask_card_title(locator_only) == App._ask_card_title(question_id_only)
    assert "another member's agent asked" in App._ask_card_title(question_id_only)


def test_the_timeout_copy_changes_subject_too() -> None:
    ordinary = App._ask_card_title(_row([{"id": "q1"}], status=STATUS_TIMED_OUT))
    bridge = App._ask_card_title(
        _row([{"id": "q1", "question_id": "q-9"}], status=STATUS_TIMED_OUT)
    )
    assert "this ask timed out" in ordinary
    assert "this request timed out" in bridge and "the sender moved on" in bridge


def test_a_malformed_question_is_not_a_bridge_carriage() -> None:
    """A row whose questions are not mappings must not raise inside a paint."""
    assert "the agent asked and moved on" in App._ask_card_title(_row([{}, "junk"]))
