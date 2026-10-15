"""D2: the decision card's title knows who asked.

A Team Bridge carriage is a request from ANOTHER member's agent. The card title
used to say "the agent asked and moved on" for every queued ask, which tells
this user that their OWN agent asked — wrong subject, and the answer travels
back to a different person. These pins hold both branches and the structural
predicate between them.

ROUND 13 ADDS THE MOUNT ITSELF. The title is only reachable if a card mounts at
all, and on the round-12 head a REAL bridge ask could not: the ask card validated
its row's questions with ``AskQuestion`` (``extra="forbid"``), while the queue's
projection (``asks/queue._question_shape``) stores the carriage's seat —
``question_id``, ``bridge``, ``human_required``, each option's ``id`` — and the
builder stamped ``persist`` without ``secret``. The row was refused, ``_mount_ask_card``
logged "carries questions this surface cannot draw", and the one ask kind that
cannot be answered on any other surface had no human card anywhere. The acceptance
below mounts from a row built end-to-end by PRODUCTION (builder -> queue
projection -> row reader) and asserts the card and its D2 title.

ROUND 15 (D3) ADDS THE QUESTION LINE. The card bounds the question to one row
(`MIN_QUESTION_ROWS`), so the ONE line that survives is the line the human
answers from: it must carry who is asking and what they are deciding on. The
§11.1 clause no longer leads it — the clause rides the carriers a MODEL reads
(see ``tests/unit/asks/test_render.py`` for the timeout listing's own pin).
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.asks.queue import _question_shape
from local_operator.harness.types import AskQuestion, AskRowQuestion
from local_operator.team_bridge.asks_bridge import bridge_request_question
from local_operator.tui import ask_open_policy
from local_operator.tui.app import OperatorApp as App
from local_operator.tui.widgets.ask_queue import (
    ASK_MARKER,
    STATUS_OPEN,
    STATUS_TIMED_OUT,
    AskRow,
    ask_rows,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


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


# ---------------------------------------------------------------------------
# The round-13 acceptance: a REAL carriage mounts the card
# ---------------------------------------------------------------------------


def _production_question() -> dict[str, Any]:
    """EXACTLY the question ``asks_bridge`` stamps, via the production builder.

    ``sender`` is the display identity the ISSUER composes from the validated
    display context (``<name> (<organization>)``, D3), not a bare name and not
    an account id.
    """
    return bridge_request_question(
        message_id="bmsg_2378",
        sender="Priya Raman (Acme Research)",
        excerpt="could you approve the Q3 spend for the enrichment pod?",
        scope={"delivery_ref": "dref_1"},
    )


def _stored_record(question: dict[str, Any], status: str = STATUS_OPEN) -> dict[str, Any]:
    """The durable ask record, through the queue's OWN projection.

    ``_question_shape`` is the whitelist the log stores and every surface folds
    back, so this is the row a live session carries: nothing here strips a key
    and no seat is synthesised.
    """
    return {
        "ask_id": "a-2378",
        "created_at": 1_700_000_000_000,
        "expires_at": 1_700_003_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": status,
        "delivered": False,
        "questions": [_question_shape(question)],
    }


@pytest.mark.asyncio
async def test_a_real_bridge_row_mounts_the_card_with_the_cross_member_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance for round 13's drawability blocker.

    End-to-end by production on every hop: ``bridge_request_question`` stamps
    the question, ``asks/queue._question_shape`` folds it into the stored row,
    ``ask_rows`` reads the row, and the app mounts its card from it. The
    assertion that matters is the one that had no reachable path on the
    round-12 head — a card EXISTS for a cross-member ask — plus the D2 title
    that names the other member rather than this user's own agent.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", False)

    record = _stored_record(_production_question())
    stored = record["questions"][0]
    # The seat and the option ids are ON the stored row — the shape the card
    # has to draw, and the ids the decide maps the human's pick back through.
    assert stored["question_id"] == "bmsg_2378"
    assert stored["human_required"] is True
    assert stored["bridge"] == {"delivery_ref": "dref_1"}
    assert [option["id"] for option in stored["options"]] == ["approve", "deny"]
    # ...and the TOOL's model still REFUSES that row: the fix is the split
    # between the tool's question and the row's, not a relaxation of the guard
    # that keeps a model from stamping provenance words on its own ask.
    with pytest.raises(ValueError):
        AskQuestion.model_validate(dict(stored))

    rows = ask_rows([record])
    session = FakeSession()
    app = App(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(3):
            await pilot.pause()
        app._sync_ask_surface(rows)
        await pilot.pause()
        app._expand_asks()
        for _ in range(10):
            await pilot.pause()
        card = app._ask_card
        assert card is not None, "a REAL cross-member ask still mounts no card"
        # D3: the ONE row the card's question bound leaves standing is the human
        # question — who is asking and what they are deciding on — not the
        # §11.1 clause, which led the card before this round and pushed the
        # sender and the request behind the cut.
        assert card.question.question == (
            "Priya Raman (Acme Research): could you approve the Q3 spend for the enrichment pod?"
        )
        assert "untrusted" not in card.question.question
        # ...and it survives the bound UNCUT: the card's own layout draws it as
        # one row at 100x30 (83 cells against the question width), where the
        # clause-led text pushed the sender and the request behind the ``…``.
        layout = card._layout()
        assert layout.question == (
            "Priya Raman (Acme Research): could you approve the Q3 spend for the enrichment pod?",
        ), layout.question
        assert card._title == (
            f"{ASK_MARKER} another member's agent asked — your answer goes back to them"
        )
        # The seat rode THROUGH the card's own validation rather than being
        # stripped on the way in, and the labels are what the human picks
        # between (the answer travels by label; the row's ids are what the door
        # maps it back to).
        assert isinstance(card.question, AskRowQuestion)
        assert card.question.question_id == "bmsg_2378"
        dumped = card.question.model_dump()
        assert [option["id"] for option in dumped["options"]] == ["approve", "deny"]
        assert [option["label"] for option in dumped["options"]] == ["Approve", "Deny"]


@pytest.mark.asyncio
async def test_a_carriage_with_no_validated_name_mounts_the_generic_asker_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3: with no validated member name the card still names an asker.

    A no-meta carriage is a real shape (``screening`` tolerates a missing sender
    block as unknown provenance), and the issuer sends "" for it rather than a
    raw account id — which would put this side's own owner, or an ``acc_…`` the
    member cannot place, in front of the person answering. Same production mount
    as the test above; only the question line differs.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", False)

    question = bridge_request_question(
        message_id="bmsg_2378",
        sender="",
        excerpt="could you approve the Q3 spend for the enrichment pod?",
        scope={"delivery_ref": "dref_1"},
    )
    rows = ask_rows([_stored_record(question)])
    session = FakeSession()
    app = App(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(3):
            await pilot.pause()
        app._sync_ask_surface(rows)
        await pilot.pause()
        app._expand_asks()
        for _ in range(10):
            await pilot.pause()
        card = app._ask_card
        assert card is not None, "the degraded carriage still mounts its card"
        assert card.question.question == (
            "another member of your organization: could you approve the Q3 spend for the "
            "enrichment pod?"
        )
        assert "acc_" not in card.question.question
