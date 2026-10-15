"""H5a — bridge requests are human-required asks (contract §10.3, #125).

The invariant under test: a bridge ask ALWAYS lands in the human queue —
the gate's clearance fork is short-circuited for the class, the queue's own
enqueue path is the one used, and the carriage that the decide receipt binds
to (``question_id`` == the request's message id) survives the shape
projection and the fold.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from local_operator.asks import policy
from local_operator.asks import store as asks_store
from local_operator.asks.queue import AskQueue, _question_shape
from local_operator.session.session import Session, _question_marked_human_required
from local_operator.team_bridge.asks_bridge import (
    BRIDGE_ASK_OPTIONS,
    EXCERPT_MAX_CHARS,
    GENERIC_ASK_SENDER,
    bridge_ask_ref,
    bridge_pending_count,
    bridge_request_question,
    enqueue_bridge_request_ask,
)
from tests.unit.asks.test_queue import BASE, FakeSession

SENDER = "Priya Raman (Acme Research)"


def test_the_question_carries_the_human_required_carriage() -> None:
    question = bridge_request_question(
        message_id="bmsg_abc",
        sender=SENDER,
        excerpt="please approve before 1700",
    )
    assert question["human_required"] is True
    assert question["question_id"] == "bmsg_abc"
    assert [option["label"] for option in question["options"]] == ["Approve", "Deny"]
    assert question["options"] == [dict(opt) for opt in BRIDGE_ASK_OPTIONS]
    # D3: WHO is asking and WHAT they are asking for, on ONE line — that is the
    # line the decision card's bound leaves standing — and nothing else: no
    # thread locator, no §11.1 clause (the clause rides the model's carriers).
    assert question["question"] == "Priya Raman (Acme Research): please approve before 1700"
    assert "\n" not in question["question"]
    assert question["secret"] is False and question["multi"] is False


def test_a_carriage_with_no_validated_name_degrades_to_the_generic_line() -> None:
    """D3: an account id identifies nobody the deciding member can recognise.

    The issuer composes the sender line from the VALIDATED display name, and
    sends "" when there is none (``session._authorize_bridge_envelope``), so the
    human's line names a generic other-organization member instead of a raw
    ``acc_…`` the reader cannot place.
    """
    question = bridge_request_question(
        message_id="bmsg_abc", sender="", excerpt="please approve before 1700"
    )
    assert question["question"] == (f"{GENERIC_ASK_SENDER}: please approve before 1700")
    assert "acc_" not in question["question"]


def test_a_missing_excerpt_is_stated_on_the_question_line() -> None:
    """D9 (design round 3, PR #2378): the `who`-alone shape is UNREACHABLE.

    An empty excerpt is a real carriage (the byte-fitter trims it under payload
    pressure, a screened body can simply be empty, and the render-time lookup
    degrades to "" by design), and a decision card asking a member to approve
    nothing — with no sign that the request text is unavailable here — is the
    same failure class D3 was about (*who* AND *what*). Both carriages state the
    absence, in the feature's own words: 51 cells for the named one, 59 for the
    unnamed, each one uncut row at the card's question width.
    """
    named = bridge_request_question(
        message_id="bmsg_abc", sender="Priya Raman (Acme Research)", excerpt=""
    )
    assert named["question"] == "Priya Raman (Acme Research): (no preview available)"
    unnamed = bridge_request_question(message_id="bmsg_abc", sender="", excerpt="")
    assert unnamed["question"] == "another member of your organization: (no preview available)"


def test_the_excerpt_is_bounded_and_flattened_client_side() -> None:
    question = bridge_request_question(
        message_id="bmsg_abc",
        sender=SENDER,
        excerpt=("line one\nline two " * 200),
    )
    text = question["question"]
    # D3: the question is ONE line, so a newline in the excerpt cannot open a
    # line of its own — it is flattened into the question the human reads.
    assert text.count("\n") == 0
    assert text.startswith("Priya Raman (Acme Research): line one line two")
    assert len(text) <= len(f"{SENDER}: ") + EXCERPT_MAX_CHARS


def test_an_ask_without_the_message_id_is_refused_at_build_time() -> None:
    """An empty question_id could never mint a valid decide receipt — the
    server binds ``question_id == :messageid`` — so it fails here, not at
    answer time."""
    with pytest.raises(ValueError):
        bridge_request_question(message_id="  ", sender=SENDER, excerpt="")


def test_the_shape_projection_keeps_the_carriage() -> None:
    """``_question_shape`` is a whitelist; the two H5a keys are ON it, or the
    gate/fold could never see them."""
    shape = _question_shape(
        bridge_request_question(message_id="bmsg_abc", sender=SENDER, excerpt="")
    )
    assert shape["human_required"] is True
    assert shape["question_id"] == "bmsg_abc"
    # The unchanged half: a generic ask carries the neutral defaults.
    generic = _question_shape({"id": "q0", "question": "which one?"})
    assert generic["human_required"] is False and generic["question_id"] == ""


def test_the_gate_short_circuits_before_the_clearance_fork(monkeypatch) -> None:
    """The load-bearing arm: with the gate ENABLED and a would-be clearance
    verdict available, a human_required question must return None (the
    UNCHANGED enqueue path) without the fork ever being consulted."""
    calls: list[str] = []

    monkeypatch.setattr(policy, "gate_enabled", lambda: True)

    import local_operator.session.clearance as clearance_mod

    def _boom(*args, **kwargs):  # pragma: no cover — the assert IS the point
        calls.append("fingerprint")
        raise AssertionError("clearance must never be consulted for a human_required ask")

    monkeypatch.setattr(clearance_mod, "fingerprint", _boom)

    stub = SimpleNamespace(
        ask_queue=lambda: object(),
        _ask_gate_diverts=set(),
    )
    import asyncio

    outcome = asyncio.run(
        # ``stub`` is a SimpleNamespace standing in for the session, and the
        # door is a static method read through the class: the cast is the
        # boundary that keeps the fixture honest without weakening the door's
        # own signature.
        Session._gate_ask(cast(Session, stub), [SimpleNamespace(secret=False, human_required=True)])
    )
    assert outcome is None
    assert calls == []

    # And the mark reads off the mapping shape too (the queue's storage form).
    assert _question_marked_human_required({"human_required": True}) is True
    assert _question_marked_human_required(SimpleNamespace(human_required=True)) is True
    assert _question_marked_human_required({"id": "q0"}) is False


def test_enqueue_rides_the_real_queue_and_the_carriage_survives_the_fold(tmp_path) -> None:
    queue = AskQueue(FakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE)
    outcome = enqueue_bridge_request_ask(
        queue,
        message_id="bmsg_abc",
        sender=SENDER,
        excerpt="please approve before 1700",
    )
    assert outcome["ok"] is True
    assert outcome["details"]["question_ids"] == ["bridge-bmsg_abc"]

    records = queue.records()
    assert bridge_pending_count(records) == 1
    record = records[0]
    assert record["status"] == asks_store.STATUS_OPEN
    carried = record["questions"][0]
    assert carried["human_required"] is True
    assert carried["question_id"] == "bmsg_abc"

    # A non-bridge ask never counts toward the badge.
    queue.enqueue([{"id": "q0", "question": "generic?"}], None)
    assert bridge_pending_count(queue.records()) == 1


def test_the_decide_receipt_pairs_the_minted_ask_instance_with_the_message() -> None:
    ref = bridge_ask_ref(ask_id="a-3f9c", question_id="bmsg_abc")
    assert ref == {"ask_id": "a-3f9c", "question_id": "bmsg_abc"}


def test_counting_ignores_settled_asks() -> None:
    open_row = {
        "status": "open",
        "questions": [{"question_id": "bmsg_abc", "human_required": True}],
    }
    answered = {"status": "answered", "questions": [{"question_id": "bmsg_def"}]}
    generic = {"status": "open", "questions": [{"id": "q0", "question": "generic"}]}
    assert bridge_pending_count([open_row, answered, generic]) == 1
