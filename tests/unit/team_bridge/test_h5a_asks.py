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
from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.session.session import Session, _question_marked_human_required
from local_operator.team_bridge.asks_bridge import (
    BRIDGE_ASK_OPTIONS,
    EXCERPT_MAX_CHARS,
    bridge_ask_ref,
    bridge_pending_count,
    bridge_request_question,
    enqueue_bridge_request_ask,
)
from tests.unit.asks.test_queue import BASE, FakeSession

SENDER = "dana@acme.example"
SUMMARY = "wants your sign-off on the Friday deploy"


def test_the_question_carries_the_human_required_carriage() -> None:
    question = bridge_request_question(
        message_id="bmsg_abc",
        sender=SENDER,
        summary=SUMMARY,
        excerpt="please approve before 1700",
    )
    assert question["human_required"] is True
    assert question["question_id"] == "bmsg_abc"
    assert [option["label"] for option in question["options"]] == ["Approve", "Deny"]
    assert question["options"] == [dict(opt) for opt in BRIDGE_ASK_OPTIONS]
    assert SENDER in question["question"] and SUMMARY in question["question"]
    assert "please approve before 1700" in question["question"]
    assert question["secret"] is False and question["multi"] is False


def test_the_excerpt_is_bounded_and_flattened_client_side() -> None:
    question = bridge_request_question(
        message_id="bmsg_abc",
        sender=SENDER,
        summary=SUMMARY,
        excerpt=("line one\nline two " * 200),
    )
    text = question["question"]
    # TB-05 (PR #2378): the §11.1 clause now rides the composition as its own
    # line, AHEAD of the sender's excerpt — the excerpt still contributes no
    # newline of its own.
    assert text.count("\n") == 2  # composition separator + the untrusted-clause line
    assert "untrusted" in text
    assert text.index("untrusted") < text.index("line one")
    assert (
        len(text)
        <= len(f"{SENDER} · {SUMMARY}\n") + len(BRIDGE_UNTRUSTED_CLAUSE) + EXCERPT_MAX_CHARS + 2
    )


def test_an_ask_without_the_message_id_is_refused_at_build_time() -> None:
    """An empty question_id could never mint a valid decide receipt — the
    server binds ``question_id == :messageid`` — so it fails here, not at
    answer time."""
    with pytest.raises(ValueError):
        bridge_request_question(message_id="  ", sender=SENDER, summary="s", excerpt="")


def test_the_shape_projection_keeps_the_carriage() -> None:
    """``_question_shape`` is a whitelist; the two H5a keys are ON it, or the
    gate/fold could never see them."""
    shape = _question_shape(
        bridge_request_question(message_id="bmsg_abc", sender=SENDER, summary="s", excerpt="")
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
        summary=SUMMARY,
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
