"""``AskQueue`` end to end against a small session double: caps, the fold, and
the delivery rules (design ``docs/design/ask-nonblocking.md`` §2.2/§2.3).

The session double is deliberately tiny — a transcript id set, a delivery sink
and the two hooks the queue uses — because the queue's contract with the session
is exactly those three things. Everything else it needs it takes from the log.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.asks import policy, render, store
from local_operator.asks.queue import AskQueue

BASE = 1_700_000_000_000


class FakeTranscript:
    def __init__(self) -> None:
        self.ids: set[str] = set()

    def has_entry(self, entry_id: str) -> bool:
        return entry_id in self.ids


class FakeSession:
    """The three things ``AskQueue`` asks of a session, and nothing else.

    ``persist_at_delivery`` models WHERE the durable write happens: the
    default (True) is the collapsed shape most cells want — reconcile hands a
    row and the row is durable immediately after — while the durability cells
    hold the two halves apart (hand-off at reconcile, append later), which is
    the real session's ordering (the append sites run in the delivery turn, not
    in reconcile).
    """

    def __init__(self, *, persist_at_delivery: bool = True) -> None:
        self.transcript = FakeTranscript()
        self.batches: list[list[Any]] = []
        self.reach: list[str] = []
        self.spawned: list[asyncio.Task[Any]] = []
        self.persist_at_delivery = persist_at_delivery
        #: The session's LIVE credential keys. Empty by default, which is also the
        #: honest answer for a double that stores nothing — and the reason the
        #: secret-lost tests have to say what is held rather than assume it.
        self.credentials: list[str] = []

    def credential_names(self) -> list[str]:
        return list(self.credentials)

    def ask_reach(self) -> list[str]:
        return self.reach

    def _spawn_background(self, coro):
        task = asyncio.ensure_future(coro)
        self.spawned.append(task)
        return task

    async def deliver_ask_messages(self, messages) -> None:
        # The HAND-OFF, not the consumption: the real session puts the message
        # on a delivery path here and the row becomes durable when that path
        # reaches its append (``persist_at_delivery`` collapses the two for the
        # cells that do not test the gap itself).
        self.batches.append(list(messages))
        if self.persist_at_delivery:
            for message in messages:
                self.transcript.ids.add(message.id)


def _questions(count: int = 1, *, secret: bool = False, text: str = "Which one?"):
    out = []
    for index in range(count):
        qid = f"key-{index}" if secret else f"q{index}"
        out.append(
            {
                "id": qid,
                "question": f"{text} ({index})" if count > 1 else text,
                "options": [],
                "multi": False,
                "secret": secret,
                "persist": False,
                "recommended": None,
            }
        )
    return out


def _queue(tmp_path: Path, session: FakeSession, now=BASE) -> AskQueue:
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: now)
    return queue


def _run(coro):
    return asyncio.run(coro)


def _dismissed_event(ask_id: str, *, at: int) -> dict[str, Any]:
    """One ``dismissed`` row as the COLD door writes it (§2.2: any process may)."""
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_DISMISSED,
        "ask_id": ask_id,
        "at": at,
        "by": {"surface": "phone"},
    }


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


def test_enqueue_returns_a_receipt_and_writes_the_log(tmp_path: Path):
    session = FakeSession()
    session.reach = ["terminal"]
    queue = _queue(tmp_path, session)
    outcome = queue.enqueue(_questions(2), None)
    assert outcome["ok"] is True
    assert "not consent" in outcome["text"].lower()
    assert "terminal" in outcome["text"]
    assert outcome["details"]["status"] == "queued"
    assert outcome["details"]["timeout_s"] == 3600
    assert outcome["details"]["expires_at"] == BASE + 3600 * 1000
    assert outcome["details"]["urgent"] is False
    assert len(outcome["details"]["question_ids"]) == 2
    events = store.read_events(queue.session_dir)
    assert [event["kind"] for event in events] == [store.EVENT_QUEUED]


def test_an_unreachable_session_gets_the_honest_receipt(tmp_path: Path):
    session = FakeSession()  # no surfaces attached
    queue = _queue(tmp_path, session)
    outcome = queue.enqueue(_questions(), None)
    assert "nobody is attached" in outcome["text"]
    # ``None`` is "no surface could be named" — the honest value, and the one the
    # receipt turns into the unreachable wording.
    assert not outcome["details"]["reach"]


def test_the_receipt_never_claims_the_user_was_told(tmp_path: Path):
    """The measured reach predicate proves a client is CONNECTED. The receipt
    must therefore claim presentation and never notice-delivery (§2.1)."""
    session = FakeSession()
    session.reach = ["desktop"]
    queue = _queue(tmp_path, session)
    text = queue.enqueue(_questions(), None)["text"].lower()
    assert "showing on" in text
    for overclaim in ("has been told", "was notified", "has been notified", "told about"):
        assert overclaim not in text


def test_an_out_of_range_timeout_is_refused_with_the_bounds(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    outcome = queue.enqueue(_questions(), 119)
    assert outcome["ok"] is False
    assert "120" in outcome["error"] and "86400" in outcome["error"]
    assert store.read_events(queue.session_dir) == []


def test_the_duration_string_is_honoured_as_seconds(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    outcome = queue.enqueue(_questions(), "2h")
    assert outcome["details"]["timeout_s"] == 7200
    assert outcome["details"]["expires_at"] == BASE + 7200 * 1000


def test_an_urgent_ask_is_derived_from_the_timeout(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    assert queue.enqueue(_questions(text="a"), 600)["details"]["urgent"] is True
    assert queue.enqueue(_questions(text="b"), 3600)["details"]["urgent"] is False


def test_the_open_cap_refuses_the_ninth_ask_by_name(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    for index in range(policy.OPEN_ASK_CAP):
        assert queue.enqueue(_questions(text=f"question {index}"), None)["ok"] is True
    refused = queue.enqueue(_questions(text="one too many"), None)
    assert refused["ok"] is False
    assert "8" in refused["error"]
    assert "do not re-ask" in refused["error"]


def test_a_duplicate_secret_key_is_refused_while_open(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    assert queue.enqueue(_questions(secret=True), None)["ok"] is True
    refused = queue.enqueue(_questions(secret=True, text="same key again"), None)
    assert refused["ok"] is False
    assert "key-0" in refused["error"]


def test_a_byte_identical_question_text_is_refused_while_open(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    assert queue.enqueue(_questions(text="Shall I proceed?"), None)["ok"] is True
    refused = queue.enqueue(_questions(text="Shall I proceed?"), None)
    assert refused["ok"] is False
    assert "do not re-ask" in refused["error"]


def test_the_same_text_is_allowed_again_once_the_first_ask_is_terminal(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    first = queue.enqueue(_questions(text="Shall I proceed?"), None)
    assert first["ok"] is True
    assert queue.respond(first["details"]["ask_id"], {"q0": ["yes"]}, by="terminal")["ok"]
    assert queue.enqueue(_questions(text="Shall I proceed?"), None)["ok"] is True


def test_the_log_is_the_truth_even_when_the_session_is_gone(tmp_path: Path):
    """Session-agnostic by construction: a second queue over the same directory
    sees the first one's asks, which is what makes a cold answer possible."""
    session = FakeSession()
    first = _queue(tmp_path, session)
    first.enqueue(_questions(text="kept"), None)
    second = AskQueue(FakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE + 1)
    assert [record["status"] for record in second.records()] == [store.STATUS_OPEN]


# ---------------------------------------------------------------------------
# the fold's refusals
# ---------------------------------------------------------------------------


def test_answering_an_unknown_ask_is_its_own_refusal(tmp_path: Path):
    outcome = _queue(tmp_path, FakeSession()).respond("a-nope", {"q0": ["yes"]})
    assert outcome["ok"] is False
    assert "not in this session's queue" in outcome["error"]


def test_a_second_answer_is_refused_as_already_answered(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["one"]}, by="terminal")["ok"] is True
    second = queue.respond(ask_id, {"q0": ["two"]}, by="phone")
    assert second["ok"] is False
    assert "already answered by terminal" in second["error"]


def test_an_expired_ask_says_so_rather_than_being_silently_dropped(tmp_path: Path):
    now = BASE + store.LATE_WINDOW_S * 1000 + 200_000
    queue = _queue(tmp_path, FakeSession(), now=now)
    # Write queued with a base clock, then answer past the window.
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: now
    outcome = queue.respond(ask_id, {"q0": ["late"]})
    assert outcome["ok"] is False
    assert "expired" in outcome["error"]


def test_an_open_ask_on_a_running_session_cannot_be_dismissed(tmp_path: Path):
    """D6 (design §15): the live door refuses an OPEN ask unless the session is
    stopped, and writes nothing. FakeSession has no ``ask_agent_running``, which
    the queue reads as running (fail closed)."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    outcome = queue.dismiss(ask_id)
    assert outcome == {"ok": False, "error": render.DISMISS_AGENT_RUNNING}
    assert "reply in chat" in outcome["error"]
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [store.EVENT_QUEUED]


def test_an_open_ask_on_a_stopped_session_is_dismissed_and_owes_a_response(tmp_path: Path):
    session = FakeSession()
    session.ask_agent_running = lambda: False  # type: ignore[attr-defined]
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.dismiss(ask_id, by="phone", reason="answered_in_chat")["ok"] is True
    events = store.read_events(queue.session_dir)
    assert events[-1]["reason"] == "answered_in_chat"
    assert events[-1]["by"] == {"surface": "phone"}
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    (message,) = session.batches[0]
    assert message.id == store.response_row_id(ask_id)
    assert message.details["status"] == "dismissed"
    assert message.details["reason"] == "answered_in_chat"
    assert message.details["by"] == {"surface": "phone"}
    record = queue._find(ask_id, queue._now())
    assert record is not None
    assert "settled in the conversation" in render.response_text(record)


def test_a_timed_out_ask_is_dismissible_on_a_running_session(tmp_path: Path):
    """Timed-out behaviour is unchanged on every door."""
    queue = _queue(tmp_path, FakeSession(), now=BASE)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 200_000
    assert queue.dismiss(ask_id, by="terminal")["ok"] is True


def test_an_unknown_dismiss_reason_is_refused_and_writes_nothing(tmp_path: Path):
    session = FakeSession()
    session.ask_agent_running = lambda: False  # type: ignore[attr-defined]
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.dismiss(ask_id, reason="because") == {
        "ok": False,
        "error": render.DISMISS_BAD_REASON,
    }
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [store.EVENT_QUEUED]


def test_a_bridge_request_stays_timed_out_only_even_when_stopped(tmp_path: Path):
    session = FakeSession()
    session.ask_agent_running = lambda: False  # type: ignore[attr-defined]
    queue = _queue(tmp_path, session)
    questions = _questions()
    questions[0]["question_id"] = "q-bridge-1"
    ask_id = queue.enqueue(questions, None)["details"]["ask_id"]
    assert queue.dismiss(ask_id) == {"ok": False, "error": render.DISMISS_BRIDGE_OPEN}


def test_a_repeat_dismissal_says_the_ask_is_already_dismissed(tmp_path: Path):
    """The LIVE door's half of the dismissed sentence (U3 = Q3 = MINOR-4).

    A second dismissal used to answer "only a timed-out ask can be dismissed; it
    is still open." on this path too — the row is dismissed, not open, and the
    sentence said the opposite of what the user had just done. The cold door
    (``asks/dismiss.py``) returns the same words, because both read
    ``render.dismiss_refusal``.
    """
    queue = _queue(tmp_path, FakeSession(), now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 200_000
    assert queue.dismiss(ask_id, by="terminal")["ok"] is True
    assert queue.dismiss(ask_id, by="terminal") == {
        "ok": False,
        "error": "you already dismissed this.",
    }


def test_decline_is_terminal_on_write(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.decline(ask_id, by="terminal")["ok"] is True
    declined = queue.find(ask_id)
    assert declined is not None and declined["status"] == store.STATUS_DECLINED
    assert queue.respond(ask_id, {"q0": ["yes"]})["ok"] is False


# ---------------------------------------------------------------------------
# withdraw: the agent-side settle (design §12)
# ---------------------------------------------------------------------------


def test_moot_withdraw_settles_writes_one_row_and_injects_nothing(tmp_path: Path):
    """The moot half, end to end at the queue's level: one ``withdrawn`` append,
    a terminal fold, NO expected row — and the answer path refuses in the
    state's own words because the box is gone everywhere."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.withdraw(ask_id, reason="moot")
    assert outcome["ok"] is True
    assert "withdrawn" in outcome["text"] and "nothing will be delivered" in outcome["text"]
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_WITHDRAWN
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds == [store.EVENT_QUEUED, store.EVENT_WITHDRAWN]
    _run(queue.reconcile())
    assert session.transcript.ids == set()
    assert session.batches == []
    # A stale surface's answer is refused, and the sentence points at chat.
    answer = queue.respond(ask_id, {"q0": ["yes"], "q1": ["no"]})
    assert answer["ok"] is False
    assert "withdrew this question" in answer["error"]
    # A second settle is refused too, with the op's own words.
    again = queue.withdraw(ask_id, reason="moot")
    assert again["ok"] is False
    assert "already withdrawn" in again["error"]


def test_a_moot_withdraw_is_allowed_on_a_timed_out_ask(tmp_path: Path):
    """A timed-out ask is still admissible — the agent retracting a question
    nobody answered is the PRIMARY shape §12 exists for."""
    queue = _queue(tmp_path, FakeSession())
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 300_000  # past the deadline, inside the window
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_TIMED_OUT
    assert queue.withdraw(ask_id, reason="moot")["ok"] is True
    settled = queue.find(ask_id)
    assert settled is not None and settled["status"] == store.STATUS_WITHDRAWN


def test_a_recorded_answer_refuses_both_withdraw_reasons_and_writes_nothing(
    tmp_path: Path,
):
    """Contract (a) at the OP level: with the answer recorded, the withdrawal
    is refused and no ``withdrawn`` row is written, so the fold carve-out is
    belt-and-braces for the true cross-process race only."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["yes"]}, by="terminal")["ok"] is True
    moot = queue.withdraw(ask_id, reason="moot")
    assert moot["ok"] is False and "already has the user's answer" in moot["error"]
    chat = queue.withdraw(ask_id, reason="answered_in_chat", answers={"q0": ["again"]})
    assert chat["ok"] is False and "not recorded" in chat["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds.count(store.EVENT_WITHDRAWN) == 0
    assert kinds.count(store.EVENT_ANSWERED) == 1


def test_withdraw_refusals_are_per_state_and_change_nothing(tmp_path: Path):
    """The settled states each no-op with a truthful sentence — declined and
    dismissed say who acted, an expiry says there is nothing to retract — and
    none of them appends anything."""
    queue = _queue(tmp_path, FakeSession())
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    assert queue.decline(ask_id, by="phone")["ok"] is True
    before = store.read_events(queue.session_dir)
    declined = queue.withdraw(ask_id, reason="moot")
    assert declined["ok"] is False and "declined" in declined["error"]
    chat = queue.withdraw(ask_id, reason="answered_in_chat", answers={"q0": ["x"]})
    assert chat["ok"] is False and "not recorded" in chat["error"]
    assert store.read_events(queue.session_dir) == before

    dismissed_queue = _queue(tmp_path, FakeSession())
    dismissed_queue._now = lambda: BASE
    second_id = dismissed_queue.enqueue(_questions(), 120)["details"]["ask_id"]
    dismissed_queue._now = lambda: BASE + 300_000
    assert dismissed_queue.dismiss(second_id, by="phone")["ok"] is True
    outcome = dismissed_queue.withdraw(second_id, reason="moot")
    assert outcome["ok"] is False and "dismissed" in outcome["error"]

    expired_queue = _queue(tmp_path, FakeSession())
    expired_queue._now = lambda: BASE
    third_id = expired_queue.enqueue(_questions(), 120)["details"]["ask_id"]
    expired_queue._now = lambda: BASE + store.LATE_WINDOW_S * 1000 + 200_000
    outcome = expired_queue.withdraw(third_id, reason="moot")
    assert outcome["ok"] is False and "expired" in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds.count(store.EVENT_WITHDRAWN) == 0


def test_the_reason_is_validated_and_moot_refuses_answers(tmp_path: Path):
    """Both refusal guards before any state is consulted, and neither writes:
    an unknown reason fails on the call, and answers on a ``moot`` are refused
    rather than silently dropped (the model meant to record words)."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    bad = queue.withdraw(ask_id, reason="settle")
    assert bad["ok"] is False and "moot" in bad["error"] and "answered_in_chat" in bad["error"]
    mismatch = queue.withdraw(ask_id, reason="moot", answers={"q0": ["words"]})
    assert mismatch["ok"] is False and "no answers" in mismatch["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds == [store.EVENT_QUEUED]


def test_answered_in_chat_records_verbatim_cells_and_names_the_chat_surface(
    tmp_path: Path,
):
    """The chat half: the user's words as cells — an EMPTY LIST for a question
    the message does not cover, the §2.4 completeness contract on the KEYS —
    attributed to the chat surface (and the message, when the caller knows
    it), and the standard response row follows through ``reconcile``."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.withdraw(
        ask_id,
        reason="answered_in_chat",
        answers={"q0": ["the audit-log one", "keep it"], "q1": []},
        message_id="m-7",
    )
    assert outcome["ok"] is True
    assert "answered from the user's chat message" in outcome["text"]
    record = queue.find(ask_id)
    assert record is not None
    assert record["status"] == store.STATUS_ANSWERED
    assert record["answers"] == {"q0": ["the audit-log one", "keep it"], "q1": []}
    assert record["answered_by"] == {"surface": "chat", "message_id": "m-7"}
    _run(queue.reconcile())
    assert store.response_row_id(ask_id) in session.transcript.ids


def test_answered_in_chat_without_a_message_id_carries_the_surface_alone(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.withdraw(ask_id, reason="answered_in_chat", answers={"q0": ["yes"]})["ok"] is True
    record = queue.find(ask_id)
    assert record is not None and record["answered_by"] == {"surface": "chat"}


def test_answered_in_chat_on_a_timed_out_ask_folds_late(tmp_path: Path):
    """Admissible where an answer is: past the deadline the fold says ``late``,
    exactly as a card answer after the deadline would — the same one response
    row, one deadline too late."""
    queue = _queue(tmp_path, FakeSession())
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 300_000
    outcome = queue.withdraw(
        ask_id, reason="answered_in_chat", answers={"q0": ["after the deadline"]}
    )
    assert outcome["ok"] is True
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_LATE


def test_answered_in_chat_refuses_a_secret_question_with_its_own_sentence(tmp_path: Path):
    """The card stays the only secret path: no masked-entry hop from chat text
    exists and none may be invented, so the whole op is refused and NOTHING is
    written — the sentinel never touches any file because no row was appended."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(1, secret=True), None)["details"]["ask_id"]
    outcome = queue.withdraw(
        ask_id, reason="answered_in_chat", answers={"key-0": ["SENTINEL-VALUE"]}
    )
    assert outcome["ok"] is False
    assert "only its card" in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds == [store.EVENT_QUEUED]
    assert "SENTINEL-VALUE" not in store.asks_log_path(queue.session_dir).read_text()


def test_answered_in_chat_refuses_a_missing_key_before_writing(tmp_path: Path):
    """The §2.4 completeness contract is on the KEYS: a question the message
    does not cover is sent as an EMPTY LIST, and a forgotten key is refused by
    name rather than settling the ask without it."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.withdraw(ask_id, reason="answered_in_chat", answers={"q0": ["yes"]})
    assert outcome["ok"] is False
    assert "q1" in outcome["error"] and "has no entry" in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert store.EVENT_ANSWERED not in kinds


def test_answered_in_chat_refuses_an_empty_map_before_writing(tmp_path: Path):
    """The §2.4 completeness contract on the EMPTY map (review round 1, NIT 5):
    `{}` covers no question — and a missing map is the same `{}` after
    defaulting — so both are refused by name before anything is written: an
    `answered` row whose cells attribute nothing would be a lie about what the
    user said."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    refusals = [
        queue.withdraw(ask_id, reason="answered_in_chat", answers={}),
        queue.withdraw(ask_id, reason="answered_in_chat", answers=None),
    ]
    for outcome in refusals:
        assert outcome["ok"] is False
        assert "q0" in outcome["error"] and "q1" in outcome["error"]
        assert "has no entry" in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert store.EVENT_ANSWERED not in kinds


def test_answering_a_withdrawn_ask_in_words_is_refused_not_dropped(tmp_path: Path):
    """The refused surfaces read the same sentence §12 puts in the state table:
    the state and the op agree about the one route left."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.withdraw(ask_id, reason="moot")["ok"] is True
    outcome = queue.respond(ask_id, {"q0": ["a real answer"]})
    assert outcome["ok"] is False
    assert (
        outcome["error"]
        == "the agent withdrew this question — if you have an answer, send it as a chat message."
    )


def test_classifying_a_wrong_key_as_unknown_is_not_the_partial_sentence(tmp_path: Path):
    """The field case behind §16/SA-D1: ``bill-merge-permission`` sent for the one
    question ``merge-permission`` was refused as a PARTIAL map, which told the
    model to send an empty list for a question it had just answered. The refusal
    names the key it did not recognise and the ids the ask holds, and nothing is
    written."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue([{**_questions()[0], "id": "merge-permission"}], None)["details"][
        "ask_id"
    ]
    outcome = queue.respond(ask_id, {"bill-merge-permission": ["yes"]})
    assert outcome["ok"] is False
    assert "'bill-merge-permission' is not a question on this ask" in outcome["error"]
    assert "'merge-permission'" in outcome["error"]
    assert "has no entry" not in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds == [store.EVENT_QUEUED]


def test_withdraw_with_a_wrong_id_names_it_and_writes_nothing(tmp_path: Path):
    """Acceptance 2 of the slice: the ASKER's own settle shares ``_whole_ask_cells``,
    so a typo is answered with the same sentence rather than the partial one — and
    no ``answered`` row is appended for the retry to race."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.withdraw(
        ask_id, reason="answered_in_chat", answers={"q0": ["yes"], "qq1": ["no"]}
    )
    assert outcome["ok"] is False
    assert "'qq1' is not a question on this ask" in outcome["error"]
    assert "has no entry" not in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds == [store.EVENT_QUEUED]


def test_an_unknown_key_beside_every_real_one_is_refused_not_stored(tmp_path: Path):
    """Acceptance 3 of the slice: a map carrying every real id plus a stray key
    used to settle the ask and write the stray cell into the log, where nothing
    can attribute it. It is now refused as a whole, on every door that shares
    ``_whole_ask_cells`` — ``respond`` here, ``revise`` below — while an empty
    answer list is still how a deliberate skip is said."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.respond(ask_id, {"q0": ["yes"], "q1": ["no"], "q9": ["stray"]})
    assert outcome["ok"] is False
    assert "'q9' is not a question on this ask" in outcome["error"]
    assert "'q0'" in outcome["error"] and "'q1'" in outcome["error"]
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_OPEN
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert store.EVENT_ANSWERED not in kinds

    assert queue.respond(ask_id, {"q0": ["yes"], "q1": []})["ok"] is True
    revised = queue.revise(ask_id, {"q0": ["yes"], "q1": [""], "q9": ["stray"]})
    assert revised["ok"] is False
    assert "'q9' is not a question on this ask" in revised["error"]


def test_the_unknown_key_echo_is_bounded(tmp_path: Path):
    """Review round 1 NIT 3: the unknown keys come from the CALLER, so the
    refusal quotes at most 5 of them, each at most 40 characters, and says how
    many it did not name. The ids the ask holds are echoed whole — the model
    chose them."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(1), None)["details"]["ask_id"]
    long_key = "x" * 200
    unknown = {long_key: ["v"], **{f"k{index}": ["v"] for index in range(6)}}
    outcome = queue.respond(ask_id, unknown)
    assert outcome["ok"] is False
    assert long_key not in outcome["error"]
    assert f"'{'x' * 40}…'" in outcome["error"]
    assert "(and 2 more)" in outcome["error"]
    assert outcome["error"].count("'k") == 4


def test_partial_chat_answer_names_the_questions_it_closed_unanswered(tmp_path: Path):
    """Acceptance 1 of the slice: a 3-question ask settled with the user's words
    in one cell and empty lists in two folds ``answered`` — the empty cell is how
    the map says "this message does not cover that question" — the receipt names
    the 2 questions that closed with it, the response row renders them as
    ``(not answered)``, and the ask leaves the open set."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(3), None)["details"]["ask_id"]
    outcome = queue.withdraw(
        ask_id,
        reason="answered_in_chat",
        answers={"q0": ["the audit-log one"], "q1": [], "q2": []},
    )
    assert outcome["ok"] is True
    assert "2 questions closed unanswered; re-ask any you still need." in outcome["text"]
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_ANSWERED
    assert record["answers"] == {"q0": ["the audit-log one"], "q1": [], "q2": []}
    text = render.response_text(record)
    assert "the audit-log one" in text
    assert text.count("(not answered)") == 2
    assert ask_id not in [row["ask_id"] for row in queue.open_records()]


def test_a_single_closed_question_is_counted_in_the_singular(tmp_path: Path):
    """The receipt reads as one sentence, not a template: 1 question, not
    "1 questions" — the count is the fact the model acts on."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.withdraw(ask_id, reason="answered_in_chat", answers={"q0": ["yes"], "q1": []})
    assert outcome["ok"] is True
    assert "1 question closed unanswered" in outcome["text"]
    assert "questions closed" not in outcome["text"]


def test_a_blank_cell_counts_as_closed_unanswered(tmp_path: Path):
    """Review round 1 m1 / QA Q-1: the count follows the ROW's rule, not the
    list's length. ``_ask_report`` drops blank text, so ``[" "]`` renders
    ``(not answered)`` — counting only ``[]`` would tell the model nothing was
    left behind while the response turn says a question was."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    outcome = queue.withdraw(
        ask_id, reason="answered_in_chat", answers={"q0": ["large"], "q1": ["  "]}
    )
    assert outcome["ok"] is True
    assert "1 question closed unanswered" in outcome["text"]
    record = queue.find(ask_id)
    assert record is not None
    assert render.response_text(record).count("(not answered)") == 1


def test_an_all_empty_chat_map_is_refused_and_points_at_moot(tmp_path: Path):
    """Review round 1 m2: every cell empty — or blank — is a model asking to
    record words nobody said, and the empty-list recipe this release teaches
    every session makes it reachable. Refused rather than rerouted to ``moot``:
    a moot writes ``withdrawn``, a different fact about who acted, and this
    model must hear that it recorded nothing. Nothing is appended either way."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    for cells in ({"q0": [], "q1": []}, {"q0": [""], "q1": [" "]}):
        outcome = queue.withdraw(ask_id, reason="answered_in_chat", answers=cells)
        assert outcome["ok"] is False
        assert "reason='moot'" in outcome["error"]
        assert "nothing was recorded" in outcome["error"]
    kinds = [event["kind"] for event in store.read_events(queue.session_dir)]
    assert kinds == [store.EVENT_QUEUED]
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_OPEN


# ---------------------------------------------------------------------------
# reconcile: which rows are written, and the N8 suppression
# ---------------------------------------------------------------------------


def test_reconcile_writes_the_response_row_for_an_answer(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    queue.respond(ask_id, {"q0": ["yes"]}, by="terminal")
    written = _run(queue.reconcile())
    assert written == [store.response_row_id(ask_id)]
    assert len(session.batches) == 1
    assert session.batches[0][0].custom_type == "ask_response"
    assert session.batches[0][0].details["status"] == store.STATUS_ANSWERED


def test_reconcile_is_idempotent(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    queue.respond(ask_id, {"q0": ["yes"]})
    assert _run(queue.reconcile()) != []
    assert _run(queue.reconcile()) == []
    assert len(session.batches) == 1


def test_reconcile_writes_only_the_timeout_row_for_a_deadline(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 200_000
    written = _run(queue.reconcile())
    assert written == [store.timeout_row_id(ask_id)]
    assert session.batches[0][0].custom_type == "ask_timeout"
    assert session.batches[0][0].attribution == "system"


def test_a_declined_ask_writes_the_response_row(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    queue.decline(ask_id, by="terminal")
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    assert session.batches[0][0].details["status"] == store.STATUS_DECLINED


def test_a_dismissed_ask_injects_nothing(tmp_path: Path):
    """AMENDED 2026-10-16 (fork A, §14), NARROWED 2026-10-19 (§15): this cell pins
    the POST-DEADLINE case the old sentence is still true for — the deadline notice
    was ALREADY delivered. A dismissal stamped before the deadline owes a response
    row instead (the cells below).

    Before the fork, ANY dismissal injected nothing, so this passed by asserting
    the whole rule. The rule narrowed rather than disappeared: a dismissal whose
    notice never reached the transcript still owes it (the next two cells), and
    one whose row is already durable owes nothing — the row is the delivery
    marker, so the dismissal has nothing left to deliver.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    session.transcript.ids.add(store.timeout_row_id(ask_id))
    queue._now = lambda: BASE + 200_000
    assert queue.dismiss(ask_id, by="terminal")["ok"] is True
    assert _run(queue.reconcile()) == []
    assert session.batches == []


def test_a_cold_dismissal_delivers_its_notice_on_reopen(tmp_path: Path):
    """THE FORK'S OWN CELL (A, §14): a dismissal does not swallow the notice.

    The shape is the one Ben hit: the session was STOPPED when the deadline
    passed, so the notice was never delivered, and the dismissal arrives from a
    surface with no runtime (the phone, an external writer — "any process may
    append answered/declined/dismissed"). The next runtime's LOAD-TIME reconcile
    must still inject the deadline notice, in its dismissed wording, because the
    agent has never been told it is waiting on an answer that is not coming.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    store.append_event(queue.session_dir, _dismissed_event(ask_id, at=BASE + 200_000))

    reopened = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: BASE + 200_100)
    assert _run(reopened.reconcile(load_time=True)) == [store.timeout_row_id(ask_id)]
    (message,) = session.batches[0]
    assert message.id == store.timeout_row_id(ask_id)
    assert message.details["status"] == store.STATUS_TIMED_OUT
    assert message.details["dismissed"] is True
    assert "dismissed it without answering" in message.details["text"]


def test_an_owed_dismissed_notice_is_delivered_once(tmp_path: Path):
    """Level-triggered and idempotent, like every other row: the marker is the
    transcript row (``ask-timeout-<id>``), so the second reconcile writes
    nothing rather than injecting the notice twice."""
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    store.append_event(queue.session_dir, _dismissed_event(ask_id, at=BASE + 200_000))

    reopened = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: BASE + 200_100)
    assert _run(reopened.reconcile(load_time=True)) == [store.timeout_row_id(ask_id)]
    assert _run(reopened.reconcile()) == []
    assert len(session.batches) == 1


def test_an_expired_ask_injects_nothing(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + store.LATE_WINDOW_S * 1000 + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + store.LATE_WINDOW_S * 1000 + 200_000
    assert _run(queue.reconcile()) == []
    assert session.batches == []


def test_a_late_answer_in_one_batch_suppresses_the_timeout_row(tmp_path: Path):
    """N8: a cold boot after an answer that arrived past the deadline must NOT
    replay "[Ask timed out] … you will be told" immediately before the answer it
    announces. Same order asserted the other way in the next test."""
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 200_000
    queue.respond(ask_id, {"q0": ["late"]}, by="terminal")
    written = _run(queue.reconcile())
    assert written == [store.response_row_id(ask_id)]
    assert [m.custom_type for m in session.batches[0]] == ["ask_response"]
    assert session.batches[0][0].details["status"] == store.STATUS_LATE


def test_a_late_answer_already_past_the_deadline_writes_both_rows_in_order(tmp_path: Path):
    """The other order: the timeout row was already delivered by an earlier
    boot, so the late answer adds ONLY the response (never a second timeout)."""
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 130_000
    # First boot: the deadline passes with nobody answering → the notice goes out.
    assert _run(queue.reconcile()) == [store.timeout_row_id(ask_id)]
    # The answer then arrives, still within the late window.
    queue._now = lambda: BASE + 200_000
    queue.respond(ask_id, {"q0": ["late"]}, by="terminal")
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    assert [m.custom_type for m in session.batches[1]] == ["ask_response"]


def test_delivered_is_sticky_after_the_fold_moves_on(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    # The answer arrives PAST the deadline, so the ask is `late` — the status
    # that later folds to `expired` (an in-window answer is `answered` for good).
    queue._now = lambda: BASE + 200_000
    queue.respond(ask_id, {"q0": ["late"]})
    _run(queue.reconcile())
    later = _queue(tmp_path, session, now=BASE + store.LATE_WINDOW_S * 1000 + 200_000)
    record = later.find(ask_id)
    assert record is not None
    assert record["status"] == store.STATUS_EXPIRED
    # The row it wrote is still in the transcript, so the flag never flips back.
    assert record["delivered"] is True


def test_a_timed_out_ask_is_still_answerable_and_folds_to_late(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    opened = queue.find(ask_id)
    assert opened is not None and opened["status"] == store.STATUS_OPEN
    queue._now = lambda: BASE + 200_000
    timed_out = queue.find(ask_id)
    assert timed_out is not None and timed_out["status"] == store.STATUS_TIMED_OUT
    assert queue.respond(ask_id, {"q0": ["late answer"]}, by="phone")["ok"] is True
    late = queue.find(ask_id)
    assert late is not None and late["status"] == store.STATUS_LATE


def test_three_asks_answered_out_of_order_produce_one_row_each(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ids = [
        queue.enqueue(_questions(text=f"question {index}"), None)["details"]["ask_id"]
        for index in range(3)
    ]
    for ask_id in (ids[2], ids[0], ids[1]):
        assert queue.respond(ask_id, {"q0": ["yes"]}, by="terminal")["ok"] is True
    written = _run(queue.reconcile())
    assert sorted(written) == sorted(store.response_row_id(ask_id) for ask_id in ids)
    assert len(session.batches) == 1  # one batch, one paid turn
    assert len(session.batches[0]) == 3


def test_the_expiry_event_carries_the_ask_id_and_no_question_text_for_a_secret(
    tmp_path: Path,
):
    """A secret ask's timeout names the KEY and never quotes the prompt: the
    question text may itself describe the credential."""
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(
        [
            {
                "id": "API_KEY",
                "question": "Paste the sk-live key",
                "options": [],
                "multi": False,
                "secret": True,
                "persist": False,
                "recommended": None,
            }
        ],
        120,
    )
    queue._now = lambda: BASE + 200_000
    _run(queue.reconcile())
    text = session.batches[0][0].details["text"]
    assert "API_KEY" in text
    assert "sk-live" not in text


def test_a_second_reconcile_does_not_write_the_timeout_after_a_late_answer(tmp_path: Path):
    """MAJOR 2: the suppression is LEVEL-TRIGGERED, not a one-batch deferral.

    The reviewer's probe, verbatim in shape: a cold late answer, then TWO
    reconciles. The one-shot form wrote the response first and the ``ask-timeout-``
    row on the very next pass, so the model read "[Ask timed out] … if they answer
    later you will be told" directly below the answer that was the reply. Reconcile
    re-runs on every answer, at every turn start and on the timer tick, so "one
    reconcile later" is the common case rather than a corner.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 200_000
    queue.respond(ask_id, {"q0": ["late"]}, by="terminal")
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    assert _run(queue.reconcile()) == []
    assert _run(queue.reconcile()) == []
    kinds = [m.custom_type for batch in session.batches for m in batch]
    assert kinds == ["ask_response"]


def test_a_deadline_that_fired_first_still_writes_its_own_row(tmp_path: Path):
    """The other side of MAJOR 2: the rule removes a CONTRADICTION, not the notice.

    An ask whose window closed with nobody watching is ``timed_out`` — no response
    exists — so its deadline row is delivered exactly as §2.5 requires, and the
    late answer that follows adds the response beside it.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 130_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 130_000
    assert _run(queue.reconcile()) == [store.timeout_row_id(ask_id)]
    assert _run(queue.reconcile()) == []
    queue._now = lambda: BASE + 200_000
    queue.respond(ask_id, {"q0": ["late"]}, by="terminal")
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    kinds = [m.custom_type for batch in session.batches for m in batch]
    assert kinds == ["ask_timeout", "ask_response"]


# ---------------------------------------------------------------------------
# the review round 1 contracts
# ---------------------------------------------------------------------------


def test_a_partial_answer_map_is_refused_and_names_the_missing_ids(tmp_path: Path):
    """QA Q1/§2.4: the submit is atomic per ask, so omitting a question is refused.

    The row that lands is TERMINAL, which is what makes a forgotten key a silent
    loss rather than a partial answer: the ask can never be completed afterwards.
    The refusal has to name the ids, because the surface that has to fix it is the
    one being told.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(2), 120)["details"]["ask_id"]
    outcome = queue.respond(ask_id, {"q0": ["yes"]}, by="desktop")
    assert outcome["ok"] is False
    assert "'q1'" in outcome["error"]
    record = queue.find(ask_id)
    assert record is not None and record["status"] == store.STATUS_OPEN
    # And the ask is still answerable in full, which is the point.
    assert queue.respond(ask_id, {"q0": ["yes"], "q1": ["eu"]}, by="desktop")["ok"] is True


def test_an_empty_list_is_how_a_skipped_question_is_sent(tmp_path: Path):
    """Completeness is checked on the KEYS, not the values: a question the user
    skipped has to be sayable, and an empty list says it."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(2), 120)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": [], "q1": ["eu"]}, by="terminal")["ok"] is True
    answered = queue.find(ask_id)
    assert answered is not None and answered["status"] == store.STATUS_ANSWERED


def test_a_value_shaped_answer_for_a_secret_question_cannot_reach_the_log(
    tmp_path: Path,
):
    """MINOR 6: the queue itself refuses a raw value, not just its caller.

    ``Session.respond_ask`` substitutes the key name BEFORE calling here; this is
    what keeps that hop load-bearing, because a cold CLI or relay route in B/C
    reaches the queue directly. Asserted by grepping the LOG for the sentinel, not
    by reading the return value — the contract is about what is durable.
    """
    sentinel = "sk-live-do-not-persist"
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(1, secret=True), 120)["details"]["ask_id"]
    assert queue.respond(ask_id, {"key-0": [sentinel]}, by="cli")["ok"] is True
    log = store.asks_log_path(store.session_dir(tmp_path, "s1")).read_text()
    assert sentinel not in log
    record = queue.find(ask_id)
    assert record is not None and record["answers"]["key-0"] == ["<not provided>"]


def test_the_key_name_itself_is_still_accepted_for_a_secret_question(tmp_path: Path):
    """The guard must not eat the LEGITIMATE cell: the key name is what
    ``Session.respond_ask`` passes after storing the value, and the model has to
    read it back."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(1, secret=True), 120)["details"]["ask_id"]
    assert queue.respond(ask_id, {"key-0": ["KEY-0"]}, by="terminal")["ok"] is True
    record = queue.find(ask_id)
    assert record is not None and record["answers"]["key-0"] == ["KEY-0"]


def test_the_key_shape_helper_matches_the_real_credential_normaliser():
    """The guard re-spells ``variables.normalize_credential_key`` rather than
    importing it (the asks package stays on the lean side of the import graph), so
    the two spellings are pinned together here: if the real normaliser changes,
    this fails instead of the guard quietly widening or narrowing."""
    from local_operator.asks.queue import _credential_key_shape
    from local_operator.variables import normalize_credential_key

    for raw in ("api key", "API-KEY", "api_key", "API_KEY", "  Api.Key  "):
        assert _credential_key_shape(raw) == normalize_credential_key(raw)


def test_the_tool_call_id_rides_the_queued_event_and_the_answer(tmp_path: Path):
    """MINOR 5: the id came from an attribute nothing in the tree ever set, so
    every ask carried ``""`` and A2's card could not link ask to call."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), 120, tool_call_id="call-7")["details"]["ask_id"]
    record = queue.find(ask_id)
    assert record is not None and record["tool_call_id"] == "call-7"
    queue.respond(ask_id, {"q0": ["yes"]}, by="terminal")
    _run(queue.reconcile())
    assert session.batches[0][0].details["tool_call_id"] == "call-7"


def test_a_secret_whose_key_is_gone_is_delivered_as_lost(tmp_path: Path):
    """MAJOR 3/§2.4: verify-on-delivery. The value lives in session memory, so a
    restart between the answer and its delivery leaves a row announcing a
    credential the runtime does not hold — the one path where the user answered
    and the agent still cannot proceed."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(1, secret=True), 120)["details"]["ask_id"]
    queue.respond(ask_id, {"key-0": ["KEY-0"]}, by="terminal")
    # The store no longer holds it (the restart the design warns about).
    session.credentials = []
    _run(queue.reconcile())
    details = session.batches[0][0].details
    assert details["secret_lost"] is True
    assert "session restarted" in details["text"]


def test_a_secret_the_session_still_holds_is_delivered_normally(tmp_path: Path):
    """The other half, and the one that keeps the gate honest: a key that IS in
    the store must not be reported lost."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(1, secret=True), 120)["details"]["ask_id"]
    queue.respond(ask_id, {"key-0": ["KEY-0"]}, by="terminal")
    session.credentials = ["KEY-0"]
    _run(queue.reconcile())
    details = session.batches[0][0].details
    assert details["secret_lost"] is False
    assert "session restarted" not in details["text"]


def test_a_declined_secret_is_not_reported_as_lost(tmp_path: Path):
    """``<not provided>`` is a decision, not a loss: nothing was ever handed
    over, so the lost copy would be a lie about what the user did."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(1, secret=True), 120)["details"]["ask_id"]
    queue.respond(ask_id, {"key-0": ["<not provided>"]}, by="terminal")
    _run(queue.reconcile())
    assert session.batches[0][0].details["secret_lost"] is False


def _stop_marker(session_dir: Path, at_ms: int, **extra: Any) -> None:
    """Write the durable stop marker a deliberate stop leaves behind.

    ``session_id`` is not decoration: the classification reader applies a
    run-covers check to this file (``attention._stop_marker_covers_run``) and the
    ask queue now applies the same conversation clause, so a marker written
    without one is evidence about nobody.
    """
    payload = {"deliberate": True, "session_id": "s1", "at": at_ms / 1000.0}
    payload.update(extra)
    (session_dir / "runtime-stop.json").write_text(json.dumps(payload))


def test_a_deadline_that_passed_while_stopped_says_so(tmp_path: Path):
    """QA Q2/§2.2: the stop rule's copy half, on the REOPEN.

    The notice is owed either way — without the sentence the user who stopped the
    session is told the agent moved on as though they had been watching. It is
    delivered by the load-time reconcile, which is the caller that passes
    ``load_time=True``.
    """
    session = FakeSession()
    session_dir = store.session_dir(tmp_path, "s1")
    session_dir.mkdir(parents=True, exist_ok=True)
    # A DELIBERATE stop, stamped between the ask and its deadline.
    _stop_marker(session_dir, BASE + 60_000)
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + 200_000
    _run(queue.reconcile(load_time=True))
    details = session.batches[0][0].details
    assert details["lapsed_while_stopped"] is True
    assert "lapsed while the session was stopped" in details["text"]


def test_a_live_deadline_lapse_is_not_annotated(tmp_path: Path):
    """M1 (review round 2): the sentence belongs to the REOPEN, not to the timing.

    The reachable false positive, verbatim: stop with an ask open, reopen BEFORE
    the deadline, let the deadline elapse while running. Every window end holds —
    the marker is inside the ask's own window — so the ask's window alone cannot
    decide it, and annotating here would be false in both halves: nothing lapsed
    because of the stop, and no notice was withheld by it.
    """
    session = FakeSession()
    session_dir = store.session_dir(tmp_path, "s1")
    session_dir.mkdir(parents=True, exist_ok=True)
    _stop_marker(session_dir, BASE + 60_000)
    queue = _queue(tmp_path, session, now=BASE)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    # The reopen, still before the deadline: nothing is owed yet.
    queue._now = lambda: BASE + 90_000
    assert _run(queue.reconcile(load_time=True)) == []
    # ...and now the deadline elapses with the session RUNNING: the live tick.
    queue._now = lambda: BASE + 200_000
    assert _run(queue.reconcile()) == [store.timeout_row_id(ask_id)]
    details = session.batches[0][0].details
    assert details["lapsed_while_stopped"] is False
    assert "lapsed while" not in details["text"]


def test_the_live_tick_never_annotates_even_when_the_marker_is_in_the_window(
    tmp_path: Path,
):
    """The gate is the CALLER, not the clock: the same marker, the same ask, the
    same deadline — only ``load_time`` differs, and it is what arms the sentence."""
    session = FakeSession()
    session_dir = store.session_dir(tmp_path, "s1")
    session_dir.mkdir(parents=True, exist_ok=True)
    _stop_marker(session_dir, BASE + 60_000)
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + 200_000
    _run(queue.reconcile())
    assert session.batches[0][0].details["lapsed_while_stopped"] is False
    # The SAME delivery through the reopen reconcile is annotated, and "the
    # same" is literal here: a fresh queue over the same directories is what a
    # reopen is, and the row is re-armed by dropping it from the fake transcript,
    # which is the only delivery marker there is.
    session.transcript.ids.clear()
    reopened = _queue(tmp_path, session, now=BASE + 200_000)
    reopened._now = lambda: BASE + 200_000
    _run(reopened.reconcile(load_time=True))
    assert session.batches[1][0].id == session.batches[0][0].id
    assert session.batches[1][0].details["lapsed_while_stopped"] is True


def test_a_marker_for_another_conversation_is_not_credited(tmp_path: Path):
    """The run-covers clause the classification reader applies: a marker is keyed
    to a conversation, so one naming a different session must never narrate this
    one's ask — the file is read from a directory the queue was merely handed."""
    session = FakeSession()
    session_dir = store.session_dir(tmp_path, "s1")
    session_dir.mkdir(parents=True, exist_ok=True)
    _stop_marker(session_dir, BASE + 60_000, session_id="somewhere-else")
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + 200_000
    _run(queue.reconcile(load_time=True))
    assert session.batches[0][0].details["lapsed_while_stopped"] is False


def test_an_involuntary_death_does_not_claim_the_session_was_stopped(tmp_path: Path):
    """The marker also records crashes (a reap, a stray kill). Telling a user their
    session "was stopped" when it died is the kind of wrong sentence that makes the
    honest ones worthless, so ``deliberate`` is required."""
    session = FakeSession()
    session_dir = store.session_dir(tmp_path, "s1")
    session_dir.mkdir(parents=True, exist_ok=True)
    _stop_marker(session_dir, BASE + 60_000, deliberate=False, mechanism="reap")
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + 200_000
    _run(queue.reconcile(load_time=True))
    details = session.batches[0][0].details
    assert details["lapsed_while_stopped"] is False
    assert "lapsed while" not in details["text"]


def test_a_stop_outside_the_asks_own_window_is_not_annotated(tmp_path: Path):
    """The window is the ask's own: an old stop marker may not narrate a deadline
    that fell while the session was running."""
    session = FakeSession()
    session_dir = store.session_dir(tmp_path, "s1")
    session_dir.mkdir(parents=True, exist_ok=True)
    # The stop happened BEFORE this ask existed.
    _stop_marker(session_dir, BASE - 600_000)
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + 200_000
    _run(queue.reconcile(load_time=True))
    assert session.batches[0][0].details["lapsed_while_stopped"] is False


# ---------------------------------------------------------------------------
# §10, #1936 — the in-flight ANSWER REVISION, bounded by delivery
# ---------------------------------------------------------------------------

#: The revision path's own sentence. NOT ``render.refusal_copy``'s table: the
#: state table must keep answering "already answered by <surface>" for a plain
#: repeat, so the delivered line belongs to the op (design §10).
DELIVERED_REFUSAL = "already delivered — send a new message"


def test_repro_the_two_question_revision_window_closes_at_delivery(tmp_path: Path):
    """THE #1936 REPRODUCTION, end to end on the queue.

    Two questions in one ask: answer it, CHANGE the answer while it is still
    undelivered (accepted, supersedes), let the delivery land, then try to change
    it again (refused). Before this op the first of those was refused — the
    reporter's "I entered an incorrect answer and I need to interrupt the session
    to ask it again".
    """
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]

    # 1. the first answer, wrong on the first question.
    assert queue.respond(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")["ok"] is True

    # 2. still undelivered: the explicit revision is ACCEPTED and supersedes.
    outcome = queue.revise(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="desktop")
    assert outcome["ok"] is True
    assert outcome["revised"] is True
    events = store.read_events(queue.session_dir)
    assert [event["kind"] for event in events] == [
        store.EVENT_QUEUED,
        store.EVENT_ANSWERED,
        store.EVENT_REVISED,
    ]
    revised = events[-1]
    assert revised["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    assert revised["by"] == {"surface": "desktop"}
    assert revised["supersedes"] == BASE

    # 3. delivery: ONE response row, and it carries the REVISED map.
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    assert session.batches[-1][0].details["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    record = queue.find(ask_id)
    assert record is not None
    assert record["status"] == store.STATUS_ANSWERED
    # The FIRST answer keeps the stamp and the attribution; only the answers move.
    assert record["answered_at"] == BASE
    assert record["answered_by"] == {"surface": "terminal"}
    assert record["answers"] == {"q0": ["yes"], "q1": ["maybe"]}

    # 4. delivered: the window has closed, in the revision path's own words.
    refused = queue.revise(ask_id, {"q0": ["maybe"], "q1": ["maybe"]})
    assert refused["ok"] is False
    assert refused["error"] == DELIVERED_REFUSAL
    # The row still says exactly what the model was told, and no second revision
    # reached the log.
    assert [event["kind"] for event in store.read_events(queue.session_dir)][-1] == (
        store.EVENT_REVISED
    )
    after = queue.find(ask_id)
    assert after is not None and after["answers"] == {"q0": ["yes"], "q1": ["maybe"]}


def test_delivered_and_the_window_key_on_the_response_row_not_the_timeout_notice(
    tmp_path: Path,
):
    """THE CONSUMPTION BOUND, at the queue's own fold (amended 2026-10-04).

    ``delivered`` is per-status now: for a LATE answer it counts the RESPONSE row
    only — the deadline notice went out but does not deliver the answer — so the
    flag reads False exactly while the revision window is open, and a revision
    there is free to make. (The old sticky hint counted the notice, so the fold
    disagreed with the window it was documented as the window.) After the response
    row lands, the flag flips True and the window is shut for good.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 120_000
    assert _run(queue.reconcile()) == [store.timeout_row_id(ask_id)]

    queue._now = lambda: BASE + 130_000
    assert queue.respond(ask_id, {"q0": ["late"]}, by="terminal")["ok"] is True
    record = queue.find(ask_id)
    assert record is not None
    assert record["status"] == store.STATUS_LATE
    assert record["delivered"] is False, "the notice does not deliver the answer"
    assert store.response_row_id(ask_id) not in session.transcript.ids

    outcome = queue.revise(ask_id, {"q0": ["late but corrected"]})
    assert outcome["ok"] is True and outcome["revised"] is True
    # The response row is what delivers a late answer, so the corrected map is
    # what lands — and only ONE row does.
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    assert session.batches[-1][0].details["answers"] == {"q0": ["late but corrected"]}

    # The append is the close: the flag flips where the row lands, and the window
    # is shut in the revision path's own sentence.
    record = queue.find(ask_id)
    assert record is not None and record["delivered"] is True
    refused = queue.revise(ask_id, {"q0": ["no"]})
    assert refused["ok"] is False and refused["error"] == DELIVERED_REFUSAL


# ---------------------------------------------------------------------------
# the consumption bound: hand-off vs durable, the in-flight guard, retries
# (amended 2026-10-04 — the window is the row's DURABLE append)
# ---------------------------------------------------------------------------


def test_handoff_alone_leaves_the_window_open_until_the_row_is_durable(tmp_path: Path):
    """HAND-OFF IS NOT CONSUMPTION: with the append not yet run, the flag is
    False, a revision is accepted, and the append re-resolves the row it lands.

    ``persist_at_delivery=False`` is the real session's ordering held apart: the
    reconcile hands the message to the delivery path (``_handed`` set, nothing
    durable), and the append happens later. The stale handed message is then
    refreshed from the fold — the carry mechanism — and only the durable append
    closes the window.
    """
    session = FakeSession(persist_at_delivery=False)
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["no"]}, by="terminal")["ok"] is True
    row_id = store.response_row_id(ask_id)
    assert _run(queue.reconcile()) == [row_id]

    record = queue.find(ask_id)
    assert record is not None
    assert record["delivered"] is False, "hand-off is not consumption"
    accepted = queue.revise(ask_id, {"q0": ["yes"]})
    assert accepted["ok"] is True and accepted["revised"] is True

    # The deliver path re-resolves content AT the append: the preview built by
    # reconcile carries the old map, the refreshed rebuild carries the revision.
    preview = session.batches[-1][0]
    assert preview.details["answers"] == {"q0": ["no"]}
    refreshed = queue.refresh_delivery_message(preview)
    assert refreshed.id == preview.id, "one response row per ask, ever"
    assert refreshed.details["answers"] == {"q0": ["yes"]}

    # The append lands: durable, published, and the window shuts.
    queue.begin_row_commit(row_id)
    session.transcript.ids.add(row_id)
    queue.finish_row_commit(row_id, durable=True)
    record = queue.find(ask_id)
    assert record is not None and record["delivered"] is True
    refused = queue.revise(ask_id, {"q0": ["maybe"]})
    assert refused["ok"] is False and refused["error"] == DELIVERED_REFUSAL


def test_a_revision_while_the_append_is_in_flight_is_refused(tmp_path: Path):
    """THE COMMIT GUARD: refresh at T0, append awaiting, revision at T1>T0.

    The append has snapshotted content already, so accepting the revision would
    be accepted-and-then-dropped — the interleaving §10 forbids by name. It is
    refused in the delivered sentence (sub-ms, conservative), and the guard
    releases on EITHER append outcome: a failed append accepts the same revision
    on retry, a successful one refuses it for good.
    """
    session = FakeSession(persist_at_delivery=False)
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["no"]})["ok"] is True
    row_id = store.response_row_id(ask_id)

    queue.begin_row_commit(row_id)
    refused = queue.revise(ask_id, {"q0": ["yes"]})
    assert refused["ok"] is False and refused["error"] == DELIVERED_REFUSAL

    # The append FAILED: the guard releases and the same revision is accepted.
    queue.finish_row_commit(row_id, durable=False)
    accepted = queue.revise(ask_id, {"q0": ["yes"]})
    assert accepted["ok"] is True and accepted["revised"] is True

    # A successful append refuses again, permanently.
    queue.begin_row_commit(row_id)
    session.transcript.ids.add(row_id)
    queue.finish_row_commit(row_id, durable=True)
    refused = queue.revise(ask_id, {"q0": ["no"]})
    assert refused["ok"] is False and refused["error"] == DELIVERED_REFUSAL


def test_a_failed_append_releases_the_handoff_so_the_next_reconcile_retries(
    tmp_path: Path,
):
    """THE LIFECYCLE: ``_handed`` is scheduling, and it does not outlive its append.

    The pre-existing hole (it is fixed here rather than preserved): the set was
    sticky, so a FAILED append was never re-handed — the row a human was waiting
    for stayed undelivered with no further attempt. Now a failed append drops the
    entry, the next reconcile re-plans the row, and an interleaved revision is
    what the retry carries.
    """
    session = FakeSession(persist_at_delivery=False)
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["no"]})["ok"] is True
    row_id = store.response_row_id(ask_id)
    assert _run(queue.reconcile()) == [row_id]
    # The dedupe holds the gap closed while the hand-off is in flight.
    assert _run(queue.reconcile()) == []

    # The append fails...
    queue.begin_row_commit(row_id)
    queue.finish_row_commit(row_id, durable=False)
    # ...a revision lands while nothing is durable...
    assert queue.revise(ask_id, {"q0": ["yes"]})["ok"] is True
    # ...and the retry re-hands, carrying the revision.
    assert _run(queue.reconcile()) == [row_id]
    assert session.batches[-1][0].details["answers"] == {"q0": ["yes"]}


def test_a_revision_from_another_surface_supersedes_while_undelivered(tmp_path: Path):
    """A revision is NOT a race, so the single-winner surface rule is not a gate.

    Design §10: while the ask is undelivered a deliberate revision from ANY
    surface is accepted, and no layer may restore a surface gate as a safety
    measure. The winner rule still governs two plain ``respond``s (pinned above).
    """
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["no"]}, by="phone")["ok"] is True
    outcome = queue.revise(ask_id, {"q0": ["yes"]}, by="desktop")
    assert outcome["ok"] is True and outcome["revised"] is True
    record = queue.find(ask_id)
    assert record is not None
    assert record["answers"] == {"q0": ["yes"]}
    assert record["answered_by"] == {"surface": "phone"}, "the FIRST surface keeps it"


def test_successive_revisions_all_land_and_the_latest_is_effective(tmp_path: Path):
    """Successive revisions are legal while the window is open, and the log keeps
    the CHAIN: each event names the write it supersedes."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["one"]})["ok"] is True
    queue._now = lambda: BASE + 1
    assert queue.revise(ask_id, {"q0": ["two"]})["ok"] is True
    queue._now = lambda: BASE + 2
    assert queue.revise(ask_id, {"q0": ["three"]})["ok"] is True

    revised = [
        event
        for event in store.read_events(queue.session_dir)
        if event["kind"] == store.EVENT_REVISED
    ]
    assert [event["supersedes"] for event in revised] == [BASE, BASE + 1]
    record = queue.find(ask_id)
    assert record is not None and record["answers"] == {"q0": ["three"]}
    assert _run(queue.reconcile()) == [store.response_row_id(ask_id)]
    assert session.batches[-1][0].details["answers"] == {"q0": ["three"]}


def test_a_revision_before_any_answer_degrades_to_the_first_answer(tmp_path: Path):
    """An ``open``/``timed_out`` ask has no recorded answer to supersede, so the
    intent degrades: one ``answered`` event, no ``revised`` (design §10)."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    outcome = queue.revise(ask_id, {"q0": ["yes"]}, by="desktop")
    assert outcome["ok"] is True
    assert outcome["revised"] is False
    assert [event["kind"] for event in store.read_events(queue.session_dir)] == [
        store.EVENT_QUEUED,
        store.EVENT_ANSWERED,
    ]


def test_a_revision_of_a_settled_refusal_keeps_the_state_tables_sentence(tmp_path: Path):
    """``declined``/``dismissed``/``expired`` are NOT the delivered case: they keep
    the state table's own sentence, byte-for-byte what every other path says."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.decline(ask_id, by="terminal")["ok"] is True
    refused = queue.revise(ask_id, {"q0": ["yes"]})
    assert refused["ok"] is False
    assert refused["error"] == "you already declined this."


def test_a_revision_past_the_late_window_says_the_ask_expired(tmp_path: Path):
    now = BASE + store.LATE_WINDOW_S * 1000 + 200_000
    queue = _queue(tmp_path, FakeSession(), now=now)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: now
    assert queue.respond(ask_id, {"q0": ["late"]})["ok"] is False  # the existing refusal
    refused = queue.revise(ask_id, {"q0": ["late"]})
    assert refused["ok"] is False
    assert "expired" in refused["error"]


def test_a_revision_keeps_the_whole_ask_map_contract(tmp_path: Path):
    """The same complete-map rule as ``respond`` — one shared implementation, so a
    surface that forgot a key is refused here too rather than losing the question."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(2), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["a"], "q1": ["b"]})["ok"] is True
    refused = queue.revise(ask_id, {"q0": ["a"]})
    assert refused["ok"] is False
    assert "q1" in refused["error"]
    assert "empty list" in refused["error"]


def test_a_revision_cannot_carry_a_secret_value_into_the_log(tmp_path: Path):
    """MINOR 6 applies to the revision path too: the guard is in the shared map
    contract, so a raw value in a secret cell is replaced whole, not filtered."""
    sentinel = "sk-live-do-not-persist"
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(1, secret=True), 120)["details"]["ask_id"]
    assert queue.respond(ask_id, {"key-0": ["KEY-0"]}, by="terminal")["ok"] is True
    assert queue.revise(ask_id, {"key-0": [sentinel]}, by="cli")["ok"] is True
    log = store.asks_log_path(store.session_dir(tmp_path, "s1")).read_text()
    assert sentinel not in log
    record = queue.find(ask_id)
    assert record is not None and record["answers"]["key-0"] == ["<not provided>"]


def test_a_plain_repeat_respond_keeps_its_exact_refusal(tmp_path: Path):
    """The one-way rule SURVIVES the amendment: a second ``respond`` — even with a
    different map — is still refused in the sentence it always used. This is the
    pin design §10 promises: a repeat tap is a retry, not a change of mind."""
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["no"]}, by="terminal")["ok"] is True
    repeat = queue.respond(ask_id, {"q0": ["yes"]}, by="phone")
    assert repeat["ok"] is False
    assert repeat["error"] == "already answered by terminal."
    assert [event["kind"] for event in store.read_events(queue.session_dir)] == [
        store.EVENT_QUEUED,
        store.EVENT_ANSWERED,
    ]


def test_a_refused_revision_cannot_buy_a_second_row_or_a_second_turn(tmp_path: Path):
    """A revision can never add a second row or a second turn (design §10). Once
    delivered, the refusal is the whole effect and reconcile stays silent."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.respond(ask_id, {"q0": ["no"]})["ok"] is True
    delivered = _run(queue.reconcile())
    assert delivered == [store.response_row_id(ask_id)]
    assert queue.revise(ask_id, {"q0": ["yes"]})["ok"] is False
    assert _run(queue.reconcile()) == []
    assert len([m for batch in session.batches for m in batch]) == 1


def test_the_session_revision_op_refuses_in_words_without_a_queue() -> None:
    """THE KILL SWITCH (design §10): with ``LOP_ASK_NONBLOCKING=0`` a real client
    never revises — the old picker submits once — and a stray op must answer in
    words rather than with a traceback or a silent success.

    Called unbound on a stub that has no queue, which is exactly the seam
    ``Session.respond_ask`` guards and the only state the guard reads.
    """
    from local_operator.session.session import Session

    class _NoQueue:
        def ask_queue(self) -> None:
            return None

    outcome = Session.revise_ask(cast(Any, _NoQueue()), "a-1", {"q0": ["yes"]})
    assert outcome["ok"] is False
    assert "predates queued asks" in outcome["error"]


def test_the_session_withdraw_op_refuses_in_words_without_a_queue() -> None:
    """THE KILL SWITCH (design §12), the same seam as the revision op: the
    tool is not mounted on the blocking arm, and a stray call — a host script,
    a stale client route — must answer in words rather than a traceback or a
    silent success."""
    from local_operator.session.session import Session

    class _NoQueue:
        def ask_queue(self) -> None:
            return None

    outcome = Session.withdraw_ask(cast(Any, _NoQueue()), "a-1", reason="moot")
    assert outcome["ok"] is False
    assert "predates queued asks" in outcome["error"]


def test_the_session_withdraw_op_forwards_the_whole_call() -> None:
    """The wrapper passes reason, answers, message_id and by through unchanged:
    the refusals and the append are the queue's, and a second translation here
    is how the two would drift."""
    from local_operator.session.session import Session

    captured: dict[str, Any] = {}

    class _Queue:
        def withdraw(self, ask_id: str, **kwargs: Any) -> dict[str, Any]:
            captured.update(ask_id=ask_id, **kwargs)
            return {"ok": True, "text": "x", "details": {}}

    class _Session:
        def ask_queue(self) -> _Queue:
            return _Queue()

    outcome = Session.withdraw_ask(
        cast(Any, _Session()),
        "a-1",
        reason="answered_in_chat",
        answers={"q0": ["words"]},
        message_id="m-1",
    )
    assert outcome["ok"] is True
    assert captured == {
        "ask_id": "a-1",
        "reason": "answered_in_chat",
        "answers": {"q0": ["words"]},
        "message_id": "m-1",
        "by": "agent",
    }


# ---------------------------------------------------------------------------
# projection and the derived index
# ---------------------------------------------------------------------------


def test_the_projection_puts_open_asks_first_and_drops_expired(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    first = queue.enqueue(_questions(text="first"), None)["details"]["ask_id"]
    second = queue.enqueue(_questions(text="second"), None)["details"]["ask_id"]
    queue.respond(first, {"q0": ["yes"]})
    rows = queue.projection()
    assert [row["ask_id"] for row in rows] == [second, first]
    assert rows[0]["status"] == store.STATUS_OPEN
    assert rows[1]["status"] == store.STATUS_ANSWERED


def test_the_index_is_written_and_read_back(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    entry = store.read_entry(tmp_path, "s1")
    assert entry is not None
    assert [row["ask_id"] for row in entry["asks"]] == [ask_id]
    assert entry["cwd"] == ""


def test_the_index_disappears_when_every_ask_is_settled_and_old(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    queue.enqueue(_questions(), 120)
    queue.respond(queue.records()[0]["ask_id"], {"q0": ["yes"]})
    queue._now = lambda: BASE + store.LATE_WINDOW_S * 1000 + 200_000
    queue._refresh()
    entry = store.read_entry(tmp_path, "s1")
    # The projection still carries the (now `expired`) ask only while it is
    # within the window; past it the entry is empty and is removed.
    assert entry is None


def test_the_timer_is_one_task_and_stops_when_nothing_is_open(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)

    async def scenario():
        queue.arm()
        first = queue._timer
        assert first is not None
        queue.arm()
        assert queue._timer is first  # still ONE task
        queue.dispose()
        assert queue._disposed is True

    _run(scenario())


def test_the_timer_does_not_arm_without_a_running_loop(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    queue.arm()  # no loop here — must not raise
    assert queue._timer is None


def test_dispose_is_idempotent(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    queue.dispose()
    queue.dispose()
    assert queue._disposed is True


def test_the_queue_never_touches_the_operator_config_dir(tmp_path: Path):
    """A guard on the fixture itself: every test here runs against ``tmp_path``,
    so a regression that reached for the live store would fail loudly."""
    queue = _queue(tmp_path, FakeSession())
    assert str(tmp_path) in str(queue.session_dir)
    assert queue.session_dir.name == "s1"


def test_now_defaults_to_wall_clock_when_no_clock_is_injected(tmp_path: Path):
    queue = AskQueue(FakeSession(), config_dir=tmp_path, session_id="s1")
    assert abs(queue._now() - int(time.time() * 1000)) < 5_000


@pytest.mark.parametrize("count", [1, 3])
def test_every_question_reaches_the_projection(tmp_path: Path, count: int):
    queue = _queue(tmp_path, FakeSession())
    queue.enqueue(_questions(count), None)
    row = queue.projection()[0]
    assert len(row["questions"]) == count
    assert all("question" in question for question in row["questions"])


# ---------------------------------------------------------------------------
# a timed-out ask: still shown, already settled (#2388)
# ---------------------------------------------------------------------------


def test_the_projection_keeps_a_timed_out_ask_but_counts_it_settled(tmp_path: Path):
    """A queue whose ONLY ask timed out keeps the row and owes nothing.

    The row stays (a settled receipt the user can dismiss), but it is not
    outstanding: the agent was told at the deadline and moved on (#2388). The
    queue still takes a late answer, which folds to ``late``.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 120_001
    rows = queue.projection()
    assert [row["ask_id"] for row in rows] == [ask_id]
    assert rows[0]["status"] == store.STATUS_TIMED_OUT
    assert store.outstanding_asks(rows) == []

    assert queue.respond(ask_id, {"q0": ["late answer"]}, by="phone")["ok"] is True
    late = queue.projection()
    assert late[0]["status"] == store.STATUS_LATE
    assert store.outstanding_asks(late) == []


def test_the_index_keeps_a_timed_out_only_queue_rather_than_going_absent(tmp_path: Path):
    """The derived index must not drop to absence for a timed-out-only queue.

    That file is the COLD reader's view (the aggregate route, the phone's list),
    and a stopped session's timed-out row must stay in it although it is no
    longer outstanding (#2388): the row is where Dismiss lives.
    """
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 120_001
    queue._refresh()
    entry = store.read_entry(tmp_path, "s1")
    assert entry is not None
    assert [row["ask_id"] for row in entry["asks"]] == [ask_id]
    assert entry["asks"][0]["status"] == store.STATUS_TIMED_OUT
    assert store.outstanding_asks(entry["asks"]) == []
    # ``now`` is injected so the reader's staleness sweep is judged at the SAME
    # instant the fold was (the fixture clock is years behind the wall clock).
    rows = store.index_asks(tmp_path, now=BASE + 120_001)
    assert [row["ask_id"] for row in rows] == [ask_id]


# ---------------------------------------------------------------------------
# The lapsed-visibility window (decisions D8, D9: `asks.lapsed_visible_for`)
# ---------------------------------------------------------------------------

HOUR_MS = 3600 * 1000


def test_projection_hides_a_lapsed_row_past_the_window_but_the_index_keeps_it(tmp_path: Path):
    """The window is a DISPLAY rule (D6): the live projection drops the row,
    while ``_refresh`` keeps writing the index at the 7 d horizon so a stopped
    session's row comes back if the operator widens the setting again."""
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    expires_at = BASE + 120_000
    queue._now = lambda: expires_at + HOUR_MS + 1
    assert queue.projection(lapsed_visible_ms=HOUR_MS) == []
    assert [r["ask_id"] for r in queue.projection(lapsed_visible_ms=2 * HOUR_MS)] == [ask_id]
    # No keyword is today's 7 d result.
    assert [r["ask_id"] for r in queue.projection()] == [ask_id]
    queue._refresh()
    entry = store.read_entry(tmp_path, "s1")
    assert entry is not None and [r["ask_id"] for r in entry["asks"]] == [ask_id]


def test_projection_keeps_answered_rows_whatever_the_window(tmp_path: Path):
    session = FakeSession()
    queue = _queue(tmp_path, session)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue.respond(ask_id, {"q0": ["yes"]})
    queue._now = lambda: BASE + 120_000 + HOUR_MS
    assert [r["ask_id"] for r in queue.projection(lapsed_visible_ms=0)] == [ask_id]


def test_the_timer_wakes_at_expires_at_plus_the_window(tmp_path: Path, monkeypatch):
    """ "Gone after 1 h" must be true with no new turn (D9 ii): the deadline
    timer, which used to stop once nothing was ``open``, also sleeps to the
    earliest ``expires_at + window`` among lapsed rows and republishes then."""
    from local_operator.asks import queue as queue_module

    clock = {"now": BASE}
    session = FakeSession()
    published: list[int] = []
    session.publish_ask_state = lambda: published.append(clock["now"])  # type: ignore[attr-defined]
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: clock["now"])
    monkeypatch.setattr(queue_module, "_lapsed_visible_ms", lambda _dir: HOUR_MS)
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    expires_at = BASE + 120_000
    # Past the deadline: the row is lapsed, nothing is ``open`` any more.
    clock["now"] = expires_at + 1
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        # Each sleep lands exactly where the timer aimed it.
        clock["now"] += int(round(seconds * 1000))

    monkeypatch.setattr(queue_module.asyncio, "sleep", fake_sleep)

    async def scenario() -> None:
        await queue._run_timer()

    _run(scenario())
    assert sleeps, "the timer returned without waiting for the lapsed row's window"
    # Ticks are bounded by MAX_TICK_MS, so the SUM is the wake instant: the
    # first millisecond PAST the window, the horizon's own strict ``>`` rule.
    hidden_at = expires_at + HOUR_MS + 1
    assert clock["now"] == hidden_at
    assert all(s * 1000 <= policy.MAX_TICK_MS for s in sleeps)
    assert published and published[-1] == hidden_at
    assert queue.projection(lapsed_visible_ms=HOUR_MS) == []
    assert [r["ask_id"] for r in queue.projection(lapsed_visible_ms=HOUR_MS + 1)] == [ask_id]


def test_the_timer_does_not_wait_on_a_lapsed_row_already_hidden(tmp_path: Path, monkeypatch):
    from local_operator.asks import queue as queue_module

    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE)
    monkeypatch.setattr(queue_module, "_lapsed_visible_ms", lambda _dir: 0)
    queue.enqueue(_questions(), 120)
    queue._now = lambda: BASE + 120_001
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(queue_module.asyncio, "sleep", fake_sleep)
    _run(queue._run_timer())
    assert sleeps == []


@pytest.mark.parametrize("with_open", [True, False], ids=["open+lapsed", "lapsed-only"])
def test_rearm_restarts_a_sleeping_timer_on_the_new_window(
    tmp_path: Path, monkeypatch, with_open: bool
):
    """QA round 1, Q2: a window narrowed while the timer sleeps. ``rearm`` must
    replace the sleeping task so the next wake is computed with the NEW window;
    ``arm`` alone kept the old task (and so the old delay), which hid the row
    40 s late with an open ask present and never with lapsed rows only."""
    from local_operator.asks import queue as queue_module

    window = {"ms": 7 * 24 * 3_600_000}
    monkeypatch.setattr(queue_module, "_lapsed_visible_ms", lambda _dir: window["ms"])
    clock = {"now": BASE}
    session = FakeSession()
    published: list[int] = []
    session.publish_ask_state = lambda: published.append(clock["now"])  # type: ignore[attr-defined]
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: clock["now"])
    queue.enqueue(_questions(), 120)
    if with_open:
        queue.enqueue(_questions(text="still open"), 86_400)
    expires_at = BASE + 120_000
    # 59 m 40 s past the lapsed row's deadline: under a 1 h window it hides 20 s on.
    clock["now"] = expires_at + 3_600_000 - 20_000
    hide_at = expires_at + 3_600_000 + 1

    async def scenario() -> None:
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds: float) -> None:
            # The clock moves by what the timer ASKED for; yield so a cancel lands.
            clock["now"] += int(round(seconds * 1000))
            await real_sleep(0)

        monkeypatch.setattr(queue_module.asyncio, "sleep", fake_sleep)
        queue.arm()
        first = queue._timer
        assert first is not None
        await real_sleep(0)  # the task is now asleep on the 7 d window's delay
        window["ms"] = 3_600_000
        clock["now"] = expires_at + 3_600_000 - 20_000
        published.clear()
        queue.rearm()
        assert queue._timer is not None and queue._timer is not first
        assert first.cancelling() or first.done()
        for _ in range(200):
            await real_sleep(0)
            if published and published[-1] >= hide_at:
                break
        queue.dispose()

    _run(scenario())
    assert published, "the re-armed timer never republished"
    assert published[-1] == hide_at
    lapsed = [r for r in queue.projection(lapsed_visible_ms=3_600_000) if r["status"] != "open"]
    assert lapsed == []


def test_a_lapsed_only_wait_does_not_refold_the_log_every_tick(tmp_path: Path, monkeypatch):
    """Review round 1, R1-4: with lapsed rows only, nothing but the clock can
    move, so the timer waits to the window edge in bounded ticks WITHOUT a log
    read and fold per tick — and an enqueue (which goes through ``arm``) still
    sends it back to re-plan for the new, earlier deadline."""
    from local_operator.asks import queue as queue_module

    monkeypatch.setattr(queue_module, "_lapsed_visible_ms", lambda _dir: HOUR_MS)
    clock = {"now": BASE}
    session = FakeSession()
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: clock["now"])
    queue.enqueue(_questions(), 120)
    clock["now"] = BASE + 120_000 + 1  # lapsed; its 1 h window edge is ~60 ticks away
    folds = {"n": 0}
    real_records = queue.records

    def counting_records(now_ms=None):
        folds["n"] += 1
        return real_records(now_ms)

    monkeypatch.setattr(queue, "records", counting_records)

    async def scenario() -> None:
        real_sleep = asyncio.sleep
        ticks = {"n": 0}

        async def fake_sleep(seconds: float) -> None:
            ticks["n"] += 1
            clock["now"] += int(round(seconds * 1000))
            if ticks["n"] == 30:
                # Mid-wait, a fresh ask arrives through the real path.
                queue.enqueue(_questions(text="new"), 120)
            await real_sleep(0)
            if ticks["n"] >= 40:
                queue.dispose()

        monkeypatch.setattr(queue_module.asyncio, "sleep", fake_sleep)
        queue.arm()
        task = queue._timer
        assert task is not None
        # ``dispose`` cancels the task at tick 40; the wait ends either way.
        await asyncio.gather(task, return_exceptions=True)
        assert ticks["n"] >= 30

    _run(scenario())
    # 30 lapsed-only ticks cost ONE plan fold, not 30; the enqueue's re-plan and
    # the open ask's every-tick loop account for the rest.
    assert folds["n"] < 20, folds


def test_rearm_never_cancels_a_reconcile_mid_hand_off(tmp_path: Path, monkeypatch):
    """Review round 2, R2-2: ``rearm`` must not cancel the timer while its
    ``reconcile`` is between adding a row id to ``_handed`` and handing the row
    to delivery. A cancel there stranded the id in ``_handed`` (only
    ``finish_row_commit`` removes it), so every later reconcile treated the
    timeout notice as already planned and it was never delivered. Here the
    delivery suspends, ``rearm`` lands mid-hand-off, and the row must still
    reach delivery exactly once."""
    from local_operator.asks import queue as queue_module

    monkeypatch.setattr(queue_module, "_lapsed_visible_ms", lambda _dir: HOUR_MS)
    clock = {"now": BASE}
    session = FakeSession()
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: clock["now"])
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    timeout_row = store.timeout_row_id(ask_id)
    clock["now"] = BASE + 120_000 + 1  # the deadline has passed: a notice is owed

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        real_deliver = session.deliver_ask_messages

        async def suspending_deliver(messages) -> None:
            entered.set()
            await release.wait()  # the hand-off is in flight here
            await real_deliver(messages)

        session.deliver_ask_messages = suspending_deliver  # type: ignore[method-assign]
        task: asyncio.Task[Any] = asyncio.ensure_future(queue.reconcile())
        queue._timer = cast("asyncio.Task[None]", task)  # the timer runs this reconcile
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert timeout_row in queue._handed
        queue.rearm()  # a window change lands mid-hand-off
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        queue.dispose()

    _run(scenario())
    delivered = [m.id for batch in session.batches for m in batch]
    assert delivered == [timeout_row], delivered
