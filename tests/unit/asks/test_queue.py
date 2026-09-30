"""``AskQueue`` end to end against a small session double: caps, the fold, and
the delivery rules (design ``docs/design/ask-nonblocking.md`` §2.2/§2.3).

The session double is deliberately tiny — a transcript id set, a delivery sink
and the two hooks the queue uses — because the queue's contract with the session
is exactly those three things. Everything else it needs it takes from the log.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy, store
from local_operator.asks.queue import AskQueue

BASE = 1_700_000_000_000


class FakeTranscript:
    def __init__(self) -> None:
        self.ids: set[str] = set()

    def has_entry(self, entry_id: str) -> bool:
        return entry_id in self.ids


class FakeSession:
    """The three things ``AskQueue`` asks of a session, and nothing else."""

    def __init__(self) -> None:
        self.transcript = FakeTranscript()
        self.batches: list[list[Any]] = []
        self.reach: list[str] = []
        self.spawned: list[asyncio.Task[Any]] = []

    def ask_reach(self) -> list[str]:
        return self.reach

    def _spawn_background(self, coro):
        task = asyncio.ensure_future(coro)
        self.spawned.append(task)
        return task

    async def deliver_ask_messages(self, messages) -> None:
        # The durable row IS the delivery marker, so the double marks it here —
        # which is exactly what the session does by persisting the message.
        self.batches.append(list(messages))
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


def test_only_a_timed_out_ask_can_be_dismissed(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    outcome = queue.dismiss(ask_id)
    assert outcome["ok"] is False
    assert "still open" in outcome["error"]


def test_decline_is_terminal_on_write(tmp_path: Path):
    queue = _queue(tmp_path, FakeSession())
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    assert queue.decline(ask_id, by="terminal")["ok"] is True
    declined = queue.find(ask_id)
    assert declined is not None and declined["status"] == store.STATUS_DECLINED
    assert queue.respond(ask_id, {"q0": ["yes"]})["ok"] is False


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
    session = FakeSession()
    queue = _queue(tmp_path, session, now=BASE + 200_000)
    queue._now = lambda: BASE
    ask_id = queue.enqueue(_questions(), 120)["details"]["ask_id"]
    queue._now = lambda: BASE + 200_000
    assert queue.dismiss(ask_id, by="terminal")["ok"] is True
    assert _run(queue.reconcile()) == []
    assert session.batches == []


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
