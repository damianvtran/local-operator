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
