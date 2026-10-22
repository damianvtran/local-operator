"""Ask RELAYS (#2520): a watcher points at another session's pending ask.

The design of record is ``docs/design/ask-nonblocking.md``; the pointer shape is
decided there and in the issue — W's log holds a REFERENCE, S's log stays the
single source of truth, and W's row status is DERIVED from S's fold. These cells
pin the consequences: one answer settles both, every refusal is reachable, and an
unresolvable pointer settles instead of lingering.

The double is the tiny one ``test_queue.py`` uses — a transcript id set, a
delivery sink and the two hooks the queue takes from a session — because the
queue's contract with a session is exactly those things.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy, render, store
from local_operator.asks.queue import AskQueue

BASE = 1_700_000_000_000
ORIGIN = "s-origin"
WATCH = "s-watch"


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
        self.spawned: list[asyncio.Task[Any]] = []

    def credential_names(self) -> list[str]:
        return []

    def ask_reach(self) -> list[str]:
        return ["terminal"]

    def _spawn_background(self, coro):
        task = asyncio.ensure_future(coro)
        self.spawned.append(task)
        return task

    async def deliver_ask_messages(self, messages) -> None:
        self.batches.append(list(messages))
        for message in messages:
            self.transcript.ids.add(message.id)


def _questions(text: str = "Ship it?", *, secret: bool = False):
    return [
        {
            "id": "key-0" if secret else "q0",
            "question": text,
            "options": [],
            "multi": False,
            "secret": secret,
            "persist": False,
            "recommended": None,
        }
    ]


def _pair(
    tmp_path: Path, *, now: int = BASE
) -> tuple[AskQueue, AskQueue, FakeSession, FakeSession]:
    """The origin queue and the watcher queue over ONE config dir.

    Two clocks rather than one: the cells that move time (the deadline ones) need
    the origin's ``now`` and the watcher's ``now`` to be the same value, and the
    cleanest way to say that is to set both from the test.
    """
    origin_session, watch_session = FakeSession(), FakeSession()
    origin = AskQueue(origin_session, config_dir=tmp_path, session_id=ORIGIN, clock=lambda: now)
    watch = AskQueue(watch_session, config_dir=tmp_path, session_id=WATCH, clock=lambda: now)
    return origin, watch, origin_session, watch_session


def _run(coro):
    return asyncio.run(coro)


def _set_now(origin: AskQueue, watch: AskQueue, now: int) -> None:
    origin._now = lambda: now
    watch._now = lambda: now


def _origin_ask(
    origin: AskQueue, text: str = "Ship it?", *, timeout: int | None = None, **kwargs
) -> str:
    outcome = origin.enqueue(_questions(text, **kwargs), timeout)
    assert outcome["ok"] is True, outcome
    return str(outcome["details"]["ask_id"])


def _relay(watch: AskQueue, origin_ask: str, *, origin: str = ORIGIN) -> dict[str, Any]:
    return watch.enqueue_relay(f"{origin}/{origin_ask}")


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_a_relay_writes_one_row_in_each_log(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    outcome = _relay(watch, origin_ask)
    assert outcome["ok"] is True
    assert outcome["details"]["relay"] == {
        "session_id": ORIGIN,
        "ask_id": origin_ask,
        "origin_status": store.STATUS_OPEN,
    }
    relay_id = outcome["details"]["ask_id"]
    assert relay_id != origin_ask
    assert [event["kind"] for event in store.read_events(watch.session_dir)] == [
        store.EVENT_RELAYED
    ]
    origin_events = store.read_events(origin.session_dir)
    assert [event["kind"] for event in origin_events] == [
        store.EVENT_QUEUED,
        store.EVENT_RELAY_SURFACED,
    ]
    surfaced = origin_events[-1]
    assert surfaced["ask_id"] == origin_ask
    assert surfaced["by"] == {"session_id": WATCH, "ask_id": relay_id}


def test_a_relay_row_derives_the_origin_questions_verbatim_and_its_deadline(tmp_path: Path):
    """The watcher cannot reword the question or move the deadline: both are the
    origin's, read from its fold on every refresh."""
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin, "Which database?", timeout=120)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    origin_record = origin.find(origin_ask)
    relay_record = watch.find(relay_id)
    assert origin_record is not None and relay_record is not None
    assert relay_record["questions"] == origin_record["questions"]
    assert relay_record["expires_at"] == origin_record["expires_at"]
    assert relay_record["timeout_s"] == origin_record["timeout_s"] == 120
    # A 2-minute window is the URGENT band (policy.is_urgent): urgency is the
    # origin's too, never re-derived from the watcher's own clock.
    assert relay_record["urgent"] is origin_record["urgent"] is True
    assert relay_record["status"] == origin_record["status"] == store.STATUS_OPEN
    # The origin learns where else it is shown, and the ORIGIN's row is otherwise
    # untouched: ``surfaced_in`` is additive, not a second status source.
    refreshed = origin.find(origin_ask)
    assert refreshed is not None
    assert refreshed["surfaced_in"] == [
        {
            "session_id": WATCH,
            "ask_id": relay_id,
            "at": BASE,
        }
    ]


def test_ordinary_ask_rows_carry_no_relay_keys(tmp_path: Path):
    """ABSENT, never empty: an ask that never met a relay keeps the same row."""
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    before = store.pending_row(origin.find(origin_ask) or {})
    assert "relay" not in before
    assert "surfaced_in" not in before
    _relay(watch, origin_ask)
    after = store.pending_row(origin.find(origin_ask) or {})
    assert "relay" not in after
    assert "surfaced_in" in after


# ---------------------------------------------------------------------------
# One answer settles both
# ---------------------------------------------------------------------------


def test_answering_the_relay_settles_the_origin(tmp_path: Path):
    origin, watch, origin_session, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    outcome = watch.respond(relay_id, {"q0": ["yes"]}, by="desktop")
    assert outcome["ok"] is True
    events = store.read_events(origin.session_dir)
    assert [event["kind"] for event in events].count(store.EVENT_ANSWERED) == 1
    origin_record = origin.find(origin_ask)
    relay_record = watch.find(relay_id)
    assert origin_record is not None and relay_record is not None
    assert origin_record["status"] == store.STATUS_ANSWERED
    assert origin_record["answers"] == {"q0": ["yes"]}
    assert relay_record["status"] == store.STATUS_ANSWERED
    assert relay_record["answers"] == {"q0": ["yes"]}
    assert relay_record["relay"]["origin_status"] == store.STATUS_ANSWERED
    # Nothing was appended to the watcher's log by the answer: the pointer has no
    # answer of its own to record, so the relay log still holds exactly one row.
    assert [event["kind"] for event in store.read_events(watch.session_dir)] == [
        store.EVENT_RELAYED
    ]
    # The ORIGIN's model is the one that gets the response row.
    _run(origin.reconcile())
    assert store.response_row_id(origin_ask) in origin_session.transcript.ids


def test_answering_the_origin_settles_the_relay(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert origin.respond(origin_ask, {"q0": ["yes"]}, by="phone")["ok"] is True
    relay_record = watch.find(relay_id)
    assert relay_record is not None
    assert relay_record["status"] == store.STATUS_ANSWERED
    assert relay_record["answered_by"] == {"surface": "phone"}


def test_a_second_answer_on_either_copy_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert watch.respond(relay_id, {"q0": ["yes"]}, by="desktop")["ok"] is True
    second = watch.respond(relay_id, {"q0": ["no"]}, by="desktop")
    assert second["ok"] is False
    # The origin's refusal wording is KEPT, behind the lead that says which ask
    # refused: a tap on the watcher's card is decided by the origin's log.
    assert "already answered by desktop" in second["error"]
    assert f"ask {origin_ask} in session {ORIGIN}" in second["error"]
    direct = origin.respond(origin_ask, {"q0": ["no"]}, by="desktop")
    assert direct["ok"] is False
    assert "already answered by desktop" in direct["error"]
    events = store.read_events(origin.session_dir)
    assert [event["kind"] for event in events].count(store.EVENT_ANSWERED) == 1


def test_declining_the_relay_settles_the_origin(tmp_path: Path):
    """Decline is the other TERMINAL settle (#2673, R1-1).

    It must land on the ORIGIN's log like an answer does: a local ``declined``
    row is re-derived away on the next fold, so the card would come back and the
    user's decision would be silently dropped.
    """
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert watch.decline(relay_id, by="terminal")["ok"] is True
    origin_record = origin.find(origin_ask)
    relay_record = watch.find(relay_id)
    assert origin_record is not None and relay_record is not None
    assert origin_record["status"] == store.STATUS_DECLINED
    assert relay_record["status"] == store.STATUS_DECLINED
    # One decision, one row, in the log that owns it — the watcher's log holds
    # the pointer and nothing else.
    assert [event["kind"] for event in store.read_events(watch.session_dir)] == [
        store.EVENT_RELAYED
    ]
    assert [event["kind"] for event in store.read_events(origin.session_dir)].count(
        store.EVENT_DECLINED
    ) == 1


def test_declining_a_settled_origin_through_the_relay_keeps_its_words(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert watch.respond(relay_id, {"q0": ["yes"]}, by="desktop")["ok"] is True
    outcome = watch.decline(relay_id, by="terminal")
    assert outcome["ok"] is False
    assert "already answered by desktop" in outcome["error"]
    assert f"ask {origin_ask} in session {ORIGIN}" in outcome["error"]


# ---------------------------------------------------------------------------
# Withdrawal, timeout, and the origin vanishing
# ---------------------------------------------------------------------------


def test_an_origin_withdrawal_settles_the_relay_as_withdrawn(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert origin.withdraw(origin_ask, reason="moot")["ok"] is True
    relay_record = watch.find(relay_id)
    assert relay_record is not None
    assert relay_record["status"] == store.STATUS_WITHDRAWN


def test_a_watcher_withdrawal_leaves_the_origin_log_byte_identical(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    log = store.asks_log_path(origin.session_dir)
    before = log.read_bytes()
    outcome = watch.withdraw(relay_id, reason="moot")
    assert outcome["ok"] is True
    assert log.read_bytes() == before
    relay_record = watch.find(relay_id)
    assert relay_record is not None
    assert relay_record["status"] == store.STATUS_WITHDRAWN
    assert watch.find(relay_id)["relay_withdrawn_at"] == BASE  # type: ignore[index]
    assert origin.find(origin_ask)["status"] == store.STATUS_OPEN  # type: ignore[index]


def test_a_watcher_withdrawal_folds_cold_as_withdrawn(tmp_path: Path):
    """R1-3: the withdrawn row is folded, not just stamped (#2673).

    Every reader that is not the live queue — the cleanup probe, the cold
    dismissal door — sees the status the watcher's own log states. Without the
    fold a retracted relay reads ``open`` forever and its session directory is
    never prunable.
    """
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    from local_operator.session.cleanup import _has_open_asks

    # Before: the pointer is outstanding, and the probe is honest about it.
    assert _has_open_asks(watch.session_dir) is True
    assert watch.withdraw(relay_id, reason="moot")["ok"] is True
    cold = store.fold(store.read_events(watch.session_dir), BASE, session_id=WATCH)
    assert [(row["ask_id"], row["status"]) for row in cold] == [(relay_id, store.STATUS_WITHDRAWN)]
    assert _has_open_asks(watch.session_dir) is False


def test_a_watcher_dismissal_of_a_lapsed_relay_sticks_on_both_doors(tmp_path: Path):
    """R1-2: a dismissal is a LOCAL view act, and the closed card stays closed.

    The live door accepts the tap on a lapsed relay (its derived status is
    ``timed_out``); the fold then applies the ``dismissed`` row, so the same card
    does not return — and the cold door reads that row from the same log, not a
    re-opened relay.
    """
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin, timeout=120)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    _set_now(origin, watch, BASE + 300_000)
    assert watch.find(relay_id)["status"] == store.STATUS_TIMED_OUT  # type: ignore[index]
    assert watch.dismiss(relay_id, by="terminal")["ok"] is True
    assert watch.find(relay_id)["status"] == store.STATUS_DISMISSED  # type: ignore[index]
    cold = store.fold(store.read_events(watch.session_dir), BASE + 300_000, session_id=WATCH)
    assert [row["status"] for row in cold] == [store.STATUS_DISMISSED]
    assert render.dismiss_refusal(cold[0]) == render.DISMISS_ALREADY_DISMISSED
    # The origin is untouched: a dismissal is a view act on the watcher's card.
    assert origin.find(origin_ask)["status"] == store.STATUS_TIMED_OUT  # type: ignore[index]


def test_an_origin_timeout_and_a_late_relay_answer_settle_both_as_late(tmp_path: Path):
    """The #2505 constraint: a lapsed ask is still answerable, and the LATE answer
    recorded on the origin is what the watcher's derived row must show."""
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin, timeout=120)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    late = BASE + 300_000
    _set_now(origin, watch, late)
    relay_record = watch.find(relay_id)
    assert relay_record is not None
    assert relay_record["status"] == store.STATUS_TIMED_OUT
    outcome = watch.respond(relay_id, {"q0": ["after the deadline"]}, by="desktop")
    assert outcome["ok"] is True
    origin_record = origin.find(origin_ask)
    relay_record = watch.find(relay_id)
    assert origin_record is not None and relay_record is not None
    assert origin_record["status"] == store.STATUS_LATE
    assert relay_record["status"] == store.STATUS_LATE
    assert relay_record["answers"] == {"q0": ["after the deadline"]}


def test_an_origin_log_that_disappears_settles_the_relay(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    store.asks_log_path(origin.session_dir).unlink()
    relay_record = watch.find(relay_id)
    assert relay_record is not None
    assert relay_record["status"] == store.STATUS_EXPIRED
    assert relay_record["relay"]["origin_gone"] is True
    assert store.is_outstanding(relay_record["status"]) is False
    # A settled pointer is not work owed: the wire's list must not show it.
    assert [row["ask_id"] for row in watch.projection()] == []
    # ... and the watcher is told its pointer went nowhere.
    _run(watch.reconcile())
    assert [message.id for message in watch._session.batches[-1]] == [store.relay_row_id(relay_id)]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_relaying_the_same_origin_ask_twice_is_refused(tmp_path: Path):
    """R1-4: one ref, one card, one surfaced note.

    A second relay of the same origin ask used to mint a second pointer — two
    cards for one question, both settling from the same log. A relay the user
    CLOSED locally is not held any more, so a deliberate re-raise still works.
    """
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    first = _relay(watch, origin_ask)
    assert first["ok"] is True
    second = _relay(watch, origin_ask)
    assert second["ok"] is False
    assert second["error"] == render.RELAY_ALREADY_HELD
    assert len([row for row in watch.records() if row.get("relay")]) == 1
    assert [event["kind"] for event in store.read_events(origin.session_dir)].count(
        store.EVENT_RELAY_SURFACED
    ) == 1
    # Closed locally, the ref is free again: the held check reads derived status.
    first_id = str(first["details"]["ask_id"])
    assert watch.withdraw(first_id, reason="moot")["ok"] is True
    assert _relay(watch, origin_ask)["ok"] is True


@pytest.mark.parametrize("ref", ["", "not-a-ref", "a/b/c", "/a-bc12", "s-origin/", " / "])
def test_a_malformed_ref_is_refused_with_its_own_sentence(tmp_path: Path, ref: str):
    _, watch, _, _ = _pair(tmp_path)
    outcome = watch.enqueue_relay(ref)
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_BAD_FORM
    assert store.read_events(watch.session_dir) == []


def test_an_unknown_origin_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    missing_session = watch.enqueue_relay(f"no-such-session/{origin_ask}")
    assert missing_session["ok"] is False
    assert missing_session["error"] == render.RELAY_ORIGIN_UNKNOWN
    missing_ask = watch.enqueue_relay(f"{ORIGIN}/a-0000")
    assert missing_ask["ok"] is False
    assert missing_ask["error"] == render.RELAY_ORIGIN_UNKNOWN


def test_a_settled_origin_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    assert origin.respond(origin_ask, {"q0": ["yes"]}, by="desktop")["ok"] is True
    outcome = _relay(watch, origin_ask)
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_ORIGIN_SETTLED


def test_a_relay_of_a_relay_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    third_session = FakeSession()
    third = AskQueue(third_session, config_dir=tmp_path, session_id="s-third", clock=lambda: BASE)
    outcome = third.enqueue_relay(f"{WATCH}/{relay_id}")
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_OF_RELAY


def test_a_secret_origin_is_refused_outright(tmp_path: Path):
    """SECURITY: a credential must never be routable through a watcher's store."""
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin, secret=True)
    outcome = _relay(watch, origin_ask)
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_SECRET_ORIGIN
    assert store.read_events(watch.session_dir) == []
    assert [event["kind"] for event in store.read_events(origin.session_dir)] == [
        store.EVENT_QUEUED
    ]


def test_relaying_this_sessions_own_ask_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    # The origin session trying to relay itself: answer it where it was asked.
    outcome = origin.enqueue_relay(f"{ORIGIN}/{origin_ask}")
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_SELF


def test_the_relay_cap_is_separate_from_the_open_ask_cap(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    ask_ids = [_origin_ask(origin, f"Question {index}?") for index in range(policy.RELAY_CAP)]
    for ask_id in ask_ids:
        assert _relay(watch, ask_id)["ok"] is True
    # The overflowing ask comes from a DIFFERENT origin session, so the bound
    # under test is the watcher's and not the origin's own OPEN_ASK_CAP.
    second = AskQueue(
        FakeSession(), config_dir=tmp_path, session_id="s-origin-2", clock=lambda: BASE
    )
    ninth = _origin_ask(second)
    over = _relay(watch, ninth, origin="s-origin-2")
    assert over["ok"] is False
    assert over["error"] == render.RELAY_CAP_REFUSAL.format(cap=policy.RELAY_CAP)
    # Relays are not the model's asking capacity: a full relay list still leaves
    # the session's OWN asks their whole budget.
    own = watch.enqueue(_questions("A fresh question?"), None)
    assert own["ok"] is True


# ---------------------------------------------------------------------------
# The watcher's settle note
# ---------------------------------------------------------------------------


def test_the_settle_note_is_delivered_exactly_once(tmp_path: Path):
    origin, watch, _, watch_session = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert watch.respond(relay_id, {"q0": ["yes"]}, by="desktop")["ok"] is True
    first = _run(watch.reconcile())
    assert first == [store.relay_row_id(relay_id)]
    messages = [message for batch in watch_session.batches for message in batch]
    assert len(messages) == 1
    note = messages[0]
    assert note.id == store.relay_row_id(relay_id)
    assert note.custom_type == "ask_relay"
    assert note.details["text"] == (
        f"relayed {relay_id} from ask {origin_ask} in session {ORIGIN} "
        "settled: answered by desktop"
    )
    # The row IS the marker: a second reconcile finds it durable and plans none.
    assert _run(watch.reconcile()) == []
    assert len([message for batch in watch_session.batches for message in batch]) == 1


def test_no_note_is_owed_while_the_relay_is_open(tmp_path: Path):
    origin, watch, _, watch_session = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    _relay(watch, origin_ask)
    assert _run(watch.reconcile()) == []
    assert watch_session.batches == []


def test_withdrawing_chat_text_through_a_relay_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    outcome = watch.withdraw(relay_id, reason="answered_in_chat", answers={"q0": ["yes"]})
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_WITHDRAW_ANSWERED_IN_CHAT
    assert [event["kind"] for event in store.read_events(watch.session_dir)] == [
        store.EVENT_RELAYED
    ]


def test_revising_through_a_relay_is_refused(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    assert watch.respond(relay_id, {"q0": ["yes"]}, by="desktop")["ok"] is True
    outcome = watch.revise(relay_id, {"q0": ["no"]}, by="desktop")
    assert outcome["ok"] is False
    assert outcome["error"] == render.RELAY_REVISE_UNSUPPORTED


def test_the_wire_carries_the_relay_provenance(tmp_path: Path):
    origin, watch, _, _ = _pair(tmp_path)
    origin_ask = _origin_ask(origin)
    relay_id = str(_relay(watch, origin_ask)["details"]["ask_id"])
    rows = {row["ask_id"]: row for row in watch.projection()}
    row = rows[relay_id]
    assert row["relay"] == {
        "session_id": ORIGIN,
        "ask_id": origin_ask,
        "origin_status": store.STATUS_OPEN,
    }
    # No second wire shape: the relay row is a PendingAsk like any other.
    assert set(row) <= {
        "ask_id",
        "created_at",
        "expires_at",
        "timeout_s",
        "urgent",
        "status",
        "delivered",
        "questions",
        "answers",
        "attachments",
        "answered_by",
        "answered_at",
        "relay",
        "surfaced_in",
        "draft_question_ids",
    }
    # The index entry on disk is the same row set, so a surface with no runtime
    # reads the provenance too.
    entry = store.read_entry(tmp_path, WATCH)
    assert entry is not None
    assert any(item.get("relay") for item in entry["asks"])
    json.dumps(entry)  # must be JSON-serialisable, like every other index entry
