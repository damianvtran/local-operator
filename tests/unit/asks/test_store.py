"""The ask log's fold, its derived index, and its append discipline.

These are the pure-stdlib half of design ``docs/design/ask-nonblocking.md``
§2.2. Everything time-dependent injects a clock: the floor is two minutes and a
test that slept through it would be a test nobody runs.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store

DAY_MS = 24 * 3600 * 1000


def _queued(
    ask_id: str = "a-1",
    *,
    at: int = 1_000_000,
    expires_at: int | None = None,
    timeout_s: int = 3600,
    urgent: bool = False,
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_QUEUED,
        "ask_id": ask_id,
        "at": at,
        "expires_at": (at + timeout_s * 1000) if expires_at is None else expires_at,
        "timeout_s": timeout_s,
        "urgent": urgent,
        "tool_call_id": "call-1",
        "questions": questions
        or [
            {
                "id": "q",
                "question": "Which?",
                "options": [],
                "multi": False,
                "secret": False,
                "persist": False,
                "recommended": None,
            }
        ],
    }


def _answered(ask_id: str = "a-1", *, at: int) -> dict[str, Any]:
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_ANSWERED,
        "ask_id": ask_id,
        "at": at,
        "by": {"surface": "terminal"},
        "answers": {"q": ["yes"]},
    }


def _dismissed(ask_id: str = "a-1", *, at: int) -> dict[str, Any]:
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_DISMISSED,
        "ask_id": ask_id,
        "at": at,
        "by": {"surface": "phone"},
    }


# ---------------------------------------------------------------------------
# The fold: one row per rule in §2.2's precedence table
# ---------------------------------------------------------------------------

BASE = 1_000_000_000_000  # a plausible epoch-ms origin; values are relative


def test_rule_8_future_deadline_is_open():
    events = [_queued(at=BASE, timeout_s=3600)]
    (record,) = store.fold(events, BASE + 1000)
    assert record["status"] == store.STATUS_OPEN
    assert record["delivered"] is False


def test_rule_1_in_window_answer_is_answered():
    events = [_queued(at=BASE, timeout_s=3600), _answered(at=BASE + 60_000)]
    (record,) = store.fold(events, BASE + 60_001)
    assert record["status"] == store.STATUS_ANSWERED
    assert record["answers"] == {"q": ["yes"]}
    assert record["answered_by"] == {"surface": "terminal"}


def test_rule_1_outranks_a_dismissal_that_landed_too():
    """Rule 1 sits ABOVE the view actions deliberately (§2.2): a dismissal is an
    action on an ask that already timed out and must never swallow an
    in-window answer, even if a hand-written log holds both."""
    events = [
        _queued(at=BASE, timeout_s=3600),
        {"v": 1, "kind": store.EVENT_DISMISSED, "ask_id": "a-1", "at": BASE + 120_000, "by": {}},
        _answered(at=BASE + 60_000),
    ]
    (record,) = store.fold(events, BASE + 121_000)
    assert record["status"] == store.STATUS_ANSWERED


def test_rule_2_decline_is_terminal_on_write():
    """A decline is terminal-on-write, and it outranks a LATE answer: rule 1
    only fires for an answer INSIDE the window, and a declined ask cannot be
    reopened by an answer that arrives afterwards (§2.2 rules 1/2/4)."""
    events = [
        _queued(at=BASE, timeout_s=120),
        {"v": 1, "kind": store.EVENT_DECLINED, "ask_id": "a-1", "at": BASE + 10, "by": {}},
        _answered(at=BASE + 200_000),
    ]
    (record,) = store.fold(events, BASE + 200_001)
    assert record["status"] == store.STATUS_DECLINED


def test_rule_3_dismissed_is_terminal_on_write():
    events = [
        _queued(at=BASE, timeout_s=120),
        {"v": 1, "kind": store.EVENT_DISMISSED, "ask_id": "a-1", "at": BASE + DAY_MS, "by": {}},
    ]
    (record,) = store.fold(events, BASE + DAY_MS + 1)
    assert record["status"] == store.STATUS_DISMISSED


def test_rule_4_answer_past_the_deadline_is_late_within_the_window():
    events = [_queued(at=BASE, timeout_s=120), _answered(at=BASE + 200_000)]
    (record,) = store.fold(events, BASE + 300_000)
    assert record["status"] == store.STATUS_LATE
    assert record["answers"] == {"q": ["yes"]}


def test_rule_5_answer_past_the_window_is_expired():
    events = [_queued(at=BASE, timeout_s=120), _answered(at=BASE + 200_000)]
    (record,) = store.fold(events, BASE + 200_000 + 8 * DAY_MS)
    assert record["status"] == store.STATUS_EXPIRED


def test_rule_6_unanswered_past_the_deadline_is_timed_out():
    events = [_queued(at=BASE, timeout_s=120)]
    (record,) = store.fold(events, BASE + 120_000)
    assert record["status"] == store.STATUS_TIMED_OUT
    (record,) = store.fold(events, BASE + 120_000 + 6 * DAY_MS)
    assert record["status"] == store.STATUS_TIMED_OUT


def test_rule_7_unanswered_past_the_window_is_expired():
    events = [_queued(at=BASE, timeout_s=120)]
    (record,) = store.fold(events, BASE + 120_000 + 8 * DAY_MS)
    assert record["status"] == store.STATUS_EXPIRED


def test_the_boundary_that_decides_late_from_expired_is_inclusive():
    """``now == expires_at + 7d`` is still LATE (§2.2 rule 4/5 use ``<=``)."""
    events = [_queued(at=BASE, timeout_s=120), _answered(at=BASE + 200_000)]
    edge = BASE + 120_000 + store.LATE_WINDOW_S * 1000
    assert store.fold(events, edge)[0]["status"] == store.STATUS_LATE
    assert store.fold(events, edge + 1)[0]["status"] == store.STATUS_EXPIRED


def test_log_order_not_timestamp_order_decides_the_queue():
    """A clock that went backwards must not reorder the queue."""
    events = [
        _queued("a-2", at=BASE + 5000),
        _queued("a-1", at=BASE),
    ]
    records = store.fold(events, BASE + 6000)
    assert [r["ask_id"] for r in records] == ["a-2", "a-1"]


def test_an_answer_without_a_queued_row_is_skipped_not_invented():
    events = [_answered(at=BASE)]
    assert store.fold(events, BASE + 1) == []


def test_unknown_event_kinds_and_torn_rows_do_not_break_the_fold():
    events = [_queued(at=BASE), {"kind": "invented", "ask_id": "a-1", "at": BASE + 1}]
    (record,) = store.fold(events, BASE + 2)
    assert record["status"] == store.STATUS_OPEN


def test_contract_b_unknown_event_kinds_are_skipped_never_raised():
    """Design §12's contract (b), pinned as a property rather than a promise.

    A build that predates an event kind — the shape every fold eventually has
    for whatever kind ships NEXT — must keep folding a log that carries that
    kind. ``withdrawn`` rides this contract today: a pre-§12 build has no branch
    for it and must find the kind opaque rather than fail. The rows below are
    hostile on purpose (no fields, junk types, unknown nested shapes) because
    tolerance that only holds for well-formed unknown rows is tolerance for
    rows that never occur — and the known events around them must still decide
    the fold.
    """
    events = [
        _queued(at=BASE, timeout_s=3600),
        {"kind": "a_kind_from_the_future"},
        {"kind": "another_future_kind", "ask_id": "a-1", "at": "not-a-number", "extra": [1]},
        _answered(at=BASE + 100),
    ]
    (record,) = store.fold(events, BASE + 200)
    assert record["status"] == store.STATUS_ANSWERED
    # The same tolerance with NO known terminal event: the unknown kinds must
    # not invent one either.
    (open_record,) = store.fold(events[:3], BASE + 200)
    assert open_record["status"] == store.STATUS_OPEN


def test_a_withdrawn_row_with_junk_fields_is_read_by_presence_alone():
    """The new kind obeys the same tolerant-reader contract it relies on: the
    fold reads presence, never a field, so a malformed ``withdrawn`` row (a
    field of the wrong type) must fold identically and raise nothing."""
    events = [
        _queued(at=BASE, timeout_s=3600),
        {"kind": store.EVENT_WITHDRAWN, "ask_id": "a-1", "at": "not-a-number"},
    ]
    (record,) = store.fold(events, BASE + 100)
    assert record["status"] == store.STATUS_WITHDRAWN


# ---------------------------------------------------------------------------
# delivered: the per-status CONSUMPTION flag (amended 2026-10-04)
# ---------------------------------------------------------------------------


def test_delivered_is_sticky_across_the_fold_to_expired():
    """An answered ask that folds to ``expired`` seven days later must not read
    as undelivered — the row it wrote is still in the transcript."""
    events = [_queued(at=BASE, timeout_s=120), _answered(at=BASE + 200_000)]
    present = {store.response_row_id("a-1")}
    fresh = store.fold(events, BASE + 300_000, present_ids=present)[0]
    old = store.fold(events, BASE + 200_000 + 8 * DAY_MS, present_ids=present)[0]
    assert fresh["status"] == store.STATUS_LATE and fresh["delivered"] is True
    assert old["status"] == store.STATUS_EXPIRED and old["delivered"] is True


def test_delivered_needs_a_row_that_exists_not_a_status():
    events = [_queued(at=BASE, timeout_s=120)]
    assert store.fold(events, BASE + 120_000)[0]["delivered"] is False
    delivered = store.fold(events, BASE + 120_000, present_ids={store.timeout_row_id("a-1")})[0]
    assert delivered["delivered"] is True


def test_delivered_is_per_status_the_late_answer_is_the_response_row_alone():
    """CONSUMPTION, not row-existence (amended 2026-10-04).

    A `late` ask's deadline notice is NOT its delivery: the flag stays false
    while only the timeout row is durable — which is exactly where the revision
    window is open (§10) — and the RESPONSE row is what flips it. Once flipped
    it never flips back (the rows are append-only), which the `expired` arm of
    the table carries for an answered-late ask seven days on.
    """
    events = [_queued(at=BASE, timeout_s=120), _answered(at=BASE + 200_000)]
    notice = {store.timeout_row_id("a-1")}
    late = store.fold(events, BASE + 300_000, present_ids=notice)[0]
    assert late["status"] == store.STATUS_LATE and late["delivered"] is False
    landed = store.fold(
        events, BASE + 300_000, present_ids=notice | {store.response_row_id("a-1")}
    )[0]
    assert landed["delivered"] is True
    # `timed_out` is the mirror: the deadline row IS what its status requires,
    # and the response row alone would not deliver a notice that has not landed.
    timed_out = [_queued(at=BASE, timeout_s=120)]
    assert store.fold(timed_out, BASE + 120_000)[0]["delivered"] is False
    assert (
        store.fold(timed_out, BASE + 120_000, present_ids={store.response_row_id("a-1")})[0][
            "delivered"
        ]
        is False
    )


def _revised(ask_id: str = "a-1", *, at: int, answers: dict[str, list[str]] | None = None):
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_REVISED,
        "ask_id": ask_id,
        "at": at,
        "by": {"surface": "desktop"},
        "answers": answers if answers is not None else {"q": ["no"]},
        "supersedes": at - 1,
    }


def test_the_latest_revision_is_effective_and_the_first_answer_keeps_its_stamp():
    """Design §10: status, ``answered_at`` and ``answered_by`` come from the FIRST
    ``answered``; the effective ``answers`` come from the LATEST ``revised``.

    Three writes, one ask: the revision does not move the moment of answering, who
    answered, or the status — only what the agent will be told.
    """
    at = BASE + 100
    events = [
        _queued(at=BASE, timeout_s=3600),
        _answered(at=at),
        _revised(at=at + 10, answers={"q": ["yes"]}),
        _revised(at=at + 20, answers={"q": ["maybe"]}),
    ]
    record = store.fold(events, at + 30)[0]
    assert record["answers"] == {"q": ["maybe"]}
    assert record["status"] == store.STATUS_ANSWERED
    assert record["answered_at"] == at
    assert record["answered_by"] == {"surface": "terminal"}
    # The write it supersedes is published so the revision path can name it.
    assert record["revised_at"] == at + 20


def test_a_revision_without_an_answered_row_does_not_invent_an_answer():
    """A ``revised`` row exists ONLY to supersede: with nothing to supersede (a
    truncated log that lost the ``answered`` row) the ask stays as it was."""
    events = [_queued(at=BASE, timeout_s=3600), _revised(at=BASE + 100)]
    record = store.fold(events, BASE + 200)[0]
    assert record["status"] == store.STATUS_OPEN
    assert "answers" not in record
    assert "revised_at" not in record


# ---------------------------------------------------------------------------
# withdrawn: the agent-side settle's row (design §12)
# ---------------------------------------------------------------------------


def _withdrawn(ask_id: str = "a-1", *, at: int) -> dict[str, Any]:
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_WITHDRAWN,
        "ask_id": ask_id,
        "at": at,
        "by": {"surface": "agent"},
    }


# ---------------------------------------------------------------------------
# dismissed: the notice a dismissal does not erase (fork A, §14)
# ---------------------------------------------------------------------------


def test_a_dismissal_after_the_deadline_still_owes_the_notice():
    """A DISMISSAL puts the ask away; it does not tell the model anything.

    The dismissal's promise is about the USER's view ("the ask is gone from my
    list"), and the deadline notice is the AGENT's fact ("you have no answer and
    must proceed"). They are different facts, so a dismissal whose notice was
    never delivered still owes it — before this the record folded to
    ``dismissed``, left ``INJECTING_STATUSES`` and dropped the row for good, so
    the agent was never told it had been waiting on nobody.
    """
    events = [_queued(at=BASE, timeout_s=120), _dismissed(at=BASE + 200_000)]
    (record,) = store.fold(events, BASE + 200_001)
    assert record["status"] == store.STATUS_DISMISSED
    assert record["dismissed_at"] == BASE + 200_000
    assert record["notice_owed"] is True
    # ``delivered_hint`` is UNCHANGED (decision D22): dismissed still means "any
    # row present", so this reads False until the notice actually lands.
    assert record["delivered"] is False
    assert store.expected_row_ids(record) == [store.timeout_row_id("a-1")]


def test_a_dismissed_notice_is_not_owed_past_the_late_window():
    """The 7 d window is the notice's own bound, and a dismissal inherits it.

    Past the window the notice is stale by exactly the rule that governs an
    unanswered deadline (``fold`` rule 7), so a dismissal there owes nothing and
    must not resurrect a row a week late.
    """
    expires_at = BASE + 120_000
    events = [_queued(at=BASE, timeout_s=120), _dismissed(at=BASE + 200_000)]
    past = expires_at + store.LATE_WINDOW_S * 1000 + 1
    (record,) = store.fold(events, past)
    assert record["status"] == store.STATUS_DISMISSED
    assert record["notice_owed"] is False
    assert store.expected_row_ids(record) == []


def test_a_dismissal_before_the_deadline_owes_nothing():
    """Only a deadline that had ALREADY passed can leave a notice undelivered.

    A hand-written log may carry a dismissal dated inside the window (no surface
    offers one — ``AskQueue.dismiss`` refuses a non-lapsed ask — but the fold is
    total, so it has to answer). Nothing was owed at that instant, and the
    dismissal does not invent an owed notice.
    """
    events = [_queued(at=BASE, timeout_s=3600), _dismissed(at=BASE + 1_000)]
    (record,) = store.fold(events, BASE + 2_000)
    assert record["status"] == store.STATUS_DISMISSED
    assert record["notice_owed"] is False
    assert store.expected_row_ids(record) == []


def test_a_dismissed_notice_already_in_the_transcript_is_not_owed():
    """The field says a row is still OWED, so a durable one clears it.

    ``notice_owed`` was read off the dates alone, so a dismissed ask whose
    ``ask-timeout-<id>`` row had already landed still said ``notice_owed: True``
    (agent review round 1, MINOR-2) — and ``expected_row_ids``, the queue's
    reconcile and every future reader take that field at its word. The fold has
    the facts (``present_ids``), so it answers the question it is asked rather
    than the rule behind it. ``delivered`` is the any-row form (see
    ``delivered_hint``): a settled row a dismissal arrived AFTER is the answer
    the model already has, which the notice's own N8 rule lets stand.
    """
    events = [_queued(at=BASE, timeout_s=120), _dismissed(at=BASE + 200_000)]
    (record,) = store.fold(events, BASE + 200_001, present_ids=[store.timeout_row_id("a-1")])
    assert record["status"] == store.STATUS_DISMISSED
    assert record["delivered"] is True
    assert record["notice_owed"] is False
    assert store.expected_row_ids(record) == []


def test_a_withdrawn_row_folds_to_withdrawn_and_leaves_the_outstanding_set():
    """Rule §12: a withdrawn row with NO answered sibling is terminal-on-write.

    It injects nothing (the dismissal rule), so it owes no row, and it is
    settled — never outstanding — for every tally that reads the one set.
    """
    events = [_queued(at=BASE, timeout_s=3600), _withdrawn(at=BASE + 100)]
    (record,) = store.fold(events, BASE + 200)
    assert record["status"] == store.STATUS_WITHDRAWN
    assert record["delivered"] is False
    assert store.is_outstanding(record["status"]) is False
    assert store.outstanding_asks([record]) == []
    assert store.expected_row_ids(record) == []
    assert store.pending_row(record)["status"] == "withdrawn"


def test_a_withdrawn_row_is_terminal_across_the_deadline_and_the_window_days():
    """Terminal-on-write in TIME too: the deadline branches never judge a
    retracted ask — before the deadline, at it, and 8 days past it."""
    events = [_queued(at=BASE, timeout_s=3600), _withdrawn(at=BASE + 100)]
    for now in (BASE + 200, BASE + 3600 * 1000, BASE + 3600 * 1000 + 8 * DAY_MS):
        assert store.fold(events, now)[0]["status"] == store.STATUS_WITHDRAWN


def test_contract_a_a_withdrawn_row_and_an_answered_row_in_both_orders():
    """Design §12's contract (a): an ``answered`` row ALWAYS wins — only a
    withdrawn row with no answered sibling folds to ``withdrawn``.

    Both write orders are pinned because both are reachable: the racing
    withdrawal may land before or after the answer that crossed its fold check
    (the check-then-append gap is cross-process), and in neither order may the
    operator's real answer be swallowed by the asker's retraction.
    """
    before = [
        _queued(at=BASE, timeout_s=3600),
        _answered(at=BASE + 100),
        _withdrawn(at=BASE + 200),
    ]
    after = [
        _queued(at=BASE, timeout_s=3600),
        _withdrawn(at=BASE + 100),
        _answered(at=BASE + 200),
    ]
    for events in (before, after):
        record = store.fold(events, BASE + 300)[0]
        assert record["status"] == store.STATUS_ANSWERED
        assert record["answers"] == {"q": ["yes"]}


def test_an_answered_late_row_also_outranks_a_later_withdrawn():
    """The carve-out reads the ANSWERED ROW, not the folded status: an answer
    past the deadline is still the operator's answer, and the withdrawal still
    loses. It folds ``late``, never ``withdrawn``."""
    events = [
        _queued(at=BASE, timeout_s=120),
        _answered(at=BASE + 200_000),
        _withdrawn(at=BASE + 300_000),
    ]
    record = store.fold(events, BASE + 400_000)[0]
    assert record["status"] == store.STATUS_LATE


def test_a_withdrawn_row_before_an_answered_late_row_still_folds_late():
    """The mirrored write order of the cell above (review round 1, NIT 4): the
    withdrawal lands FIRST and the past-deadline answer AFTER it — the answered
    branch still sits above the withdrawn branch, so the fold reads ``late``,
    never ``withdrawn``, and keeps the words."""
    events = [
        _queued(at=BASE, timeout_s=120),
        _withdrawn(at=BASE + 200_000),
        _answered(at=BASE + 300_000),
    ]
    record = store.fold(events, BASE + 400_000)[0]
    assert record["status"] == store.STATUS_LATE
    assert record["answers"] == {"q": ["yes"]}


def test_expected_rows_are_per_kind_and_late_needs_the_response_alone():
    """A `late` ask owes ONE row, and that is the level-triggered rule.

    The response SUPERSEDES the deadline row for good (review round 1, MAJOR 2): a
    per-batch suppression let the next reconcile write the timeout row below the
    answer it announced. The two rows can still both EXIST for one ask — when the
    deadline genuinely fired first, in its own reconcile — which is why the
    delivery marker stays per (ask, kind) rather than one boolean.
    """
    late = {"ask_id": "a-1", "status": store.STATUS_LATE}
    assert store.expected_row_ids(late) == [store.response_row_id("a-1")]
    timed_out = {"ask_id": "a-1", "status": store.STATUS_TIMED_OUT}
    assert store.expected_row_ids(timed_out) == [store.timeout_row_id("a-1")]
    answered = {"ask_id": "a-1", "status": store.STATUS_ANSWERED}
    assert store.expected_row_ids(answered) == [store.response_row_id("a-1")]
    dismissed = {"ask_id": "a-1", "status": store.STATUS_DISMISSED}
    assert store.expected_row_ids(dismissed) == []
    withdrawn = {"ask_id": "a-1", "status": store.STATUS_WITHDRAWN}
    assert store.expected_row_ids(withdrawn) == []
    # A dismissed ask that HAD a row keeps its delivered flag; the row is the
    # marker, not the status.
    assert store.pending_row({"ask_id": "a-1", "status": "dismissed"})["delivered"] is False


# ---------------------------------------------------------------------------
# The log on disk
# ---------------------------------------------------------------------------


def test_append_then_read_round_trips(tmp_path: Path):
    session = store.session_dir(tmp_path, "s1")
    session.mkdir(parents=True)
    assert store.append_event(session, _queued())
    assert store.append_event(session, _answered(at=BASE + 10))
    events = store.read_events(session)
    assert [e["kind"] for e in events] == [store.EVENT_QUEUED, store.EVENT_ANSWERED]


def test_read_events_tolerates_a_torn_final_line(tmp_path: Path):
    session = store.session_dir(tmp_path, "s1")
    session.mkdir(parents=True)
    store.append_event(session, _queued())
    with store.asks_log_path(session).open("ab") as handle:
        handle.write(b'{"kind": "answered", "ask_i')  # a crash mid-write
    events = store.read_events(session)
    assert len(events) == 1


def test_read_events_of_a_missing_session_is_empty(tmp_path: Path):
    assert store.read_events(store.session_dir(tmp_path, "nope")) == []


def test_concurrent_appenders_keep_every_row(tmp_path: Path):
    """The O_APPEND + LOCK_NB contract: N processes, N*K rows, no torn line.

    Run for real (the writers are separate processes, which is the whole point
    of the file), and kept tiny because a concurrency test that is flaky is
    worse than none: 4 writers x 25 rows.
    """
    session = store.session_dir(tmp_path, "s1")
    session.mkdir(parents=True)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(session), f"a-{i}", 25)) for i in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    events = store.read_events(session)
    assert len(events) == 100
    by_id: dict[str, int] = {}
    for event in events:
        by_id[event["ask_id"]] = by_id.get(event["ask_id"], 0) + 1
    assert by_id == {f"a-{i}": 25 for i in range(4)}


def _append_many(session_dir: str, ask_id: str, count: int) -> None:
    from local_operator.asks import store as _store

    for index in range(count):
        _store.append_event(Path(session_dir), _queued(ask_id, at=BASE + index))


# ---------------------------------------------------------------------------
# The derived index
# ---------------------------------------------------------------------------


def _write_index(tmp_path: Path, session_id: str = "s1", asks: list[dict[str, Any]] | None = None):
    return store.write_entry(
        tmp_path,
        session_id,
        cwd="/tmp",
        asks=asks if asks is not None else [store.pending_row({**_queued(), "status": "open"})],
    )


def test_index_round_trips_and_then_disappears_when_emptied(tmp_path: Path):
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    assert _write_index(tmp_path) is not None
    entry = store.read_entry(tmp_path, "s1")
    assert entry and entry["schema"] == store.INDEX_SCHEMA
    assert store.write_entry(tmp_path, "s1", cwd="/tmp", asks=[]) is None
    assert store.read_entry(tmp_path, "s1") is None


def test_a_corrupt_index_entry_reads_as_absent(tmp_path: Path):
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    _write_index(tmp_path)
    store.entry_path(tmp_path, "s1").write_text("{not json")
    assert store.read_entry(tmp_path, "s1") is None


def test_an_unknown_schema_reads_as_absent(tmp_path: Path):
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    _write_index(tmp_path)
    path = store.entry_path(tmp_path, "s1")
    payload = json.loads(path.read_text())
    payload["schema"] = store.INDEX_SCHEMA + 1
    path.write_text(json.dumps(payload))
    assert store.read_entry(tmp_path, "s1") is None


def test_the_index_self_heals_from_the_log(tmp_path: Path):
    """A deleted index is rebuilt by the next writer — the wakes/ contract."""
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    _write_index(tmp_path)
    store.entry_path(tmp_path, "s1").unlink()
    assert store.read_index(tmp_path) == {}
    _write_index(tmp_path)
    assert "s1" in store.read_index(tmp_path)


def test_read_index_sweeps_an_entry_whose_session_directory_is_gone(tmp_path: Path):
    """The §2.2 TTL sweep, and the unit test the note names: delete the session
    directory by hand, reopen the index → the entry is gone."""
    session = tmp_path / "sessions" / "s1"
    session.mkdir(parents=True)
    _write_index(tmp_path)
    assert "s1" in store.read_index(tmp_path)
    session.rmdir()
    assert store.read_index(tmp_path) == {}
    assert not store.entry_path(tmp_path, "s1").exists()


def test_read_index_sweeps_an_entry_whose_asks_are_all_old_and_terminal(tmp_path: Path):
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    old = store.pending_row(
        {
            **_queued(at=1_000_000, timeout_s=120),
            "status": "timed_out",
            "expires_at": 1_120_000,
        }
    )
    _write_index(tmp_path, asks=[old])
    assert store.read_index(tmp_path, now=1_120_000 + 8 * DAY_MS) == {}
    # ... and the same entry is KEPT while it is still within the window.
    _write_index(tmp_path, asks=[old])
    assert "s1" in store.read_index(tmp_path, now=1_120_000 + 6 * DAY_MS)


def test_read_index_keeps_an_entry_with_an_open_ask_however_old(tmp_path: Path):
    """An open ask is answerable whenever the user comes back; age alone must
    never sweep it (only a missing session directory does)."""
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    _write_index(tmp_path, asks=[store.pending_row({**_queued(at=1_000_000), "status": "open"})])
    assert "s1" in store.read_index(tmp_path, now=1_000_000 + 400 * DAY_MS)


def test_read_index_skips_staged_temp_files(tmp_path: Path):
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    _write_index(tmp_path)
    (store.asks_dir(tmp_path) / ".s1.abc.tmp").write_text("{}")
    assert list(store.read_index(tmp_path)) == ["s1"]


def test_entry_is_stale_is_a_pure_question():
    entry = {"asks": []}
    assert store.entry_is_stale(entry, 0, session_exists=True) is True
    assert store.entry_is_stale(entry, 0, session_exists=False) is True


def test_now_ms_and_new_ask_id_are_usable():
    before = int(time.time() * 1000)
    assert store.now_ms() >= before
    assert store.new_ask_id().startswith("a-")
    assert store.new_ask_id({"a-1"}) != "a-1"


@pytest.mark.parametrize("bad", ["", "   "])
def test_new_ask_id_is_never_empty(bad: str):
    assert store.new_ask_id() != bad


def test_the_append_is_0600(tmp_path: Path):
    session = store.session_dir(tmp_path, "s1")
    session.mkdir(parents=True)
    store.append_event(session, _queued())
    mode = os.stat(store.asks_log_path(session)).st_mode & 0o777
    assert mode == 0o600


def test_append_creates_the_session_directory(tmp_path: Path):
    session = store.session_dir(tmp_path, "fresh")
    assert store.append_event(session, _queued())
    assert store.asks_log_path(session).exists()


# ---------------------------------------------------------------------------
# The outstanding set: ONE spelling of "the user can still act on this ask"
# ---------------------------------------------------------------------------


def test_a_timed_out_ask_is_settled_but_still_injects_its_notice():
    """The rule the wire's tally, the index and every surface share (#2388).

    An ask settles when the AGENT is told — by its deadline or by an answer —
    not only when the user acts: a timed-out ask is no longer outstanding, so no
    surface counts it as owed. Settling it must not cost the agent its notice:
    ``timed_out`` still injects, and the fold still requires exactly the
    deadline row.
    """
    assert store.OUTSTANDING_STATUSES == frozenset({store.STATUS_OPEN})
    assert store.is_outstanding(store.STATUS_OPEN)
    assert not store.is_outstanding(store.STATUS_TIMED_OUT)
    assert store.STATUS_TIMED_OUT in store.INJECTING_STATUSES
    (record,) = store.fold([_queued(at=BASE, timeout_s=3600)], BASE + 3_600_001)
    assert record["status"] == store.STATUS_TIMED_OUT
    assert store.expected_row_ids(record) == [store.timeout_row_id("a-1")]
    for settled in (
        store.STATUS_TIMED_OUT,
        store.STATUS_ANSWERED,
        store.STATUS_LATE,
        store.STATUS_DECLINED,
        store.STATUS_DISMISSED,
        store.STATUS_WITHDRAWN,
        store.STATUS_EXPIRED,
    ):
        assert not store.is_outstanding(settled)
    # A status nobody can name is settled, never outstanding: a surface must not
    # be told to offer an answer for an ask it cannot classify.
    assert not store.is_outstanding(None)
    assert not store.is_outstanding("")


def test_open_asks_is_the_cap_set_and_a_timed_out_ask_holds_no_slot():
    """The cap counts OPEN asks only, and so does the outstanding set (#2388).

    A timed-out ask must not hold a slot a fresh question needs, and it is no
    longer owed either: the agent was told and moved on. It stays SHOWN (the
    row is a settled receipt), which is the readers' business, not this set's.
    """
    events = [_queued(at=BASE, timeout_s=3600)]
    (record,) = store.fold(events, BASE + 3_600_001)
    assert record["status"] == store.STATUS_TIMED_OUT
    assert store.open_asks([record]) == []
    assert store.outstanding_asks([record]) == []


# ---------------------------------------------------------------------------
# Image answers: refs, never bytes (the Other door's attachments)
# ---------------------------------------------------------------------------

_REF = {"attachment": "0123456789abcdef0123456789abcdef", "mime_type": "image/png", "bytes": 4242}


def test_a_text_only_answer_folds_without_an_attachments_key() -> None:
    (record,) = store.fold([_queued(), _answered(at=1_500_000)], now=1_600_000)
    assert "attachments" not in record
    assert "attachments" not in store.pending_row(record)


def test_the_fold_keeps_the_first_answers_refs_across_a_revision() -> None:
    """D5: a revision replaces the TEXT; the pictures are immutable once sent."""
    answered = {**_answered(at=1_500_000), "attachments": {"q": [dict(_REF)]}}
    revised = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_REVISED,
        "ask_id": "a-1",
        "at": 1_550_000,
        "by": {"surface": "desktop"},
        "answers": {"q": ["changed my mind"]},
        "supersedes": 1_500_000,
    }

    (record,) = store.fold([_queued(), answered, revised], now=1_600_000)

    assert record["answers"] == {"q": ["changed my mind"]}
    assert record["attachments"] == {"q": [_REF]}
    assert store.pending_row(record)["attachments"] == {"q": [_REF]}


def test_a_revision_cannot_smuggle_attachments_in() -> None:
    """Even a hand-written ``revised`` event carrying the key contributes nothing."""
    revised = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_REVISED,
        "ask_id": "a-1",
        "at": 1_550_000,
        "by": {"surface": "desktop"},
        "answers": {"q": ["x"]},
        "attachments": {"q": [dict(_REF)]},
    }
    (record,) = store.fold([_queued(), _answered(at=1_500_000), revised], now=1_600_000)
    assert "attachments" not in record


def _stale_open_entry(root: Path, session_id: str, now: int) -> None:
    """An index entry still saying ``open`` one minute past its deadline — what a
    runtime stopped before the deadline leaves behind (it never wrote the fold
    again)."""
    (root / store.SESSIONS_DIRNAME / session_id).mkdir(parents=True)
    store.write_entry(
        root,
        session_id,
        cwd="/tmp/w",
        asks=[
            {
                "ask_id": "stale-1",
                "created_at": now - 600_000,
                "expires_at": now - 60_000,
                "timeout_s": 540,
                "urgent": False,
                "status": store.STATUS_OPEN,
                "delivered": False,
                "questions": [],
            }
        ],
    )


def test_a_stale_open_row_past_its_deadline_reads_timed_out_from_every_index_reader(
    tmp_path: Path,
) -> None:
    """The index holds the status as last WRITTEN; the readers judge it at ``now``.

    Without the re-derivation the phone showed a lapsed ask with no control (the
    queue would accept a dismiss — it folds the log to ``timed_out`` — but the
    surface read the index's stale ``open``), the legacy mirror painted it as a
    live card, and every count called it waiting.
    """
    now = int(time.time() * 1000)
    _stale_open_entry(tmp_path, "s-stale", now)
    assert [row["status"] for row in store.session_asks(tmp_path, "s-stale", now=now)] == [
        store.STATUS_TIMED_OUT
    ]
    assert [row["status"] for row in store.index_asks(tmp_path, now=now)] == [
        store.STATUS_TIMED_OUT
    ]


# ---------------------------------------------------------------------------
# The lapsed-visibility window (decisions D6–D8: `asks.lapsed_visible_for`)
# ---------------------------------------------------------------------------

HOUR_MS = 3600 * 1000


def _lapsed_and_answered_entry(root: Path, session_id: str, expires_at: int) -> None:
    """One timed-out ask and one answered ask sharing the SAME deadline.

    The pair is what pins D6's scope: the window is about LAPSED rows only, so
    an answered row with the identical ``expires_at`` must survive the window
    that hides its timed-out neighbour (an answer receipt is not an expired
    question)."""
    (root / store.SESSIONS_DIRNAME / session_id).mkdir(parents=True)
    store.write_entry(
        root,
        session_id,
        cwd="/tmp/w",
        asks=[
            store.pending_row(
                {
                    **_queued("lapsed-1", at=expires_at - 120_000, timeout_s=120),
                    "expires_at": expires_at,
                    "status": store.STATUS_TIMED_OUT,
                }
            ),
            store.pending_row(
                {
                    **_queued("answered-1", at=expires_at - 120_000, timeout_s=120),
                    "expires_at": expires_at,
                    "status": store.STATUS_ANSWERED,
                }
            ),
        ],
    )


def test_a_zero_window_hides_a_timed_out_row_but_keeps_an_answered_one(tmp_path: Path):
    _lapsed_and_answered_entry(tmp_path, "s1", BASE)
    rows = store.index_asks(tmp_path, now=BASE + 1, lapsed_visible_ms=0)
    assert [row["ask_id"] for row in rows] == ["answered-1"]


def test_no_window_keyword_is_todays_seven_day_result(tmp_path: Path):
    _lapsed_and_answered_entry(tmp_path, "s1", BASE)
    default = store.index_asks(tmp_path, now=BASE + 6 * DAY_MS)
    assert sorted(row["ask_id"] for row in default) == ["answered-1", "lapsed-1"]
    explicit = store.index_asks(
        tmp_path, now=BASE + 6 * DAY_MS, lapsed_visible_ms=store.LATE_WINDOW_S * 1000
    )
    assert explicit == default
    # And the default IS the 7 d choice of the setting, not a second number.
    assert store.LAPSED_VISIBLE_FOR_S[store.DEFAULT_LAPSED_VISIBLE_FOR] == store.LATE_WINDOW_S


def test_a_stale_open_row_past_its_deadline_is_hidden_after_the_window(tmp_path: Path):
    """D5's re-derived ``open`` past ``expires_at`` IS lapsed, so the window
    applies to it exactly as to a written ``timed_out``."""
    now = BASE
    _stale_open_entry(tmp_path, "s-stale", now)  # expires_at = now - 60 s
    assert [
        r["ask_id"] for r in store.index_asks(tmp_path, now=now, lapsed_visible_ms=HOUR_MS)
    ] == ["stale-1"]
    later = now - 60_000 + HOUR_MS + 1
    assert store.index_asks(tmp_path, now=later, lapsed_visible_ms=HOUR_MS) == []


def test_an_open_row_before_its_deadline_ignores_the_window(tmp_path: Path):
    (tmp_path / store.SESSIONS_DIRNAME / "s1").mkdir(parents=True)
    store.write_entry(
        tmp_path,
        "s1",
        cwd="/tmp",
        asks=[store.pending_row({**_queued(at=BASE, timeout_s=3600), "status": "open"})],
    )
    assert len(store.index_asks(tmp_path, now=BASE + 1, lapsed_visible_ms=0)) == 1


def test_session_asks_honours_the_window(tmp_path: Path):
    _lapsed_and_answered_entry(tmp_path, "s1", BASE)
    assert [
        r["ask_id"] for r in store.session_asks(tmp_path, "s1", now=BASE + 1, lapsed_visible_ms=0)
    ] == ["answered-1"]
    assert len(store.session_asks(tmp_path, "s1", now=BASE + 1)) == 2


def test_the_sweep_keeps_an_entry_whose_only_row_is_hidden_by_the_window(tmp_path: Path):
    """The window is DISPLAY only (D6): the index's TTL sweep stays at 7 d, so
    widening the setting again brings the row back for a stopped session."""
    (tmp_path / store.SESSIONS_DIRNAME / "s1").mkdir(parents=True)
    store.write_entry(
        tmp_path,
        "s1",
        cwd="/tmp",
        asks=[
            store.pending_row(
                {
                    **_queued(at=BASE - 120_000, timeout_s=120),
                    "expires_at": BASE,
                    "status": "timed_out",
                }
            )
        ],
    )
    now = BASE + 2 * HOUR_MS
    assert store.index_asks(tmp_path, now=now, lapsed_visible_ms=HOUR_MS) == []
    entry = store.read_entry(tmp_path, "s1")
    assert entry is not None
    assert store.entry_is_stale(entry, now, session_exists=True) is False
    assert store.entry_path(tmp_path, "s1").exists()
    assert len(store.index_asks(tmp_path, now=now)) == 1


@pytest.mark.parametrize(
    ("choice", "seconds"),
    [("at_expiry", 0), ("1h", 3600), ("24h", 86_400), ("7d", 7 * 86_400)],
)
def test_the_choice_table_maps_every_choice_from_expires_at(choice: str, seconds: int):
    assert store.LAPSED_VISIBLE_FOR_S[choice] == seconds


def test_is_lapsed_names_timed_out_only():
    assert store.LAPSED_STATUSES == frozenset({store.STATUS_TIMED_OUT})
    assert store.is_lapsed(store.STATUS_TIMED_OUT) is True
    for status in (
        "open",
        "answered",
        "declined",
        "late",
        "dismissed",
        "withdrawn",
        "expired",
        None,
    ):
        assert store.is_lapsed(status) is False, status


# ---------------------------------------------------------------------------
# Relay rows (#2520) and the tolerance that makes them safe to introduce
# ---------------------------------------------------------------------------


def _relayed(
    ask_id: str = "a-r", *, at: int = 1_000_000, origin: str = "s-other"
) -> dict[str, Any]:
    return {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_RELAYED,
        "ask_id": ask_id,
        "at": at,
        "relay": {"session_id": origin, "ask_id": "a-1"},
    }


def test_a_relay_row_folds_without_a_queued_event():
    """The one ask shape with no ``queued`` row: the watcher records a REFERENCE.

    The fold emits the base record and nothing more — the questions, deadline and
    status are the ORIGIN's and are filled in by the queue's derivation, which is
    the only reader allowed to open another session's log.
    """
    records = store.fold([_relayed()], 1_000_000, session_id="s-watch")
    assert len(records) == 1
    record = records[0]
    assert record["ask_id"] == "a-r"
    assert record["session_id"] == "s-watch"
    assert record["relay"] == {"session_id": "s-other", "ask_id": "a-1"}
    assert record["questions"] == []
    assert record["status"] == store.STATUS_OPEN
    # No deadline of its own: the origin owns it.
    assert record["expires_at"] == 0
    assert record["timeout_s"] == 0


def test_a_relay_row_carries_the_watchers_withdrawal_stamp():
    withdrawn = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_WITHDRAWN,
        "ask_id": "a-r",
        "at": 1_000_500,
        "by": {"surface": "agent"},
    }
    records = store.fold([_relayed(), withdrawn], 1_001_000, session_id="s-watch")
    assert records[0]["relay_withdrawn_at"] == 1_000_500
    # FOLDED, not just stamped (#2673, R1-3): every COLD reader — the cleanup
    # probe, the cold dismissal door — sees a settled relay, so the session is
    # prunable and the doors agree without reading the origin.
    assert records[0]["status"] == store.STATUS_WITHDRAWN


def test_a_relay_row_folds_a_local_dismissal_over_a_withdrawal():
    """A dismissed relay reads dismissed, and outranks a withdrawal (§12)."""
    dismissed = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_DISMISSED,
        "ask_id": "a-r",
        "at": 1_000_600,
        "by": {"surface": "terminal"},
    }
    withdrawn = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_WITHDRAWN,
        "ask_id": "a-r",
        "at": 1_000_500,
        "by": {"surface": "agent"},
    }
    assert store.fold([_relayed(), dismissed], 1_001_000, session_id="s-watch")[0][
        "status"
    ] == store.STATUS_DISMISSED
    both = store.fold([_relayed(), withdrawn, dismissed], 1_001_000, session_id="s-watch")
    assert both[0]["status"] == store.STATUS_DISMISSED
    assert both[0]["relay_dismissed_at"] == 1_000_600
    # The wire row must not leak the fold's internal stamps.
    assert "relay_dismissed_at" not in store.pending_row(both[0])


def test_an_old_shaped_log_folds_byte_identically_and_ignores_unknown_kinds():
    """ADDITIVE, or the introduction would not be safe on a mixed tree.

    Two assertions in one cell because they are one property: an old log (known
    kinds only) and the same log plus an event kind this reader has never heard
    of must produce the SAME records, byte for byte — and neither may grow a
    relay/surfaced key that an older surface would then have to understand.
    """
    old_shaped = [_queued(), _answered(at=1_000_100)]
    unknown = {
        "v": store.EVENT_SCHEMA,
        "kind": "a-kind-from-a-later-revision",
        "ask_id": "a-1",
        "at": 1_000_050,
        "whatever": {"nested": True},
    }
    baseline = store.fold(old_shaped, 1_000_200, session_id="s1")
    with_unknown = store.fold([*old_shaped, unknown], 1_000_200, session_id="s1")
    assert with_unknown == baseline
    assert json.dumps(with_unknown, sort_keys=True) == json.dumps(baseline, sort_keys=True)
    assert "relay" not in baseline[0]
    assert "surfaced_in" not in baseline[0]
    # And the WIRE row for an ordinary ask carries none of the relay keys either.
    row = store.pending_row(baseline[0])
    assert set(row) == {
        "ask_id",
        "created_at",
        "expires_at",
        "timeout_s",
        "urgent",
        "status",
        "delivered",
        "questions",
        "answers",
        "answered_by",
        "answered_at",
    }


def test_a_relay_wire_row_carries_its_provenance_and_nothing_empty():
    record = store.fold([_relayed()], 1_000_000, session_id="s-watch")[0]
    row = store.pending_row(record)
    assert row["relay"] == {"session_id": "s-other", "ask_id": "a-1"}
    # Nothing empty rides beside it: the reference is the whole provenance.
    assert set(row) == {
        "ask_id",
        "created_at",
        "expires_at",
        "timeout_s",
        "urgent",
        "status",
        "delivered",
        "questions",
        "relay",
    }


def test_surfaced_rows_attach_to_the_origin_record_only():
    surfaced = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_RELAY_SURFACED,
        "ask_id": "a-1",
        "at": 1_000_050,
        "by": {"session_id": "s-watch", "ask_id": "a-r"},
    }
    records = store.fold([_queued(), surfaced], 1_000_100, session_id="s1")
    assert records[0]["surfaced_in"] == [
        {
            "session_id": "s-watch",
            "ask_id": "a-r",
            "at": 1_000_050,
        }
    ]
    # The surfaced row is NOT a status event: the origin's own precedence decides.
    assert records[0]["status"] == store.STATUS_OPEN
    assert len(records) == 1


def test_relay_row_ids_are_their_own_namespace():
    assert store.relay_row_id("a-1") == "ask-relay-a-1"
    assert store.relay_row_id("a-1") not in (
        store.response_row_id("a-1"),
        store.timeout_row_id("a-1"),
    )
