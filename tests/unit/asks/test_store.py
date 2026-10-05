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


def test_is_outstanding_covers_open_and_timed_out_and_no_settled_status():
    """The rule the wire's tally, the index and every surface share.

    It is the flip precondition for ``LOP_ASK_NONBLOCKING``: a timed-out but
    UNANSWERED ask is still outstanding — a late answer reaches the agent
    (design §2.2, the spec's item 3) — while a settled one is not.
    """
    assert store.is_outstanding(store.STATUS_OPEN)
    assert store.is_outstanding(store.STATUS_TIMED_OUT)
    for settled in (
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


def test_outstanding_asks_and_pending_asks_are_one_set_by_construction():
    """``pending_asks`` (notice owed) delegates to the outstanding set.

    They coincide because an ask owes its timeout notice exactly while it is
    outstanding; a second spelling of either rule is what produced the wrong
    wire tally, so there is only one.
    """
    events = [_queued(at=BASE, timeout_s=3600)]
    (record,) = store.fold(events, BASE + 3_600_001)
    assert record["status"] == store.STATUS_TIMED_OUT
    assert [row["ask_id"] for row in store.outstanding_asks([record])] == ["a-1"]
    assert store.pending_asks([record]) == store.outstanding_asks([record])


def test_open_asks_is_the_narrower_cap_set():
    """The cap counts OPEN asks only; the display set counts outstanding ones.

    A timed-out ask must not hold a slot a fresh question needs, and it must not
    vanish from a surface either — two questions, two sets, one definition each.
    """
    events = [_queued(at=BASE, timeout_s=3600)]
    (record,) = store.fold(events, BASE + 3_600_001)
    assert record["status"] == store.STATUS_TIMED_OUT
    assert store.open_asks([record]) == []
    assert len(store.outstanding_asks([record])) == 1
