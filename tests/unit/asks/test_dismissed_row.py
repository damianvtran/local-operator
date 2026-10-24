"""SA-D8 (design ``docs/design/ask-nonblocking.md`` §17): a dismissed ask settles
into a visible row instead of vanishing.

WHY THIS FILE EXISTS. Dismissal used to be the one settled outcome with no row:
``store._shown_rows`` and ``AskQueue.projection`` each dropped ``dismissed`` by
name, from when dismiss was a ``timed_out``-only view act. §15 made it a
SETTLING act on an open ask that owes the agent a response, so the receipt is
the only place the operator can audit what they decided — and the reason
(``moot`` vs ``answered_in_chat``) exists nowhere else afterwards.

The rule splits on TIMING, and every cell below drives BOTH readers — the
derived index (``session_asks`` / ``index_asks``, via ``store._shown_rows``) and
the live projection (``AskQueue.projection``) — because the defect one shared
predicate exists to prevent is a surface hiding a row another still shows. The
expected answer is asserted, not just the agreement: two readers that both
wrongly dropped the row would agree perfectly.

The queue's double and its helpers come from ``test_queue`` rather than being
re-spelled here: one session shape for the whole ask suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.asks import store
from tests.unit.asks.test_queue import FakeSession, _questions, _queue

BASE = 1_700_000_000_000
DAY_MS = 24 * 3600 * 1000
HOUR_MS = 3600 * 1000
TIMEOUT_S = 3600
EXPIRES_AT = BASE + TIMEOUT_S * 1000


def _dismissed_queue(
    tmp_path: Path, *, dismiss_after_ms: int, reason: str = "moot"
) -> tuple[Any, str]:
    """One ask queued at ``BASE``, dismissed ``dismiss_after_ms`` later.

    The dismissal's own ``at`` is what decides the case: before ``expires_at`` it
    is §15's settling act, at or after it the lapsed row being put away.
    """
    queue = _queue(tmp_path, FakeSession(), now=BASE)
    ask_id = str(queue.enqueue(_questions(), TIMEOUT_S)["details"]["ask_id"])
    queue._now = lambda: BASE + dismiss_after_ms
    assert queue.dismiss(ask_id, by="desktop", reason=reason)["ok"] is True
    return queue, ask_id


def _index_rows(queue: Any, tmp_path: Path, now: int) -> list[dict[str, Any]]:
    """The index exactly as this queue's own writer would leave it at ``now``.

    ``AskQueue._refresh`` is ``store.write_entry(..., asks=self.projection(now))``
    and nothing else, so the two readers are fed the one row list a running
    session publishes. A reader-side rule that disagreed with the projection
    would then show up as a difference, and a rule wrong in the WRITER shows up
    against the expected statuses each cell asserts.
    """
    rows = queue.projection(now)
    store.write_entry(tmp_path, "s1", cwd="/tmp/w", asks=rows)
    return rows


def _readers(
    queue: Any, tmp_path: Path, now: int, *, window: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Both readers' rows at ``now``: the live projection and the derived index."""
    projection = (
        queue.projection(now, lapsed_visible_ms=window)
        if window is not None
        else queue.projection(now)
    )
    kwargs = {} if window is None else {"lapsed_visible_ms": window}
    return {
        "projection": projection,
        "session_asks": store.session_asks(tmp_path, "s1", now=now, **kwargs),
        "index_asks": store.index_asks(tmp_path, now=now, **kwargs),
    }


def _statuses(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["status"]) for row in rows]


def _index_row(**patch: Any) -> dict[str, Any]:
    """One frozen ``PendingAsk`` row as an index entry holds it."""
    row: dict[str, Any] = {
        "ask_id": "a-1",
        "created_at": BASE,
        "expires_at": EXPIRES_AT,
        "timeout_s": TIMEOUT_S,
        "urgent": False,
        "status": store.STATUS_DISMISSED,
        "delivered": True,
        "questions": [],
    }
    row.update(patch)
    return row


# ---------------------------------------------------------------------------
# The four cases of the rule, on both readers
# ---------------------------------------------------------------------------


def test_a_pre_deadline_dismissal_is_a_settled_row_on_both_readers(tmp_path: Path):
    """Case 1: §15's dismissal is a receipt, and it says why.

    The row is shown a day past the ask's own deadline — long after the queue
    stopped offering it — because a receipt's horizon is the answered row's, not
    the ask's.
    """
    queue, ask_id = _dismissed_queue(tmp_path, dismiss_after_ms=600_000, reason="answered_in_chat")
    now = EXPIRES_AT + DAY_MS
    _index_rows(queue, tmp_path, now)

    readers = _readers(queue, tmp_path, now)
    for name, rows in readers.items():
        assert _statuses(rows) == [store.STATUS_DISMISSED], name
        assert rows[0]["ask_id"] == ask_id, name
        # The reason rides the row on BOTH readers: the projection from the fold,
        # the index from the row the writer published.
        assert rows[0]["dismissed_reason"] == "answered_in_chat", name
        assert rows[0]["dismissed_at"] == BASE + 600_000, name


def test_the_receipt_ends_strictly_after_the_answered_horizon(tmp_path: Path):
    """Case 1's edge: 7 d from ``expires_at`` inclusive, gone at +1 ms.

    Strictly past, like the horizon beside it (``_shown_rows``' own
    ``now - expires_at > 7 d``): the boundary is pinned on both sides so a
    ``>=``/``>`` slip is a failure rather than a rounding difference.
    """
    queue, _ = _dismissed_queue(tmp_path, dismiss_after_ms=600_000)
    _index_rows(queue, tmp_path, EXPIRES_AT + store.LATE_WINDOW_S * 1000)

    at_horizon = _readers(queue, tmp_path, EXPIRES_AT + store.LATE_WINDOW_S * 1000)
    for name, rows in at_horizon.items():
        assert _statuses(rows) == [store.STATUS_DISMISSED], name

    past_horizon = _readers(queue, tmp_path, EXPIRES_AT + store.LATE_WINDOW_S * 1000 + 1)
    for name, rows in past_horizon.items():
        assert rows == [], name


def test_a_dismissal_at_or_after_the_deadline_still_leaves_both_readers(tmp_path: Path):
    """Case 2: dismissing a LAPSED row is "put it away", and it must still clear.

    §13 promises the lapsed row is shown WITH Dismiss, and the only effect that
    gesture can have is that the row leaves — a receipt here would turn the
    control into a relabel.
    """
    queue, _ = _dismissed_queue(tmp_path, dismiss_after_ms=TIMEOUT_S * 1000 + 60_000)
    now = EXPIRES_AT + 2 * HOUR_MS
    _index_rows(queue, tmp_path, now)

    for name, rows in _readers(queue, tmp_path, now).items():
        assert rows == [], name


def test_a_dismissed_index_row_without_the_stamp_stays_hidden(tmp_path: Path):
    """Case 3: an index entry from a build before ``dismissed_at``.

    Absence means the old behaviour. This row cannot be produced by this build,
    which is why it is written by hand: the shipped case is an index file another
    machine's older runtime left behind.
    """
    (tmp_path / store.SESSIONS_DIRNAME / "s1").mkdir(parents=True)
    store.write_entry(tmp_path, "s1", cwd="/tmp/w", asks=[_index_row()])

    for name, rows in _readers(
        _queue(tmp_path, FakeSession()), tmp_path, EXPIRES_AT + DAY_MS
    ).items():
        # The projection reads the LOG, which holds no dismissal at all — so this
        # cell's evidence is the two index readers; the projection's equivalent
        # (no reason recorded) is covered by the fallback in the TUI/phone tests.
        if name == "projection":
            assert rows == [], name
        else:
            assert rows == [], name


def test_an_expired_row_stays_hidden_on_both_readers(tmp_path: Path):
    """``expired`` is still dropped by name — §17 changes only ``dismissed``.

    The entry is written from the FOLD rather than from the projection, because
    the projection has already dropped the row at this clock (that is one half of
    what is being pinned); the index reader must drop it for its own reason.
    """
    queue = _queue(tmp_path, FakeSession(), now=BASE)
    queue.enqueue(_questions(), TIMEOUT_S)
    now = EXPIRES_AT + store.LATE_WINDOW_S * 1000 + 1
    (record,) = store.fold(store.read_events(queue.session_dir), now)
    assert record["status"] == store.STATUS_EXPIRED
    store.write_entry(tmp_path, "s1", cwd="/tmp/w", asks=[store.pending_row(record, None)])

    for name, rows in _readers(queue, tmp_path, now).items():
        assert rows == [], name


# ---------------------------------------------------------------------------
# The window the receipt ignores, and the window the lapsed row keeps
# ---------------------------------------------------------------------------


def test_lapsed_visible_for_does_not_move_a_dismissed_receipt(tmp_path: Path):
    """Case 1's second half: the setting is for rows settled by the DEADLINE.

    ``at_expiry`` and an hour are the two settings that would otherwise delete an
    open-ask receipt at (or an hour after) the ask's own deadline — shorter than
    the 7 d every answered ask gets. Both are exercised beside a timed-out row
    sharing the same deadline, which must keep obeying the window. The lapsed row
    is written straight into the entry: the window's own tests live beside the
    readers (``test_store.test_session_asks_honours_the_window``), and this cell is
    about the two rows being judged DIFFERENTLY in one read.
    """
    queue, _ = _dismissed_queue(tmp_path, dismiss_after_ms=600_000, reason="moot")
    now = EXPIRES_AT + 2 * HOUR_MS
    asks = queue.projection(now) + [
        _index_row(ask_id="a-lapsed", status=store.STATUS_TIMED_OUT, delivered=True)
    ]
    store.write_entry(tmp_path, "s1", cwd="/tmp/w", asks=asks)

    for window in (0, HOUR_MS):
        rows = store.session_asks(tmp_path, "s1", now=now, lapsed_visible_ms=window)
        assert _statuses(rows) == [store.STATUS_DISMISSED], window
        assert rows[0]["dismissed_reason"] == "moot"
        # The lapsed row is what the window is FOR, and it still obeys it.
        assert "a-lapsed" not in {str(row["ask_id"]) for row in rows}
        # The live projection applies the same window to the same population.
        projected = queue.projection(now, lapsed_visible_ms=window)
        assert _statuses(projected) == [store.STATUS_DISMISSED], window


def test_a_zero_stamp_reads_as_absent_on_both_readers(tmp_path: Path):
    """Review round 1, m3: ``fold`` records ``int(at or 0)``, so a damaged or
    hand-written dismissal stamps ``0``.

    The projection used to read that 0 as an EARLY stamp and show the row, while
    ``pending_row`` omits a falsy stamp from the index row — so the index reader
    hid the same ask. One row, two answers, which is the drift the shared
    predicate exists to stop. Absence and zero are one statement here: a row that
    cannot prove an early dismissal has not earned a receipt.
    """
    queue = _queue(tmp_path, FakeSession(), now=BASE)
    ask_id = str(queue.enqueue(_questions(), TIMEOUT_S)["details"]["ask_id"])
    store.append_event(
        queue.session_dir,
        {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_DISMISSED,
            "ask_id": ask_id,
            "at": 0,
            "by": {"surface": "phone"},
            "reason": "moot",
        },
    )
    (record,) = queue.records(EXPIRES_AT + HOUR_MS)
    assert record["status"] == store.STATUS_DISMISSED
    assert record["dismissed_at"] == 0

    assert queue.projection(EXPIRES_AT + HOUR_MS) == []
    row = store.pending_row(record, None)
    assert "dismissed_at" not in row, "a falsy stamp is not published"
    store.write_entry(tmp_path, "s1", cwd="/tmp/w", asks=[row])
    assert store.session_asks(tmp_path, "s1", now=EXPIRES_AT + HOUR_MS) == []


# ---------------------------------------------------------------------------
# Relay rows (case 4)
# ---------------------------------------------------------------------------


def test_a_dismissed_relay_row_stays_hidden(tmp_path: Path):
    """Case 4: a pointer to another conversation's ask is out of scope.

    Exercised on the index reader (a relay row is what a watcher's entry holds,
    and ``expires_at`` there is 0 until a derivation copies the origin's — which
    is exactly why relay-ness is passed IN rather than inferred from the fields)
    and at the predicate itself, which is where ``AskQueue.projection`` gets it.
    """
    (tmp_path / store.SESSIONS_DIRNAME / "s1").mkdir(parents=True)
    store.write_entry(
        tmp_path,
        "s1",
        cwd="/tmp/w",
        asks=[
            _index_row(
                ask_id="a-relay",
                expires_at=0,
                dismissed_at=BASE,
                dismissed_reason="moot",
                relay={"session_id": "s-origin", "ask_id": "a-origin"},
            )
        ],
    )
    assert store.session_asks(tmp_path, "s1", now=BASE + DAY_MS) == []
    assert store.index_asks(tmp_path, now=BASE + DAY_MS) == []
    assert (
        store.dismissed_hidden(
            store.STATUS_DISMISSED,
            BASE,
            EXPIRES_AT,
            BASE + DAY_MS,
            relay=True,
        )
        is True
    )


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


def test_the_reason_rides_a_dismissed_row_and_nothing_else():
    """``pending_row`` adds the pair on a dismissed row only.

    Both halves matter: a surface needs the reason to draw a receipt, and every
    OTHER status's row has to stay byte-identical to what every build before these
    keys sent — that is what makes the change additive on the wire.
    """
    from tests.unit.asks.test_store import _answered

    queued = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_QUEUED,
        "ask_id": "a-1",
        "at": BASE,
        "expires_at": EXPIRES_AT,
        "timeout_s": TIMEOUT_S,
        "urgent": False,
        "tool_call_id": "",
        "questions": _questions(),
    }
    dismissed = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_DISMISSED,
        "ask_id": "a-1",
        "at": BASE + 1000,
        "reason": "answered_in_chat",
    }
    (record,) = store.fold([queued, dismissed], BASE + 2000)
    row = store.pending_row(record, None)
    assert row["dismissed_at"] == BASE + 1000
    assert row["dismissed_reason"] == "answered_in_chat"

    (open_row,) = store.fold([queued], BASE + 2000)
    assert "dismissed_at" not in store.pending_row(open_row, None)
    assert "dismissed_reason" not in store.pending_row(open_row, None)
    (answered_row,) = store.fold([queued, _answered(at=BASE + 500)], BASE + 1000)
    assert "dismissed_at" not in store.pending_row(answered_row, None)
