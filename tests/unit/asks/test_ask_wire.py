"""The A2 WIRE: presence, the fold's rows, the legacy mirror, the aggregate view.

Design ``docs/design/ask-nonblocking.md`` §4 (the frozen wire and the N2/N3
rules) and §7's A2 cell. The property every test here defends is the same one
stated twice in the design: **presence is the capability proxy**. While
``asks.policy.NONBLOCKING_ASK`` is off, the ask fields must be ABSENT from the
frontend state, the projection and the list rows — exactly as an old core omits
them — because a client that saw the field would take the queued path against a
backend that still blocks.

The queue itself is faked only where the wire meets it (a session exposing
``ask_queue()``), and the queue in those fakes is the REAL ``AskQueue``: the
fold, the cap and the ordering under test are A1's, and a stub that reimplemented
them would test the stub.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.asks import policy, store
from local_operator.asks.queue import AskQueue
from local_operator.asks.render import mirror_card
from local_operator.mobile.projection import ProjectionFold, fold_messages_to_entries
from local_operator.mobile.types import (
    PendingAskWire,
    SessionProjection,
    _projection_from_json,
    ask_mirror_request,
)
from local_operator.session.frontend_state import (
    FrontendSessionState,
    PendingAskState,
    ask_wire,
)

BASE = 1_700_000_000_000


class FakeTranscript:
    def __init__(self) -> None:
        self.ids: set[str] = set()

    def has_entry(self, entry_id: str) -> bool:
        return entry_id in self.ids


class FakeSession:
    """A session double exposing exactly what the wire path reads.

    ``ask_queue`` is a CALLABLE returning the real queue — the same shape
    ``Session.ask_queue`` has — so the flag gate and the "no host" gate are
    exercised as the production code spells them.
    """

    def __init__(self, queue: AskQueue | None) -> None:
        self.transcript = FakeTranscript()
        self._queue = queue
        self.batches: list[list[Any]] = []
        self.state: list[tuple[Any, Any]] = []
        self.spawned: list[Any] = []

    def ask_queue(self) -> AskQueue | None:
        return self._queue

    def ask_reach(self) -> list[str]:
        return []

    def publish_ask_state(self) -> None:
        self.state.append(ask_wire(self))

    def _spawn_background(self, coro: Any) -> Any:
        """Record-and-close, rather than schedule.

        The queue schedules a reconcile on every terminal transition; a double
        that really ran it would leave a live task behind on the loop the test
        then closes (and a warning at interpreter shutdown that the next reader
        has to explain). The mirror test drives ``reconcile_asks`` itself, so
        nothing here depends on the scheduled copy having run.
        """
        self.spawned.append(coro)
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        return None

    async def deliver_ask_messages(self, messages: Any) -> None:
        self.batches.append(list(messages))
        for message in messages:
            self.transcript.ids.add(message.id)


def _questions(
    count: int = 1, *, text: str = "Which one?", secret: bool = False
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"key-{index}" if secret else f"q{index}",
            "question": f"{text} ({index})" if count > 1 else text,
            "options": [] if secret else [{"label": "yes"}, {"label": "no"}],
            "multi": False,
            "secret": secret,
            "persist": False,
            "recommended": None,
        }
        for index in range(count)
    ]


def _queue(tmp_path: Path, session: FakeSession, now: int = BASE) -> AskQueue:
    return AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: now)


def _live_session(tmp_path: Path, now: int = BASE) -> tuple[FakeSession, AskQueue]:
    """A session with a real queue that samples the clock the test moves."""
    clock = {"now": now}
    session = FakeSession(None)
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: clock["now"])
    session._queue = queue
    setattr(queue, "_test_clock", clock)
    return session, queue


# ---------------------------------------------------------------------------
# N2 — presence is the capability proxy
# ---------------------------------------------------------------------------


def test_the_frontend_state_omits_the_ask_fields_while_dark() -> None:
    """Flag off ⇒ the keys are ABSENT from the serialized state, not empty.

    The whole A2→F rollout rests on this: the field would otherwise ship while
    the server default was still blocking, and a new client would take the
    queued path against a blocking backend.
    """
    state = FrontendSessionState(session_id="s1", epoch="e")
    payload = state.model_dump(mode="json")
    assert "asks" not in payload
    assert "asks_open" not in payload


def test_the_frontend_state_publishes_them_once_live() -> None:
    state = FrontendSessionState(
        session_id="s1",
        epoch="e",
        asks=[PendingAskState(ask_id="a-1", status="open")],
        asks_open=1,
    )
    payload = state.model_dump(mode="json")
    assert payload["asks"][0]["ask_id"] == "a-1"
    assert payload["asks_open"] == 1


def test_the_projection_omits_them_while_dark_and_publishes_when_live() -> None:
    projection = SessionProjection(session_id="s1", pid=1, kind="daemon")
    assert "asks" not in projection.to_json()
    assert "asks_open" not in projection.to_json()
    projection.asks = [PendingAskWire(ask_id="a-1", status="open")]
    projection.asks_open = 1
    payload = projection.to_json()
    assert payload["asks"][0]["ask_id"] == "a-1"
    assert payload["asks_open"] == 1


def test_ask_wire_is_absence_without_a_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No queue (flag off, or a host with no ask surface) ⇒ ``(None, None)``."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    assert ask_wire(FakeSession(None)) == (None, None)

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
    session, _queue_ = _live_session(tmp_path)
    assert ask_wire(session) == (None, None)


def test_ask_wire_reports_the_fold_and_the_outstanding_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    session, queue = _live_session(tmp_path)
    queue.enqueue(_questions(), 600)
    queue.enqueue(_questions(text="Another?"), 600)
    rows, outstanding = ask_wire(session)
    assert rows is not None and outstanding == 2
    assert [row["status"] for row in rows] == ["open", "open"]
    assert all("ask_id" in row for row in rows)
    # The full question rides (options and flags included), because a surface
    # that can only see an id cannot draw a picker.
    assert rows[0]["questions"][0]["options"][0]["label"] == "yes"


def test_a_timed_out_but_unanswered_ask_still_counts_as_outstanding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The flip precondition: ``asks_open`` counts OUTSTANDING asks.

    An ask that timed out and was never answered used to drop out of the tally
    (0, and with no other row present the queue read as absent everywhere but
    the live bar) while the bar still offered it and a late answer still reached
    the agent (design §2.2, the spec's item 3). It is outstanding while the user
    can still act on it; answering it late settles it and the tally falls.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    clock = {"now": BASE}
    session = FakeSession(None)
    queue = AskQueue(session, config_dir=tmp_path, session_id="s1", clock=lambda: clock["now"])
    session._queue = queue
    ask_id = queue.enqueue(_questions(), 600)["details"]["ask_id"]

    clock["now"] += 600_001  # past the deadline, inside the 7-day late window
    rows, outstanding = ask_wire(session)
    assert rows is not None
    assert [row["status"] for row in rows] == ["timed_out"]
    assert outstanding == 1, "a timed-out ask the user can still answer is outstanding"

    assert queue.respond(ask_id, {"q0": ["late answer"]}, by="phone")["ok"] is True
    rows, outstanding = ask_wire(session)
    assert rows is not None
    assert outstanding == 0, "an answered-late ask is settled"
    assert [row["status"] for row in rows] == ["late"]


def test_the_queue_publishes_through_the_session_on_every_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The queue's own change path is what makes the wire move without a refresh."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    session, queue = _live_session(tmp_path)
    outcome = queue.enqueue(_questions(), None)
    assert outcome["ok"] is True
    assert session.state, "enqueue must publish the ask state"
    rows, opens = session.state[-1]
    assert opens == 1
    queue.decline(str(outcome["details"]["ask_id"]), by="test")
    rows, opens = session.state[-1]
    assert opens == 0 and rows[0]["status"] == "declined"


# ---------------------------------------------------------------------------
# the fold's ordering and cap (the frozen wire shape's own rules)
# ---------------------------------------------------------------------------


def test_open_asks_come_first_and_newest_first(tmp_path: Path) -> None:
    session, queue = _live_session(tmp_path)
    clock = getattr(queue, "_test_clock")
    first = queue.enqueue(_questions(text="first"), None)["details"]["ask_id"]
    clock["now"] += 1000
    second = queue.enqueue(_questions(text="second"), None)["details"]["ask_id"]
    clock["now"] += 1000
    queue.decline(first, by="test")
    rows = queue.projection()
    # The OPEN one leads, even though the declined one is older; among equals
    # (and within each status) the newest leads.
    assert [row["ask_id"] for row in rows] == [second, first]


def test_the_projection_is_capped_at_the_frozen_twenty(tmp_path: Path) -> None:
    """The wire list is ``PROJECTION_CAP`` rows, newest first, open ones in front.

    Written straight to the LOG rather than through ``enqueue``: the enqueue path
    refuses past ``OPEN_ASK_CAP`` (8) opens, so a 20-row list is only reachable
    once earlier asks have settled — which is exactly the state the cap exists
    for (a week of questions, most of them answered).
    """
    session, queue = _live_session(tmp_path)
    session._queue = queue
    for index in range(policy.PROJECTION_CAP + 5):
        ask_id = f"a-{index:02d}"
        at = BASE + index * 1000
        store.append_event(
            queue.session_dir,
            {
                "v": store.EVENT_SCHEMA,
                "kind": store.EVENT_QUEUED,
                "at": at,
                "ask_id": ask_id,
                "created_at": at,
                "expires_at": at + 3_600_000,
                "timeout_s": 3600,
                "urgent": False,
                "questions": _questions(text=f"ask {index}"),
            },
        )
        store.append_event(
            queue.session_dir,
            {
                "v": store.EVENT_SCHEMA,
                "kind": store.EVENT_DECLINED,
                "at": at + 10,
                "ask_id": ask_id,
                "by": {"surface": "test"},
            },
        )
    # One OPEN ask, the OLDEST of the settled batch: it must lead the list even
    # though it is the least recent, because "open first" is the ordering rule.
    store.append_event(
        queue.session_dir,
        {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_QUEUED,
            "at": BASE + 900,
            "ask_id": "a-open",
            "created_at": BASE + 900,
            "expires_at": BASE + 3_600_000,
            "timeout_s": 3600,
            "urgent": False,
            "questions": _questions(text="still open"),
        },
    )
    rows = queue.projection()
    assert len(rows) == policy.PROJECTION_CAP
    assert rows[0]["ask_id"] == "a-open" and rows[0]["status"] == "open"
    assert [row["ask_id"] for row in rows[1:4]] == ["a-24", "a-23", "a-22"]


# ---------------------------------------------------------------------------
# the legacy mirror (§4): one card, three publishers, one spelling
# ---------------------------------------------------------------------------


def test_the_mirror_card_is_the_oldest_open_ask(tmp_path: Path) -> None:
    session, queue = _live_session(tmp_path)
    clock = getattr(queue, "_test_clock")
    first = queue.enqueue(_questions(text="oldest"), None)["details"]["ask_id"]
    clock["now"] += 1000
    queue.enqueue(_questions(text="newest"), None)
    card = mirror_card(queue.projection())
    assert card is not None
    assert card["ask_id"] == first
    assert card["request_id"] == store.mirror_request_id(first, 0)
    assert card["title"] == "oldest"


def test_the_mirror_card_is_none_when_nothing_is_open(tmp_path: Path) -> None:
    session, queue = _live_session(tmp_path)
    ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
    queue.decline(ask_id, by="test")
    assert mirror_card(queue.projection()) is None


def test_the_fold_fronts_the_mirrored_card_and_keeps_pending_count_approvals(
    tmp_path: Path,
) -> None:
    """Publisher 3 (``_sync_pending``): the phone's card, and the count rule."""
    projection = SessionProjection(session_id="s1", pid=1, kind="daemon")
    fold = ProjectionFold(projection)
    fold.set_asks(
        [
            {
                "ask_id": "a-9",
                "status": "open",
                "created_at": 5,
                "questions": [
                    {"id": "q0", "question": "Which env?", "options": [{"label": "stg"}]}
                ],
            }
        ],
        1,
    )
    assert projection.pending is not None
    assert projection.pending.request_id == "a-9.0"
    assert projection.pending.kind == "ask"
    assert projection.pending.title == "Which env?"
    assert projection.pending.question_total == 1
    # ``pending_count`` stays the APPROVAL queue's length: an open ask is not a
    # blocking gate, and a badge that counted it would tell the user an approval
    # was waiting when the agent had merely asked.
    assert projection.pending_count == 0
    assert projection.asks_open == 1


def test_the_mirror_ignores_a_timed_out_ask(tmp_path: Path) -> None:
    """A dead ask must not be painted as "waiting for you" (§4)."""
    projection = SessionProjection(session_id="s1", pid=1, kind="daemon")
    fold = ProjectionFold(projection)
    fold.set_asks([{"ask_id": "a-9", "status": "timed_out", "created_at": 5, "questions": []}], 0)
    assert projection.pending is None


def test_absence_clears_the_mirror_without_a_false_count() -> None:
    projection = SessionProjection(session_id="s1", pid=1, kind="daemon")
    fold = ProjectionFold(projection)
    fold.set_asks([{"ask_id": "a-9", "status": "open", "created_at": 5, "questions": []}], 1)
    fold.set_asks(None, None)
    assert projection.asks is None and projection.asks_open is None
    assert projection.pending is None


def test_the_mirrored_request_id_maps_back_to_the_queue() -> None:
    """The one spelling, both directions — a drift here loses old clients' taps."""
    assert store.parse_mirror_request_id(store.mirror_request_id("a-3f9c", 2)) == ("a-3f9c", 2)
    # Everything that is NOT a mirrored id, including a live picker's hex token
    # and an approval's id, must fall through to the blocking path.
    assert store.parse_mirror_request_id("9f1c2ab3") is None
    assert store.parse_mirror_request_id("a-3f9c") is None
    assert store.parse_mirror_request_id("a-3f9c.x") is None
    assert store.parse_mirror_request_id("") is None


def test_the_mirror_helper_builds_the_card_the_old_client_answers() -> None:
    card = mirror_card(
        [
            {
                "ask_id": "a-3f9c",
                "status": "open",
                "created_at": 7,
                "questions": [
                    {
                        "id": "q0",
                        "question": "Deploy where?",
                        "options": [{"label": "stg"}, {"label": "prod", "description": "careful"}],
                    }
                ],
            }
        ]
    )
    request = ask_mirror_request(card or {})
    assert request.request_id == "a-3f9c.0"
    assert request.kind == "ask"
    assert [option.label for option in request.options] == ["stg", "prod"]
    assert request.options[1].description == "careful"


# ---------------------------------------------------------------------------
# the skew matrix (§4/§7) — both directions, exercised for real
# ---------------------------------------------------------------------------


def test_an_old_core_gives_a_new_client_nothing_to_render() -> None:
    """old core + new client: no ``asks`` key ⇒ absence ⇒ today's view."""
    projection = _projection_from_json(
        {"session_id": "s1", "pid": 0, "kind": "daemon", "pending_kind": "approval"},
        _record(),
    )
    assert projection.asks is None
    assert projection.asks_open is None
    assert "asks" not in projection.to_json()


def test_a_new_core_reaches_an_old_client_through_the_mirror() -> None:
    """new core + old client: the legacy ``pending`` card still answers."""
    projection = SessionProjection(session_id="s1", pid=1, kind="daemon")
    ProjectionFold(projection).set_asks(
        [{"ask_id": "a-1", "status": "open", "created_at": 1, "questions": _questions()}], 1
    )
    payload = projection.to_json()
    assert payload["pending"]["kind"] == "ask"
    assert payload["pending"]["request_id"] == "a-1.0"
    # The old client's answer path reads ``pending.request_id`` and sends
    # ``ask_answer``; the synthetic id is what the runtime maps onto the queue.
    assert store.parse_mirror_request_id(payload["pending"]["request_id"]) == ("a-1", 0)


def test_the_mirrored_answer_resolves_onto_the_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runtime's ``ask_answer`` half of the skew matrix, exercised for real.

    ``ServingSessionHandle._mirror_ask_answer`` is called with a session double
    that owns a REAL queue: the mapped answer must land as an ``answered`` event
    on the log, under the QUESTION id the harness asked with rather than the
    synthetic request id the old client echoed back.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    session, queue = _live_session(tmp_path)
    outcome = queue.enqueue(_questions(text="Deploy where?"), None)
    ask_id = outcome["details"]["ask_id"]

    def _respond(ask: str, answers: Any, by: str = "unknown") -> Any:
        return queue.respond(ask, answers, by=by)

    def _decline(ask: str, by: str = "unknown") -> Any:
        return queue.decline(ask, by=by)

    def _answer_one(ask: str, key: str, values: Any, by: str = "unknown") -> Any:
        return queue.answer_one(ask, key, values, by=by)

    session.respond_ask = _respond  # type: ignore[attr-defined]
    session.decline_ask = _decline  # type: ignore[attr-defined]
    session.answer_ask_question = _answer_one  # type: ignore[attr-defined]

    class _Handle:
        _session = session

    async def _reconcile(now: int | None = None) -> None:
        await queue.reconcile(now)

    session.reconcile_asks = _reconcile  # type: ignore[attr-defined]

    async def _drive() -> str | None:
        detail = await ServingSessionHandle._mirror_ask_answer(
            cast(Any, _Handle()), store.mirror_request_id(ask_id, 0), "yes"
        )
        return detail

    detail = asyncio.run(_drive())
    assert detail == "answered"
    events = store.read_events(queue.session_dir)
    answered = [event for event in events if event["kind"] == store.EVENT_ANSWERED]
    assert len(answered) == 1
    assert answered[0]["answers"] == {"q0": ["yes"]}
    record = queue.find(ask_id)
    assert record is not None and record["status"] == "answered"


def test_a_non_mirrored_request_id_is_left_to_the_blocking_path() -> None:
    """``None`` ⇒ the caller's picker/approval logic runs, untouched."""
    from local_operator.session.runtime.serving import ServingSessionHandle

    class _Handle:
        _session = None  # type: ignore[assignment]

    assert (
        asyncio.run(
            ServingSessionHandle._mirror_ask_answer(cast(Any, _Handle()), "not-an-ask", "yes")
        )
        is None
    )


# ---------------------------------------------------------------------------
# the fold's rows (§4: EntryKind ask_response / ask_timeout)
# ---------------------------------------------------------------------------


def test_the_timeout_row_carries_the_frozen_details() -> None:
    from local_operator.harness.message_types import ASK_TIMEOUT_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage

    message = CustomMessage(
        custom_type=ASK_TIMEOUT_MESSAGE_TYPE,
        id="ask-timeout-a-1",
        details={
            "ask_id": "a-1",
            "status": "timed_out",
            "waited_s": 600,
            "urgent": True,
            "text": "the model's own notice",
        },
    )
    rows = fold_messages_to_entries([message])
    assert len(rows) == 1
    assert rows[0].kind == "ask_timeout"
    assert rows[0].details["ask_id"] == "a-1"
    assert rows[0].details["status"] == "timed_out"
    assert rows[0].details["waited_s"] == 600
    assert rows[0].details["urgent"] is True
    assert rows[0].details["text"] == "the model's own notice"


def test_the_response_row_carries_the_answer_payload() -> None:
    from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage

    message = CustomMessage(
        custom_type=ASK_RESPONSE_MESSAGE_TYPE,
        id="ask-response-a-1",
        details={
            "ask_id": "a-1",
            "status": "late",
            "questions": _questions(),
            "answers": {"q0": ["yes"]},
            "at": BASE,
            "text": "the answer",
        },
    )
    rows = fold_messages_to_entries([message])
    assert rows[0].kind == "ask_response"
    assert rows[0].details["status"] == "late"
    assert rows[0].details["answers"] == {"q0": ["yes"]}
    assert rows[0].details["questions"][0]["id"] == "q0"


# ---------------------------------------------------------------------------
# the aggregate view (§4: GET /api/asks, GET /v1/desktop/asks)
# ---------------------------------------------------------------------------


def test_the_aggregate_read_is_index_backed_and_names_its_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    session, queue = _live_session(tmp_path)
    queue.enqueue(_questions(text="Where?"), None)
    # ``now`` is the test clock, not the wall clock: the aggregate read applies
    # the SAME 7-day horizon the writer does, so a 2023-dated fixture read at
    # 2026's real time would be swept as expired — which is the rule working.
    rows = store.index_asks(tmp_path, now=BASE)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["status"] == "open"
    # No session object, no runtime: the index alone answers, which is the whole
    # reason the aggregate view can be served from a cold process.
    assert store.index_asks(tmp_path / "nothing-here", now=BASE) == []


def test_the_aggregate_read_puts_open_asks_first_across_sessions(tmp_path: Path) -> None:
    settled = store.pending_row(
        {
            "ask_id": "a-settled",
            "created_at": BASE - 1000,
            "expires_at": BASE,
            "status": store.STATUS_DECLINED,
            "questions": [],
        }
    )
    open_row = store.pending_row(
        {
            "ask_id": "a-open",
            "created_at": BASE - 5000,
            "expires_at": BASE + 10_000,
            "status": store.STATUS_OPEN,
            "questions": [],
        }
    )
    for session_id in ("s-settled", "s-open"):
        # The session DIRECTORY is what keeps an entry alive: an entry whose
        # session dir is gone is an orphan and the reader sweeps it (the TTL rule
        # the index owns, and the reason the aggregate read cannot invent rows).
        (tmp_path / store.SESSIONS_DIRNAME / session_id).mkdir(parents=True)
    store.write_entry(tmp_path, "s-settled", cwd="/tmp/one", asks=[settled])
    store.write_entry(tmp_path, "s-open", cwd="/tmp/two", asks=[open_row])
    rows = store.index_asks(tmp_path, now=BASE)
    assert [row["ask_id"] for row in rows] == ["a-open", "a-settled"]
    assert {row["session_id"] for row in rows} == {"s-open", "s-settled"}
    assert {row["cwd"] for row in rows} == {"/tmp/one", "/tmp/two"}


def test_the_aggregate_read_applies_the_one_horizon(tmp_path: Path) -> None:
    """Past ``expires_at + 7 d`` the row is gone from every reader, not just this one."""
    stale = store.pending_row(
        {
            "ask_id": "a-old",
            "created_at": BASE - 100_000_000,
            "expires_at": BASE - (store.LATE_WINDOW_S * 1000) - 1000,
            "status": store.STATUS_ANSWERED,
            "questions": [],
        }
    )
    (tmp_path / store.SESSIONS_DIRNAME / "s1").mkdir(parents=True)
    store.write_entry(tmp_path, "s1", cwd="/tmp", asks=[stale])
    assert store.index_asks(tmp_path, now=BASE) == []


def _record() -> Any:
    """A minimal ``SessionRecord`` stand-in for the wire rebuild."""
    from local_operator.session.runtime.types import SessionRecord

    return SessionRecord(
        session_id="s1",
        pid=1,
        kind="daemon",
        conversation_name="",
        cwd="/tmp",
        model_label="",
        control_port=0,
        control_key="",
    )


# ---------------------------------------------------------------------------
# the desktop answer body (§4: the additive fields on POST .../answers)
# ---------------------------------------------------------------------------


def test_the_answers_body_accepts_all_three_shapes() -> None:
    """The gate shapes are unchanged; the queued-ask shape needs no epoch."""
    from local_operator.server.routes.desktop_sessions import Answer

    approval = Answer(epoch="e1", request_id="r1", approved=True)
    assert approval.approved is True and approval.ask_id is None

    picker = Answer(epoch="e1", request_id="r1", value="yes", question_index=0)
    assert picker.value == "yes" and picker.question_index == 0

    # No epoch: an ask outlives the owner that queued it, so its answer cannot
    # be required to name one.
    queued = Answer(ask_id="a-1", answers={"q0": ["yes"]})
    assert queued.epoch == "" and queued.answers == {"q0": ["yes"]}

    declined = Answer(ask_id="a-1", decline=True)
    assert declined.decline is True


def test_the_answers_body_refuses_the_shapes_that_mix_rules() -> None:
    from pydantic import ValidationError

    from local_operator.server.routes.desktop_sessions import Answer

    # A gate answer still needs both identity fields: the queued-ask shape
    # loosened nothing about the two blocking ones.
    bodies: tuple[dict[str, Any], ...] = (
        {"request_id": "r1", "approved": True},
        {"epoch": "e1", "approved": True},
        {"epoch": "e1", "request_id": "r1"},
        {"epoch": "e1", "request_id": "r1", "value": "x"},
        {"epoch": "e1", "request_id": "r1", "approved": True, "question_index": 0},
        {"ask_id": "a-1"},
        {"ask_id": "a-1", "answers": {}},
        {"ask_id": "a-1", "answers": {"q0": "yes"}},
        {"ask_id": "a-1", "answers": {"q0": ["yes"]}, "decline": True},
    )
    for body in bodies:
        with pytest.raises(ValidationError):
            Answer(**body)


# ---------------------------------------------------------------------------
# the legacy path is INCREMENTAL, the new ops stay ATOMIC (§4 A2 addendum)
# ---------------------------------------------------------------------------


def _multi_question_ask(queue: Any, count: int = 2) -> str:
    outcome = queue.enqueue(_questions(count, text="Which one?"), None)
    assert outcome["ok"] is True
    return str(outcome["details"]["ask_id"])


def _status(queue: Any, ask_id: str) -> str:
    """The ask's folded status, with a missing record as the assertion itself.

    ``AskQueue.find`` returns ``None`` for an id the log no longer holds, so a
    bare ``queue.find(id)["status"]`` is a subscript on an optional — and a
    silently-missing record would raise a ``TypeError`` the test reads as a
    crash rather than as the wrong outcome it is.
    """
    record = queue.find(ask_id)
    assert record is not None, f"{ask_id} is not on this queue"
    return str(record["status"])


def test_the_legacy_path_settles_a_multi_question_ask_one_tap_at_a_time(tmp_path: Path) -> None:
    """The review round 1 blocker: the mirror could SEE multi-question asks and
    never settle one, because every tap was a partial map for an atomic op."""
    session, queue = _live_session(tmp_path)
    ask_id = _multi_question_ask(queue)

    first = queue.answer_one(ask_id, "q0", ["yes"], by="mirror")
    assert first == {"ok": True, "settled": False, "waiting": ["q1"]}
    assert _status(queue, ask_id) == "open", "a partial must not settle the ask"
    assert (
        store.read_events(queue.session_dir)[-1]["kind"] == store.EVENT_QUEUED
    ), "nothing durable yet"

    second = queue.answer_one(ask_id, "q1", ["no"], by="mirror")
    assert second["ok"] is True and second["settled"] is True
    record = queue.find(ask_id)
    assert record is not None and record["status"] == "answered"
    answered = [
        e for e in store.read_events(queue.session_dir) if e["kind"] == store.EVENT_ANSWERED
    ]
    # ONE atomic write, carrying BOTH cells: the log never holds a partial.
    assert len(answered) == 1
    assert answered[0]["answers"] == {"q0": ["yes"], "q1": ["no"]}


def test_a_repeat_tap_on_the_same_question_is_refused(tmp_path: Path) -> None:
    session, queue = _live_session(tmp_path)
    ask_id = _multi_question_ask(queue)
    assert queue.answer_one(ask_id, "q0", ["yes"], by="mirror")["ok"] is True
    again = queue.answer_one(ask_id, "q0", ["no"], by="mirror")
    assert again["ok"] is False and "already answered" in again["error"]
    assert _status(queue, ask_id) == "open"


def test_an_unknown_question_id_is_refused_by_name(tmp_path: Path) -> None:
    session, queue = _live_session(tmp_path)
    ask_id = _multi_question_ask(queue)
    outcome = queue.answer_one(ask_id, "nope", ["yes"], by="mirror")
    assert outcome["ok"] is False and "not a question" in outcome["error"]


def test_the_partial_rides_the_published_rows_so_the_card_advances(tmp_path: Path) -> None:
    """The mirrored card must move off the question the old client just tapped."""
    session, queue = _live_session(tmp_path)
    ask_id = _multi_question_ask(queue)
    before = mirror_card(queue.projection(drafts=True))
    assert before is not None and before["request_id"] == f"{ask_id}.0"
    queue.answer_one(ask_id, "q0", ["yes"], by="mirror")
    after = mirror_card(queue.projection(drafts=True))
    assert after is not None and after["request_id"] == f"{ask_id}.1"
    assert after["title"].endswith("(1)")
    # ...and the DURABLE index never sees the draft: an answer a runtime death
    # would erase must not be presented as a settled one to a cross-session view.
    index_rows = store.index_asks(tmp_path, now=BASE)
    assert index_rows and "draft_question_ids" not in index_rows[0]
    assert "answers" not in index_rows[0]


def test_the_new_ops_still_refuse_a_partial_map(tmp_path: Path) -> None:
    """The atomic contract is unchanged: only the mirror's bridge is incremental."""
    session, queue = _live_session(tmp_path)
    ask_id = _multi_question_ask(queue)
    outcome = queue.respond(ask_id, {"q0": ["yes"]}, by="desktop")
    assert outcome["ok"] is False and "has no entry" in outcome["error"]
    assert _status(queue, ask_id) == "open"


def test_a_declined_ask_drops_its_draft(tmp_path: Path) -> None:
    session, queue = _live_session(tmp_path)
    ask_id = _multi_question_ask(queue)
    queue.answer_one(ask_id, "q0", ["yes"], by="mirror")
    assert queue.decline(ask_id, by="mirror")["ok"] is True
    assert queue.draft_question_ids(ask_id) == []


def test_a_deadline_reclaims_the_partial_and_the_closed_row_carries_no_draft(
    tmp_path: Path,
) -> None:
    """Review round 2's one new minor: the timed-out transition is DERIVED in the
    fold, so it never passes through ``_settled`` — the tap had to be reclaimed
    where the deadline is observed, and a closed ask's row must never state it."""
    session, queue = _live_session(tmp_path)
    clock = getattr(queue, "_test_clock")
    ask_id = _multi_question_ask(queue)
    queue.answer_one(ask_id, "q0", ["yes"], by="mirror")

    # While the ask is OPEN the tap rides the rows — that is what advances the
    # mirrored card — so this cell discriminates rather than asserting absence
    # against a field that is never there.
    open_row = next(row for row in queue.projection(drafts=True) if row["ask_id"] == ask_id)
    assert open_row["draft_question_ids"] == ["q0"]

    # Let the deadline elapse WITHOUT a reconcile first: this is the window the
    # review measured, where the fold already says ``timed_out`` (it reads the
    # clock) while the tap is still in memory, and a publish in it folds the
    # clock rather than waiting for the tick.
    record = queue.find(ask_id)
    assert record is not None
    clock["now"] = int(record["expires_at"]) + 1
    late_row = next(row for row in queue.projection(drafts=True) if row["ask_id"] == ask_id)
    assert late_row["status"] == store.STATUS_TIMED_OUT
    assert "draft_question_ids" not in late_row

    # ...and the live reconcile (the armed ``ask_timeout`` wake, and the boot
    # drain) RECLAIMS the entry rather than leaving it to the runtime's lifetime.
    asyncio.run(queue.reconcile(clock["now"]))
    assert queue.draft_question_ids(ask_id) == []


def test_the_session_bridge_stores_a_secret_at_the_tap_and_keeps_the_key(tmp_path: Path) -> None:
    """The legacy path must not put a secret value anywhere durable, not even in
    a draft held across taps."""
    session, queue = _live_session(tmp_path)
    outcome = queue.enqueue(_questions(1, secret=True), None)
    assert outcome["ok"] is True
    ask_id = str(outcome["details"]["ask_id"])
    stored: dict[str, list[str]] = {}

    class _Variables:
        def store(self, name: str, value: str) -> None:  # pragma: no cover - shape only
            stored[name] = [value]

    class _Session(FakeSession):
        _variables = _Variables()

        def journal_credential_change(self, *args: Any, **kwargs: Any) -> None:
            return None

    # ``answer_ask_question`` needs the real substitution hop, so it is exercised
    # through a small stand-in session rather than the module-level double.
    from local_operator.session.session import Session

    holder = _Session(queue)
    holder._queue = queue
    monkey = Session.answer_ask_question
    outcome = monkey(cast(Any, holder), ask_id, "key-0", ["sk-live-value"], by="mirror")
    assert outcome["ok"] is True and outcome["settled"] is True
    events = store.read_events(queue.session_dir)
    answers = [e for e in events if e["kind"] == store.EVENT_ANSWERED][0]["answers"]
    assert answers == {"key-0": ["key-0"]} or "sk-live-value" not in json.dumps(events)
    assert "sk-live-value" not in json.dumps(events)


# ---------------------------------------------------------------------------
# the wire bound (§4 A2 addendum; review round 1 findings 2/4/5/8)
# ---------------------------------------------------------------------------


def _ask_row(index: int, *, questions: int = 1, text: str = "Which one?") -> dict[str, Any]:
    return {
        "ask_id": f"a-{index:04x}",
        "created_at": BASE + index,
        "expires_at": BASE + 3_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": "open",
        "delivered": False,
        "questions": [
            {
                "id": f"q{question}",
                "question": f"{text} {question}",
                "options": [{"label": "yes", "description": ""}],
                "multi": False,
                "secret": False,
                "persist": False,
                "recommended": None,
            }
            for question in range(questions)
        ],
    }


def test_the_bound_caps_counts_as_well_as_text() -> None:
    """The first revision exempted the first row from the budget entirely, so one
    ask with a hundred long questions could spend the whole frame."""
    from local_operator.session.frontend_state import (
        ASK_WIRE_OPTIONS_MAX,
        ASK_WIRE_QUESTIONS_MAX,
        bound_ask_rows,
    )

    row = _ask_row(0, questions=400, text="x" * 500)
    row["questions"][0]["options"] = [
        {"label": "y" * 300, "description": "z" * 300} for _ in range(200)
    ]
    kept, dropped = bound_ask_rows([row])
    assert dropped is True
    questions = kept[0]["questions"]
    assert len(questions) <= ASK_WIRE_QUESTIONS_MAX
    for question in questions:
        assert question["question"].endswith("…") and len(question["question"]) <= 201
        assert len(question["options"]) <= ASK_WIRE_OPTIONS_MAX
        for option in question["options"]:
            assert len(option["label"]) <= 61 and len(option["description"]) <= 81


def test_the_first_row_is_clipped_to_the_budget_rather_than_exempt() -> None:
    from local_operator.session.frontend_state import bound_ask_rows

    row = _ask_row(0, questions=6, text="q" * 5_000)
    kept, _dropped = bound_ask_rows([row], budget=1_000)
    charged = sum(len(q["question"]) for q in kept[0]["questions"])
    assert charged <= 1_000, charged


def test_the_snapshot_marks_a_clipped_list_rather_than_shipping_a_bare_count() -> None:
    """QA round 1 Q2: 20 long asks shipped as 9 rows beside ``asks_open: 20``."""
    from local_operator.session.frontend_state import (
        FrontendStateStore,
        sync_wire_payload,
    )

    state = FrontendSessionState(
        session_id="s1",
        epoch="e",
        asks=[
            PendingAskState(**row)  # type: ignore[arg-type]
            for row in [_ask_row(i, questions=3, text="q" * 400) for i in range(20)]
        ],
        asks_open=20,
    )
    store = FrontendStateStore(state)
    payload = sync_wire_payload(store.subscribe(lambda _u: None).sync)
    snapshot = payload["snapshot"]
    assert len(snapshot["asks"]) < 20, "the bound must have dropped something"
    assert snapshot["asks_truncated"] is True
    # The count keeps its meaning (the session's open asks); the flag is what
    # stops a client drawing a prefix as if it were the whole list.
    assert snapshot["asks_open"] == 20


def test_a_complete_list_carries_no_truncation_flag() -> None:
    from local_operator.session.frontend_state import (
        FrontendStateStore,
        sync_wire_payload,
    )

    state = FrontendSessionState(
        session_id="s1",
        epoch="e",
        asks=[  # type: ignore[arg-type]
            PendingAskState(**_ask_row(0)),
            PendingAskState(**_ask_row(1)),
        ],
        asks_open=2,
    )
    payload = sync_wire_payload(FrontendStateStore(state).subscribe(lambda _u: None).sync)
    assert "asks_truncated" not in payload["snapshot"]


def test_the_delta_route_is_bounded_too() -> None:
    """The queue's own change path drives ``mutate``, so a snapshot-only bound
    would hold for the first frame and leak on every one after it."""
    from local_operator.session.frontend_state import FrontendStateStore

    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e"))
    update = store.mutate(
        asks=[_ask_row(i, questions=3, text="q" * 400) for i in range(20)], asks_open=20
    )
    assert update is not None
    assert len(update.changes["asks"]) < 20
    assert update.changes["asks_truncated"] is True


def test_the_yield_measures_the_real_payload_not_a_rebuilt_envelope() -> None:
    """Review round 1 finding 2: the first revision rebuilt a push-shaped envelope
    from the snapshot, omitting ``epoch``/``sequence``/``live_cursor`` and using
    the wrong frame shape — ~220 B under against ~110 B of slack, so the branch
    could fail to fire and the whole frame would degrade instead."""
    from local_operator.session import frontend_state as fs

    state = FrontendSessionState(
        session_id="s1",
        epoch="e",
        asks=[PendingAskState(**_ask_row(0))],  # type: ignore[arg-type]
        asks_open=1,
    )
    payload = fs.sync_wire_payload(fs.FrontendStateStore(state).subscribe(lambda _u: None).sync)
    snapshot = payload["snapshot"]
    limit = fs._MODEL_CATALOGUE_LINE_LIMIT

    # Park the REAL frame just under the line...
    snapshot["cwd"] = "x" * max(0, limit - fs._frame_line_bytes(payload) - 40)
    assert fs._frame_line_bytes(payload) <= limit
    fs._yield_asks_when_the_frame_has_no_room(snapshot, payload)
    assert snapshot.get("asks"), "an under-line frame keeps its asks"

    # ...then push it over, by less than the envelope delta the old measurement
    # got wrong: the asks must give way (the old shape would have kept them).
    snapshot = payload["snapshot"]
    snapshot["cwd"] += "x" * 200
    assert fs._frame_line_bytes(payload) > limit
    fs._yield_asks_when_the_frame_has_no_room(snapshot, payload)
    assert "asks" not in snapshot and "asks_open" not in snapshot
    assert "asks_truncated" not in snapshot, "the flag goes with the rows it describes"
