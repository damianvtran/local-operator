"""The relay's half of the queued-ask wire (design §4/§5.3): the aggregate the
phone's asks sheet reads, the row count its list chip reads, and the command
route its answer form posts to.

WHY THIS FILE EXISTS. ``GET /api/asks``, ``SessionListRow.asks_open`` and the
``ask_respond``/``ask_decline``/``ask_dismiss`` ops all landed with the wire PR,
and the mobile web client (the phone surface) is what consumes them — so these
assertions are about what THE PHONE IS TOLD, over the real app and the real
index, not about the queue's own store (which has ``tests/unit/asks``).

The two presence rules are the ones worth pinning here, because both are
invisible in a green unit suite and load-bearing on the client:

* ``GET /api/asks`` is INDEX-BACKED: it must answer with no runtime running at
  all, because a queued ask outlives the runtime that asked it.
* ``asks_open`` is ABSENT when the runtime does not publish asks — and it may
  be a present ``0``, which the WIRE FIX made a LIVE value: a runtime running
  the queued engine with nothing waiting says exactly that, while a runtime
  that cannot say anything at all omits the key. The phone's capability proxy
  is the presence of ``asks`` OR ``asks_open``, never the count.

Nothing here touches the operator's machine: the config root is the isolated
one ``conftest`` installs, the records are this test's own pids, and the one
real runtime is an in-process ``RuntimeServer`` on a loopback socket.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

import httpx
import pytest
from starlette.testclient import TestClient

from local_operator.asks import store as ask_store
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, _dial, build_app
from local_operator.mobile.types import SessionProjection, SessionRecord
from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.mobile.test_daemon import FakeHandle

SESSION_A = "aaa111222333"
SESSION_B = "bbb444555666"


def _client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _client_and_daemon() -> tuple[TestClient, MobileDaemon]:
    """``_client()`` plus the daemon, for the cells that read its list table."""
    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client, daemon


def _question(qid: str, text: str = "ship it?") -> dict[str, object]:
    return {
        "id": qid,
        "question": text,
        "options": [{"label": "yes", "description": ""}, {"label": "no", "description": ""}],
        "multi": False,
        "secret": False,
        "persist": False,
    }


def _pending(patch: dict[str, object] | None = None) -> dict[str, object]:
    now = int(time.time() * 1000)
    row: dict[str, object] = {
        "ask_id": "ask-open-1",
        "created_at": now,
        "expires_at": now + 900_000,
        "timeout_s": 900,
        "urgent": False,
        "status": "open",
        "delivered": False,
        "questions": [_question("q1")],
    }
    row.update(patch or {})
    return row


def test_the_aggregate_serves_nothing_before_anything_is_queued() -> None:
    assert _client().get("/api/asks").json() == {"asks": []}


def test_the_aggregate_is_index_backed_and_names_each_row_s_own_session() -> None:
    """Answered with no runtime running, and with the two facts a row needs to
    be answerable against its OWN conversation route."""
    _session_dir(SESSION_A)
    _session_dir(SESSION_B)
    ask_store.write_entry(
        config_dir(),
        SESSION_A,
        cwd="/tmp/aaa",
        asks=[
            _pending(),
            _pending({"ask_id": "ask-answered", "status": "answered", "delivered": True}),
            _pending({"ask_id": "ask-dismissed", "status": "dismissed"}),
            _pending({"ask_id": "ask-expired", "status": "expired"}),
        ],
    )
    ask_store.write_entry(
        config_dir(),
        SESSION_B,
        cwd="/tmp/bbb",
        asks=[_pending({"ask_id": "ask-open-b"})],
    )

    rows = _client().get("/api/asks").json()["asks"]
    ids = [row["ask_id"] for row in rows]
    # Dismissed and expired rows leave the population; what is left is the two
    # open asks and the answered one, and the OPEN ones LEAD (the route's own
    # rank, which is the order the phone's sheet renders without re-sorting).
    assert sorted(ids) == ["ask-answered", "ask-open-1", "ask-open-b"]
    assert set(ids[:2]) == {"ask-open-1", "ask-open-b"}
    by_id = {row["ask_id"]: row for row in rows}
    assert by_id["ask-open-1"]["session_id"] == SESSION_A
    assert by_id["ask-open-1"]["cwd"] == "/tmp/aaa"
    assert by_id["ask-open-b"]["session_id"] == SESSION_B


def _session_dir(session_id: str) -> None:
    """A session directory, because the index sweeps entries whose session is
    gone (``store.entry_is_stale``): a durable conversation is what makes an ask
    still answerable, and an entry with no session at all is an orphan the
    reader is right to drop."""
    (config_dir() / "sessions" / session_id).mkdir(parents=True, exist_ok=True)


def _record(session_id: str, pid: int) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=f"session {session_id}",
        cwd="/synthetic",
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )


def _entry(daemon: MobileDaemon, record: SessionRecord, projection: SessionProjection) -> None:
    entry = SessionEntry(record)
    entry.projection = projection
    daemon.session_projections[record.session_id] = projection
    daemon.table.entries[record.pid] = entry


def test_the_list_row_carries_asks_open_only_when_the_runtime_publishes_asks() -> None:
    """PRESENCE is the capability proxy (design §4, the WIRE FIX): a runtime that
    does not publish asks must be indistinguishable from an old one, so the key
    is OMITTED — and a LIVE runtime with nothing waiting sends a present ``0``,
    which is now legal and load-bearing (it is how the phone tells a live
    queued engine from a blocking one on an empty queue)."""
    _session_dir(SESSION_A)
    _session_dir(SESSION_B)
    _session_dir("sess-empty-live")
    daemon = MobileDaemon(port=0, password="pw123")
    publishing = SessionProjection(
        session_id=SESSION_A, pid=101, kind="tui", conversation_name="publishing"
    )
    publishing.asks = []  # type: ignore[assignment]
    publishing.asks_open = 2
    _entry(daemon, _record(SESSION_A, 101), publishing)

    silent = SessionProjection(
        session_id=SESSION_B, pid=102, kind="tui", conversation_name="silent"
    )
    _entry(daemon, _record(SESSION_B, 102), silent)

    # Live-but-empty: the queue runs, nothing is waiting. The tally rides as a
    # present 0 (the WIRE FIX); only a runtime that cannot answer omits it.
    empty_live = SessionProjection(
        session_id="sess-empty-live", pid=103, kind="tui", conversation_name="live empty"
    )
    empty_live.asks_open = 0
    _entry(daemon, _record("sess-empty-live", 103), empty_live)

    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    rows = {row["session_id"]: row for row in client.get("/api/sessions").json()["sessions"]}
    assert rows[SESSION_A]["asks_open"] == 2
    assert "asks_open" not in rows[SESSION_B]
    assert rows["sess-empty-live"]["asks_open"] == 0


class _AskHandle(FakeHandle):
    """The mobile suite's minimal handle, plus the queued-ask op family.

    SUBCLASSED rather than written again: ``test_daemon.FakeHandle`` already
    satisfies ``SessionHandle``, and a second hand-rolled stub set is both
    duplicated protocol surface and a standing pyright error (the first version
    of this file carried its own and failed the type gate with "``prompt`` is
    not present"). What this class ADDS is the three ops the queued wire
    introduced, recorded separately from the inherited ``calls`` so an
    assertion about an ask cannot be satisfied by some other op's record.

    The point of the cells below is that family's shape — atomic per ask,
    refusals carrying the queue's own sentence, and a runtime WITHOUT the ops
    refused in words. ``test_an_older_runtime_refuses_in_words`` gets the last
    one by shadowing ``ask_respond`` with ``None``, which is exactly the
    condition ``_dispatch``'s ``getattr`` probe tests for.
    """

    def __init__(self, *, refusal: str = "") -> None:
        super().__init__()
        self.ask_calls: list[tuple[str, dict[str, object]]] = []
        self._refusal = refusal
        self._projection = SessionProjection(
            session_id="s-ask",
            pid=0,
            kind="tui",
            conversation_name="asking",
            cwd="/tmp",
            model_label="test/model",
        )

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def _maybe_refuse(self) -> None:
        if self._refusal:
            raise ValueError(self._refusal)

    async def ask_respond(self, ask_id, answers, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.ask_calls.append(("ask_respond", {"ask_id": ask_id, "answers": answers, "by": by}))
        return "answered"

    async def ask_revise(self, ask_id, answers, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.ask_calls.append(("ask_revise", {"ask_id": ask_id, "answers": answers, "by": by}))
        return "revised"

    async def ask_decline(self, ask_id, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.ask_calls.append(("ask_decline", {"ask_id": ask_id, "by": by}))
        return "declined"

    async def ask_dismiss(self, ask_id, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.ask_calls.append(("ask_dismiss", {"ask_id": ask_id, "by": by}))
        return "dismissed"


@pytest.mark.asyncio
async def test_a_queued_ask_is_answered_through_the_command_route() -> None:
    """The phone's answer form posts ONE frame for the whole ask (§2.4: atomic
    per ask, so there is no per-question race to lose)."""
    handle = _AskHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        record = None
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            found = [pair for pair in registry.scan() if pair[1] == "live"]
            if found:
                record = found[0][0]
                break
            await asyncio.sleep(0.05)
        assert record is not None

        daemon = MobileDaemon(port=0, password="pw123")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            # Wait for the dial to be established (its first projection is the
            # receipt) before posting: a request sent into a half-open dial
            # times out, and the timeout would be the fixture's fault rather
            # than the route's.
            for _ in range(100):
                if entry.projection is not None:
                    break
                await asyncio.sleep(0.05)
            assert entry.projection is not None
            # ASGITransport on THIS loop, not TestClient: the daemon's dial
            # connection lives on the running loop, so a blocking client (which
            # drives the app on its own thread) would starve it and every
            # request would time out — a fixture artefact that reads exactly
            # like a route defect.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(daemon)),
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw123")},
            ) as client:
                reply = await client.post(
                    f"/api/sessions/{record.session_id}/command",
                    json={
                        "op": "ask_respond",
                        "ask_id": "ask-1",
                        "answers": {"q1": ["yes"], "q2": []},
                    },
                )
            assert reply.status_code == 200, reply.text
            assert reply.json()["ok"] is True
            assert handle.ask_calls[-1] == (
                "ask_respond",
                {"ask_id": "ask-1", "answers": {"q1": ["yes"], "q2": []}, "by": ""},
            )

            # §10 (#1936): the SAME atomic body over the SAME route, as a
            # REVISION. A separate op rather than a flag — the intent has to be
            # explicit on the wire, so the runtime never has to compare values.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(daemon)),
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw123")},
            ) as client:
                reply = await client.post(
                    f"/api/sessions/{record.session_id}/command",
                    json={
                        "op": "ask_revise",
                        "ask_id": "ask-1",
                        "answers": {"q1": ["no"], "q2": []},
                    },
                )
            assert reply.status_code == 200, reply.text
            assert reply.json()["ok"] is True
            assert handle.ask_calls[-1] == (
                "ask_revise",
                {"ask_id": "ask-1", "answers": {"q1": ["no"], "q2": []}, "by": ""},
            )
        finally:
            dial.cancel()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_a_refused_answer_keeps_the_queue_s_own_sentence() -> None:
    """Single-winner by the log: the loser is told WHO answered, in the queue's
    words — the phone renders ``error`` verbatim and must not paraphrase it."""
    handle = _AskHandle(refusal="already answered by desktop.")
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = [pair for pair in registry.scan() if pair[1] == "live"]
            if found:
                record = found[0][0]
                break
            await asyncio.sleep(0.05)
        assert record is not None

        daemon = MobileDaemon(port=0, password="pw123")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            # Wait for the dial to be established (its first projection is the
            # receipt) before posting: a request sent into a half-open dial
            # times out, and the timeout would be the fixture's fault rather
            # than the route's.
            for _ in range(100):
                if entry.projection is not None:
                    break
                await asyncio.sleep(0.05)
            assert entry.projection is not None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(daemon)),
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw123")},
            ) as client:
                reply = await client.post(
                    f"/api/sessions/{record.session_id}/command",
                    json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
                )
            assert reply.status_code == 422, reply.text
            assert reply.json()["error"] == "already answered by desktop."
        finally:
            dial.cancel()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_an_older_runtime_refuses_in_words_rather_than_with_a_stack_trace() -> None:
    """A handle without the op family (a build that predates queued asks) must
    answer with the sentence the phone shows, so the client can say why."""
    handle = _AskHandle()
    handle.ask_respond = None  # type: ignore[assignment]
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = [pair for pair in registry.scan() if pair[1] == "live"]
            if found:
                record = found[0][0]
                break
            await asyncio.sleep(0.05)
        assert record is not None
        daemon = MobileDaemon(port=0, password="pw123")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            # Wait for the dial to be established (its first projection is the
            # receipt) before posting: a request sent into a half-open dial
            # times out, and the timeout would be the fixture's fault rather
            # than the route's.
            for _ in range(100):
                if entry.projection is not None:
                    break
                await asyncio.sleep(0.05)
            assert entry.projection is not None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(daemon)),
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw123")},
            ) as client:
                reply = await client.post(
                    f"/api/sessions/{record.session_id}/command",
                    json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
                )
            assert reply.status_code == 422, reply.text
            assert "predates queued asks" in reply.json()["error"]
        finally:
            dial.cancel()
    finally:
        registrant.close()


def test_answering_an_ask_for_a_gone_conversation_says_it_can_never_be_read() -> None:
    """THE NEGATIVE, and it is as load-bearing as the engage itself.

    An ask outlives its runtime, but it does NOT outlive its CONVERSATION. A
    session id with no durable transcript has no log a runtime could own, so
    engaging cannot help: the answer could never be read. The route must say
    that in its own words rather than the generic "session not connected",
    because a surface that reported success for an unreadable answer is worse
    than one that refuses (design §2.4's durability is a promise about a log,
    not about a process).
    """
    client = _client()
    reply = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
    )
    assert reply.status_code == 409
    assert reply.json()["code"] == "ask_session_gone"
    assert "no longer exists" in reply.json()["error"]
    assert reply.json()["error"].endswith(".")


def test_the_gone_sentence_is_the_ops_own() -> None:
    """ONE code, THREE sentences — and the op picks (design round 1, D2 = UX U2;
    UX round 2, U9 for the third).

    ``dismiss — send no reply`` promises that NOTHING is sent, so answering it
    with "…ask again" prescribed the opposite of what the user asked for. A
    ``decline``/``revise`` was then given the same re-ask clause, which is
    addressed to someone who was not asking: handing the decision over leaves no
    question for the user to put again. The queue's refusals are per-STATE; this
    one is per-OP, because what is lost differs.
    """
    client = _client()
    answer = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
    )
    assert answer.json()["error"] == (
        "this conversation no longer exists — the ask can never be read; "
        "start a new conversation and ask again."
    )
    dismissed = client.post(
        f"/api/sessions/{SESSION_A}/command", json={"op": "ask_dismiss", "ask_id": "ask-1"}
    )
    assert dismissed.status_code == 409
    assert dismissed.json()["code"] == "ask_session_gone"
    assert dismissed.json()["error"] == (
        "this conversation no longer exists — nothing was sent; the ask can never be read."
    )
    # ``decline`` and ``revise`` deliver a DECISION: the ask is unreadable and the
    # user has nothing left to re-ask, so they take the handoff wording — which is
    # NOT the sentence a surface shows before asking: the phone's pre-emptive
    # state uses ``_ASK_GONE_RESPONSE`` (the answer's, remedy included) on a DEAD
    # row, because the row it draws offers no control whose op could be known.
    for op, body in (("ask_decline", {}), ("ask_revise", {"answers": {"q1": ["no"]}})):
        reply = client.post(
            f"/api/sessions/{SESSION_A}/command", json={"op": op, "ask_id": "ask-1", **body}
        )
        assert reply.json()["error"] == (
            "this conversation no longer exists — the ask can never be read."
        )


def test_the_aggregate_says_whether_answering_will_have_to_engage() -> None:
    """``runtime_live`` and ``durable`` are the two facts the card cannot derive
    (design round 1, D1/D4 = UX U1/U3/U4; design round 2, D6 = UX U7).

    An index row is durable but knows nothing about whether answering it is a
    warm round trip or a ~30 s engage — that is a property of the relay's own
    dial table. ``durable`` is the second half: `runtime_live` answers "can the
    relay deliver right now", not "will anything EVER read this", and over a
    conversation with no transcript both are false while the op is a terminal
    409. Each is absent (not false) from an older daemon, so a reader must treat
    absence as unknown.
    """
    for session, ask_id in ((SESSION_A, "ask-open-1"), (SESSION_B, "ask-open-b")):
        _session_dir(session)
    _session_dir("ccc777888999")
    ask_store.write_entry(config_dir(), SESSION_A, cwd="/tmp/aaa", asks=[_pending()])
    ask_store.write_entry(
        config_dir(), SESSION_B, cwd="/tmp/bbb", asks=[_pending({"ask_id": "ask-open-b"})]
    )
    ask_store.write_entry(
        config_dir(), "ccc777888999", cwd="/tmp/ccc", asks=[_pending({"ask_id": "ask-open-c"})]
    )
    # ONE durable conversation: a transcript is what the refusal predicate reads.
    _write_transcript(SESSION_A)

    daemon = MobileDaemon(port=0, password="pw123")
    # A dialled, CONNECTED owner for A...
    live = SessionEntry(_record(SESSION_A, 101))
    live.writer = object()  # type: ignore[assignment]
    daemon.table.entries[101] = live
    # ...and an entry for C that has not finished dialling (no writer): the op
    # cannot be delivered there either, so it is not live for this purpose.
    dialling = SessionEntry(_record("ccc777888999", 102))
    daemon.table.entries[102] = dialling

    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    rows = {row["ask_id"]: row for row in client.get("/api/asks").json()["asks"]}
    assert rows["ask-open-1"]["runtime_live"] is True
    assert rows["ask-open-b"]["runtime_live"] is False
    assert rows["ask-open-c"]["runtime_live"] is False
    assert rows["ask-open-1"]["durable"] is True
    assert rows["ask-open-b"]["durable"] is False
    assert rows["ask-open-c"]["durable"] is False


def _write_transcript(session_id: str) -> None:
    """A transcript is what ``_durable_user_session_dir`` looks for, so an ask
    can be answered INTO it; a session directory without one is the state the
    route refuses with ``ask_session_gone``."""
    (config_dir() / "sessions" / session_id / "transcript.jsonl").write_text("")


class _FakeAskClient:
    """The engaged client's half of the contract the relay's cold arm uses.

    Records the op and its body — the assertion the route exists to make is
    that the ANSWER reaches the freshly engaged runtime, so a stub whose calls
    are not inspectable would let the route pass while delivering nothing.
    """

    def __init__(self, *, refusal: str = "") -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._refusal = refusal
        self.closed = False

    def _maybe_refuse(self) -> None:
        if self._refusal:
            raise RuntimeError(self._refusal)

    def close(self) -> None:
        self.closed = True

    async def ask_respond(self, ask_id, answers, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.calls.append(("ask_respond", {"ask_id": ask_id, "answers": answers, "by": by}))
        return "answered"

    async def ask_revise(self, ask_id, answers, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.calls.append(("ask_revise", {"ask_id": ask_id, "answers": answers, "by": by}))
        return "revised"

    async def ask_decline(self, ask_id, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.calls.append(("ask_decline", {"ask_id": ask_id, "by": by}))
        return "declined"

    async def ask_dismiss(self, ask_id, by="") -> str:  # noqa: ANN001
        self._maybe_refuse()
        self.calls.append(("ask_dismiss", {"ask_id": ask_id, "by": by}))
        return "dismissed"


def _durable_session() -> None:
    """A runtime-less but DURABLE user session: the folder, a real transcript
    (the route's own gate reads it), and one open ask in the durable index."""
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    _session_dir(SESSION_A)
    asyncio.run(
        Transcript(config_dir() / "sessions" / SESSION_A).append_message(
            Message.user("existing", id="existing")
        )
    )
    ask_store.write_entry(config_dir(), SESSION_A, cwd="/tmp/gone", asks=[_pending()])


def _durable_ask_session(ask_id: str, *, expired: bool) -> None:
    """The same durable, runtime-less session, with a real ask IN THE LOG.

    The cold dismissal folds the LOG, so an index row with no ``queued`` event
    behind it is not dismissible at all (there is no ask) — a test that seeded
    only the index would be asserting the refusal while believing it tested the
    door. ``expired`` picks the two states a dismissal distinguishes: past its
    deadline (the one state it is offered on) and still inside it.
    """
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    _session_dir(SESSION_A)
    directory = config_dir() / "sessions" / SESSION_A
    asyncio.run(Transcript(directory).append_message(Message.user("existing", id="existing")))
    now = int(time.time() * 1000)
    ask_store.append_event(
        directory,
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "ask_id": ask_id,
            "at": now - 3_600_000,
            "created_at": now - 3_600_000,
            # Past the deadline and inside the 7 d window (the state a dismissal
            # is offered on, and the one whose notice is still owed), or still
            # running: the refusal cell's state.
            "expires_at": now - 60_000 if expired else now + 3_600_000,
            "timeout_s": 3600,
            "urgent": False,
            "tool_call_id": "",
            "questions": [_question("q1")],
        },
    )


def _cold_engage(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeAskClient, list[tuple[str, object]]]:
    """Stand up a durable, runtime-less session and record the engage.

    Returns ``(client, engagements)`` where each engagement is
    ``(session_id, errand)``. Patching ``engage_session_client`` — not the
    inner ``engage_runtime`` — keeps the assertion at the seam the route uses,
    which is what pins "the same engage path the prompt uses".
    """
    from local_operator.mobile import attach_client

    _durable_session()
    client = _FakeAskClient()
    engagements: list[tuple[str, object]] = []

    async def fake_engage(config_dir, session_id, work, **kwargs):  # noqa: ANN001, ANN003, ANN202
        engagements.append((session_id, work))
        return client, "runtime ready"

    monkeypatch.setattr(attach_client, "engage_session_client", fake_engage)
    return client, engagements


@pytest.mark.parametrize(
    "op,body,answered_with",
    [
        ("ask_respond", {"ask_id": "ask-1", "answers": {"q1": ["yes"]}}, "answered"),
        ("ask_revise", {"ask_id": "ask-1", "answers": {"q1": ["no"]}}, "revised"),
        ("ask_decline", {"ask_id": "ask-1"}, "declined"),
    ],
)
def test_an_answer_to_a_stopped_session_engages_it(monkeypatch, op, body, answered_with) -> None:
    """THE ENGAGE (design §2.4), for the three settle ops that NEED a runtime.

    The operator's bug was exactly this: a queued ask is durable so the agent
    can move on, and the relay REFUSED the answer once the runtime had exited —
    "session not connected" — so the answer was accepted by the phone and read
    by nobody. Decline settles the ask just as an answer does, so it takes the
    SAME arm: one engage path. ``ask_dismiss`` USED to be the fourth and is not
    any more (fork A, §14): the next cell pins its own arm, and the split is the
    point — an answer has to reach a runtime, a dismissal does not.
    """
    from local_operator.session.runtime.launch import AskErrand

    client, engagements = _cold_engage(monkeypatch)
    reply = _client().post(f"/api/sessions/{SESSION_A}/command", json={"op": op, **body})

    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "detail": answered_with}
    # The errand is the ASK errand, carrying the id for the log line, and the
    # session engaged is the one the phone answered against.
    assert len(engagements) == 1
    session_id, errand = engagements[0]
    assert session_id == SESSION_A
    assert isinstance(errand, AskErrand)
    assert errand.ask_id == "ask-1"
    # The answer reached the freshly engaged runtime with the atomic body.
    assert client.calls[-1][0] == op
    if op in ("ask_respond", "ask_revise"):
        assert client.calls[-1][1]["answers"] == body["answers"]
    # The engaged connection is released rather than left open by the route.
    assert client.closed is True


def test_a_cold_dismiss_never_engages(monkeypatch) -> None:
    """FORK A (§14): dismissing on a stopped conversation boots NOTHING.

    The engage arm exists because a queued ask's READER is a runtime. A dismissal
    has no reader: it is one append plus an index rewrite, squared against the
    fold, and the notice it may still owe is delivered by the next real reopen.
    Before the fork this op engaged (measured end to end: a boot, a deadline
    notice and a PAID TURN), which is the promise the phone's card was already
    making — "dismiss — send no reply".
    """
    client, engagements = _cold_engage(monkeypatch)
    _durable_ask_session("ask-lapsed", expired=True)

    reply = _client().post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_dismiss", "ask_id": "ask-lapsed"},
    )

    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "detail": "dismissed"}
    assert engagements == []
    assert client.calls == []
    # The dismissal is DURABLE — the phone's settled row is not a paint.
    events = ask_store.read_events(config_dir() / "sessions" / SESSION_A)
    assert [event["kind"] for event in events][-1] == ask_store.EVENT_DISMISSED


def _durable_lapsed_ask(ask_id: str) -> None:
    """A durable, runtime-less session whose LAPSED ask is in the LOG and the INDEX.

    Both halves are the point. ``dismiss_cold`` folds the LOG (an ask is only
    dismissible when the log says it lapsed), and the phone's screen reads the
    INDEX: ``_durable_projection`` builds a stopped conversation's ``asks`` from
    the derived index (``asks.store.session_asks``), which is what the dock and
    the header chip draw.
    """
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    _session_dir(SESSION_A)
    directory = config_dir() / "sessions" / SESSION_A
    asyncio.run(Transcript(directory).append_message(Message.user("existing", id="existing")))
    now = int(time.time() * 1000)
    ask_store.append_event(
        directory,
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "ask_id": ask_id,
            "at": now - 3_600_000,
            "created_at": now - 3_600_000,
            # Past the deadline (the only state a dismissal is offered on) and
            # well inside the 7 d window a lapsed row stays visible for.
            "expires_at": now - 60_000,
            "timeout_s": 3600,
            "urgent": False,
            "tool_call_id": "",
            "questions": [_question("q1")],
        },
    )
    ask_store.write_entry(
        config_dir(),
        SESSION_A,
        cwd=str(directory),
        asks=[
            _pending(
                {
                    "ask_id": ask_id,
                    "status": "timed_out",
                    "created_at": now - 3_600_000,
                    "expires_at": now - 60_000,
                }
            )
        ],
    )


def test_a_cold_dismiss_republishes_the_conversation_s_own_asks(monkeypatch) -> None:
    """UX round 1, U1 (MAJOR): the no-boot path pushes a frame, minus the engage.

    The dock and the header chip read ``projection.asks``, and nothing refreshed
    them: the card's ``onSettled`` re-reads ``/api/asks`` for the SHEET, so the
    sheet said "1 timed out" while the dock still read "2 questions timed out"
    at +45 s, until the user left the screen. Main did not have this because its
    engage pushed a frame. The dismissal now recaptures the durable projection
    and publishes it to the session's subscribers — what
    ``_engage_and_publish``'s repaint does, without starting anything.
    """
    from local_operator.mobile.daemon import _durable_projection

    _cold_engage(monkeypatch)
    _durable_lapsed_ask("ask-lapsed")
    before = _durable_projection(SESSION_A)
    assert before is not None, "the fixture must project asks for this cell to mean anything"
    assert [row.ask_id for row in before.asks or ()] == ["ask-lapsed"]

    client, daemon = _client_and_daemon()
    subscriber: "asyncio.Queue[dict[str, object]]" = asyncio.Queue()
    daemon.table.session_subscribers[SESSION_A] = {subscriber}
    reply = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_dismiss", "ask_id": "ask-lapsed"},
    )
    assert reply.status_code == 200, reply.text

    frame = subscriber.get_nowait()
    assert frame["session_id"] == SESSION_A
    # PRESENT and empty: the rows are gone from the frame the screen paints, and
    # the tally rides beside them (the client's capability proxy is either field).
    assert cast("list[dict[str, object]]", frame.get("asks")) == []
    # Still the cold door: nothing was dialled and nothing was started.
    assert daemon.table.entries == {}


@pytest.mark.parametrize(
    "evict_payload", [True, False], ids=["payload-evicted", "payload-retained"]
)
def test_a_cold_dismiss_repaints_with_the_current_fold(monkeypatch, evict_payload) -> None:
    """Agent review round 2, M-3: the repaint carries the ended receipt — and the
    CURRENT fold.

    ``_repaint_stopped_screen`` and the SSE route both fold with the daemon's own
    ended rule ("a stopped screen is ended iff an ended entry for it exists"),
    because the phone's composer reads ``projection.ended``: a conversation this
    daemon has watched die must not come back to life on the phone because a
    dismissal repainted it. The republish spelling omitted the flag, so the frame
    the subscriber received NEXT said ``ended=False`` over a screen whose last
    frame said ``True`` — until some later frame corrected it.

    The two payload states are the fence's own two branches, and a dismissal must
    be honest in both: the payload EVICTED under cache pressure (the ledger turns a
    fresh fold into the retained epoch — the state this capture exists for), and the
    payload RETAINED (``state.terminal and not terminal`` serves that frame back
    verbatim, which for a repaint that follows a write is the FRAME FROM BEFORE THE
    WRITE — the dock and the chip would keep showing the ask the user just
    dismissed, which is the U1 symptom itself).
    """
    from local_operator.mobile.daemon import _projection_frame

    _cold_engage(monkeypatch)
    _durable_lapsed_ask("ask-lapsed")
    client, daemon = _client_and_daemon()
    record = _record(SESSION_A, 9_999_911)
    entry = SessionEntry(record)
    entry.ended = True  # the scan's own verdict on a pid that is gone
    daemon.table.entries[record.pid] = entry
    subscriber: "asyncio.Queue[dict[str, object]]" = asyncio.Queue(maxsize=8)
    daemon.table.session_subscribers[SESSION_A] = {subscriber}

    async def seed() -> None:
        # The frame the watching phone received when the death was proved — the
        # scan's capture, pushed to this same subscriber set (daemon.py's dead-pid
        # branch). That frame is what the dock and the chip are drawn from, and the
        # ask below is the one the phone is about to dismiss.
        dead = await daemon._capture_durable_projection(
            SESSION_A, record=record, terminal=True, ended=True
        )
        assert dead is not None and [row.ask_id for row in dead.asks or ()] == ["ask-lapsed"]
        subscriber.put_nowait(_projection_frame(dead))
        if evict_payload:
            daemon.session_projections.pop(SESSION_A, None)
        assert daemon._projection_generations[SESSION_A].terminal is True

    asyncio.run(seed())
    assert subscriber.get_nowait()["ended"] is True

    reply = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_dismiss", "ask_id": "ask-lapsed"},
    )
    assert reply.status_code == 200, reply.text

    frame = subscriber.get_nowait()
    # The rows the dismissal just settled are gone from the frame that repaints the
    # screen; the receipt the screen already had is kept.
    assert frame["asks"] == []
    assert frame["ended"] is True


def test_a_dismissal_that_landed_is_not_failed_by_a_broken_repaint(monkeypatch, caplog) -> None:
    """Agent review round 2, M-4: the repaint is a courtesy, the dismissal is durable.

    The capture at the end of this route reads the same disk the dismissal has
    already written, and it only decides what is DRAWN: what it produces is what
    the next fold would produce anyway. Catching ``_StaleProjection`` alone meant
    any other raise (a session directory that vanished under the fold, a full
    disk) reached the route's own except ladder and answered 422/502 for a
    dismissal that had ALREADY landed — so the phone painted a failure over a
    durable success and the user tapped again. Answer the dismissal; log the paint.
    """
    _cold_engage(monkeypatch)
    _durable_lapsed_ask("ask-lapsed")
    client, daemon = _client_and_daemon()
    subscriber: "asyncio.Queue[dict[str, object]]" = asyncio.Queue()
    daemon.table.session_subscribers[SESSION_A] = {subscriber}

    async def broken_capture(session_id, **kwargs):  # noqa: ANN001, ANN003, ANN202
        raise OSError("the fold could not read the session directory")

    monkeypatch.setattr(daemon, "_capture_durable_projection", broken_capture)
    with caplog.at_level(logging.DEBUG, logger="local_operator.mobile.daemon"):
        reply = client.post(
            f"/api/sessions/{SESSION_A}/command",
            json={"op": "ask_dismiss", "ask_id": "ask-lapsed"},
        )

    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "detail": "dismissed"}
    # Nothing was painted, and nothing was claimed: the subscriber waits for the
    # next honest frame rather than being handed one built from a failed read.
    assert subscriber.empty()
    # The claim that survives the broken paint is the durable one.
    events = ask_store.read_events(config_dir() / "sessions" / SESSION_A)
    assert events[-1]["kind"] == ask_store.EVENT_DISMISSED
    assert any("repaint failed" in record.getMessage() for record in caplog.records)


def test_a_cold_dismiss_from_the_phone_records_the_remote_surface(monkeypatch) -> None:
    """Agent review round 1, MINOR-3: the phone sends no ``by``, so name the door.

    ``daemon.py`` passes ``by=str(body.get("by") or "")`` and the phone's
    ``ask_dismiss`` body carries no ``by`` at all, which recorded ``by:
    {"surface": ""}`` — an audit line that names nobody, where the LIVE door
    records ``"remote"`` for the same tap (``serving.py``: ``by or "remote"``).
    """
    _cold_engage(monkeypatch)
    _durable_ask_session("ask-lapsed", expired=True)
    reply = _client().post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_dismiss", "ask_id": "ask-lapsed"},
    )
    assert reply.status_code == 200, reply.text
    events = ask_store.read_events(config_dir() / "sessions" / SESSION_A)
    assert events[-1]["kind"] == ask_store.EVENT_DISMISSED
    assert events[-1]["by"] == {"surface": "remote"}


def test_a_cold_dismiss_of_an_open_ask_refuses_in_the_queues_own_words(monkeypatch) -> None:
    """Only ``ask_owner_present`` falls back to the dial (fork A).

    A refusal is the answer, not a reason to boot a runtime: an open ask is not
    dismissible (no surface offers it, and the queue refuses it), so engaging to
    ask a runtime the same question would spend a boot to be told the same thing.
    The sentence is the queue's, verbatim, because a stale tap reads the same
    words on the phone as on the TUI.
    """
    client, engagements = _cold_engage(monkeypatch)
    _durable_ask_session("ask-open-1", expired=False)
    reply = _client().post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_dismiss", "ask_id": "ask-open-1"},
    )

    assert reply.status_code == 422, reply.text
    assert reply.json()["error"] == "only a timed-out ask can be dismissed; it is still open."
    assert engagements == []
    assert client.calls == []


def test_a_refused_cold_answer_keeps_the_session_claimed_active(monkeypatch) -> None:
    """A REFUSAL is not a failed engage, and the list must not say it is.

    The engage and the op are one call on the cold arm, so a raise is usually a
    refusal from a runtime that IS live (an expired ask, a race lost to another
    surface). Discarding the provisional claim there flashes the session out of
    the active section on a refusal — a (brief) lie about a runtime that exists
    — so the ask arm RETAINS it and lets ``retain_provisional_active``'s
    one-scan settle remove the claim if discovery cannot confirm it.

    THE CONTRAST IS THE POINT (agent review round 1, MINOR-1): the cell below
    pins the prompt arm's OPPOSITE behaviour, so the two reasons stay distinct
    in the code rather than one rollback being shared by accident.
    """
    from local_operator.mobile import attach_client

    _durable_session()
    client_double = _FakeAskClient(refusal="this ask expired 7 days ago")

    async def fake_engage(config_dir, session_id, work, **kwargs):  # noqa: ANN001, ANN003, ANN202
        return client_double, "runtime ready"

    monkeypatch.setattr(attach_client, "engage_session_client", fake_engage)
    client, daemon = _client_and_daemon()
    reply = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
    )
    assert reply.status_code == 422, reply.text
    assert SESSION_A in daemon.table.provisional_active


def test_a_failed_prompt_drops_the_provisional_claim(monkeypatch) -> None:
    """The prompt arm's rollback, pinned beside the ask arm's retention.

    A failed prompt means the message never reached a runtime, so the optimistic
    "this session is coming up" claim is FALSE and must be retracted at once —
    the opposite of the ask arm's refusal, where the runtime exists and only the
    op was refused.
    """
    from local_operator.mobile import attach_client

    _durable_session()

    async def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise ConnectionError("daemon restarted")

    monkeypatch.setattr(attach_client, "continue_command", fail)
    client, daemon = _client_and_daemon()
    reply = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={
            "op": "prompt",
            "command_id": "12345678-1234-5678-1234-567812345678",
            "text": "retry me",
        },
    )
    assert reply.status_code == 502, reply.text
    assert SESSION_A not in daemon.table.provisional_active


def test_an_entry_with_no_writer_takes_the_engage_arm(monkeypatch) -> None:
    """``entry is None`` was too narrow a condition for the cold arm.

    An entry can be REGISTERED and still undeliverable — the dial has not
    landed, or its socket is gone — and ``daemon.request`` needs ``entry.writer``
    for exactly that reason. Under the narrower condition such an op fell through
    to the generic "session not connected" while the aggregate had already
    published ``runtime_live: false``, i.e. told the phone to expect an engage.
    One predicate now answers both, so the surface's promise and the route's
    behaviour cannot drift.
    """
    from local_operator.mobile import attach_client

    _durable_session()
    client_double = _FakeAskClient()
    engagements: list[tuple[str, object]] = []

    async def fake_engage(config_dir, session_id, work, **kwargs):  # noqa: ANN001, ANN003, ANN202
        engagements.append((session_id, work))
        return client_double, "runtime ready"

    monkeypatch.setattr(attach_client, "engage_session_client", fake_engage)
    client, daemon = _client_and_daemon()
    daemon.table.entries[4242] = SessionEntry(_record(SESSION_A, 4242))  # no writer

    reply = client.post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
    )
    assert reply.status_code == 200, reply.text
    assert [session for session, _ in engagements] == [SESSION_A]
    assert client_double.calls[-1][0] == "ask_respond"


def test_a_refusal_from_the_engaged_runtime_keeps_the_queue_s_own_sentence(monkeypatch) -> None:
    """The cold arm's refusals are the RUNTIME's, verbatim: an expired ask is
    the ask's own sentence ("expired 7 days ago"), not this route's generic
    "session not connected" — the user's next move depends on which it is."""
    from local_operator.mobile import attach_client

    _durable_session()
    client = _FakeAskClient(
        refusal="this ask expired 7 days ago — ask again if it is still needed."
    )

    async def fake_engage(config_dir, session_id, work, **kwargs):  # noqa: ANN001, ANN003, ANN202
        return client, "runtime ready"

    monkeypatch.setattr(attach_client, "engage_session_client", fake_engage)
    reply = _client().post(
        f"/api/sessions/{SESSION_A}/command",
        json={"op": "ask_respond", "ask_id": "ask-1", "answers": {"q1": ["yes"]}},
    )
    assert reply.status_code == 422, reply.text
    assert "expired 7 days ago" in reply.json()["error"]


def test_the_frame_is_validated_at_the_boundary() -> None:
    """A malformed ask frame is refused BEFORE it reaches a runtime (the op
    family's own validation), which is what keeps a typo from becoming a
    half-applied ask on the far side."""
    client = _client()
    for body in (
        {"op": "ask_respond", "ask_id": "", "answers": {"q1": ["yes"]}},
        {"op": "ask_respond", "ask_id": "ask-1"},
        {"op": "ask_decline"},
        # §10's revision is the same body, so it must be validated the same way.
        {"op": "ask_revise", "ask_id": "", "answers": {"q1": ["yes"]}},
        {"op": "ask_revise", "ask_id": "ask-1"},
        {"op": "ask_revise", "ask_id": "ask-1", "answers": "yes"},
    ):
        reply = client.post(f"/api/sessions/{SESSION_A}/command", json=body)
        assert reply.status_code == 422, body
        assert reply.json()["error"], body


def test_a_settled_row_survives_its_runtime_and_still_names_its_conversation() -> None:
    """The index lives OUTSIDE the session directory (that is its whole point),
    so the phone can still show — and answer — an ask whose runtime is gone."""
    _session_dir(SESSION_A)
    ask_store.write_entry(config_dir(), SESSION_A, cwd="/tmp/gone", asks=[_pending()])
    # No live record for it: the ask outlives the runtime that asked it, which
    # is the whole reason this route reads the index instead of dialling.
    assert not any(record.session_id == SESSION_A for record, _ in registry.scan())
    rows = _client().get("/api/asks").json()["asks"]
    assert [row["ask_id"] for row in rows] == ["ask-open-1"]
    assert rows[0]["cwd"] == "/tmp/gone"


# ---------------------------------------------------------------------------
# Image answers through the relay (ask-attachments-v1): forward or refuse, never strip
# ---------------------------------------------------------------------------


class _KeepingAskHandle(_AskHandle):
    """An owner whose ``ask_respond`` takes ``attachments`` -- it advertises the token."""

    async def ask_respond(  # type: ignore[override]
        self, ask_id, answers, by="", attachments=None
    ) -> str:  # noqa: ANN001
        self.ask_calls.append(
            (
                "ask_respond",
                {
                    "ask_id": ask_id,
                    "answers": answers,
                    "by": by,
                    "kinds": {
                        q: [b.type for b in blocks] for q, blocks in (attachments or {}).items()
                    },
                },
            )
        )
        return "answered"


async def _post_ask_command(handle: _AskHandle, body: dict[str, object]) -> httpx.Response:
    """Dial ``handle`` through a real relay and POST one command frame at it."""
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = [pair for pair in registry.scan() if pair[1] == "live"]
            if found:
                record = found[0][0]
                break
            await asyncio.sleep(0.05)
        assert record is not None
        daemon = MobileDaemon(port=0, password="pw123")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            for _ in range(100):
                if entry.projection is not None:
                    break
                await asyncio.sleep(0.05)
            assert entry.projection is not None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(daemon)),
                base_url="http://fixture",
                cookies={COOKIE_NAME: sign_cookie("pw123")},
            ) as client:
                return await client.post(f"/api/sessions/{record.session_id}/command", json=body)
        finally:
            dial.cancel()
    finally:
        registrant.close()


def _png_b64_small() -> str:
    import base64
    import io

    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), (10, 120, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_the_relay_forwards_an_image_answer_to_a_capable_owner() -> None:
    handle = _KeepingAskHandle()
    reply = await _post_ask_command(
        handle,
        {
            "op": "ask_respond",
            "ask_id": "ask-1",
            "answers": {"q1": ["see this"]},
            "images": [
                {"question_id": "q1", "data_b64": _png_b64_small(), "mime_type": "image/png"}
            ],
        },
    )
    assert reply.status_code == 200, reply.text
    (call,) = handle.ask_calls
    assert call[1]["kinds"] == {"q1": ["image"]}


@pytest.mark.asyncio
async def test_the_relay_refuses_rather_than_strips_for_an_owner_without_the_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.session.runtime import server as runtime_server
    from local_operator.session.runtime.types import ASK_ATTACHMENTS_UNSUPPORTED

    # PIN THE RELAY'S OWN GATE, not the owner's backstop. The owner refuses an
    # image answer with the SAME sentence (``_answer_attachment_kwargs``), and a
    # handle that never saw the call records nothing either way -- so with the
    # relay gate removed this cell still passed (mutant: ``daemon.py`` gate
    # condition -> ``False``). What differs is whether the frame REACHED the
    # owner at all: a real pre-feature owner has no dispatch backstop and would
    # record the text and ignore the pictures. The spy stands in for "the frame
    # arrived": it must never be called.
    reached_owner: list[str] = []
    real = runtime_server._answer_attachment_kwargs  # noqa: SLF001

    async def spy(op, method, wire_images, answers):  # noqa: ANN001
        reached_owner.append(op)
        return await real(op, method, wire_images, answers)

    monkeypatch.setattr(runtime_server, "_answer_attachment_kwargs", spy)

    handle = _AskHandle()  # ``ask_respond(ask_id, answers, by)``: no ``attachments``
    reply = await _post_ask_command(
        handle,
        {
            "op": "ask_respond",
            "ask_id": "ask-1",
            "answers": {"q1": ["see this"]},
            "images": [
                {"question_id": "q1", "data_b64": _png_b64_small(), "mime_type": "image/png"}
            ],
        },
    )
    assert reply.status_code == 422, reply.text
    assert reply.json()["error"] == ASK_ATTACHMENTS_UNSUPPORTED
    assert handle.ask_calls == [], "the text half of the answer must NOT have been recorded"
    assert reached_owner == [], "the relay must refuse BEFORE a frame is written to the owner"


@pytest.mark.asyncio
async def test_a_malformed_images_list_is_a_422_at_the_relay_door() -> None:
    handle = _KeepingAskHandle()
    reply = await _post_ask_command(
        handle,
        {
            "op": "ask_respond",
            "ask_id": "ask-1",
            "answers": {"q1": ["x"]},
            "images": [{"data_b64": "QUJD", "mime_type": "image/png"}],
        },
    )
    assert reply.status_code == 422, reply.text
    assert handle.ask_calls == []


def test_the_aggregate_follows_a_window_written_after_the_daemon_started() -> None:
    """``asks.lapsed_visible_for`` (decision D8) on the phone's sheet: a
    ``config.yml`` written while the daemon is up applies on the NEXT request,
    with no restart, and only to the lapsed row."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    now = int(time.time() * 1000)
    _session_dir(SESSION_A)
    lapsed = {"expires_at": now - 2 * 3_600_000, "created_at": now - 2 * 3_600_000 - 900_000}
    ask_store.write_entry(
        config_dir(),
        SESSION_A,
        cwd="/tmp/aaa",
        asks=[
            _pending({"ask_id": "ask-lapsed", "status": "timed_out", **lapsed}),
            _pending({"ask_id": "ask-answered", "status": "answered", **lapsed}),
        ],
    )
    client = _client()
    before = [row["ask_id"] for row in client.get("/api/asks").json()["asks"]]
    assert sorted(before) == ["ask-answered", "ask-lapsed"]

    settings_io._store(
        ConfigManager(config_dir()), settings_io.BY_KEY["asks.lapsed_visible_for"].path, "1h"
    )
    after = [row["ask_id"] for row in client.get("/api/asks").json()["asks"]]
    assert after == ["ask-answered"]
