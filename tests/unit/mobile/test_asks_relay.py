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
* ``asks_open`` is ABSENT — not ``0`` — when the runtime does not publish asks,
  because the phone's capability proxy is the field's PRESENCE (a ``0`` would
  say "supported, nothing waiting" about a runtime that cannot say anything).

Nothing here touches the operator's machine: the config root is the isolated
one ``conftest`` installs, the records are this test's own pids, and the one
real runtime is an in-process ``RuntimeServer`` on a loopback socket.
"""

from __future__ import annotations

import asyncio
import time

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
    """PRESENCE is the capability proxy (design §4): a runtime that does not
    publish asks must be indistinguishable from an old one, so the key is
    OMITTED rather than sent as 0."""
    _session_dir(SESSION_A)
    _session_dir(SESSION_B)
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

    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    rows = {row["session_id"]: row for row in client.get("/api/sessions").json()["sessions"]}
    assert rows[SESSION_A]["asks_open"] == 2
    assert "asks_open" not in rows[SESSION_B]


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
        ("ask_dismiss", {"ask_id": "ask-1"}, "dismissed"),
    ],
)
def test_an_answer_to_a_stopped_session_engages_it(monkeypatch, op, body, answered_with) -> None:
    """THE ENGAGE (design §2.4), for all four settle ops.

    The operator's bug was exactly this: a queued ask is durable so the agent
    can move on, and the relay REFUSED the answer once the runtime had exited —
    "session not connected" — so the answer was accepted by the phone and read
    by nobody. Decline and dismiss settle the ask just as an answer does, so
    they take the SAME arm: one engage path, four ops.
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
