"""Desktop read receipts: a cold durable path that never admits or binds.

The desktop control surface reaches sessions through `DesktopSessionBridge`,
which acquires a `AttachedSession` and can START an owner. A read receipt must
not do any of that: the user is looking at a conversation that already ended,
frequently with no owner alive at all, and marking it read is not a reason to
spawn a process. These tests pin that separation plus the ordering rules the
shared receipt clock depends on.
"""

import asyncio
import contextlib
import sqlite3
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attention import AttentionStore

# A 12-character id in exactly the shape a session id has, but uppercase.
#
# Written out as a literal rather than derived from a live id with ``.upper()``,
# and that is a fix rather than a style choice: ids are ``uuid4().hex[:12]``, so
# roughly one draw in ``(16/10)**12`` (~282) contains no letter at all, and
# ``.upper()`` of such an id is that id unchanged -- a perfectly VALID session.
# The arms below then demand a rejection for a real conversation (and a 422 for
# a real batch item), so they redden a shard at a rate rare enough to pass review
# and recur in CI. Uppercase hex cannot be a generated id (``.hex`` only ever
# emits lowercase), so this id is refused for its CASE on every draw, which is
# the arm under test.
UPPERCASE_SESSION_ID = "ABCDEF012345"


def _publish(root, session_id: str, anchor: str, kind: str = "complete") -> str:
    token = str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(f"session/{session_id}", token, anchor, kind)
    return token


@pytest.mark.asyncio
async def test_a_read_receipt_never_acquires_a_session_or_starts_an_owner(tmp_path, monkeypatch):
    """The cold path is the point: no bridge, no attach, no spawn.

    `DesktopSessions.session()` is the only other way in, and it constructs a
    `AttachedSession` that will start an owner for a cold session. Reading is not
    an admission, so this route must not reach it -- an exploding `session()`
    is how that stays true if someone later "simplifies" the implementation.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")

    def forbidden(*args, **kwargs):
        raise AssertionError("a read receipt must not acquire a session bridge")

    monkeypatch.setattr(DesktopSessions, "session", forbidden)
    state = await pool.acknowledge_attention(sid, token)
    assert state["unseen"] is False and state["revision"] == [1, 1]
    assert not (tmp_path / "sessions" / sid / ".session.pid").exists()
    assert not pool.bridges


@pytest.mark.asyncio
async def test_only_a_real_user_session_in_this_root_can_be_acknowledged(tmp_path):
    """Identity is validated the same way the bridge validates it.

    A path from the caller, a traversal, or another root's session id must be
    a 404 rather than a receipt written against a fabricated conversation.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")
    for bogus in ("../../etc", "not-hex", "a" * 12, UPPERCASE_SESSION_ID, ""):
        with pytest.raises(KeyError):
            await pool.acknowledge_attention(bogus, token)
    # A valid-shaped id that is not a session on disk is equally unknown.
    with pytest.raises(KeyError):
        await pool.acknowledge_attention("0123456789ab", token)
    # Unchanged by every rejection above.
    assert AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")["unseen"]
    assert (await pool.acknowledge_attention(sid, token))["unseen"] is False


@pytest.mark.asyncio
async def test_a_delayed_receipt_for_an_older_completion_never_clears_a_newer_one(tmp_path):
    """A slow client acknowledging A must not mark B read.

    The renderer captures the token with the anchor it actually saw, so a receipt
    that arrives after the next turn finished is addressed to the OLD outcome.
    Two wrong answers are available and both are refused here: advancing to "now"
    (which would silently swallow an unread result -- the exact failure the
    mobile bodyless `/seen` had before this contract existed) and answering 200
    for a receipt that cannot make the conversation read, which is what stranded
    the operator's checkmark (both shipped clients latch on a resolved call).
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    first = _publish(tmp_path, sid, "result-1")
    second = _publish(tmp_path, sid, "result-2")

    # The refusal TYPE is read by name rather than imported: the name is what
    # the surfaces key on (`detail.code` / the mobile `code`), and a name check
    # keeps this module importable against a pre-fix tree, so the assertion that
    # fails there is about BEHAVIOUR (no refusal, or a moved receipt) rather
    # than about a symbol having been added.
    with pytest.raises(ValueError) as refused:
        await pool.acknowledge_attention(sid, first)
    assert type(refused.value).__name__ == "SupersededCompletionToken", refused.value
    untouched = AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")
    assert untouched["unseen"] is True and untouched["revision"] == [2, 0]

    caught_up = await pool.acknowledge_attention(sid, second)
    assert caught_up["unseen"] is False and caught_up["revision"] == [2, 2]

    # Reordered duplicate delivery of the old receipt converges, never regresses.
    assert (await pool.acknowledge_attention(sid, first))["unseen"] is False


@pytest.mark.asyncio
async def test_the_route_refuses_a_superseded_token_with_a_machine_code(tmp_path, monkeypatch):
    """The wire answer a client has to be able to act on, over the REAL route.

    A 200 whose body said `unseen: true` was the defect the operator reported:
    both shipped clients treat a resolved `sessions.seen` as "read" and stop
    retrying, so the one completion they were looking at stayed unseen forever.
    The refusal is therefore a 409 carrying the machine code they need to take
    the re-arm path instead of backing off as if the store had failed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-desktop-token")
    app = FastAPI()
    app.include_router(desktop_sessions.router)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    sid = await pool.create(str(tmp_path))
    stale = _publish(tmp_path, sid, "result-1")
    current = _publish(tmp_path, sid, "result-2")
    store = AttentionStore(tmp_path / "attention.db")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer synthetic-desktop-token"},
    ) as client:
        route = f"/v1/desktop/sessions/{sid}/seen"
        refused = await client.post(route, json={"completion_token": stale})
        assert refused.status_code == 409
        assert refused.json()["detail"]["code"] == "superseded_completion_token"
        assert "current token" in refused.json()["detail"]["message"]
        assert store.state(f"session/{sid}")["unseen"] is True

        read = await client.post(route, json={"completion_token": current})
        assert read.status_code == 200
        assert read.json()["result"]["unseen"] is False
        assert store.state(f"session/{sid}")["unseen"] is False

        # And the two refusals stay distinguishable by TYPE on the class a
        # caller catches, not only by the code on the wire.
        unknown = await client.post(route, json={"completion_token": str(uuid.uuid4())})
        assert unknown.status_code == 409
        assert unknown.json()["detail"] == "unknown completion token"


@pytest.mark.asyncio
async def test_an_unknown_or_foreign_token_is_refused_without_touching_state(tmp_path):
    """Tokens are membership-checked against THIS conversation.

    Rejecting a well-formed token that belongs to another session matters more
    than rejecting garbage: the ids are uniform, so a mixed-up client would
    otherwise clear a conversation the user never opened.
    """
    pool = DesktopSessions(tmp_path)
    mine = await pool.create(str(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    theirs = await pool.create(str(other))
    foreign = _publish(tmp_path, theirs, "their-result")
    _publish(tmp_path, mine, "my-result")

    for bad in (foreign, str(uuid.uuid4()), "not-a-uuid"):
        with pytest.raises(ValueError):
            await pool.acknowledge_attention(mine, bad)
    store = AttentionStore(tmp_path / "attention.db")
    assert store.state(f"session/{mine}")["unseen"]
    assert store.state(f"session/{theirs}")["unseen"]


@pytest.mark.asyncio
async def test_the_session_list_reports_durable_receipts_without_an_owner(tmp_path):
    """Unread state is a cold read, one connection for the whole list.

    The list is painted before any conversation is opened, so it cannot depend
    on a live owner or a per-row database connection.
    """
    pool = DesktopSessions(tmp_path)
    read = await pool.create(str(tmp_path))
    unread = await pool.create(str(tmp_path))
    token = _publish(tmp_path, read, "seen-result")
    _publish(tmp_path, unread, "unseen-result")
    await pool.acknowledge_attention(read, token)

    rows = {row["id"]: row["attention"] for row in (await pool.list(50)).rows}
    assert rows[read]["unseen"] is False and rows[read]["revision"] == [1, 1]
    assert rows[unread]["unseen"] is True
    assert rows[unread]["conversation_id"] == f"session/{unread}"
    # A session that never completed a turn is present and simply has nothing.
    quiet = await pool.create(str(tmp_path))
    assert (await pool.list(50)).rows[0]["id"] is not None
    assert {row["id"]: row["attention"] for row in (await pool.list(50)).rows}[quiet][
        "completion_token"
    ] is None


@pytest.mark.asyncio
async def test_concurrent_receipts_and_publications_converge(tmp_path):
    """Independent processes write this store; the API must not serialize it.

    Interleaving a burst of acknowledgements with a new publication has exactly
    two correct outcomes per acknowledgement -- read through the token it named,
    or refused because a newer completion is already current -- and one outcome
    that is never correct: accepted while the conversation reads as unread.
    Neither may lose the newer completion or report a receipt ahead of what was
    acknowledged.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")

    async def publish_later():
        await asyncio.sleep(0.01)
        return await asyncio.to_thread(_publish, tmp_path, sid, "result-2")

    async def acknowledge_once():
        try:
            return await pool.acknowledge_attention(sid, token)
        except ValueError as refused:
            assert type(refused).__name__ == "SupersededCompletionToken", refused
            return None

    receipts, second = await asyncio.gather(
        asyncio.gather(*[acknowledge_once() for _ in range(8)]), publish_later()
    )
    store = AttentionStore(tmp_path / "attention.db")
    final = store.state(f"session/{sid}")
    published, acknowledged = final["revision"]
    assert published == 2 and acknowledged == 1
    assert final["unseen"] is True and final["anchor_id"] == "result-2"
    accepted = [state for state in receipts if state is not None]
    assert accepted, "every acknowledgement was refused; the race did not run"
    for state in accepted:
        # An accepted receipt belongs to the token it named: the watermark reads
        # 1, and the state names `result-1` as the completion it is about. An
        # acknowledgement that ran before the publication was therefore HONEST
        # when it answered -- the newer completion simply landed after it, which
        # is the whole reason the clients must read the returned state rather
        # than the fact that the call resolved.
        assert state["revision"][1] == 1 and state["completion_token"] == token
    # The newer completion's own receipt is the one that closes it.
    assert store.acknowledge(f"session/{sid}", second)["unseen"] is False


def _hold_write_lock(root):
    """Hold the store's write lock from a second connection.

    REAL contention rather than a simulated error, and deterministic *because*
    the lock is held for the whole attempt: the store's own busy timeout expires
    against this holder and raises ``SQLITE_BUSY``, which the shared classifier
    answers with the retryable 503. Nothing races here -- the outcome is fixed
    by the fact that this connection never lets go until the caller does.
    """
    conn = sqlite3.connect(root / "attention.db", timeout=5)
    conn.execute("BEGIN IMMEDIATE")
    return conn


def _break_store(root) -> None:
    """Drop the table a read needs, leaving a file that still opens.

    The UNCLASSIFIED store failure: an error the classifier cannot name as
    contention, a full disk or a missing file, so it takes the deliberate
    default -- 500 ``store_unavailable``, "retrying will not help". Contention
    is exercised for real by :func:`_hold_write_lock` instead.
    """
    import sqlite3 as _sqlite3

    with contextlib.closing(_sqlite3.connect(root / "attention.db")) as conn:
        conn.execute("DROP TABLE completions")
        conn.commit()


@pytest.mark.asyncio
async def test_a_store_error_costs_one_poll_not_the_whole_loop(tmp_path):
    """A transient store failure must not end cross-process read sync.

    The loop was a bare `while True`, so ONE `database is locked` stopped the
    poller for the life of the bridge: the phone and the TUI would clear an
    unread completion while the desktop kept showing it, forever and silently.
    That is a new way to hide an unread result -- the failure this feature
    exists to prevent.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")
    async with pool.session(sid) as bridge:
        for _ in range(200):
            if bridge.attention.get("completion_token"):
                break
            await asyncio.sleep(0.01)
        assert bridge.attention["unseen"] is True

        _break_store(tmp_path)
        # The fixture must actually reach the failing branch, or this test
        # passes while exercising nothing.
        with pytest.raises(sqlite3.Error):
            AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")
        bridge.attention_poll_key = None
        await asyncio.sleep(2.5)
        assert (
            bridge.attention_task is not None and not bridge.attention_task.done()
        ), "the poll loop died on a transient store error"

        # Recovery is the point: once the store is healthy again the loop must
        # pick the change up on its own, with no remount. Deleting the file is
        # how the schema is rebuilt from scratch by the store itself.
        (tmp_path / "attention.db").unlink()
        recovered = _publish(tmp_path, sid, "result-2")
        for _ in range(300):
            if bridge.attention.get("completion_token") == recovered:
                break
            await asyncio.sleep(0.01)
        assert bridge.attention["completion_token"] == recovered
        assert bridge.attention["unseen"] is True
    assert token != recovered


@pytest.mark.asyncio
async def test_a_dead_poll_task_can_never_strand_the_session_owner(tmp_path):
    """Teardown must dispose the owner even when the poller already failed.

    `_detach` awaited the attention task FIRST and suppressed only
    `CancelledError`, so awaiting an already-failed task re-raised: leaving a
    conversation raised, `dispose()` never ran, and the owner plus its
    subscriptions leaked while `users` had already reached 0. A read receipt
    must never strand a session owner.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _publish(tmp_path, sid, "result-1")
    disposed: list[bool] = []
    async with pool.session(sid) as bridge:
        remote = bridge.remote
        assert remote is not None
        original = remote.dispose

        async def record_dispose():
            disposed.append(True)
            await original()

        remote.dispose = record_dispose  # type: ignore[method-assign]

        # Kill the task the way a store error would, and PROVE it is dead
        # before asserting teardown survives it.
        assert bridge.attention_task is not None
        bridge.attention_task.cancel()
        with contextlib.suppress(BaseException):
            await bridge.attention_task

        async def already_failed() -> None:
            raise sqlite3.OperationalError("database is locked")

        bridge.attention_task = asyncio.create_task(already_failed())
        await asyncio.sleep(0)
        assert bridge.attention_task.done() and bridge.attention_task.exception() is not None

    # Leaving the conversation did not raise, and teardown completed.
    assert disposed == [True], "the owner session was never disposed"
    assert bridge.attention_task is None
    assert bridge.remote is None and bridge.users == 0
    assert not bridge.unsubscribers, "frontend subscriptions leaked"


@pytest.mark.asyncio
async def test_a_degraded_store_never_breaks_opening_or_listing(tmp_path):
    """Unread badges are decoration; they must not fail primary navigation.

    Before this field existed neither path touched `attention.db`, so letting a
    busy sidecar 500 the session list and the transcript is a straight
    availability regression on paths that have nothing to do with receipts.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _publish(tmp_path, sid, "result-1")
    async with pool.session(sid) as bridge:
        healthy = await bridge.snapshot()
        assert healthy["payload"]["frontend"]["snapshot"]["attention"]["unseen"] is True
        assert (await pool.list(50)).rows[0]["attention"]["unseen"] is True

        _break_store(tmp_path)
        with pytest.raises(sqlite3.Error):
            AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")

        # The conversation still OPENS, and the list still lists.
        degraded = await bridge.snapshot()
        assert degraded["payload"]["frontend"]["snapshot"]["session_id"] == sid
        rows = (await pool.list(50)).rows
        assert [row["id"] for row in rows] == [sid]
        # Omitted rather than fabricated: absent state is "not ackable" on the
        # client, which is correct. A false "read" would not be.
        assert "attention" not in rows[0]


# ---------------------------------------------------------------------------
# The bulk route: POST /v1/desktop/attention/seen
#
# Same cold contract as the per-session route above, reached by a gesture that
# means "these" rather than "this one". The wire shape is frozen in the design
# (`DESIGN.md` §3.1), so these tests pin the SHAPE as well as the behaviour --
# including the two things a caller depends on and cannot re-derive: which
# bucket an item landed in, and that a `read` entry is the store's own state
# dict rather than a response model that would add a `supported: null`.
# ---------------------------------------------------------------------------

TOKEN_ENV = "LOCAL_OPERATOR_DESKTOP_TOKEN"


@contextlib.asynccontextmanager
async def _bulk_client(tmp_path, monkeypatch):
    """A real ASGI client over the real router, at an isolated config root."""
    monkeypatch.setenv(TOKEN_ENV, "bulk-token")
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer bulk-token"},
    ) as client:
        yield client, pool


async def _session_with_completion(pool, tmp_path, anchor: str) -> tuple[str, str]:
    session_id = await pool.create(str(tmp_path))
    return session_id, _publish(tmp_path, session_id, anchor)


@pytest.mark.asyncio
async def test_the_bulk_route_reports_a_verdict_per_item(tmp_path, monkeypatch):
    """One call, three buckets, and every item answered -- including the misses.

    A batch that clears nothing is still 200: the verdicts ARE the answer, and a
    non-2xx would make the renderer discard the partial result it did get.
    """
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")
        superseded_id, superseded_token = await _session_with_completion(pool, tmp_path, "result-2")
        _publish(tmp_path, superseded_id, "result-2b")
        absent = "0123456789ab"

        response = await client.post(
            "/v1/desktop/attention/seen",
            json={
                "items": [
                    {"session_id": sid, "completion_token": token},
                    {"session_id": superseded_id, "completion_token": superseded_token},
                    {"session_id": absent, "completion_token": str(uuid.uuid4())},
                ]
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["message"] == "Completion receipts marked read."
        assert body["status"] == 200
        assert body["result"]["superseded"] == [superseded_id]
        assert body["result"]["unknown"] == [absent]
        assert len(body["result"]["read"]) == 1
        state = body["result"]["read"][0]
        assert state["conversation_id"] == f"session/{sid}"
        assert state["completion_token"] == token and state["unseen"] is False
        # R8: the store's own dict, never `AttentionState`. That model defaults
        # `supported` to None, and the renderer treats `null` as "you know this
        # one" rather than "inherit" -- which would switch off its visible-read
        # receipt for the conversation this batch just cleared.
        assert "supported" not in state, state
        # And the store really moved, for the one item that was ackable.
        store = AttentionStore(tmp_path / "attention.db")
        assert store.state(f"session/{sid}")["unseen"] is False
        assert store.state(f"session/{superseded_id}")["unseen"] is True


@pytest.mark.asyncio
async def test_a_batch_that_clears_nothing_is_still_a_200(tmp_path, monkeypatch):
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, _token = await _session_with_completion(pool, tmp_path, "result-1")
        response = await client.post(
            "/v1/desktop/attention/seen",
            json={"items": [{"session_id": sid, "completion_token": str(uuid.uuid4())}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == {"read": [], "superseded": [], "unknown": [sid]}


@pytest.mark.asyncio
async def test_the_bulk_route_refuses_a_malformed_batch_with_422(tmp_path, monkeypatch):
    """The admission rules, one per arm, all of them before any store call."""
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")
        item = {"session_id": sid, "completion_token": token}
        cases = {
            "empty": {"items": []},
            "missing": {},
            "over the cap": {"items": [item] * 501},
            "short id": {"items": [dict(item, session_id="abc")]},
            "uppercase id": {"items": [dict(item, session_id=UPPERCASE_SESSION_ID)]},
            "non-uuid token": {"items": [dict(item, completion_token="now")]},
            # `extra="forbid"` like every other Input in this module: a
            # conversation identity is DERIVED, so a caller cannot name one.
            "caller-named identity": {"items": [dict(item, conversation_id="session/x")]},
            "unknown top-level field": {"items": [item], "all": True},
        }
        for name, payload in cases.items():
            response = await client.post("/v1/desktop/attention/seen", json=payload)
            assert response.status_code == 422, (name, response.status_code, response.text)
        assert AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")["unseen"] is True


@pytest.mark.asyncio
async def test_the_bulk_route_is_cold_and_never_acquires_a_bridge(tmp_path, monkeypatch):
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")

        def forbidden(*args, **kwargs):
            raise AssertionError("a bulk read receipt must not acquire a session bridge")

        monkeypatch.setattr(DesktopSessions, "session", forbidden)
        response = await client.post(
            "/v1/desktop/attention/seen",
            json={"items": [{"session_id": sid, "completion_token": token}]},
        )
        assert response.status_code == 200, response.text
        assert not (tmp_path / "sessions" / sid / ".session.pid").exists()
        assert not pool.bridges


@pytest.mark.asyncio
async def test_the_bulk_route_requires_the_desktop_credential(tmp_path, monkeypatch):
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")
        payload = {"items": [{"session_id": sid, "completion_token": token}]}
        # An EMPTY Authorization rather than no header at all: this client sets a
        # default bearer, and httpx merges per-request headers over it, so an
        # absent key would keep the fixture's own token and assert nothing.
        for headers in ({"Authorization": ""}, {"Authorization": "Bearer wrong-token"}):
            response = await client.post(
                "/v1/desktop/attention/seen", json=payload, headers=headers
            )
            assert response.status_code == 401, headers
        # The refusal happens before the handler: nothing was cleared.
        assert AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")["unseen"] is True


@pytest.mark.asyncio
async def test_the_bulk_capability_is_advertised_with_its_own_key(tmp_path, monkeypatch):
    """`completion_ack_bulk`, beside `completion_ack` rather than on top of it.

    The renderer gates the control on this key, so a key that never reaches the
    wire is a feature no client can discover. It is deliberately NOT a bump of
    `completion_ack`: the per-session ack must keep working against a backend
    that lacks the batch route, which is why the two versions answer different
    questions and why nothing else may be gated on either.
    """
    async with _bulk_client(tmp_path, monkeypatch) as (client, _pool):
        features = (await client.get("/v1/capabilities")).json()["result"]["features"]
        assert features["completion_ack_bulk"] == 1
        assert features["completion_ack"] == 1, "the per-session ack keeps its version"


@pytest.mark.asyncio
async def test_store_contention_costs_the_whole_batch_and_writes_nothing(tmp_path, monkeypatch):
    """R4 end to end: the 503 promises nothing moved, and one transaction keeps it.

    The retryable arm. Contention is created by a second connection holding the
    write lock for the whole attempt, so the store's own busy timeout raises
    ``SQLITE_BUSY`` through the identical call path a busy daemon would -- not a
    patched exception that only proves the test's own stub.
    """
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")
        other, other_token = await _session_with_completion(pool, tmp_path, "result-2")
        holder = _hold_write_lock(tmp_path)
        try:
            response = await client.post(
                "/v1/desktop/attention/seen",
                json={
                    "items": [
                        {"session_id": sid, "completion_token": token},
                        {"session_id": other, "completion_token": other_token},
                    ]
                },
            )
        finally:
            holder.rollback()
            holder.close()
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "store_busy", response.text
        assert "busy" in detail["message"]
        # The remedy rides the sentence because a client paints this text and
        # reads only the code: "it will catch up on its own" would be unearned for
        # a write the caller initiated (QA round 2, Q1).
        assert "Try again in a moment" in detail["message"], detail["message"]
        assert "catch up on its own" not in detail["message"], detail["message"]
        # …and this route is not a message send, so the send path's nouns are
        # absent from every arm of its refusal (QA round 2, Q1).
        assert "message" not in detail["message"], detail["message"]
        assert "send it again" not in detail["message"], detail["message"]
        with contextlib.closing(sqlite3.connect(tmp_path / "attention.db")) as conn:
            receipts = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        assert receipts == 0, "a refused batch left a receipt behind"


@pytest.mark.asyncio
async def test_an_unreadable_store_costs_the_whole_batch_and_writes_nothing(tmp_path, monkeypatch):
    """R4's other arm: the store failure the classifier cannot name as transient.

    ``session/store_failures`` splits the three sqlite conditions rather
    than answering all of them with the contention sentence; this route must
    inherit that split instead of flattening it, so an unreadable store answers
    the 500 that says retrying will not help. The batch-wide guarantee is the
    same in both arms: nothing was written.
    """
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")
        other, other_token = await _session_with_completion(pool, tmp_path, "result-2")
        _break_store(tmp_path)

        response = await client.post(
            "/v1/desktop/attention/seen",
            json={
                "items": [
                    {"session_id": sid, "completion_token": token},
                    {"session_id": other, "completion_token": other_token},
                ]
            },
        )
        assert response.status_code == 500, response.text
        assert response.json()["detail"]["code"] == "store_unavailable", response.text
        with contextlib.closing(sqlite3.connect(tmp_path / "attention.db")) as conn:
            receipts = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        assert receipts == 0, "a refused batch left a receipt behind"


@pytest.mark.asyncio
async def test_a_full_volume_answers_with_the_receipts_own_sentence(tmp_path, monkeypatch):
    """QA round 2, Q1: the receipts route is not a message send, and says so.

    The classifier's out-of-space sentence belongs to the send path — "the message
    could not be written ... and send it again" — and a bulk read receipt has no
    message in it and sends nothing. Driven through the real handler, because the
    sentence is chosen at the route's arm, and the assertion that the send path's
    phrasings are ABSENT is what makes it evidence rather than a restatement of
    the code.
    """
    async with _bulk_client(tmp_path, monkeypatch) as (client, pool):
        sid, token = await _session_with_completion(pool, tmp_path, "result-1")

        def full(self, items):  # noqa: ANN001 — mirrors the bound method's shape
            error = sqlite3.OperationalError("database or disk is full")
            error.sqlite_errorname = "SQLITE_FULL"
            raise error

        monkeypatch.setattr(AttentionStore, "acknowledge_many", full)
        response = await client.post(
            "/v1/desktop/attention/seen",
            json={"items": [{"session_id": sid, "completion_token": token}]},
        )

    assert response.status_code == 507, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "store_out_of_space", response.text
    assert "out of disk space" in detail["message"], detail["message"]
    assert "nothing was written" in detail["message"], detail["message"]
    assert "then try again" in detail["message"], detail["message"]
    assert "message" not in detail["message"], detail["message"]
    assert "send it again" not in detail["message"], detail["message"]
    with contextlib.closing(sqlite3.connect(tmp_path / "attention.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 0


def test_every_receipts_refusal_arm_is_composed_for_this_route():
    """The three arms, pinned as copy: what a client paints, per condition.

    The codes and the statuses are the classifier's (and are asserted elsewhere);
    this pins the SENTENCES, which are this route's, because that split is the
    whole finding.
    """
    from local_operator.server.utils.store_failures import (
        STORE_BUSY,
        STORE_OUT_OF_SPACE,
        STORE_UNAVAILABLE,
        StoreFailure,
    )

    arms = (
        # code, what the sentence must make true for a receipt clear
        (STORE_BUSY, "nothing was written", "Try again in a moment."),
        (STORE_OUT_OF_SPACE, "nothing was written", "Free some space on the volume holding"),
        (STORE_UNAVAILABLE, "could not be written", "Retrying will not help;"),
    )
    for code, claim, remedy in arms:
        failure = StoreFailure(500, code, "SENTINEL-SEND-PATH-PROSE", 40, False)
        text = desktop_sessions.receipts_refusal(failure, None)
        # The classifier's own sentence never reaches this route's client...
        assert "SENTINEL-SEND-PATH-PROSE" not in text, text
        assert "message" not in text and "send it again" not in text, text
        # ...and the arm is still true about the operation and actionable.
        assert claim in text, (code, text)
        assert remedy in text, (code, text)


@pytest.mark.asyncio
async def test_the_single_receipt_route_answers_in_its_own_nouns(tmp_path, monkeypatch):
    """M1: BOTH receipt routes compose their copy, pinned at each handler.

    ``POST /v1/desktop/sessions/{session_id}/seen`` is the shipped
    ``sessions.seen`` contract, and it was left on the classifier's send-path
    sentences (a full volume: "the message could not be written ... and send it
    again"; contention: "it will catch up on its own") after the bulk route was
    fixed -- the same defect class graded MAJOR twice in this PR. Dropping the
    composer at either route has to fail a test, so this one drives both arms on
    THIS route rather than asserting the composer's own output.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-desktop-token")
    app = FastAPI()
    app.include_router(desktop_sessions.router)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")
    store = AttentionStore(tmp_path / "attention.db")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer synthetic-desktop-token"},
    ) as client:
        route = f"/v1/desktop/sessions/{sid}/seen"

        # A real full volume, raised from the store call the route makes.
        def full(self, conversation, observed):  # noqa: ANN001 — mirrors the method
            error = sqlite3.OperationalError("database or disk is full")
            error.sqlite_errorname = "SQLITE_FULL"
            raise error

        with monkeypatch.context() as patch:
            patch.setattr(AttentionStore, "acknowledge", full)
            response = await client.post(route, json={"completion_token": token})
        assert response.status_code == 507, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "store_out_of_space", response.text
        assert "Free some space on the volume holding" in detail["message"], detail["message"]
        assert "message" not in detail["message"], detail["message"]
        assert "send it again" not in detail["message"], detail["message"]
        assert store.state(f"session/{sid}")["unseen"] is True

        # Real contention: a second writer holding the store's write lock.
        holder = sqlite3.connect(tmp_path / "attention.db")
        holder.execute("BEGIN IMMEDIATE")
        try:
            busy = await client.post(route, json={"completion_token": token})
        finally:
            holder.rollback()
            holder.close()
        assert busy.status_code == 503, busy.text
        busy_detail = busy.json()["detail"]
        assert busy_detail["code"] == "store_busy", busy.text
        assert "Try again in a moment" in busy_detail["message"], busy_detail["message"]
        assert "catch up on its own" not in busy_detail["message"], busy_detail["message"]
        assert "message" not in busy_detail["message"], busy_detail["message"]
        assert store.state(f"session/{sid}")["unseen"] is True

        # …and the healthy path still clears, so the composer did not cost the
        # route its job.
        read = await client.post(route, json={"completion_token": token})
        assert read.status_code == 200, read.text
        assert read.json()["result"]["unseen"] is False
