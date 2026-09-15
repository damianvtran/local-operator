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

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attention import AttentionStore


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
    for bogus in ("../../etc", "not-hex", "a" * 12, sid.upper(), ""):
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

    rows = {row["id"]: row["attention"] for row in await pool.list(50)}
    assert rows[read]["unseen"] is False and rows[read]["revision"] == [1, 1]
    assert rows[unread]["unseen"] is True
    assert rows[unread]["conversation_id"] == f"session/{unread}"
    # A session that never completed a turn is present and simply has nothing.
    quiet = await pool.create(str(tmp_path))
    assert (await pool.list(50))[0]["id"] is not None
    assert {row["id"]: row["attention"] for row in await pool.list(50)}[quiet][
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


def _break_store(root) -> None:
    """Drop the table a read needs, leaving a file that still opens.

    Real contention (`database is locked` past the 2 s timeout) cannot be
    scheduled deterministically in a test; a missing table raises the same
    `sqlite3.Error` family through the identical call path, which is what the
    guards are written against.
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
        assert (await pool.list(50))[0]["attention"]["unseen"] is True

        _break_store(tmp_path)
        with pytest.raises(sqlite3.Error):
            AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")

        # The conversation still OPENS, and the list still lists.
        degraded = await bridge.snapshot()
        assert degraded["payload"]["frontend"]["snapshot"]["session_id"] == sid
        rows = await pool.list(50)
        assert [row["id"] for row in rows] == [sid]
        # Omitted rather than fabricated: absent state is "not ackable" on the
        # client, which is correct. A false "read" would not be.
        assert "attention" not in rows[0]
