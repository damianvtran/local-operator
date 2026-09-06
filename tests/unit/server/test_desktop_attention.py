"""Desktop read receipts: a cold durable path that never admits or binds.

The desktop control surface reaches sessions through `DesktopSessionBridge`,
which acquires a `RemoteSession` and can START an owner. A read receipt must
not do any of that: the user is looking at a conversation that already ended,
frequently with no owner alive at all, and marking it read is not a reason to
spawn a process. These tests pin that separation plus the ordering rules the
shared receipt clock depends on.
"""

import asyncio
import uuid

import pytest

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
    `RemoteSession` that will start an owner for a cold session. Reading is not
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

    The renderer captures the token with the anchor it actually saw, so a
    receipt that arrives after the next turn finished is addressed to the OLD
    outcome. Advancing to "now" (or to the latest sequence) would silently
    swallow an unread result -- the exact failure the mobile bodyless `/seen`
    had before this contract existed.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    first = _publish(tmp_path, sid, "result-1")
    second = _publish(tmp_path, sid, "result-2")

    late = await pool.acknowledge_attention(sid, first)
    assert late["unseen"] is True
    assert late["completion_token"] == second and late["revision"] == [2, 1]

    caught_up = await pool.acknowledge_attention(sid, second)
    assert caught_up["unseen"] is False and caught_up["revision"] == [2, 2]

    # Reordered duplicate delivery of the old receipt converges, never regresses.
    assert (await pool.acknowledge_attention(sid, first))["unseen"] is False


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
    two correct outcomes -- read through the latest, or unread BECAUSE the new
    completion landed after the last receipt. Neither may lose the newer
    completion or report a receipt ahead of what was acknowledged.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")

    async def publish_later():
        await asyncio.sleep(0.01)
        return await asyncio.to_thread(_publish, tmp_path, sid, "result-2")

    receipts, _ = await asyncio.gather(
        asyncio.gather(*[pool.acknowledge_attention(sid, token) for _ in range(8)]),
        publish_later(),
    )
    final = AttentionStore(tmp_path / "attention.db").state(f"session/{sid}")
    published, acknowledged = final["revision"]
    assert published == 2 and acknowledged == 1
    assert final["unseen"] is True and final["anchor_id"] == "result-2"
    for state in receipts:
        assert state["revision"][1] == 1
