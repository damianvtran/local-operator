"""The desktop stream's reconnect dwell (design D2 of ``docs/design/desktop-stream-gap-storm.md``).

WHY THIS FILE EXISTS. The desktop stream is the bridge's ONLY reference
(``DesktopSessions.session`` acquires before the headers and releases in its
``finally``), so before D2 a transport break was a detach: the facade disposed,
the epoch rotated, and the client's ~500 ms reconnect paid a full re-engage and a
mandatory ``gap``. Every reopen repainted the whole conversation, which is the
oscillation the operator reported. D2 keeps the subscription -- and with it the
facade, the watch lease and the runtime -- for ``RECONNECT_DWELL_S``.

WHAT IS ASSERTABLE WITHOUT WAITING 20 SECONDS. The dwell is deadline-driven on
the MODULE clock rather than one ``asyncio.sleep``, precisely so these tests can
move the deadline instead of living through it: the same ``SimpleNamespace``
clock substitution ``test_desktop_sessions.py`` already uses. Every case below
is measured in milliseconds.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import BRIDGE_COUNT, DesktopSessions

pytestmark = pytest.mark.asyncio


async def _drain(stream: Any) -> list[dict[str, Any]]:
    """The handshake of ``stream``: its ``open`` and its ``snapshot``."""
    return [await anext(stream), await anext(stream)]


async def _arm_dwell(pool: DesktopSessions, sid: str) -> Any:
    """Open a stream, close it, and hand back the bridge holding the dwell.

    ``aclose()`` is the ASGI-disconnect shape: the generator's ``finally`` runs
    with the subscription still registered and nothing against it, which is
    exactly the state D2 arms on.
    """
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        await _drain(stream)
        await stream.aclose()
        assert sub.dwelling, "closing a stream with nothing against it arms the dwell"
        return bridge


async def test_a_reconnect_inside_the_dwell_keeps_the_epoch_and_replays_the_outage(
    tmp_path,
):
    """The property the whole change exists for: a reopen is PATCHED, not repainted."""
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        opened, _ = await _drain(stream)
        epoch, cursor = opened["epoch"], opened["seq"]
        await stream.aclose()

        remote = bridge.remote
        assert remote is not None, "the dwell holds the facade, not just the subscription"

        # The outage: frames published while the viewer's transport is gone. The
        # dwelling subscription is skipped by `publish` (it has no reader), and
        # every frame still lands in the replay the returning client asks for.
        published = []
        for n in range(3):
            bridge.publish("event", {"value": n})
            published.append(bridge.sequence)
        assert sub.queue.empty(), "a dwelling subscription is not fed frames it cannot read"

    # The viewer comes back, on a warm bridge, with the cursor it learned.
    async with pool.session(sid) as reopened:
        assert reopened is bridge, "the re-acquire is warm: the facade was never disposed"
        assert reopened.epoch == epoch, "and the epoch did not rotate"
        again = reopened.subscribe()
        resumed = reopened.events(again, epoch=epoch, after_seq=cursor)
        # The handshake's order is open -> REPLAY -> snapshot, so the replay is
        # read between the two rather than after both.
        open_frame = await anext(resumed)
        assert open_frame["payload"]["gap"] is False, (
            "a reconnect inside the dwell must NOT be a gap: the replay covers the "
            "outage, which is what makes the statement honourable"
        )
        replayed = [await anext(resumed) for _ in published]
        assert [frame["seq"] for frame in replayed] == published, (
            "and every frame published during the outage is replayed, in order, "
            "before the snapshot"
        )
        assert (await anext(resumed))["type"] == "snapshot"
        await resumed.aclose()


async def test_the_dwell_expiry_detaches_and_rotates_the_epoch(tmp_path, monkeypatch):
    """The other half of the contract: the window is BOUNDED, and what it held goes."""
    now = 1000.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(module, "DWELL_TICK_S", 0.01)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    bridge = await _arm_dwell(pool, sid)
    epoch = bridge.epoch
    sub = next(iter(bridge.subscribers.values()))
    assert sub.dwell_until == pytest.approx(now + module.RECONNECT_DWELL_S)
    task = bridge._dwell_tasks.get(sub.id)
    assert task is not None, "one task per dwelling subscription"

    # Before the deadline: nothing has happened.
    await asyncio.sleep(0.05)
    assert not task.done(), "the dwell does not expire early"
    assert bridge.remote is not None and sub.id in bridge.subscribers

    # Past it.
    now += module.RECONNECT_DWELL_S + 1
    await asyncio.wait_for(task, timeout=2)
    assert sub.id not in bridge.subscribers, "the subscription is dropped"
    assert not bridge._dwell_tasks, "and its task is retired"
    assert bridge.remote is None, "the facade is disposed"

    # A later open is therefore COLD: the epoch rotates and the client is told
    # its cursor cannot be replayed. That is the honest full repaint, and it is
    # what the dwell's boundedness buys.
    async with pool.session(sid) as reopened:
        assert reopened is bridge
        assert reopened.epoch != epoch, "the expired dwell rotated the epoch"
        stream = reopened.events(reopened.subscribe(), epoch=epoch, after_seq=0)
        opened, _ = await _drain(stream)
        assert opened["payload"]["gap"] is True, "and the stale cursor is a gap"
        await stream.aclose()


async def test_zero_disables_the_dwell(tmp_path, monkeypatch):
    """`RECONNECT_DWELL_S = 0` is CONTRACT, not a test hook.

    At zero nothing is created, `dwelling` is never set, and the last release
    detaches in the same call -- byte for byte the pre-D2 behaviour. That is what
    every existing "the last release detaches" assertion is written against.
    """
    monkeypatch.setattr(module, "RECONNECT_DWELL_S", 0)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        await _drain(stream)
        await stream.aclose()
        assert not sub.dwelling, "zero never sets the flag"
        assert not bridge._dwell_tasks, "and never creates a task"
        assert sub.id not in bridge.subscribers, "the subscription is popped instead"
        assert not bridge.dwelling, "so the bridge is not dwelling either"

    assert bridge.remote is None, "and the last release detached, in the same call"


async def test_a_dwelling_bridge_is_not_evictable_and_close_ends_the_dwell(tmp_path):
    """A dwelling bridge is protected from pool pressure, and teardown ends it."""
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    bridge = await _arm_dwell(pool, sid)
    sub = next(iter(bridge.subscribers.values()))

    assert bridge.dwelling is True
    assert pool._evictable(bridge) is False, (
        "otherwise BRIDGE_COUNT pressure could delete a bridge that still holds a "
        "facade, a presence assertion and a runtime nothing can reach"
    )

    await bridge.close()
    assert not bridge.subscribers, "close pops the dwelling subscription"
    assert not bridge._dwell_tasks, "and cancels its timer"
    assert bridge.remote is None, "and disposes the facade"
    assert sub.id not in bridge.subscribers


async def test_the_dwell_holds_residency_without_claiming_attention(tmp_path, monkeypatch):
    """What the dwell asserts to the owner: still here, NOT still watching.

    Term 3 of the runtime's residency rule wants a fresh lease and
    `desktop_visible or desktop_can_notify`. Claiming `visible` for a viewer that
    has gone would tell the notification ladder a person is reading a session
    they walked away from -- and suppress the rung that matters most here, "a turn
    finished while my window was reconnecting".
    """
    now = 500.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    async with pool.session(sid) as bridge:
        writes: list[dict[str, Any]] = []

        async def record(**kwargs: Any) -> None:
            writes.append(kwargs)

        # The writer is REPLACED ON THE REAL REMOTE rather than the remote being
        # replaced by a stand-in: the handshake below reads
        # `remote.frontend_state.history_cursor` (D1's page gate), and a bare
        # SimpleNamespace has no store for it.
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "update_desktop_watch", record)
        sub = bridge.subscribe()
        sub.visible = sub.can_notify = True
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        await _drain(stream)
        await stream.aclose()
        assert sub.dwelling

        writes.clear()
        await bridge.refresh_watch()
        assert writes, "the dwell still asserts presence, or the runtime exits under it"
        assert writes[-1] == {
            "visible": False,
            "can_notify": True,
        }, "residency through the notify half, and no claim that anyone is watching"


async def test_an_outage_longer_than_the_replay_window_still_gaps(tmp_path, monkeypatch):
    """The honest fallback is part of the contract, not a failure of the dwell."""
    monkeypatch.setattr(module, "REPLAY_COUNT", 2)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        opened, _ = await _drain(stream)
        epoch, cursor = opened["epoch"], opened["seq"]
        await stream.aclose()
        for n in range(3):
            bridge.publish("event", {"value": n})

    async with pool.session(sid) as reopened:
        assert reopened is bridge and reopened.epoch == epoch
        again = reopened.subscribe()
        resumed = reopened.events(again, epoch=epoch, after_seq=cursor)
        open_frame, _ = await _drain(resumed)
        assert open_frame["payload"]["gap"] is True, (
            "a cursor older than the oldest retained frame cannot be patched, and "
            "the client is told so rather than handed a silently short replay"
        )
        await resumed.aclose()


async def test_a_dwelling_subscription_does_not_hold_the_daemon(tmp_path):
    """`_live_leases` excludes it by default; `refresh_watch` includes it.

    A viewer that has ALREADY lost its transport may not pin a build update for
    `RECONNECT_DWELL_S`: the successor daemon serves the reconnect, and the
    announcement is what the client is already reacting to. The end-to-end form
    of this is `tests/unit/server/test_serve_retire.py`'s drain, which must keep
    passing unchanged.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    bridge = await _arm_dwell(pool, sid)
    assert bridge.dwelling

    assert bridge.users == 0, "the viewer's release already landed"
    assert pool.in_flight_reason() is None, "a dwell must not be a reason this daemon cannot leave"

    assert bridge._live_leases() == [], "excluded by default"
    held = bridge._live_leases(include_dwelling=True)
    assert held and all(s.dwelling for s in held), (
        "and included for the one caller that asks, which is what keeps the "
        "runtime resident across the window"
    )


async def test_the_dwell_is_inside_the_watch_lease(tmp_path):
    """An inequality, pinned: one presence assertion carries the whole window."""
    assert 0 < module.RECONNECT_DWELL_S < module.WATCH_TTL, (
        "a dwell longer than the lease would let presence lapse mid-window, so a "
        "future raise of the constant has to come here and say what it did about "
        "the lease"
    )


# --------------------------------------------------------------- §12 (item 3) --
#
# EVICTION CLOSES WHAT IT DROPS, AND A DWELLING BRIDGE IS THE LAST RESORT.
#
# NOT THE FIX FOR THE REPORTED DEFECT, and these cases must not be read as one:
# measured pool occupancy is 25 of 64 at its worst and every daemon restart resets
# the pool. What they protect is that D2 does not INTRODUCE a refusal -- a
# dwelling bridge is unevictable for the length of its window, so without the
# second tier a pool at BRIDGE_COUNT could reach `ValueError("Too many active
# desktop sessions")` in a state that used to be impossible -- and that the
# eviction path stops leaking the bridge it drops.


async def test_eviction_closes_what_it_drops(tmp_path):
    """The plain `del` never closed the victim; this is what that cost.

    A dropped bridge that is not closed keeps its facade, its runtime, its
    subscribers and its presence, and a later open of that session builds a
    SECOND bridge with its own epoch while the first is still attached.
    """
    pool = DesktopSessions(tmp_path)
    first = await pool.create(str(tmp_path))
    # A bridge exists once a session is ACQUIRED; `create` writes the directory.
    async with pool.session(first):
        pass
    for _ in range(BRIDGE_COUNT - 1):
        other = await pool.create(str(tmp_path))
        async with pool.session(other):
            pass
    assert len(pool.bridges) == BRIDGE_COUNT, "the pool is at its cap"

    evicted = pool.bridges[first]
    assert evicted.users == 0, "and the first bridge is idle"

    victim = pool._evict_one()
    assert victim is not None, "with an idle bridge present there is always a victim"
    assert victim.session_id not in pool.bridges, "and it is out of the pool"

    await victim.close()
    assert victim.remote is None, "closed, so a later open cannot find a second bridge"
    assert not victim.subscribers and not victim._dwell_tasks


async def test_the_dwelling_bridge_is_the_last_resort(tmp_path, monkeypatch):
    """An idle bridge goes first; a dwelling one only when nothing else is left.

    Taking the dwelling one ENDS its dwell, so its viewer gets one honest gap on
    reconnect rather than being refused the open outright -- which is the trade
    this item makes.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    dwelling = await _arm_dwell(pool, sid)
    assert dwelling.dwelling

    # An ordinary idle bridge exists, so it is the victim and the dwell survives.
    other_sid = await pool.create(str(tmp_path))
    async with pool.session(other_sid) as other:
        pass
    assert other.users == 0 and not other.dwelling
    chosen = pool._evict_one()
    assert (
        chosen is not None and chosen.session_id == other_sid
    ), "an idle bridge has nothing to lose and goes first"
    assert chosen is not dwelling
    assert dwelling.dwelling, "and the dwell is untouched by taking it"

    # Nothing but the dwelling bridge is left, so it is taken -- and its dwell
    # must end with it rather than leaking a facade nothing can reach.
    await chosen.close()
    last = pool._evict_one()
    assert last is not None, "with only dwelling bridges left, there is still a victim"
    assert last is dwelling, "and it is the last resort"
    await last.close()
    assert not last.dwelling, "and taking it ends the dwell"
    assert not last._dwell_tasks and last.remote is None


async def test_eviction_never_awaits_inside_the_pool_lock(tmp_path):
    """The regression guard for the ORDERING, which is the one way to be wrong.

    `close()` awaits the bridge's own lock and the owner connection's teardown;
    holding the pool lock across either is the multi-second open this pool's
    `forget` docstring exists to prevent. So the close is asserted to happen with
    the pool lock RELEASED.
    """
    pool = DesktopSessions(tmp_path)
    seen: list[bool] = []

    sid = await pool.create(str(tmp_path))
    async with pool.session(sid):
        pass
    for _ in range(BRIDGE_COUNT - 1):
        other = await pool.create(str(tmp_path))
        async with pool.session(other):
            pass
    victim = pool._evict_one()
    assert victim is not None, "the pool is at its cap and every bridge is idle"

    original = victim.close

    async def spy() -> None:
        seen.append(pool.lock.locked())
        await original()

    victim.close = spy  # type: ignore[method-assign]
    # The production ordering: evict under the lock, close after it.
    async with pool.lock:
        assert victim.session_id not in pool.bridges
    await victim.close()
    assert seen == [False], (
        "the close must run with the pool lock RELEASED; `True` here is the "
        "multi-second open the ordering exists to prevent"
    )
    assert sid  # the created session is untouched by this case


async def test_a_read_release_still_detaches_on_the_spot(tmp_path):
    """The dwell is scoped to STREAMS, and this is the other side of that scope.

    A read route's `release()` detaches immediately. Applying the dwell to every
    `session()` read would silently extend residency across the whole read
    surface -- every `sessions.get`, every `/history`, every `/mcp` -- and change
    what every test that asserts a read detaches is describing.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    async with pool.session(sid) as bridge:
        assert bridge.users == 1

    assert bridge.dwelling is False, "no stream lost a transport here"
    assert bridge.remote is None, "so the read's release detached, on the spot"
    assert not bridge._dwell_tasks
