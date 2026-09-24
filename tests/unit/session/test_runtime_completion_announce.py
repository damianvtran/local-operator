"""Rung 4: when the RUNTIME raises a completion's banner, and when it must not.

A detached runtime is the only surface left when nothing is running, and before
this arm existed a completion with no TUI and no desktop app was announced by
NOBODY. The arm is the last rung of a ladder rather than a race, and the
distinction is the whole test: the runtime learns about a completion EARLIER
than every other surface (at turn settle, against the feed's 100 ms poll and the
TUI's 1 s tick), so an arm that claimed unconditionally would win every
completion and make both richer paths dead.

So eligibility is decided BEFORE the claim, and these tests parametrise the four
rungs — a watching surface, a notify-capable desktop, a running TUI, and
nothing — asserting for each whether a banner was raised and whether the
delivery watermark was spent.

The rig is the production one (``test_busy_settles.py``'s shape): a real
``Session`` behind the real ``ServingSessionHandle``. Only three things are
doubled, and each is a boundary rather than a judgement: the OS spawn, the
machine-wide presence read, and the "is a TUI running" probe.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import local_operator.session.runtime.presence as presence_module
import local_operator.tui.notify as notify_module
from local_operator.paths import config_dir
from local_operator.session.attention import AttentionStore
from local_operator.session.runtime.presence import desktop_delivery_present
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session


@pytest.fixture(autouse=True)
def _notification_gate_off() -> Iterator[None]:
    """Opt this module IN to the notification path, deliberately and visibly.

    ``tests/conftest.py`` arms ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` for every test
    (and at import time, so spawned children inherit it), and the runtime's
    rung-4 arm now returns SETTLED before it claims anything while that switch is
    on — which is the whole subject here: these tests parametrise the LADDER, so
    the ladder must be reachable.

    Set and restored by hand rather than through ``monkeypatch``: that fixture
    is function-scoped and SHARED with the tests, so a test calling
    ``monkeypatch.undo()`` would re-arm the gate mid-test.
    """
    prior = os.environ.pop("LOCAL_OPERATOR_NO_NOTIFICATIONS", None)
    yield
    if prior is not None:
        os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = prior


@pytest.fixture
def banners(monkeypatch):
    """Record every rung-4 banner, and let a test force a failed spawn.

    Doubled at ``tui/notify.detached_notify`` — the same funnel every other
    delivery on this path uses — so the assertion is about the CALL rather than
    about a real OS notification, which is what the design's back-end evidence
    asks for ("assert the spawn argv, not a real OS banner").
    """
    calls: list[dict[str, str]] = []
    state = {"delivered": True}

    def fake(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        calls.append({"title": title, "body": body, "session_id": session_id, "subtitle": subtitle})
        return bool(state["delivered"])

    monkeypatch.setattr(notify_module, "detached_notify", fake)
    return calls, state


async def _rig(
    tmp_path: Path,
    monkeypatch,
    *,
    watching: tuple[str, ...] = (),
    tui: bool = False,
) -> tuple[Any, ServingSessionHandle]:
    """A real session under the production handle, with two seams doubled only.

    ``_watching_surfaces`` is the per-session live connection table, which the
    runtime server owns and which has its own suite (``test_desktop_*`` and the
    server tests); doubling it here keeps this file about the LADDER. Everything
    else is real, including the presence file and the viewer registry.
    """
    directory = tmp_path / "sess"
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, ScriptedStream([]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    monkeypatch.setattr(handle, "_watching_surfaces", lambda: frozenset(watching))
    if tui:
        _publish_tui_viewer()
    return session, handle


def _publish_tui_viewer() -> None:
    """A live TUI viewer record in the isolated config root.

    Real rather than patched, because that is the signal rung 3 is built on and
    the whole point of it is that it is MACHINE-wide: ``scan_viewers`` reaps a
    dead pid and a stale heartbeat, so a TUI that crashed cannot keep every
    runtime silent forever.
    """
    import os

    from local_operator.session.runtime.viewers import (
        TUI_SURFACE,
        ViewerRecord,
        publish_viewer,
    )

    publish_viewer(
        ViewerRecord(pid=os.getpid(), surface=TUI_SURFACE, control_port=1, control_key="k"),
        config_dir(),
    )


def _publish_desktop_presence(**window: Any):
    """A real ``run/desktop/delivery.json``, through the production writer."""
    from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher

    publisher = DesktopDeliveryPublisher(config_dir())
    publisher.update(
        "sub-1",
        can_notify=True,
        can_notify_kinds=["complete", "error"],
        window={"exists": True, "focused": True, "visible": True, "minimized": False, **window},
    )
    presence_module.reset_cache()
    return publisher


def _publish(kind: str, session_id: str, anchor: str = "a1") -> str:
    """Publish a completion into the machine-wide store, as a turn would."""
    import uuid

    token = str(uuid.uuid4())
    AttentionStore(config_dir() / "attention.db").publish(
        f"session/{session_id}", token, anchor, kind
    )
    return token


async def _arm(handle: ServingSessionHandle, session_id: str, kind: str = "complete") -> str:
    token = _publish(kind, session_id)
    await asyncio.to_thread(handle._announce_completion)
    return token


def _delivered(session_id: str, token: str) -> bool:
    """Whether the delivery watermark has passed this completion's row.

    Read straight out of the store rather than by attempting a claim.
    ``claim_delivery`` SPENDS the watermark — "exactly one caller ever wins" —
    so asking with it would be a destructive probe, and a test that consumed the
    claim it was inspecting would report the opposite of the truth for every
    assertion after it.
    """
    import sqlite3

    path = config_dir() / "attention.db"
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT sequence FROM completions WHERE conversation=? AND token=?",
            (f"session/{session_id}", token),
        ).fetchone()
        assert row is not None, "no completion row for this token"
        delivered = conn.execute(
            "SELECT delivered FROM deliveries WHERE conversation=?",
            (f"session/{session_id}",),
        ).fetchone()
    return bool(delivered) and int(delivered[0]) >= int(row[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("watching", "desktop", "tui", "expected"),
    [
        # Rung 1 — a surface is showing the card in band.
        (("attach",), False, False, False),
        (("viewer",), False, False, False),
        # Rung 2 — a desktop app can attempt the completion banner itself.
        ((), True, False, False),
        # Rung 3 — a TUI's own background announcer owns it.
        ((), False, True, False),
        # Rung 4 — nothing else is eligible, so the runtime must speak.
        ((), False, False, True),
    ],
)
async def test_exactly_one_rung_delivers_the_completion(
    tmp_path: Path, monkeypatch, banners, watching, desktop, tui, expected
) -> None:
    session, handle = await _rig(tmp_path, monkeypatch, watching=watching, tui=tui)
    publisher = _publish_desktop_presence() if desktop else None
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        token = await _arm(handle, session_id)
        assert bool(calls) is expected, (calls, watching, desktop, tui)
        # The claim follows eligibility, never the other way round: a rung that
        # did not deliver must leave the watermark free for the surface that
        # does, and the rung that delivered must have spent it.
        assert _delivered(session_id, token) is expected
        if expected:
            assert calls[0]["session_id"] == session_id
            assert calls[0]["title"]
            assert calls[0]["subtitle"], "the state category must ride the banner"
    finally:
        if publisher is not None:
            publisher.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_watching_surface_is_never_raced_for_the_claim(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """Rung 1 is checked BEFORE the store is even read for eligibility.

    A surface that is displaying the session shows the card in band; spending
    the watermark for it would also suppress the in-band path's own report.
    """
    session, handle = await _rig(tmp_path, monkeypatch, watching=("attach",))
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        token = await _arm(handle, session_id)
        assert calls == []
        assert _delivered(session_id, token) is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_failed_spawn_hands_the_claim_back(tmp_path: Path, monkeypatch, banners) -> None:
    """A watermark asserting a banner nobody received is the silent hole.

    Exactly the hole ``release_delivery`` exists to close, and the reason the
    argument ordering matters: a spawn that reports nothing went out must leave
    the next surface free to try.
    """
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, state = banners
    state["delivered"] = False
    try:
        session_id = handle._session_id_for_resume()
        token = await _arm(handle, session_id)
        assert calls, "the arm did not attempt a banner at all"
        assert _delivered(session_id, token) is False, "the claim was not handed back"
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["interrupted"])
async def test_a_kind_outside_the_bridge_set_is_never_announced(
    tmp_path: Path, monkeypatch, banners, kind
) -> None:
    """``interrupted`` is the user's own Ctrl+C a moment ago.

    It is the only non-notifiable kind ``AttentionStore`` can hold — the column
    is constrained to complete/error/interrupted — which is why the arm's kind
    check is a plain ``BRIDGE_NOTIFIABLE_KINDS`` membership test rather than a
    gate-specific branch: ``ask``/``approval`` never reach the store at all.
    """
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        token = await _arm(handle, session_id, kind=kind)
        assert calls == []
        assert _delivered(session_id, token) is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_gate_path_keeps_its_per_session_lease(tmp_path: Path, monkeypatch) -> None:
    """DESIGN REVIEW B2(b): the machine-wide presence may not reach a GATE.

    The feed carries completions only, so widening the gate's suppression to
    the machine-wide lease would silence a background session's parked `ask`
    with nothing to replace it — a regression against today. The gate path
    therefore keeps reading the per-SESSION reachability table, and this asserts
    the two predicates were not conflated.
    """
    session, handle = await _rig(tmp_path, monkeypatch)
    publisher = _publish_desktop_presence()

    class _LeaseStub:
        def notification_surfaces(self):
            return frozenset()

    try:
        handle._registrant = _LeaseStub()
        assert handle._desktop_notification_available() is False
        # The machine-wide lease IS present and does NOT move that answer.
        assert desktop_delivery_present(config_dir(), "complete") is True
        handle._registrant = _LeaseStubWithDesktop()
        assert handle._desktop_notification_available() is True
    finally:
        publisher.close()
        await session.dispose()


class _LeaseStubWithDesktop:
    def notification_surfaces(self):
        return frozenset({"desktop"})


@pytest.mark.asyncio
async def test_an_already_claimed_completion_is_not_announced_twice(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """Two turns' worth of ticks must not become two banners for one completion."""
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        await _arm(handle, session_id)
        assert len(calls) == 1
        # The SAME completion, seen again with nothing having changed.
        await asyncio.to_thread(handle._announce_completion)
        assert len(calls) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_seen_completion_is_never_re_announced(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """A read completion is not news — the watermark, not a timer, decides."""
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        token = _publish("complete", session_id)
        AttentionStore(config_dir() / "attention.db").acknowledge(f"session/{session_id}", token)
        await asyncio.to_thread(handle._announce_completion)
        assert calls == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_turn_settled_hook_still_carries_both_consumers(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """ONE SLOT, TWO CONSUMERS, and neither may be dropped.

    ``Session.on_turn_settled`` already carried the record's ``busy`` settle.
    The runtime's handler chains the completion arm onto it rather than growing
    a second hook on ``Session``; dropping either call leaves a record stuck
    busy or a completion announced by nobody, and both are silent failures.
    """
    session, handle = await _rig(tmp_path, monkeypatch)
    try:
        assert session.on_turn_settled is not None
        busy: list[bool] = []
        monkeypatch.setattr(handle, "_publish_busy_soon", lambda: busy.append(True))
        session.on_turn_settled()
        assert busy == [True], "the busy settle was dropped from the hook"
        assert handle._completion_task is not None, "the completion arm was not scheduled"
        handle._completion_task.cancel()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_arm_stays_silent_while_the_handle_is_disposing(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """A runtime on its way out must not raise a banner for a dead process."""
    session, handle = await _rig(tmp_path, monkeypatch)
    try:
        handle._disposing = True
        handle._schedule_completion_announce()
        assert handle._completion_task is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_runtime_reads_the_real_presence_file(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """The rung-2 wiring, against a real ``run/desktop/delivery.json``.

    The parametrised case above already covers it; this one exists to make the
    read path explicit — the arm imports ``desktop_delivery_present`` by name at
    call time and reads the host's aggregate, so a patch on a different name
    would let the ladder pass while nothing was wired.
    """
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners
    publisher = _publish_desktop_presence()
    try:
        session_id = handle._session_id_for_resume()
        token = _publish("complete", session_id)
        await asyncio.to_thread(handle._announce_completion)
        assert calls == [], "a notify-capable desktop must silence the runtime"
        assert _delivered(session_id, token) is False
        # Withdraw the lease and the runtime must speak again, which is what
        # makes this a LIVE predicate rather than a mode set at boot.
        publisher.close()
        presence_module.reset_cache()
        await asyncio.to_thread(handle._announce_completion)
        assert len(calls) == 1, "the runtime stayed silent for a withdrawn lease"
        assert calls[0]["session_id"] == session_id
    finally:
        publisher.close()
        await session.dispose()


# -- R7: the retry ladder --------------------------------------------------
#
# Review round 1 found that the ONLY scheduled attempt was the turn-settled
# task: a failed spawn released the claim and scheduled nothing, and deferring
# to a desktop lease that then disappeared ended the task. With no other
# announcer the completion lost its banner permanently. The round-1 test looked
# recovered only because it called the arm a second time BY HAND — which is the
# thing production could not do, so the tests below never call the arm directly.


async def _until(predicate, timeout_s: float = 3.0) -> None:
    """Poll until ``predicate()`` or fail loudly. Avoids asserting on a sleep."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never became true")


#: Slack the ladder guard carries over the ladder's own schedule. The guard is
#: derived, never a literal: see `_ladder_exhausted`.
_LADDER_GUARD_SLACK_S = 10.0


async def _ladder_exhausted(
    handle: ServingSessionHandle, *, timeout_s: float | None = None
) -> None:
    """Wait until the announcement ladder has STOPPED, or fail loudly.

    The ladder runs one attempt per task and REPLACES ``handle._completion_task``
    when it schedules the next rung — see the comment at that assignment in
    ``serving.py``, which is the contract this rests on — so a slot still holding
    the task that just finished is the ladder's own end-of-ladder signal: no
    successor was scheduled, and that attempt's handback landed inside the task
    before it returned. Awaiting those tasks IS awaiting the event R7 is about; a
    banner CALL COUNT is only a proxy for it, and a wrong one in both directions:

    - ``len(calls) >= 4`` goes true the moment the fourth attempt ENTERS its
      sink — inside ``detached_notify``, BEFORE that attempt's
      ``release_delivery`` — so the read lands on the claim the attempt is still
      holding (CI run 35377466211 died on the same line, and the base rate is
      measured at 8 in 120 runs under CI-like concurrency).
    - A count is still a wall-clock bet even when it is not read too early:
      ``_until``'s 3 s bound expires before a slowed ladder runs out its rungs,
      and that is the SAME defect reported as the other signature,
      ``AssertionError: the condition never became true``.

    ``timeout_s`` defaults to the ladder's OWN schedule plus slack, read from the
    module the ladder schedules from, so a caller that compresses the delays
    (``_fast_ladder``) compresses the guard with them and a caller that does not
    gets one wider than the production 2 + 8 + 30 s. A literal here would be a
    bound a caller could silently undercut: the compressed test would be guarded
    by 30 s while standing in for a 40 s schedule.
    """
    if timeout_s is None:
        import local_operator.session.runtime.serving as serving_module

        timeout_s = sum(serving_module._COMPLETION_RETRY_DELAYS_S) + _LADDER_GUARD_SLACK_S
    guard = asyncio.timeout(timeout_s)
    try:
        async with guard:
            while True:
                task = handle._completion_task
                if task is None:
                    # NEVER SCHEDULED, which from here is indistinguishable from
                    # an exhausted ladder — the slot is the only handle this
                    # helper has on it. The caller pins which one it is: the rung
                    # count below, so this cannot pass vacuously either way.
                    return
                await task
                if handle._completion_task is task:
                    return
    except TimeoutError as error:
        if not guard.expired():
            raise
        raise AssertionError("the completion ladder never exhausted") from error


def _fast_ladder(monkeypatch, *delays: float) -> None:
    """Compress the ladder's DELAYS, never its shape.

    The real constants are chosen against real timeouts (see
    ``_COMPLETION_RETRY_DELAYS_S``); asserting them here would spend about 40 s
    proving arithmetic. What is under test is that a retry happens at all, how
    it terminates, and that it reaches the same eligibility gate each time.
    """
    import local_operator.session.runtime.serving as serving_module

    monkeypatch.setattr(serving_module, "_COMPLETION_RETRY_DELAYS_S", delays or (0.05, 0.05, 0.05))


@pytest.mark.asyncio
async def test_a_failed_spawn_is_retried_without_being_rearmed_by_hand(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """R7's first half: the ladder has to retry a transient rung-4 failure.

    The sink fails its FIRST attempt and succeeds after, so a banner can only
    appear if something scheduled the second attempt — and nothing in this test
    does. Under the round-1 code ``calls`` would hold exactly one entry and the
    completion would be announced by nobody.
    """
    _fast_ladder(monkeypatch)
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners
    attempts = {"n": 0}

    def flaky(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        attempts["n"] += 1
        calls.append({"title": title, "body": body, "session_id": session_id, "subtitle": subtitle})
        return attempts["n"] > 1

    monkeypatch.setattr(notify_module, "detached_notify", flaky)
    try:
        session_id = handle._session_id_for_resume()
        token = _publish("complete", session_id)
        handle._schedule_completion_announce()
        await _until(lambda: len(calls) >= 2)
        assert attempts["n"] == 2, "the ladder retried more than the failure needed"
        assert _delivered(session_id, token) is True, "the retry did not land the banner"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_deferred_desktop_lease_is_rechecked_once_it_goes_away(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """R7's second half: a DEFERRAL is not a settlement.

    The reviewer's second reproduction: cache a live lease, drop it, settle, and
    wait — ``fresh_presence_now=False; attempts_after_expiry=0``. The deferral is
    a read of a CACHED answer, so the only thing that can catch the app leaving
    is a second attempt after the cache expires.

    Both knobs are compressed and their ORDER is what the test asserts: the
    retry delay must exceed the cache TTL, or the second attempt re-reads the
    same stale lease and the ladder burns out inside the window. The real pair
    is 2 s / 2 s, which is why the first rung of the ladder is exactly the cache
    TTL rather than an arbitrary round number.
    """
    import local_operator.session.runtime.serving as serving_module

    monkeypatch.setattr(serving_module, "_COMPLETION_RETRY_DELAYS_S", (0.25, 0.25, 0.25))
    monkeypatch.setattr(presence_module, "PRESENCE_CACHE_TTL_S", 0.05)
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners
    publisher = _publish_desktop_presence()
    try:
        # Warm the cache while the app is up: the FIRST attempt must defer.
        assert desktop_delivery_present(config_dir(), "complete") is True
        session_id = handle._session_id_for_resume()
        token = _publish("complete", session_id)
        publisher.close()

        handle._schedule_completion_announce()
        await _until(lambda: len(calls) >= 1)
        assert calls[0]["session_id"] == session_id
        assert _delivered(session_id, token) is True
    finally:
        publisher.close()
        presence_module.reset_cache()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_ladder_is_bounded_and_stops_when_nothing_can_deliver(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """The bound, asserted rather than assumed.

    A retry that never ends would be the unbounded runtime residency the review
    forbids, so the ladder has to EXHAUST: with a sink that always fails, the
    attempt count is exactly ``1 + len(delays)`` and then stops. Nothing further
    is scheduled, and the durable unseen mark is left alone so the completion
    still reads as unread everywhere.
    """
    import local_operator.session.runtime.serving as serving_module

    _fast_ladder(monkeypatch, 0.05, 0.05)
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, state = banners
    state["delivered"] = False
    try:
        session_id = handle._session_id_for_resume()
        token = _publish("complete", session_id)
        handle._schedule_completion_announce()
        await _until(lambda: len(calls) >= 3)
        await asyncio.sleep(0.2)  # any further retry would land inside this
        assert len(calls) == 3, f"the ladder ran past its bound: {len(calls)} attempts"
        assert len(serving_module._COMPLETION_RETRY_DELAYS_S) == 2
        assert _delivered(session_id, token) is False, "a failed banner must not spend the claim"
        # The event is still visible where it matters most.
        from local_operator.session.attention import AttentionStore as _Store

        assert (
            _Store(config_dir() / "attention.db").state(f"session/{session_id}")["unseen"] is True
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_exception_after_the_claim_hands_it_back(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """R7: the release branch must survive a RAISE, not just a ``False``.

    The claim is taken before the raise, so an exception escaping past the
    ``if not delivered`` branch left the watermark asserting a banner nobody
    received — a completion that was neither announced nor left claimable by the
    next surface. The raise is forced through the real ``detached_notify``
    funnel so the arm's own exception path is the one under test.
    """
    _fast_ladder(monkeypatch)
    session, handle = await _rig(tmp_path, monkeypatch)
    calls, _state = banners

    def exploding(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        calls.append({"title": title, "body": body, "session_id": session_id, "subtitle": subtitle})
        raise OSError("no notification helper")

    monkeypatch.setattr(notify_module, "detached_notify", exploding)
    try:
        session_id = handle._session_id_for_resume()
        token = _publish("complete", session_id)
        handle._schedule_completion_announce()
        # Every attempt fails, so the ladder exhausts and the claim is free.
        #
        # Wait for the LADDER, never for a count of banner calls: `len(calls) >= 4`
        # becomes true as the fourth attempt ENTERS its sink, inside
        # `detached_notify` and before that attempt hands the claim back, so the
        # assertion below can read the watermark the attempt is still holding.
        # Both reported signatures are that one defect — see `_ladder_exhausted`.
        await _ladder_exhausted(handle)
        # ...and then pin the RUNG COUNT the wait no longer proves. Waiting on the
        # ladder's own end makes the wait honest, but a ladder whose successor is
        # created and never stored in `_completion_task` also looks exhausted after
        # ONE rung — QA round 1's `leakslot` mutation, which this test passed — and
        # so does a completion that rung 2 or 3 answers, where no delivery is ever
        # attempted and the claim assertion below would pass vacuously because
        # nothing took the claim. The count is the effective schedule, so a
        # compressed ladder (`_fast_ladder`) expects its own number of rungs.
        import local_operator.session.runtime.serving as serving_module

        rungs = 1 + len(serving_module._COMPLETION_RETRY_DELAYS_S)
        assert len(calls) == rungs, (
            f"the ladder ran {len(calls)} rung(s), expected {rungs}: either no delivery "
            f"was attempted at all, or a rung scheduled a successor the handle never "
            f"stored in `_completion_task`"
        )
        assert _delivered(session_id, token) is False, "the claim survived a raise"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_machine_started_session_never_announces_a_completion(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """A session the operator's own listings HIDE must not reach his lock screen.

    THE RULE, and the reason it is the runtime that owns it: the completion of a
    delegated run belongs to the session that asked for it, which is already
    showing it, and the operator was never told this run existed. Before this
    gate the run's own process was the one leak left on the path — every listing
    filtered the hidden origin and this arm did not, so a throwaway
    ``lop exec`` put its last assistant line on the screen.

    Both halves of the outcome are asserted, because either alone is a weaker
    claim: nothing was spawned, AND no delivery claim was taken, so the durable
    unseen mark is exactly as eligible as it was for whatever surface does own
    the row.
    """
    from local_operator.resume import ORIGIN_AGENT_SHELL, mark_session_origin

    session, handle = await _rig(tmp_path, monkeypatch)
    mark_session_origin(session._transcript.directory, ORIGIN_AGENT_SHELL)
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        token = await _arm(handle, session_id)
        assert calls == [], "a hidden session raised a banner"
        assert _delivered(session_id, token) is False, "a hidden session spent the claim"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_workstream_the_operator_asked_for_still_announces(
    tmp_path: Path, monkeypatch, banners
) -> None:
    """THE MIRROR CASE, and it matters as much as the gate.

    ``agent-workstream`` is a USER origin: the operator asked for the run, so it
    is listed — and a gate that silenced it would be the very bug this change
    exists to fix, one rung up. The pair with the cell above is the whole point:
    the two sessions differ only by the marker.
    """
    from local_operator.resume import ORIGIN_AGENT_WORKSTREAM, mark_session_origin

    session, handle = await _rig(tmp_path, monkeypatch)
    mark_session_origin(
        session._transcript.directory,
        ORIGIN_AGENT_WORKSTREAM,
        opened_by={"session": "req000000001"},
    )
    calls, _state = banners
    try:
        session_id = handle._session_id_for_resume()
        token = await _arm(handle, session_id)
        assert calls, "the workstream the operator asked for was silenced"
        assert calls[0]["session_id"] == session_id
        assert _delivered(session_id, token) is True
    finally:
        await session.dispose()
