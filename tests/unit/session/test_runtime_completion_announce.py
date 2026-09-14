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
