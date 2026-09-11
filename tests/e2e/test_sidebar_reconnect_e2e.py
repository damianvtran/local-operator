"""A viewer that lost its owner reconnects itself, against a real runtime.

WHAT THIS STAGE ADDS OVER THE UNIT TESTS. The unit file
(``tests/unit/tui/test_sidebar_connect_retry.py``) drives a `AttachedSession`
subclass that reproduces the silent-bind outcome directly. It pins the policy,
but it cannot prove the state it simulates is the one a real disruption
produces. These tests take the disruption itself — a real `RuntimeServer`, real
attach sockets, a real eviction — and assert the user-visible outcome.

THE TRIGGER, AND WHY IT IS NOT THE ROOT CAUSE. `ATTACH_MAX_CLIENTS` is 4 per
runtime and the cap evicts LRU with no goodbye, so a viewer learns it was
dropped only through EOF, which routes to `_on_disconnected` and sets
`_recovering`. That is the dominant trigger on a heavily-multiplexed host
(several TUIs, a desktop proxy, a mobile bridge, sidebar sources, all on one
runtime). But the defect is a property of ANY owner loss:
``test_a_plain_socket_loss_reconnects_with_no_attach_pressure`` reproduces it
from a bare socket close with zero attach clients, which is why the fix is at
the bind postcondition rather than at the cap.

THE TWO WINDOWS ARE BOTH ASSERTED, and that is deliberate. A disruption leaves
the facade `_recovering` for up to `COLD_FALLBACK_S`, after which `_go_cold`
releases it. Reselecting inside that window used to fail; reselecting after it
always worked. Parametrising both is what keeps this a regression guard in
BOTH directions — the ``after`` case passed before the fix and must keep
passing, so a "fix" that broke the healthy path would be caught here.

ISOLATION. `headless_tui_env` gives every test its own config dir, the runtimes
are in-process, and the session ids are synthetic. Nothing here can attach to,
evict or disturb a session outside the test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import ATTACH_MAX_CLIENTS
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    user_message,
    wait_for_adoption,
)

#: How long a settled connect is waited for. A BACKSTOP, not an assertion: the
#: loop below exits the moment the task finishes, so a healthy run never spends
#: it. Sized well above the retry budget (~13 s) so a slow CI runner cannot
#: turn a passing reconnect into a timeout.
_SETTLE_BACKSTOP_S = 60.0

#: Gate recoveries a single reconnect may spend before it counts as a spin.
#:
#: The defect turned `post_display_hook`'s recovery branch into a hot loop —
#: measured on this tree at 1,268 recoveries in ONE reselect, because a cold
#: session published as connected can never satisfy the gate and the branch
#: bought a full-screen relayout on every frame until the timer fired. After
#: the fix the same reselect spends 0 or 1, the same as a healthy first connect.
#:
#: Sized to separate those two populations by orders of magnitude rather than to
#: pin an exact frame count, which legitimately varies with which paint carries
#: the completed compositor map.
_GATE_RECOVERY_CEILING = 8


async def _runtime(config: Path, session_id: str) -> RuntimeServer:
    """A real in-process runtime owning a two-message saved transcript."""
    directory = config / "sessions" / session_id
    await seed_transcript(
        directory,
        [user_message(f"{session_id} q"), assistant_message(f"{session_id} saved answer")],
    )
    owner = build_session(directory, ScriptedStream([]), cwd=config)
    handle = ServingSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    return server


async def _never_take_over() -> None:
    raise AssertionError("a viewer must not take over during this test")


async def _settle(app: OperatorApp, pilot, source: SessionInteraction) -> list[tuple[bool, bool]]:
    """Pump until the connect task finishes, sampling what the user can see.

    Each sample is ``(display_only, is_cold)``. The false-connected state the
    defect produced is ``(False, True)`` — a live-looking transcript over a
    session with no runtime attached — so collecting the samples lets the
    caller assert that state never existed, without asserting how long
    anything took.
    """
    seen: list[tuple[bool, bool]] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SETTLE_BACKSTOP_S
    while loop.time() < deadline:
        await pilot.pause()
        seen.append((source.display_only, bool(getattr(source.session, "is_cold", False))))
        await asyncio.sleep(0.05)
        current = source.connection_task
        if current is None:
            break
        if current.done():
            # A FINISHED TASK IS NOT A SETTLED CONNECT. Every retry round is a
            # NEW task, re-armed from the previous one's `finally` through
            # `call_soon`, so between rounds the current task is briefly `done`
            # with its successor not yet installed. Breaking on `done()` alone
            # returned mid-chain and reported the in-flight `display_only=True`
            # as the outcome — which is how this read as a failure under load,
            # where a slower machine simply lands the sample inside a backoff
            # rather than after the last attempt. Settled means the task
            # finished AND nothing replaced it.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if source.connection_task is current:
                break
    await pilot.pause()
    seen.append((source.display_only, bool(getattr(source.session, "is_cold", False))))
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("reselect_after_s, window", [(1.0, "during"), (10.0, "after")])
async def test_an_evicted_viewer_reconnects_without_ever_looking_connected(
    headless_tui_env: Path,
    reselect_after_s: float,
    window: str,
) -> None:
    """Evict the sidebar's viewer with real clients, then reselect it.

    ``during``: the reselect lands while the facade is still `_recovering`, so
    the bind cannot succeed yet. Before the fix this either latched terminally
    or — the case this asserts — committed a cold session and showed a live
    transcript for 15 s before reverting. Now the connect retries past
    `COLD_FALLBACK_S` and lands connected, with no intermediate live-looking
    state at all.

    ``after``: the reselect lands past the recovery bound. This case worked
    before the fix and must keep working; it is the guard against a retry
    policy that slows down or breaks the healthy path.
    """
    config = headless_tui_env
    servers = {name: await _runtime(config, name) for name in ("origin", "target")}

    def find_owner(_config_dir, session_id):
        server = servers.get(session_id)
        return (server._record, server._record.pid) if server else (None, None)

    async def resume(session_id):
        return await AttachedSession.connect(
            servers[session_id]._record,
            session_id,
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )

    app = OperatorApp(lambda: resume("origin"), resume_factory=resume)
    with patch("local_operator.mobile.attach_client.find_runtime_record", find_owner):
        async with app.run_test(size=(120, 36)) as pilot:
            await wait_for_adoption(app, pilot)
            await asyncio.wait_for(app._sidebar_navigation.select("target"), 30)
            source = app._sidebar_sources["target"]
            if source.connection_task is not None:
                await asyncio.wait_for(asyncio.shield(source.connection_task), 30)
            await pilot.pause()
            # Narrowed once: the source's session is typed as the general
            # protocol, but the recovery state under test (`is_cold`,
            # `_recovering`) is the viewer facade's.
            viewer = source.session
            assert isinstance(viewer, AttachedSession)
            assert not source.display_only
            assert not viewer.is_cold
            # PRECONDITION for the counter assertion below: the gate really is
            # armed on this rig, so a small count there cannot be satisfied by a
            # switch that never consulted it (#856's own note on why
            # `_sidebar_gate_reached` exists beside `_sidebar_gate_recoveries`).
            assert app._sidebar_gate_reached > 0

            # `ATTACH_MAX_CLIENTS` other attach-class clients on the SAME
            # runtime: the ordinary shape of a multiplexed host, and enough to
            # evict the sidebar's viewer through the LRU cap.
            hogs = [
                await AttachedSession.connect(
                    servers["target"]._record,
                    "target",
                    config_dir=config,
                    takeover_factory=_never_take_over,
                    display_window=True,
                )
                for _ in range(ATTACH_MAX_CLIENTS)
            ]
            try:
                loop = asyncio.get_running_loop()
                until = loop.time() + reselect_after_s
                while loop.time() < until:
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                # PRECONDITION: the eviction actually landed, so a pass cannot
                # mean "nothing was ever broken".
                assert viewer.is_cold
                if window == "during":
                    assert viewer._recovering
                else:
                    assert not viewer._recovering

                # THE USER ACTION: selecting the session again from the sidebar.
                recoveries_before = app._sidebar_gate_recoveries
                app._start_sidebar_connection(source)
                seen = await _settle(app, pilot, source)

                assert source.display_only is False
                assert source.connection_error == ""
                assert viewer.is_cold is False
                # THE HEADLINE PROPERTY: at no sampled point was a cold session
                # presented as connected. That state is what the user saw for
                # 15 s, and it is what the readiness gate can never satisfy.
                assert (False, True) not in seen
                # The readiness gate is genuinely reachable now, rather than
                # structurally refused for the life of the switch.
                assert app._sidebar_gate_surface_ready(source)
                # AND THE SPIN IS GONE, in #856's counters. A cold session
                # published as connected made the gate unsatisfiable, so
                # `post_display_hook`'s recovery branch bought a full-screen
                # relayout on EVERY frame until the 15 s timer fired — measured
                # on this tree at 1,268 refusals, all of them recoveries.
                #
                # A CEILING, not an equality, and deliberately loose. #856's
                # unit rig reaches a flat zero, but this one drives real
                # runtimes through a real eviction and spends 0 or 1 depending
                # on which paint carries the completed compositor map — a
                # painted-map refusal that has nothing to do with `is_cold`
                # (verified at the refusal: current=True, display_only=False,
                # cold=False, hist=True). Pinning 0 would assert that timing,
                # not this fix.
                #
                # What IS this fix's property is that the count is BOUNDED
                # rather than proportional to the 15 s window: the defect made
                # this a hot spin whose size was set by how long the timer ran.
                # The ceiling separates 1,268 from 0-1 by three orders of
                # magnitude, so a spin cannot come in under it while honest
                # paint jitter cannot exceed it. A COUNT is also
                # load-independent — a busy machine changes how long the frames
                # take, not how many the gate refuses.
                assert app._sidebar_gate_recoveries - recoveries_before <= _GATE_RECOVERY_CEILING
            finally:
                for hog in hogs:
                    await hog.dispose()


@pytest.mark.asyncio
async def test_a_loss_in_the_bind_to_paint_window_heals_instead_of_latching(
    headless_tui_env: Path,
) -> None:
    """THE WINDOW ITSELF, which the two tests above only approach.

    ``during`` reselects a session that is ALREADY cold, so the connect body's
    bind postcondition refuses it before anything is committed. The operator's
    report is the other ordering: the session was live when the bind check ran
    and the owner was lost in the gap between that check and the first painted
    frame — the window ``post_display_hook``'s readiness gate owns. A cold
    session committed in that window satisfies no frame the gate can ever
    accept (its FIRST check is ``is_cold``), so pre-fix the gate refused every
    frame for 15 s, each refusal buying a forced full-screen relayout (measured
    on the architect's rig: 15.10 s, 1,820 refusals, all recoveries), and the
    timer's ``SurfaceNotReady`` — terminal on first by #883's design — latched a
    loss the user's reselect healed in 0.17 s.

    The injection is deterministic rather than raced: ``ensure_display_current``
    is the first await AFTER the bind postcondition and BEFORE presentation, so
    evicting the viewer inside it puts a REAL ``ATTACH_MAX_CLIENTS`` eviction
    exactly in the window while every other line of the real connect body runs.
    Racing it (as the architect's offset sweep does) is load-dependent: on a
    fast host the connect's fast path lands first and no frame is ever armed.
    """
    config = headless_tui_env
    servers = {name: await _runtime(config, name) for name in ("origin", "target")}

    def find_owner(_config_dir, session_id):
        server = servers.get(session_id)
        return (server._record, server._record.pid) if server else (None, None)

    async def resume(session_id):
        return await AttachedSession.connect(
            servers[session_id]._record,
            session_id,
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )

    app = OperatorApp(lambda: resume("origin"), resume_factory=resume)
    hogs: list[AttachedSession] = []
    with patch("local_operator.mobile.attach_client.find_runtime_record", find_owner):
        async with app.run_test(size=(120, 36)) as pilot:
            await wait_for_adoption(app, pilot)
            await asyncio.wait_for(app._sidebar_navigation.select("target"), 30)
            source = app._sidebar_sources["target"]
            if source.connection_task is not None:
                await asyncio.wait_for(asyncio.shield(source.connection_task), 30)
            await pilot.pause()
            viewer = source.session
            assert isinstance(viewer, AttachedSession)
            assert not source.display_only
            assert not viewer.is_cold

            original_display = AttachedSession.ensure_display_current

            async def evicting_display(self) -> None:
                await original_display(self)
                if self is viewer and not hogs:
                    # REAL eviction: the same number of genuine attach clients
                    # as the cap, so the LRU drop is the runtime's own.
                    hogs.extend(
                        [
                            await AttachedSession.connect(
                                servers["target"]._record,
                                "target",
                                config_dir=config,
                                takeover_factory=_never_take_over,
                                display_window=True,
                            )
                            for _ in range(ATTACH_MAX_CLIENTS)
                        ]
                    )

            try:
                with patch.object(AttachedSession, "ensure_display_current", evicting_display):
                    # The state the connect task is always entered from: the
                    # saved excerpt is on screen, and the user asks for the
                    # session again.
                    source.display_only = True
                    recoveries_before = app._sidebar_gate_recoveries
                    app._start_sidebar_connection(source)
                    seen = await _settle(app, pilot, source)

                # PRECONDITION: the eviction really landed in the window, so a
                # pass cannot mean "nothing was ever lost".
                assert hogs, "the eviction never fired; this test proves nothing"
                assert (
                    viewer.is_cold is False
                ), "the reconnect did not heal: the owner was never recovered"
                assert source.display_only is False
                assert source.connection_error == ""
                # THE HEADLINE: the transient loss healed with NO user action,
                # and a cold session was never presented as LIVE on the way.
                #
                # Only `(False, cold)` is forbidden. `(True, True)` — the saved
                # excerpt on screen while the owner is gone — is the HONEST
                # state (that is what "Saved · Connecting…" means), and the
                # point of this fix is that it is a state the app RETRIES out
                # of rather than a verdict it latches.
                assert (False, True) not in seen
                # NO SPIN: the commit->paint window is where the 15 s refusal
                # storm lived, so its counter is the sharpest reading here.
                assert app._sidebar_gate_recoveries - recoveries_before <= _GATE_RECOVERY_CEILING
            finally:
                for hog in hogs:
                    await hog.dispose()


@pytest.mark.asyncio
async def test_a_plain_socket_loss_reconnects_with_no_attach_pressure(
    headless_tui_env: Path,
) -> None:
    """GENERALITY: the fix addresses owner loss, not the attach cap.

    Raising `ATTACH_MAX_CLIENTS` was explicitly rejected as a fix for exactly
    this reason. Here the socket is dropped the way a send timeout or an owner
    blip drops it — one viewer, zero other clients, no cap involved — and the
    facade lands in the same `is_cold` + `_recovering` state, so a connect that
    trusts `_ensure_bound`'s silent return fails identically.
    """
    config = headless_tui_env
    session_id = "target"
    server = await _runtime(config, session_id)

    def find_owner(_config_dir, requested):
        return (server._record, server._record.pid) if requested == session_id else (None, None)

    with patch("local_operator.mobile.attach_client.find_runtime_record", find_owner):
        viewer = await AttachedSession.saved_preview(
            session_id,
            config_dir=config,
            cwd=str(config),
            takeover_factory=_never_take_over,
        )
        try:
            await viewer._ensure_bound()
            await viewer.ensure_display_current()
            assert not viewer.is_cold
            assert server.attach_clients() == 1

            client = viewer._client
            assert client is not None
            client.close()
            await asyncio.sleep(0)
            for _ in range(20):
                await asyncio.sleep(0.05)
                if viewer.is_cold:
                    break
            # No cap, no eviction, no competing client — and the same state.
            assert viewer.is_cold
            assert viewer._recovering
            assert server.attach_clients() == 0

            # THE DEFECT, in the two calls the connect body makes: the bind
            # neither binds nor raises, so a caller that does not check the
            # postcondition proceeds against a cold session.
            await viewer._ensure_bound()
            assert viewer.is_cold, (
                "_ensure_bound bound the facade; this test no longer covers the "
                "silent-return outcome the fix exists to catch"
            )
        finally:
            await viewer.dispose()
