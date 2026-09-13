"""The PREWARM leg of the sidebar: a speculative lease that heals on the click.

WHY A SEPARATE MODULE. The sibling file (``test_sidebar_reconnect_e2e.py``)
leases its source with ``saved_preview`` — the CLICK branch — and never
exercises the speculative one. That is the branch that latched for the
operator: ``saved_preview`` builds a VIEWER (``_can_go_cold = True``), while
the prewarm branch called ``AttachedSession.connect`` with no contract asked
for, so the facade it created returned from ``_ensure_bound``'s first guard
(``if not self._can_go_cold or self._disposed: return``) on every round and
spent the whole retry budget on no-ops. The source cache hands that same facade
to the user's later click, so the click could not heal it either. These cases
live in their own module rather than being appended to the sibling because open
PR #1025 also edits that file; keeping them apart lets both land without a
merge conflict.

WHAT THE FIX CHANGES, IN THE TWO SHAPES THAT MATTER.

* A plain socket loss while the prewarm facade is ``_recovering`` is bounded
  either way: the recovery loop releases the facade at ``COLD_FALLBACK_S``. The
  first case pins that the postcondition the sibling file asserts for a
  ``saved_preview`` source also holds for a prewarm-created one — the click
  commits, never as a cold session presented as connected, and the readiness
  gate is genuinely reachable.
* A STOP is the shape production logged, and it is the one no bound on the
  recovery loop can reach: ``_on_disconnected``'s deliberate-stop branch
  returns before setting ``_recovering``, and ``_recover_runtime`` returns at
  its entry guard, so the facade is cold with NO loop running and nothing left
  to release it. The operator's two latches (19:23:54, 19:24:25) both logged
  ``recovering=False`` for exactly this reason, with a live, dialable owner.
  The second case reproduces it against real runtimes: a peer asks the owner to
  stop, and then the conversation comes back as a new runtime for the same id.

ISOLATION. ``headless_tui_env`` gives every test its own config dir, the
runtimes are in-process, the session ids are synthetic, and nothing here reads
or touches a session outside the test — no inherited ``CMUX_*`` id is used to
name anything.

NO WALL-CLOCK ASSERTIONS. Per AGENTS.md "Timing, flakes" the postcondition is
asserted structurally: which state was ever presented (``display_only`` against
``is_cold``), whether the gate is reachable, how many gate recoveries the
connect spent, and how many attach clients the owner holds. The retry budget
itself is pinned by ``tests/unit/tui/test_sidebar_connect_retry.py``, as a
relationship between two constants.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
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
#: loop exits the moment the task finishes, so a healthy run never spends it.
#: Sized well above the retry budget (~13 s) so a slow runner cannot turn a
#: passing heal into a timeout.
_SETTLE_BACKSTOP_S = 60.0

#: Gate recoveries a single reconnect may spend before it counts as a spin.
#: Same ceiling and same reasoning as the sibling file: the defect made the
#: recovery branch a hot loop (measured at 1,268 refusals in one reselect), and
#: a healthy connect spends 0 or 1 depending on which paint carries the
#: completed compositor map. A count separates those populations by orders of
#: magnitude and, unlike an elapsed time, does not move with machine load.
_GATE_RECOVERY_CEILING = 8


async def _runtime(config: Path, session_id: str, *, stoppable: bool = False) -> RuntimeServer:
    """A real in-process runtime owning a two-message saved transcript.

    ``stoppable`` wires the hook the real runtime installs in
    ``process.amain``. Without it a socket ``stop`` op falls back to disposing
    the session in place, which leaves this runtime serving — fine for a rig
    that never stops anything, and useless for one that does. Assigned on the
    concrete handle (the ``SessionHandle`` protocol does not declare the hook),
    so the assignment stays type-checked rather than going through a cast.
    """
    directory = config / "sessions" / session_id
    await seed_transcript(
        directory,
        [user_message(f"{session_id} q"), assistant_message(f"{session_id} saved answer")],
    )
    owner = build_session(directory, ScriptedStream([]), cwd=config)
    handle = ServingSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    if stoppable:
        # A `def` rather than a lambda: the hook's declared type returns None,
        # and the teardown task must be created without being returned into it.
        def stop_this_runtime() -> None:
            asyncio.get_running_loop().create_task(server.aclose())

        handle.on_stop_requested = stop_this_runtime
    return server


async def _never_take_over() -> None:
    raise AssertionError("a viewer must not take over during this test")


async def _settle(app: OperatorApp, pilot, source: SessionInteraction) -> list[tuple[bool, bool]]:
    """Pump until the connect task finishes, sampling what the user can see.

    Each sample is ``(display_only, is_cold)``. The state the defect produced is
    ``(False, True)`` — a live-looking transcript over a session with no runtime
    attached — so collecting the samples lets the caller assert that state never
    existed, without asserting how long anything took. A finished task is not a
    settled connect: every retry round is a NEW task re-armed from the previous
    one's ``finally``, so the task is briefly ``done`` with no successor yet.
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
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if source.connection_task is current:
                break
    await pilot.pause()
    seen.append((source.display_only, bool(getattr(source.session, "is_cold", False))))
    return seen


def _status_text(app: OperatorApp) -> str:
    assert app._status is not None
    return app._status.render_text(200).plain


def _assert_healed(
    app: OperatorApp, source: SessionInteraction, seen: list[tuple[bool, bool]]
) -> None:
    """The postcondition, in one place so both cases assert the same thing."""
    viewer = source.session
    assert isinstance(viewer, AttachedSession)
    assert source.display_only is False, "the connect did not commit"
    assert source.connection_error == ""
    assert viewer.is_cold is False, "the click settled on a cold session"
    # At no sampled point was a cold session presented as connected: that is the
    # state the user saw for 15 s before the fix, and it is what the readiness
    # gate can never satisfy.
    assert (False, True) not in seen
    assert app._sidebar_gate_surface_ready(source)
    status = _status_text(app)
    assert "Reconnect failed" not in status
    assert "Select again to retry" not in status


@pytest.mark.asyncio
async def test_a_prewarmed_source_heals_after_a_plain_socket_loss(headless_tui_env: Path) -> None:
    """GENERALITY for the prewarm leg: no attach cap, one viewer, a bare close.

    The production-dominant owner loss is socket-level (``reader eof`` 13,435,
    ``reader reset`` 1,667 and 204 busy-owner send timeouts in ``mobile.log`` on
    this host) rather than the attach cap the sibling file uses, and the cap
    appears ZERO times in that log. Dropping the socket the way a send timeout
    drops it leaves the prewarm facade in the same ``is_cold`` state, so a click
    that trusts ``_ensure_bound``'s silent return fails identically.
    """
    config = headless_tui_env
    servers = {name: await _runtime(config, name) for name in ("origin", "target")}

    # ``**_probe`` absorbs the keyword the engage loop's DISCOVERY path passes
    # (`find_runtime_record(..., check_zombie=False)`): a rig that accepts only
    # the two positional arguments turns that call into a `TypeError`, which the
    # connect reports as its `connection_error`. The sibling file's rigs carry
    # the same parameter for the same reason.
    def find_owner(_config_dir, session_id, **_probe):
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
            source = await app._lease_sidebar_source("target", speculative=True)
            viewer = source.session
            assert isinstance(viewer, AttachedSession)
            # PRECONDITION: the speculative lease attached to the LIVE owner, so
            # a pass cannot mean "nothing was ever leased".
            assert viewer.is_cold is False
            assert servers["target"].attach_clients() == 1

            # The loss itself, with no other client on the runtime: exactly the
            # shape a send timeout or an owner blip produces.
            client = viewer._client
            assert client is not None
            client.close()
            for _ in range(40):
                await asyncio.sleep(0.05)
                if viewer.is_cold:
                    break
            assert viewer.is_cold, "the socket loss never landed"

            # THE USER ACTION: the click, which the cache routes back to the
            # very same source the prewarm branch built. It lands WHILE the
            # recovery loop is still running (asserted below), so this is also
            # the double-engage hazard pinned at the real level: a caller's bind
            # and the loop's own release must not leave two attach clients on
            # one facade, which the `attach_clients() == 1` after the heal
            # measures. ``_ensure_bound`` returns at its `_recovering` guard
            # while the loop holds the dial, and the cold arm returns from the
            # loop before the facade is bindable.
            recoveries_before = app._sidebar_gate_recoveries
            assert viewer._recovering is True, (
                "the click did not land during recovery; the double-engage "
                "property this case also covers is no longer exercised"
            )
            await asyncio.wait_for(app._sidebar_navigation.select("target"), 30)
            assert app._sidebar_sources["target"] is source
            seen = await _settle(app, pilot, source)

            _assert_healed(app, source, seen)
            assert servers["target"].attach_clients() == 1, "the heal did not dial"
            assert app._sidebar_gate_recoveries - recoveries_before <= _GATE_RECOVERY_CEILING


@pytest.mark.asyncio
async def test_a_stopped_prewarm_source_heals_once_its_conversation_is_back(
    headless_tui_env: Path,
) -> None:
    """THE SHAPE THE OPERATOR'S TWO LATCHES LOGGED: ``recovering=False``.

    Both production latches carry ``recovering=False``, i.e. no recovery loop
    was running, so no bound on that loop could have reached them: a stop leaves
    a facade cold WITHOUT ever setting ``_recovering``. The stop here is a real
    one — a peer viewer asks the owner to stop, the owner announces ``stopping``
    to everyone attached and exits — and the conversation then comes back as a
    new runtime for the same id, which is the production state: a live, dialable
    record and a stale row whose facade could not dial it.

    Measured on the pre-fix tree at this base: 8 rounds, every one of them
    ``guard='G1 can_go_cold/disposed'``, ``DIALS: 0``, and the band reading
    ``Saved · Reconnect failed · Select again to retry`` while the owner was
    alive and dialable.
    """
    config = headless_tui_env
    servers = {"origin": await _runtime(config, "origin")}
    # The real stop rung: the socket `stop` op ends this runtime, unpublishing
    # its record and closing every viewer's socket just after the `stopping`
    # announcement, exactly as `lop stop` does.
    servers["target"] = await _runtime(config, "target", stoppable=True)

    # ``**_probe`` absorbs the keyword the engage loop's DISCOVERY path passes
    # (`find_runtime_record(..., check_zombie=False)`): a rig that accepts only
    # the two positional arguments turns that call into a `TypeError`, which the
    # connect reports as its `connection_error`. The sibling file's rigs carry
    # the same parameter for the same reason.
    def find_owner(_config_dir, session_id, **_probe):
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
            source = await app._lease_sidebar_source("target", speculative=True)
            viewer = source.session
            assert isinstance(viewer, AttachedSession)
            assert viewer.is_cold is False

            peer = await AttachedSession.connect(
                servers["target"]._record,
                "target",
                config_dir=config,
                takeover_factory=_never_take_over,
                display_window=True,
            )
            try:
                await peer.request_stop()
                for _ in range(200):
                    await asyncio.sleep(0.05)
                    if viewer.is_cold:
                        break
                # PRECONDITIONS, and both are the production shape: cold, with
                # no recovery loop to bound and (below) an owner that is alive.
                assert viewer.is_cold, "the stop never landed"
                assert viewer._recovering is False, (
                    "a recovery loop was running; this case exists for the shape "
                    "production logged, which has none"
                )

                # The conversation comes back: a successor runtime publishes a
                # record for the same id, exactly as a resume does.
                servers["target"] = await _runtime(config, "target")
                assert servers["target"]._record is not None

                recoveries_before = app._sidebar_gate_recoveries
                await asyncio.wait_for(app._sidebar_navigation.select("target"), 30)
                # THE PRECONDITION FOR THE WHOLE CASE: the click reused the
                # facade the PREWARM branch built, rather than minting its own.
                assert app._sidebar_sources["target"] is source
                seen = await _settle(app, pilot, source)

                _assert_healed(app, source, seen)
                assert servers["target"].attach_clients() == 1, "the heal did not dial"
                assert app._sidebar_gate_recoveries - recoveries_before <= _GATE_RECOVERY_CEILING
            finally:
                await peer.dispose()
