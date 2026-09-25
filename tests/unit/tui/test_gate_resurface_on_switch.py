"""A live gate must come back when the user switches back to its session.

THE BUG. Session A raises an ``ask`` question or a tool-permission request.
The user switches to session B and back to A. The card is gone, and the turn
behind it is still blocked on an answer that now has no surface to arrive
through.

WHY A TEST AT THIS SEAM AND NOWHERE ELSE. The prompt card is not part of the
replayed transcript: a switch rebuilds the transcript from settled history
rows, and a live gate card is not a row. Its only route back on screen is a
re-armed gate bridge --
``_commit_sidebar_session`` -> ``AttachedSession.resume_viewer_gates`` ->
``_maybe_start_gate``. So the assertion has to be driven through the REAL
``_prepare_sidebar_session`` / ``_commit_sidebar_session`` pair (the harness
shape is borrowed from ``test_parked_source_seam``), with a REAL gate raised
through the owner handle (the shape is borrowed from
``test_reload_gate_delivery_e2e``). A unit test on either half alone cannot
see the drop, because each half is individually correct.

WHAT THE INSTRUMENTATION IS FOR. ``_maybe_start_gate`` is a five-guard ladder
in which EVERY early return is a silent drop -- no card, no notice, no log.
Knowing that the card did not come back is worthless on its own; the finding
is WHICH guard returned. ``_GateLadderProbe`` below replays the ladder's
conditions read-only just before each real call and records the first one that
trips, so a failure prints the named guard rather than leaving the next reader
to bisect it.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import AskOption, AskQuestion, Message, TextContent
from local_operator.session.attached import AttachedSession, _pending_request
from local_operator.session.protocol import SessionProtocol
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import EventController
from local_operator.tui.session_interaction import SessionInteraction
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over


def _rows(count: int = 3) -> list[Message]:
    return [
        Message(
            id=f"gate-row-{index:04}",
            role="assistant",
            content=[TextContent(text=f"Settled row {index:04}")],
            stop_reason="stop",
        )
        for index in range(count)
    ]


@asynccontextmanager
async def _remote(tmp_path: Path, name: str):
    """A real owner runtime plus the real ``AttachedSession`` viewer over it.

    Yields BOTH halves, unlike ``test_parked_source_seam._remote``: raising a
    genuine gate needs the owner handle, and answering it needs the viewer.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"gateswitch-{name}"
    await seed_transcript(directory, _rows())
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await AttachedSession.connect(
        server._record,
        directory.name,
        config_dir=config,
        takeover_factory=_never_take_over,
        display_window=True,
    )
    try:
        yield remote, handle
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


def _attached(session: SessionProtocol | None) -> AttachedSession:
    """The app's current session, as the concrete viewer the rig handed it.

    ``OperatorApp._session`` is typed ``SessionProtocol | None`` because the app
    accepts any session facade, but every case in these files drives a real
    ``AttachedSession`` over an in-process runtime and probes its gate internals
    (``_gates_detached``, ``_gate_task``) — which no protocol declares, and which
    ``_install_probe`` needs the concrete class for.

    So this is the narrowing AND a real premise check, not a cast: a rig that
    left the app holding anything else has not set up the scenario under test,
    and saying so here beats an ``AttributeError`` sixty lines later.
    """
    assert isinstance(
        session, AttachedSession
    ), f"expected the app to hold an AttachedSession, got {type(session).__name__}"
    return session


# --- instrumentation ---------------------------------------------------------


#: The ladder in ``AttachedSession._maybe_start_gate``, in evaluation order,
#: labelled with the same G1-G6 names as its debug lines. Each check answers
#: "does THIS guard return early?". Replayed read-only just before each real
#: call, so the readout names a guard rather than a theory about one. It
#: carries no line numbers: they move with every edit to the file.
def _first_guard_that_drops(session: AttachedSession) -> str:
    if session._disposed or not session._ready_for_events:
        return "G1 disposed-or-not-ready"
    pending = _pending_request(session.pending_gate)
    if pending is None:
        return "G2a no-pending-gate"
    if session._gate_task is not None:
        return "G2b gate-task-already-set"
    if session._gate_identity(pending) == session._gate_answered_key:
        return "G3 identity-equals-answered-key"
    background = (
        session._gates_detached and session._background_approval and pending.kind == "approval"
    )
    if session._gates_detached and not background:
        return "G4 gates-detached"
    if not session.can_ever_bind:
        return "G6 viewer-can-never-bind"
    if pending.kind == "approval" and (session._approval_handler is not None or background):
        return ""
    if pending.kind == "ask" and session._ask_handler is not None:
        return ""
    return "G5 handler-is-None"


@dataclass
class _GateLadderProbe:
    """Records every ``_maybe_start_gate`` call on one session, with its verdict."""

    session: AttachedSession
    calls: list[dict[str, Any]] = field(default_factory=list)

    def readout(self) -> dict[str, Any]:
        """The state the ladder reads, sampled NOW rather than at call time."""
        pending = _pending_request(self.session.pending_gate)
        task = self.session._gate_task
        return {
            "_gates_detached": self.session._gates_detached,
            "_gate_task": None if task is None else f"<task done={task.done()}>",
            "_gate_answered_key": self.session._gate_answered_key,
            "_ready_for_events": self.session._ready_for_events,
            "_disposed": self.session._disposed,
            "_background_approval": self.session._background_approval,
            "_keep_gate_reply": self.session._keep_gate_reply,
            "pending_gate": None if pending is None else (pending.kind, pending.request_id),
            "_ask_handler_is_None": self.session._ask_handler is None,
            "_approval_handler_is_None": self.session._approval_handler is None,
            "guard_that_would_drop": _first_guard_that_drops(self.session) or "(none: starts)",
        }

    def since(self, marker: int) -> list[dict[str, Any]]:
        return self.calls[marker:]

    @property
    def marker(self) -> int:
        return len(self.calls)


def _install_probe(monkeypatch: pytest.MonkeyPatch, session: AttachedSession) -> _GateLadderProbe:
    probe = _GateLadderProbe(session)
    original = AttachedSession._maybe_start_gate

    def wrapper(self: AttachedSession, pending: Any = None) -> None:
        if self is not session:
            return original(self, pending)
        before = probe.readout()
        original(self, pending)
        after_task = self._gate_task
        probe.calls.append(
            {
                **before,
                "started_a_bridge": after_task is not None,
            }
        )

    monkeypatch.setattr(AttachedSession, "_maybe_start_gate", wrapper)
    return probe


def _format(calls: list[dict[str, Any]], title: str) -> str:
    if not calls:
        return f"\n{title}: _maybe_start_gate was NEVER CALLED on this leg."
    lines = [f"\n{title}: {len(calls)} call(s) to _maybe_start_gate"]
    for index, call in enumerate(calls, 1):
        lines.append(f"  call {index}: guard={call['guard_that_would_drop']}")
        for key in (
            "_gates_detached",
            "_gate_task",
            "_gate_answered_key",
            "_ready_for_events",
            "_background_approval",
            "_keep_gate_reply",
            "pending_gate",
            "_ask_handler_is_None",
            "_approval_handler_is_None",
            "started_a_bridge",
        ):
            lines.append(f"      {key} = {call[key]!r}")
    return "\n".join(lines)


# --- the test ----------------------------------------------------------------


async def _pump(pilot, count: int = 12) -> None:
    for _ in range(count):
        await pilot.pause()


async def _pump_until(pilot, predicate, tries: int = 400) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await pilot.pause()
    return predicate()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval", "ask"])
async def test_a_live_gate_resurfaces_when_the_user_switches_back(
    tmp_path, monkeypatch, kind: str
) -> None:
    """Switch away from a session holding a live gate, come back, see the card.

    The turn behind the gate is still blocked either way; the question this
    asks is whether the user has any surface left to answer it through.
    """
    async with (
        _remote(tmp_path, "alpha") as (alpha, alpha_handle),
        _remote(tmp_path, "beta") as (beta, _beta_handle),
        _remote(tmp_path, "gamma") as (gamma, _gamma_handle),
    ):
        # Boot ON alpha: `_commit_sidebar_session` refuses to switch away from
        # anything that is not a real viewer, so alpha has to be the app's own
        # session for the switch-away leg to reach the code under test.
        async def factory():
            return alpha

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            booted = await _pump_until(pilot, lambda: app._session is alpha, tries=200)
            assert booted, "the app never adopted the alpha viewer"

            sources: dict[str, SessionInteraction] = {alpha.session_id: app._interaction}
            app._sidebar_sources[alpha.session_id] = app._interaction
            for remote in (beta, gamma):
                source = SessionInteraction(remote)
                sources[remote.session_id] = source
                app._sidebar_sources[remote.session_id] = source

            async def lease(session_id, *, speculative=False):
                # Reproduces `_lease_sidebar_source`'s park-at-birth wiring
                # rather than skipping it; only external discovery is
                # redirected. Same override as test_parked_source_seam.
                source = sources[session_id]
                source.preparations += 1
                if source.controller is None:
                    app._interactions[id(source.session)] = source
                    source.controller = EventController(source.session, app)
                    app._event_sources[source.controller] = source
                    source.controller.set_parked(True)
                    source.controller.subscribe()
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]

            async def visit(session_id: str) -> None:
                prepared = await app._prepare_sidebar_session(session_id)
                ready = app._commit_sidebar_session(
                    session_id, prepared, app._sidebar_navigation.generation
                )
                await _pump(pilot, 16)
                if ready is not None and not ready.done():
                    ready.cancel()

            # --- raise a REAL gate on alpha, on screen ----------------------
            # Without this an approval auto-answers and never mounts a card.
            app._set_approve_all(False)
            alpha_handle._auto_approve = False

            probe = _install_probe(monkeypatch, alpha)

            if kind == "approval":
                gate_task = asyncio.create_task(
                    alpha_handle._approval_gate("write", "Save one record")
                )
                mounted = await _pump_until(pilot, lambda: app._approval is not None)
            else:
                gate_task = asyncio.create_task(
                    alpha_handle._ask_gate(
                        [
                            AskQuestion(
                                id="destination",
                                question="Choose a destination",
                                options=[AskOption(label="Here"), AskOption(label="There")],
                            )
                        ]
                    )
                )
                mounted = await _pump_until(pilot, lambda: app._ask_screen is not None)

            def card_is_mounted() -> bool:
                return (app._approval if kind == "approval" else app._ask_screen) is not None

            try:
                assert mounted, (
                    f"the {kind} card never mounted on the session that raised it; "
                    "the reproduction never reached its premise"
                )
                assert alpha.pending_gate is not None
                assert not gate_task.done(), "the gate answered itself before the switch"

                # --- leg 1: switch AWAY to beta -------------------------------
                away_marker = probe.marker
                await visit(beta.session_id)
                away_calls = probe.since(away_marker)

                assert not card_is_mounted(), (
                    "the outgoing session's card is still on screen after the "
                    "switch; the premise of this test no longer holds"
                )
                assert not gate_task.done(), (
                    "switching away ANSWERED the gate: navigation invented an "
                    "answer the user never gave"
                )

                # --- leg 2: switch BACK to alpha ------------------------------
                back_marker = probe.marker
                await visit(alpha.session_id)
                # The bridge is re-armed asynchronously; give it the same turn
                # budget the mount above got before concluding it never came.
                resurfaced = await _pump_until(pilot, card_is_mounted)
                back_calls = probe.since(back_marker)

                alpha_source = sources[alpha.session_id]
                diagnosis = (
                    _format(away_calls, "LEG 1 (alpha -> beta, switching AWAY)")
                    + _format(back_calls, "LEG 2 (beta -> alpha, switching BACK)")
                    + "\n\nSTATE ON THE RETURN LEG, sampled after the commit settled:"
                    + "".join(f"\n  {key} = {value!r}" for key, value in probe.readout().items())
                    + f"\n  source.display_only = {alpha_source.display_only!r}"
                    + f"\n  source.retired = {alpha_source.retired!r}"
                    + f"\n  session.is_cold = {alpha.is_cold!r}"
                    + f"\n  session.display_history_current = {alpha.display_history_current!r}"
                    + f"\n  app._session is alpha = {app._session is alpha!r}"
                    + f"\n  gate_task.done() = {gate_task.done()!r}"
                    + f"\n  app._approval = {app._approval!r}"
                    + f"\n  app._ask_screen = {app._ask_screen!r}"
                )

                if os.environ.get("GATE_PROBE_DUMP"):
                    print(diagnosis)

                # The turn is STILL BLOCKED. That is what makes a missing card a
                # stuck session rather than a cosmetic loss, so it is asserted
                # before the card itself: if this ever fails the bug is a
                # different, worse one.
                assert not gate_task.done(), (
                    "the gate resolved across the round trip without the user "
                    "answering it" + diagnosis
                )

                assert resurfaced, (
                    f"REPRODUCED: the live {kind} card did not come back when the "
                    "user returned to the session holding it. The turn is still "
                    "blocked and there is no surface to answer it through." + diagnosis
                )

                # Answerable, not merely present: a disabled card is the same
                # dead end with a different appearance.
                card = app._approval if kind == "approval" else app._ask_screen
                assert card is not None
                assert not card.disabled, (
                    "the card came back DISABLED -- visible, unanswerable, and "
                    "the turn stays blocked" + diagnosis
                )
            finally:
                gate_task.cancel()
                await asyncio.gather(gate_task, return_exceptions=True)


# --- ANGLE A: the display_only latch ------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval", "ask"])
async def test_a_display_only_commit_rearms_its_gate_bridge(
    tmp_path, monkeypatch, kind: str
) -> None:
    """A ``display_only`` commit re-arms the gate bridge, and the card follows.

    THE NAMED DEFENDER of the commit site. ``_commit_sidebar_session`` guards
    ``_submit_boot_prompt`` with ``not source.display_only``, because submitting
    starts a turn and a saved preview must never do that. ``resume_viewer_gates``
    sits outside that guard, because it submits nothing: it clears the
    ``_gates_detached`` latch and re-runs the ``_maybe_start_gate`` ladder. The
    flag is also sticky. ``_prepare_sidebar_session`` ORs into it
    (``source.display_only or session.is_cold or not
    session.display_history_current``) and only the connect/bind path clears
    it. So a source that goes ``display_only`` once, for a reason as transient
    as a display resync in flight at the instant of preparation, still has to
    get its bridge back on this and every later return.

    ISOLATED, so the commit site is the only thing that can pass. Three other
    routes also clear the latch, and each would satisfy an outcome assertion:
    the level-triggered reconcile, the settled-navigation arm of
    ``_sidebar_navigation_pending``, and the reconnect's ``refresh=True``
    re-commit (which clears ``display_only`` first). All three are suppressed
    for the ``display_only`` return leg, and the assertion made right after that
    commit is on the LATCH rather than on a card.

    HOW THE FLAG IS SET, and why this is honest. The third disjunct is driven
    for real: ``session.display_history_current`` is ``not
    self._display_invalidated``, and ``_display_invalidated`` is set by
    ``_invalidate_display_history`` on a history-generation move. This test moves
    it through the OWNER's transcript with a real ``append_compaction``, which
    bumps ``_history_generation`` exactly as a production compaction does. It
    then holds the viewer's refresh off the loop so preparation observes the
    invalidated state, which is the race the user hits on a busy session.
    Nothing assigns ``display_only`` by hand.

    The rest of the test is the user-visible outcome: once the owner heals, the
    card comes back where the user is, and again after an ordinary round trip.
    """
    async with (
        _remote(tmp_path, "adelta") as (alpha, alpha_handle),
        _remote(tmp_path, "bdelta") as (beta, _beta_handle),
    ):

        async def factory():
            return alpha

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            booted = await _pump_until(pilot, lambda: app._session is alpha, tries=200)
            assert booted, "the app never adopted the alpha viewer"

            sources: dict[str, SessionInteraction] = {alpha.session_id: app._interaction}
            app._sidebar_sources[alpha.session_id] = app._interaction
            beta_source = SessionInteraction(beta)
            sources[beta.session_id] = beta_source
            app._sidebar_sources[beta.session_id] = beta_source

            async def lease(session_id, *, speculative=False):
                source = sources[session_id]
                source.preparations += 1
                if source.controller is None:
                    app._interactions[id(source.session)] = source
                    source.controller = EventController(source.session, app)
                    app._event_sources[source.controller] = source
                    source.controller.set_parked(True)
                    source.controller.subscribe()
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]

            async def visit(session_id: str) -> None:
                prepared = await app._prepare_sidebar_session(session_id)
                ready = app._commit_sidebar_session(
                    session_id, prepared, app._sidebar_navigation.generation
                )
                await _pump(pilot, 16)
                if ready is not None and not ready.done():
                    ready.cancel()

            app._set_approve_all(False)
            alpha_handle._auto_approve = False
            probe = _install_probe(monkeypatch, alpha)

            if kind == "approval":
                gate_task = asyncio.create_task(
                    alpha_handle._approval_gate("write", "Save one record")
                )
                mounted = await _pump_until(pilot, lambda: app._approval is not None)
            else:
                gate_task = asyncio.create_task(
                    alpha_handle._ask_gate(
                        [
                            AskQuestion(
                                id="destination",
                                question="Choose a destination",
                                options=[AskOption(label="Here"), AskOption(label="There")],
                            )
                        ]
                    )
                )
                mounted = await _pump_until(pilot, lambda: app._ask_screen is not None)

            def card_is_mounted() -> bool:
                return (app._approval if kind == "approval" else app._ask_screen) is not None

            alpha_source = sources[alpha.session_id]
            # Bound BEFORE the `try` because the `finally` releases it: created at
            # its point of use below, it is unbound on any early failure and the
            # cleanup then depends on catching `NameError` to notice.
            release_sync = asyncio.Event()
            try:
                assert mounted, f"the {kind} card never mounted; the premise is unmet"
                assert not alpha_source.display_only, (
                    "alpha was ALREADY display_only before the scenario started; "
                    "this test would prove nothing"
                )

                # --- switch AWAY, which detaches the gates --------------------
                await visit(beta.session_id)
                assert not card_is_mounted()
                assert alpha._gates_detached, "the switch away did not detach alpha's gates"

                # --- drive display_history_current False FOR REAL -------------
                # A real compaction marker on the OWNER's transcript bumps
                # `_history_generation` (transcript.py:1593-1594). The viewer
                # sees the generation move on the next canonical delta and calls
                # `_invalidate_display_history`, which is what makes
                # `display_history_current` False.
                # The healing refresh must still be IN FLIGHT when preparation
                # reads the flag -- that ordering IS the race. It is held open
                # at the honest place: the owner round-trip the refresh is
                # waiting on. `_refresh_display_history` awaits
                # `client.frontend_sync()` before it can clear
                # `_display_invalidated`, so a sync that has not answered yet is
                # exactly a production-slow owner. Nothing about the viewer's
                # own logic is patched.
                client = alpha._client
                assert client is not None
                sync_entered = asyncio.Event()
                real_sync = client.frontend_sync

                async def held_sync(*args, **kwargs):
                    sync_entered.set()
                    await release_sync.wait()
                    return await real_sync(*args, **kwargs)

                monkeypatch.setattr(client, "frontend_sync", held_sync)

                await alpha_handle._session._transcript.append_compaction(
                    summary="compacted mid-navigation",
                    first_kept_entry_id="gate-row-0002",
                    tokens_before=1234,
                )
                alpha._invalidate_display_history()
                invalidated = await _pump_until(
                    pilot, lambda: not alpha.display_history_current, tries=40
                )

                # --- return to alpha while the display is invalidated ---------
                # Every OTHER route that clears `_gates_detached` is suppressed
                # for this leg (see the docstring), so the latch assertion below
                # can only be satisfied by the commit site's own call.
                assert alpha._gates_detached, "the premise needs a detached bridge"
                back_marker = probe.marker
                with (
                    patch.object(
                        OperatorApp, "_reconcile_gate_surface", lambda self, candidate: None
                    ),
                    patch.object(
                        OperatorApp, "_sidebar_navigation_pending", lambda self, session_id: None
                    ),
                    patch.object(
                        OperatorApp,
                        "_start_sidebar_connection",
                        lambda self, candidate, **_kwargs: None,
                    ),
                ):
                    await visit(alpha.session_id)
                    rearmed_by_commit = not alpha._gates_detached
                    resurfaced = await _pump_until(pilot, card_is_mounted, tries=120)
                back_calls = probe.since(back_marker)

                diagnosis = (
                    _format(back_calls, "RETURN LEG with display_history_current False")
                    + "\n\nSTATE ON THE RETURN LEG:"
                    + "".join(f"\n  {key} = {value!r}" for key, value in probe.readout().items())
                    + f"\n  source.display_only = {alpha_source.display_only!r}"
                    + f"\n  session.display_history_current = {alpha.display_history_current!r}"
                    + f"\n  session.is_cold = {alpha.is_cold!r}"
                    + f"\n  invalidation_observed = {invalidated!r}"
                    + f"\n  app._session is alpha = {app._session is alpha!r}"
                    + f"\n  gate_task.done() = {gate_task.done()!r}"
                )
                if os.environ.get("GATE_PROBE_DUMP"):
                    print(diagnosis)

                assert alpha_source.display_only, (
                    "the return leg did not actually latch display_only, so this "
                    "test did not exercise the guard it exists for" + diagnosis
                )
                assert rearmed_by_commit, (
                    "REGRESSION: the display_only commit did NOT clear "
                    "_gates_detached. Every other route was suppressed for this leg, "
                    "so the commit site's resume_viewer_gates() is the call that is "
                    "missing; with it gated on `not source.display_only` again, a "
                    "source that latched display_only once loses its gate card." + diagnosis
                )
                assert not gate_task.done(), (
                    "the gate resolved without the user answering it" + diagnosis
                )
                assert not resurfaced, (
                    "the card came back on the invalidated leg; the premise of "
                    "the stickiness argument below no longer holds" + diagnosis
                )

                # --- HEAL EVERYTHING, then take a CLEAN round trip ------------
                # The leg above is contaminated: the held sync also leaves
                # `_ready_for_events` False, so G1 fires and G4 is not yet the
                # proven cause. Releasing the sync removes that confound
                # entirely -- the session becomes fully live and current again,
                # a state in which the CONTROL test above passes. What does NOT
                # heal is `source.display_only`: nothing on the sidebar path
                # clears it (the sole clear is app.py:7973, in the connect/bind
                # path). So a further, entirely ordinary switch away and back --
                # no invalidation, no cold owner, nothing in flight -- isolates
                # the latch as the single remaining difference from the control.
                release_sync.set()
                healed = await _pump_until(
                    pilot,
                    lambda: alpha._ready_for_events and alpha.display_history_current,
                    tries=200,
                )

                # FIRST, the scenario the user actually reports: they are back on
                # session A and they STAY there. A `display_only` commit arms
                # `_start_sidebar_connection` (app.py:7638-7648), which re-prepares
                # with `refresh=True`, clears `display_only` at app.py:7973 and
                # re-commits -- so the card may come back with no further input.
                # Whether it does is the difference between "stuck forever" and
                # "stuck until the reconnect lands", and it is measured, not
                # assumed.
                in_place_marker = probe.marker
                healed_in_place = await _pump_until(pilot, card_is_mounted, tries=300)
                in_place_calls = probe.since(in_place_marker)

                clean_marker = probe.marker
                await visit(beta.session_id)
                await visit(alpha.session_id)
                clean_resurfaced = await _pump_until(pilot, card_is_mounted, tries=120)
                clean_calls = probe.since(clean_marker)

                clean_diagnosis = (
                    diagnosis
                    + _format(
                        in_place_calls,
                        "\nSTAYING PUT on alpha after the owner healed "
                        f"(healed_in_place={healed_in_place})",
                    )
                    + _format(
                        clean_calls,
                        "\nCLEAN ROUND TRIP afterwards (healed session, no invalidation)",
                    )
                    + "\n\nSTATE AFTER THE CLEAN ROUND TRIP:"
                    + "".join(f"\n  {key} = {value!r}" for key, value in probe.readout().items())
                    + f"\n  source.display_only = {alpha_source.display_only!r}"
                    + f"\n  session.display_history_current = "
                    f"{alpha.display_history_current!r}"
                    + f"\n  session.is_cold = {alpha.is_cold!r}"
                    + f"\n  healed = {healed!r}"
                    + f"\n  gate_task.done() = {gate_task.done()!r}"
                )
                if os.environ.get("GATE_PROBE_DUMP"):
                    print(clean_diagnosis)

                assert healed, (
                    "the session never became live and current again, so the "
                    "clean round trip below is not clean" + clean_diagnosis
                )
                assert not gate_task.done(), (
                    "the gate resolved without the user answering it" + clean_diagnosis
                )
                assert healed_in_place, (
                    f"REPRODUCED (H1): the user returns to the session holding the "
                    f"live {kind} gate and the card is GONE. The owner is healthy "
                    "again and the turn is still blocked, but staying on the "
                    "session does not bring the prompt back: the commit saw "
                    "source.display_only and skipped resume_viewer_gates() "
                    "(app.py:7537), so _gates_detached stayed True and guard "
                    "G4 dropped every re-arm attempt." + clean_diagnosis
                )
                assert clean_resurfaced, (
                    f"REPRODUCED (H1), display_only is a PERMANENT latch: the "
                    f"session is fully live and current again ({kind} gate still "
                    "unanswered), and an ordinary switch away-and-back -- the exact "
                    "round trip the control test passes -- still does not bring the "
                    "card back. source.display_only stayed True, so the commit at "
                    "app.py:7537 skipped resume_viewer_gates(), _gates_detached "
                    "stayed True, and guard G4 dropped the gate again. The "
                    "turn is blocked with no surface to answer it, permanently." + clean_diagnosis
                )
            finally:
                release_sync.set()
                gate_task.cancel()
                await asyncio.gather(gate_task, return_exceptions=True)


# --- ANGLE B: can the CURRENT session be a local owner? -----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval", "ask"])
async def test_a_local_owner_session_refuses_the_switch_before_unmounting_its_card(
    tmp_path, monkeypatch, kind: str
) -> None:
    """The unmount is unconditional, but it is UNREACHABLE for a local owner.

    THE CONCERN. ``_commit_sidebar_session`` suspends the outgoing gates only
    ``if _is_viewer(previous)`` (app.py:7382-7383), while the block that
    unmounts the cards and clears ``_ask_screen``/``_approval``
    (app.py:7386-7393) has no such condition. A local owner session has no
    ``pending_gate`` projection and no ``_run_ask``/``_run_approval`` bridge at
    all -- its prompt is only the mounted widget plus the coroutine blocked in
    ``_request_user_choice_on_app_loop``. If that unmount could run for a local
    owner, the card would be destroyed with nothing anywhere able to rebuild
    it: strictly worse than the viewer case, where a re-arm at least exists.

    THE ANSWER IS NO, ON TWO INDEPENDENT GROUNDS, and this test pins the second.

    (1) ``lop``'s TUI never builds a local owner in the first place. The
    factory at cli.py:8351-8371 returns an ``AttachedSession`` on every path --
    ``AttachedSession.connect`` when a live runtime record exists and
    ``AttachedSession.cold`` otherwise -- and ``AttachedSession.owns_runtime``
    is hard-coded ``False`` (attached.py:6901-6908). The takeover factory
    cli.py:8280-8301 raises rather than converting a viewer into an owner, and
    says so: "THE OWNER PATH IS GONE FROM `lop`". ``Session.owns_runtime`` is
    ``True`` (session.py:3333-3335) but only ``lop exec``, the headless REPL,
    the server and the mobile daemon build one, and none of those has a sidebar.

    (2) Even if one existed, the ORDER protects it. The
    ``not _is_viewer(previous)`` check raises at app.py:7328-7329, which is
    fifty-odd lines ABOVE the unmount at 7386. ``SessionNavigation.
    _prepare_and_commit`` catches that ``RuntimeError`` (session_navigation.py:
    176-178) and abandons the switch, so the conversation stays on screen with
    its card intact. Ordering is what makes this safe, and ordering is exactly
    the kind of property a later edit can break silently -- moving the unmount
    above the guard would introduce the severe bug with no test objecting. This
    test fails if that happens.

    So the local-owner case is NOT a second instance of the bug. It is a case
    the guard refuses, and what is asserted here is the refusal plus the
    survival of the card, not a hypothesis about it.
    """
    async with (
        _remote(tmp_path, "owner") as (alpha, alpha_handle),
        _remote(tmp_path, "bowner") as (beta, _beta_handle),
    ):

        async def factory():
            return alpha

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            booted = await _pump_until(pilot, lambda: app._session is alpha, tries=200)
            assert booted, "the app never adopted the alpha viewer"

            sources: dict[str, SessionInteraction] = {alpha.session_id: app._interaction}
            app._sidebar_sources[alpha.session_id] = app._interaction
            beta_source = SessionInteraction(beta)
            sources[beta.session_id] = beta_source
            app._sidebar_sources[beta.session_id] = beta_source

            async def lease(session_id, *, speculative=False):
                source = sources[session_id]
                source.preparations += 1
                if source.controller is None:
                    app._interactions[id(source.session)] = source
                    source.controller = EventController(source.session, app)
                    app._event_sources[source.controller] = source
                    source.controller.set_parked(True)
                    source.controller.subscribe()
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]

            app._set_approve_all(False)
            alpha_handle._auto_approve = False

            if kind == "approval":
                gate_task = asyncio.create_task(
                    alpha_handle._approval_gate("write", "Save one record")
                )
                mounted = await _pump_until(pilot, lambda: app._approval is not None)
            else:
                gate_task = asyncio.create_task(
                    alpha_handle._ask_gate(
                        [
                            AskQuestion(
                                id="destination",
                                question="Choose a destination",
                                options=[AskOption(label="Here"), AskOption(label="There")],
                            )
                        ]
                    )
                )
                mounted = await _pump_until(pilot, lambda: app._ask_screen is not None)

            def card_is_mounted() -> bool:
                return (app._approval if kind == "approval" else app._ask_screen) is not None

            try:
                assert mounted, f"the {kind} card never mounted; the premise is unmet"

                # The REAL `lop` factory never yields one of these, so the local
                # owner is simulated at the ONE predicate the commit path reads:
                # `_is_viewer`, which is `not session.owns_runtime`
                # (app.py:41951). Patching the predicate rather than swapping in
                # a `Session` keeps every other part of the seam genuine, and it
                # is the exact question app.py:7328 asks.
                real_is_viewer = app_module._is_viewer

                def owner_shaped(session: Any) -> bool:
                    # Only ALPHA -- the outgoing/current session -- reads as a
                    # local owner. Beta must stay a viewer or the commit refuses
                    # at app.py:7315 for the incoming session instead, which is
                    # a different guard and would not test this one.
                    if session is alpha:
                        return False
                    return real_is_viewer(session)

                monkeypatch.setattr(app_module, "_is_viewer", owner_shaped)

                prepared = await app._prepare_sidebar_session(beta.session_id)
                with pytest.raises(RuntimeError) as raised:
                    app._commit_sidebar_session(
                        beta.session_id, prepared, app._sidebar_navigation.generation
                    )

                assert "runtime-backed current session" in str(raised.value), (
                    "the commit refused for some OTHER reason than the local-owner "
                    f"guard at app.py:7328: {raised.value!r}"
                )

                await _pump(pilot, 12)
                assert card_is_mounted(), (
                    "SEVERE: the refused switch still unmounted the local owner's "
                    f"{kind} card. The unmount at app.py:7386-7393 is "
                    "unconditional, and it must stay BELOW the guard at "
                    "app.py:7328 -- a local owner has no gate bridge, so a card "
                    "destroyed here can never be rebuilt by anything."
                )
                assert not gate_task.done(), "the refused switch answered the gate"
            finally:
                monkeypatch.undo()
                gate_task.cancel()
                await asyncio.gather(gate_task, return_exceptions=True)
