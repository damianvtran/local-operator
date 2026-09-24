"""The guards the gate-resurface FIX itself needs, once a reconcile exists.

WHY A THIRD FILE. Its two siblings pin that a pending gate COMES BACK. The fix
that makes them pass adds a level-triggered reconcile
(``OperatorApp._reconcile_gate_surface``, called from
``_source_frontend_changed``) which re-arms the bridge on STATE rather than on a
navigation edge. That inverts the risk: the interesting failures are no longer
"the card never came back" but "a card came back that must NOT have", and they
need their own assertions rather than a footnote in a resurface test.

Two of those are load-bearing enough that the fix is wrong without them:

* **G3, the answered-gate fence.** ``_maybe_start_gate``'s third guard
  (attached.py) refuses a pending gate whose identity equals
  ``_gate_answered_key``. The reconcile raises how often that guard is
  consulted by orders of magnitude — it runs on every frontend delta of the
  visible source instead of once per navigation — so a fence that was
  previously asked once per switch is now asked continuously. If it does not
  hold, the user answers a question and the question comes straight back, which
  is a worse bug than the one being fixed.
* **The cross-session rule.** A late gate from session A must never mount while
  B is displayed (``tests/unit/tui/test_sidebar_source_state.py`` pins the
  original form). The reconcile is a NEW route to a mount, so it has to obey
  the same rule; ``_is_current`` is its first check precisely for this.

The third covers EDIT 1 of the fix directly: ``_sidebar_gate_surface_ready``
used to answer "ready" for a ``display_only`` frame only when NO card was
mounted, which is what sent a correctly-painted frame into the 15 s
``SurfaceNotReady`` timer. Its replacement shares one predicate with the live
branch (``_sidebar_gate_card_ready``), and the property worth pinning is the
NARROWNESS of the relaxation: a correctly-BOUND card is ready, an UNBOUND one
is still refused. A relaxation that accepted any mounted card would re-introduce
the identity confusion the whole bug is made of.

The rig, the probe and the click driver are the real-entry file's, imported so
there is one harness rather than three that can drift apart.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.session_sidebar import SessionSidebar
from tests.unit.tui.test_gate_resurface_on_switch import (
    _attached,
    _install_probe,
    _pump,
    _pump_until,
)
from tests.unit.tui.test_gate_resurface_real_entry import (
    _RESURFACE_TURNS,
    _app,
    _click,
    _dump,
    _one_question,
    _raise_ask,
    _rig,
    _state,
)


@pytest.mark.asyncio
async def test_an_answered_gate_does_not_remount_under_the_reconcile(tmp_path, monkeypatch) -> None:
    """G3 must hold against the reconcile, or answering a question re-asks it.

    DRIVEN THROUGH THE REAL PATH, not by calling ``_maybe_start_gate``. The
    thing under test is the reconcile added to ``_source_frontend_changed``, and
    a direct call to the ladder would prove only that the ladder still has a G3
    — not that the new caller reaches it with the state that makes it fire. So
    the question is answered ON SCREEN with a real keypress, and then the
    frontend is driven to deliver deltas, which is what calls the reconcile.

    THE SINGLE-QUESTION ASK IS THE POINT. Case E in the sibling file uses a
    two-question ask and measures that Q2 correctly survives, because a fresh
    ``request_id`` per question means G3 cannot match it. Here the ask holds ONE
    question, so once it is answered there is no successor for the fence to be
    confused with: any card that appears afterwards is the answered question
    coming back, which is unambiguously the bug this pins.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted; the premise is unmet"

                    # ANSWER IT, for real, the way the user does.
                    await pilot.press("enter")
                    answered = await _pump_until(
                        pilot, lambda: gate_task.done(), tries=_RESURFACE_TURNS
                    )
                    assert answered, (
                        "the keypress never answered the gate, so there is no answered "
                        "gate to fence" + _state(app, alpha, source, probe)
                    )
                    await _pump(pilot, 30)
                    assert (
                        app._ask_screen is None
                    ), "the answered card is still on screen; it should have settled" + _state(
                        app, alpha, source, probe
                    )

                    # Now drive the reconcile HARD. Every one of these is a
                    # frontend delta on the visible source, which is exactly the
                    # event that calls `_reconcile_gate_surface`.
                    marker = probe.marker
                    for _ in range(25):
                        app._source_frontend_changed(source)
                        await _pump(pilot, 4)

                    remounted = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=60
                    )
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  ladder calls during reconcile = {len(probe.since(marker))}"
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not remounted, (
                        "REGRESSION (G3): a question the user ALREADY ANSWERED came "
                        "back on screen under the new reconcile. The answered-gate "
                        "fence (_gate_answered_key) did not hold against a caller "
                        "that consults it on every frontend delta, so the user is "
                        "re-asked something they have already decided." + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_the_reconcile_never_mounts_a_hidden_sessions_gate(tmp_path, monkeypatch) -> None:
    """Session A's gate must not appear while the user is looking at B.

    THE RULE IS OLDER THAN THE FIX (``test_sidebar_source_state.py`` pins its
    original form) and the reconcile is a new way to break it: it re-arms a
    bridge from a hook that fires for EVERY leased source, not only the visible
    one. ``_is_current`` is its first check, and this is what asserts that check
    is load-bearing rather than decorative.

    The reconcile is invoked DIRECTLY on the hidden source here, which is
    stronger than waiting for a delta to arrive on its own: it proves the guard
    refuses even when the call is made, rather than proving only that no call
    happened to occur during the test window.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha = _attached(app._session)
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted on alpha; the premise is unmet"

                    # Go to beta. Alpha keeps its unanswered question.
                    await _click(app, pilot, "beta")
                    assert (
                        getattr(app._session, "session_id", "") == "beta"
                    ), "the switch away never landed on beta"
                    assert app._ask_screen is None, "alpha's card is still on screen over beta"

                    alpha_source = app._sidebar_sources.get("alpha")
                    assert alpha_source is not None, "alpha's source was not retained"
                    assert (
                        alpha.pending_gate is not None
                    ), "alpha's gate left canonical state, so nothing could mount anyway"

                    # Drive the reconcile ON THE HIDDEN SOURCE, repeatedly.
                    for _ in range(25):
                        app._source_frontend_changed(alpha_source)
                        await _pump(pilot, 4)

                    leaked = await _pump_until(pilot, lambda: app._ask_screen is not None, tries=60)
                    diagnosis = (
                        _state(app, alpha, alpha_source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not leaked, (
                        "REGRESSION: the reconcile mounted session ALPHA's gate card "
                        "while the user was looking at BETA. A card that belongs to a "
                        "hidden conversation appeared over a visible one, and "
                        "answering it would answer a question the user cannot see the "
                        "context for." + diagnosis
                    )
                    assert not gate_task.done(), (
                        "alpha's gate was answered while it was off screen" + diagnosis
                    )
                    assert getattr(app._session, "session_id", "") == "beta", (
                        "the reconcile moved the user off beta" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_a_gate_arriving_with_no_navigation_coming_still_mounts(
    tmp_path, monkeypatch
) -> None:
    """THE NAMED DEFENDER of the level-triggered reconcile (EDIT 3).

    WHAT THIS EDIT UNIQUELY BUYS, and nothing else in the suite covers it.
    Both re-arms of the gate bridge are EDGES: they fire from a navigation.
    ``_gates_detached`` therefore heals only if another navigation happens to
    come along. A gate that arrives at a CURRENT source whose bridge is already
    detached, with no navigation in flight and none coming, is dropped at G4 and
    has no further edge to wait for — the turn stays blocked with nothing on
    screen. The reconcile is level-triggered precisely so that state, rather
    than a transition, is what brings the card back.

    MEASURED, NOT ASSUMED: removing the ``_reconcile_gate_surface`` call from
    ``_source_frontend_changed`` leaves every other test in all three gate files
    green (19 passed), because edit 2 heals each of their scenarios through a
    navigation. This is the one assertion that goes red, which is what makes it
    the edit's defender rather than a restatement of its siblings.

    THE STATE IS BUILT WITH THE REAL SUSPEND, not by assigning the latch: the
    detached bridge comes from ``_suspend_sidebar_gates``, the same call a
    departure makes, and the gate is a genuine unanswered ask on the owner. No
    navigation is dispatched at any point after it, so an edge-triggered re-arm
    has nothing to fire on.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                # Detach the bridge on the source the user is LOOKING AT, with
                # the same call a departure makes, and with nothing pending yet.
                assert alpha.pending_gate is None
                app._suspend_sidebar_gates(source)
                assert alpha._gates_detached, "the suspend did not detach the gates"

                # The gate arrives now. `_apply_pending_gate` -> `_maybe_start_gate`
                # drops it at G4 because the bridge is detached; the question is
                # whether anything ever brings it back with no navigation coming.
                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    arrived = await _pump_until(
                        pilot, lambda: alpha.pending_gate is not None, tries=_RESURFACE_TURNS
                    )
                    assert (
                        arrived
                    ), "the gate never reached the viewer, so this proves nothing" + _state(
                        app, alpha, source, probe
                    )

                    mounted = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert mounted, (
                        "REGRESSION: a gate arrived at the session the user is looking "
                        "at while its bridge was detached, and NO card mounted. Both "
                        "re-arms are edge-triggered on navigation, so with no switch "
                        "coming there is no edge left to heal this: the turn stays "
                        "blocked with nothing on screen saying so. This is what the "
                        "level-triggered reconcile in _source_frontend_changed "
                        "exists to prevent." + diagnosis
                    )
                    assert not gate_task.done(), (
                        "the gate was answered without the user" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_the_navigation_settled_rearm_heals_a_display_only_source(
    tmp_path, monkeypatch
) -> None:
    """THE NAMED DEFENDER of the re-arm at ``_sidebar_navigation_pending``.

    WHY THIS TEST HAD TO BE WRITTEN SEPARATELY, and it is the whole point of it.
    The fix unfuses ``resume_viewer_gates`` from ``not display_only`` at TWO
    sites. Mutation-testing them one at a time showed the commit site
    (``_commit_sidebar_session``) is defended by
    ``test_a_display_only_commit_rearms_its_gate_bridge``, but re-fusing
    the OTHER one — the ``session_id == ""`` arm of
    ``_sidebar_navigation_pending`` — left the entire suite GREEN (18 passed).
    The level-triggered reconcile added by the same fix heals the gate before
    any assertion can notice that this arm did nothing, so the arm looks like
    dead code to anything that measures only the user-visible outcome. Dead-
    looking code with no test on it is deleted by the next refactor, and this
    one is load-bearing: it is the route that heals a detached bridge after a
    SUPERSEDED or FAILED navigation, where no commit runs at all (cases A and B
    in ``test_gate_resurface_real_entry.py`` depend on it entirely).

    HOW THE SEAM IS ISOLATED. Both of the other routes back are suppressed for
    the duration of the measurement — the commit-site re-arm and the reconcile
    — so the ONLY thing that can clear ``_gates_detached`` is the arm under
    test. The suppression is on the APP, not on the session: nothing about
    ``resume_viewer_gates`` or the ladder is stubbed, so what is measured is
    genuinely "did THIS call site fire", and the assertion is the latch it
    clears rather than a card that several routes could have produced.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted; the premise is unmet"

                    # ISOLATE FIRST, then detach. Order matters and the reason
                    # is itself the finding: `_suspend_sidebar_gates` arms a
                    # done-callback onto `_source_frontend_changed`, so with the
                    # reconcile live the detach is undone within a few turns and
                    # the premise below cannot even be established. That is the
                    # masking this test exists to defeat, observed directly.
                    monkeypatch.setattr(
                        OperatorApp, "_reconcile_gate_surface", lambda self, candidate: None
                    )
                    monkeypatch.setattr(
                        OperatorApp,
                        "_commit_sidebar_session",
                        lambda self, *a, **k: pytest.fail(
                            "the commit route ran; this test no longer isolates the "
                            "navigation-settled arm"
                        ),
                    )

                    # Detach the bridge exactly as a departure does, then make
                    # the source `display_only` — the state in which the arm
                    # under test used to be skipped.
                    app._suspend_sidebar_gates(source)
                    await _pump(pilot, 10)
                    source.display_only = True
                    assert alpha._gates_detached, "the suspend did not detach the gates"

                    # The settled-navigation publication, which is what
                    # `_prepare_and_commit`'s `finally` performs on EVERY
                    # navigation including a superseded or failed one.
                    app._sidebar_navigation_pending("")
                    await _pump(pilot, 20)

                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not alpha._gates_detached, (
                        "REGRESSION: the settled-navigation re-arm at "
                        "`_sidebar_navigation_pending` did NOT clear _gates_detached "
                        "for a display_only source. This is the route that heals a "
                        "superseded or failed navigation, where no commit ever runs; "
                        "with it suppressed, such a switch strands the gate bridge "
                        "with G4 dropping every attempt." + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_a_display_only_frame_is_ready_only_with_a_CORRECTLY_BOUND_card(
    tmp_path, monkeypatch
) -> None:
    """EDIT 1, and the narrowness of it.

    ``_sidebar_gate_surface_ready`` answered False for ANY ``display_only``
    frame that had a card mounted, which meant a frame painted perfectly well
    was declared unpaintable: ``_await_sidebar_frame`` sat out its 15 s timer,
    raised ``SurfaceNotReady``, and latched ``display_only`` with a "Reconnect
    failed" notice. The relaxation makes a ``display_only`` frame run the same
    card check the live branch runs.

    BOTH HALVES ARE ASSERTED, because only the pair pins the relaxation at the
    right width:

    * a card whose ``source_binding`` matches the source's current gate identity
      makes the frame READY — this is the case that used to hang;
    * a card that is mounted but NOT bound to this gate is still REFUSED — a
      relaxation to "any card is fine" would accept a card belonging to another
      session or to a superseded gate view, which is the identity confusion this
      whole bug is made of.

    The difference between the two legs is ONE component of the binding tuple,
    mutated on the card itself, so nothing else can explain a divergence in the
    verdict.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted; the premise is unmet"
                    card = app._ask_screen
                    assert card is not None

                    # The frame is `display_only`, which is the state whose
                    # readiness verdict is under test. Set on the SOURCE, which
                    # is where `_prepare_sidebar_session` latches it.
                    source.display_only = True
                    bound = (
                        source.token,
                        app._sidebar_gate_identity(source),
                        source.gate_view_generation,
                    )
                    assert card.source_binding == bound, (
                        "the mounted card is not bound to this source's gate, so the "
                        "'correctly bound' leg below would be vacuous: "
                        f"{card.source_binding!r} != {bound!r}" + _state(app, alpha, source, probe)
                    )

                    # THE REAL PREDICATE, `_sidebar_gate_surface_ready`, not the
                    # helper it delegates to. Asking the helper directly would
                    # pass even with the display_only branch reverted, because
                    # the revert reinstates an EARLY RETURN in the caller and
                    # leaves the helper untouched — verified by mutation.
                    # So the frame evidence the caller demands is built for real
                    # from the live compositor.
                    from textual._compositor import LayoutUpdate  # noqa: F401

                    visible = app.screen._compositor._visible_widgets
                    assert visible, "the compositor painted nothing; no frame evidence"
                    app._sidebar_displayed_frame = (
                        source.token,
                        app._sidebar_navigation.generation,
                        getattr(source.session, "display_history_revision", 0),
                        source.presentation_revision,
                        {id(widget): geometry for widget, geometry in visible.items()},
                    )

                    ready_when_bound = app._sidebar_gate_surface_ready(source)

                    # Now break ONE component of the binding and ask again.
                    card.source_binding = (
                        source.token,
                        app._sidebar_gate_identity(source),
                        source.gate_view_generation + 1,
                    )
                    ready_when_unbound = app._sidebar_gate_surface_ready(source)

                    diagnosis = (
                        f"\n  ready_when_bound = {ready_when_bound!r}"
                        f"\n  ready_when_unbound = {ready_when_unbound!r}"
                        f"\n  source.display_only = {source.display_only!r}"
                        + _state(app, alpha, source, probe)
                    )
                    _dump(diagnosis)

                    assert ready_when_bound, (
                        "a display_only frame holding a CORRECTLY BOUND gate card was "
                        "refused. That refusal is what sent the frame into the 15 s "
                        "_await_sidebar_frame timer, raised SurfaceNotReady and "
                        "latched 'Reconnect failed' on a frame that was painted "
                        "correctly." + diagnosis
                    )
                    assert not ready_when_unbound, (
                        "a display_only frame holding an UNBOUND card was accepted as "
                        "ready. The relaxation is too wide: a card belonging to a "
                        "superseded gate view (or another session) would be treated as "
                        "this gate's surface." + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("history", ["current", "stale"])
async def test_a_display_only_frame_whose_gate_cannot_be_presented_still_paints(
    tmp_path, monkeypatch, history: str
) -> None:
    """A card-less preview is ready at once, and the card follows when it can.

    THE STATE. The user returns to a session that holds an unanswered gate
    while its viewer is not ready for events (``_ready_for_events`` False: a
    display resync in flight, with the socket up). That is ``is_cold`` through
    its third disjunct, so preparation latches ``display_only``. It is also G1
    of ``_maybe_start_gate``, so no card can mount in this state. The frame
    must not wait for one: the connect that ends the state only starts once the
    frame resolves (``connect_after_paint``). A frame that waited would wait on
    itself, sit out ``_await_sidebar_frame``'s 15 s timer and end in the terminal
    ``SurfaceNotReady`` ("Could not open conversation") over a correctly painted
    preview.

    BOTH SUB-SHAPES, because they fail differently when the escape is missing.
    With the display history still current the paint-recovery branch fires on
    every refused frame (a relayout spin). With it stale that branch is skipped
    and the frame waits out the timer. ``stale`` is driven for real, through a
    held ``frontend_sync`` and ``_invalidate_display_history``. ``current``
    clears ``_ready_for_events`` directly, since a real refresh always
    invalidates the history first and cannot produce this sub-shape on its own.

    THE SECOND HALF IS THE POINT OF THE PR. Resolving the frame must not cost
    the card: once the viewer is ready again the card has to come back without
    the user switching. That is why the escape lives in the frame verdict and
    not in the card predicate the reconcile also asks.

    Settlement is measured in turns (``_RESURFACE_TURNS``), not seconds. The
    ``committed_id`` it waits on is only set after the frame future resolves
    successfully, so a pass cannot come from the timer firing.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    release_sync = asyncio.Event()
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted; the premise is unmet"

                    await _click(app, pilot, "beta")
                    assert app._sidebar_navigation.committed_id == "beta"
                    assert app._ask_screen is None, "the outgoing card is still on screen"

                    if history == "stale":
                        client = alpha._client
                        assert client is not None
                        real_sync = client.frontend_sync

                        async def held_sync(*args, **kwargs):
                            await release_sync.wait()
                            return await real_sync(*args, **kwargs)

                        monkeypatch.setattr(client, "frontend_sync", held_sync)
                        alpha._invalidate_display_history()
                        assert await _pump_until(
                            pilot, lambda: not alpha._ready_for_events, tries=100
                        ), "the held resync never cleared _ready_for_events"
                        assert not alpha.display_history_current
                    else:
                        alpha._ready_for_events = False
                        assert alpha.display_history_current
                    assert alpha.is_cold, "the premise needs is_cold via _ready_for_events"

                    reached_at = app._sidebar_gate_reached
                    recoveries_at = app._sidebar_gate_recoveries
                    app.post_message(SessionSidebar.Selected("alpha"))
                    painted = await _pump_until(
                        pilot,
                        lambda: app._sidebar_navigation.committed_id == "alpha",
                        tries=_RESURFACE_TURNS,
                    )
                    diagnosis = (
                        f"\n  history = {history!r}"
                        f"\n  painted = {painted!r}"
                        f"\n  gate_reached delta = {app._sidebar_gate_reached - reached_at}"
                        f"\n  gate_recoveries delta = "
                        f"{app._sidebar_gate_recoveries - recoveries_at}"
                        + _state(app, alpha, source, probe)
                    )
                    _dump(diagnosis)

                    assert painted, (
                        "the display_only frame for a session whose gate cannot be "
                        "presented never resolved. It waits for a card that G1 refuses "
                        "to mount, and the only exit left is the 15 s timer's terminal "
                        "SurfaceNotReady." + diagnosis
                    )
                    assert app._sidebar_gate_reached > reached_at, (
                        "the frame resolved without reaching the paint gate, so this "
                        "did not test the gate's verdict" + diagnosis
                    )
                    assert app._sidebar_gate_recoveries == recoveries_at, (
                        "the paint gate bought recovery relayouts for a refusal a "
                        "repaint cannot fix" + diagnosis
                    )
                    assert source.display_only and not alpha._ready_for_events, (
                        "the state moved on before the verdict could be read; the "
                        "premise does not hold" + diagnosis
                    )
                    assert app._ask_screen is None and alpha.pending_gate is not None

                    # And the card comes back once the viewer is ready, with no
                    # switch. `stale` heals for real. `current` restores the flag
                    # and delivers the frontend notification a delta would.
                    if history == "stale":
                        release_sync.set()
                    else:
                        alpha._ready_for_events = True
                        app._source_frontend_changed(source)
                    resurfaced = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )
                    assert resurfaced, (
                        "the preview painted, but the card never came back once the "
                        "viewer was ready for events" + _state(app, alpha, source, probe)
                    )
                    assert not gate_task.done(), "the gate resolved without an answer"

                    # THE PLACEMENT OF THE ESCAPE. Above, the card can also come
                    # back through the connect path, so that half cannot tell
                    # an escape in the frame verdict from one in
                    # `_sidebar_gate_card_ready`. This leg can. A card-less,
                    # detached `display_only` source whose viewer IS ready has
                    # exactly one route back: the level-triggered reconcile,
                    # which `_suspend_sidebar_gates`' done-callback reaches
                    # through `_source_frontend_changed`. An escape in the card
                    # predicate would tell the reconcile the preview is fine and
                    # the card would stay gone. The connect is suppressed, so
                    # it cannot heal this leg instead.
                    monkeypatch.setattr(
                        OperatorApp,
                        "_start_sidebar_connection",
                        lambda self, candidate, **_kwargs: None,
                    )
                    await _pump(pilot, 20)
                    source.display_only = True
                    app._suspend_sidebar_gates(source)
                    healed = await _pump_until(
                        pilot,
                        lambda: app._ask_screen is not None and not alpha._gates_detached,
                        tries=_RESURFACE_TURNS,
                    )
                    assert healed, (
                        "the reconcile did not bring back the card of a detached, "
                        "card-less display_only source whose viewer is ready. The "
                        "display_only escape is answering the reconcile's question "
                        "as well as the frame's." + _state(app, alpha, source, probe)
                    )
                finally:
                    release_sync.set()
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_a_delta_inside_a_departing_detach_does_not_rearm_the_bridge(
    tmp_path, monkeypatch
) -> None:
    """The reconcile must not undo ``detach_viewer_gates`` on a departing session.

    ``/resume`` onto a live owner (``_attach_or_refuse``) detaches the outgoing
    session's gates behind an ``await`` and then disposes the session. For the
    whole of that await ``self._interaction`` is still the outgoing source, so
    a frontend delta landing there reaches the reconcile looking like "current,
    pending gate, no ready card". Nothing in the latch tells a departure from a
    sidebar suspension: both set ``_gates_detached``. The reconcile tells them
    apart by the window it runs in (``_session_transition_pending``), and this
    pins that.

    DRIVEN THROUGH THE REAL SEAM. The transition runs through
    ``_run_session_transition`` into the real ``_attach_or_refuse``, dialling a
    real in-process owner. The only thing added is the delta, delivered from
    inside the real ``detach_viewer_gates`` after its own await, which is the
    window the race needs. The latch is read right there, before the dispose
    that follows can hide the outcome.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted; the premise is unmet"

                    real_detach = alpha.detach_viewer_gates
                    seen: dict[str, object] = {}

                    async def detach_with_a_delta(**kwargs):
                        await real_detach(**kwargs)
                        seen["detached_before_delta"] = alpha._gates_detached
                        seen["current"] = app._interaction is source
                        app._source_frontend_changed(source)
                        seen["detached_after_delta"] = alpha._gates_detached

                    monkeypatch.setattr(alpha, "detach_viewer_gates", detach_with_a_delta)
                    app._run_session_transition(app._attach_or_refuse(rig.config, "beta"))
                    landed = await _pump_until(
                        pilot,
                        lambda: getattr(app._session, "session_id", "") == "beta",
                        tries=_RESURFACE_TURNS,
                    )
                    diagnosis = f"\n  seen = {seen!r}" + _state(app, alpha, source, probe)
                    _dump(diagnosis)

                    assert landed, "the /resume attach never landed on beta" + diagnosis
                    assert seen.get("detached_before_delta") is True and seen.get("current"), (
                        "the delta did not land inside the departing detach, so the "
                        "window under test was not exercised" + diagnosis
                    )
                    assert seen.get("detached_after_delta") is True, (
                        "REGRESSION: a frontend delta inside /resume's detach cleared "
                        "_gates_detached, re-arming a gate bridge for the conversation "
                        "being disposed" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_a_delta_inside_the_relaunch_window_does_not_rearm_the_bridge(
    tmp_path, monkeypatch
) -> None:
    """The reconcile must also stay out of the RELAUNCH window (``_restart_plan``).

    The third disjunct of the reconcile's guard, and the one nothing else
    exercises: ``_session_transition_pending`` is covered by the sibling case
    above, and ``_sidebar_navigation.requested_id`` by the rearm cases beside it.
    A relaunch is the same shape as a departure for a different reason —
    ``_detach_relaunch_gates`` withdraws every viewer bridge (``preserve_answers``
    so the next viewer receives the unanswered gate from canonical state) and the
    process re-execs, so a frontend delta that lands in that window must not put a
    bridge back on a session about to be disposed. Nothing in
    ``_gates_detached`` tells that apart from a sidebar suspension, which is why
    the window is read from the plan.

    DRIVEN THROUGH THE REAL SEAM. The withdrawal is the real
    ``_detach_relaunch_gates``; the delta is delivered through the real
    ``_source_frontend_changed``, which is what a socket read calls. Only
    ``_restart_plan`` itself is set by hand: constructing a ``RestartPlan`` means
    ``make_plan``/``stash_plan`` over ``sys.argv``, and the plan's contents are
    not what the guard reads — its PRESENCE is.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted; the premise is unmet"

                    app._restart_plan = object()
                    await app._detach_relaunch_gates()
                    await _pump(pilot, 16)

                    # THE GUARD'S OWN PREMISE, asserted rather than assumed: every
                    # earlier return in the reconcile has to be FALSE, or this test
                    # would pass without reaching the disjunct it is about. The
                    # earlier ones are `requested_id` (nothing was requested here)
                    # and `_sidebar_gate_card_ready` — hence the check that the
                    # withdrawal really did take the card off the screen.
                    card_ready = app._sidebar_gate_card_ready(source, require_paint=False)
                    detached_before = alpha._gates_detached
                    assert detached_before, (
                        "the relaunch withdrawal did not leave the current source "
                        "detached, so there is no latch for the delta to undo and this "
                        "test proves nothing"
                    )
                    assert not card_ready and app._ask_screen is None, (
                        "the card is still mounted, so the reconcile would return at "
                        "`_sidebar_gate_card_ready` and never reach the plan disjunct"
                    )

                    app._source_frontend_changed(source)
                    await _pump(pilot, 16)

                    diagnosis = (
                        f"\n  detached before the delta = {detached_before!r}"
                        f"\n  detached after the delta = {alpha._gates_detached!r}"
                        f"\n  _restart_plan is not None = {app._restart_plan is not None!r}"
                        + _state(app, alpha, source, probe)
                    )
                    _dump(diagnosis)

                    assert alpha._gates_detached, (
                        "REGRESSION: a frontend delta inside the relaunch window cleared "
                        "_gates_detached and re-armed a gate bridge on a session whose "
                        "runtime is about to be disposed by the re-exec" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()
