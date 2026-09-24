"""The gate-resurface bug driven through the REAL user entry point.

WHAT THIS FILE ADDS TO ITS SIBLING. ``test_gate_resurface_on_switch.py`` is the
control harness: it drives ``_prepare_sidebar_session`` /
``_commit_sidebar_session`` directly and overrides ``_lease_sidebar_source``,
which is the happy path with the rig holding it straight. Everything here goes
through what the USER touches instead:

* ``SessionSidebar.Selected`` -> ``OperatorApp.on_session_sidebar_selected``
  -> ``_select_sidebar_session`` -> ``SessionNavigation.select`` -> ``_navigate``
  -> ``_prepare_and_commit``, including its exception handling at
  session_navigation.py:176-178;
* the keyboard route, ``_switch_session_from`` -> ``post_message(Selected)``,
  which is what ``action_switch_session`` delegates to once it has a catalogue;
* the REAL ``_lease_sidebar_source``, over real in-process runtimes discovered
  through a patched ``find_runtime_record`` (the same rig
  ``tests/e2e/test_sidebar_reconnect_e2e.py`` uses). Nothing about the app's own
  leasing, parking, subscribing or gate wiring is stubbed.

FIVE CONDITIONS, EACH A REGRESSION GATE. Every case carries the
``_maybe_start_gate`` guard instrumentation borrowed from the sibling file, so a
failure names the guard that dropped the card rather than a theory about one.

* **A — rapid switching.** A superseded navigation still re-arms the bridge.
  See :func:`test_a_rapid_switching_still_rearms_the_gate_bridge`.
* **B — abandoned/failed commit.** A commit that raises after
  ``_suspend_sidebar_gates`` leaves a torn state, and the next selection heals
  it. See
  :func:`test_a_commit_that_fails_after_suspending_gates_heals_on_the_next_click`.
* **C — gate arrives while backgrounded.** It mounts on return. See
  :func:`test_a_gate_raised_while_the_session_is_backgrounded_mounts_on_return`.
* **D — a source whose reconnect can never complete.** Its ``display_only``
  latch never clears, and its card must still come back. See
  :func:`test_a_source_that_can_never_bind_loses_its_gate_card`.
* **E — the answered-key latch against a second question.** The second question
  of one ask still resurfaces. See
  :func:`test_a_second_question_in_the_same_ask_resurfaces_across_a_switch`.

THE THREE ROUTES BACK FOR A DETACHED BRIDGE, since every case above is about
one of them:

* the commit: ``_commit_sidebar_session`` calls ``resume_viewer_gates()`` for
  every source it commits, ``display_only`` or not;
* the settled navigation: ``_sidebar_navigation_pending``'s ``session_id == ""``
  arm, reached from ``_prepare_and_commit``'s ``finally`` on EVERY settled
  navigation, including a failed or superseded one where no commit runs at all
  (A and B depend on it);
* the reconcile: ``_reconcile_gate_surface``, level-triggered from
  ``_source_frontend_changed``, which re-arms the visible session whenever it
  owes a card it lacks.

None of them is gated on ``display_only``. Re-arming a bridge submits nothing,
so the flag that stops a saved preview from starting a turn has no business
stopping it. D is the case that pins that: its latch can never clear.

HERMETICITY. Every test pins ``LOCAL_OPERATOR_CONFIG_DIR`` at its own
``tmp_path``, patches ``find_runtime_record`` to see only its own in-process
runtimes, and REFUSES ``_spawn_runtime`` outright — a case that reached for a
real runtime subprocess would fail rather than quietly start one and pass.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.session.attached import AttachedSession
from local_operator.session.catalog import CatalogEntry, SessionRow
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.session_sidebar import SessionSidebar
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript

# The instrumentation IS the sibling file's. Imported rather than copied so the
# guard ladder replay cannot drift into two versions that disagree.
from tests.unit.tui.test_gate_resurface_on_switch import (
    _attached,
    _format,
    _install_probe,
    _pump,
    _pump_until,
    _rows,
)

#: Turn budget for "the card came back". Sized well above what a healthy re-arm
#: takes (measured 1-3 calls to `_maybe_start_gate`, all within ~30 turns) so a
#: loaded runner cannot turn a pass into a failure. A COUNT, not a clock.
_RESURFACE_TURNS = 400


async def _never_take_over() -> Any:
    raise AssertionError("a viewer must not take over during this test")


class _Rig:
    """Real in-process runtimes plus the app that views them.

    One object because every case needs the same four things wired together and
    torn down in the same order: the runtimes, the discovery patch that makes
    only THOSE runtimes findable, the app booted onto one of them, and the
    handles that raise gates on the owner side.
    """

    def __init__(self, config: Path) -> None:
        self.config = config
        self.servers: dict[str, RuntimeServer] = {}
        self.handles: dict[str, ServingSessionHandle] = {}

    async def runtime(self, session_id: str) -> None:
        directory = self.config / "sessions" / session_id
        await seed_transcript(directory, _rows())
        owner = build_session(directory, ScriptedStream([]), cwd=self.config)
        handle = ServingSessionHandle(owner, asyncio.get_running_loop(), cwd=str(self.config))
        server = RuntimeServer(handle, kind="daemon")
        await server.start_in_process()
        # Every case in this file is about a gate the user must answer, so no
        # owner may answer one for itself.
        handle._auto_approve = False
        self.servers[session_id] = server
        self.handles[session_id] = handle

    def find_owner(self, _config_dir: Any, session_id: str, **_probe: Any) -> tuple[Any, Any]:
        server = self.servers.get(session_id)
        if server is None or server._closed.is_set():
            return (None, None)
        return (server._record, server._record.pid)

    async def dispose(self) -> None:
        for session_id, server in self.servers.items():
            server.close()
            await self.handles[session_id].dispose()


async def _rig(config: Path, monkeypatch: pytest.MonkeyPatch, *names: str) -> _Rig:
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

    # HERMETICITY, asserted rather than assumed: the real `_lease_sidebar_source`
    # and the real `_ensure_bound` are in play here, and both can reach
    # `engage_runtime`, which starts a DETACHED `lop` runtime process. A case that
    # gets there must fail loudly instead of passing after ~30 s of real spawning.
    def refuse_spawn(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            f"this test tried to spawn a real runtime process for {session_id!r}; "
            "the rig is supposed to serve every session in-process"
        )

    monkeypatch.setattr("local_operator.session.runtime.launch._spawn_runtime", refuse_spawn)

    rig = _Rig(config)
    for name in names:
        await rig.runtime(name)
    return rig


def _app(rig: _Rig, boot: str, *, viewer: bool = False) -> OperatorApp:
    """The app, booted onto ``boot`` through a factory shaped like the CLI's.

    ``viewer`` picks WHICH owner-loss contract the facade carries, and it is the
    same single knob ``AttachedSession.connect`` documents. False is what
    ``lop`` itself builds at cli.py:8313 (the legacy attach contract); the
    sidebar's own leases pass True from ``_lease_sidebar_source``. Both are
    exercised in this file, because D turns on exactly that difference.
    """

    async def resume(session_id: str | None) -> AttachedSession:
        # `str | None` because that is `OperatorApp`'s `resume_factory` contract
        # (app.py:3419): None asks for a brand new conversation. This rig only
        # ever resumes a seeded runtime, so None here is a bug in the test.
        assert session_id is not None
        return await AttachedSession.connect(
            rig.servers[session_id]._record,
            session_id,
            config_dir=rig.config,
            takeover_factory=_never_take_over,
            display_window=True,
            viewer=viewer,
        )

    return OperatorApp(lambda: resume(boot), resume_factory=resume)


async def _click(app: OperatorApp, pilot: Any, session_id: str, *, rounds: int = 10) -> None:
    """The user's click on a sidebar row, driven to settlement.

    ``post_message`` rather than ``_select_sidebar_session``: the handler
    ``on_session_sidebar_selected`` is part of what is under test — it is where
    the same-session no-op, the ``cancel()`` of an in-flight navigation and the
    ``intend("")`` publication live — so the message has to go through Textual's
    pump exactly as ``SessionSidebar`` posts it.

    Settlement is awaited on the navigation's own task, and re-awaited because
    ``_navigate`` LOOPS: a ``PreparationInvalidated`` retry replaces the task
    under us, and a superseding click creates a new one.
    """
    app.post_message(SessionSidebar.Selected(session_id))
    for _ in range(rounds):
        await _pump(pilot, 20)
        task = app._sidebar_navigation._task
        if task is None or task.done():
            break
        try:
            await asyncio.wait_for(asyncio.shield(task), 30)
        except asyncio.TimeoutError:  # pragma: no cover - a wedged navigation
            break
        except Exception:
            # A failed navigation is a CASE here, not an error: `_prepare_and_commit`
            # already reported it through `_sidebar_navigation_failed`, and what
            # this file measures is the state it left behind.
            break
    await _pump(pilot, 20)


def _entries(*ids: str) -> list[CatalogEntry]:
    """The sidebar's own ranking, as ``_switch_session_from`` reads it."""
    now = time.time()
    return [
        CatalogEntry(
            SessionRow(
                sid,
                now - 60 * (index + 1),
                f"Session {sid}",
                created_at=now - 60 * (index + 1),
            )
        )
        for index, sid in enumerate(ids)
    ]


async def _raise_ask(rig: _Rig, session_id: str, *questions: AskQuestion) -> asyncio.Task[Any]:
    """A REAL ask gate on the owner side, unanswered."""
    return asyncio.create_task(rig.handles[session_id]._ask_gate(list(questions)))


def _one_question() -> AskQuestion:
    return AskQuestion(
        id="destination",
        question="Choose a destination",
        options=[AskOption(label="Here"), AskOption(label="There")],
    )


def _state(app: OperatorApp, session: Any, source: SessionInteraction, probe: Any) -> str:
    """Everything the guard ladder and the re-arm route read, in one block."""
    lines = ["\n\nSTATE, sampled after the leg settled:"]
    for key, value in probe.readout().items():
        lines.append(f"  {key} = {value!r}")
    lines += [
        f"  source.display_only = {source.display_only!r}",
        f"  source.retired = {source.retired!r}",
        f"  source.can_never_bind = {source.can_never_bind!r}",
        f"  source.connection_error = {source.connection_error!r}",
        f"  source.connect_attempts = {source.connect_attempts!r}",
        f"  source.connection_task = {source.connection_task!r}",
        f"  source is app._interaction = {source is app._interaction!r}",
        f"  session.is_cold = {getattr(session, 'is_cold', None)!r}",
        f"  session.can_ever_bind = {getattr(session, 'can_ever_bind', None)!r}",
        f"  session._recovering = {getattr(session, '_recovering', None)!r}",
        f"  session.display_history_current = "
        f"{getattr(session, 'display_history_current', None)!r}",
        f"  app._session id = {getattr(app._session, 'session_id', None)!r}",
        f"  navigation.committed_id = {app._sidebar_navigation.committed_id!r}",
        f"  navigation.requested_id = {app._sidebar_navigation.requested_id!r}",
        f"  app._ask_screen = {app._ask_screen!r}",
        f"  app._approval = {app._approval!r}",
    ]
    return "\n".join(lines)


def _dump(text: str) -> None:
    if os.environ.get("GATE_PROBE_DUMP"):
        print(text)


# --- A: RAPID SWITCHING -------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["click-burst-2", "click-burst-3", "keyboard-burst"])
async def test_a_rapid_switching_still_rearms_the_gate_bridge(
    tmp_path, monkeypatch, route: str
) -> None:
    """NEGATIVE RESULT. A superseded navigation does not strand the gate.

    THE HYPOTHESIS, and it was the leading one. ``SessionNavigation.select``
    bumps ``generation`` and cancels the in-flight task (session_navigation.py:
    122-127), and ``_prepare_and_commit`` re-reads the generation immediately
    before ``_commit`` (:157). A commit that had ALREADY run
    ``_suspend_sidebar_gates`` (app.py:7383) but was fenced out before the
    resume at app.py:7539 would leave ``_gates_detached`` True with nothing to
    re-arm it. All three bursts here start a second navigation before the first
    can commit: two clicks A->B->A, three clicks A->B->C->A, and the keyboard
    form, where ``_switch_session_from`` is called three times in ONE event
    batch — the auto-repeat shape its own comments describe (round 5, U7).

    WHY IT DOES NOT REPRODUCE, measured rather than reasoned. There is a SECOND
    re-arm of the bridge and the first pass never saw it, because it never
    dispatched a real navigation: ``_prepare_and_commit``'s ``finally`` calls
    ``pending("")`` on every settled navigation (session_navigation.py:184-187),
    and ``_sidebar_navigation_pending``'s empty-id arm (app.py:6645-6646) calls
    ``current.resume_viewer_gates()`` on whatever viewer ended up on screen. So
    a superseded commit's detach is undone by the SETTLEMENT of the burst, not
    by its commit. The readout below shows exactly that: G4 fires on the
    intermediate legs and then a later call starts the bridge with
    ``_gates_detached`` False.

    This is a real property worth pinning, not merely an absence: it is the
    mechanism that makes A, B and C all heal, and an edit that moved the
    ``resume_viewer_gates`` out of that arm would turn every one of them into
    the reported bug. VERIFIED BY MUTATION rather than asserted: deleting the
    ``current.resume_viewer_gates()`` call at app.py:6646 fails all three
    parametrizations of this test with exactly the reported symptom (and also
    kills ``before_adopt`` in case B), so these negatives are watching the
    mechanism they name rather than passing for an unrelated reason.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta", "gamma")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(
                    pilot, lambda: app._session is not None, tries=300
                ), "the app never adopted a session; the rig never reached its premise"
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)

                gate_task = await _raise_ask(rig, "alpha", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted on the session that raised it"
                    assert not gate_task.done(), "the gate answered itself before the burst"

                    marker = probe.marker
                    if route == "click-burst-2":
                        # Both messages posted before either can dispatch: the
                        # second `select` supersedes the first mid-flight.
                        app.post_message(SessionSidebar.Selected("beta"))
                        app.post_message(SessionSidebar.Selected("alpha"))
                    elif route == "click-burst-3":
                        app.post_message(SessionSidebar.Selected("beta"))
                        app.post_message(SessionSidebar.Selected("gamma"))
                        app.post_message(SessionSidebar.Selected("alpha"))
                    else:
                        # The KEYBOARD route. `action_switch_session` reads the
                        # sidebar's catalogue and delegates here; three calls in
                        # one batch is the held-key auto-repeat, which is the
                        # shape `intent_id` exists for.
                        rows = _entries("alpha", "beta", "gamma")
                        app._switch_session_from(rows, 1)
                        app._switch_session_from(rows, 1)
                        app._switch_session_from(rows, 1)

                    # Drain the whole burst. `_navigate` LOOPS on retries and a
                    # superseding select replaces the task, so one await is not
                    # settlement.
                    for _ in range(8):
                        await _pump(pilot, 20)
                        task = app._sidebar_navigation._task
                        if task is None or task.done():
                            break
                        try:
                            await asyncio.wait_for(asyncio.shield(task), 30)
                        except Exception:
                            break
                    await _pump(pilot, 30)

                    # Wherever the burst landed, the user then deliberately
                    # returns to the session holding the gate.
                    if getattr(app._session, "session_id", "") != "alpha":
                        await _click(app, pilot, "alpha")
                    resurfaced = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )

                    diagnosis = (
                        _format(probe.since(marker), f"BURST ({route})")
                        + _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not gate_task.done(), (
                        "the burst ANSWERED the gate: navigation invented an answer "
                        "the user never gave" + diagnosis
                    )
                    assert app._session is alpha, (
                        "the burst did not end on the session holding the gate, so "
                        "this leg proves nothing about the re-arm" + diagnosis
                    )
                    assert resurfaced, (
                        f"the live ask card did not come back after a {route}. If this "
                        "is failing, the supersession hypothesis (A) has become real: "
                        "check which guard the readout names and whether "
                        "_sidebar_navigation_pending's empty-id arm still calls "
                        "resume_viewer_gates (app.py:6645-6646)." + diagnosis
                    )
                    card = app._ask_screen
                    assert card is not None and not card.disabled, (
                        "the card came back DISABLED — visible, unanswerable, and the "
                        "turn stays blocked" + diagnosis
                    )
                    assert not alpha._gates_detached, (
                        "the bridge is on screen but the facade still reads detached; "
                        "the next gate in this session would be dropped" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- B: ABANDONED / FAILED COMMIT --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["before_adopt", "after_adopt"])
async def test_a_commit_that_fails_after_suspending_gates_heals_on_the_next_click(
    tmp_path, monkeypatch, where: str
) -> None:
    """NEGATIVE RESULT, with a real intermediate tear worth recording.

    THE INJECTION. ``_commit_sidebar_session`` suspends the outgoing gates at
    app.py:7383 and unmounts both cards at 7386-7393, then does ~150 lines of
    widget work before reaching ``resume_viewer_gates()`` at 7539. A failure
    anywhere in that window is caught by
    ``SessionNavigation._prepare_and_commit`` (session_navigation.py:176-178),
    which calls ``_failed`` and abandons the switch. Both parametrizations raise
    inside that window, on either side of ``_adopt_session`` — the line that
    actually moves ``self._interaction`` (app.py:7446) — because the two sides
    leave genuinely different wreckage, and only measurement says which:

    * ``before_adopt`` (in ``_apply_sidebar_presentation``): ownership never
      moved. The app is still on the outgoing session, and the card is STILL
      MOUNTED — the raise happens after the unmount at 7386, so this is the
      re-arm putting it back, not the unmount having been skipped.
    * ``after_adopt`` (in ``_sync_draft_recoveries``): ownership DID move. The
      app is on the incoming session with ``committed_id`` never set, the
      outgoing card is gone, and ``_gates_detached`` is left True on a session
      nobody is looking at. That is the torn state the hypothesis predicted —
      and it is not permanent.

    WHAT HEALS IT is the same mechanism A measures: the ``finally`` at
    session_navigation.py:179-187 runs on the exception path too, so
    ``pending("")`` fires and ``_sidebar_navigation_pending`` re-arms the viewer
    that ended up on screen. For ``before_adopt`` that viewer IS the one holding
    the gate, which is why its card is back before the test even asks. For
    ``after_adopt`` the user's next click on that row re-arms it normally.

    So a failed commit is not the reported bug. What it IS is a state where the
    card is gone while the turn is blocked and NOTHING on screen says so until
    the user happens to click back — recorded here as the measured shape rather
    than asserted as a defect, because the recovery is real.
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

                    # A ONE-SHOT failure inside the commit body. Injected on a
                    # method the commit calls between 7383 and 7539 rather than
                    # by patching the commit itself, so every other line of the
                    # real body still runs and the wreckage is the real wreckage.
                    target = (
                        "_apply_sidebar_presentation"
                        if where == "before_adopt"
                        else "_sync_draft_recoveries"
                    )
                    real = getattr(OperatorApp, target)
                    fired = {"count": 0}

                    def boom(self: OperatorApp, *args: Any, **kwargs: Any) -> Any:
                        if fired["count"] == 0 and self is app:
                            fired["count"] = 1
                            raise RuntimeError(f"injected widget failure in {target}")
                        return real(self, *args, **kwargs)

                    monkeypatch.setattr(OperatorApp, target, boom)

                    failed_marker = probe.marker
                    await _click(app, pilot, "beta")
                    torn = (
                        _format(probe.since(failed_marker), f"FAILED COMMIT ({where})")
                        + _state(app, alpha, source, probe)
                        + f"\n  injection fired = {fired['count']!r}"
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(torn)

                    assert fired["count"] == 1, (
                        f"the injected failure in {target} never ran, so no commit was "
                        "abandoned and this test measured nothing" + torn
                    )
                    assert not gate_task.done(), "the abandoned commit ANSWERED the gate" + torn
                    assert app._sidebar_navigation.committed_id == "", (
                        "the navigation recorded a commit even though the commit body "
                        "raised" + torn
                    )

                    # The user clicks the row holding the gate. For `before_adopt`
                    # they are already on it, and `on_session_sidebar_selected`'s
                    # same-session arm (app.py:8235) handles that click instead of
                    # starting a navigation — which is itself part of the path.
                    return_marker = probe.marker
                    await _click(app, pilot, "alpha")
                    resurfaced = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )
                    diagnosis = (
                        torn
                        + _format(probe.since(return_marker), "RETURN LEG after the failure")
                        + _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not gate_task.done(), (
                        "the gate resolved without the user answering it" + diagnosis
                    )
                    assert resurfaced, (
                        "REPRODUCED (B): a commit that raised after _suspend_sidebar_gates "
                        "left the gate unreachable even after the user clicked back onto "
                        "the session. The turn is blocked with no surface to answer it." + diagnosis
                    )
                    assert not alpha._gates_detached, (
                        "the card is on screen but the facade still reads detached" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- C: A GATE THAT ARRIVES WHILE THE SESSION IS BACKGROUNDED ----------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ask", "approval"])
async def test_a_gate_raised_while_the_session_is_backgrounded_mounts_on_return(
    tmp_path, monkeypatch, kind: str
) -> None:
    """NEGATIVE RESULT for the commonest real shape the first pass never tried.

    THE SHAPE: a background agent hits an approval while you work elsewhere.
    Session A is NOT on screen and has no gate at all when the user leaves it,
    so there is no card to snapshot and no ``gate_draft`` — the sibling file's
    whole mechanism is absent here. The gate then ARRIVES at a viewer whose
    ``_gates_detached`` is already True: ``_apply_pending_gate`` (attached.py:
    5576-5586) calls ``_maybe_start_gate``, which drops it at G4, correctly,
    because there is no screen to mount it on.

    WHAT HAD TO BE PROVEN is that the arrival is not LOST by that drop — that
    returning to the session mounts a card for a gate the viewer learned about
    while parked. It is: the return leg re-arms through ``resume_viewer_gates``
    and the bridge starts from ``self.pending_gate``, which is canonical owner
    state rather than anything the card left behind. Both gate kinds are covered
    because the approval path has an extra arm (``_background_approval``) that
    can answer in the background, and a rig where that arm fired would prove the
    opposite of what this asserts — so the test also pins that it did not.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(100, 30)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                # Explicitly OFF, so the background-approval arm cannot answer
                # for the user and turn this into a vacuous pass.
                app._set_approve_all(False)
                source.draft.approve_all = False
                probe = _install_probe(monkeypatch, alpha)

                # Leave alpha FIRST, with nothing pending anywhere.
                assert alpha.pending_gate is None
                await _click(app, pilot, "beta")
                assert (
                    getattr(app._session, "session_id", "") == "beta"
                ), "the switch away never landed on beta"
                assert alpha._gates_detached, "the switch away did not detach alpha's gates"

                # NOW the backgrounded owner raises the gate.
                if kind == "ask":
                    gate_task = await _raise_ask(rig, "alpha", _one_question())
                else:
                    gate_task = asyncio.create_task(
                        rig.handles["alpha"]._approval_gate("write", "Save one record")
                    )
                try:
                    arrived = await _pump_until(
                        pilot, lambda: alpha.pending_gate is not None, tries=_RESURFACE_TURNS
                    )
                    assert arrived, (
                        "the gate never reached the backgrounded viewer at all, so the "
                        "return leg below would prove nothing" + _state(app, alpha, source, probe)
                    )
                    assert app._ask_screen is None and app._approval is None, (
                        "a card for a BACKGROUNDED session mounted over the session the "
                        "user is actually looking at" + _state(app, alpha, source, probe)
                    )
                    assert not gate_task.done(), (
                        "the gate was answered while its session was off screen; the "
                        "background arm answered for the user" + _state(app, alpha, source, probe)
                    )

                    marker = probe.marker
                    await _click(app, pilot, "alpha")

                    def mounted() -> bool:
                        return (app._ask_screen if kind == "ask" else app._approval) is not None

                    resurfaced = await _pump_until(pilot, mounted, tries=_RESURFACE_TURNS)
                    diagnosis = (
                        _format(probe.since(marker), f"RETURN LEG ({kind} raised while parked)")
                        + _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not gate_task.done(), (
                        "the gate resolved without the user answering it" + diagnosis
                    )
                    assert resurfaced, (
                        f"REPRODUCED (C): a {kind} raised while the session was in the "
                        "background never got a card when the user returned to it. The "
                        "turn is blocked and the arrival was dropped at the guard the "
                        "readout names." + diagnosis
                    )
                    card = app._ask_screen if kind == "ask" else app._approval
                    assert card is not None and not card.disabled, (
                        "the card mounted DISABLED, so the turn stays blocked" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- D: A SOURCE WHOSE RECONNECT CAN NEVER COMPLETE --- THE REPRODUCTION -----


@pytest.mark.asyncio
async def test_a_source_that_can_never_bind_loses_its_gate_card(tmp_path, monkeypatch) -> None:
    """A session whose ``display_only`` latch can never clear still gets its card.

    THE CHAIN this pins, every link checked by the readout this test prints:

    1. The user leaves session A. ``_commit_sidebar_session`` suspends A's gates
       (``suspend_viewer_gates``, ``_gates_detached = True``) and unmounts both
       cards unconditionally.
    2. A's owner is STOPPED, from another terminal's ``/stop all`` or a shell
       ``lop stop``. The runtime announces it on the wire and closes.
       ``_on_disconnected``'s deliberate-stop branch returns BEFORE setting
       ``_recovering``, so there is no recovery loop, and on the boot facade
       ``_can_go_cold`` is False, so ``can_ever_bind`` answers False for the
       rest of the process. This is not a contrived facade: it is what the CLI
       builds (``AttachedSession.connect`` with no ``viewer=True``), and
       ``can_ever_bind``'s own docstring names this as the reachable False.
    3. The user clicks back onto A. ``_prepare_sidebar_session`` latches
       ``display_only`` (``or session.is_cold``), and nothing on this path can
       clear it: the connect armed after the paint raises at the bind
       postcondition and takes the ``terminal_reason`` arm, which sets no retry.
    4. The card must come back anyway, on the first return and on every later
       one. The commit's ``resume_viewer_gates()`` and the settled-navigation
       re-arm both run for a ``display_only`` source, so ``_gates_detached``
       clears and the ladder mounts the card. Gating either of them on
       ``display_only`` drops the card here for good, with G4 refusing every
       attempt and nothing on screen mentioning the pending question.

    WHAT THIS DOES AND DOES NOT CLAIM. The card loss is real and permanent
    without the re-arm. The BLOCKED TURN behind it is not fully modelled: a
    stopped owner's turn dies with its process, so in production the gate
    behind the card goes too. Here the owner is in-process and its gate
    coroutine survives, which is a property of the rig. What generalises is
    the LATCH: any source that reaches ``display_only`` with a reconnect that
    cannot complete needs a route back that does not depend on the latch, and
    the un-bindable stop is the shape reachable end to end today.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    # `viewer=False`: the boot contract `lop` itself builds, and the one on which
    # a deliberate stop is terminal. See the docstring's step 2.
    app = _app(rig, "alpha", viewer=False)
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
                    assert (
                        not source.display_only
                    ), "alpha was already display_only before the scenario started"

                    # Step 1: leave A. The card goes, the gates detach.
                    await _click(app, pilot, "beta")
                    assert alpha._gates_detached, "the switch away did not detach alpha's gates"
                    assert app._ask_screen is None, "the outgoing card is still on screen"

                    # Step 2: A's owner is stopped deliberately, announced on the
                    # wire exactly as `lop stop` announces it.
                    rig.servers["alpha"].announce_stop()
                    await _pump(pilot, 10)
                    await rig.servers["alpha"].aclose()
                    cold = await _pump_until(pilot, lambda: alpha.is_cold, tries=600)
                    assert cold, (
                        "the stop never made the viewer cold, so the un-bindable premise "
                        "does not hold" + _state(app, alpha, source, probe)
                    )
                    assert not alpha.can_ever_bind, (
                        "the facade can still bind, so the self-heal at app.py:7973 is "
                        "available and this is not the case under test"
                        + _state(app, alpha, source, probe)
                    )

                    # Step 3: the user clicks back onto A.
                    marker = probe.marker
                    await _click(app, pilot, "alpha", rounds=20)
                    resurfaced = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )

                    # Step 4/5: and it is PERMANENT, not transient. Two further
                    # ordinary round trips, each given a full settle, so a pass
                    # cannot mean "it was going to arrive eventually".
                    for _ in range(2):
                        await _click(app, pilot, "beta")
                        await _click(app, pilot, "alpha", rounds=20)
                        resurfaced = resurfaced or await _pump_until(
                            pilot, lambda: app._ask_screen is not None, tries=200
                        )

                    diagnosis = (
                        _format(probe.since(marker), "RETURN LEGS onto an un-bindable owner")
                        + _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                        + f"\n  alpha.pending_gate is not None = {alpha.pending_gate is not None!r}"
                    )
                    _dump(diagnosis)

                    # The premise of the whole complaint: the owner still has an
                    # unanswered question on its books.
                    assert alpha.pending_gate is not None, (
                        "the pending gate left canonical state, so there is no question "
                        "left to surface and this test proves nothing" + diagnosis
                    )
                    assert source.display_only, (
                        "display_only never latched, so the guard this case exists for "
                        "was not exercised" + diagnosis
                    )

                    assert resurfaced, (
                        "REPRODUCED (D): the user returns to a session that still has an "
                        "unanswered gate on the owner's books, and the card is GONE — "
                        "permanently, across three separate returns. display_only is "
                        "latched True with no reconnect that can ever clear it "
                        "(can_ever_bind is False), so the commit skips "
                        "resume_viewer_gates() at app.py:7537 AND the second re-arm at "
                        "app.py:6645 is skipped for the same reason. _gates_detached "
                        "stays True and guard G4 drops every attempt. Nothing on "
                        "screen mentions the pending question." + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- E: THE ANSWERED-KEY LATCH AGAINST A SECOND QUESTION ---------------------


@pytest.mark.asyncio
async def test_a_second_question_in_the_same_ask_resurfaces_across_a_switch(
    tmp_path, monkeypatch
) -> None:
    """NEGATIVE RESULT for G3, probed where the identity actually advances.

    THE CONCERN. ``_maybe_start_gate``'s third guard (attached.py:5595) drops a
    pending gate whose identity equals ``_gate_answered_key``, and
    ``_gate_identity`` is ``(kind, request_id, question_index)`` — with
    ``question_index`` carried end to end precisely because one ask REQUEST can
    hold several questions (attached.py:5566-5574). A latch that compared too
    coarsely would drop question two after question one was answered, and a
    switch in between is exactly the event that would expose it: the answered
    key is set by ``_run_ask`` (attached.py:5679) on the way out of Q1, and the
    re-arm on the return reads it.

    THE PROBE. A genuine two-question ask, answered one question at a time by
    the owner's own gate (``serving.py``'s ``ask_gate`` loops over the questions,
    pushing each as its own ``PendingRequest`` with ``question_index=index``).
    Q1 survives a round trip, is ANSWERED on screen with a real keypress, and
    then Q2 — a new request id AND a new index — survives a second round trip.

    IT DOES NOT REPRODUCE, and the readout says why precisely: this owner mints
    a FRESH ``request_id`` per question, so Q2's identity differs in two of
    three components and G3 cannot match it. The measured
    ``_gate_answered_key`` on the second trip is ``None`` — the key is set on
    the facade that answered, and the answer went out on a bridge that had
    already been replaced. Worth pinning anyway: it is the guard most likely to
    start matching too coarsely if ``_gate_identity`` is ever simplified.
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

                gate_task = await _raise_ask(
                    rig,
                    "alpha",
                    AskQuestion(
                        id="first",
                        question="Question one",
                        options=[AskOption(label="A1"), AskOption(label="B1")],
                    ),
                    AskQuestion(
                        id="second",
                        question="Question two",
                        options=[AskOption(label="A2"), AskOption(label="B2")],
                    ),
                )
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the first question never mounted"
                    first = alpha.pending_gate
                    assert (
                        first is not None and first.question_index == 0
                    ), f"the owner did not project question one first: {first!r}"

                    # Trip one, with Q1 up.
                    await _click(app, pilot, "beta")
                    await _click(app, pilot, "alpha")
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    ), (
                        "question one did not come back; that is the CONTROL case and it "
                        "must hold before the second question means anything"
                        + _state(app, alpha, source, probe)
                    )

                    # ANSWER Q1 for real, on screen, with the keyboard.
                    await pilot.press("enter")
                    advanced = await _pump_until(
                        pilot,
                        lambda: alpha.pending_gate is not None
                        and alpha.pending_gate.question_index == 1,
                        tries=_RESURFACE_TURNS,
                    )
                    second = alpha.pending_gate
                    assert advanced, (
                        "answering question one never advanced the owner to question two, "
                        f"so the index latch is untested: pending={second!r}"
                        + _state(app, alpha, source, probe)
                    )
                    assert second is not None
                    assert (
                        "ask",
                        second.request_id,
                        second.question_index,
                    ) != ("ask", first.request_id, first.question_index), (
                        "question two carries the SAME gate identity as question one; G3 "
                        "would drop it by construction" + _state(app, alpha, source, probe)
                    )

                    # Trip two, with Q2 up. This is the leg G3 could break.
                    marker = probe.marker
                    await _click(app, pilot, "beta")
                    await _click(app, pilot, "alpha")
                    resurfaced = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )
                    diagnosis = (
                        _format(probe.since(marker), "TRIP 2, question two up")
                        + _state(app, alpha, source, probe)
                        + f"\n  question one identity = {('ask', first.request_id, 0)!r}"
                        + f"\n  question two identity = "
                        f"{('ask', second.request_id, second.question_index)!r}"
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert not gate_task.done(), (
                        "the ask resolved without the user answering question two" + diagnosis
                    )
                    assert resurfaced, (
                        "REPRODUCED (E): the SECOND question of an ask did not come back "
                        "after a switch, while the first one did. Check whether "
                        "_gate_answered_key (G3) matched an identity it should not "
                        "have." + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()
