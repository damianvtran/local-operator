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
  latch never clears, and it is offered NO card — the band's stopped-session
  verdict is the honest surface, and an answer given on the card could not
  reach an owner (UX round 1, U1). See
  :func:`test_a_source_that_can_never_bind_is_offered_no_gate_card`.
* **E — the answered-key latch against a second question.** The second question
  of one ask still resurfaces. See
  :func:`test_a_second_question_in_the_same_ask_resurfaces_across_a_switch`.
* **F — an answer that never reached the owner.** The receipt is lifted and the
  operator is told once. See
  :func:`test_an_answer_that_never_reached_the_owner_writes_no_receipt`.
* **G — the same stop on a SIDEBAR-LEASED source.** The viewer contract carries
  ``_can_go_cold``, so ``can_ever_bind`` is true on every switched-to session
  and the first version of this guard could never fire there. See
  :func:`test_a_sidebar_leased_source_whose_owner_was_stopped_offers_no_card`.
* **H — a stopped session that comes back.** The stop term must lift on the
  successor's sync, or the guard would refuse a live question. See
  :func:`test_a_session_restarted_after_its_stop_still_surfaces_a_new_gate`.
* **I — the band over a returned card.** It yields to the card, in the muted
  ink, and stops instructing once the card is answered. See
  :func:`test_the_band_yields_to_a_returned_card_and_stops_when_it_is_answered`.
* **J — the undelivered channel's discriminators.** A refusal and a superseded
  race are not delivery failures, and a retraction is tied to the reply that
  failed. See the three ``test_a_refused_answer_is_not_reported_as_undelivered``
  / ``…_superseded_race…`` / ``…_retracted_receipt…`` cases.
* **K — the undelivered notice's copy and its retirement.** One row at 60
  columns, and gone when the card is answerable again. See
  :func:`test_the_undelivered_notice_fits_one_row_and_is_retired`.

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
from local_operator.session.errors import OperatorAuthorityRequired
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.approval import ApprovalBlock
from local_operator.tui.widgets.session_sidebar import SessionSidebar
from local_operator.tui.widgets.transcript import NoticeBlock
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


def _spy_undelivered(session: AttachedSession) -> list[Any]:
    """Record every report on the session's undelivered channel, then forward it.

    WHY A SPY AND NOT THE TRANSCRIPT. The notice this channel writes is RETIRED
    the moment an answerable card is back for the source (UX round 2, U5), and a
    refused or superseded approval re-offers its card at once — so a wrong report
    can be written and taken down again before a test samples the screen. The
    channel itself is the fact under test; the screen is only its symptom.
    """
    calls: list[Any] = []
    forward = session._gate_undelivered_handler

    def spy(*args: Any) -> None:
        calls.append(args)
        if forward is not None:
            forward(*args)

    session._gate_undelivered_handler = spy
    return calls


class _Rig:
    """Real in-process runtimes plus the app that views them.

    One object because every case needs the same four things wired together and
    torn down in the same order: the runtimes, the discovery patch that makes
    only THOSE runtimes findable, the app booted onto one of them, and the
    handles that raise gates on the owner side.
    """

    def __init__(self, config: Path, operator_cap: bytes | None = None) -> None:
        self.config = config
        #: The spawner's capability (the ``operator_cap`` fixture), for the one
        #: case that needs an ALLOW to take effect on the owner. Without it an
        #: in-process owner refuses every Allow (#1310), which is the default
        #: this file wants everywhere else: a refusal is its own pinned outcome,
        #: and a case that silently needs delivery would otherwise pass on it.
        self.operator_cap = operator_cap
        self.servers: dict[str, RuntimeServer] = {}
        self.handles: dict[str, ServingSessionHandle] = {}

    async def runtime(self, session_id: str) -> None:
        directory = self.config / "sessions" / session_id
        await seed_transcript(directory, _rows())
        owner = build_session(directory, ScriptedStream([]), cwd=self.config)
        handle = ServingSessionHandle(owner, asyncio.get_running_loop(), cwd=str(self.config))
        server = RuntimeServer(handle, kind="daemon", operator_cap=self.operator_cap)
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


async def _rig(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    *names: str,
    operator_cap: bytes | None = None,
) -> _Rig:
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

    rig = _Rig(config, operator_cap)
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


# --- D: A SOURCE WHOSE RECONNECT CAN NEVER COMPLETE ---------------------------


@pytest.mark.asyncio
async def test_a_source_that_can_never_bind_is_offered_no_gate_card(tmp_path, monkeypatch) -> None:
    """An un-bindable source is offered NO card; the band is the honest surface.

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
       ``display_only`` (``or session.is_cold``). The commit's
       ``resume_viewer_gates()`` and the settled-navigation re-arm BOTH still
       run — ``test_a_display_only_commit_rearms_its_gate_bridge`` and the
       sibling file's ``test_the_navigation_settled_rearm_heals_a_display_only_
       source`` defend that, and both must keep running: a ``display_only``
       source whose viewer CAN still bind is the shape this PR exists to heal.
       What the ladder does with this one is different, and that is the
       regression this case now guards: ``can_ever_bind`` is False, so
       ``_maybe_start_gate`` drops at G6 and no bridge starts at all.
    4. The card is therefore ABSENT on the first return and on every later one,
       and the band carries the stopped-session verdict — the surface whose own
       next step (``/resume <id>``) is runnable from the state it names.

    WHY WITHHELD RATHER THAN ANSWERABLE. This test used to assert the opposite,
    and the reversal is the fix rather than a change of mind (UX review round 1,
    U1). The card is mounted from the viewer's STALE ``pending_gate``, so an
    answer given on it is posted to an owner that is gone: measured, an ask
    un-mounted silently with no receipt and the gate still unresolved, and an
    approval wrote a FALSE ``✓ allowed`` receipt for a decision that reached
    nobody. The old docstring's own note carried the reason already — "a stopped
    owner's turn dies with its process, so in production the gate behind the card
    goes too" — which is exactly the state in which the card must not be offered.
    What generalises from the rig is the 1:1 of card and gate: on this source
    there is no gate left for a card to answer.

    WHAT THIS DOES AND DOES NOT CLAIM. The un-bindable stop is the reachable
    shape of ``can_ever_bind`` False today, and this rig's in-process owner
    keeps the gate coroutine alive past the stop, which a real stop would not.
    Both are stated: the card's ABSENCE here is decided by the viewer's
    predicate and its latched ``pending_gate``, neither of which depends on the
    rig's surviving coroutine.
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

                    # Step 3: the user clicks back onto A. The commit's
                    # `resume_viewer_gates()` and the settled-navigation re-arm
                    # both run (they are not gated on `display_only`), and the
                    # ladder then refuses to start a bridge because the viewer
                    # can never bind.
                    marker = probe.marker
                    await _click(app, pilot, "alpha", rounds=20)
                    offered = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )

                    # Step 4: PERMANENT, not transient. Two further ordinary
                    # round trips, each given a full settle, so a pass cannot
                    # mean "it was going to arrive eventually".
                    for _ in range(2):
                        await _click(app, pilot, "beta")
                        await _click(app, pilot, "alpha", rounds=20)
                        offered = offered or await _pump_until(
                            pilot, lambda: app._ask_screen is not None, tries=200
                        )

                    diagnosis = (
                        _format(probe.since(marker), "RETURN LEGS onto an un-bindable owner")
                        + _state(app, alpha, source, probe)
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                        + f"\n  alpha.pending_gate is not None = {alpha.pending_gate is not None!r}"
                        + f"\n  alpha.can_ever_bind = {alpha.can_ever_bind!r}"
                        + (
                            "\n  band = "
                            f"{app._status._connection if app._status is not None else None!r}"
                        )
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
                    # THE ROUTE WAS TAKEN. `_gates_detached` is cleared by the
                    # commit's `resume_viewer_gates()`, so a False here proves
                    # the refusal is the LADDER's (G6) and not the latch's (G4):
                    # a rig whose latch was still set would drop at G4 and this
                    # test would be measuring the old bug rather than the fix.
                    assert not alpha._gates_detached, (
                        "the commit did not even clear `_gates_detached` for this "
                        "display_only source, so the ladder was never reached and "
                        "this test is not measuring the G6 drop" + diagnosis
                    )
                    assert not offered, (
                        "OFFERED (D): a session whose owner can never take an answer "
                        "was offered an answerable card, on the first return or a "
                        "later one. Pressing a key on it discards the answer in "
                        "silence for an ask and writes a FALSE `✓ allowed` receipt "
                        "for an approval (UX round 1, U1); the card's own gate was "
                        "mounted from the viewer's stale `pending_gate`, and no "
                        "owner on the other end can take it. `can_ever_bind` is "
                        "False for the life of the process, so guard G6 drops every "
                        "attempt — measured by the readout above." + diagnosis
                    )
                    # THE BAND IS THE HONEST SURFACE, and it is the only one left:
                    # it has to name the state and its runnable next step.
                    band = app._status._connection if app._status is not None else None
                    assert band is not None and "was stopped" in band and "/resume alpha" in band, (
                        f"the band is {band!r} rather than the stopped-session "
                        "verdict this source's own state names" + diagnosis
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


# --- F: AN ANSWER THAT NEVER REACHED THE OWNER -------------------------------


@pytest.mark.asyncio
async def test_an_answer_that_never_reached_the_owner_writes_no_receipt(
    tmp_path, monkeypatch
) -> None:
    """A settled card whose answer did not land leaves no receipt, and says so once.

    THE SHAPE, and it is the one the card cannot know from the answer alone. An
    approval card is up and answerable; the operator presses ``y``; the POST that
    carries the answer to the owner fails. The card resolves on the KEYPRESS and
    the transcript receipt is written in the same breath (``request_tool_approval``
    appends ``ApprovalBlock.receipt`` in its ``finally``), while
    ``AttachedSession._run_approval`` posts the answer one await LATER — so by the
    time the failure is known there is a ``✓ allowed write Save one record`` row
    on screen for a decision that reached nobody. Nothing downstream can revise a
    transcript row, so the correction has to reach back to the block:
    ``TranscriptView.remove_block``, the seam ``/clear`` and the boot hint already
    use.

    TWO HALVES, BOTH ASSERTED, because either alone is a lie: the false record is
    GONE, and the operator is TOLD — once — in the sentence that names the
    outcome (``_gate_reply_undelivered_text``).

    WHAT IS FAKED, AND WHAT IS NOT. The runtime, the owner, its gate, the card,
    the keystroke, the swallow arms and the notice are all real. Only the
    transport's answer is, because a socket dying exactly between two awaits
    cannot be scheduled from a test: it is raised from ``client.approval_answer``,
    which is precisely the boundary the product itself treats as fallible —
    ``ConnectionError`` out of it is swallowed two frames up by design.

    WHAT THIS DOES NOT PIN. Whether a card is offered AGAIN after the notice. The
    gate is still unanswered, so the ladder may start a second bridge — correct
    while the viewer can still bind (the question is open, and the operator should
    be able to answer it once the session is back), and refused one level down by
    the ladder's G6 on ``can_ever_bind`` for a session whose owner is gone
    (``test_a_source_that_can_never_bind_is_offered_no_gate_card``). This rig fakes
    only the transport, so its owner still looks perfectly bindable and a second
    card is expected here. The receipt, the sentence and the unanswered gate are
    what must hold either way, and those are what this asserts.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
                assert await _pump_until(
                    pilot, lambda: app._session is not None, tries=300
                ), "the app never adopted a session; the rig never reached its premise"
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)
                client = alpha._client
                assert client is not None, "the premise is a bound facade with a live owner"

                gate_task = asyncio.create_task(
                    rig.handles["alpha"]._approval_gate("write", "Save one record")
                )
                #: The posts the app attempted. One entry is how this test knows the
                #: keypress was TAKEN and a delivery attempted — the fact the whole case
                #: turns on, and one the card's own presence cannot tell us (see the
                #: docstring: whether a card is offered again afterwards depends on
                #: whether the viewer can still bind, which this rig leaves alone).
                posted: list[Any] = []
                try:
                    # Mounted AND FOCUSED: the premise is a card the operator can
                    # answer, and a `y` delivered while the dock card does not hold
                    # focus goes to the composer instead and answers nothing.
                    # Asserted rather than assumed because the two are separate
                    # events in the pump.
                    assert await _pump_until(
                        pilot,
                        lambda: app._approval is not None and app.focused is app._approval,
                        tries=300,
                    ), "the approval card never mounted with focus; the premise is unmet"

                    async def dead_socket(*args: Any, **_kwargs: Any) -> Any:
                        posted.append(args[0] if args else None)
                        raise ConnectionError("the owner's socket is gone")

                    monkeypatch.setattr(client, "approval_answer", dead_socket)

                    await pilot.press("y")
                    # The post happens one await after the settle, and the notice is
                    # scheduled with `call_later`, so the correction lands on a later
                    # turn than the keypress.
                    await _pump(pilot, 40)

                    blocks = app._transcript_view().blocks()
                    receipts = [b for b in blocks if isinstance(b, ApprovalBlock)]
                    notices = [b.text() for b in blocks if isinstance(b, NoticeBlock) and b.text()]
                    undelivered = [text for text in notices if "Answer not delivered" in text]
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  transcript notices = {notices!r}"
                        + f"\n  approval receipts on the transcript = {len(receipts)!r}"
                        + f"\n  posts attempted = {len(posted)!r}"
                        + f"\n  owner gate done = {gate_task.done()!r}"
                        + f"\n  app.focused = {app.focused!r}"
                    )
                    _dump(diagnosis)

                    # The premise: the key was taken, so a delivery WAS attempted.
                    # Without it this case could pass by the card never having been
                    # answerable at all.
                    assert len(posted) == 1, (
                        "the answer was never posted, so there was no undelivered reply "
                        "to report and this test proves nothing" + diagnosis
                    )
                    assert not gate_task.done(), (
                        "the owner's gate resolved, so the answer somehow arrived and "
                        "there was nothing undelivered to report" + diagnosis
                    )
                    assert not receipts, (
                        "a `✓ allowed` receipt survived an answer that never reached the "
                        "owner: the transcript claims a call was authorised that nobody "
                        'received, and it is the strongest "it worked" affordance the '
                        "row has" + diagnosis
                    )
                    assert len(undelivered) == 1, (
                        "an undelivered answer must be said exactly once, in the "
                        "unavailable-until-connected register" + diagnosis
                    )
                    assert "not connected" in undelivered[0] or "was stopped" in undelivered[0], (
                        "the notice does not name the outcome, so the operator is told "
                        "nothing about what happened to the answer they gave" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- G: THE SAME STOP ON A SIDEBAR-LEASED SOURCE (the viewer contract) --------


@pytest.mark.asyncio
async def test_a_sidebar_leased_source_whose_owner_was_stopped_offers_no_card(
    tmp_path, monkeypatch
) -> None:
    """Condition G: the stop gate has to hold on the VIEWER contract too.

    WHY D IS NOT ENOUGH. D drives the facade ``lop`` itself builds
    (``viewer=False``), whose deliberate-stop arm leaves ``can_ever_bind`` False
    for the life of the process — so G6 could refuse it on that term alone.
    EVERY sidebar lease is the other contract: ``_lease_sidebar_source`` builds
    ``_can_go_cold = True`` facades, and that flag is one of ``can_ever_bind``'s
    own disjuncts, so on the switched-to sessions the predicate had no false term
    at all and the guard that exists to withhold this card could never fire.
    Measured before this case existed, on this rig (agent review round 3, A9 = QA
    Q2, reproduced independently in both streams): the card mounted AND focused
    over ``Saved · This session was stopped; /resume beta reopens it``, was
    re-offered on every visit, and each answer produced an "undelivered" notice
    for a question no owner could ever receive.

    THE SESSION UNDER TEST IS BETA, not the boot session, and that is what makes
    the facade a sidebar lease. It is also the shape the PR's own defect
    statement is about: raise a question, switch away, come back.

    THE PREMISES ARE ASSERTED, because either one failing turns this case into a
    restatement of D: ``beta._can_go_cold`` True (the viewer contract) and
    ``beta.can_ever_bind`` True after the stop (so the refusal cannot be the
    first term), alongside the stale ``pending_gate`` that would otherwise mount
    the card.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                app._set_approve_all(False)

                # Step 1: lease BETA through the real sidebar, which is what
                # builds the viewer-contract facade this case is about.
                await _click(app, pilot, "beta")
                beta, source = _attached(app._session), app._interaction
                assert getattr(app._session, "session_id", None) == "beta", (
                    "the switch never landed on beta, so this case is not driving a "
                    "sidebar lease at all"
                )
                assert beta._can_go_cold is True, (
                    "beta was not leased through `_lease_sidebar_source`, so its facade "
                    "carries the boot contract and this case duplicates D"
                )
                probe = _install_probe(monkeypatch, beta)

                gate_task = await _raise_ask(rig, "beta", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the ask card never mounted on beta; the premise is unmet"
                    assert not source.display_only, "beta was already display_only"

                    # Step 2: leave, and stop beta's owner while the user is away.
                    await _click(app, pilot, "alpha")
                    assert beta._gates_detached, "the switch away did not detach beta's gates"
                    assert app._ask_screen is None, "the outgoing card is still on screen"

                    rig.servers["beta"].announce_stop()
                    await _pump(pilot, 10)
                    await rig.servers["beta"].aclose()
                    assert await _pump_until(
                        pilot, lambda: beta.is_cold, tries=600
                    ), "the stop never made the viewer cold"

                    diagnosis = (
                        _state(app, beta, source, probe)
                        + f"\n  beta._deliberate_stop = {beta._deliberate_stop!r}"
                        + f"\n  beta.is_cold = {beta.is_cold!r}"
                        + f"\n  beta.pending_gate is not None = "
                        f"{beta.pending_gate is not None!r}"
                    )
                    _dump(diagnosis)

                    assert beta.pending_gate is not None, (
                        "the viewer's gate went away with the stop, so no card could be "
                        "offered and this case would pass without the guard" + diagnosis
                    )
                    assert beta.can_ever_bind, (
                        "beta reports it can never bind, so the refusal would come from "
                        "G6's first term and the sidebar term would go untested" + diagnosis
                    )
                    assert beta._deliberate_stop, (
                        "the viewer was never told the session was stopped, so the guard "
                        "has nothing to refuse on and the card will be offered" + diagnosis
                    )

                    # Step 3: back to beta, twice, each given a full settle.
                    marker = probe.marker
                    await _click(app, pilot, "beta", rounds=20)
                    offered = await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )
                    await _click(app, pilot, "alpha")
                    await _click(app, pilot, "beta", rounds=20)
                    offered = offered or await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                    )

                    diagnosis = (
                        _format(probe.since(marker), "RETURN LEGS onto a stopped sidebar lease")
                        + _state(app, beta, source, probe)
                        + f"\n  beta._deliberate_stop = {beta._deliberate_stop!r}"
                        + f"\n  beta.can_ever_bind = {beta.can_ever_bind!r}"
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                        + "\n  band = "
                        f"{app._status._connection if app._status is not None else None!r}"
                    )
                    _dump(diagnosis)

                    assert not beta._gates_detached, (
                        "the commit did not clear `_gates_detached`, so the ladder was "
                        "never reached and this case is not measuring G6" + diagnosis
                    )
                    assert not offered, (
                        "OFFERED (G): a sidebar-leased session whose owner was stopped was "
                        "offered an answerable card. `can_ever_bind` is True for every "
                        "sidebar lease, so this is the term G6 needs — refusing here is "
                        "what makes the withheld card the fix rather than an apology for "
                        "it." + diagnosis
                    )
                    band = app._status._connection if app._status is not None else None
                    assert band is not None and "was stopped" in band and "/resume beta" in band, (
                        f"the band is {band!r} rather than the stopped-session verdict "
                        "this source's own state names" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- H: A STOPPED SESSION THAT COMES BACK ------------------------------------


@pytest.mark.asyncio
async def test_a_session_restarted_after_its_stop_still_surfaces_a_new_gate(
    tmp_path, monkeypatch
) -> None:
    """Condition H: the guard must not cost a session that someone brings back.

    THE OTHER SIDE OF G, and the thing the guard's second term could get wrong.
    A stop ends the TURN — and the parked gate with it, which is why withholding
    the card is honest — but it does not end the SESSION. Another TUI, a shell
    turn, ``--resume``, or a fresh ``lop`` on the same id all bring the session
    back, and when they do, the viewer binds to that successor and
    ``_finish_sync`` clears ``_deliberate_stop``. From that moment the term G6
    reads is false again, and the next gate the successor raises has to mount and
    be answerable — a guard that withheld it would be worse than the bug it
    fixes, because there would be no way to answer a live question.

    DRIVEN THROUGH THE SAME ROUTE AS G: beta is leased by the sidebar, its owner
    is stopped while the user is away, and the card is withheld. The owner is
    then RESTARTED in-process for the same session id, the user returns, and a
    FRESH ask raised by the successor is answered on screen with a real keypress
    — asserted by the owner's own gate returning the answer, so a card that
    merely mounted and swallowed the keystroke cannot pass.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                app._set_approve_all(False)

                await _click(app, pilot, "beta")
                beta, source = _attached(app._session), app._interaction
                probe = _install_probe(monkeypatch, beta)
                first = await _raise_ask(rig, "beta", _one_question())
                try:
                    assert await _pump_until(
                        pilot, lambda: app._ask_screen is not None, tries=300
                    ), "the first ask never mounted"

                    # The stop, while the user is on alpha.
                    await _click(app, pilot, "alpha")
                    rig.servers["beta"].announce_stop()
                    await _pump(pilot, 10)
                    await rig.servers["beta"].aclose()
                    assert await _pump_until(pilot, lambda: beta.is_cold, tries=600)
                    assert beta._deliberate_stop, (
                        "the stop was not recorded on the viewer, so this case is not "
                        "exercising the term it exists for"
                    )

                    # The owner comes back: a new runtime for the same session id,
                    # which is what `lop --resume` and a fresh turn both produce.
                    await rig.runtime("beta")
                    assert not first.done()

                    await _click(app, pilot, "alpha")
                    await _click(app, pilot, "beta", rounds=20)
                    rebound = await _pump_until(
                        pilot,
                        lambda: not beta.is_cold and not beta._deliberate_stop,
                        tries=_RESURFACE_TURNS,
                    )
                    diagnosis = (
                        _state(app, beta, source, probe)
                        + f"\n  beta._deliberate_stop = {beta._deliberate_stop!r}"
                        + f"\n  beta.is_cold = {beta.is_cold!r}"
                        + f"\n  beta.can_ever_bind = {beta.can_ever_bind!r}"
                        + "\n  band = "
                        f"{app._status._connection if app._status is not None else None!r}"
                    )
                    _dump(diagnosis)
                    assert rebound, (
                        "the viewer never bound to the restarted owner, so the claim that "
                        "the stop term lifts on a successful sync is untested" + diagnosis
                    )

                    # A FRESH gate from the successor has to come back and be
                    # answerable. The stopped turn's gate died with it, so this is
                    # the one route on which a card is owed.
                    second = await _raise_ask(rig, "beta", _one_question())
                    try:
                        assert await _pump_until(
                            pilot, lambda: app._ask_screen is not None, tries=_RESURFACE_TURNS
                        ), (
                            "REFUSED A RECOVERABLE SESSION: the restarted owner raised a "
                            "question and no card was offered, so the stop term is still "
                            "suppressing a card that can be answered" + diagnosis
                        )
                        await pilot.press("enter")
                        delivered = await asyncio.wait_for(second, 30)
                        assert delivered == {"destination": ["Here"]}, (
                            "the successor's gate did not receive the answer given on "
                            f"screen; it returned {delivered!r}" + diagnosis
                        )
                    finally:
                        second.cancel()
                        await asyncio.gather(second, return_exceptions=True)
                finally:
                    first.cancel()
                    await asyncio.gather(first, return_exceptions=True)
    finally:
        await rig.dispose()


# --- I: THE BAND OVER A RETURNED, THEN ANSWERED, CARD ------------------------


@pytest.mark.asyncio
async def test_the_band_yields_to_a_returned_card_and_stops_when_it_is_answered(
    tmp_path, monkeypatch
) -> None:
    """Conditions I: D1's band copy, D7's ink, and D5's half nobody finished.

    THE STATE is D1's: a session whose viewer is not ready for events (a display
    resync in flight, so ``display_only`` is latched and ``is_cold`` is true
    through its third disjunct) and whose connect has ended in a verdict — the
    band's own card-less sentence, ``Select again to retry``. Two things then
    have to hold, and the second is the half the last round left open (design
    round 2, D5 = QA Q6):

    * while the card is back and answerable, the band must NOT go on offering a
      retry, and it must not carry it in the alarm ink either: `Answer the
      question above` is an instruction, and painting it red under a card whose
      own highlight is the accent green reads as "something is wrong with this
      question" (D7).
    * once the card is ANSWERED the band must stop saying that, because there is
      nothing above it to answer — measured still saying it at +20 s with the
      composer contradicting it in the line below.

    HOW THE STATE IS REACHED, and one deliberate shortcut. The held
    ``frontend_sync`` and the invalidated display history are real, and the
    click back onto the session is the real route. What is set directly is
    ``connection_error``, because the honest route to that verdict is
    ``_await_sidebar_frame``'s 15 s timer (measured at 16.5 s in the round-2
    frames) and what THIS case pins is the band's answer to the state, not the
    clock that produces it — which
    ``test_a_display_only_frame_whose_gate_cannot_be_presented_still_paints``
    already drives for real. The band is read on the frame the card arrives,
    because that is the frame the branch exists for.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", "beta")
    app = _app(rig, "alpha")
    release_sync = asyncio.Event()
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
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
                    assert app._ask_screen is None, "the outgoing card is still on screen"

                    client = alpha._client
                    assert client is not None
                    real_sync = client.frontend_sync

                    async def held_sync(*args: Any, **kwargs: Any) -> Any:
                        await release_sync.wait()
                        return await real_sync(*args, **kwargs)

                    monkeypatch.setattr(client, "frontend_sync", held_sync)
                    alpha._invalidate_display_history()
                    assert await _pump_until(
                        pilot, lambda: not alpha._ready_for_events, tries=100
                    ), "the held resync never cleared _ready_for_events"

                    await _click(app, pilot, "alpha", rounds=20)
                    assert source.display_only, "the returning source never latched display_only"
                    # THE CONNECT THAT IS STILL RUNNING GOES FIRST. It is parked on the
                    # held sync, and left alive its own retry arm clears
                    # `connection_error` back to "" ("Connecting…") on its next round,
                    # which raced the verdict planted below: 2 of 3 runs read the band
                    # as `Connecting…` and never reached the state this case pins.
                    # Cancelled rather than waited out because its cancellation arm
                    # skips both the status write and the re-arm, which is exactly
                    # the state a SPENT connect leaves behind.
                    connect = source.connection_task
                    if connect is not None and not connect.done():
                        connect.cancel()
                        await asyncio.gather(connect, return_exceptions=True)
                    # The verdict a spent connect leaves, without its 15 s clock.
                    source.connection_error = "the connection could not be established"
                    source.can_never_bind = False
                    app._show_sidebar_connection(source)
                    await _pump(pilot, 10)

                    before = app._status._connection if app._status is not None else None
                    assert before is not None and "Select again to retry" in before, (
                        f"the card-less sentence is not on the band ({before!r}), so the "
                        "state this case needs was never reached"
                    )

                    # Read the band on every frame the card is up, so the frame the
                    # branch exists for cannot be missed.
                    seen: list[tuple[str, bool]] = []

                    def card_up() -> bool:
                        if app._ask_screen is None:
                            return False
                        seen.append(
                            (
                                "" if app._status is None else app._status._connection,
                                bool(app._status is not None and app._status._connection_muted),
                            )
                        )
                        return True

                    release_sync.set()
                    assert await _pump_until(pilot, card_up, tries=_RESURFACE_TURNS), (
                        "the card never came back once the viewer was ready, so the band "
                        "state it should have announced was never reached"
                        + _state(app, alpha, source, probe)
                    )
                    await _pump(pilot, 10)

                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  band frames while the card was up = {seen!r}"
                    )
                    _dump(diagnosis)

                    assert any("Answer the question above" in band for band, _ in seen), (
                        "the band kept the connect's card-less sentence over a question "
                        "that is right there and answerable — design round 1's D1" + diagnosis
                    )
                    assert not any(
                        "Answer the question above" in band and not muted for band, muted in seen
                    ), (
                        "the instruction sentence was painted in the connection-failure "
                        "ink, so a card whose own highlight is the accent green is "
                        "announced by a red line (design round 2, D7)" + diagnosis
                    )

                    # Answer it for real, and the band must stop instructing.
                    await pilot.press("enter")
                    answered = await _pump_until(
                        pilot,
                        lambda: app._ask_screen is None and alpha.pending_gate is None,
                        tries=_RESURFACE_TURNS,
                    )
                    assert (
                        answered
                    ), "the returned card did not resolve on a real keypress" + _state(
                        app, alpha, source, probe
                    )
                    await _pump(pilot, 30)
                    after = app._status._connection if app._status is not None else None
                    _dump(_state(app, alpha, source, probe) + f"\n  band after = {after!r}")
                    assert after is not None and "Answer the question above" not in after, (
                        "the band is still pointing the operator at a question that has "
                        "been answered and taken off the screen (design round 2, D5 = QA "
                        f"Q6): band = {after!r}"
                    )
                finally:
                    release_sync.set()
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


# --- J: THE UNDELIVERED CHANNEL'S THREE DISCRIMINATIONS ----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("link_at_refusal", ["up", "dropping"])
async def test_a_refused_answer_is_not_reported_as_undelivered(
    tmp_path, monkeypatch, link_at_refusal
) -> None:
    """Condition J1: the owner's REFUSAL is not a delivery failure (A10 = Q3 = D4).

    ``OperatorAuthorityRequired`` subclasses ``RuntimeError``, so the transport
    clause in ``_run_approval`` matched it and the pane was told its answer was
    never delivered — false twice over: the session is connected, and the owner
    RECEIVED the answer and refused it. The row recording this pane's answer was
    retracted with it, which is precisely what the refusal notice's own
    ``not applied — `` prefix exists to avoid: the row has to RECORD that the
    answer was given here, corrected in its first words, not disappear.

    NOTHING IS FAKED. The rig's owners are built WITHOUT the spawner's operator
    capability (see ``_Rig.operator_cap``), which is exactly the #1310 shape: a
    follower pane whose ``y`` crosses the socket, reaches the owner, and is
    REFUSED there. The refusal therefore comes back through the real
    ``AttachClient.approval_answer`` and the real ``_run_approval`` clauses —
    the end-to-end route agent review round 3 asked for, since the handler-level
    pin in ``test_approvals_ux`` cannot see an arm ordered ahead of it.

    ``link_at_refusal="dropping"`` is the one injected fault, and it is what makes
    the dedicated ``except OperatorAuthorityRequired`` arm load-bearing on its
    own: with the link up, the wire rule beside it (``_gate_reply_reached_the_
    owner``) ALSO reads a connected refusal as "reached", so deleting the arm
    alone would pass. Here the socket is reported down at the instant the
    refusal is classified (restored on the next loop turn) — the owner refused
    and then went away, e.g. stopped right after answering. The owner still
    RECEIVED and RULED on the answer, so it is a refusal whatever the link does
    next; only the explicit arm says so.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)
                reported = _spy_undelivered(alpha)
                client = alpha._client
                assert client is not None

                gate_task = asyncio.create_task(
                    rig.handles["alpha"]._approval_gate("write", "Save one record")
                )
                posted: list[Any] = []
                try:
                    assert await _pump_until(
                        pilot,
                        lambda: app._approval is not None and app.focused is app._approval,
                        tries=300,
                    ), "the approval card never mounted with focus; the premise is unmet"

                    # A RECORDING WRAPPER, not a fake: the real post still runs, and
                    # what it raised is kept so the premise ("the OWNER refused") is
                    # asserted rather than assumed.
                    real_answer = client.approval_answer

                    async def recorded(*args: Any, **kwargs: Any) -> Any:
                        try:
                            return await real_answer(*args, **kwargs)
                        except BaseException as error:
                            posted.append(error)
                            if link_at_refusal == "dropping":
                                client._connected = False
                                asyncio.get_running_loop().call_soon(
                                    setattr, client, "_connected", True
                                )
                            raise

                    monkeypatch.setattr(client, "approval_answer", recorded)
                    await pilot.press("y")
                    assert await _pump_until(
                        pilot,
                        lambda: any(
                            isinstance(b, NoticeBlock) and "not applied" in b.text()
                            for b in app._transcript_view().blocks()
                        ),
                        tries=600,
                    ), "the owner's refusal never reached the screen" + _state(
                        app, alpha, source, probe
                    )
                    await _pump(pilot, 30)

                    blocks = app._transcript_view().blocks()
                    receipts = [b for b in blocks if isinstance(b, ApprovalBlock)]
                    notices = [b.text() for b in blocks if isinstance(b, NoticeBlock) and b.text()]
                    undelivered = [text for text in notices if "Answer not delivered" in text]
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  transcript notices = {notices!r}"
                        + f"\n  approval receipts on the transcript = {len(receipts)!r}"
                        + f"\n  posts attempted = {len(posted)!r}"
                        + f"\n  client.connected = {client.connected!r}"
                        + f"\n  undelivered reports = {reported!r}"
                        + f"\n  gate_task.done() = {gate_task.done()!r}"
                    )
                    _dump(diagnosis)

                    assert len(posted) == 1 and isinstance(posted[0], OperatorAuthorityRequired), (
                        "the real post did not come back as the owner's #1310 refusal, so "
                        f"this case is not measuring it: {posted!r}" + diagnosis
                    )
                    assert client.connected, (
                        "the premise is a CONNECTED pane whose owner refused; a "
                        "disconnected one takes the transport arm legitimately" + diagnosis
                    )
                    assert not undelivered and not reported, (
                        "the pane was told its answer was never delivered when the owner "
                        "had received it and refused it: the sentence is false in both "
                        "clauses and points at the connection instead of the operator key"
                        + diagnosis
                    )
                    assert len(receipts) == 1, (
                        "the receipt recording this pane's answer was retracted; the "
                        "refusal notice's own `not applied —` prefix is what corrects "
                        "that row, and retracting it throws away the record that the "
                        "answer was given here at all" + diagnosis
                    )
                    assert any("not applied" in text for text in notices), (
                        "the refusal's own notice never arrived, so this case cannot show "
                        "which surface owns the outcome" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_a_superseded_race_is_not_reported_as_a_disconnection(tmp_path, monkeypatch) -> None:
    """Condition J2: the first-answer-wins race is not a connection problem (A11 = Q4).

    Another front end settles the owner's gate between this pane's keypress and
    its post, and the owner answers the stale tap with its own verdict ("that
    approval is no longer waiting"). The reply REACHED the owner — it was read
    and ruled on — so the transport clause must not turn it into "Send
    unavailable until connected" on a pane whose socket is up: the operator is
    sent to fix a connection instead of doing nothing, which is what the pre-fix
    head did (the swallow arm's ordinary outcome, silent by design).

    The receipt is left alone here for the same reason the refusal case leaves it
    alone: it records the answer this pane gave, and the arm that owns this
    outcome is the swallow's, not the host's.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)
                reported = _spy_undelivered(alpha)
                client = alpha._client
                assert client is not None

                gate_task = asyncio.create_task(
                    rig.handles["alpha"]._approval_gate("write", "Save one record")
                )
                posted: list[Any] = []
                try:
                    assert await _pump_until(
                        pilot,
                        lambda: app._approval is not None and app.focused is app._approval,
                        tries=300,
                    ), "the approval card never mounted with focus; the premise is unmet"

                    async def raced(*args: Any, **_kwargs: Any) -> Any:
                        posted.append(args[0] if args else None)
                        raise RuntimeError("that approval is no longer waiting")

                    monkeypatch.setattr(client, "approval_answer", raced)
                    await pilot.press("y")
                    await _pump(pilot, 60)

                    blocks = app._transcript_view().blocks()
                    receipts = [b for b in blocks if isinstance(b, ApprovalBlock)]
                    notices = [b.text() for b in blocks if isinstance(b, NoticeBlock) and b.text()]
                    undelivered = [text for text in notices if "Answer not delivered" in text]
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  transcript notices = {notices!r}"
                        + f"\n  approval receipts on the transcript = {len(receipts)!r}"
                        + f"\n  posts attempted = {len(posted)!r}"
                        + f"\n  client.connected = {client.connected!r}"
                        + f"\n  undelivered reports = {reported!r}"
                    )
                    _dump(diagnosis)

                    assert len(posted) == 1, (
                        "the answer was never posted, so this is not the race under test"
                        + diagnosis
                    )
                    assert client.connected, (
                        "the pane is not connected, so the transport arm fires legitimately "
                        "and this case proves nothing" + diagnosis
                    )
                    assert not undelivered and not reported, (
                        "an ordinary stale-answer race on a CONNECTED pane was reported as "
                        "a connection problem: nothing was disconnected, and the operator "
                        "is told to wait for a link that is already up" + diagnosis
                    )
                    assert len(receipts) == 1, (
                        "the row recording this pane's answer was retracted by the "
                        "transport arm, which no longer owns this outcome" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()


@pytest.mark.asyncio
async def test_a_retracted_receipt_belongs_to_the_reply_that_failed(
    tmp_path, monkeypatch, operator_cap
) -> None:
    """Condition J3: only THIS reply's receipt may be taken back (A12 = QA Q5).

    The two facts are not the same one. A reply that failed to land is not always
    the reply that wrote the retained row: an approval answered with NO card at
    all — an allow-all latch, a background approval — writes no receipt, so a
    slot holding "the last receipt" lets an undelivered post delete a DELIVERED
    row belonging to an unrelated gate. The session now reports the gate identity
    with the drop and the host matches it, which is what this case pins: an
    allow-all approval whose post fails must leave the earlier, delivered
    ``✓ allowed`` on the transcript untouched.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha", operator_cap=operator_cap)
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(110, 32)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)
                client = alpha._client
                assert client is not None

                def receipts() -> list[Any]:
                    return [
                        b for b in app._transcript_view().blocks() if isinstance(b, ApprovalBlock)
                    ]

                def notices() -> list[str]:
                    return [
                        b.text()
                        for b in app._transcript_view().blocks()
                        if isinstance(b, NoticeBlock) and b.text()
                    ]

                # Step 1: a DELIVERED approval, with its card, answered for real.
                first = asyncio.create_task(
                    rig.handles["alpha"]._approval_gate("write", "Save one record")
                )
                try:
                    assert await _pump_until(
                        pilot,
                        lambda: app._approval is not None and app.focused is app._approval,
                        tries=300,
                    ), "the first approval card never mounted with focus"
                    await pilot.press("y")
                    assert await _pump_until(
                        pilot, lambda: bool(receipts()) and app._approval is None, tries=300
                    ), (
                        "the delivered approval left no receipt to protect"
                        + _state(app, alpha, source, probe)
                        + f"\n  transcript notices = {notices()!r}"
                        + f"\n  approval pending = {app._approval is not None!r}"
                    )
                    delivered_blocks = receipts()
                    delivered_gate = await asyncio.wait_for(first, 30)
                    assert (
                        delivered_gate is True
                    ), f"the first approval was not delivered to the owner: {delivered_gate!r}"
                finally:
                    first.cancel()
                    await asyncio.gather(first, return_exceptions=True)

                # Step 2: a second approval answered with NO card (the latch) whose
                # post fails. Its own reply wrote no receipt of its own, so nothing
                # may be taken off the transcript for it.
                app._set_approve_all(True)
                second = asyncio.create_task(
                    rig.handles["alpha"]._approval_gate("write", "Save another record")
                )
                try:

                    async def dead_socket(*_args: Any, **_kwargs: Any) -> Any:
                        raise ConnectionError("the owner's socket is gone")

                    monkeypatch.setattr(client, "approval_answer", dead_socket)
                    assert await _pump_until(
                        pilot,
                        lambda: "Answer not delivered" in " ".join(notices()),
                        tries=600,
                    ), (
                        "the card-less approval was never reported as undelivered, so the "
                        "retraction this case is about never ran"
                        + _state(app, alpha, source, probe)
                    )
                    await _pump(pilot, 30)

                    remaining = receipts()
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  receipts before = {len(delivered_blocks)!r}"
                        + f"\n  receipts after = {len(remaining)!r}"
                        + f"\n  notices = {notices()!r}"
                        + f"\n  _last_gate_receipt = {app._last_gate_receipt!r}"
                    )
                    _dump(diagnosis)

                    assert len(remaining) == len(delivered_blocks), (
                        "an undelivered approval that wrote NO receipt of its own removed "
                        "the DELIVERED row belonging to the previous gate: the retraction "
                        "is not tied to the reply that failed" + diagnosis
                    )
                finally:
                    second.cancel()
                    await asyncio.gather(second, return_exceptions=True)
    finally:
        await rig.dispose()


# --- K: THE UNDELIVERED NOTICE'S COPY AND ITS RETIREMENT ---------------------


@pytest.mark.asyncio
async def test_the_undelivered_notice_fits_one_row_and_is_retired(tmp_path, monkeypatch) -> None:
    """Condition K: the sentence has to be readable, and it has to end (D6, U5).

    TWO FACTS ABOUT ONE SURFACE.

    * IT FITS. Under a mounted card at 60 columns the transcript viewport is a
      single row, so the sentence the last round shipped — 57 cells — wrapped and
      the only text on screen was its last fragment, ``connected.``: a truncation
      that reads as the opposite of the message (design round 2, D6). The
      replacement leads with the outcome, and this case asserts the rendered
      block is ONE row rather than a character count, so a copy change that
      overflows fails here rather than in someone's terminal.

    * IT ENDS. The sentence is present-progressive about a connection state, and
      UX round 2 measured it still up at t=33 s with the card back and
      answerable, at t=37.6 s after a message had been sent successfully, and
      idle at t=44.9 s (U5). Retired when an answerable card is on screen for the
      source, which is the frame the claim stops being true on.

    The drop is real: the transport's answer is the one thing faked, from the
    boundary the product treats as fallible.
    """
    rig = await _rig(tmp_path / "config", monkeypatch, "alpha")
    app = _app(rig, "alpha")
    try:
        with patch("local_operator.mobile.attach_client.find_runtime_record", rig.find_owner):
            async with app.run_test(size=(60, 16)) as pilot:
                assert await _pump_until(pilot, lambda: app._session is not None, tries=300)
                alpha, source = _attached(app._session), app._interaction
                app._set_approve_all(False)
                probe = _install_probe(monkeypatch, alpha)
                reported = _spy_undelivered(alpha)
                client = alpha._client
                assert client is not None

                gate_task = asyncio.create_task(
                    rig.handles["alpha"]._approval_gate("write", "Save one record")
                )
                try:
                    assert await _pump_until(
                        pilot,
                        lambda: app._approval is not None and app.focused is app._approval,
                        tries=300,
                    ), "the approval card never mounted with focus; the premise is unmet"

                    # THE LINK GOES DOWN WITH THE POST, not just the post. Faking only
                    # the post left the socket up, so the app — correctly — re-offered
                    # the still-pending gate at once and retired the notice before it
                    # could be measured (1 run in 3 under load). A dead socket is also
                    # a disconnected client, and on this boot facade that is
                    # `can_ever_bind`'s false term, so G6 holds the card back until the
                    # link is restored below: the state the sentence describes.
                    async def dead_socket(*_args: Any, **_kwargs: Any) -> Any:
                        client._connected = False
                        raise ConnectionError("the owner's socket is gone")

                    monkeypatch.setattr(client, "approval_answer", dead_socket)
                    await pilot.press("y")
                    await _pump(pilot, 60)

                    def undelivered_blocks() -> list[Any]:
                        return [
                            b
                            for b in app._transcript_view().blocks()
                            if isinstance(b, NoticeBlock) and "Answer not delivered" in b.text()
                        ]

                    assert await _pump_until(
                        pilot, lambda: bool(undelivered_blocks()), tries=600
                    ), (
                        "no undelivered notice was written, so this case has nothing to "
                        "measure"
                        + _state(app, alpha, source, probe)
                        + f"\n  undelivered reports = {reported!r}"
                        + f"\n  retained notices = {app._gate_reply_notices!r}"
                        + f"\n  approval answered = {getattr(app._approval, 'answered', None)!r}"
                        + f"\n  focused = {app.focused!r}"
                    )
                    block = undelivered_blocks()[0]
                    rows = block.region.height
                    diagnosis = (
                        _state(app, alpha, source, probe)
                        + f"\n  notice text = {block.text()!r}"
                        + f"\n  notice region = {block.region!r}"
                        + f"\n  transcript content_region = "
                        f"{app._transcript_view().content_region!r}"
                    )
                    _dump(diagnosis)
                    assert rows == 1, (
                        "the undelivered notice wraps at 60 columns, so the only text on "
                        "screen under a card is its last fragment (design round 2, D6)" + diagnosis
                    )

                    # TWO ROWS THE RETIREMENT MUST NOT TOUCH, planted beside the real
                    # one: an ordinary warning that this channel never wrote, and an
                    # undelivered row owned by ANOTHER session (a token that is not
                    # alpha's). A card coming back on alpha is evidence about alpha's
                    # link only, so both are still true afterwards.
                    bystander = app._system_notice_block("an unrelated warning", "warning")
                    elsewhere = app._system_notice_block(
                        "Answer not delivered — not connected. (another session)", "warning"
                    )
                    app._gate_reply_notices.append(("some-other-source", elsewhere))

                    # THE LINK IS BACK: the gate is re-offered and the operator can
                    # answer it, which is exactly the state the sentence denies.
                    client._connected = True
                    alpha.resume_viewer_gates()
                    assert await _pump_until(
                        pilot, lambda: app._approval is not None, tries=_RESURFACE_TURNS
                    ), (
                        "the card never came back, so the retirement this case is about "
                        "was never exercised" + _state(app, alpha, source, probe)
                    )
                    await _pump(pilot, 30)
                    left = [b for b in undelivered_blocks() if b is not elsewhere]
                    _dump(_state(app, alpha, source, probe) + f"\n  notices left = {len(left)!r}")
                    assert not left, (
                        "the 'not connected' sentence is still on screen with the card back "
                        "and answerable under it (UX round 2, U5)" + diagnosis
                    )
                    on_screen = app._transcript_view().blocks()
                    assert bystander in on_screen, (
                        "retiring the undelivered row also took down a notice this channel "
                        "never wrote" + diagnosis
                    )
                    assert elsewhere in on_screen and app._gate_reply_notices == [
                        ("some-other-source", elsewhere)
                    ], (
                        "alpha's card coming back retired ANOTHER session's undelivered "
                        "row, whose link nothing has shown to be back" + diagnosis
                    )
                finally:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
    finally:
        await rig.dispose()
