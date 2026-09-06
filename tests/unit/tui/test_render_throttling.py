"""What each animated surface is allowed to cost when nothing has changed.

Every assertion here is a COUNT of invalidation, not a duration. The work these
tests guard was diagnosed on a machine whose wall clock was worthless — a fixed
quantum held its CPU time (81.6 -> 101.5 ms) while wall time inflated 4.6-7.8x
under unrelated load — so a timing assertion here would be a flake generator
that proves nothing. A row that is not rewritten emits no escape sequences and
a tick that posts no ``messages.Layout`` cannot reflow the screen, whatever the
box is doing at the time.

Two invariants run through the whole file and are each pinned by name:

* animation may only ever change RATE, never content; and
* a session that is never told about focus must never end up throttled.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple, cast

import pytest
from textual import messages
from textual.events import AppBlur, AppFocus
from textual.widgets import Static

from local_operator.harness.comms import SubagentComms
from local_operator.session.session import Session
from local_operator.tui import animation
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_panel import SPINNER_INTERVAL_S, SubagentPanel
from local_operator.tui.widgets.subagent_view import SPINNER_FRAMES, SubagentView
from local_operator.tui.widgets.transcript import UserBlock, WorkingBlock
from local_operator.tui.widgets.welcome import WelcomeView

from .test_band_panels import FakeSession, _async_factory, _fake_jobs, _Job


async def _open_running_child(pilot: Any, app: OperatorApp) -> SubagentView:
    """A running child page over a seeded conversation, settled and painted."""
    for _ in range(80):
        await pilot.pause()
        if app._session is not None:
            break
    app._append_block(UserBlock("audit the ingest path"))
    app._refresh_band()
    await pilot.pause()
    app._open_subagent_view("sub-1")
    await pilot.pause()
    await pilot.pause()
    return app.query_one(SubagentView)


def _running_app() -> OperatorApp:
    session = FakeSession()
    session.jobs = _fake_jobs(_Job("sub-1", "audit the ingest path", status="running"))
    return OperatorApp(_async_factory(session))


class _SpinnerTimers(NamedTuple):
    """The three animated surfaces' timer objects, captured at one instant.

    Named rather than a bare tuple because these are only ever compared PER
    SURFACE: a tuple `!=` passes when any single element differs, which is the
    hole review round 1 (R1) and QA round 1 (Q1) both found by mutation. Fields
    make the pairing explicit at the assertion site, so reordering the capture
    cannot silently compare the dock against the band.
    """

    view: Any
    panel: Any
    band: Any


def _running_app_with_running_grandchild() -> OperatorApp:
    """A running child that itself owns a running child, so the DOCK animates.

    ``_running_app`` is enough for any surface the child PAGE owns, but not for
    the subagent panel: ``_subagent_roster`` scopes the dock to the direct
    children of the job whose page is open, so a lone ``sub-1`` leaves the dock
    legitimately empty and its spinner legitimately stopped. A test that wants
    to observe the panel's cadence has to give it something to animate ABOUT,
    or it is asserting against a surface the app is entitled to keep still.

    The lineage is recorded through the real ``SubagentComms`` rather than a
    stub with a ``children`` method, because the roster is resolved from the
    ownership graph and a fake that answers only the one call the current code
    makes would stop being a fixture the moment that resolution changes.
    """
    parent = _Job("sub-1", "audit the ingest path", status="running")
    grandchild = _Job("sub-1a", "read the ingest schema", status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(parent, grandchild)
    comms = SubagentComms(cast(Session, session))
    comms.record_launch(parent.id, parent.label)
    comms.record_launch(grandchild.id, grandchild.label, parent_job_id=parent.id)
    session._subagent_comms = comms
    return OperatorApp(_async_factory(session))


class _UpdateSpy:
    """Records every ``Static.update`` on the page's three chrome rows."""

    def __init__(self, view: SubagentView) -> None:
        self.calls: dict[str, list[tuple[Any, dict[str, Any]]]] = {
            "title": [],
            "breadcrumb": [],
            "rule": [],
        }
        for name, key in (
            ("_title", "title"),
            ("_breadcrumb", "breadcrumb"),
            ("_rule", "rule"),
        ):
            widget: Static = getattr(view, name)
            original = widget.update

            def update(
                renderable: Any = "",
                *,
                _key: str = key,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                self.calls[_key].append((renderable, kwargs))
                return _original(renderable, **kwargs)

            widget.update = update  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_breadcrumb_is_painted_without_a_relayout() -> None:
    """The breadcrumb passes ``layout=False`` like the title and rule beside it.

    It is ``height: 1`` in the sheet exactly as they are, so it cannot move the
    box; the default ``layout=True`` cleared every ancestor's arrangement cache
    and posted ``messages.Layout``, which made the deliberate ``layout=False``
    on its two siblings buy nothing at all — one relayout on the same tick
    reflows the same screen.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        spy = _UpdateSpy(view)
        # Force a full repaint: the memo is keyed on facts that have not moved.
        view._chrome_state = None
        view._breadcrumb_key = None
        view._rule_key = None
        view._paint_chrome()

        assert spy.calls["breadcrumb"], "the breadcrumb was never painted"
        for _renderable, kwargs in spy.calls["breadcrumb"]:
            assert kwargs.get("layout") is False
        # Stated together, because the value of any one of the three is
        # conditional on the other two: one defaulted update relayouts anyway.
        for row in ("title", "rule"):
            for _renderable, kwargs in spy.calls[row]:
                assert kwargs.get("layout") is False


@pytest.mark.asyncio
async def test_a_spinner_tick_repaints_only_the_row_carrying_the_spinner() -> None:
    """The glyph rides the title, so a tick must not touch breadcrumb or rule.

    ``_chrome_state`` carries the spinner and ``_tick`` advances the index
    before calling ``_paint_chrome``, so the memo misses on essentially every
    tick of a running child (measured live: 291 misses in 313 calls over two
    seconds). Before the per-row keys that meant the breadcrumb and the rule
    were rewritten with byte-identical strings 12.5 times a second.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        assert view._running, "fixture must be a RUNNING child or there is no spinner"
        spy = _UpdateSpy(view)

        # Count the ADVANCES, never compare two indices. `_spinner_index` is
        # advanced modulo `len(SPINNER_FRAMES)`, so it is a phase on a cycle
        # and not a monotonic counter: any run whose total tick count is a
        # multiple of the frame count lands back on the frame it started from,
        # and `index != before` then reads a moving spinner as a stopped one.
        # The total is NOT under this test's control — the view's own spinner
        # timer is live and fires during `pilot.pause()`, so the driven ticks
        # below are a lower bound on how many actually happen. That is what
        # made CI fail `assert 3 != 3` on one leg and pass on another, and it
        # is why the phase is observed through a counting wrapper instead.
        advances = 0
        real_tick = view._tick

        def counting_tick() -> None:
            nonlocal advances
            advances += 1
            real_tick()

        view._tick = counting_tick  # type: ignore[method-assign]
        try:
            ticks = len(SPINNER_FRAMES) + 1
            for _ in range(ticks):
                view._tick()
                await pilot.pause()
        finally:
            del view._tick  # type: ignore[attr-defined]

        assert advances >= ticks, "the animation stopped advancing"
        # The cadence is untouched: every tick still repaints the glyph. It is
        # `>=` rather than `==` because the 1 Hz job poll also refreshes the
        # page, and this test is about which ROWS a tick touches, not about
        # owning every paint that happens during it.
        assert len(spy.calls["title"]) >= ticks
        assert spy.calls["breadcrumb"] == []
        assert spy.calls["rule"] == []


@pytest.mark.asyncio
async def test_unchanged_chrome_strings_are_not_rewritten() -> None:
    """Re-painting with identical facts writes nothing to the terminal.

    ``_paint_chrome`` is called from ``show()``, which runs on every child
    event and on the 1 Hz poll, so the common case is a refresh that changes
    nothing — and its own docstring says such a refresh must repaint nothing.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        view._paint_chrome()
        spy = _UpdateSpy(view)

        for _ in range(5):
            view._paint_chrome()

        assert spy.calls["title"] == []
        assert spy.calls["breadcrumb"] == []
        assert spy.calls["rule"] == []


@pytest.mark.asyncio
async def test_a_changed_breadcrumb_still_repaints() -> None:
    """The skip is a memo, not a freeze: new facts must reach the screen."""
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        view._paint_chrome()
        spy = _UpdateSpy(view)

        view._ancestors = ["orchestrator", "reviewer"]
        view._paint_chrome()

        assert len(spy.calls["breadcrumb"]) >= 1
        painted = spy.calls["breadcrumb"][-1][0].plain
        assert "orchestrator" in painted and "reviewer" in painted


@pytest.mark.asyncio
async def test_a_spinner_tick_posts_no_layout_message() -> None:
    """The end-to-end statement of the two fixes above, at the screen.

    Counting ``messages.Layout`` rather than reflows keeps the assertion at the
    boundary the widget actually controls: a Layout message is what clears the
    arrangement caches and makes the compositor re-arrange every widget behind
    the page.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        screen = app.screen
        posted: list[Any] = []
        original_post = screen.post_message

        def post_message(message: Any) -> Any:
            posted.append(message)
            return original_post(message)

        screen.post_message = post_message  # type: ignore[method-assign]
        for _ in range(10):
            view._tick()
            await pilot.pause()

        assert [m for m in posted if isinstance(m, messages.Layout)] == []


# -- focus gating -----------------------------------------------------------


@pytest.mark.asyncio
async def test_blur_slows_every_animated_surface_and_focus_restores_it() -> None:
    """One blur re-rates the page, the panel and the band; focus puts them back.

    The rate is asserted through each surface's own recorded interval rather
    than by timing a real timer: the point is which cadence was requested, and
    a duration assertion under load measures the box, not the app.

    EVERY SURFACE HERE MUST BE GENUINELY ANIMATING FIRST, and that is a
    correctness requirement rather than tidiness. ``SPINNER_INTERVAL_S`` (0.08)
    is BOTH the focused cadence and the value every ``_spinner_rate`` is
    constructed with, so ``rate == SPINNER_INTERVAL_S`` is satisfied by a
    surface that never started a spinner at all. Asserting it alone cannot tell
    "correctly animating at the fast rate" from "never animated", and the blur
    assertions below then fail against an untouched default rather than against
    a rate the fan-out declined to change.

    That is not hypothetical: it is what made this test the repo's dominant
    flake (7 of 11 CI failures over 40 runs, always ``assert 0.08 == 1.0`` at
    the PANEL line while the VIEW line above it passed). The panel was rowless,
    because with a child page open ``_subagent_roster`` scopes the dock to the
    DIRECT CHILDREN of the viewed job and this fixture's ``sub-1`` had none —
    so the app's 1 Hz ``_poll_subagents`` correctly cleared the roster and
    ``_tick`` correctly stopped a timer with nothing left to animate. The old
    ``panel._start_spinner()`` fabricated an animation the app then reclaimed,
    and whether it survived to the assertion depended on whether a poll landed
    in between — a race that only loses under load, which is why it read as a
    timing flake and is not one.

    So the panel is given a running GRANDCHILD through the real
    ``SubagentComms`` graph: a reason to animate that the app's own poll
    re-affirms every tick instead of revoking. Each surface is then pinned by
    TIMER IDENTITY across the transition (AGENTS.md, "prefer a structural
    invariant to a numeric one") — re-rating REPLACES a Textual timer, since
    its interval is fixed at creation, so a surface that was skipped keeps its
    object and a surface that was re-rated does not. That distinguishes the two
    states the bare number cannot, and it holds no matter how loaded the box is.
    """
    app = _running_app_with_running_grandchild()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        panel = app._subagent_panel
        assert panel is not None
        band = app._status
        assert band is not None
        band._streaming = True
        band._sync_spinner_timer()
        await pilot.pause()

        # Liveness BEFORE cadence: see the docstring — the rate assertions
        # under this line are vacuous without it.
        assert view._spinner_timer is not None, "the child page is not animating"
        assert panel._spinner_timer is not None, "the dock has no live spinner to re-rate"
        assert band._spinner_timer is not None, "the band is not animating"
        assert view._spinner_rate == SPINNER_INTERVAL_S
        assert panel._spinner_rate == SPINNER_INTERVAL_S
        assert band._spinner_rate == SPINNER_INTERVAL_S
        blurred_from = _SpinnerTimers(
            view._spinner_timer, panel._spinner_timer, band._spinner_timer
        )

        app._set_animation_focused(False)
        await pilot.pause()

        assert not animation.animation_focused()
        assert view._spinner_rate == animation.BLURRED_SPINNER_INTERVAL_S
        assert panel._spinner_rate == animation.BLURRED_SPINNER_INTERVAL_S
        assert band._spinner_rate == animation.BLURRED_SPINNER_INTERVAL_S
        # Slowed, never stopped: a frozen spinner and a finished job look the
        # same, and this app uses motion as its word for "alive".
        assert view._spinner_timer is not None
        assert panel._spinner_timer is not None
        assert band._spinner_timer is not None
        # The structural half of the claim, asserted PER SURFACE. A re-rate is
        # a REPLACED timer, so an identical object here means that surface was
        # silently skipped even though the number above happened to read
        # correctly — a `_spinner_rate` reporting the new cadence while the
        # timer still fires at the old one is exactly the lying-field shape
        # this test exists to remove, and one assertion per surface is what
        # makes it detectable. Stated three times rather than as one tuple
        # compare: `!=` on a tuple passes when ANY element differs, so two
        # surfaces could keep their timers and it would still go green
        # (review round 1, R1 / QA round 1, Q1 — both mutation-proved it by
        # having the panel record the wanted rate without replacing its timer).
        assert view._spinner_timer is not blurred_from.view, "the child page was not re-rated"
        assert panel._spinner_timer is not blurred_from.panel, "the dock was not re-rated"
        assert band._spinner_timer is not blurred_from.band, "the band was not re-rated"

        focused_from = _SpinnerTimers(
            view._spinner_timer, panel._spinner_timer, band._spinner_timer
        )

        app._set_animation_focused(True)
        await pilot.pause()

        assert view._spinner_rate == SPINNER_INTERVAL_S
        assert panel._spinner_rate == SPINNER_INTERVAL_S
        assert band._spinner_rate == SPINNER_INTERVAL_S
        assert view._spinner_timer is not focused_from.view, "the child page was not restored"
        assert panel._spinner_timer is not focused_from.panel, "the dock was not restored"
        assert band._spinner_timer is not focused_from.band, "the band was not restored"


@pytest.mark.asyncio
async def test_refocus_repaints_so_no_surface_shows_a_stale_frame() -> None:
    """The reduced rate may cost frames; it may never leave stale state.

    The clock, the status and the glyph on the child page are all painted from
    the throttled timer, so returning to the window has to repaint immediately
    rather than wait up to a second for the next slow tick.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        app._set_animation_focused(False)
        await pilot.pause()

        # State moves while the terminal is away. The job's start_time is
        # pinned rather than the painted string left alone, because the 1 Hz
        # job poll recomputes elapsed from the ledger on every fire and would
        # overwrite an injected `_elapsed` with the real duration ("100d+"
        # against the 2023 fixture timestamp) whenever a poll landed inside
        # the window — which is exactly how this test failed one CI leg and
        # passed another. Setting the START makes every recomputation agree
        # with the assertion instead of racing it.
        session = app._session
        assert session is not None
        job = cast(Any, session).jobs.list()[0]
        job.start_time = time.time() - 252.0
        view._elapsed = "4m12s"
        spy = _UpdateSpy(view)
        app._set_animation_focused(True)
        await pilot.pause()

        assert spy.calls["title"], "refocus did not repaint the title"
        assert "4m12s" in spy.calls["title"][-1][0].plain


@pytest.mark.asyncio
async def test_blur_does_not_stop_a_settled_page_from_staying_settled() -> None:
    """Re-rating must never START a timer a surface correctly stopped.

    A settled child has no spinner by design; a naive "restart the timer on
    focus change" would spin a finished job forever.
    """
    session = FakeSession()
    session.jobs = _fake_jobs(_Job("sub-1", "audit the ingest path", status="completed"))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        assert view._spinner_timer is None

        app._set_animation_focused(False)
        await pilot.pause()
        assert view._spinner_timer is None

        app._set_animation_focused(True)
        await pilot.pause()
        assert view._spinner_timer is None


@pytest.mark.asyncio
async def test_a_session_that_never_hears_about_focus_is_never_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is FOCUSED, because some hosts send no focus events at all.

    "We were never told" and "the user is looking at this" have to be the same
    state, or a session on such a host animates at the reduced rate forever.
    """
    assert animation.animation_focused() is True
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open_running_child(pilot, app)
        assert view._spinner_rate == SPINNER_INTERVAL_S
        # The shimmer pin has to come OFF for this line to mean anything. The
        # suite sets LOCAL_OPERATOR_NO_SHIMMER=1 (conftest), under which
        # `motion_enabled()` is False whatever focus says — so the disjunct
        # this assertion used to carry was unconditionally true and the test
        # asserted nothing (agent review R5). Turning motion on isolates the
        # FOCUS gate, which is the one under test here.
        monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)
        assert animation.motion_enabled() is True
        working = WorkingBlock("thinking")
        app._append_block(working)
        await pilot.pause()
        assert working._tick_ms == WorkingBlock._FRAME_MS


@pytest.mark.asyncio
async def test_working_block_falls_to_the_static_cadence_on_blur(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 30 fps shimmer line reuses its EXISTING static mode, not a new one.

    ``_STATIC_FRAME_MS`` already exists for shimmer-off and is already correct
    (frozen glyph, a repaint only when the clock's second changes), so a
    blurred window and a shimmer-off window are one tested state rather than
    two. Measured: 6.8% of a core and ~30 paints/s animated against 3.5% and
    0.8 paints/s static.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        working = WorkingBlock("thinking")
        app._append_block(working)
        # The app fans out to the block it OWNS, which is how a real turn wires
        # it (`_start_working_block`); a block merely appended by a test is not
        # on that path.
        app._working_block = working
        await pilot.pause()
        working._animated = True
        working._tick_ms = WorkingBlock._FRAME_MS

        app._set_animation_focused(False)
        await pilot.pause()
        assert working._tick_ms == WorkingBlock._STATIC_FRAME_MS
        assert working._timer is not None, "throttled, not stopped"

        # Refocus with the shimmer pin OFF, so "restored" is distinguishable
        # from "never changed". Asserting _STATIC_FRAME_MS on both sides of the
        # transition (as this did) passes even if refocus does nothing at all
        # (agent review R5). The AND-ing of the two gates is asserted
        # separately below.
        monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)
        app._set_animation_focused(True)
        await pilot.pause()
        assert working._tick_ms == WorkingBlock._FRAME_MS, "refocus did not restore"

        # And with the shimmer setting off, focus alone must NOT restore
        # motion: the gate is AND-ed rather than replaced.
        monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
        app._set_animation_focused(False)
        await pilot.pause()
        app._set_animation_focused(True)
        await pilot.pause()
        assert working._tick_ms == WorkingBlock._STATIC_FRAME_MS


@pytest.mark.asyncio
async def test_blur_never_drops_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blurred session still paints everything it is told.

    The throttle touches repaint TIMERS only. Transcript blocks, the ledger and
    the working line's label are all event-driven, so they must land on a
    blurred terminal exactly as they do on a focused one — anything else would
    be a session that quietly loses output while the user is in another window.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._set_animation_focused(False)
        await pilot.pause()

        app._append_block(UserBlock("this arrived while the window was blurred"))
        await pilot.pause()
        rows = [getattr(block, "renderable", "") for block in app._transcript_view().blocks()]
        plain = " ".join(getattr(row, "plain", str(row)) for row in rows)
        assert "this arrived while the window was blurred" in plain


@pytest.mark.asyncio
async def test_welcome_pulse_stops_unfocused_and_resumes_focused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The splash's glow is 92% of what an idle session writes to the terminal.

    Gated on focus ONLY: the focused cadence and appearance are untouched, so
    this asserts the timers exist again after refocus rather than that they
    changed rate.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)
    monkeypatch.setattr("local_operator.tui.shimmer.settings_get", lambda *a, **k: True)
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        welcome = app.query_one(WelcomeView)
        welcome.set_visible(True)
        welcome._sync_pulse_timer()
        welcome._sync_tip_timer()
        await pilot.pause()
        assert welcome._pulse_timer is not None
        assert welcome._tip_timer is not None

        app._set_animation_focused(False)
        await pilot.pause()
        # Not merely paused: with the gate closed no timer exists at all, which
        # is the same shape the shimmer kill switch already produces.
        assert welcome._pulse_timer is None
        assert welcome._tip_timer is None
        # And the mark is back at its DEFINED resting frame, not the phase the
        # glow happened to be holding when the user looked away.
        assert welcome._mark_color is None
        assert welcome._tip_index == 0

        app._set_animation_focused(True)
        await pilot.pause()
        assert welcome._pulse_timer is not None
        assert welcome._tip_timer is not None


@pytest.mark.asyncio
async def test_shimmer_disabled_path_is_unchanged_by_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kill switch still wins on its own: focus cannot re-enable animation.

    ``motion_enabled()`` is an AND, so a developer or a CI job that turned
    animation off gets a still frame whether or not the terminal is focused.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        welcome = app.query_one(WelcomeView)
        welcome.set_visible(True)
        welcome._sync_pulse_timer()
        await pilot.pause()
        assert welcome._pulse_timer is None

        app._set_animation_focused(True)
        await pilot.pause()
        assert welcome._pulse_timer is None
        assert not animation.motion_enabled()


@pytest.mark.asyncio
async def test_a_keystroke_unthrottles_an_app_that_never_got_a_focus_event() -> None:
    """Typing into a blurred-looking session must restore full animation.

    Textual sets its `app_focus` reactive DIRECTLY when a key or mouse-down
    arrives while it believes the app is blurred, and that assignment posts no
    AppFocus event. A gate wired only to `on_app_focus` therefore never hears
    about it, and a host that reports blur but not focus leaves a session the
    user is actively typing into stuck at the reduced rate. Verified against
    the real app before the fix: `app_focus` True, animation gate False.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app.post_message(AppBlur())
        await pilot.pause()
        await pilot.pause()
        assert not animation.animation_focused()

        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()

        assert app.app_focus is True
        assert animation.animation_focused() is True


def test_focus_flag_defaults_focused_and_reports_only_real_changes() -> None:
    """The flag's contract, without an app: default True, idempotent sets.

    The return value is what lets the app skip the fan-out — Textual reasserts
    focus on every keypress, so an unconditional resync would stop and restart
    four timers per character typed.
    """
    animation.reset_animation_focus()
    assert animation.animation_focused() is True
    assert animation.set_animation_focused(True) is False
    assert animation.set_animation_focused(False) is True
    assert animation.set_animation_focused(False) is False
    assert animation.set_animation_focused(True) is True
    animation.reset_animation_focus()


@pytest.mark.asyncio
async def test_panel_repaints_in_full_on_refocus() -> None:
    """The dock's numbers must be current the moment the window is looked at.

    The panel has ONE repaint point (``_tick``), so refocus marks it dirty
    rather than painting from a second path; the next tick is then a full
    repaint including the re-read stats.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        panel = app.query_one(SubagentPanel)
        panel._start_spinner()
        await pilot.pause()
        panel._dirty = False

        app._set_animation_focused(False)
        await pilot.pause()
        app._set_animation_focused(True)

        assert panel._dirty is True


def test_the_band_rerenders_on_refocus() -> None:
    """The band's clock and token counts must be current on refocus.

    The band is the one throttled surface that is NOT a ``Widget`` — it drives
    a ``Static`` dock and is re-rated by the app rather than by a Textual event
    of its own — so nothing else in the suite holds its refocus repaint. It is
    covered here because the band carries the session clock and the token
    counts, not just a glyph: those are the numbers a returning user reads
    first, and at the blurred cadence they would otherwise be up to a second
    stale at the moment the window is looked at.

    Mutation-tested: deleting the ``refresh()`` in ``sync_animation_rate``
    survived the whole suite before this test existed (agent review R4).
    """
    from local_operator.tui.widgets.status_line import StatusLine
    from tests.unit.tui.test_status_line import _dock

    animation.reset_animation_focus()
    band = StatusLine(_dock(200))
    # Only a STREAMING band has a spinner timer to re-rate; an idle one has
    # already stopped and must stay stopped.
    band.update(streaming=True)
    dock = cast(Any, band)._dock

    animation.set_animation_focused(False)
    band.sync_animation_rate()
    dock.painted = None

    animation.set_animation_focused(True)
    band.sync_animation_rate()

    assert dock.painted is not None, "the band did not re-render on refocus"
    assert dock.layout is False, "the band must never ask for a layout pass"
    animation.reset_animation_focus()


def test_an_idle_band_stays_stopped_across_a_focus_change() -> None:
    """Re-rating must never START a spinner on a band that is not streaming.

    ``sync_animation_rate`` replaces the timer to change its interval, and the
    hazard in that shape is arming a surface that had correctly stopped.
    """
    from local_operator.tui.widgets.status_line import StatusLine
    from tests.unit.tui.test_status_line import _dock

    animation.reset_animation_focus()
    band = StatusLine(_dock(200))
    dock = cast(Any, band)._dock
    before = len(dock.intervals)

    animation.set_animation_focused(False)
    band.sync_animation_rate()
    animation.set_animation_focused(True)
    band.sync_animation_rate()

    assert len(dock.intervals) == before, "an idle band was given a spinner"
    animation.reset_animation_focus()


@pytest.mark.asyncio
async def test_a_blur_and_refocus_cycle_keeps_the_keyboard() -> None:
    """Alt-tabbing away and back must leave the user able to type.

    Textual's ``Reactive._check_watchers`` invokes BOTH watcher spellings —
    ``_watch_app_focus`` and then ``watch_app_focus`` — and the base
    implementation is not idempotent: on blur it stashes ``screen.focused``
    and then clears focus, so a second call re-reads the ``None`` it just
    wrote and destroys the memory. Refocus then restores nothing and the user
    has to click before the keyboard works again.

    That is invisible to every other test here (the whole file passed 18/18
    with the bug present) and to all four static gates, because it is a
    property of the app's INPUT path rather than of its paint path. Caught by
    agent review round 3 (R12); this test is what holds it.
    """
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        focused_before = app.screen.focused
        assert focused_before is not None, "fixture must start with a focused widget"

        app.post_message(AppBlur())
        await pilot.pause()
        app.post_message(AppFocus())
        await pilot.pause()

        assert app.screen.focused is focused_before, "refocus lost the keyboard"


@pytest.mark.asyncio
async def test_blurring_the_splash_repaints_it_to_its_resting_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing the pulse/tip STATE is not enough; the screen must be redrawn.

    Both stop methods reset their state so the next frame drawn is the defined
    resting one — but with the timer stopped there IS no next frame, so the
    terminal kept rendering whatever the animation was holding when focus went
    away: a mark frozen mid-swell (a different brightness per window), and a
    tip row showing an arbitrary entry that then swapped to ``TIPS[0]`` a
    tenth of a second after refocus. Stale while nobody is looking and moving
    exactly when someone is, in a change whose whole point is removing motion
    (design review D1/D2).

    Asserted through ``refresh``, because the defect is invisible in the
    widget's own attributes — those were already correct.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)
    monkeypatch.setattr("local_operator.tui.shimmer.settings_get", lambda *a, **k: True)
    app = _running_app()
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        welcome = app.query_one(WelcomeView)
        welcome.set_visible(True)
        welcome._sync_pulse_timer()
        welcome._sync_tip_timer()
        await pilot.pause()

        # Put both animations somewhere OTHER than their resting frame, which
        # is the only state in which the missing repaint is observable.
        welcome._mark_color = "#a08040"
        welcome._tip_index = 2

        repaints = 0
        original = welcome.refresh

        def counting_refresh(*args: Any, **kwargs: Any) -> Any:
            nonlocal repaints
            repaints += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(welcome, "refresh", counting_refresh)
        app._set_animation_focused(False)
        await pilot.pause()

        assert welcome._mark_color is None
        assert welcome._tip_index == 0
        # TWO, not one: the pulse and the tip are separate stops with separate
        # repaints, and one shared counter at `>= 1` goes green when either
        # fix alone is removed — over the exact stale-frame defect this test
        # exists to guard (agent review R16). The count is deterministic here
        # because both animations were put off their resting frame above.
        assert repaints >= 2, "a stop cleared its state without repainting the stale frame"
