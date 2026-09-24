"""Canonical paints on the TUI are spaced by their cost; urgent ones are not.

WHY. On a viewer of a loaded parent (12 stepping lanes, 240 settled children)
one canonical paint cost 40-60 ms of the TUI's CPU and every delta scheduled
one on the next loop turn, so Enter waited 2-4 s p50 to be handled. The fix
spaces non-urgent paints by ``cost / share`` (``_frontend_apply_delay``) and
exempts what the user is waiting on: the working signal, an approval/ask card,
the completion receipt, and a child starting, settling or leaving.

These are STRUCTURAL assertions per AGENTS.md "Timing, flakes": which paint
path a delta takes (next turn vs spaced timer) and how many paints a burst
produces. The paint's measured cost is set directly rather than timed, and the
real Textual scheduler is recorded rather than awaited, so no assertion here
depends on how fast the host is.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
)
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    # An inherited CMUX_* variable lets a headless app rename the operator's
    # real workspaces; HOME is redirected because the cache root derives from it.
    for key in tuple(os.environ):
        if key.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(OperatorApp, "_start_update_check", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


class _Viewer(FakeSession):
    """A fake owner carrying a REAL store, so revisions are the production ones."""

    def __init__(self) -> None:
        super().__init__()
        self._store = FrontendStateStore(FrontendSessionState(session_id="sess", epoch="e1"))

    @property
    def frontend_state(self) -> FrontendSessionState:
        return self._store.state

    def frontend_revision(self) -> Any:
        return self._store.revision()

    def push(self, **changes: Any) -> FrontendUpdate:
        update = FrontendUpdate(
            epoch="e1", sequence=self._store._state.sequence + 1, changes=changes
        )
        self._store.apply_update(update)
        return update


def _job(job_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": job_id, "type": "task", "status": "running", "label": job_id, **fields}


class _Recorder:
    """Stands in for ``call_later``/``set_timer``: which path, and with what delay."""

    def __init__(self, app: OperatorApp) -> None:
        self.now: list[Any] = []
        self.spaced: list[tuple[float, Any]] = []
        self.stopped = 0
        recorder = self

        class _Timer:
            def stop(self) -> None:
                recorder.stopped += 1

        def call_later(callback: Any, *args: Any, **_kw: Any) -> bool:
            # The arming hop is how a spaced paint reaches ``set_timer`` (it is
            # armed on the app's own context); run it so the timer is observed.
            if getattr(callback, "__name__", "") == "_arm_frontend_apply_timer":
                callback(*args)
                return True
            self.now.append((callback, args))
            return True

        def set_timer(delay: float, callback: Any, **_kw: Any) -> Any:
            self.spaced.append((delay, callback))
            return _Timer()

        app.call_later = call_later  # type: ignore[method-assign]
        app.set_timer = set_timer  # type: ignore[method-assign]


async def _booted(viewer: _Viewer):  # type: ignore[no-untyped-def]
    app = OperatorApp(lambda: _factory(viewer))
    return app


def _as_if_a_paint_just_cost(app: OperatorApp, seconds: float) -> None:
    """Put the app where a real paint of ``seconds`` has just finished."""
    import time

    app._frontend_apply_scheduled = False
    app._frontend_apply_timer = None
    app._frontend_paint_cost_s = seconds
    app._frontend_painted_at = time.perf_counter()


@pytest.mark.asyncio
async def test_a_costly_roster_delta_is_spaced_not_painted_next_turn() -> None:
    viewer = _Viewer()
    viewer.push(jobs=[_job("a", latest_details={"progress": "one"})])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)  # records the painted lifecycle
        recorder = _Recorder(app)
        _as_if_a_paint_just_cost(app, 0.040)
        # Progress text on a running child: the churn a busy roster is made of.
        app._on_frontend_update(viewer.push(jobs=[_job("a", latest_details={"progress": "two"})]))
        assert recorder.now == []
        assert len(recorder.spaced) == 1
        delay, _callback = recorder.spaced[0]
        # 40 ms at a 25% share spaces the next paint ~120 ms after the last.
        assert 0.0 < delay <= app_module._frontend_apply_delay(0.040)


@pytest.mark.asyncio
async def test_a_burst_inside_one_window_costs_one_paint_and_it_paints_the_last_state() -> None:
    """N deltas inside one spacing window -> one paint, reading the LATEST state."""
    viewer = _Viewer()
    viewer.push(jobs=[_job("a")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)
        recorder = _Recorder(app)
        _as_if_a_paint_just_cost(app, 0.040)
        painted: list[str] = []
        app._apply_frontend_state = lambda state: painted.append(  # type: ignore[method-assign]
            state.jobs[0].latest_details["progress"]
        )
        for n in range(25):
            app._on_frontend_update(
                viewer.push(jobs=[_job("a", latest_details={"progress": f"step {n}"})])
            )
        # ceil(window / floor) bounds how many paints a window may hold; one
        # window here, and the whole burst coalesced into its single paint.
        assert len(recorder.spaced) + len(recorder.now) == 1
        _delay, callback = recorder.spaced[0]
        callback()
        assert painted == ["step 24"], "the last snapshot must be the one painted"


async def _path_taken(**changes: Any) -> str:
    """Push ``changes`` right after a costly paint and report "now" or "spaced"."""
    viewer = _Viewer()
    viewer.push(jobs=[_job("a"), _job("b")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)
        recorder = _Recorder(app)
        _as_if_a_paint_just_cost(app, 0.040)
        app._on_frontend_update(viewer.push(**changes))
        assert len(recorder.now) + len(recorder.spaced) == 1
        return "now" if recorder.now else "spaced"


# ONE TEST PER EXEMPTION, as the design's risk list requires ("enforce the
# exemption list with a test per field, not a comment"): each would silently
# become a spaced paint if its entry were dropped.


@pytest.mark.asyncio
async def test_the_working_signal_is_painted_next_turn() -> None:
    assert await _path_taken(streaming=True) == "now"


@pytest.mark.asyncio
async def test_an_approval_or_ask_card_is_painted_next_turn() -> None:
    gate = {"request_id": "r1", "kind": "approval", "title": "Run bash?"}
    assert await _path_taken(pending_gate=gate) == "now"


@pytest.mark.asyncio
async def test_the_completion_receipt_is_painted_next_turn() -> None:
    assert await _path_taken(attention={"completion_token": "t1", "anchor_id": "a"}) == "now"


@pytest.mark.asyncio
async def test_a_child_settling_is_painted_next_turn() -> None:
    assert await _path_taken(jobs=[_job("a", status="failed"), _job("b")]) == "now"


@pytest.mark.asyncio
async def test_a_child_starting_is_painted_next_turn() -> None:
    assert await _path_taken(jobs=[_job("a"), _job("b"), _job("c")]) == "now"


@pytest.mark.asyncio
async def test_a_queued_child_being_admitted_is_painted_next_turn() -> None:
    viewer = _Viewer()
    viewer.push(jobs=[_job("a", queued=True)])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)
        recorder = _Recorder(app)
        _as_if_a_paint_just_cost(app, 0.040)
        app._on_frontend_update(viewer.push(jobs=[_job("a", queued=False)]))
        assert len(recorder.now) == 1 and recorder.spaced == []


@pytest.mark.asyncio
async def test_progress_churn_and_scalars_are_spaced() -> None:
    """The control: without these, every 'now' above could be a default."""
    assert (
        await _path_taken(jobs=[_job("a", latest_details={"progress": "moved"}), _job("b")])
        == "spaced"
    )
    assert await _path_taken(context_tokens=123) == "spaced"


@pytest.mark.asyncio
async def test_an_urgent_delta_pulls_a_spaced_paint_forward() -> None:
    viewer = _Viewer()
    viewer.push(jobs=[_job("a")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)
        recorder = _Recorder(app)
        _as_if_a_paint_just_cost(app, 0.040)
        app._on_frontend_update(viewer.push(jobs=[_job("a", latest_details={"progress": "x"})]))
        assert len(recorder.spaced) == 1 and recorder.now == []
        app._on_frontend_update(
            viewer.push(pending_gate={"request_id": "r1", "kind": "ask", "title": "Which?"})
        )
        assert len(recorder.now) == 1, "the paint is pulled onto the next turn"
        painted: list[Any] = []
        app._apply_frontend_state = painted.append  # type: ignore[method-assign]
        callback, args = recorder.now[0]
        callback(*args)
        assert recorder.stopped == 1, "and the spaced timer is cancelled"
        assert len(painted) == 1 and painted[0].pending_gate is not None
        assert app._frontend_apply_scheduled is False


@pytest.mark.asyncio
async def test_retiring_the_session_cancels_a_spaced_paint() -> None:
    viewer = _Viewer()
    viewer.push(jobs=[_job("a")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)
        recorder = _Recorder(app)
        _as_if_a_paint_just_cost(app, 0.040)
        app._on_frontend_update(viewer.push(jobs=[_job("a", latest_details={"progress": "x"})]))
        app._invalidate_pending_frontend_state()
        assert recorder.stopped == 1
        assert app._frontend_apply_scheduled is False


@pytest.mark.asyncio
async def test_the_live_prompt_backstop_runs_even_when_the_band_is_gated() -> None:
    """Review round 1, F2: the card's only trigger here is this backstop.

    The card's footer is derived from focus and draft state, neither of which
    emits anything the card hears, so it is re-asked directly on every canonical
    paint. Gating that with the band would leave the routine token/phase delta
    (which moves no roster row) re-checked only by the 1 Hz poll.
    """
    viewer = _Viewer()
    viewer.push(jobs=[_job("a")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._session = viewer
        band_calls = 0
        stale_checks = 0

        def band() -> None:
            nonlocal band_calls
            band_calls += 1

        class _Prompt:
            def repaint_if_stale(self) -> None:
                nonlocal stale_checks
                stale_checks += 1

        app._refresh_band = band  # type: ignore[method-assign]
        app._live_prompt = lambda: _Prompt()  # type: ignore[method-assign]
        app._apply_frontend_state(viewer.frontend_state)
        assert (band_calls, stale_checks) == (1, 1)
        # A scalar delta: the band is gated, the card is still re-asked.
        viewer.push(activity_phase="responding", context_tokens=10)
        app._apply_frontend_state(viewer.frontend_state)
        assert band_calls == 1, "no roster row moved, so the band is skipped"
        assert stale_checks == 2, "the backstop must still run"


@pytest.mark.asyncio
async def test_a_session_switch_does_not_space_the_new_sessions_first_paint() -> None:
    """Review round 1, N2: the spacing describes the roster just left."""
    viewer = _Viewer()
    viewer.push(jobs=[_job("a")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._invalidate_pending_frontend_state()
        app._paint_frontend_session(viewer)
        _as_if_a_paint_just_cost(app, 0.040)
        app._band_painted_revision = (viewer, viewer.frontend_revision())
        app._frontend_painted_lifecycle = (viewer, "e1", 0)
        app._invalidate_pending_frontend_state()
        assert app._frontend_paint_cost_s == 0.0
        assert app._frontend_painted_at is None
        assert app._band_painted_revision is None
        assert app._frontend_painted_lifecycle is None
        recorder = _Recorder(app)
        app._on_frontend_update(viewer.push(jobs=[_job("a", latest_details={"progress": "later"})]))
        assert len(recorder.now) == 1, "the first paint after a switch is not spaced"
        assert recorder.spaced == []


@pytest.mark.asyncio
async def test_the_band_is_not_repainted_when_no_collection_moved() -> None:
    viewer = _Viewer()
    viewer.push(jobs=[_job("a")])
    app = await _booted(viewer)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        refreshes = 0

        def refreshed() -> None:
            nonlocal refreshes
            refreshes += 1

        app._session = viewer
        app._refresh_band = refreshed  # type: ignore[method-assign]
        app._apply_frontend_state(viewer.frontend_state)
        assert refreshes == 1
        viewer.push(activity_phase="responding", context_tokens=10)
        app._apply_frontend_state(viewer.frontend_state)
        assert refreshes == 1, "a scalar delta cannot move the band"
        viewer.push(jobs=[_job("a", latest_details={"progress": "moved"})])
        app._apply_frontend_state(viewer.frontend_state)
        assert refreshes == 2


def test_the_spacing_is_bounded_by_its_floor_and_ceiling() -> None:
    delay = app_module._frontend_apply_delay
    assert delay(0.0) == app_module._FRONTEND_APPLY_FLOOR_S
    assert delay(0.001) == app_module._FRONTEND_APPLY_FLOOR_S
    assert delay(0.040) == pytest.approx(0.040 / app_module._FRONTEND_APPLY_MAX_SHARE - 0.040)
    assert delay(10.0) == app_module._FRONTEND_APPLY_CEILING_S
