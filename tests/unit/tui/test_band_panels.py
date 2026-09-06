"""The dock band panels (todo + subagent) — the new chrome above the composer.

The build wave wrote these widgets but never mounted or tested them; the
integration committed them into ``app.py`` compose() with a 1 Hz poll and
Subagent* event handlers. These tests drive the REAL mounted panels through
the app, so visibility toggling (empty vs populated), the refresh path, and
the trajectory modal are asserted the way a user reaches them — not as
isolated widget calls that can pass while the wiring is dead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult

from local_operator.harness.types import ImageContent
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import SubagentEnded, SubagentStarted
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.status_line import format_agents
from local_operator.tui.widgets.subagent_panel import (
    MAX_SUBAGENT_ROWS,
    Density,
    SubagentPanel,
    SubagentRow,
    SummaryCounts,
    compose_summary,
)
from local_operator.tui.widgets.todo_panel import MAX_TODO_ROWS, TodoPanel


@pytest.mark.asyncio
async def test_todo_panel_reads_selected_child_snapshot_without_root_leakage(tmp_path) -> None:
    from local_operator.session.transcript import Transcript
    from local_operator.tools.builtin import TODO_STORE

    child = Transcript(tmp_path / "child")
    await child.append_custom(
        "todo_snapshot",
        {
            "items": [
                {
                    "name": "Child",
                    "items": [{"text": "child only", "status": "pending", "reason": ""}],
                }
            ]
        },
    )
    TODO_STORE["root"] = [
        {"name": "Root", "items": [{"text": "root only", "status": "pending", "reason": ""}]}
    ]
    try:
        panel = TodoPanel()
        session = type("Session", (), {"session_id": "root"})()
        panel.sync(session, session_id="child", transcript_directory=str(child.directory))
        # Historical transcript fallback is deliberately off-loop; the first
        # sync schedules one worker read and the callback repaints from cache.
        for _ in range(50):
            if "child only" in str(panel._body.content):
                break
            await asyncio.sleep(0.01)
        assert "child only" in str(panel._body.content)
        assert "root only" not in str(panel._body.content)
        panel.sync(session, session_id="missing")
        assert panel.display is False
    finally:
        TODO_STORE.pop("root", None)
        TODO_STORE.pop("child", None)
        TODO_STORE.pop("missing", None)


class FakeSession:
    """Minimal SessionProtocol the app can boot against."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.aborts: list[str] = []
        self.disposed = False
        self._handlers: list[Any] = []
        self.jobs: Any = None
        self._subagent_comms: Any = None
        self._history: list[Any] = []
        self.asides: list[list[Any]] = []
        self.adopted: list[list[Any]] = []

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def model_label(self) -> str:
        return "test/model"

    @property
    def model(self) -> Any:
        return None

    @property
    def effective_model(self) -> Any:
        # The fake never falls back, so selection and effective agree.
        return self.model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    def queued_steering(self) -> list[Any]:
        return []

    def steer_message(self, message: Any) -> None:
        pass

    def recall_steering(self, message: Any) -> bool:
        return False

    def set_approval_handler(self, handler: Any | None) -> None:
        self.approval_handler = handler

    def set_ask_handler(self, handler: Any | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot; fakes only
        # need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def running_subagents(self) -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    @property
    def conversation_name(self) -> str:
        return self.conversation_name_state.text

    @property
    def conversation_name_state(self) -> ConversationName:
        # The real holder, created on first read: `user_set` precedence (a
        # human rename outranks every generated title, forever) is behaviour
        # the TUI reads before it spends a re-title call, so a fake that
        # reimplemented it as a bare string would hide a regression in it.
        state = getattr(self, "_name_state", None)
        if state is None:
            state = self._name_state = ConversationName()
        return state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return self.conversation_name_state.set(text, user_set=user_set)

    async def complete_once(self, system: str, prompt: str) -> str:
        return ""

    async def dispose(self) -> None:
        self.disposed = True

    def history(self) -> list[Any]:
        return self._history

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Recorded, not answered: the aside's no-trace contract is proven
        # against the real Session in tests/unit/session/test_aside.py. Here
        # the only thing that must hold is that the app can call it.
        self.asides.append(list(turns))
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        self.adopted.append(list(messages))

    async def compact_now(self) -> CompactionOutcome:
        # No history to compact: this fake never carries a conversation, which
        # is the state a real session answers with the same refusal.
        return CompactionOutcome(
            ran=False, reason="nothing_to_compact", detail="nothing to compact"
        )


def _async_factory(session: FakeSession):
    """The app's factory must return an awaitable; boot does ``await`` on it."""

    async def factory() -> FakeSession:
        return session

    return factory


def _fake_jobs(*jobs: Any) -> Any:
    class _Manager:
        def list(self, *, owner_id: str | None = None) -> list[Any]:
            return list(jobs)

        def get(self, job_id: str, *, owner_id: str | None = None) -> Any:
            for job in jobs:
                if str(getattr(job, "id", "")) == job_id:
                    return job
            return None

    return _Manager()


class _Job:
    def __init__(self, job_id: str, label: str, status: str = "running") -> None:
        self.id = job_id
        self.type = "task"
        self.status = status
        self.label = label
        self.start_time = 1_700_000_000.0
        # Mirrors ``AsyncJob.started_at``: when the runner actually began, or
        # ``None`` if it never did. Defaulted to ``None`` so a fixture that
        # does not think about it says "never ran" — the same default the real
        # model carries, and the discriminator ``cancel()`` stamps on.
        self.started_at: float | None = None
        # Mirrors ``AsyncJob.queued``: admitted to the ledger but held behind
        # the capacity gate, which carries ``status == "running"`` while not
        # actually running. Declared so a test can set it without inventing an
        # attribute the real model has.
        self.queued: bool = False
        self.result_text: str | None = None
        self.error_text: str | None = None
        self.settled_at: float | None = None
        self.trajectory: list[dict[str, Any]] | None = None
        # Mirrors ``AsyncJob.agent_role``/``effort``: the child's role and
        # effort tier, recorded at launch. Defaulted to the real model's
        # not-recorded conventions — a plain fixture is a ``task`` with no tier
        # — so a test that does not think about them gets the common case.
        self.agent_role: str | None = "task"
        self.effort: str | None = None


@pytest.fixture(autouse=True)
def _clean_todo_store():
    """Reset the shared todo store between tests (it is module-global in the
    tool registry, so one test's seed leaks into the next otherwise)."""
    from local_operator.tools import builtin

    builtin.TODO_STORE.clear()
    yield
    builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_band_panels_hidden_when_empty_and_shown_when_populated() -> None:
    """Both panels collapse to zero height with nothing to show, and appear
    once their ledger/store has content — the visibility is per-panel, so an
    idle session costs the layout nothing."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        # Wait for the SESSION, not a frame count. The app paints before its
        # session exists (the factory is awaited in a boot worker) and both
        # panels read their content off it, so a repaint that lands in that
        # window finds no jobs and no todos and hides them both — which reads
        # as this assertion failing for a reason that has nothing to do with
        # visibility. The same race costs `test_app_pilot`'s first test on
        # `main`; here it is deterministic enough to fix rather than tolerate.
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        assert app._session is not None, "the session never booted"
        todo = app.query_one(TodoPanel)
        sub = app.query_one(SubagentPanel)
        assert todo.display is False
        assert sub.display is False

        # Give the session a task job + a todo list, then repaint via the same
        # path the 1 Hz poll uses.
        session.jobs = _fake_jobs(_Job("sub-1", "summarize workspace"))
        builtin.TODO_STORE["sess"] = [
            {"text": "wire the band", "status": "done"},
            {"text": "capture frames", "status": "pending"},
        ]
        app._refresh_band()
        await pilot.pause()
        assert sub.display is True
        assert todo.display is True

        # Draining both sides hides them again.
        session.jobs = _fake_jobs()
        builtin.TODO_STORE["sess"] = []
        app._refresh_band()
        await pilot.pause()
        assert sub.display is False
        assert todo.display is False


@pytest.mark.asyncio
async def test_subagent_panel_bounds_to_newest_slice_and_expands_in_start_order() -> None:
    """The default roster preserves the relevant tail without reversing it;
    its keyboard disclosure reveals every retained child, and the press after
    that collapses the roster AND shrinks the panel to the summary row in one
    step (#525 design §1: the roster's collapse is folded into the forward
    cycle rather than being a fourth stop). The press after that hides it,
    and the one after brings the preview back exactly as it was."""
    session = FakeSession()
    jobs = [_Job(f"sub-{index:02d}", f"task {index:02d}") for index in range(1, 13)]
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(SubagentPanel)

        visible = [row.job_id for row in panel.query(SubagentRow) if row.display]
        assert visible == [f"sub-{index:02d}" for index in range(7, 13)]
        assert panel.predicted_rows() == MAX_SUBAGENT_ROWS
        assert str(panel._affordance.content) == "+6 earlier · ctrl+g to expand"

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert [row.job_id for row in panel.query(SubagentRow) if row.display] == [
            f"sub-{index:02d}" for index in range(1, 13)
        ]
        assert str(panel._affordance.content) == "ctrl+g to shrink"

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert panel._expanded is False
        assert [row.job_id for row in panel.query(SubagentRow) if row.display] == []
        assert app.focused is app.query_one(Editor)

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.HIDDEN
        assert panel.display is False

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.FULL
        assert [row.job_id for row in panel.query(SubagentRow) if row.display] == visible
        assert str(panel._affordance.content) == "+6 earlier · ctrl+g to expand"


@pytest.mark.asyncio
async def test_clicking_subagent_disclosure_preserves_composer_focus_and_input() -> None:
    """Pointer disclosure is only a visibility gesture; unlike ``ctrl+g``, it
    must not silently opt the user into keyboard roster navigation or divert
    the next character away from their draft."""
    session = FakeSession()
    session.jobs = _fake_jobs(
        *[_Job(f"sub-{index:02d}", f"task {index:02d}") for index in range(1, 13)]
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.text = "draft"
        editor.move_cursor(editor._end_of_buffer())
        editor.focus()

        await pilot.click("#subagent-affordance")
        await pilot.pause()
        assert app.focused is editor
        assert app.query_one(SubagentPanel)._expanded is True
        await pilot.press("x")
        assert editor.text == "draftx"

        # The second click on the affordance is the forward cycle: collapse
        # AND shrink to the summary row, with the composer still focused.
        await pilot.click("#subagent-affordance")
        await pilot.pause()
        assert app.focused is editor
        panel = app.query_one(SubagentPanel)
        assert panel._expanded is False
        assert panel.density is Density.SUMMARY
        await pilot.press("y")
        assert editor.text == "draftxy"

        # The header is the pointer target in every density (#525 design §3):
        # in summary the affordance row is gone, so the caption is what the
        # user has to click to keep cycling, and it too leaves the draft alone.
        await pilot.click("#subagent-header")
        await pilot.pause()
        assert panel.density is Density.HIDDEN
        assert app.focused is editor
        await pilot.press("z")
        assert editor.text == "draftxyz"
        # Hidden has no header to click; the key is the way back.
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.FULL
        assert panel._expanded is False
        await pilot.click("#subagent-header")
        await pilot.pause()
        # With overflow the header click is the expand step, same as the key
        # would be, and still does not enter navigation.
        assert panel._expanded is True
        assert app.focused is editor
        await pilot.press("w")
        assert editor.text == "draftxyzw"


@pytest.mark.asyncio
async def test_subagent_expansion_traverses_every_row_and_escape_restores_composer() -> None:
    session = FakeSession()
    session.jobs = _fake_jobs(
        *[_Job(f"sub-{index:02d}", f"task {index:02d}") for index in range(1, 31)]
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.text = "draft survives roster navigation"

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert app.focused is app.query_one(SubagentRow)
        for _ in range(29):
            await pilot.press("down")
        await pilot.pause()
        assert isinstance(app.focused, SubagentRow)
        assert app.focused.job_id == "sub-30"
        panel = app.query_one(SubagentPanel)
        # The expanded roster virtualizes to one viewport: traversal advances
        # the logical window instead of accumulating off-screen DOM rows.
        assert sum(row.display for row in panel.query(SubagentRow)) < 30
        assert panel._navigation_index == 29

        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is editor
        assert editor.text == "draft survives roster navigation"


@pytest.mark.asyncio
async def test_subagent_disclosure_hides_without_overflow_and_resize_recovers_geometry() -> None:
    session = FakeSession()
    session.jobs = _fake_jobs(
        *[_Job(f"sub-{index:02d}", f"task {index:02d}") for index in range(1, 31)]
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        await pilot.resize_terminal(50, 16)
        for _ in range(4):
            await pilot.pause()
        await pilot.resize_terminal(100, 30)
        for _ in range(4):
            await pilot.pause()
        assert app.screen.virtual_size == app.screen.size, {
            "panel": app.query_one(SubagentPanel).size,
            "list": app.query_one(SubagentPanel)._list.size,
            "band": app.query_one("#band").size,
            "predicted": app.query_one(SubagentPanel).predicted_rows(),
        }

        session.jobs = _fake_jobs(*[_Job(f"small-{index}", f"small {index}") for index in range(6)])
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        assert panel._affordance.display is False
        assert panel.predicted_rows() == 7


@pytest.mark.asyncio
async def test_todo_panel_renders_items_with_done_and_pending() -> None:
    """The todo panel's body shows the list with done items struck through —
    the readable mid-conversation progress surface, not a bare count."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": "wire the band", "status": "done"},
            {"text": "capture frames", "status": "pending"},
        ]
        app._refresh_band()
        await pilot.pause()
        todo = app.query_one(TodoPanel)
        body_plain = str(todo._body.content).replace("\n", " ").replace(" ", "")
        assert "wiretheband" in body_plain
        assert "captureframes" in body_plain
        # The fraction says what it counts, and with nothing abandoned there is
        # no dropped segment to state.
        assert "1/2resolved" in body_plain
        assert "dropped" not in body_plain


@pytest.mark.asyncio
async def test_todo_panel_renders_blocked_and_dropped_and_counts_resolved() -> None:
    """All four statuses, in the tool's own marks, with the header counting
    progress toward a FINISHED list: done and dropped need no more work, while
    blocked stays visibly open (with what it waits on) because it is the one row
    the user has to act on. A blocked item counted as complete is how a stalled
    list reads as a finished one. Each settled/open state that needs saying says
    it in one grammar (``text — state``), so no two statuses are a single glyph
    apart."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": "wire the band", "status": "done"},
            {"text": "pick a domain", "status": "blocked", "reason": "needs the user"},
            {"text": "old plan", "status": "dropped"},
            {"text": "capture frames", "status": "pending"},
            {"text": "from the future", "status": "unheard-of"},
        ]
        app._refresh_band()
        await pilot.pause()
        todo = app.query_one(TodoPanel)
        body = str(todo._body.content)

        assert "- [x] wire the band" in body
        assert "- [~] pick a domain — blocked: needs the user" in body
        assert "- [-] old plan — dropped" in body
        assert "- [ ] capture frames" in body
        # An unknown future status renders as OPEN, never as complete.
        assert "- [ ] from the future" in body
        # done + dropped out of five; blocked, pending and the unknown are open.
        # The word is what stops the sum reading as five finished items, and the
        # abandoned count is stated rather than folded in.
        assert body.split("\n")[0] == "Todos · 2/5 resolved · 1 dropped"


def _style_at(row: Text, needle: str) -> Style:
    """The style covering ``needle``'s first cell in a built row.

    Rich types a span's style as ``str | Style`` (a span may carry a style
    NAME), so it is parsed rather than returned as-is: every caller here reads
    ``.color`` off the result.
    """
    start = row.plain.index(needle)
    for span in reversed(row.spans):
        if span.start <= start < span.end:
            return Style.parse(span.style) if isinstance(span.style, str) else span.style
    raise AssertionError(f"no span covers {needle!r} in {row.plain!r}")


def _ink(style: Style) -> str:
    assert style.color is not None and style.color.triplet is not None
    return style.color.triplet.hex.lower()


@pytest.mark.asyncio
async def test_todo_header_names_its_arithmetic_and_the_abandoned_count() -> None:
    """``n/total`` counted abandoned work as progress with no word for it: one
    ``done`` beside four ``dropped`` rendered ``Todos · 5/5``, which is exactly
    what a finished plan looks like. The sum stays — a fully triaged list is not
    stalled — and the words that make it honest come back: ``resolved``, plus
    the abandoned count stated BESIDE the fraction rather than hidden inside
    it."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": "ship the guardrail", "status": "done"},
            {"text": "add the phase headers", "status": "dropped"},
            {"text": "rewrite the store", "status": "dropped"},
            {"text": "extend the tool schema", "status": "dropped"},
            {"text": "backfill the old sessions", "status": "dropped"},
        ]
        app._refresh_band()
        await pilot.pause()
        todo = app.query_one(TodoPanel)
        assert str(todo._body.content).split("\n")[0] == "Todos · 5/5 resolved · 4 dropped"


@pytest.mark.asyncio
async def test_todo_rows_separate_done_from_dropped_and_pending_from_blocked() -> None:
    """Inside each tier the statuses were one dim character apart: ``- [x]``
    beside ``- [-]`` made a finished item and an abandoned one read as the same
    thing, and ``blocked`` sat in ``pending``'s ink with its reason at 4.18:1 —
    the least legible text in the band, on the one row the user has to act on.

    Open-vs-settled is still luminance and must not regress; WHICH state inside
    a tier is a word."""
    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        done = panel._item_row("finished work", "done")
        dropped = panel._item_row("abandoned work", "dropped")
        pending = panel._item_row("open work", "pending")
        blocked = panel._item_row("stuck work", "blocked", "needs the user")
        bare = panel._item_row("stuck work", "blocked")

        # Open vs settled: settled work is quiet and struck, open work is not.
        for row, word in ((done, "finished"), (dropped, "abandoned")):
            style = _style_at(row, word)
            assert style.strike is True
            assert _ink(style) == theme_mod.semantic_color("dim")
        assert not _style_at(pending, "open").strike
        assert not _style_at(blocked, "stuck").strike

        # The two settled states are no longer one glyph apart, and the tag is
        # NOT struck so it stays readable on a crossed-out row.
        assert done.plain == "- [x] finished work"
        assert dropped.plain == "- [-] abandoned work — dropped"
        assert not _style_at(dropped, "— dropped").strike

        # The blocked row is the loudest ink in the band rather than the
        # quietest, and it names its state whether or not a reason came with it
        # (which is what the tool's own receipt already writes).
        assert _ink(_style_at(pending, "open")) == theme_mod.semantic_color("muted")
        assert _ink(_style_at(blocked, "stuck")) == theme_mod.semantic_color("fg")
        assert blocked.plain == "- [~] stuck work — blocked: needs the user"
        assert bare.plain == "- [~] stuck work — blocked"
        # The part that says what it waits on is legible now, not `dim`.
        assert _ink(_style_at(blocked, "needs the user")) == theme_mod.semantic_color("muted")


@pytest.mark.asyncio
async def test_todo_rows_clip_with_a_visible_ellipsis_at_a_narrow_width() -> None:
    """The rows' own ``no_wrap``/``overflow="ellipsis"`` pair never fired:
    ``#band`` is ``width: auto`` and its slots are ``1fr``, so the band measured
    the widest row — 129 cells inside a 50-cell screen — and the container cut
    the rest flush against the edge. A blocked reason stopped mid-word with
    nothing saying it continued. Clipping against the screen makes the marker
    real and stops the band overflowing sideways at the same time."""
    from rich.cells import cell_len

    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(52, 24)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {
                "text": "decide the compaction default",
                "status": "blocked",
                "reason": "waiting on the user to confirm the ceiling and the percentage",
            }
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        row = str(panel._body.content).split("\n")[1]

        assert panel._row_cells() == app.screen.size.width - 2  # the dock's rail
        # Never wider than the budget, and never more than one cell under it:
        # rstripping the cut can hand a cell back, which is the whole point of
        # doing it before the marker goes on.
        assert panel._row_cells() - 1 <= cell_len(row) <= panel._row_cells()
        assert row.endswith("…")
        # Never "word …": the same row must not truncate in two typographic
        # styles one column apart.
        assert not row.endswith(" …")
        # The band no longer drives the screen's width from a natural row size.
        assert app.screen.query_one("#todo-body").size.width == panel._row_cells()
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)


@pytest.mark.asyncio
async def test_todo_row_cap_follows_the_screen_and_the_marker_counts_the_hidden() -> None:
    """An absolute cap of eight asked for ten rows inside a twelve-row screen,
    and ``Screen { overflow: hidden }`` swallowed the difference silently: the
    header and the first three rows were clipped ABOVE the top edge while the
    surviving ``… 4 more todos`` counted only the tail the panel dropped itself
    — seven were actually unseen. The cap answers to the height now, and the
    marker counts every item the reader cannot see."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": f"step {n} of the plan", "status": "pending"} for n in range(1, 13)
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        lines = str(panel._body.content).split("\n")

        assert app.screen.size.height == 12
        # 12 rows of screen, 8 of them the dock's, leaving header + 2 + marker.
        #
        # The band's top inset (`#band.has-slot`) is NOT among them at this
        # size: it is withheld at or below a 12-row screen, where the dock
        # already crowds the terminal and the row would come out of this
        # panel's items. So the arithmetic here is the same as before the inset
        # existed — which is the point of the floor, and why this test is
        # unchanged from `main` while the taller sizes shifted by a row.
        assert panel._body_rows() == 4
        assert len(lines) == 4
        assert lines[0] == "Todos · 0/12 resolved"
        assert lines[1:3] == ["- [ ] step 1 of the plan", "- [ ] step 2 of the plan"]
        assert lines[-1] == "… 10 more todos"
        # Nothing clipped upward, and the band fits the screen it is drawn in.
        assert app.screen.query_one("#todo-body").region.y >= 0
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)

    # The floor: a 12-row TERMINAL leaves three rows for the whole panel, so the
    # marker is the row that goes — an item is worth more than a count the
    # header's own denominator already implies, and dropping it is what keeps
    # the band inside the screen at all.
    #
    # Driven mid-session, which is the state a todo list exists in. The BOOT
    # splash asks for one content row where an empty transcript asks for none
    # (measured: `TranscriptView` region height 1 vs 0 at this size), so at this
    # one size the splash is a row over on its own; the panel is already at its
    # floor and cannot pay for it.
    short = OperatorApp(_async_factory(FakeSession()))
    async with short.run_test(size=(100, 12)) as pilot:
        await pilot.pause()
        await pilot.press(*"go")
        await pilot.press("enter")
        await pilot.pause()
        short._refresh_band()
        await pilot.pause()
        panel = short.query_one(TodoPanel)
        lines = str(panel._body.content).split("\n")

        assert short.screen.size.height == 10
        assert panel._body_rows() == 2
        assert lines == ["Todos · 0/12 resolved", "- [ ] step 1 of the plan"]
        assert short.screen.query_one("#todo-body").region.y >= 0
        assert tuple(short.screen.virtual_size) == tuple(short.screen.size)


@pytest.mark.asyncio
async def test_todo_shows_a_ninth_row_rather_than_a_one_item_marker() -> None:
    """``… 1 more todo`` costs exactly the row the item costs, so the panel
    spends it on the item. The height is identical either way, which is what
    keeps going one over the cap safe at the boundary."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": f"step {n} of the plan", "status": "pending"}
            for n in range(1, MAX_TODO_ROWS + 2)
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        lines = str(panel._body.content).split("\n")

        assert panel._body_rows() == MAX_TODO_ROWS + 2
        assert len(lines) == MAX_TODO_ROWS + 2  # header + every item, no marker
        assert lines[-1] == f"- [ ] step {MAX_TODO_ROWS + 1} of the plan"
        assert "more todos" not in str(panel._body.content)


@pytest.mark.asyncio
async def test_todo_panel_repaints_when_only_a_blocker_reason_changes() -> None:
    """The equality guard keys on the reason too: re-blocking an item with a new
    reason changes what the row says, and a guard blind to it would leave the
    stale wait on screen."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": "pick a domain", "status": "blocked", "reason": "waiting on legal"}
        ]
        app._refresh_band()
        await pilot.pause()
        todo = app.query_one(TodoPanel)
        assert "waiting on legal" in str(todo._body.content)

        builtin.TODO_STORE["sess"][0]["reason"] = "waiting on the user"
        app._refresh_band()
        await pilot.pause()
        assert "waiting on the user" in str(todo._body.content)


@pytest.mark.asyncio
async def test_subagent_panel_opens_full_page_view() -> None:
    """Clicking/selecting a subagent row opens the full-page view rendering
    the job's retained events — the click-through a user reaches the child's
    work through."""
    session = FakeSession()
    job = _Job("sub-1", "summarize workspace", status="running")
    job.trajectory = [
        {"type": "message_start", "message": {"role": "assistant", "id": "m1"}},
        {
            "type": "message_update",
            "message": {"id": "m1", "role": "assistant"},
            "delta": "child did it",
        },
        {
            "type": "message_end",
            "message": {
                "id": "m1",
                "role": "assistant",
                "content": [{"type": "text", "text": "child did it"}],
            },
        },
        {
            "type": "tool_execution_start",
            "tool_call_id": "c1",
            "tool_name": "read",
            "args": {"path": "x"},
        },
        {
            "type": "tool_execution_end",
            "tool_call_id": "c1",
            "tool_name": "read",
            "result": {"is_error": False},
        },
    ]
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        sub = app.query_one(SubagentPanel)
        assert sub.display is True
        # open the page through the same callback the row uses
        app._open_subagent_view("sub-1")
        await pilot.pause()
        from local_operator.tui.widgets.subagent_view import SubagentView

        view = app.query_one(SubagentView)
        rendered = " ".join(view.rendered_rows())
        assert "child did it" in rendered, rendered
        assert "summarize workspace" in rendered, rendered


@pytest.mark.asyncio
async def test_todo_panel_hides_on_store_error_or_empty() -> None:
    """A missing/empty store degrades to a hidden panel — never a crash."""
    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        todo = app.query_one(TodoPanel)
        app._refresh_band()
        await pilot.pause()
        assert todo.display is False


@pytest.mark.asyncio
async def test_subagent_band_does_not_crush_the_transcript() -> None:
    """D-01 regression: a populated subagent band must stay as tall as its
    rows, not stretch to fill the screen and push the transcript to one row.

    The inner rows container defaulted to ``1fr``, so one subagent ballooned
    the band across the whole terminal. This measures the real layout: the
    band's rendered height must stay small (a handful of rows) and the
    transcript must keep the bulk of the vertical budget.
    """
    session = FakeSession()
    session.jobs = _fake_jobs(_Job("sub-1", "summarize workspace"))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()

        from local_operator.tui.widgets.subagent_panel import SubagentPanel
        from local_operator.tui.widgets.transcript import TranscriptView

        sub = app.query_one(SubagentPanel)
        transcript = app.query_one(TranscriptView)
        input_shell = app.query_one("#input-shell")
        assert sub.display is True
        # The band (header + one row + slot padding) must be a few rows, NOT
        # the full height. The transcript must keep the great majority.
        band_height = sub.region.height
        trans_height = transcript.region.height
        assert band_height <= 4, f"band ballooned to {band_height} rows"
        assert trans_height >= 10, f"transcript crushed to {trans_height} rows"
        # D-15-01: the band's content must sit ABOVE the composer (the input
        # shell), never overlapping it — a sibling bottom-dock previously
        # painted the band's rows blank behind the input. The subagent panel's
        # bottom must end at or above the input shell's top.
        assert (
            sub.region.bottom <= input_shell.region.y
        ), f"subagent band ({sub.region}) overlaps input shell ({input_shell.region})"
        # And the row's text is actually rendered (not blanked).
        row = sub.query("SubagentRow")[0]
        assert "summarize workspace" in str(getattr(row, "content", ""))


@pytest.mark.asyncio
async def test_reordering_skips_a_row_the_list_does_not_own_yet() -> None:
    """`mount` and `remove` are deferred in Textual, so between two syncs closer
    together than the DOM's apply tick `_rows` can name a widget that is not a
    child yet - and `move_child` raises `WidgetError` on it rather than
    ignoring it.

    Observed as a hard crash in the band refresh, roughly one full-suite run in
    ten and never reproducibly in isolation, so the state is constructed
    directly here rather than raced for.
    """
    from local_operator.tui.widgets.subagent_panel import SubagentPanel, SubagentRow

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield SubagentPanel(lambda *a: None)

    app = Host()
    async with app.run_test() as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        panel._sync_rows([_Job("b", "Second")])
        await pilot.pause()
        assert list(panel._list.children), "the fixture never mounted a row"

        # Exactly the state a deferred mount leaves: named in `_rows`, absent
        # from the DOM, and sorted BEFORE the row that is mounted.
        orphan = SubagentRow("a", panel._on_open)
        panel._rows = {"a": orphan, **panel._rows}

        panel._sync_rows([_Job("a", "First"), _Job("b", "Second")])

        assert orphan not in panel._list.children
        assert [row for row in panel._list.children] == [panel._rows["b"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("height", "children"),
    [(20, 4), (24, 4), (28, 6), (32, 10)],
)
async def test_the_band_inset_never_costs_the_subagent_list_its_screen(
    height: int, children: int
) -> None:
    """The dock's top inset is only taken when there is a row to spare.

    `TodoPanel` charges itself for that row (`_band_inset_rows`), but
    `SubagentPanel` has NO row budget: it mounts one row per task job
    unconditionally. So on the subagent side the inset's row is charged to
    nobody and comes straight out of the transcript — measured at 100x16 with
    six children as virtual height 15 against a 14-row screen, where the same
    frame without the inset fits exactly.

    `AGENTS.md` is explicit that a screen whose virtual size exceeds its actual
    size is always a bug here: `Screen { overflow: hidden }` does not report it,
    it silently clips a row off the top. The inset is therefore gated — on the
    screen height and on the child count, both of which are known synchronously
    and so cannot change between the frame that paints and the frame after it.
    Gates that MEASURED the laid-out band were tried first and removed: every
    one of them flipped mid-repaint and traded this overflow for visible
    motion.

    Driven at MID heights on purpose. The seam suite runs everything at 40 rows,
    which is why this class of overflow was invisible to it.

    The cases here are the ones where the band FITS without the inset, which is
    what makes them a statement about the inset rather than about the panel. A
    tall enough child list overflows this app on its own — ten children need
    twelve rows, which no 16-row screen holds beside the composer, here and
    equally on `main` — and that pre-existing defect (`SubagentPanel` has no row
    cap, unlike `TodoPanel`) is out of scope: its fix is a row budget and an
    `… N more` line on that panel, not a spacing change.

    Choosing the cells is therefore part of the test. They were picked from a
    100-configuration sweep (screen heights 12-40 x todo-only / subagent-only /
    both) as cells where `main` does NOT overflow, so an overflow here is this
    change's doing and nothing else's. Over that whole sweep the inset overflows
    in a strict SUBSET of `main`'s cells — it fixes four and introduces none —
    which is the property this samples.
    """
    session = FakeSession()
    session.jobs = _fake_jobs(*[_Job(f"sub-{i}", f"child task {i}") for i in range(children)])
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, height)) as pilot:
        await pilot.pause()
        app._refresh_band()
        for _ in range(4):
            await pilot.pause()

        screen = app.screen
        assert tuple(screen.virtual_size) == tuple(screen.size), (
            f"the band overflowed the screen at {height} rows with {children} children: "
            f"virtual={tuple(screen.virtual_size)} size={tuple(screen.size)}"
        )


@pytest.mark.asyncio
async def test_the_band_inset_survives_resizing_across_its_floor() -> None:
    """Shrinking and re-growing the terminal lands on the same frame each time.

    The dock's inset is gated on a SCREEN DIMENSION, and this is the property
    that makes that gate safe where the measured ones this replaced were not: a
    screen height cannot change between two frames of one repaint, so the answer
    cannot flip mid-paint. It can still be wrong across a RESIZE, which is the
    one input that does change it — and a gate with hysteresis would leave the
    band in a different state at 14 rows depending on whether the user got there
    by shrinking or by growing.

    Walked down past the floor and back up, asserting the state is a function of
    the height alone. The overflow assertion is deliberately paired with it: at
    these sizes `main` overflows at six of the ten steps and this walks through
    with one, so the floor is not buying stillness at the cost of the screen.
    """
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        assert app._session is not None, "the session never booted"
        builtin.TODO_STORE[session.session_id] = [
            {"text": f"step {n}", "status": "pending"} for n in range(5)
        ]
        try:
            seen: dict[int, bool] = {}
            for height in (30, 16, 14, 13, 12, 13, 14, 16, 30):
                await pilot.resize_terminal(100, height)
                # NO manual `_refresh_band()`. The production resize path is
                # what is under test: `on_resize` schedules the band's
                # re-decision itself, and an earlier version of this test
                # supplied that call by hand — which made it pass while the
                # real app sat on the previous height's answer until the 1 Hz
                # poll. A test that provides the trigger it is meant to be
                # checking for asserts nothing.
                for _ in range(6):
                    await pilot.pause()

                screen = app.screen
                inset = app.query_one("#band").has_class("has-slot")
                # Same height, same answer — whichever direction it arrived from.
                if height in seen:
                    assert seen[height] == inset, (
                        f"the inset is {inset} at {height} rows going one way and "
                        f"{seen[height]} the other: the gate has hysteresis"
                    )
                seen[height] = inset
                assert tuple(screen.virtual_size) == tuple(screen.size), (
                    f"the band overflowed the screen at {height} rows: "
                    f"virtual={tuple(screen.virtual_size)} size={tuple(screen.size)}"
                )
            # And the gate did something across this walk, or the test proves
            # nothing: tall screens take the row and short ones do not.
            assert seen[30] is True
            assert seen[12] is False
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(("height", "children"), [(15, 5), (16, 6), (17, 7), (18, 8), (20, 10)])
async def test_the_inset_is_never_what_tips_a_long_subagent_list_over(
    height: int, children: int
) -> None:
    """The dock's inset must not be the row that overflows the screen.

    ``SubagentPanel`` has no row cap — it mounts one row per task job, unlike
    ``TodoPanel``, which budgets against the screen — so a long enough child
    list overruns the dock on its own, on this branch and on ``main`` alike.
    That defect is out of scope here. Not making it WORSE is not: the inset's
    row is exactly what tips a list that currently just fits.

    Each case is A/B'd against the identical frame with the class forced off, in
    the same checkout, so the comparison isolates the inset rather than
    comparing two builds.

    The cells are exactly those where the PREVIOUS head granted the inset and
    the inset then caused an overflow — found by sweeping heights 14-26 against
    1-10 children on that head and keeping every cell where the A/B diverged.
    Choosing them that way is what makes this a regression test rather than a
    restatement: on the fixed head the gate WITHHOLDS the row in all five, which
    is the fix, and the assertion below holds either by the row being refused or
    by it being granted without cost.

    An earlier version parameterized cells where the gate withheld the inset on
    both heads. Those compare one build against itself and can only ever pass —
    the test was green and vacuous.
    """

    async def _overflows(*, suppress_inset: bool) -> bool:
        session = FakeSession()
        session.jobs = _fake_jobs(*[_Job(f"sub-{i}", f"child {i}") for i in range(children)])
        app = OperatorApp(_async_factory(session))
        if suppress_inset:
            original = type(app)._sync_band_inset

            def without_inset(self: Any) -> None:
                original(self)
                self.query_one("#band").remove_class("has-slot")

            app._sync_band_inset = without_inset.__get__(app)  # type: ignore[method-assign]
        async with app.run_test(size=(100, height)) as pilot:
            for _ in range(80):
                await pilot.pause()
                if app._session is not None:
                    break
            assert app._session is not None, "the session never booted"
            for _ in range(4):
                app._refresh_band()
                for _ in range(3):
                    await pilot.pause()
            screen = app.screen
            return tuple(screen.virtual_size) != tuple(screen.size)

    # The A/B is only meaningful where the inset is actually GRANTED. Where the
    # gate withholds it the two runs are the same build painting the same frame,
    # and any difference between them is the subagent panel's own pre-existing
    # overflow — which varies run to run at the cells where its row count sits
    # exactly on the screen height, and would make this test flaky about a
    # defect it does not own.
    async def _inset_granted() -> bool:
        """Whether the gate actually gives this configuration the row."""
        session = FakeSession()
        session.jobs = _fake_jobs(*[_Job(f"sub-{i}", f"child {i}") for i in range(children)])
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=(100, height)) as pilot:
            for _ in range(80):
                await pilot.pause()
                if app._session is not None:
                    break
            for _ in range(4):
                app._refresh_band()
                for _ in range(3):
                    await pilot.pause()
            return app.query_one("#band").has_class("has-slot")

    # A cell where the gate withholds the row compares the same frame with
    # itself; the assertion still holds, and holds trivially. Recorded rather
    # than skipped so a reader can see which of the two ways each cell passes.
    granted = await _inset_granted()

    with_inset = await _overflows(suppress_inset=False)
    without_inset = await _overflows(suppress_inset=True)

    assert not (with_inset and not without_inset), (
        f"the inset caused an overflow at {height} rows with {children} children "
        f"that the same frame without it does not have "
        f"(inset granted here: {granted})"
    )


#: Conversation rows a resumed session must still have on the shortest terminal
#: the operator actually uses (24 rows). Deliberately smaller than the panels'
#: own transcript floors: this is the USER-FACING guarantee — enough rows to
#: read a reply in place — where those constants are the mechanism, swept per
#: panel and free to move as long as this holds (UX round 2, U7).
_MIN_READABLE_TRANSCRIPT_ROWS = 3


@pytest.mark.asyncio
async def test_a_restored_dock_leaves_the_conversation_readable_on_a_short_terminal() -> None:
    """The dock is chrome: it may shorten itself, but it may not take the screen.

    Before the roster was restored on the first frame, a resumed session opened
    with an EMPTY dock, so the panels' absolute row budgets were only ever
    reached once children were running — by which point the user had asked for
    them. Restoring the roster (this PR) makes a cold resume paint a full
    roster, plan and wake list immediately, and on a short terminal the three
    panels filled the column between them: measured at 100x24, the operator's
    own height, ZERO conversation rows (UX round 2, U7).

    Asserted as the STRUCTURAL property rather than a row count, which would
    re-pin the swept constants every time they move: the transcript keeps rows,
    every child stays reachable, and the way back is on screen. A count would
    also be the wrong assertion for the reason the module notes give — the
    budgets answer to the screen, so the number is a function of the size.
    """
    from local_operator.tools import builtin

    session = FakeSession()
    # A resumed session's shape: more children than any preview shows, plus a
    # plan, which is what puts three panels in the band at once.
    session.jobs = _fake_jobs(
        *(_Job(f"sub-{n}", f"child number {n}", status="completed") for n in range(1, 20))
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": f"step {n} of the plan", "status": "pending"} for n in range(1, 13)
        ]
        app._refresh_band()
        for _ in range(8):
            await pilot.pause()

        transcript = app.query_one("#transcript")
        panel = app.query_one(SubagentPanel)

        # The FLOOR, not merely "not zero". A `> 0` assertion passes on the
        # unfixed panel — this fixture leaves it two rows where the operator's
        # real session, which also docks a wake list, had none — so it would
        # pin nothing. The panels' budgets are what guarantee the floor, and
        # asserting it is what keeps their swept constants answerable to a test.
        assert transcript.size.height >= _MIN_READABLE_TRANSCRIPT_ROWS, (
            f"the restored dock left {transcript.size.height} conversation rows; "
            "the dock is chrome and may not take the screen"
        )
        # Nothing is clipped upward and the dock fits the screen it is drawn in.
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)
        # Every child stays REACHABLE even where none is listed: the caption
        # carries the total and the affordance carries the way to them.
        assert panel._affordance.display, "no escape route from a truncated roster"
        assert len(panel._rows) == 19, "restored children were dropped, not folded"
        if not any(row.display for row in panel._rows.values()):
            assert "19" in str(
                panel._header.content
            ), "the compact caption must carry the count it is standing in for"


@pytest.mark.asyncio
async def test_ctrl_g_still_reaches_every_child_from_the_compact_roster() -> None:
    """The escape route has to work at the size that needs it.

    U7's severity was not that the roster was large but that ``ctrl+g`` could
    not recover the conversation: the COLLAPSED dock alone already exceeded the
    short screen, so the affordance existed in principle and was unreachable in
    practice. The toggle's own refusal was part of it — it was gated on the flat
    ``_PREVIEW_JOB_ROWS`` constant, so on a screen whose budget hid rows below
    that number the key was a silent no-op.
    """
    from local_operator.tools import builtin

    session = FakeSession()
    # FEWER children than the flat `_PREVIEW_JOB_ROWS` ceiling, which is the
    # case the old gate got wrong: with a height-aware budget the preview hides
    # rows well before six, so a refusal keyed on the constant made `ctrl+g` a
    # silent no-op on exactly the screens where it is the only way to them.
    # A docked plan beside it is what shrinks the budget that far — a roster
    # alone keeps its full preview and would not exercise the gate at all.
    session.jobs = _fake_jobs(
        *(_Job(f"sub-{n}", f"child number {n}", status="completed") for n in range(1, 5))
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": f"step {n} of the plan", "status": "pending"} for n in range(1, 13)
        ]
        app._refresh_band()
        for _ in range(8):
            await pilot.pause()
        panel = app.query_one(SubagentPanel)
        collapsed_visible = sum(1 for row in panel._rows.values() if row.display)

        await pilot.press("ctrl+g")
        for _ in range(8):
            await pilot.pause()
        expanded_visible = sum(1 for row in panel._rows.values() if row.display)
        assert (
            expanded_visible > collapsed_visible
        ), "ctrl+g did not reveal any child; the escape route is a no-op"
        # Expanding is an explicit request for the roster, so it MAY borrow from
        # the transcript — but it must still leave the composer on screen.
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)

        # The press after the expanded roster collapses it AND shrinks the
        # panel to the summary row (#525 cycle), which on this short screen
        # is the strongest form of "gives the conversation its rows back".
        await pilot.press("ctrl+g")
        for _ in range(8):
            await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert sum(1 for row in panel._rows.values() if row.display) == 0
        assert (
            app.query_one("#transcript").size.height > 0
        ), "collapsing did not give the conversation its rows back"
        # Two more presses round the cycle: hidden, then the preview as it was.
        await pilot.press("ctrl+g")
        await pilot.press("ctrl+g")
        for _ in range(8):
            await pilot.pause()
        assert panel.density is Density.FULL
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)
        # Coming back from hidden is the ARRIVAL path — the todo list grew into
        # the rows while the panel was gone, and the two slots re-divide the
        # band over the next polls exactly as they do on `main` when a roster
        # first appears beside a docked plan (`slot_rows` keeps the larger of
        # measured and predicted, so the todo's stale height withholds rows
        # for a frame). Converge through the poll rather than asserting the
        # first frame; the settled preview must be the one the session began on.
        for _ in range(4):
            app._refresh_band()
            for _ in range(4):
                await pilot.pause()
        assert sum(1 for row in panel._rows.values() if row.display) == collapsed_visible


async def _boot_with_jobs(app: OperatorApp, pilot: Any) -> SubagentPanel:
    """Wait for the session, paint the band, and return the panel."""
    for _ in range(80):
        await pilot.pause()
        if app._session is not None:
            break
    app._refresh_band()
    for _ in range(4):
        await pilot.pause()
    return app.query_one(SubagentPanel)


@pytest.mark.asyncio
async def test_ctrl_g_cycles_full_summary_hidden_with_three_children() -> None:
    """#525 case 1: the common ≤6-child roster, where the key used to no-op.

    Every stop is checked against the geometry the app budgets from
    (``predicted_rows``: header+3 → 1 → not displayed → header+3) and against
    the screen never becoming scrollable, the invariant the band tests share.
    """
    session = FakeSession()
    session.jobs = _fake_jobs(*[_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)])
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        editor = app.query_one(Editor)
        assert panel.density is Density.FULL
        assert panel.predicted_rows() == 4
        # No overflow, so the hint lives in the caption (design §3).
        # A gap, not the counts' ` · ` seam, so the hint does not scan as a
        # statistic beside the label (round 1, D3).
        assert panel.summary_text() == "Subagents   ctrl+g"
        assert panel._affordance.display is False

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert panel.display is True
        assert panel.predicted_rows() == 1
        assert panel._list.display is False
        assert panel.summary_text().startswith("Subagents · 3 running ")
        assert panel.summary_text().endswith(" · ctrl+g")
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)
        assert app.focused is editor

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.HIDDEN
        assert panel.display is False
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)
        # The 1 Hz poll must not un-hide it (design §8).
        app._refresh_band()
        await pilot.pause()
        assert panel.display is False
        assert panel._spinner_timer is None, "a hidden panel must not keep ticking"

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.FULL
        assert panel.display is True
        assert panel._expanded is False
        assert panel.predicted_rows() == 4
        assert sum(1 for row in panel._rows.values() if row.display) == 3
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)
        assert panel._spinner_timer is not None, "running rows must animate again"


@pytest.mark.asyncio
async def test_ctrl_g_with_overflow_expands_then_summarises_then_hides() -> None:
    """#525 case 2: with 30 children the full state has two stops.

    Expanded enters navigation (focus on a row, as before); the NEXT press
    leaves navigation, collapses the roster and shrinks to summary in one
    step; then hidden; then the preview.
    """
    session = FakeSession()
    session.jobs = _fake_jobs(*[_Job(f"sub-{n:02d}", f"task {n:02d}") for n in range(1, 31)])
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        editor = app.query_one(Editor)
        # With overflow the affordance says the hint; the caption does not
        # repeat it.
        assert panel.summary_text() == "Subagents"
        assert str(panel._affordance.content) == "+24 earlier · ctrl+g to expand"

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel._expanded is True
        assert isinstance(app.focused, SubagentRow)

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert panel._expanded is False
        assert app.focused is editor
        assert panel.predicted_rows() == 1
        # The spinner frame between the count and the hotkey varies per tick.
        text = panel.summary_text()
        assert text.startswith("Subagents · 30 running ") and text.endswith(" · ctrl+g")

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.HIDDEN
        assert panel.display is False

        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.FULL
        assert panel._expanded is False
        assert sum(1 for row in panel._rows.values() if row.display) == 6
        assert str(panel._affordance.content) == "+24 earlier · ctrl+g to expand"
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)


def test_summary_row_counts_each_state_and_sheds_segments_in_order() -> None:
    """#525 case 3: the summary row's content and its width ladder.

    Zero segments are omitted, the spinner is present iff something is
    running, and shedding drops whole segments in the documented order
    without ever truncating into ``ctrl+g``.
    """
    counts = SummaryCounts(running=1, queued=2, done=3, failed=1, cancelled=1, interrupted=1)
    wide = compose_summary(counts, spinner_glyph="⣾", width=200).plain
    assert wide == (
        "Subagents · 1 running ⣾ · 3 done · 1 failed · 2 queued · 1 cancelled · 1 interrupted"
        " · ctrl+g"
    )
    # Ink: `failed` is the only danger segment; the hotkey is muted.
    row = compose_summary(counts, spinner_glyph="⣾", width=200)
    assert _ink(_style_at(row, "1 failed")) == theme_mod.semantic_color("danger")
    assert _ink(_style_at(row, "ctrl+g")) == theme_mod.semantic_color("muted")
    assert _ink(_style_at(row, "3 done")) == theme_mod.semantic_color("dim")

    # Zero counts are omitted, and no spinner when nothing runs.
    settled = compose_summary(SummaryCounts(done=4, failed=1), spinner_glyph="", width=200).plain
    assert settled == "Subagents · 4 done · 1 failed · ctrl+g"
    assert "⣾" not in settled and "running" not in settled

    # The ladder, from a width that fits everything down to one that fits
    # only the compressed failed glyph and the hotkey. Each step is a strict
    # subset of the segments before it and the hotkey is always intact.
    floor = "⣾ · ✗1 · ctrl+g"
    seen: list[str] = []
    for width in range(len(wide), 8, -1):
        text = compose_summary(counts, spinner_glyph="⣾", width=width).plain
        assert text.endswith("ctrl+g"), (width, text)
        # Below the floor there is nothing left to shed; the renderer clips.
        assert len(text) <= width or text == floor, (width, text)
        if not seen or seen[-1] != text:
            seen.append(text)
    # The documented order: cancelled, interrupted, queued, done, label,
    # running count (glyph stays), failed count (glyph stays).
    assert seen[0] == wide
    assert (
        seen[1] == "Subagents · 1 running ⣾ · 3 done · 1 failed · 2 queued · 1 interrupted · ctrl+g"
    )
    assert seen[2] == "Subagents · 1 running ⣾ · 3 done · 1 failed · 2 queued · ctrl+g"
    assert seen[3] == "Subagents · 1 running ⣾ · 3 done · 1 failed · ctrl+g"
    assert seen[4] == "Subagents · 1 running ⣾ · 1 failed · ctrl+g"
    assert seen[5] == "1 running ⣾ · 1 failed · ctrl+g"
    assert seen[6] == "⣾ · 1 failed · ctrl+g"
    assert seen[7] == floor
    assert seen[-1] == floor


@pytest.mark.asyncio
async def test_summary_row_at_fifty_columns_keeps_the_hotkey() -> None:
    """#525 case 3, on the real panel: 50x16 sheds segments, never ctrl+g."""
    session = FakeSession()
    session.jobs = _fake_jobs(
        _Job("sub-1", "child one"),
        _Job("sub-2", "child two"),
        _Job("sub-3", "child three"),
        _Job("sub-4", "child four", status="completed"),
        _Job("sub-5", "child five", status="completed"),
        _Job("sub-6", "child six", status="failed"),
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(50, 16)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        # Height-forced compact on this screen paints the SAME summary row
        # (design §8: one vocabulary for the two zero-row shapes) — so go to
        # summary explicitly and assert the row either way.
        while panel.density is not Density.SUMMARY:
            await pilot.press("ctrl+g")
            await pilot.pause()
        from rich.cells import cell_len

        text = panel.summary_text()
        assert text.endswith("ctrl+g"), text
        assert "3 running" in text and "1 failed" in text, text
        # `cell_len`, not `len`: `compose_summary` budgets in CELLS because
        # the spinner and `✗` are width-sensitive, so a character count
        # asserts a different quantity than the code guarantees and passes
        # only while the glyphs happen to be single-width (round 1, F6).
        assert cell_len(text) <= panel._row_width(), (text, panel._row_width())
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)


@pytest.mark.asyncio
async def test_a_failure_promotes_hidden_to_summary_and_a_start_does_not() -> None:
    """#525 case 4: re-emergence rules.

    Hidden + a child STARTING stays hidden (a preference for a small dock is
    a preference; the band's counter says children run). Hidden + a child
    FAILING becomes summary — one row, never full — and clears the user pin
    so the next press re-hides. Summary + failure stays summary with the
    count moved.
    """
    session = FakeSession()
    jobs = [_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)]
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        await pilot.press("ctrl+g")
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.HIDDEN
        assert panel._user_density is True

        # A fourth child starts: still hidden.
        jobs.append(_Job("sub-4", "child task 4"))
        session.jobs = _fake_jobs(*jobs)
        app.post_message(SubagentStarted("sub-4", "child task 4"))
        await pilot.pause()
        await pilot.pause()
        assert panel.density is Density.HIDDEN
        assert panel.display is False

        # A cancel is not a failure: still hidden.
        jobs[1].status = "cancelled"
        app.post_message(SubagentEnded("sub-2", "child task 2", "cancelled"))
        await pilot.pause()
        await pilot.pause()
        assert panel.density is Density.HIDDEN

        # A failure breaks through, to ONE row, and unpins.
        jobs[0].status = "failed"
        app.post_message(SubagentEnded("sub-1", "child task 1", "failed"))
        await pilot.pause()
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert panel.display is True
        assert panel.predicted_rows() == 1
        assert panel._user_density is False
        text = panel.summary_text()
        assert "1 failed" in text and "2 running" in text and "1 cancelled" in text, text
        assert tuple(app.screen.virtual_size) == tuple(app.screen.size)

        # Summary + another failure: still summary, count moves.
        jobs[2].status = "failed"
        app.post_message(SubagentEnded("sub-3", "child task 3", "failed"))
        await pilot.pause()
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert "2 failed" in panel.summary_text(), panel.summary_text()

        # The next press reads naturally: summary → hidden.
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.HIDDEN


@pytest.mark.asyncio
async def test_collapsed_preview_keeps_failed_children_over_completed_ones() -> None:
    """#525 case 6: the budgeted slice is a priority pick, not the newest tail.

    Eight children, budget six, the two OLDEST failed and the two NEWEST
    completed: the failed pair is shown, the affordance says ``+2 more``
    (the hidden set is no longer the roster's prefix), and DOM order is
    unchanged.
    """
    session = FakeSession()
    jobs = [_Job(f"sub-{n:02d}", f"task {n:02d}") for n in range(1, 9)]
    jobs[0].status = "failed"
    jobs[1].status = "failed"
    jobs[6].status = "completed"
    jobs[7].status = "completed"
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        assert panel._preview_job_rows() == 6
        visible = [row.job_id for row in panel.query(SubagentRow) if row.display]
        assert visible == ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
        assert str(panel._affordance.content) == "+2 more · ctrl+g to expand"

        # When the hidden set IS the prefix the old wording stands.
        for job in jobs:
            job.status = "running"
        session.jobs = _fake_jobs(*jobs)
        app._refresh_band()
        await pilot.pause()
        visible = [row.job_id for row in panel.query(SubagentRow) if row.display]
        assert visible == [f"sub-{n:02d}" for n in range(3, 9)]
        assert str(panel._affordance.content) == "+2 earlier · ctrl+g to expand"


@pytest.mark.asyncio
async def test_display_dock_seeds_the_initial_density_and_live_applies_unless_pinned(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#525 case 7: the setting is where a session STARTS, never an override.

    ``display.dock: summary`` in an isolated config paints summary on the
    first non-empty sync; ``ctrl+g`` still cycles from it; a live write
    applies while the user has not chosen, and is ignored once they have.
    """
    import yaml

    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.tui.settings import settings_reload

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"values": {"display.dock": "summary"}}))
    settings_reload()
    # Writes go through the facade the `/settings` page uses, so the process
    # watcher's snapshot (which the fast-path reader prefers once the app has
    # started one) and the display cache both move, exactly as a page edit
    # would move them. A bare file write would be read by nobody until the
    # watcher's next poll.
    manager = ConfigManager(tmp_path)
    dock = settings_io.BY_KEY["display.dock"]
    try:
        session = FakeSession()
        session.jobs = _fake_jobs(*[_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)])
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            panel = await _boot_with_jobs(app, pilot)
            assert panel.density is Density.SUMMARY
            assert panel._user_density is False
            assert panel.predicted_rows() == 1

            # A live write with no user choice applies.
            settings_io.write_setting(manager, dock, "hidden")
            app._apply_dock_density()
            await pilot.pause()
            assert panel.density is Density.HIDDEN
            assert panel.display is False

            # ctrl+g cycles from wherever the setting put it.
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert panel.density is Density.FULL
            assert panel._user_density is True

            # Now pinned: a live write does not fight the user.
            settings_io.write_setting(manager, dock, "summary")
            app._apply_dock_density()
            await pilot.pause()
            assert panel.density is Density.FULL

            # A session swap forgets the pin and re-reads the setting.
            panel.reset_density()
            app._refresh_band()
            await pilot.pause()
            assert panel.density is Density.SUMMARY
            assert panel._user_density is False
    finally:
        settings_reload()


@pytest.mark.asyncio
async def test_hiding_the_roster_beside_a_todo_list_never_shrinks_the_transcript() -> None:
    """Round 1 F1/D1/Q1: the rows a hidden roster frees go to the CONVERSATION.

    The coverage gap the whole round turned on. Every other density test docks
    the roster alone, where the arithmetic is trivially right; with a todo
    panel sharing the band each panel sizes against the other's CURRENT
    height, so `display = False` made the roster contribute zero rows, the
    shared transcript floor evaporated, and the todo panel absorbed the rows
    the user had just asked for — the dock ending TALLER than in `full`
    (measured 100x24: transcript 8 → 11 → 5, dock 14 → 11 → 17).

    Asserts the invariant rather than three fixed numbers: the exact heights
    depend on the todo panel's own budget ladder, but "pressing ctrl+g never
    costs the transcript rows" is the promise the feature makes.
    """
    from local_operator.tools import builtin

    session = FakeSession()
    session.jobs = _fake_jobs(*[_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)])
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        todo = app.query_one(TodoPanel)
        builtin.TODO_STORE["sess"] = [
            {"text": f"step {n} of the plan", "status": "pending"} for n in range(1, 13)
        ]
        # Let the BAND settle before measuring anything. The todo panel walks
        # to its own budget over a couple of polls, and a baseline captured
        # mid-walk attributes that panel's convergence to the keypress —
        # which is a measurement artifact, not the defect under test.
        for _ in range(3):
            app._refresh_band()
            for _ in range(6):
                await pilot.pause()
        assert todo.display, "this test is meaningless without a todo panel docked"

        transcript = app.query_one("#transcript")

        async def settled_height() -> int:
            """The transcript once the band has stopped moving.

            Both panels re-budget against each other over a couple of polls,
            so every reading here is taken the SAME way — settled — or the
            comparison measures pump depth rather than the density change.
            """
            for _ in range(3):
                app._refresh_band()
                for _ in range(6):
                    await pilot.pause()
            return transcript.size.height

        heights = [await settled_height()]
        densities = [panel.density]
        for _ in range(2):
            await pilot.press("ctrl+g")
            for _ in range(4):
                await pilot.pause()
            heights.append(await settled_height())
            densities.append(panel.density)

        assert densities == [Density.FULL, Density.SUMMARY, Density.HIDDEN], densities
        # The invariant. `>=` rather than `>`: on a short screen a density step
        # may be absorbed entirely by the floor, which is fine — what must
        # never happen is the conversation paying for the user's own request
        # for more room.
        assert heights[1] >= heights[0], f"summary took transcript rows: {heights}"
        assert heights[2] >= heights[1], f"hiding took transcript rows: {heights}"
        # And the dock must not grow across the same presses.
        assert app.query_one("#input-dock").size.height <= 24, heights


@pytest.mark.asyncio
async def test_children_settling_while_summarised_return_the_dock_to_idle() -> None:
    """Round 1 F2: a settled dock in `summary` must stop animating.

    `row.running` is written only in `SubagentRow.paint`, which never runs in
    `summary` because no row is displayed — so the tick's stop condition read
    a flag frozen at the last `full` frame and kept a 12.5 fps timer on the
    keyboard thread for the rest of the session (measured 12.7 ticks/sec on a
    fully settled dock against 1.0/sec in `full`). The children settle AFTER
    the press here, which is precisely the ordering every existing summary
    test misses.
    """
    session = FakeSession()
    jobs = [_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)]
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert panel._spinner_timer is not None, "a running roster should animate"

        # Settle every child WHILE summarised: no row repaints, so the rows'
        # own flags stay stale and only the ledger knows.
        for job in jobs:
            job.status = "completed"
        session.jobs = _fake_jobs(*jobs)
        app._refresh_band()
        for _ in range(8):
            await pilot.pause()

        assert all(
            row.running for row in panel._rows.values()
        ), "precondition: the stale row flags are exactly what made this a bug"
        assert "3 done" in panel.summary_text(), panel.summary_text()

        # The timer stops itself from INSIDE `_tick`, so wait for that tick to
        # happen rather than for a pump count: at a 0.08 s interval a fixed
        # number of `pause()` calls bets on how much wall time a loaded xdist
        # worker gets, which is the clock-dependence AGENTS.md warns about (it
        # failed 2 runs in 3 under `-n2` while passing alone every time).
        for _ in range(60):
            if panel._spinner_timer is None:
                break
            await asyncio.sleep(0.05)
            await pilot.pause()
        assert panel._spinner_timer is None, "settled summary is still animating (F2)"

        # And it comes back when work does, rather than being stopped for good.
        jobs.append(_Job("sub-4", "child task 4"))
        session.jobs = _fake_jobs(*jobs)
        app._refresh_band()
        for _ in range(8):
            await pilot.pause()
        assert panel._spinner_timer is not None, "a new child did not restart the spinner"


@pytest.mark.asyncio
async def test_the_band_counts_queued_children_so_a_hidden_dock_is_never_silent() -> None:
    """Round 1 U2: hiding is justified by the band, so the band must speak.

    The design lets a LIVE panel disappear because `◍ N agents` still reports
    the children. That counter excluded queued children, so a hidden dock
    whose children were all queued reported nothing anywhere — the panel's
    own stated fallback, empty for exactly the state the panel was showing.
    """
    session = FakeSession()
    jobs = [_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)]
    for job in jobs:
        job.queued = True
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        assert app._job_count("task") == 3, "queued children are still delegated work"

        await pilot.press("ctrl+g")
        for _ in range(4):
            await pilot.pause()
        # Summary names them as queued, which is the state the band must not
        # then contradict by falling silent.
        assert "3 queued" in panel.summary_text(), panel.summary_text()

        await pilot.press("ctrl+g")
        for _ in range(4):
            await pilot.pause()
        assert panel.density is Density.HIDDEN
        # The fallback the design leans on is present rather than empty.
        assert app._job_count("task") == 3
        assert format_agents(app._job_count("task")) == "3 agents"


@pytest.mark.asyncio
async def test_the_gate_admitting_a_queued_child_updates_the_summary_caption() -> None:
    """Round 2 Q2/F7: the counts memo must see a queued→running promotion.

    The one ledger move that changes NEITHER the job's identity nor its
    status. A queued child already carries ``status == "running"``
    (`summary_counts` buckets on ``facts.queued`` first), and
    ``AsyncJobManager.start_queued`` clears ``job.queued`` in place when the
    capacity gate opens — so a memo keyed on ``(id, status)`` alone returns
    the pre-admission counts and the caption reports children as queued
    while they are actually running, indefinitely, until some unrelated job
    happens to move its status.

    In `summary` that caption is the panel's ONLY representation, which is
    what made a cheap-path optimisation into a user-visible wrong number.
    """
    session = FakeSession()
    jobs = [_Job(f"sub-{n}", f"child task {n}") for n in range(1, 4)]
    # One running, two parked behind the gate — `queued` set without touching
    # `status`, exactly as the manager holds them.
    jobs[1].queued = True
    jobs[2].queued = True
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        panel = await _boot_with_jobs(app, pilot)
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert panel.density is Density.SUMMARY
        assert "2 queued" in panel.summary_text(), panel.summary_text()
        assert "1 running" in panel.summary_text(), panel.summary_text()

        # The gate opens: `start_queued` flips this flag in place and leaves
        # `status` alone (`harness/jobs.py`, which even guards on the status
        # already being "running").
        jobs[1].queued = False
        jobs[2].queued = False
        session.jobs = _fake_jobs(*jobs)
        app._refresh_band()
        for _ in range(8):
            await pilot.pause()

        caption = panel.summary_text()
        assert "3 running" in caption, caption
        assert "queued" not in caption, caption
