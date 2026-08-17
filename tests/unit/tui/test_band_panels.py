"""The dock band panels (todo + subagent) — the new chrome above the composer.

The build wave wrote these widgets but never mounted or tested them; the
integration committed them into ``app.py`` compose() with a 1 Hz poll and
Subagent* event handlers. These tests drive the REAL mounted panels through
the app, so visibility toggling (empty vs populated), the refresh path, and
the trajectory modal are asserted the way a user reaches them — not as
isolated widget calls that can pass while the wiring is dead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult

from local_operator.harness.types import ImageContent
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_panel import SubagentPanel
from local_operator.tui.widgets.todo_panel import MAX_TODO_ROWS, TodoPanel
from local_operator.session.naming import ConversationName


class FakeSession:
    """Minimal SessionProtocol the app can boot against."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.aborts: list[str] = []
        self.disposed = False
        self._handlers: list[Any] = []
        self.jobs: Any = None
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

    def set_model(self, model: Any) -> None:
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

    def set_approval_handler(self, handler: Any | None) -> None:
        self.approval_handler = handler

    def set_ask_handler(self, handler: Any | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot; fakes only
        # need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

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
        self.result_text: str | None = None
        self.error_text: str | None = None
        self.settled_at: float | None = None
        self.trajectory: list[dict[str, Any]] | None = None


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
        await pilot.pause()
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
