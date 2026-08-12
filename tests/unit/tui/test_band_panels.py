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
from textual.app import App, ComposeResult

from local_operator.harness.types import ImageContent
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_panel import SubagentPanel
from local_operator.tui.widgets.todo_panel import TodoPanel


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
        return ""

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return (text or "").strip()

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
        assert "1/2" in body_plain


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
