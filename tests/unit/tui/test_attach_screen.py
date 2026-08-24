"""AttachScreen pilot tests — banner, transcript rows, steer routing,
pending gates, owner death, and /detach — against a REAL OperatorApp host
(CSS-bearing, per AGENTS.md's visual-validation rule)."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.attach_screen import AttachScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def make_record(pid: int = 4242) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id="sess-42",
        conversation_name="Attach Demo",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
    )


def make_projection(
    *, streaming: bool = False, pending: PendingRequest | None = None
) -> SessionProjection:
    return SessionProjection(
        session_id="sess-42",
        pid=4242,
        kind="tui",
        conversation_name="Attach Demo",
        cwd="/tmp",
        model_label="test/model",
        streaming=streaming,
        activity="auditing merged MRs" if streaming else "",
        transcript=[
            TranscriptEntry(id="u1", kind="user", text="summarise the ingest path"),
            TranscriptEntry(id="a1", kind="assistant", text="The ingest path has three stages"),
            TranscriptEntry(id="t1", kind="tool", tool_name="bash", summary="grep -r ingest src/"),
            TranscriptEntry(id="n1", kind="notice", text="queued as steering"),
        ],
        pending=pending,
    )


class StubClient:
    """Stands in for AttachClient once the screen is up: records routed ops."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, tuple[object, ...]]] = []
        self.connected = True

    async def connect(self, record, session_id):  # noqa: ANN001, ANN202
        pass

    async def prompt(self, text: str) -> str:
        self.sent.append(("prompt", (text,)))
        return "prompt sent"

    async def steer(self, text: str) -> str:
        self.sent.append(("steer", (text,)))
        return "steering queued"

    async def approval_answer(self, request_id: str, approved: bool) -> str:
        self.sent.append(("approval", (request_id, approved)))
        return "approved" if approved else "denied"

    async def ask_answer(self, request_id: str, value: str) -> str:
        self.sent.append(("ask", (request_id, value)))
        return "answered"

    def close(self) -> None:
        self.connected = False


def _text(widget) -> str:  # noqa: ANN001
    from rich.text import Text

    r = widget.renderable if hasattr(widget, "renderable") else widget.content
    return r.plain if isinstance(r, Text) else str(r)


async def _attached_screen(app: OperatorApp) -> tuple[AttachScreen, StubClient]:
    screen = AttachScreen(make_record(), "sess-42")
    app.push_screen(screen)
    # Let the (unreachable) connect task fail into the dead-state first; the
    # stub replaces it for deterministic routing assertions.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    stub = StubClient()
    screen._owner_dead = False
    screen._client = stub  # type: ignore[assignment]
    return screen, stub


@pytest.mark.asyncio
async def test_banner_and_transcript_rows() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, _ = await _attached_screen(app)
        screen._render_projection(make_projection())
        await pilot.pause()
        assert "attached" in _text(screen._banner)
        assert "pid 4242" in _text(screen._banner)
        assert "Attach Demo" in _text(screen._banner)
        assert "/detach" in _text(screen._banner)
        rows = [c for c in screen._transcript.children]
        texts = [_text(r) for r in rows]
        assert "❯ summarise the ingest path" in texts
        assert "The ingest path has three stages" in texts
        assert "· grep -r ingest src/" in texts


@pytest.mark.asyncio
async def test_submit_routes_to_steer_when_streaming() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, stub = await _attached_screen(app)
        screen._render_projection(make_projection(streaming=True))
        await pilot.pause()
        screen._composer.value = "focus on the retry path"
        screen._composer.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert stub.sent == [("steer", ("focus on the retry path",))]


@pytest.mark.asyncio
async def test_submit_routes_to_prompt_when_idle() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, stub = await _attached_screen(app)
        screen._render_projection(make_projection())
        await pilot.pause()
        screen._composer.value = "what are the three stages?"
        screen._composer.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert stub.sent == [("prompt", ("what are the three stages?",))]


@pytest.mark.asyncio
async def test_pending_approval_renders_and_y_n_answers() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, stub = await _attached_screen(app)
        screen._render_projection(
            make_projection(
                pending=PendingRequest(request_id="r1", kind="approval", title="bash: rm -rf build")
            )
        )
        await pilot.pause()
        assert screen._pending.display
        assert "rm -rf build" in _text(screen._pending)
        # Composer holds focus; y/n still answer because the handler checks
        # focus only for ask pickers. Press with the composer BLURRED.
        screen._composer.blur() if hasattr(screen._composer, "blur") else None
        screen.set_focus(None)
        await pilot.press("y")
        await pilot.pause()
        assert ("approval", ("r1", True)) in stub.sent


@pytest.mark.asyncio
async def test_owner_death_state() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, _ = await _attached_screen(app)
        screen._render_projection(make_projection())
        await pilot.pause()
        screen._owner_exited("owner exited")
        await pilot.pause()
        assert "owner exited" in _text(screen._banner)
        assert "resume here" in _text(screen._pending)
        assert screen._owner_dead


@pytest.mark.asyncio
async def test_detach_command_pops_the_screen() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, _ = await _attached_screen(app)
        screen._render_projection(make_projection())
        await pilot.pause()
        screen._composer.value = "/detach"
        screen._composer.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert screen not in app.screen_stack


@pytest.mark.asyncio
async def test_escape_detaches() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, _ = await _attached_screen(app)
        screen._render_projection(make_projection())
        await pilot.pause()
        screen.set_focus(screen._transcript) if hasattr(screen, "set_focus") else None
        await pilot.press("escape")
        await pilot.pause()
        assert screen not in app.screen_stack
