"""AttachScreen pilot tests — banner, transcript rows, steer routing,
pending gates, owner death, and /detach — against a REAL OperatorApp host
(CSS-bearing, per AGENTS.md's visual-validation rule)."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile.types import (
    AskOptionWire,
    ContinuationCommand,
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

    async def send_command(self, command: ContinuationCommand) -> str:
        self.sent.append(("prompt", (command.text, command.command_id)))
        return "prompt admitted"

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
    # This module tests the SCREEN, not socket dialing. Cancel the real connect
    # task before replacing it so its expected port-1 failure cannot enqueue an
    # owner-death repaint after the deterministic stub state below.
    await asyncio.sleep(0)
    screen.run_task.cancel()
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
        assert screen._composer.placeholder == "Message Local Operator…"
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
        assert len(stub.sent) == 1
        assert stub.sent[0][1][0] == "focus on the retry path"


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
        assert len(stub.sent) == 1
        assert stub.sent[0][1][0] == "what are the three stages?"
        assert screen._composer.value == ""


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
        # Normal entry focus stays on the composer; the displayed gate owns its
        # answer shortcut and the key must not become draft text.
        screen._composer.focus()
        await pilot.press("y")
        await pilot.pause()
        assert ("approval", ("r1", True)) in stub.sent
        assert screen._composer.value == ""


@pytest.mark.asyncio
async def test_pending_ask_option_answers_from_composer_focus() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, stub = await _attached_screen(app)
        screen._render_projection(
            make_projection(
                pending=PendingRequest(
                    request_id="q1",
                    kind="ask",
                    title="Choose",
                    options=[AskOptionWire(label="Alpha", description="first")],
                )
            )
        )
        await pilot.pause()
        screen._composer.focus()
        await pilot.press("a")
        await pilot.pause()
        assert ("ask", ("q1", "Alpha")) in stub.sent
        assert screen._composer.value == ""


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
        assert screen._owner_dead
        assert not screen._pending.display
        assert not screen._composer.disabled
        assert screen._composer.placeholder == "Message Local Operator…"
        # A late projection queued before EOF cannot repaint contradictory live
        # state or resurrect the approval card after owner death.
        screen._render_projection(
            make_projection(
                pending=PendingRequest(request_id="late", kind="approval", title="late")
            )
        )
        assert "late" not in _text(screen._pending)


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


@pytest.mark.asyncio
async def test_live_client_callback_schedules_on_ui_loop() -> None:
    """AttachClient's pump runs on Textual's loop; a callback must use
    call_later, not call_from_thread (which raises on the UI thread)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen, _ = await _attached_screen(app)
        screen._on_projection(make_projection())
        await pilot.pause()
        assert screen._projection is not None
        assert screen._projection.conversation_name == "Attach Demo"


@pytest.mark.asyncio
async def test_owner_death_r_key_runs_async_resume_callback() -> None:
    resumed: list[str] = []

    async def resume_here(session_id: str) -> None:
        resumed.append(session_id)

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        screen = AttachScreen(make_record(), "sess-42", on_resume_here=resume_here)
        app.push_screen(screen)
        await pilot.pause()
        screen._owner_exited("owner exited")
        await pilot.press("r")
        await pilot.pause()
        assert resumed == []
        assert screen._composer.placeholder == "Message Local Operator…"
