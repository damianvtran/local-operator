"""A single /goal Enter must reach the real engine and execute a real tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.tools.builtin import build_write_tool
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import UserBlock
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    drain,
    text_turn,
    tool_call_turn,
    transcript_text,
    wait_for_adoption,
)
from tests.e2e.test_tui_e2e import _has_tool_call, _has_tool_result, _transcript_records
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_one_goal_command_executes_and_persists_a_real_tool(headless_tui_env: Path):
    target = headless_tui_env / "goal-artifact.txt"
    request = "Write the goal artifact"
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Writing the goal artifact.",
                tool_name="write",
                tool_call_id="goal-write-1",
                arguments={"path": str(target), "content": "goal command started this turn"},
            ),
            text_turn("The goal artifact is ready."),
        ]
    )
    directory = headless_tui_env / "sessions" / "goal-submit"
    session = build_session(directory, stream, tools=[build_write_tool()], cwd=headless_tui_env)
    # Naming is another provider request, not part of the turn under test.
    session.set_conversation_name("Goal regression", user_set=True)

    async def factory():
        return session

    app = OperatorApp(factory)
    try:
        with bounded(60, "goal composer submission"):
            async with app.run_test(size=(100, 30)) as pilot:
                await wait_for_adoption(app, pilot)
                await drain(pilot)
                app._editor().load_text(f"/goal {request}")
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await drain(pilot, cycles=20)
                assert session.goal == request
                assert target.read_text() == "goal command started this turn"
                assert len(stream.requests) == 2
                assert [row.text() for row in app.query(UserBlock)] == [request]
                assert "The goal artifact is ready." in transcript_text(app)
                history = [item for item in session.history() if isinstance(item, Message)]
                users = [message for message in history if message.role == "user"]
                assert len(users) == 1
                content = users[0].content[0]
                assert isinstance(content, TextContent)
                assert content.text == request
                assert any(message.role == "tool" for message in history)
                records = _transcript_records(directory)
                assert _has_tool_call(records, "write")
                assert _has_tool_result(records, "write")
    finally:
        await dispose_quietly(session)
