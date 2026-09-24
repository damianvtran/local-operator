"""A single /goal Enter must reach the real engine and execute a real tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.rows import is_harness_chrome
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
            # THE JUDGE. Setting a goal with `/goal <text>` arms it, so this turn's
            # END is the edge that starts a `complete_aside` — the app's own judge
            # call, off the record. `CONTINUE` here is deliberate: it is what makes
            # the continuation row exist at all, which is the row the assertions
            # below prove is recorded and never painted.
            text_turn("VERDICT: CONTINUE\nThe artifact exists but has not been verified."),
            text_turn("Verified: the artifact is there and the goal is met."),
            # The second judge call, on the continuation turn it just admitted.
            text_turn("VERDICT: ACHIEVED\nThe artifact exists and the goal is met."),
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
                # FIVE provider calls, and each one is named: the goal turn's tool
                # call, its summary, the judge's verdict, the continuation the
                # verdict admitted, and the judge's second verdict (the one that
                # settles). A number without that list reads as flake the next time
                # somebody adds a turn.
                assert len(stream.requests) == 5
                # THE CHROME ROW IS NEVER PAINTED. The continuation is a real
                # persisted user-role row (asserted below), and this is the
                # assertion that no surface paints the harness's words as the
                # human's: exactly one UserBlock, carrying what was typed.
                assert [row.text() for row in app.query(UserBlock)] == [request]
                assert "The goal artifact is ready." in transcript_text(app)
                history = [item for item in session.history() if isinstance(item, Message)]
                users = [message for message in history if message.role == "user"]
                # TWO durable user rows: what the human typed, then the chrome
                # prompt the judge admitted. The transcript records why the
                # conversation continued; the column above shows only the human.
                assert len(users) == 2
                content = users[0].content[0]
                assert isinstance(content, TextContent)
                assert content.text == request
                chrome = users[1].content[0]
                assert isinstance(chrome, TextContent)
                assert is_harness_chrome(chrome.text)
                assert users[1].provider_payload == {"harness_injected": True}
                assert session.goal_status == "done", "the judge's second verdict settles"
                assert any(message.role == "tool" for message in history)
                records = _transcript_records(directory)
                assert _has_tool_call(records, "write")
                assert _has_tool_result(records, "write")
    finally:
        await dispose_quietly(session)
