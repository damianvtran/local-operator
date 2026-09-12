"""Design round 3: what a person sees when they paste a failover notice verbatim.

Three cases over the same short conversation, driven through the real
``OperatorApp`` (so ``local_operator.tcss`` applies):

* ``no_paste``  — the conversation without the paste (the control);
* ``pasted``    — the same conversation with the notice pasted as a whole message;
* ``quoted``    — the notice quoted INSIDE a sentence, which the head test spares.

The control is what makes the answer legible: if ``pasted`` and ``no_paste`` are
the same frame, the paste leaves no trace at all on screen.

    env -u NO_COLOR -u CMUX_WORKSPACE_ID -u CMUX_SESSION_ID -u CMUX_PANE_ID \
        TERM=xterm-256color .venv/bin/python /tmp/d993r3_paste_shot.py pasted out.svg
"""

import asyncio
import sys

sys.path.insert(0, "/Users/damian/workspace/repos/lo-switch-leak")

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.harness.types import Message, TextContent  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text  # noqa: E402

NOTICE = (
    "[model switch] You are now running as zai/glm-5.3 (was anthropic/claude-opus-5).\n"
    "Reason: provider failure"
)

ASK = "why did the model just change?"
ANSWER = (
    "The primary model's provider returned an error on that request, so the session "
    "fell back to the next route for it and put the notice in the transcript."
)
THANKS = "thanks — is that expected to keep happening?"


def rows_for(case: str) -> list[Message]:
    head = [Message.user(ASK), Message(role="assistant", content=[TextContent(text=ANSWER)])]
    if case == "no_paste":
        return [*head, Message.user(THANKS)]
    if case == "pasted":
        return [*head, Message(role="user", content=[TextContent(text=NOTICE)]), Message.user(THANKS)]
    if case == "quoted":
        return [*head, Message.user(f"I saw this line — {NOTICE.replace(chr(10), ' ')} — is that expected?")]
    raise SystemExit(f"unknown case {case!r}")


async def main() -> None:
    case, out = sys.argv[1], sys.argv[2]
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(118, 42)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(rows_for(case)))
        for _ in range(40):
            await pilot.pause()
        print(f"--- {case}: painted transcript ---")
        print(_transcript_text(app))
        save_capture(app, out)


asyncio.run(main())
