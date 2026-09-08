"""Capture /goal's real composer-to-turn behavior before and after the fix."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import scripts.probe_isolation  # noqa: F401 — isolate before application imports
from local_operator.tui.app import OperatorApp
from scripts.visual_capture import save_capture
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    drain,
    text_turn,
    transcript_text,
    wait_for_adoption,
)


async def main() -> None:
    # A headless app must not inherit identities that can rename a live workspace.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            os.environ.pop(key)
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    stream = ScriptedStream([text_turn("Working on the release checklist.")])
    session = build_session(root / "sessions" / "goal-probe", stream, cwd=root)
    session.set_conversation_name("Goal submission", user_set=True)

    async def factory():
        return session

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            app._editor().load_text("/goal Prepare the release checklist")
            await pilot.press("enter")
            await drain(pilot, cycles=10)
            save_capture(app, str(output / "first.svg"))
            await app.workers.wait_for_complete()
            await drain(pilot, cycles=20)
            save_capture(app, str(output / "settled.svg"))
            result = {
                "goal": session.goal,
                "transcript": transcript_text(app),
                "requests": [r.model_dump(mode="json") for r in stream.requests],
            }
            (output / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
    finally:
        await dispose_quietly(session)


if __name__ == "__main__":
    asyncio.run(main())
