"""Capture the status band of a freshly booted viewer, with NOTHING typed.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/eager_boot_shot.py out.svg

Boots the real ``OperatorApp`` against the machine's real ``~/.local-operator``
(so a real provider and real MCP servers are configured) on a fresh session
id, presses no keys, waits for the mount engage to bind, and saves the frame.
Prints the fields the band reads so the picture can be backed by numbers.

This is the frame the eager-runtime change is judged by: before it, the band
showed cwd and the configured model NAME only until the first keystroke; after
it, the effective model, effort, MCP roster and context reading are on the
first settled frame. Run it from a checkout of ``main`` for the before-frame
(see AGENTS.md, "Visual validation" — never stash to get one).

Boots on the CURRENT directory (`os.getcwd()`), so the splash and the band
name the same cwd. Uses a fresh session id every run and leaves nothing behind: an unused runtime
is retired on quit, and the deferred session directory is never materialised.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG = Path.home() / ".local-operator"


async def main() -> None:
    out = sys.argv[1]
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(CONFIG)

    from local_operator.session.remote import RemoteSession
    from local_operator.tui.app import OperatorApp

    session_id = uuid.uuid4().hex[:12]

    async def _never() -> None:
        raise AssertionError("takeover was not expected")

    viewer = await RemoteSession.cold(
        session_id, config_dir=CONFIG, cwd=os.getcwd(), takeover_factory=_never
    )

    async def factory() -> RemoteSession:
        return viewer

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 32)) as pilot:
        # NO key presses anywhere in this script: the whole point is what the
        # band shows before the user touches anything.
        for _ in range(40):
            await pilot.pause()
        # Break on the BINDING (the engage publishes a record, then dials,
        # and only the dial fills the band) — a capture script, so a bounded
        # poll rather than a subscription is acceptable here.
        for _ in range(120):
            await asyncio.sleep(0.25)
            await pilot.pause()
            if not getattr(app._session, "is_cold", True):
                break
        # Settle budget for the MCP roster push: the servers report in one by
        # one over the first couple of seconds after the bind (the count ticks
        # up on the band), and the frame should show the finished roster.
        for _ in range(40):
            await asyncio.sleep(0.1)
            await pilot.pause()
        state = viewer.frontend_state
        model = state.effective_model
        assert model is not None
        print(f"session:          {session_id}")
        print(f"is_cold:          {viewer.is_cold}")
        print(f"effective model:  {model.provider}/{model.model_id}")
        print(f"mcp servers:      {len(state.mcp_servers or [])}")
        print(f"context window:   {state.context_window}")
        app.save_screenshot(out)
    await viewer.dispose()


if __name__ == "__main__":
    asyncio.run(main())
