"""Capture the status band of a freshly booted viewer, with NOTHING typed.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/eager_boot_shot.py out.svg --live
    .venv/bin/python scripts/eager_boot_shot.py out.svg --isolated

With explicit ``--live``, boots the real ``OperatorApp`` against ``~/.local-operator``
(so a real provider and real MCP servers are configured) on a fresh session
id, presses no keys, waits for the mount engage to bind, and saves the frame.
Prints the fields the band reads so the picture can be backed by numbers.
``--isolated`` instead captures real unconfigured boot with a temporary HOME and
config, no provider credentials or integrations. It validates capture transport,
not a successful live runtime bind; the finite gallery runs only this safe mode.

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

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

# This probe intentionally differs from the offline fixture census: live mode
# exercises configured providers/MCP. Requiring an explicit choice prevents a
# gallery or an unsuspecting operator from starting their real integrations.
CAPTURE_REQUIRES_LIVE_OPT_IN = True


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Use real configured providers/MCP")
    mode.add_argument(
        "--isolated", action="store_true", help="Capture real unconfigured boot safely"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if args.isolated:
        isolate_capture()
        config = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    else:
        config = Path.home() / ".local-operator"
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config)

    from local_operator.session.remote import RemoteSession
    from local_operator.tui.app import OperatorApp

    session_id = uuid.uuid4().hex[:12]

    async def _never() -> None:
        raise AssertionError("takeover was not expected")

    viewer = await RemoteSession.cold(
        session_id, config_dir=config, cwd=os.getcwd(), takeover_factory=_never
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
        for _ in range(0 if args.isolated else 120):
            await asyncio.sleep(0.25)
            await pilot.pause()
            if not getattr(app._session, "is_cold", True):
                break
        # Settle budget for the MCP roster push: the servers report in one by
        # one over the first couple of seconds after the bind (the count ticks
        # up on the band), and the frame should show the finished roster.
        for _ in range(0 if args.isolated else 40):
            await asyncio.sleep(0.1)
            await pilot.pause()
        state = viewer.frontend_state
        model = state.effective_model
        if args.live:
            assert model is not None, "configured live runtime did not bind"
        print(f"session:          {session_id}")
        print(f"is_cold:          {viewer.is_cold}")
        label = (
            f"{model.provider}/{model.model_id}"
            if model is not None and model.provider and model.model_id
            else "unconfigured"
        )
        print(f"effective model:  {label}")
        print(f"mcp servers:      {len(state.mcp_servers or [])}")
        print(f"context window:   {state.context_window}")
        save_capture(app, out)
    await viewer.dispose()


if __name__ == "__main__":
    asyncio.run(main())
