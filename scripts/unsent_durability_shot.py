"""Capture the frames a user gets when a cold viewer's bind fails.

Two modes, one per half of the ``dea45f5bdae2`` incident:

* ``prompt`` — a typed message whose bind fails against a busy runtime. The
  frame shows whether the composer still holds the text and what the failure
  notice says. Before the durability fix the composer is EMPTY and the notice
  reads as status ("the runtime is not responding"), which is the loss.
* ``splash`` — a foreground slash engage that fails on a still-cold viewer.
  The frame shows the splash's model-row word and notice row. Before the fix
  the splash carries the facade's PROVISIONAL label (`test/mock`) — the
  viewer's own copy from local config, which on a cold viewer reads as
  connected — and no failure row at all; the verdict lives only in a
  transcript notice under the splash.

Drives the REAL ``OperatorApp`` over a real ``RemoteSession.cold`` (so the
production stylesheet applies — the point of a capture), with the bind failed
exactly the way the incident failed it: ``_ensure_bound`` raising
``RuntimeUnresponsiveError`` inside ``RemoteSession.prompt`` / the
``_bind_then_dispatch`` seam.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/unsent_durability_shot.py prompt|splash out.svg [WIDTHxHEIGHT]
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.session.remote import (  # noqa: E402
    _SYNC_UNRESPONSIVE_REASON,
    RemoteSession,
    RuntimeUnresponsiveError,
)
from local_operator.tui.app import OperatorApp  # noqa: E402

MESSAGE = "I downloaded the store version of the extension, can you try it"


async def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__.rsplit("Usage:", 1)[1])
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode not in ("prompt", "splash"):
        sys.exit("mode must be 'prompt' or 'splash'")
    width, height = (int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "110x30").split("x"))

    config = Path.home() / "config"
    config.mkdir(parents=True, exist_ok=True)
    (Path.home() / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    session_id = uuid.uuid4().hex[:12]

    async def _never():
        raise AssertionError("a viewer never takes over")

    async def factory() -> RemoteSession:
        return await RemoteSession.cold(
            session_id, config_dir=Path.home(), cwd=str(Path.home()), takeover_factory=_never
        )

    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
    app = OperatorApp(factory)

    async with app.run_test(size=(width, height)) as pilot:
        for _ in range(200):
            await pilot.pause()
            if app._session is not None:
                break
        session = app._session
        assert session is not None

        async def unresponsive(*args, **kwargs) -> None:
            raise RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON)

        session._ensure_bound = unresponsive  # type: ignore[method-assign]

        if mode == "prompt":
            from textual import events

            from local_operator.tui.widgets.editor import Editor

            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            app.post_message(events.Paste(MESSAGE))
            await pilot.pause()
            await pilot.press("enter")
        else:
            app._bind_then_dispatch("/goal test the fix")
        for _ in range(400):
            await pilot.pause()
            await asyncio.sleep(0.01)

        save_capture(app, out)
        print(f"captured {mode} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
