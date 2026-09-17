"""Capture a settled turn with and without its mid-turn narration.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/narration_toggle_shot.py OUT.svg [COLSxROWS] [MODE]

``MODE`` is ``show`` (default — ``display.narration`` ON, which is the shipped
behaviour and therefore the BEFORE frame) or ``hide`` (the flag OFF, the
AFTER). The pair is the whole point: the feature is judged on whether a reader
can find the answer in the settled transcript, and a frame of either mode alone
says nothing about that.

The seeded tree is deliberately a FULL turn — prompt, narration, two tool rows,
answer. Narration is only confusable with an answer when both are on screen in
the same ink, so a capture of the narration by itself would miss the defect
this setting exists to fix.

The transcript is driven through the app's own event handlers rather than by
mounting blocks, because the removal under test lives in
``on_assistant_message_end``: a hand-mounted tree would paint a frame the app
never produces.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()


def _seed_env() -> None:
    """Pin the icon gate to Nerd glyphs BEFORE the app is imported.

    Same reasoning as ``wake_shot.py``: the gate reads the environment at
    row-build time and this capture runs under an isolated HOME, so without a
    marker every tool row falls back to ASCII and the frame stops matching what
    the operator sees.
    """
    for var in (
        "GHOSTTY_RESOURCES_DIR",
        "GHOSTTY_BIN",
        "KITTY_WINDOW_ID",
        "WEZTERM_PANE",
        "WEZTERM_EXECUTABLE",
        "TERM_PROGRAM",
        "LOCAL_OPERATOR_NO_NERD_ICONS",
    ):
        os.environ.pop(var, None)
    os.environ["GHOSTTY_BIN"] = "/usr/local/bin/ghostty"


_seed_env()

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import (  # noqa: E402
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.settings import settings_reload  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

PROMPT = "how long before a gate times out?"
NARRATION = "Let me check the config first."
ANSWER = (
    "The approval gate times out after **30 seconds**, set by "
    "`approval.gate_timeout` in `config.yml`. After that the call is refused "
    "rather than left waiting."
)


def _write_flag(*, show: bool) -> None:
    """Set ``display.narration`` through the REAL registry and reader.

    Written to the isolated config ``isolate_capture()`` re-homed us into, so
    the capture exercises the same flat-dotted key a user's toggle writes —
    patching the reader would capture a frame no configuration produces.
    """
    settings_io.write_setting(
        ConfigManager(config_dir()), settings_io.BY_KEY["display.narration"], show
    )
    settings_reload()


def _tool(app: OperatorApp, call_id: str, name: str, args: dict[str, object], result: str) -> None:
    """A FINISHED tool row — settled ink, not the live-turn accent."""
    card = ToolCard(call_id, name, args)
    app._append_block(card)
    card.mark_done(result)


async def _model_call(pilot, app: OperatorApp, text: str, *, narration: bool) -> None:
    """One model call through the app's own handlers, start to finalize."""
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    app.post_message(AssistantDelta(text))
    await pilot.pause()
    app.post_message(
        AssistantMessageEnd(
            text,
            stop_reason="toolUse" if narration else "stop",
            has_tool_calls=narration,
        )
    )
    await pilot.pause()
    await pilot.pause()


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    mode = sys.argv[3] if len(sys.argv) > 3 else "show"

    _write_flag(show=(mode == "show"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._append_block(UserBlock(PROMPT))
        await pilot.pause()

        # The mid-turn call: prose, then the tools it announced.
        await _model_call(pilot, app, NARRATION, narration=True)
        _tool(app, "t1", "read", {"path": "config.yml"}, "84 lines")
        _tool(app, "t2", "grep", {"pattern": "gate_timeout"}, "1 match")
        await pilot.pause()

        # The final call: the answer, which survives in both modes.
        await _model_call(pilot, app, ANSWER, narration=False)

        # A second settled frame: a first paint that differs from this one is a
        # reflow the user sees as motion (AGENTS.md, "Animation and multi-frame
        # changes"), and the status band is waited on so two captures of the
        # same tree differ in the ledger and nothing else.
        await pilot.pause()
        await settle_status_line(pilot, app)
        screen = app.screen
        print(
            f"mode={mode} size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar}",
            file=sys.stderr,
        )
        save_capture(app, out)


asyncio.run(main())
