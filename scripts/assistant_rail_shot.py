"""Capture the assistant rail against everything it has to be told apart from.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/assistant_rail_shot.py OUT.svg [COLSxROWS] [THEME]

The seeded tree is chosen so one frame answers every question the treatment can
be wrong about, because a frame of an assistant message ALONE cannot show any of
them:

* a ``UserBlock`` prompt directly above — the rail borrows that rule's GEOMETRY
  and deliberately not its colour, so the two bars have to be visible in one
  frame or "they are still told apart" is an assertion rather than an
  observation;
* a MID-TURN PROGRESS sentence between that prompt and the answer, with its own
  settled tool card — the rail marks the ANSWER, so the one frame this capture
  must contain is a progress sentence and an answer on screen together, or "the
  rail is on the answer and not on the narration" is again an assertion. It is
  painted through the REAL event path (``AssistantMessageEnd`` finalized into
  tool calls) rather than mounted as a settled block: the classification is made
  from exactly those two fields on arrival, so a hand-built block would capture a
  state the app never produces;
* a settled tool card — the ledger spine is the other vertical ink on the
  screen, and a rail that reads as a third spine is a regression in the
  transcript's structure even when it is correct per-block;
* prose containing a BLOCKQUOTE, a bullet list and a fenced code block — the
  blockquote is the load-bearing one, because Rich paints its bar with the very
  glyph the rail uses, and the frame is where "two bars, and you can tell which
  is which" is checked;
* a multi-paragraph answer, so the blank separator rows show the rail running
  CONTINUOUSLY rather than breaking into one segment per paragraph.

The tree is ONE turn, and there is deliberately no second prompt/answer pair
after it: the transcript follows the tail, so a second turn pushed the progress
sentence off the bottom of a 30-row frame — the two blocks this capture exists to
compare cannot both be in the frame that way.

``THEME`` (default ``dark``) selects the palette, because the rail's ``label``
ink moves per theme and the decision that it stays legible and stays distinct
from the prompt's ``signal`` is a claim about every palette, not about one.

Set ``RAIL_SHOT_OFF=1`` to capture with ``display.rail`` OFF. The flag is
forced on the CONSUMING module rather than by writing a config file, the same
seam the unit tests use, so the frame shows what a user who turned the setting
off would see without this capture leaving state behind in their config. The
rail-OFF frame is not decoration: "off restores the pre-rail build" is a claim
about a rendered frame, and the pair is what lets a reviewer check it.

``SURFACE`` (default ``transcript``) selects WHICH surface is captured.
``subagent`` renders the delegated-job page instead, and it is not optional
coverage: the rail appears there, and every prose block on that page is a model
response — so it is the page where a mark that distinguishes progress from the
answer is either doing work or just spending the inset. The fixture it folds has
both kinds (a sentence before a tool batch, then the closing sentence), which is
what makes the pair checkable from one frame rather than argued from the source.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()

import os  # noqa: E402

import local_operator.tui.widgets.assistant as _assistant_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import (  # noqa: E402
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_band_panels import FakeSession as BandSession  # noqa: E402
from tests.unit.tui.test_band_panels import _async_factory, _fake_jobs  # noqa: E402
from tests.unit.tui.test_subagent_view import TRAJECTORY, _job_with  # noqa: E402

#: Multi-paragraph, and every construct that paints its own furniture. The
#: blockquote is here for the glyph collision; the fence is here because a code
#: line beginning with a digit is the case the copy path refuses to strip.
ANSWER = (
    "The ingest path reads each source once and writes a manifest, so a "
    "re-run is cheap.\n"
    "\n"
    "Three things are worth knowing before you change it:\n"
    "\n"
    "- a source that fails three times is quarantined, never dropped\n"
    "- the manifest is what the reconciler reads on the next pass\n"
    "- retries are backed off, so a flapping source cannot spin the loop\n"
    "\n"
    "> The quarantine is deliberate: losing a source silently is worse than\n"
    "> stopping loudly.\n"
    "\n"
    "Re-run a single source with:\n"
    "\n"
    "```sh\n"
    "1 ingest --source billing --force\n"
    "```\n"
    "\n"
    "That is the whole loop."
)

#: The mid-turn progress sentence: prose a model call streamed before it
#: finalized into a tool call. Short enough to sit on one row at 100 columns,
#: which is what puts it in the frame beside the answer instead of above it.
PROGRESS = "The app quit during that window. Let me check its state and bring it back."


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer — ``finalize_text`` is what the stream does at message
    end, and an unsettled block keeps its live-turn ink, which would make this
    a capture of the streaming treatment rather than of the rail."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _tool(app: OperatorApp, call_id: str, name: str, args: dict[str, object], result: str) -> None:
    """A FINISHED tool row, so the ledger spine beside the rail is settled ink
    rather than the live-turn accent."""
    card = ToolCard(call_id, name, args)
    app._append_block(card)
    card.mark_done(result)


async def _stream(
    app: OperatorApp,
    pilot: Any,
    text: str,
    *,
    stop_reason: str | None,
    has_tool_calls: bool,
) -> None:
    """Paint one model call through the app's own handlers.

    The mount, the finalize and the classification are all owned by these
    handlers (``on_assistant_delta``/``on_assistant_message_end``), so posting
    the events is what makes the frame the one the product produces for a call
    that ends in tool calls — the same sequence ``test_narration_toggle`` drives.
    """
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    app.post_message(AssistantDelta(text))
    await pilot.pause()
    app.post_message(
        AssistantMessageEnd(text, stop_reason=stop_reason, has_tool_calls=has_tool_calls)
    )
    await pilot.pause()


async def _seed(app: OperatorApp, pilot: Any) -> None:
    """The reported turn: prompt, progress sentence, its tool card, the answer."""
    app._append_block(UserBlock("how does the ingest path handle a failing source?"))
    await pilot.pause()
    await _stream(app, pilot, PROGRESS, stop_reason="toolUse", has_tool_calls=True)
    _tool(app, "t1", "read", {"path": "src/ingest/manifest.py"}, "412 lines")
    await pilot.pause()
    app._append_block(_answer(ANSWER))


async def _open_subagent(app: OperatorApp, pilot: Any, job_id: str) -> None:
    """Drive the real page open, the way ``test_subagent_view`` drives it.

    The session has to arrive before the page can be asked for; the poll is the
    same bounded one the tests use rather than a fixed sleep, so a slow import
    lengthens the wait instead of producing a frame of an empty page.
    """
    for _ in range(80):
        await pilot.pause()
        if app._session is not None:
            break
    app._open_subagent_view(job_id)
    for _ in range(8):
        await pilot.pause()


def _force_rail_off() -> None:
    """Pin ``display.rail`` OFF for this capture only.

    Patched on the module that READS it, delegating every other key to the real
    reader, so the frame differs from its ON counterpart in the rail and in
    nothing else.
    """
    real = _assistant_mod.settings_get
    _assistant_mod.settings_get = lambda key, default=None: (  # type: ignore[assignment]
        False if key == "display.rail" else real(key, default)
    )


async def main() -> None:
    if os.environ.get("RAIL_SHOT_OFF") == "1":
        _force_rail_off()
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    theme = sys.argv[3] if len(sys.argv) > 3 else None
    surface = sys.argv[4] if len(sys.argv) > 4 else "transcript"

    if surface == "subagent":
        # The delegated-job page, built from the same trajectory fixture the
        # subagent tests fold, so the frame shows the shape that ships rather
        # than one composed for the capture.
        job = _job_with(TRAJECTORY)
        job.prompt = "audit the ingest path"
        session = BandSession()
        session.jobs = _fake_jobs(job)
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            if theme is not None:
                app._apply_theme(theme)
                await pilot.pause()
            await _open_subagent(app, pilot, str(job.id))
            await pilot.pause()
            await settle_status_line(pilot, app)
            screen = app.screen
            print(
                f"size={screen.size} virtual={screen.virtual_size} "
                f"vscroll={screen.show_vertical_scrollbar}",
                file=sys.stderr,
            )
            save_capture(app, out)
        return

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if theme is not None:
            # Through the APP's own path, and only once it is running.
            # ``OperatorApp.__init__`` sets the theme from config, so a
            # ``theme_mod.set_theme`` before construction is silently overridden
            # and every frame comes out in the default ramp — which looks like a
            # working capture and is not one. ``_apply_theme`` is what the theme
            # picker calls, so this frame is the one a user would see.
            app._apply_theme(theme)
            await pilot.pause()
        await _seed(app, pilot)
        await pilot.pause()
        await pilot.pause()

        # A second settled frame: a first paint that differs from this one is a
        # reflow the user sees as motion (AGENTS.md, "Animation and multi-frame
        # changes"). The status band is waited on too, so a capture of this tree
        # differs from another capture of the same tree in the ledger and
        # nothing else.
        await settle_status_line(pilot, app)
        screen = app.screen
        print(
            f"size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar}",
            file=sys.stderr,
        )
        save_capture(app, out)


asyncio.run(main())
