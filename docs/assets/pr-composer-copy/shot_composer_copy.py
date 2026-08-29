"""Render the composer's click-selection states from the REAL OperatorApp.

Uses the real app (CSS_PATH applied) driven by `pilot.click(times=...)`, which
is the gesture the bug is about: a double-click to select a word, a
triple-click to select the line, and the Ctrl+C that should copy what they
selected.

Both THEMES are captured. The light ramp is not a nicety here: the selection
band's ink was unset, so selected text fell back to Textual's `#e0e0e0` and on
paper that was 1.003:1 against the band — the selection erased the text instead
of highlighting it (design round 1, D1). A dark-only capture cannot show that,
which is how the first round of frames missed it.

Every frame prints the state it was captured in, so a reader can check the
picture against the numbers rather than trusting the filename.

Usage: shot_composer_copy.py <outdir> [repo-root]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/tmp/lop-composer-copy")

from local_operator.tui import theme as theme_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.markdown_theme import install_markdown_theme  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

DRAFT = "summarise the ingest path please"

#: Paragraphs split by a blank line — the commonest shape of a real prompt, and
#: the one whose blank row used to answer the gesture with nothing and then eat
#: the draft on the following Ctrl+C (design round 1, D2).
PARAGRAPH_DRAFT = "first paragraph of my prompt\n\nsecond paragraph here"

#: A paragraph followed by two shift+enters — the user sitting on a blank LAST
#: row while they think of the next sentence. `_line_break_span` must stay
#: collapsed there (there is no following row to take, which is also what keeps
#: the empty composer correct, D7), so the gesture paints NOTHING and the frame
#: is byte-identical to the one before it. That identical frame is the point:
#: it is why the Ctrl+C which follows used to clear the draft, and why the
#: remedy had to be on the gesture rather than on the range (design round 2,
#: D2). The `07`/`08` pair is what shows the draft surviving that press.
TRAILING_BLANK_DRAFT = "first paragraph of my prompt\n\n"


async def _settle(pilot, editor: Editor, row: int) -> None:
    """Wait for the composer to stop moving before aiming a click at ``row``.

    The dock migrates for several frames after a multi-line draft loads, and
    `pilot.click` resolves its offset against the region at click time, so a
    click aimed early lands a row high and the frame shows the wrong gesture.
    """
    stable = 0
    previous = None
    for _ in range(40):
        await pilot.pause()
        current = editor.region
        stable = stable + 1 if current == previous and editor.size.height > row else 0
        previous = current
        if stable >= 4:
            return


async def shot(
    outdir: Path,
    name: str,
    times: int,
    press_copy: bool,
    *,
    theme: str = "dark",
    draft: str = DRAFT,
    column: int = 18,
    row: int = 0,
    arm_exit_hint: bool = False,
) -> None:
    # The theme has to be passed to the CONSTRUCTOR, not merely set beforehand:
    # `OperatorApp.__init__` re-applies its `theme_name` argument (which
    # defaults to dark) over whatever the module-level ramp was, so a
    # `set_theme` call before construction is silently overwritten and every
    # "light" frame comes out byte-identical to its dark twin.
    theme_mod.set_theme(theme)
    install_markdown_theme()
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name=theme)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        if arm_exit_hint:
            # The ladder has to be armed on an EMPTY composer, because that is
            # the press that paints `ctrl+c again to exit`. The draft is typed
            # in afterwards so the barren rung is reachable at all: it is now
            # gated on there being a draft to protect (agent review round 3,
            # R3-1), and the stale-hint defect only shows where both are true
            # (design review round 3, D3-2).
            await pilot.press("ctrl+c")
            await pilot.pause()
        editor.load_text(draft)
        await _settle(pilot, editor, row)
        app._clipboard = "PREEXISTING"

        await pilot.click(
            editor,
            offset=(editor.gutter.left + column, editor.gutter.top + row),
            times=times,
        )
        await pilot.pause()
        await pilot.pause()

        if press_copy:
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.pause()

        app.save_screenshot(str(outdir / f"{name}.svg"))
        toast = app.query_one(Toast)
        # `exit_hint` is printed for every frame so the D3-2 pair can be checked
        # against a number as well as against the picture.
        print(
            f"{name}: theme={theme} selection={editor.selection} "
            f"selected_text={editor.selected_text!r} "
            f"clipboard={app._clipboard!r} toast={toast.message!r} "
            f"exit_hint={app._exit_hint is not None} "
            f"draft={editor.text!r}"
        )


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    for theme in ("dark", "light"):
        prefix = "" if theme == "dark" else "light-"
        await shot(outdir, f"{prefix}01-double-click", 2, False, theme=theme)
        await shot(outdir, f"{prefix}02-double-click-ctrl-c", 2, True, theme=theme)
        await shot(outdir, f"{prefix}03-triple-click", 3, False, theme=theme)
        await shot(outdir, f"{prefix}04-triple-click-ctrl-c", 3, True, theme=theme)
        # The blank row between two paragraphs: the gesture must answer with a
        # live range, and the Ctrl+C after it must not clear the draft (D2).
        await shot(
            outdir,
            f"{prefix}05-blank-line-double-click",
            2,
            False,
            theme=theme,
            draft=PARAGRAPH_DRAFT,
            column=0,
            row=1,
        )
        await shot(
            outdir,
            f"{prefix}06-blank-line-ctrl-c",
            2,
            True,
            theme=theme,
            draft=PARAGRAPH_DRAFT,
            column=0,
            row=1,
        )
        # The blank LAST row (D2). The gesture paints nothing here by design,
        # so `07` is deliberately identical to the resting composer — the value
        # is in `08`, where the draft is STILL THERE after the press that used
        # to scrap it.
        await shot(
            outdir,
            f"{prefix}07-trailing-blank-double-click",
            2,
            False,
            theme=theme,
            draft=TRAILING_BLANK_DRAFT,
            column=0,
            row=2,
        )
        await shot(
            outdir,
            f"{prefix}08-trailing-blank-ctrl-c",
            2,
            True,
            theme=theme,
            draft=TRAILING_BLANK_DRAFT,
            column=0,
            row=2,
        )
        # D3-2: the absorbed press must WITHDRAW a live exit hint. The barren
        # rung returned early without resetting the ladder, so the screen kept
        # promising `ctrl+c again to exit` after a press that made no exit —
        # the exact stale promise the neighbouring draft rung was written to
        # prevent. `09` is the armed hint, `10` is the frame after the absorbed
        # press: on this branch the hint is GONE and the draft is still there.
        await shot(
            outdir,
            f"{prefix}09-exit-hint-armed",
            2,
            False,
            theme=theme,
            draft=TRAILING_BLANK_DRAFT,
            column=0,
            row=2,
            arm_exit_hint=True,
        )
        await shot(
            outdir,
            f"{prefix}10-exit-hint-absorbed-press",
            2,
            True,
            theme=theme,
            draft=TRAILING_BLANK_DRAFT,
            column=0,
            row=2,
            arm_exit_hint=True,
        )
    theme_mod.set_theme("dark")


asyncio.run(main())
