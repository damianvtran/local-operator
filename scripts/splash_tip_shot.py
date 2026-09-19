"""Capture the splash's ROTATING TIP ROW, on the real app, at a chosen ring position.

The tip row is the one part of the welcome view that changes while nothing else
does: it is painted once per ``TIP_ROTATE_INTERVAL_S`` and must occupy exactly one
row at every width that draws it at all. A frame of the default ring position
therefore says nothing about the entries a change adds, which is why this script
takes the position explicitly.

Usage::

    # the frame at ring position 12, on a 100x30 terminal
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
      scripts/splash_tip_shot.py out.svg --tip 12
    # the same tip at the NARROWEST terminal that draws it whole: the widget
    # is 4 columns narrower than the terminal, so 63x30 is the threshold today
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
      scripts/splash_tip_shot.py narrow.svg --tip 12 --size 63x30
    # no capture: print every ring position's rendered row and its cell width
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
      scripts/splash_tip_shot.py --census

The ring is advanced with the view's OWN ``_tip_tick`` — the method the rotation
timer calls — rather than by waiting out ``TIP_ROTATE_INTERVAL_S`` per position
(thirteen positions would be two and a half minutes of nothing per frame).
``run_test`` drives the real ``OperatorApp``, so ``local_operator.tcss`` is
applied; the lightweight test hosts declare no ``CSS_PATH`` and are useless for
judging this row.

Every run prints the numbers the frame is judged by: the ring position and its
resolved sentence, that row's rendered width in cells, ``TIP_MIN_WIDTH`` (the
threshold that decides whether the row is drawn at all), the frame's row count
and whether anything wrapped onto a second row.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE app imports: isolate HOME, config and caches

from rich.cells import cell_len  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import welcome  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def _rows(app: OperatorApp) -> list[str]:
    """The splash's rows as plain text, in paint order.

    ``WelcomeView.render()`` returns a ``Group`` of one ``Text`` per row, each
    already padded and truncated to the widget, so this is the painted block
    rather than a re-render at some other size.
    """
    view = app.query_one(welcome.WelcomeView)
    renderable = view.render()
    lines = getattr(renderable, "renderables", None)
    if lines is None:
        plain = getattr(renderable, "plain", None)
        return [plain] if plain is not None else []
    return [line.plain for line in lines]


def _tip_rows(app: OperatorApp) -> list[str]:
    """Every rendered row whose CONTENT opens with the tip glyph.

    Stripped first: the row is centred over the widget's width, so an unstripped
    match finds nothing on any terminal wider than the tip itself.
    """
    return [row.strip() for row in _rows(app) if row.strip().startswith(welcome.TIP_GLYPH)]


async def _census(size: tuple[int, int]) -> int:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = app.query_one(welcome.WelcomeView)
        for index in range(len(welcome.TIPS)):
            while view._tip_index != index:
                view._tip_tick()
            await pilot.pause()
            rows = _tip_rows(app)
            sentence = welcome._resolve_tip(welcome.TIPS[index])
            # The row is PADDED and CENTRED over the widget, so the figure that
            # matters is the stripped one: glyph + sentence, which is what
            # `TIP_MIN_WIDTH` is measured against.
            width = cell_len(rows[0]) if rows else 0
            print(
                f"tip {index:2d}  rows={len(rows)}  width={width:3d}  "
                f"frame_rows={len(_rows(app))}  {sentence!r}"
            )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--tip",
        type=int,
        default=0,
        help="Ring position to capture (0 is the pinned opening tip).",
    )
    parser.add_argument(
        "--size",
        default="100x30",
        help="Terminal size as WxH (63x30 is the narrowest terminal that draws the row whole).",
    )
    parser.add_argument(
        "--census",
        action="store_true",
        help="Print every ring position's rendered row instead of capturing one.",
    )
    args = parser.parse_args()

    width, height = (int(part) for part in args.size.split("x", 1))
    size = (width, height)
    if args.census:
        return await _census(size)
    if args.output is None:
        parser.error("an output path is required unless --census is given")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = app.query_one(welcome.WelcomeView)
        while view._tip_index != args.tip:
            view._tip_tick()
        await pilot.pause()
        rows = _rows(app)
        tip_rows = _tip_rows(app)
        sentence = welcome._resolve_tip(welcome.TIPS[args.tip])
        # Stripped already: the painted row is centred and padded to the widget,
        # and the number this change is judged by is the CONTENT width against
        # `TIP_MIN_WIDTH`.
        content_width = cell_len(tip_rows[0]) if tip_rows else 0
        print(f"ring position:    {args.tip} of {len(welcome.TIPS)}")
        print(f"tip sentence:     {sentence!r}")
        print(f"drawn tip rows:   {len(tip_rows)}")
        print(f"tip content wide: {content_width} cells")
        print(f"TIP_MIN_WIDTH:    {welcome.TIP_MIN_WIDTH} cells")
        print(f"frame rows:       {len(rows)}")
        print(f"terminal:         {width}x{height}")
        save_capture(app, args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
