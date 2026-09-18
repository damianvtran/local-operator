"""Capture a NESTED roster: the dock scoped to an open child that has children.

Evidence for the nesting change. Two facts this frame exists to show, both in one
picture of the real app with its production CSS:

1. the header names WHOSE children the rows are — `Subagents of Leaf work` —
   because the roster lists one level at a time (`app._subagent_roster`), so the
   bare word names a different list on every level; and
2. a row whose child has children of its own carries `⊞N`, which is the only
   thing distinguishing it from a leaf before the reader drills in.

Run before and after the change to get the pair: the BEFORE frame (no scope, no
mark) is what a reader saw when a mid-level page and the root looked identical.

    .venv/bin/python scripts/nested_roster_shot.py /tmp/nested 120x40
    # the narrow dock: the mark's floor, and the scope's live-width bound
    .venv/bin/python scripts/nested_roster_shot.py /tmp/nested 60x30
    rsvg-convert /tmp/nested/nested-root-120x40.svg -o /tmp/nested-root.png
    rsvg-convert /tmp/nested/nested-scoped-120x40.svg -o /tmp/nested-scoped.png
    rsvg-convert /tmp/nested/nested-root-60x30.svg -o /tmp/nested-narrow.png
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.subagent_panel import SubagentPanel  # noqa: E402
from tests.unit.tui.test_subagent_scope import (  # noqa: E402
    FakeSession,
    _async_factory,
    install,
    nested_state,
    row_text,
)


async def capture(output: Path, size: tuple[int, int]) -> None:
    state = nested_state()
    session: object = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        # Frame 1 — the ROOT roster: the mid-level child is a row here, and its
        # own children are what `⊞N` marks. Before the change this row and a
        # leaf are byte-identical.
        save_capture(app, str(output / f"nested-root-{size[0]}x{size[1]}.svg"))
        print(f"root header: {panel._header.content!r}")
        for job_id in ("manager", "leaf"):
            try:
                print(f"  row {job_id}: {row_text(panel, job_id)!r}")
            except KeyError:
                print(f"  row {job_id}: not on this roster")
        # Frame 2 — the MID-LEVEL page open: the roster re-scopes to that
        # child's own children and the header names whose they are.
        app._open_subagent_view("leaf")
        for _ in range(6):
            await pilot.pause()
        save_capture(app, str(output / f"nested-scoped-{size[0]}x{size[1]}.svg"))
        print(f"scoped header: {panel._header.content!r}")
        for job_id in ("leaf", "grandchild"):
            try:
                print(f"  row {job_id}: {row_text(panel, job_id)!r}")
            except KeyError:
                print(f"  row {job_id}: not on this roster")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path, help="directory for the two frames")
    parser.add_argument("size", nargs="?", default="120x40")
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.split("x"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    asyncio.run(capture(args.outdir, (width, height)))


if __name__ == "__main__":
    main()
