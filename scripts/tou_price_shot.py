"""Capture the `/model` picker's DeepSeek rows, for the time-of-use tariff round.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/tou_price_shot.py OUT.svg [COLSxROWS] [live|peak|off-peak]

WHY: `deepseek-flash` stores DeepSeek's PEAK list rates, and the picker's price
column is the one user-visible surface on which the tariff has to be legible. A
green test pins the string; only a rendered frame says the window tag fits the
row at a real width, and that the flat `deepseek-chat` row sitting beside it
still reads as deliberate rather than as a missing tag.

The rows come from the REAL pipeline — `ProviderController.initial_catalogue`
-> `picker_rows` — rather than from hand-typed numbers, so a change that only
works in a unit test cannot produce a correct frame here. `initial_catalogue` is
what the app paints its first `/model` frame from, and for DeepSeek it is the
DISCOVERY path (`merge_models` over the bundled registry rows), which is exactly
the hop a `CatalogueEntry`-only change would miss.

MOMENT: `live` (the default) lets the schedule read the machine's own clock,
which is the honest capture. `peak`/`off-peak` freeze `tariff.now_utc` to a
known Monday instant so the other window can be captured without waiting for the
clock to reach it; the window actually used is printed either way.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE app imports: isolate HOME, config and caches

from local_operator.providers.catalogue import picker_rows  # noqa: E402
from local_operator.providers.controller import ProviderController  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The instants the two non-live captures freeze to. Both are Mondays in the two
#: windows: 07:00 UTC is inside 06:00-10:00 (peak), 12:00 UTC is outside every
#: window (off-peak).
FROZEN = {
    "peak": datetime(2026, 9, 14, 7, 0, tzinfo=timezone.utc),
    "off-peak": datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc),
}


class _NoCredentials:
    """The credential-store slice `initial_catalogue` asks for.

    An empty one, because no credential affects a PRICE: `usable_providers()`
    only decides the `connected` column. The rows are then flipped connected
    below, because `_numbers` renders the literal ``login required`` in place of
    the whole numbers run — and the numbers run is the only thing this frame is
    evidence about.
    """

    def list_credentials(self, provider: Any = None, include_disabled: bool = False) -> list[Any]:
        return []


def _deepseek_rows() -> list[Any]:
    """The picker's rows for the `deepseek` filter, through the real pipeline."""
    entries = ProviderController(auth_store=cast(Any, _NoCredentials())).initial_catalogue()
    deepseek = [
        dataclasses.replace(entry, connected=True)
        for entry in entries
        if entry.provider == "deepseek"
    ]
    assert deepseek, "the DeepSeek catalogue came back empty"
    rows, _hidden = picker_rows(deepseek, usable=None)
    return rows


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows_dim = sys.argv[2].split("x")
        size = (int(cols), int(rows_dim))
    window = sys.argv[3] if len(sys.argv) > 3 else "live"

    when = FROZEN.get(window)
    if when is not None:
        # Imported lazily so this script still runs against the tree BEFORE the
        # tariff module existed: that is how the before-frame is captured.
        from local_operator.model import tariff

        tariff.now_utc = lambda: when
    effective = when or datetime.now(timezone.utc)

    rows = _deepseek_rows()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        picker = app._editor().model_picker
        picker.set_rows(rows, current="deepseek/deepseek-flash", status="")
        picker.open()
        # Let the overlay's layout SETTLE before the capture: one pause paints a
        # frame whose rows are still sized to the pre-open composer width.
        for _ in range(4):
            await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        picker._repaint()
        await pilot.pause()

        # The numbers behind the frame: a tag that fits in isolation can still be
        # the thing that truncates the id beside it, and only the assembled row
        # shows that.
        print(
            f"window={window} moment={effective:%Y-%m-%d %H:%M}Z size={picker.size}",
            file=sys.stderr,
        )
        for row in rows:
            print(
                f"  {row.provider}/{row.model_id}: "
                f"numbers={picker._numbers(row)!r} price={picker._price(row)!r}",
                file=sys.stderr,
            )
        for line in picker.render_rows(picker.size.width):
            print(f"  |{line.plain}|", file=sys.stderr)
        save_capture(app, out)


if __name__ == "__main__":
    asyncio.run(main())
