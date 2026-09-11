"""PROTOTYPE — THROWAWAY. Still frames of the /resume picker variants.

Usage:
    python scripts/prototype_resume_shot.py OUT.svg VARIANT [COLSxROWS] [query] [downs]

`downs` steers the cursor after the query settles. Round 2's only stacked query
frame landed on a fuzzy hit, so the stacked context line was never demonstrated;
the cursor has to be steered onto a body match to capture it (D17, and the
designer's "what I did not verify").

Give it a NUMBER to press `down` that many times, or a session-name SUBSTRING to
land on that session wherever it currently sits. Prefer the substring: the store
is live, rows are ordered newest-first, and an active session bubbles to the top
between two captures minutes apart — a fixed count silently lands on a different
session and the frame stops showing what it was captured to show.

Frames go to /tmp, never into the repo tree. See
``~/workspace/PROPOSAL-resume-picker.md``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 1. Capture the REAL store BEFORE isolation re-homes HOME.
REAL_STORE = Path(
    os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or (Path.home() / ".local-operator")
)

# 2. Isolation MUST precede every local_operator import.
import scripts.probe_isolation  # noqa: F401,E402

# 3. Only now may local_operator be imported.
from scripts.visual_capture import save_capture  # noqa: E402

from local_operator.tui.widgets.prototype_resume_data import PreviewData  # noqa: E402
from local_operator.tui.widgets.prototype_resume_picker import (  # noqa: E402
    PrototypeResumeApp,
)


async def shoot(
    out: Path, variant: str, size: tuple[int, int], query: str, target: str = ""
) -> None:
    app = PrototypeResumeApp(PreviewData(REAL_STORE), variant, query)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if query:
            for char in query:
                await pilot.press(char)
            await pilot.pause()
        steps = 0
        if target.isdigit():
            steps = int(target)
        elif target:
            rows = app.screen._rows
            steps = next(
                (i for i, r in enumerate(rows) if target.lower() in (r.name or r.id).lower()),
                -1,
            )
            if steps < 0:
                raise SystemExit(f"no visible row matches {target!r} (query={query!r})")
        for _ in range(steps):
            await pilot.press("down")
        await pilot.pause()
        # Report what the cursor ACTUALLY landed on, so a frame can never be
        # described by the row it was aimed at rather than the row it captured.
        screen = app.screen
        row = screen._rows[screen._cursor]
        save_capture(app, out)
        return row.name or row.id


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    variant = argv[1] if len(argv) > 1 else "A"
    cols, _, rows = (argv[2] if len(argv) > 2 else "100x30").partition("x")
    query = argv[3] if len(argv) > 3 else ""
    target = argv[4] if len(argv) > 4 else ""
    landed = asyncio.run(shoot(out, variant, (int(cols), int(rows)), query, target))
    print(f"{out}  {cols}x{rows}  variant {variant}  cursor={landed!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
