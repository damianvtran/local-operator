"""PROTOTYPE — THROWAWAY. Still frames of the /resume picker variants.

Usage:
    python scripts/prototype_resume_shot.py OUT.svg VARIANT [COLSxROWS] [query]

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


async def shoot(out: Path, variant: str, size: tuple[int, int], query: str) -> None:
    app = PrototypeResumeApp(PreviewData(REAL_STORE), variant, query)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if query:
            for char in query:
                await pilot.press(char)
            await pilot.pause()
        await pilot.pause()
        save_capture(app, out)


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    variant = argv[1] if len(argv) > 1 else "A"
    cols, _, rows = (argv[2] if len(argv) > 2 else "100x30").partition("x")
    query = argv[3] if len(argv) > 3 else ""
    asyncio.run(shoot(out, variant, (int(cols), int(rows)), query))
    print(f"{out}  {cols}x{rows}  variant {variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
