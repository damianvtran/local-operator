"""Capture the `/session` Tool surface block across widths and ledger states.

Usage: python scripts/tool_surface_shot.py OUTDIR 66x38 [tools|denials|nodispatch|nested]

Sibling of ``session_report_shot.py``, which captures the WHOLE diagnostics
screen with one populated fixture. This one is about a single block of it, and
it exists because that block's defects live in two dimensions the whole-screen
capture cannot reach:

* **Width bands**, not widths. Design round 2 (D6) found the ``Tool-side
  errors`` note asserting the inverse of the truth (``7.4% dispatched`` — read
  as "7.4% got dispatched" where 27 of 31 did) across terminal columns 66-72
  only. Frames at 68 and 88 straddle a band like that without landing in it, so
  a note ladder has to be swept rather than sampled.
* **Ledger STATES.** The rows are drawn conditionally — a ` └ ` sub-row appears
  only when its count is nonzero — so the shape that carries a defect may not
  exist in the populated fixture at all. D7 (the headline counting the
  operator's own gate denials as ``failed``) is invisible in ``tools`` and is
  the entire headline in ``nodispatch``.

Every state seeds REAL ``tool_calls`` rows through ``AnalyticsStore`` and drives
the real ``OperatorApp`` through the real ``/session`` command, so what is
captured is the stylesheet-loaded screen rather than a bare test host — the
lightweight hosts in the test files declare no ``CSS_PATH`` and would not show a
spacing or colour change at all (``AGENTS.md``, "Visual validation").

No provider request, live session or operator config is used: ``probe_isolation``
re-homes ``HOME`` and the config dir before any application import, and each
capture asserts the operator's own ledger was untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from typing import cast  # noqa: E402

from rich.text import Text  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.session_report_shot import DiagnosticSession, seed_populated  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: ``state -> [(tool_name, origin, fault, count)]``, written verbatim into
#: ``tool_calls``. Chosen to be the states the two round-2 MAJORs live in, plus
#: the two neighbours that must not regress:
#:
#: ``tools``      the review's own fixture — 35 recorded, both origins, one
#:                denial and every fault class, so the subtraction chain is at
#:                full depth. This is the width-band state.
#: ``denials``    the DEFAULT posture with the approval gate on: denials
#:                dominate but real faults exist too, so a headline that counts
#:                denials as failures is wrong by a number rather than wholly.
#: ``nodispatch`` every recorded call excluded. D7's sharpest form: the old
#:                headline read `0 ok · 4 failed` on a session where nothing
#:                failed, which is the operator's founding report verbatim.
#: ``nested``     a FAILED eval-bridge call, which is why `failed` subtracts the
#:                excluded rather than summing the two model-origin fault
#:                classes — that sum leaves this call in neither headline term.
STATES: dict[str, list[tuple[str, str, str, int]]] = {
    "tools": [
        ("read", "model", "", 25),
        ("edit", "model", "invalid_arguments", 2),
        ("read", "model", "duplicate_id", 1),
        ("reed_file", "model", "unknown_tool", 1),
        ("web_fetch", "model", "execution", 2),
        ("bash", "model", "denied", 1),
        ("read", "nested", "", 3),
    ],
    "denials": [
        ("read", "model", "", 4),
        ("bash", "model", "denied", 7),
        ("edit", "model", "invalid_arguments", 1),
        ("web_fetch", "model", "execution", 1),
    ],
    "nodispatch": [("bash", "model", "denied", 4)],
    "nested": [
        ("read", "model", "", 8),
        ("web_fetch", "model", "execution", 2),
        ("read", "nested", "", 4),
        ("web_fetch", "nested", "execution", 2),
    ],
}


def seed_tool_calls(session_id: str, state: str) -> None:
    """Write the state's tool-call rows into the ambient (isolated) ledger.

    Rows are inserted through the store's own writer rather than by hand-rolled
    SQL so the capture exercises the same path the harness uses; a schema drift
    that broke recording would break this script rather than being papered over.
    """
    store = AnalyticsStore()
    rows: list[tuple[int, str, str, str, str, float]] = []
    ts = 1788602400000
    for tool_name, origin, fault, count in STATES[state]:
        for i in range(count):
            rows.append((ts + len(rows) * 1000, session_id, tool_name, origin, fault, 120.0 + i))
    store.record_tool_calls(rows)
    store.close()


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows_ = sys.argv[2].split("x")
    size = (int(cols), int(rows_))
    state = sys.argv[3] if len(sys.argv) > 3 else "tools"

    session = DiagnosticSession()
    session.set_conversation_name("Investigate request latency")
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="capture",
        generation=6,
        context_tokens=28400,
        context_is_estimate=False,
        context_window=200000,
    )
    # The request ledger too: without it the screen has no session to report on
    # and the tool block never renders.
    seed_populated(session.session_id)
    seed_tool_calls(session.session_id, state)

    ledger = default_db_path()
    before_digest = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        # ``_report_text`` is reached through ``getattr`` because ``app.screen``
        # is typed as the base ``Screen``; the cast names the renderable the
        # session screen actually returns rather than silencing the check.
        text = getattr(screen, "_report_text", None)
        painted = cast("Text", text()).plain if callable(text) else ""

        # SCROLL TO THE BLOCK, never a fixed number of pagedowns. The block's
        # line number moves with the width (rows above it rewrap) and with the
        # state (a ` └ ` row is only drawn when nonzero), so a constant page
        # count silently overshoots: at 88 columns three pagedowns landed past
        # it, and two states then exported byte-identical frames of the section
        # BELOW it while their `block` text — read from `_report_text()`, not
        # from the frame — was correctly different. A capture that does not
        # contain the thing it is capturing is worse than no capture, because it
        # looks like evidence.
        lines = painted.splitlines()
        target = next((i for i, ln in enumerate(lines) if "Tool surface" in ln), 0)
        scroll = getattr(screen, "_scroll", None)
        if scroll is not None:
            # One line of headroom above the header so the block is not flush
            # against the top edge, which reads as a cropped section.
            scroll.scroll_to(y=max(0, target - 1), animate=False)
            await pilot.pause()
        await pilot.pause()
        save_capture(app, str(out / f"{state}-{size[0]}.svg"))
        block = []
        for line in painted.splitlines():
            if "Tool surface" in line:
                block = [line]
            elif block:
                if line.strip() and not line.startswith("  ") and "└" not in line:
                    break
                block.append(line)
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "state": state,
            "size": size,
            "screen": type(screen).__name__,
            "screen_size": list(screen.size),
            "virtual_size": list(screen.virtual_size),
            # A scrollbar is a silent two-cell width loss and, on this screen, a
            # bug: the transcript scrolls, the block does not.
            "scrollbar": bool(getattr(screen, "show_vertical_scrollbar", False)),
            "geometry_clean": list(screen.size) == list(screen.virtual_size)
            and not getattr(screen, "show_vertical_scrollbar", False),
            "block": [ln.rstrip() for ln in block if ln.strip()],
            "ledger_unchanged": before_digest
            == (hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None),
        }
        (out / f"{state}-{size[0]}.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
