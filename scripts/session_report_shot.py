"""Capture the real /session slash path with synthetic ledger data.

Usage: python scripts/session_report_shot.py OUTDIR 80x24 populated|empty|unavailable
Copy this script into a preserved base worktree to capture the pre-command path
with the same fixture. No provider request, live session or operator config is used.
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
from dataclasses import replace  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.harness.types import ModelSpec  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402


class DiagnosticSession(FakeSession):
    frontend_state: FrontendSessionState

    @property
    def session_id(self) -> str:
        # A REAL-SHAPED ID: the harness generates ``uuid4().hex[:12]``
        # (``harness/types.py``, ``fork.py``), and every session directory on a
        # live machine is exactly 12 characters. The previous 40-character
        # demo string was ~3.3x anything reachable, so it manufactured a header
        # crop at 50 columns that no operator can hit — a defect in the
        # evidence, not in the screen.
        return "a3f9c21b7e40"

    @property
    def model(self) -> ModelSpec:
        return ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6", context_window=200000)

    @property
    def effective_model(self) -> ModelSpec:
        return ModelSpec(provider="openai", model_id="gpt-5.2", context_window=200000)

    @property
    def model_label(self) -> str:
        return "anthropic/claude-sonnet-4-6"

    @property
    def effective_model_label(self) -> str:
        return "openai/gpt-5.2"


#: The populated demo session, as (context, output, reasoning, duration_ms,
#: purpose, ok) per request. Module scope so a sibling capture script can
#: seed the SAME session — one fixture, one place to change it.
POPULATED_SHAPE = [
    # (context, output, reasoning, duration_ms, purpose, ok)
    (8_400, 900, 120, 1_850, "turn", True),
    (14_200, 1_400, 260, 2_310, "turn", True),
    (21_800, 620, 0, 1_240, "aside", True),
    (33_500, 2_900, 800, 4_120, "turn", True),
    (46_100, 1_100, 180, 2_050, "turn", True),
    (52_700, 480, 0, 980, "naming", True),
    (61_400, 3_600, 1_240, 5_870, "turn", True),
    (74_900, 1_750, 320, 2_640, "turn", True),
    (88_300, 210, 0, 1_420, "turn", False),
    (96_800, 2_150, 540, 3_180, "turn", True),
    (112_400, 1_320, 210, 2_260, "turn", True),
    (128_900, 4_800, 1_900, 7_240, "turn", True),
    (141_200, 760, 0, 1_510, "aside", True),
    (158_600, 2_400, 620, 3_450, "turn", True),
    (172_300, 1_180, 240, 2_180, "turn", True),
    (186_700, 5_200, 2_100, 8_960, "turn", True),
    # The compaction pair: expensive reads, and the context it buys back.
    (194_100, 3_100, 0, 6_320, "compaction", True),
    (42_600, 1_450, 280, 2_390, "turn", True),
]


def seed_populated(session_id: str) -> None:
    """Write the demo session into the ambient (isolated) ledger.

    A session that VARIES, because a constant fixture makes the two best new
    charts render as a wall of identical bars and a degenerate
    ``2,800 ms-2,800 ms`` timing range (QA Q3, echoed by the design and code
    rounds). Those frames are correct renderings of uniform input, but they are
    what a reader judges the feature by, and they make a working chart look
    broken. ``POPULATED_SHAPE`` is what a real session does: context grows as
    history accumulates, a compaction resets it, a couple of turns are large,
    one fails, and a cheap model handles the short asides. No number is
    load-bearing here - only the spread is.

    Shared with the cost-mode capture so the two never seed different sessions.
    """
    store = AnalyticsStore()
    snapshots = []
    for i, (context, output, reasoning, duration, purpose, ok) in enumerate(POPULATED_SHAPE):
        # The cheap model takes the short non-turn work, so `By model` has a
        # real split to draw rather than one dominant row.
        aside = purpose in ("aside", "naming")
        cache_read = int(context * 0.72)
        snapshots.append(
            replace(
                _snap(
                    session_id=session_id,
                    provider="openai" if aside else "anthropic",
                    model_id="gpt-5.2-mini" if aside else "claude-sonnet-4-6",
                    context=context,
                    input_tokens=context - cache_read - 800,
                    cache_read=cache_read,
                    cache_write=800,
                    output_tokens=output,
                    reasoning=reasoning,
                    # Roughly list price for the tokens, so `t` (cost) gives a
                    # different ordering rather than a rescale of the same one.
                    cost_micro=int(context * 0.55 + output * 8.5),
                    chars={
                        "system_prompt": 400,
                        "custom_instructions": 110,
                        "tool_inventory": 150,
                        "tool_schemas": 250,
                        # Conversation and tool results grow with context; a
                        # flat split makes `Where input went` identical on
                        # every frame.
                        "conversation": 300 + i * 95,
                        "tool_results": 120 + i * 60,
                    },
                    ts_ms=1788602400000 + i * 47000,
                    ok=ok,
                ),
                request_id=f"logical-request-{i:03d}",
                purpose=purpose,
                outcome="ok" if ok else "error",
                duration_ms=duration,
                ttft_ms=int(duration * 0.19) + 180,
                preparation_ms=18 + (i % 5) * 7,
            )
        )
    store.record_batch(snapshots)
    store.close()


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = sys.argv[2].split("x")
    size = (int(cols), int(rows))
    scenario = sys.argv[3] if len(sys.argv) > 3 else "populated"
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
    if scenario == "populated":
        seed_populated(session.session_id)
    elif scenario == "unavailable":
        path = default_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic corrupt ledger")
    ledger = default_db_path()
    before_digest = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        save_capture(app, str(out / "opened.svg"))
        await pilot.pause()
        save_capture(app, str(out / "settled.svg"))
        if type(app.screen).__name__ == "SessionScreen":
            for page in range(1, 7):
                await pilot.press("pagedown")
                await pilot.pause()
                save_capture(app, str(out / f"page-{page}.svg"))
        await pilot.press("end")
        await pilot.pause()
        save_capture(app, str(out / "bottom.svg"))
        screen = app.screen
        scroll = getattr(screen, "_scroll", None)
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "scenario": scenario,
            "size": size,
            "screen": type(screen).__name__,
            "screen_geometry": {
                "size": list(screen.size),
                "virtual_size": list(screen.virtual_size),
                "region": list(screen.region),
            },
            "prompts": session.prompts,
            "ledger_unchanged": before_digest
            == (hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None),
            "scroll": (
                {
                    "size": list(scroll.size),
                    "virtual_size": list(scroll.virtual_size),
                    "max_x": scroll.max_scroll_x,
                    "max_y": scroll.max_scroll_y,
                }
                if scroll
                else None
            ),
        }
        await pilot.press("escape")
        await pilot.pause()
        save_capture(app, str(out / "closed.svg"))
        metrics["composer_focused_after_close"] = app.focused is app.query_one(Editor)
        (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics))


if __name__ == "__main__":
    asyncio.run(main())
