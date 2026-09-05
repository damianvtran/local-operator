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
        return "session-diagnostic-demo-0123456789abcdef"

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
        store = AnalyticsStore()
        snapshots = []
        for i in range(18):
            snapshots.append(
                replace(
                    _snap(
                        session_id=session.session_id,
                        provider="anthropic" if i < 12 else "openai",
                        model_id="claude-sonnet-4-6" if i < 12 else "gpt-5.2",
                        context=28400,
                        input_tokens=4400,
                        cache_read=23000,
                        cache_write=1000,
                        output_tokens=1700,
                        reasoning=400,
                        cost_micro=18500,
                        chars={
                            "system_prompt": 400,
                            "tool_inventory": 150,
                            "tool_schemas": 250,
                            "conversation": 1400,
                            "tool_results": 600,
                        },
                        ts_ms=1788602400000 + i * 10000,
                    ),
                    request_id=f"logical-request-{i:03d}-0123456789abcdef",
                    purpose="turn" if i < 16 else "compaction",
                    outcome="ok",
                    duration_ms=2800 + i * 15,
                    ttft_ms=350 + i * 10,
                    preparation_ms=24,
                )
            )
        store.record_batch(snapshots)
        store.close()
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
