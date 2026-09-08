"""Capture /session's failure accounting and tool-call rows, real slash path.

Usage: python scripts/session_failure_shot.py OUTDIR 100x30 healthy|failing|tools

Exists because the operator's bug report ("15 req · 15 failed" on a session
where nothing failed) is invisible to the unit suite: the fixtures asserted an
``outcome`` vocabulary the recorder cannot emit. Every scenario here seeds the
REAL vocabulary the provider path writes (``stop``/``toolUse``/``length``, or an
exception class name), so a frame from this script is evidence about the
shipping product rather than about a fixture.

- ``healthy``  every request ok, mixed real outcomes -> must show NO failures.
- ``failing``  three genuinely failed requests -> must still report them.
- ``tools``    healthy plus recorded tool calls -> the measured tool rows.

No provider request, live session or operator config is used; the capture
sandboxes HOME before any application import.
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
import json  # noqa: E402
from dataclasses import replace  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.harness.types import ModelSpec  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402


class DiagnosticSession(FakeSession):
    frontend_state: FrontendSessionState

    @property
    def session_id(self) -> str:
        return "b71d40e9a2c5"

    @property
    def model(self) -> ModelSpec:
        return ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6", context_window=200000)

    @property
    def effective_model(self) -> ModelSpec:
        return self.model

    @property
    def model_label(self) -> str:
        return "anthropic/claude-sonnet-4-6"

    @property
    def effective_model_label(self) -> str:
        return "anthropic/claude-sonnet-4-6"


#: (context, output, duration_ms, purpose, ok, outcome). The outcome strings are
#: the ones ``_record_stream`` actually writes: ``str(stop_reason)`` from the
#: provider adapters' finish-reason map, or an exception class name on a failure.
#: The literal ``"ok"`` is deliberately absent — nothing in the product emits it,
#: and a fixture that does is how this defect shipped.
HEALTHY_SHAPE = [
    (8_400, 900, 1_850, "turn", True, "toolUse"),
    (14_200, 1_400, 2_310, "turn", True, "toolUse"),
    (21_800, 620, 1_240, "turn", True, "stop"),
    (33_500, 2_900, 4_120, "turn", True, "toolUse"),
    (46_100, 1_100, 2_050, "turn", True, "toolUse"),
    (52_700, 480, 980, "naming", True, "stop"),
    (61_400, 3_600, 5_870, "turn", True, "toolUse"),
    (74_900, 1_750, 2_640, "turn", True, "toolUse"),
    (88_300, 210, 1_420, "turn", True, "stop"),
    (96_800, 2_150, 3_180, "turn", True, "toolUse"),
    (112_400, 1_320, 2_260, "turn", True, "toolUse"),
    (128_900, 4_800, 7_240, "turn", True, "length"),
    (141_200, 760, 1_510, "aside", True, "stop"),
    (158_600, 2_400, 3_450, "turn", True, "toolUse"),
    (172_300, 1_180, 2_180, "turn", True, "stop"),
    (186_700, 5_200, 8_960, "turn", True, "toolUse"),
]

#: The same session with three REAL failures, so the guard is shown to still
#: fire. ``ok=False`` is the recorded truth; the outcome string only labels it.
FAILING_SHAPE = HEALTHY_SHAPE[:13] + [
    (158_600, 0, 3_450, "turn", False, "ProviderError"),
    (172_300, 0, 2_180, "turn", False, "ProviderError"),
    (186_700, 0, 8_960, "turn", False, "refusal"),
]

#: Tool calls for the ``tools`` scenario: (tool_name, origin, fault). Mirrors a
#: plausible run — mostly clean, one hallucinated tool, two schema violations,
#: one duplicate emission, plus non-model faults that must stay OUT of the
#: validity numerator.
TOOL_SHAPE = (
    [("read", "model", "")] * 9
    + [("bash", "model", "")] * 7
    + [("grep", "model", "")] * 4
    + [("edit", "model", "")] * 3
    + [("eval", "model", "")] * 2
    + [("read", "nested", "")] * 3
    + [("reed_file", "model", "unknown_tool")]
    + [("edit", "model", "invalid_arguments"), ("bash", "model", "invalid_arguments")]
    + [("read", "model", "duplicate_id")]
    + [("web_fetch", "model", "execution"), ("bash", "model", "execution")]
    + [("bash", "model", "denied")]
)


def seed(session_id: str, shape: list[tuple[int, int, int, str, bool, str]]) -> None:
    store = AnalyticsStore()
    snapshots = []
    for i, (context, output, duration, purpose, ok, outcome) in enumerate(shape):
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
                    reasoning=int(output * 0.2),
                    cost_micro=int(context * 0.55 + output * 8.5),
                    chars={
                        "system_prompt": 400,
                        "custom_instructions": 110,
                        "tool_inventory": 150,
                        "tool_schemas": 250,
                        "conversation": 300 + i * 95,
                        "tool_results": 120 + i * 60,
                    },
                    ts_ms=1788602400000 + i * 47000,
                    ok=ok,
                ),
                request_id=f"logical-request-{i:03d}",
                purpose=purpose,
                outcome=outcome,
                duration_ms=duration,
                ttft_ms=int(duration * 0.19) + 180,
                preparation_ms=18 + (i % 5) * 7,
            )
        )
    store.record_batch(snapshots)
    store.close()


def seed_tool_calls(session_id: str) -> None:
    store = AnalyticsStore()
    store.record_tool_calls(
        [
            (1788602400000 + i * 3100, session_id, name, origin, fault, 40.0 + i)
            for i, (name, origin, fault) in enumerate(TOOL_SHAPE)
        ]
    )
    store.close()


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = sys.argv[2].split("x")
    size = (int(cols), int(rows))
    scenario = sys.argv[3] if len(sys.argv) > 3 else "healthy"
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
    seed(session.session_id, FAILING_SHAPE if scenario == "failing" else HEALTHY_SHAPE)
    if scenario == "tools":
        seed_tool_calls(session.session_id)

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        save_capture(app, str(out / "opened.svg"))
        screen = app.screen
        pages = []
        for page in range(1, 8):
            await pilot.press("pagedown")
            await pilot.pause()
            save_capture(app, str(out / f"page-{page}.svg"))
            pages.append(page)
        scroll = getattr(screen, "_scroll", None)
        # Geometry, not just pixels: virtual_size > size on THIS screen is
        # always a bug (a scrollbar costs two cells and reflows the transcript).
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "scenario": scenario,
            "size": size,
            "screen": type(screen).__name__,
            "screen_size": list(screen.size),
            "screen_virtual_size": list(screen.virtual_size),
            "screen_scrollbar": bool(getattr(screen, "show_vertical_scrollbar", False)),
            "scroll": (
                {
                    "size": list(scroll.size),
                    "virtual_size": list(scroll.virtual_size),
                    "max_y": scroll.max_scroll_y,
                }
                if scroll
                else None
            ),
        }
        # The rendered text, so a frame's claim can be grepped rather than
        # eyeballed. Taken from the screen's OWN body widget, which is the exact
        # Text the SVG paints — not a re-render with different inputs.
        text = getattr(screen, "_report_text", None)
        if text is not None:
            metrics["lines"] = str(text()).splitlines()
        (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "lines"}))
        for line in metrics.get("lines", []):
            print(line)


if __name__ == "__main__":
    asyncio.run(main())
