"""Capture the `/model` picker's aggregator rows, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/model_picker_shot.py OUT.svg [COLSxROWS]

Exists for the `radient/auto` metadata round: the router's row is the one
under test, and the two facts it must state — a 1M context window and a
usage-based price rather than the word `free` — are only judgeable as a
rendered frame, because the price column is width-constrained and a longer
label is exactly what would push the numbers run off the edge.

Drives the real ``OperatorApp`` rather than a bare widget host on purpose:
the lightweight hosts in the test files declare no ``CSS_PATH``, so
``local_operator.tcss`` never applies to them and a still captured from one
cannot show what the user sees (see AGENTS.md, "Visual validation").

The router's own row is built by running the REAL pipeline over a literal
`/v1/models` payload — listing entry -> ``DiscoveredModel`` -> ``_price`` ->
``ModelRow`` — rather than by hand-typing the numbers. That is what makes the
before/after pair evidence: the payload is fixed, so the only thing that can
change between the two captures is the code under test. Both payload shapes
are captured, because a user's cached listing may still be either one.

The remaining rows are literal context: they exist so the three price states
(priced / genuinely free / unknown) stay distinguishable in the same frame.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.model.discovery import _row_from_openai_entry  # noqa: E402
from local_operator.providers.controller import _price  # noqa: E402
from local_operator.providers.registry import get_provider_definition  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.model_picker import ModelRow  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: What Radient's `/v1/models` quotes for the router TODAY — no context length,
#: no architecture, and pricing written as an explicit `"0"`. Every user whose
#: cached listing predates the server fix still resolves through this shape.
CACHED_PAYLOAD = {
    "id": "auto",
    "name": "Automatic",
    "pricing": {"prompt": "0", "completion": "0"},
}

#: What it quotes once the server-side half of this round lands: the meta-route
#: shape OpenRouter already publishes for `openrouter/auto`, where `"-1"` means
#: "the cost depends on which model this routes to".
FIXED_PAYLOAD = {
    "id": "auto",
    "name": "Automatic",
    "context_length": 1_048_576,
    "pricing": {"prompt": "-1", "completion": "-1"},
    "architecture": {"input_modalities": ["text", "image", "file"]},
}


def _router_row(provider: str, payload: dict[str, object], label: str) -> ModelRow:
    """One picker row for ``payload``, built the way the running app builds it.

    Deliberately goes through the same three seams the live catalogue does, so
    a fix that only works in a unit test cannot produce a correct frame here.
    The window falls back to the resolved SPEC when the listing states none,
    which is precisely the substitution that made the router read as 128k.
    """
    definition = get_provider_definition(provider)
    assert definition is not None
    row = _row_from_openai_entry(payload)
    assert row is not None
    context_window = row.context_window or build_model_spec(provider, row.id).context_window
    return ModelRow(
        provider=provider,
        model_id=row.id,
        label=label,
        context_window=context_window,
        input_price=_price(row.input_price, definition, free=row.free),
        output_price=_price(row.output_price, definition, free=row.free),
        aggregated=True,
        routed=row.routed,
    )


ROWS = [
    _router_row("radient", CACHED_PAYLOAD, "Automatic (cached listing)"),
    _router_row("radient", FIXED_PAYLOAD, "Automatic (fixed listing)"),
    ModelRow(
        provider="anthropic",
        model_id="claude-opus-5",
        label="Claude Opus 5",
        context_window=200_000,
        input_price=15.0,
        output_price=75.0,
    ),
    ModelRow(
        provider="openrouter",
        model_id="google/gemma-4-31b-it:free",
        label="Gemma 4 31B",
        context_window=131_072,
        input_price=0.0,
        output_price=0.0,
        aggregated=True,
    ),
    ModelRow(
        provider="openai",
        model_id="gpt-5.4",
        label="GPT-5.4",
        context_window=400_000,
        input_price=1.25,
        output_price=10.0,
    ),
]


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        picker = app._editor().model_picker
        picker.set_rows(ROWS, current="radient/auto", status="")
        picker.open()
        # Let the overlay's layout SETTLE before the capture. One pause paints a
        # frame whose rows are still sized to the pre-open composer width, and
        # the ids come out truncated in a way the running app never shows.
        for _ in range(4):
            await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        # `_repaint` renders against `self.size.width`, which is still the
        # pre-open width on the frame `open()` itself paints. Repainting once
        # the overlay has been laid out is what the running app gets from the
        # next resize/keystroke; without it the capture shows truncated ids the
        # user never sees.
        picker._repaint()
        await pilot.pause()

        # The numbers behind the frame: a label that fits in isolation can
        # still overflow the row, and only the widget's own geometry says so.
        print(f"picker.size={picker.size}", file=sys.stderr)
        for row in ROWS:
            print(
                f"  {row.provider}/{row.model_id}: "
                f"numbers={picker._numbers(row)!r} price={picker._price(row)!r}",
                file=sys.stderr,
            )
        # The PAINTED lines, not just the fields: a price that fits the column in
        # isolation can still be the thing that truncates the id beside it, and
        # only the assembled row shows that.
        for line in picker.render_rows(picker.size.width):
            print(f"  |{line.plain}|", file=sys.stderr)
        app.save_screenshot(out)


if __name__ == "__main__":
    # Guarded so the row fixtures above can be imported by a geometry probe
    # without the import also capturing a frame.
    asyncio.run(main())
