"""Capture the /settings provider/model suggestion editors for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/settings_suggest_shot.py OUT.svg [COLSxROWS] [STATE]

Drives the REAL :class:`OperatorApp`, which is the only host that loads
``local_operator.tcss`` (AGENTS.md, "Visual validation"). A scratch config dir
is used for every capture so the frames show a KNOWN configuration and can never
touch the developer's own settings — the same discipline ``settings_shot.py``
follows.

STATE selects what the page is showing:

    provider   the Default provider editor open, typed "a", suggestion dropdown
               with ghost-text completion of the top provider match
    provider-empty  the Default provider editor open with an empty buffer,
               showing the full provider suggestion list
    model      the Default model editor open, typed "opus", fuzzy suggestion
               dropdown with ghost-text completion of the top model match
    model-empty  the Default model editor open with an empty buffer, showing
               the catalogue ordered as /model orders it
    custom     the Default model editor holding a value the catalogue does NOT
               know (a custom endpoint id), proving suggestions assist but do
               not hard-constrain — the frame shows no ghost and the buffer is
               committable as typed
    provider-windowed  the Default provider editor open against a catalogue
               LARGER than the visible window, highlight moved into the middle,
               so the frame shows the `pos/total` count cue on the highlighted
               row and the windowed 8-of-N list (review round 1, U3)

The provider and model catalogues are INJECTED (same shape the app resolves for
the real page) so the frame is identical on every machine.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = tempfile.mkdtemp(prefix="lo-suggest-shot-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

from local_operator.config import ConfigManager  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.model_picker import ModelRow  # noqa: E402
from local_operator.tui.widgets.settings_view import SettingsView  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

TEAMS = [("lopdev", "manager · 6 roles", "ships local-operator changes end to end")]
AGENTS = [("coder", "role · effort med", "implements one bounded slice")]
PROVIDERS = [("anthropic", "signed in"), ("openrouter", "api key")]

#: A representative slice of the provider login catalogue, as ``(id, label)``.
PROVIDER_CATALOGUE = [
    ("anthropic", "Anthropic"),
    ("openai", "OpenAI"),
    ("openrouter", "OpenRouter"),
    ("google", "Google"),
    ("mistral", "Mistral"),
    ("deepseek", "DeepSeek"),
    ("alibaba", "Alibaba"),
    ("radient", "Radient"),
]

#: A representative slice of the model catalogue, as ModelRows.
MODEL_CATALOGUE = [
    ModelRow("anthropic", "claude-opus-5", "Claude Opus 5", 200_000, 15.0, 75.0),
    ModelRow("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5", 200_000, 3.0, 15.0),
    ModelRow("openai", "gpt-5.2", "GPT-5.2", 400_000, 2.5, 10.0),
    ModelRow("openai", "gpt-5.2-mini", "GPT-5.2 mini", 400_000, 0.4, 1.6),
    ModelRow("google", "gemini-3-pro", "Gemini 3 Pro", 1_000_000, 2.0, 12.0),
    ModelRow(
        "openrouter",
        "anthropic/claude-opus-5",
        "Claude Opus 5",
        200_000,
        16.0,
        78.0,
        aggregated=True,
    ),
    ModelRow("deepseek", "deepseek-chat", "DeepSeek Chat", 128_000, 0.3, 1.1),
]


def _seed() -> None:
    m = ConfigManager(Path(_SCRATCH))
    m.set_config_value("hosting", "anthropic")
    m.set_config_value("model_name", "claude-opus-5")


def _select(view: SettingsView, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            view._scroll_to_selection()
            return
    raise SystemExit(f"no row for {key}")


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    state = sys.argv[3] if len(sys.argv) > 3 else "provider"

    _seed()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        for turn in range(1, 3):
            app._append_block(UserBlock(f"Turn {turn}: change the default model?"))
            prose = AssistantBlock()
            prose.update_text(f"Answer {turn}: /settings has it under Model.")
            app._append_block(prose)
        await pilot.pause()

        app._open_settings_view()
        view = app.query_one(SettingsView)
        view.load(
            teams=TEAMS,
            agents=AGENTS,
            providers=PROVIDERS,
            provider_catalogue=PROVIDER_CATALOGUE,
            model_catalogue=MODEL_CATALOGUE,
        )
        await pilot.pause()

        def _type(text: str) -> None:
            for ch in text:
                view._buffer = view._buffer[: view._caret] + ch + view._buffer[view._caret :]
                view._caret += 1
            view._repaint()

        if state == "provider":
            _select(view, "hosting")
            view.action_activate()
            view._buffer = ""
            view._caret = 0
            _type("a")
        elif state == "provider-empty":
            _select(view, "hosting")
            view.action_activate()
            view._buffer = ""
            view._caret = 0
            view._repaint()
        elif state == "model":
            _select(view, "model_name")
            view.action_activate()
            view._buffer = ""
            view._caret = 0
            _type("opus")
        elif state == "model-empty":
            _select(view, "model_name")
            view.action_activate()
            view._buffer = ""
            view._caret = 0
            view._repaint()
        elif state == "custom":
            _select(view, "model_name")
            view.action_activate()
            view._buffer = ""
            view._caret = 0
            _type("my-endpoint/custom-model-v2")
        elif state == "provider-windowed":
            # A catalogue larger than the 8-row window, with the highlight moved
            # into the middle, so the frame proves the `pos/total` cue and the
            # windowing behaviour a short list never exercises. Re-load with the
            # bigger catalogue rather than the module default.
            view.load(
                teams=TEAMS,
                agents=AGENTS,
                providers=PROVIDERS,
                provider_catalogue=[(f"prov{n:02d}", f"Provider {n:02d}") for n in range(12)],
                model_catalogue=MODEL_CATALOGUE,
            )
            _select(view, "hosting")
            view.action_activate()
            view._buffer = ""
            view._caret = 0
            view._suggest_index = 3
            view._repaint()
        else:
            raise SystemExit(f"unknown state {state}")

        await pilot.pause()
        app.save_screenshot(out)
        print(
            f"state={state} buffer={view._buffer!r} "
            f"suggestions={view.suggestion_labels_for_test()!r} "
            f"ghost={view.ghost_text_for_test()!r}"
        )


asyncio.run(main())
