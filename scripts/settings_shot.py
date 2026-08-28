"""Capture the ``/settings`` page for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/settings_shot.py OUT.svg [COLSxROWS] [STATE]

Modelled on ``scripts/ask_shot.py``: it drives the REAL :class:`OperatorApp`,
which is the only host that loads ``local_operator.tcss`` — the lightweight
hosts in the test files declare no ``CSS_PATH``, so a still captured from one
of them cannot show a stylesheet change at all (AGENTS.md, "Visual
validation").

A scratch config dir is used for every capture (``LOCAL_OPERATOR_CONFIG_DIR``),
so the frames show a KNOWN configuration rather than the developer's own — the
same class of trap ``approval_shot.py`` records for ``tool_approval_mode``,
where a machine set to ``auto`` renders a frame with no prompt in it and the
surface looks broken rather than skipped. It also means a capture can never
write to the real config.

STATE selects what the page is showing:

    overview   the page as it opens (default)
    enum       an enum row expanded into its choices
    error      a text editor open with a rejected value and its inline error
    cascade    the failover cascade editor with a chain open
    teams      the read-only teams pane
    agents     the read-only agents pane
    retired    scrolled to the retired section
    frames     TWO consecutive frames (OUT.svg and OUT.frame2.svg) to prove the
               opening layout settles rather than reflowing after paint
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The scratch config dir has to exist BEFORE the app imports anything that
# resolves `config_dir()`, which is why this runs above the local imports.
_SCRATCH = tempfile.mkdtemp(prefix="lo-settings-shot-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

from local_operator.config import ConfigManager  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.settings_view import SettingsView  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Rows the read-only pane shows. Injected rather than resolved from a real
#: registry so the frame is the same on every machine — a capture whose pane
#: depended on the developer's own teams would not be comparable.
TEAMS = [
    ("lopdev", "manager · 6 roles", "ships local-operator changes end to end"),
    ("research", "manager · 3 roles", "background reading and summarisation"),
]
AGENTS = [
    ("architect", "role · effort hi", "structural and cross-subsystem decisions"),
    ("coder", "role · effort med", "implements one bounded slice"),
    ("reviewer", "role · effort hi", "posts the agent review round"),
]
PROVIDERS = [
    ("anthropic", "signed in"),
    ("openrouter", "api key"),
]


def _seed_config() -> None:
    """Give the scratch config a few NON-default values.

    A frame of an all-default config cannot show the changed-vs-default
    styling, which is half of what the page is for — so the capture writes the
    same handful of settings a real user would have touched.
    """
    manager = ConfigManager(Path(_SCRATCH))
    manager.set_config_value("hosting", "anthropic")
    manager.set_config_value("model_name", "claude-opus-5")
    manager.set_config_value("tool_approval_mode", "auto")
    manager.set_config_value("display.shimmer", False)
    retry = dict(manager.get_config_value("retry", {}) or {})
    retry["maxRetries"] = 4
    retry["fallbackChains"] = {
        "default": ["anthropic/claude-opus-5", "openrouter/deepseek/deepseek-chat"],
        "cheap": ["openrouter/qwen/qwen3-coder"],
    }
    manager.set_config_value("retry", retry)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    state = sys.argv[3] if len(sys.argv) > 3 else "overview"

    _seed_config()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # Seed a conversation, so "does leaving the page put my transcript
        # back?" is an answerable question rather than an empty-screen no-op —
        # the same reason ask_shot.py seeds one.
        for turn in range(1, 4):
            app._append_block(UserBlock(f"Turn {turn}: can I change the default model?"))
            prose = AssistantBlock()
            prose.update_text(f"Answer {turn}: yes — /settings has it under Model.")
            app._append_block(prose)
        await pilot.pause()

        app._open_settings_view()
        view = app.query_one(SettingsView)
        view.load(teams=TEAMS, agents=AGENTS, providers=PROVIDERS)
        await pilot.pause()

        if state == "frames":
            # CONSECUTIVE frames. A two-pane layout inside a mode is exactly
            # where a post-paint reflow shows: if frame 1 differs from frame 2,
            # the user sees motion on open whether or not anyone intended it.
            app.save_screenshot(out)
            await pilot.pause()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".frame2.svg"))
        elif state == "enum":
            _select(view, "tool_approval_mode")
            view.action_activate()
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "error":
            _select(view, "retry.maxRetries")
            view.action_activate()
            await pilot.pause()
            view._buffer = "9999"
            view._commit_edit()
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "cascade":
            _select_chain(view, "default")
            view.action_activate()
            await pilot.pause()
            app.save_screenshot(out)
        elif state in ("teams", "agents"):
            while view._pane != state:
                view.action_pane(1)
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "retired":
            for _ in range(len(view._rows)):
                view.action_jump(1)
                break
            await pilot.pause()
            app.save_screenshot(out)
        else:
            app.save_screenshot(out)

        # The geometry behind the pixels (AGENTS.md step 4). A scrollbar on the
        # SCREEN is always a bug on this app — the body scrolls, the dock is
        # docked — and it costs two cells of width silently.
        screen = app.screen
        print(
            f"state={state} size={size} "
            f"screen.size={tuple(screen.size)} "
            f"screen.virtual_size={tuple(screen.virtual_size)} "
            f"screen.show_vertical_scrollbar={screen.show_vertical_scrollbar} "
            f"body.size={tuple(view._body.size)} "
            f"body.virtual_size={tuple(view._body.virtual_size)} "
            f"body.show_vertical_scrollbar={view._body.show_vertical_scrollbar} "
            f"pane.size={tuple(view._pane_view.size)} "
            f"rows={len(view._rows)} "
            f"hints={view.rendered_hints()!r}"
        )


def _select(view: SettingsView, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.setting is not None and row.setting.key == key and row.kind == "setting":
            view._selected = index
            view._repaint()
            # Scroll it into view, or the capture frames a cursor that is not
            # on screen and the expansion under it is invisible.
            view._scroll_to_selection()
            return
    raise SystemExit(f"no row for {key}")


def _select_chain(view: SettingsView, chain: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "chain" and row.chain == chain:
            view._selected = index
            view._repaint()
            view._scroll_to_selection()
            return
    raise SystemExit(f"no chain row for {chain}")


asyncio.run(main())
