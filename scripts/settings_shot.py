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
    cascade-row  the failover cascade SETTING row highlighted (OUT.svg) and
               activated (OUT.open.svg). What `enter` does on that row is the
               subject of #440: it used to open a free-text editor seeded with
               the cascade's Python repr, and committing that repr destroyed
               every chain, so a still of the resting row alone cannot show
               the fix
    cascade-corrupt  the page as a #440 VICTIM sees it (OUT.svg), then after
               pressing `r` (OUT.cleared.svg). The scratch config is seeded
               with the repr string the shipped bug stored, so this is the one
               state that photographs the aftermath rather than the bug: the
               value column and the group line under it used to state
               contradictory things, and `r` used to refuse to help
    confirm    `d` on a chain row, asking before it deletes the whole chain
    confirm-long  the same ask on a 26-character chain name, which is what
               shows whether the ask is budgeted against the row it is
               painted into (D12) rather than against the settings list
    teams      the read-only teams pane
    agents     the read-only agents pane
    reset-default  a row that IS at its shipped default (OUT.svg), then the
               same row after pressing `r` (OUT.pressed.svg). `r` used to WRITE
               here — on a config with no file at all it created one — so the
               subject of this pair is the footer: whether the page offers a
               key that has nothing to restore. A third frame,
               OUT.offdefault.svg, is the SAME row once it is off-default,
               which is what proves the hint is being withheld by state rather
               than removed (#440)
    retired    scrolled to the retired section — the read-only bottom row the
               clamp now parks users on, whose footer must not advertise keys
               that cannot act on it
    top        TRAVELLED back to the first row by held `up`, which is the frame
               that shows whether the section header owning it is on screen
    fork       the Fork section's rows, scrolled into view
    fork-open  the same, with `fork.mode` expanded into its choices
    fork-placement       the cmux placement row, scrolled into view
    fork-placement-open  the same, expanded into workspace/surface
    theme      the Theme row highlighted (OUT.svg) and activated (OUT.open.svg),
               which is the affordance review round 1 m1 changed: TEXT opened a
               free-text editor, ENUM expands the registry's themes as choices
    frames     TWO consecutive frames (OUT.svg and OUT.frame2.svg) to prove the
               opening layout settles rather than reflowing after paint
    boot       the page opened from the BOOT/splash state, with NO conversation
               seeded, plus OUT.splash.svg taken immediately after leaving it.
               The boot layout is a whole second layout (docked centred card,
               bottom-aligned transcript), so a settings frame captured over a
               populated transcript cannot show what it does to this page — and
               the splash frame is what proves leaving puts the composition
               back rather than approximating it.
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

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.settings_io import Setting  # noqa: E402
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
# THREE providers, not two. The provider section sheds by height, and with two
# entries the frames could not show the difference between shedding honestly and
# dropping content: at 20-26 rows the pane ate signed-in providers and painted a
# bare `providers` header, which reads as "none configured" (design round 3,
# D11). Three is the smallest roster where a fold to `… N more` is visibly a
# count rather than a single missing row.
PROVIDERS = [
    ("anthropic", "signed in"),
    ("openrouter", "api key"),
    ("openai", "api key"),
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
    # A LONG chain name alongside the short ones. `default` is 7 characters and
    # fits at every width, so a confirm frame built on it demonstrates no
    # truncation behaviour at all — which is why the committed `confirm-80x24`
    # was captioned for a shed it did not show (design round 3, D13), and why
    # the over-clipping at 100 and 140 columns went unphotographed (D12).
    retry["fallbackChains"] = {
        "default": ["anthropic/claude-opus-5", "openrouter/deepseek/deepseek-chat"],
        "cheap": ["openrouter/qwen/qwen3-coder"],
        "openrouter-budget-fallback": [
            "openrouter/deepseek/deepseek-chat",
            "openrouter/qwen/qwen3-coder",
        ],
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
        # the same reason ask_shot.py seeds one. The `boot` state deliberately
        # seeds NOTHING: its whole subject is the layout the splash puts up,
        # and one appended block retires it.
        if state != "boot":
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

        geometry: str | None = None
        if state == "boot":
            # Opened over the splash. The second frame is taken AFTER leaving,
            # because the failure this state exists to photograph has two
            # halves: the page has to take the screen, and the composition it
            # displaced has to come back whole (splash, card clamp, and the
            # rows `_sync_boot_composition` reserves below the card).
            app.save_screenshot(out)
            # Measured while the page is still up: the geometry line below is
            # the numbers behind THIS frame, and the view is unmounted a
            # moment from now.
            geometry = _geometry(app, view, state, size)
            app._close_settings_view()
            await pilot.pause()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".splash.svg"))
            welcome = app._welcome
            geometry += (
                " || after leaving: "
                f"boot={app.screen.has_class('boot')} "
                f"boot-card={app.screen.has_class('boot-card')} "
                f"welcome.display={welcome.display if welcome is not None else None} "
                f"welcome.height={welcome.size.height if welcome is not None else None} "
                f"dock.height={app.query_one('#input-dock').size.height}"
            )
        elif state == "frames":
            # CONSECUTIVE frames. A two-pane layout inside a mode is exactly
            # where a post-paint reflow shows: if frame 1 differs from frame 2,
            # the user sees motion on open whether or not anyone intended it.
            app.save_screenshot(out)
            await pilot.pause()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".frame2.svg"))
        elif state == "fork":
            _select(view, "fork.mode")
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "fork-open":
            # ENUM rows expand in place, so the choices and their descriptions
            # are only visible activated — which is the state a user picking
            # `switch` actually sees.
            _select(view, "fork.mode")
            view.action_activate()
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "fork-placement":
            _select(view, "fork.cmux_placement")
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "fork-placement-open":
            _select(view, "fork.cmux_placement")
            view.action_activate()
            await pilot.pause()
            app.save_screenshot(out)
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
        elif state == "cascade-corrupt":
            # TWO frames of the RECOVERY, not of the bug. The corrupt value is
            # written through the page's own writer (see `_corrupt_cascade`) so
            # the frame shows a state a real config can be in, and the second
            # frame is what `r` leaves behind — which is the half a still of
            # the broken page alone cannot show.
            _corrupt_cascade()
            view._manager.reload()
            _select(view, "retry.fallbackChains")
            await pilot.pause()
            view._repaint()
            # The GROUP has to be in frame, not just the cursor. U1 is a
            # contradiction BETWEEN the value column and the line under it, so
            # a frame scrolled to the setting row alone — which is where
            # `_scroll_to_selection` stops, since the row is already visible —
            # photographs one half of it and proves nothing.
            _scroll_to_show_group(view)
            await pilot.pause()
            app.save_screenshot(out)
            geometry = _geometry(app, view, state, size)
            view.action_reset()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".cleared.svg"))
            geometry += f" || after r: notice={view.notice_text!r}"
        elif state == "cascade-row":
            # TWO frames, for the reason the `theme` state takes two: the bug
            # is in what ACTIVATION does, not in how the row rests. Before the
            # #440 fix the second frame shows an inline editor holding
            # `{'default': [...]}` on a row that has no scalar to edit.
            _select(view, "retry.fallbackChains")
            await pilot.pause()
            app.save_screenshot(out)
            view.action_activate()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".open.svg"))
        elif state == "confirm-long":
            # The SAME ask on a 26-character chain name. The `confirm` frames
            # use `default` (7 characters), which fits everywhere and therefore
            # photographs none of the width behaviour the ask actually has
            # (design round 3, D12/D13).
            _select_chain(view, "openrouter-budget-fallback")
            view._delete_hop()
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "confirm":
            # The UX round 1 U5 fix: `d` on a CHAIN row asks first, because it
            # destroys every hop in it and immediate-write has no undo. A still
            # of the chain row alone would not show the ask, which lives in the
            # detail line.
            _select_chain(view, "default")
            view._delete_hop()
            await pilot.pause()
            app.save_screenshot(out)
        elif state in ("teams", "agents"):
            while view._pane != state:
                view.action_pane(1)
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "theme":
            # TWO frames: the row at rest, then the row activated. The m1 fix
            # changes what activation DOES (free-text editor -> choice
            # expansion), so a still of the resting row alone would not show it.
            _select(view, "tui.theme")
            await pilot.pause()
            app.save_screenshot(out)
            view.action_activate()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".open.svg"))
        elif state == "reset-default":
            # THREE frames. The `r` affordance is a footer question, and a
            # single still of a row cannot answer it: the pair that matters is
            # the same row at default and off-default, because only the
            # difference between them shows the hint is state-driven rather
            # than simply gone. `tui.shimmer` is used because `_seed_config`
            # writes it, so both sides are reachable on one config.
            _select(view, "display.shimmer")
            settings_io.reset_setting(view._manager, _require("display.shimmer"))
            view._manager.reload()
            view._repaint()
            await pilot.pause()
            app.save_screenshot(out)
            geometry = _geometry(app, view, state, size)
            view.action_reset()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".pressed.svg"))
            geometry += f" || after r at default: notice={view.notice_text!r}"
            settings_io.write_setting(view._manager, _require("display.shimmer"), False)
            view._manager.reload()
            view._repaint()
            await pilot.pause()
            app.save_screenshot(out.replace(".svg", ".offdefault.svg"))
            geometry += f" || off-default hints={view.rendered_hints()!r}"
        elif state == "retired":
            for _ in range(len(view._rows)):
                view.action_jump(1)
                break
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "top":
            # TRAVELLED to the top, not opened at it. The two are not the same
            # frame: arriving by held `up` is what settled the viewport one row
            # down, hiding the `Model` header that names the highlighted row,
            # and the clamp is what made users dwell there (UX round 1, U1).
            view.action_jump(1)
            await pilot.pause()
            for _ in range(80):
                view.action_move(-1)
            await pilot.pause()
            app.save_screenshot(out)
        else:
            app.save_screenshot(out)

        print(geometry if geometry is not None else _geometry(app, view, state, size))


def _geometry(app: OperatorApp, view: SettingsView, state: str, size: tuple[int, int]) -> str:
    """The numbers behind the pixels (AGENTS.md step 4).

    A scrollbar on the SCREEN is always a bug on this app — the body scrolls,
    the dock is docked — and it costs two cells of width silently. The dock and
    view heights are here because a mode that fails to take the screen shows up
    as arithmetic (view + dock < screen) before it shows up as a pixel anyone
    notices.

    The HORIZONTAL numbers are here because the collision is not vertical at
    every size. At 100x30 the boot composition reserves no rows at all, so the
    page gets the same height on both sides and only the card differs: clamped
    to 73 cells at column 12 rather than spanning 96 from column 1. A geometry
    line reporting rows alone made that pair look like it showed nothing
    (design round 1, D1). `boot-card` IS the clamp; `#input-shell`'s width and x
    are what it does. `dock.outer` is reported beside `dock.height` for the same
    reason — the composition's reserve lives in the dock's padding, so the inner
    height alone is identical on both sides.
    """
    screen = app.screen
    dock = app.query_one("#input-dock")
    shell = app.query_one("#input-shell")
    return (
        f"state={state} size={size} "
        f"screen.size={tuple(screen.size)} "
        f"screen.virtual_size={tuple(screen.virtual_size)} "
        f"screen.show_vertical_scrollbar={screen.show_vertical_scrollbar} "
        f"boot={screen.has_class('boot')} "
        f"boot-card={screen.has_class('boot-card')} "
        f"dock.display={dock.display} dock.height={dock.size.height} "
        f"dock.outer={dock.outer_size.height} "
        f"shell.width={shell.size.width} shell.x={shell.region.x} "
        f"view.height={view.size.height} "
        f"view.width={view.size.width} view.x={view.region.x} "
        f"body.size={tuple(view._body.size)} "
        f"body.virtual_size={tuple(view._body.virtual_size)} "
        f"body.show_vertical_scrollbar={view._body.show_vertical_scrollbar} "
        f"pane.size={tuple(view._pane_view.size)} "
        f"rows={len(view._rows)} "
        f"hints={view.rendered_hints()!r}"
    )


def _scroll_to_show_group(view: SettingsView) -> None:
    """Put the highlighted row AND the rows it owns inside the viewport.

    ``_scroll_to_selection`` stops as soon as the CURSOR is visible, which for
    a row near the bottom edge leaves its group below the fold. The cascade's
    frames need both, so this scrolls until the last row of the group fits and
    then keeps the owning row on screen.
    """
    last = view._selected
    for index in range(view._selected + 1, len(view._rows)):
        if view._rows[index].kind not in ("empty", "chain", "hop", "hop_add", "chain_add"):
            break
        last = index
    height = view._body.size.height
    view._body.scroll_to(y=max(0, min(view._selected - 1, last - height + 2)), animate=False)


def _require(key: str) -> Setting:
    """Resolve a shipped setting or fail loudly.

    A capture that silently skipped a missing key would produce a frame of the
    wrong state and caption it as the right one.
    """
    setting = settings_io.resolve_key(key)
    if setting is None:
        raise SystemExit(f"no setting {key}")
    return setting


def _corrupt_cascade() -> None:
    """Store the wreckage #440 left in a real user's ``config.yml``.

    Written through ``settings_io``'s own writer rather than into the YAML by
    hand, so the captured frame shows a state the page itself can produce: the
    pre-fix editor seeded its buffer with ``str(mapping)`` and committing that
    repr stored it as a STRING, which is exactly what this writes.
    """
    setting = settings_io.resolve_key("retry.fallbackChains")
    if setting is None:  # the row is in the shipped registry
        raise SystemExit("no cascade setting")
    settings_io.write_setting(
        ConfigManager(Path(_SCRATCH)),
        setting,
        "{'default': ['anthropic/claude-opus-5', 'openrouter/deepseek']}x",
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
