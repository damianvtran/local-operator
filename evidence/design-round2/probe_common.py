"""Shared driver for design round 2 of PR #891.

Real OperatorApp, real keystrokes, isolated HOME/config, CMUX_* scrubbed and
synthetic IDs. Never assigns `editor.text` — that bypasses the `_on_key`
funnel the whole feature lives in.
"""

from __future__ import annotations

import os
import sys

WT = "/tmp/d891r2/wt"
sys.path.insert(0, WT)

# Scrub every inherited CMUX_* BEFORE anything imports app modules: an
# inherited CMUX_WORKSPACE_ID has let headless tests rename real workspaces.
for _k in [k for k in os.environ if k.startswith("CMUX_")]:
    del os.environ[_k]
os.environ["CMUX_WORKSPACE_ID"] = "synthetic-d891r2-ws"
os.environ["CMUX_SESSION_ID"] = "synthetic-d891r2-sess"

import scripts.probe_isolation  # noqa: F401,E402  (isolates HOME/config on import)

from scripts.visual_capture import save_capture  # noqa: E402

import local_operator.tui.app as app_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

# Assert we are measuring the worktree under review, not the shared checkout
# or the installed uv tool. A probe against the wrong module file is the
# confident-wrong-answer failure mode.
assert app_mod.__file__.startswith(WT), f"WRONG MODULE: {app_mod.__file__}"

SECRET = "hunter2-typed-test"  # synthetic only


def new_app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


async def type_text(pilot, text: str) -> None:
    """Real keystrokes, one per character."""
    for ch in text:
        await pilot.press("space" if ch == " " else ch)
    await pilot.pause()


def editor_of(app):
    from local_operator.tui.widgets.editor import Editor

    return app.query_one(Editor)


def frame_text(app) -> str:
    """The whole screen AS PAINTED — the repo's own compositor idiom.

    `screen.render_line` returns blank strips outside a real paint cycle, which
    silently made every absence reading a false clean. `_compositor.render_strips`
    is what tests/unit/tui/test_app_pilot.py uses and what actually carries text.
    """
    return "\n".join(strip.text.rstrip() for strip in app.screen._compositor.render_strips())


def notice_row_text(app) -> str:
    """The picker's notice row as painted, sliced out of the composited frame."""
    from local_operator.tui.widgets.command_picker import CommandPicker

    picker = app.query_one(CommandPicker)
    if not picker.display or picker.region is None:
        return ""
    region = picker.region
    rows = frame_text(app).splitlines()
    out = []
    for y in range(region.y, min(region.y + region.height, len(rows))):
        out.append(rows[y][region.x : region.x + region.width].rstrip())
    return "\n".join(line for line in out if line.strip())


def assert_instrument(app, must_contain: str, label: str) -> None:
    """Positive control: refuse to report an absence from a dead instrument."""
    frame = frame_text(app)
    rows = len([ln for ln in frame.splitlines() if ln.strip()])
    if must_contain not in frame:
        raise AssertionError(
            f"INSTRUMENT DEAD at {label}: {must_contain!r} absent, "
            f"non-blank rows={rows}, len={len(frame)}"
        )
