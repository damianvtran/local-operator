"""Reproduce the ``retry.fallbackChains`` data-loss bug (#440).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/cascade_repro.py

``action_activate`` has no ``Kind.CASCADE`` branch, so ``enter`` on the failover
cascade SETTING row falls through to ``_begin_edit``, which seeds a free-text
editor with ``str(dict)`` — a Python ``repr``. Committing it writes that repr
into ``retry.fallbackChains`` as a STRING, ``read_chains`` then returns ``{}``,
and the user's whole cascade is gone with no way back (``r`` cannot restore it
because the stored value is no longer a mapping).

Drives the REAL :class:`OperatorApp` against a scratch config dir seeded with a
real cascade, and reports the STORED value before and after — the stored value
being the thing that is destroyed, which a "did an editor open" probe would not
show.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = tempfile.mkdtemp(prefix="lo-cascade-repro-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

import yaml  # noqa: E402

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.settings_view import SettingsView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

_CHAINS = {"default": ["anthropic/claude-opus-5", "openrouter/deepseek"]}


def _stored() -> object:
    """The RAW value on disk, not what ``read_chains`` makes of it."""
    config = Path(_SCRATCH) / "config.yml"
    if not config.exists():
        return None
    values = yaml.safe_load(config.read_text()).get("values", {})
    return values.get("retry", {}).get("fallbackChains")


async def main() -> None:
    manager = ConfigManager(Path(_SCRATCH))
    settings_io.write_chains(manager, dict(_CHAINS))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        for index, row in enumerate(view._rows):
            if row.kind == "setting" and row.setting is not None:
                if row.setting.key == "retry.fallbackChains":
                    view._selected = index
                    break
        else:  # pragma: no cover - the row is in the shipped registry
            raise AssertionError("no cascade setting row")

        print(f"cascade row: {view._rows[view._selected].setting.key}")
        print(f"chains BEFORE:            {settings_io.read_chains(view._manager)}")

        # A real `enter`, through the app's own binding, not a direct call.
        await pilot.press("enter")
        await pilot.pause()
        print(f"after real 'enter':       editing? {view._editing is not None} | "
              f"buffer: {view._buffer[:44]!r}")

        # Type one character so the commit is not the no-op "nothing typed"
        # path, then accept it exactly as a user would.
        await pilot.press("x")
        await pilot.press("enter")
        await pilot.pause()

        view._manager.reload()
        print(f"chains AFTER enter-commit: {settings_io.read_chains(view._manager)}")
        print(f"raw yaml: {_stored()!r}")
        print(f"stored value is a mapping: {isinstance(_stored(), dict)}")


asyncio.run(main())
