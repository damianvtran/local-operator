"""Data-loss repro: does a fuzzy `/mcp remove` row FILL or FIRE on Enter?

`Editor._argument_is_destructive` matches DESTRUCTIVE_COMMANDS against the
COMMAND WORD ("logout"), so "/mcp" is not covered and the new remove rows are
unprotected. This drives the REAL key path against a temp HOME and reports what
happened to the config on disk.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/lop-mcp")

_home = Path(tempfile.mkdtemp(prefix="gate-home-"))
_cwd = Path(tempfile.mkdtemp(prefix="gate-cwd-"))
(_home / ".local-operator").mkdir(parents=True)
CFG = _home / ".local-operator" / "mcp.json"
CFG.write_text(
    json.dumps(
        {
            "mcpServers": {
                "filesystem": {"type": "stdio", "command": "npx"},
                "grafana": {"type": "http", "url": "https://grafana.example/mcp"},
            }
        },
        indent=2,
    )
)
os.environ["HOME"] = str(_home)
os.chdir(_cwd)

from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeMcpManager,
    McpSession,
    _factory,
    _set_editor_line,
)

from local_operator.session.mcp_status import McpStartupOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402


async def main() -> None:
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _ = load_all_mcp_configs(os.getcwd())
    manager = FakeMcpManager(list(configs), [])
    manager._configs = dict(configs)
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 26)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)

        print("servers on disk BEFORE :", sorted(json.loads(CFG.read_text())["mcpServers"]))
        # A fuzzy subsequence query that spells nothing, narrowed to ONE row.
        _set_editor_line(editor, "/mcp remove fsy")
        for _ in range(8):
            await pilot.pause()
        print("typed buffer           : '/mcp remove fsy'")
        print("picker rows            :", [n for n, _ in editor.picker.suggestions()])
        print("editor says destructive:", editor._argument_is_destructive())
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()
        print("buffer AFTER enter     :", repr(editor.text))
        after = sorted(json.loads(CFG.read_text())["mcpServers"])
        print("servers on disk AFTER  :", after)
        if "filesystem" not in after:
            print("\nRESULT: DATA LOSS — 'filesystem' was deleted by the fuzzy query 'fsy'.")
            print("        Enter FIRED the row instead of filling it.")
        else:
            print("\nRESULT: SAFE — the row filled and the config is intact.")


asyncio.run(main())
