"""Render the `/mcp` argument picker over a mixed-source MCP config.

  env -u NO_COLOR TERM=xterm-256color .venv/bin/python .evidence/shot.py <outdir>

Uses the REAL OperatorApp so local_operator.tcss applies (the lightweight test
hosts declare no CSS_PATH and would not show a stylesheet change at all), and a
TEMP HOME so the operator's own config is untouched. The config deliberately
mixes an owned stdio server, an owned non-OAuth http server, an owned OAuth
server and a Claude-imported one: the whole point of the new `remove` rows is
the source column that tells them apart.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/lop-mcp")

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

_home = Path(tempfile.mkdtemp(prefix="mcp-shot-home-"))
_cwd = Path(tempfile.mkdtemp(prefix="mcp-shot-cwd-"))
(_home / ".local-operator").mkdir(parents=True)
(_home / ".local-operator" / "mcp.json").write_text(
    json.dumps(
        {
            "mcpServers": {
                "filesystem": {"type": "stdio", "command": "npx", "args": ["-y", "fs-mcp"]},
                "grafana": {"type": "http", "url": "https://grafana.example/mcp"},
                "linear": {
                    "type": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "auth": {"type": "oauth"},
                },
            }
        },
        indent=2,
    )
)
(_home / ".claude.json").write_text(
    json.dumps({"mcpServers": {"notion": {"type": "http", "url": "https://mcp.notion.com/mcp"}}})
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

LINES = ["/mcp ", "/mcp remove ", "/mcp login "]


async def main() -> None:
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _sources = load_all_mcp_configs(os.getcwd())
    manager = FakeMcpManager(list(configs), ["linear", "grafana"])
    manager._configs = dict(configs)
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 26)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)
        for line in LINES:
            _set_editor_line(editor, line)
            for _ in range(8):
                await pilot.pause()
            rows = [(n, c.detail, c.alert) for n, c in editor.picker.suggestions()]
            slug = line.strip().replace("/", "").replace(" ", "-")
            print(f"{line!r} -> {rows}")
            app.save_screenshot(str(OUT / f"{slug}.svg"))


asyncio.run(main())
