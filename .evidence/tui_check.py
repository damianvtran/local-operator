"""Exercise the REAL `/mcp` TUI paths end to end against a temp HOME.

Drives the actual OperatorApp through `_type_command` (the path a user's
keystrokes take) and reads back the transcript plus the on-disk config, so
every line below is a real command with its real output and real side effect.
The operator's own ~/.local-operator/mcp.json is never read or written, and
the synthetic config carries no secrets.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/lop-mcp")

_home = Path(tempfile.mkdtemp(prefix="mcp-tui-home-"))
_cwd = Path(tempfile.mkdtemp(prefix="mcp-tui-cwd-"))
(_home / ".local-operator").mkdir(parents=True)
GLOBAL = _home / ".local-operator" / "mcp.json"
GLOBAL.write_text(
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
# A server local-operator can SEE but must never DELETE.
(_home / ".claude.json").write_text(
    json.dumps({"mcpServers": {"notion": {"type": "http", "url": "https://mcp.notion.com/mcp"}}})
)
os.environ["HOME"] = str(_home)
os.chdir(_cwd)

from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeMcpManager,
    McpSession,
    _factory,
    _transcript_text,
    _type_command,
)

from local_operator.session.mcp_status import McpStartupOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402


def _servers() -> list[str]:
    return sorted(json.loads(GLOBAL.read_text())["mcpServers"])


async def main() -> None:
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _ = load_all_mcp_configs(os.getcwd())
    manager = FakeMcpManager(list(configs), ["linear", "grafana"])
    manager._configs = dict(configs)
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))

    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()

        for line in (
            "mcp list",
            "mcp add demo-stdio npx -y demo-mcp",
            "mcp add demo-http https://demo.example/mcp",
            "mcp add demo-stdio npx",
            "mcp add demo-http https://demo.example/mcp extra",
            "mcp add oops",
            "mcp remove demo-http",
            "mcp remove notion",
            "mcp remove ghost",
            "mcp frobnicate",
            "mcp list",
        ):
            before = len(_transcript_text(app))
            await _type_command(pilot, app, line)
            for _ in range(10):
                await pilot.pause()
            print(f"\n=== /{line}")
            print(_transcript_text(app)[before:].strip())
            print(f"--- global mcp.json now: {_servers()}")

    print("\n=== final ~/.local-operator/mcp.json")
    print(GLOBAL.read_text())


asyncio.run(main())
