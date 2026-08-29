"""M1/M3/N1/N2 verification on the REAL /mcp path against a temp HOME."""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/lop-mcp")

_home = Path(tempfile.mkdtemp(prefix="m1-home-"))
_cwd = _home / "proj"
_cwd.mkdir(parents=True)
(_home / ".local-operator").mkdir(parents=True)
GLOBAL = _home / ".local-operator" / "mcp.json"
GLOBAL.write_text(
    json.dumps({"mcpServers": {"owned": {"type": "http", "url": "https://owned.example/mcp"}}})
)
# Foreign, LOWER priority than the global file we write -> our entry would shadow it.
(_home / ".claude.json").write_text(
    json.dumps({"mcpServers": {"notion": {"type": "http", "url": "https://mcp.notion.com/mcp"}}})
)
# Project .mcp.json, HIGHER priority -> our write would be invisible.
(_cwd / ".mcp.json").write_text(
    json.dumps({"mcpServers": {"proj": {"type": "http", "url": "https://proj.example/mcp"}}})
)
os.environ["HOME"] = str(_home)
os.chdir(_cwd)

from local_operator.session.mcp_status import McpStartupOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeMcpManager,
    McpSession,
    _factory,
    _transcript_text,
    _type_command,
)


def effective(name):
    from local_operator.mcp.config import load_all_mcp_configs

    cfgs, srcs = load_all_mcp_configs(os.getcwd())
    c = cfgs.get(name)
    return (
        getattr(c, "url", None),
        srcs.get(name, "").replace(str(_home), "~").replace(str(_cwd), "<cwd>"),
    )


async def main():
    from local_operator.mcp.config import load_all_mcp_configs

    cfgs, _ = load_all_mcp_configs(os.getcwd())
    mgr = FakeMcpManager(list(cfgs), [])
    mgr._configs = dict(cfgs)
    app = OperatorApp(lambda: _factory(McpSession(manager=mgr, startup=McpStartupOutcome())))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        for label, line, watch in (
            (
                "M1 shadow (foreign, lower priority)",
                "mcp add notion https://evil.example/mcp",
                "notion",
            ),
            (
                "M1 no-effect (project, higher priority)",
                "mcp add proj https://mine.example/mcp",
                "proj",
            ),
            (
                "M1 control: owned name still updatable?",
                "mcp add owned https://x.example/mcp",
                "owned",
            ),
            ("M3 hint accuracy", "mcp add gw https://gw.example/mcp", "gw"),
            ("N1 list extra tokens", "mcp list junk", None),
            ("N2 remove extra tokens", "mcp remove owned extra", "owned"),
        ):
            before_url = effective(watch) if watch else None
            n = len(_transcript_text(app))
            await _type_command(pilot, app, line)
            for _ in range(10):
                await pilot.pause()
            print(f"\n=== {label}\n$ /{line}")
            print(_transcript_text(app)[n:].strip())
            if watch:
                print(f"   effective {watch}: before={before_url} after={effective(watch)}")


asyncio.run(main())
