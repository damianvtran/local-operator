"""M2 repro on the REAL key path: does a fuzzy `/mcp reauth` Enter fire?

Same shape as gate_repro.txt, aimed at the verb the audit found ungated. A
subsequence query that spells nothing (`lnr` -> `linear`) must FILL the row,
not run it -- running it calls `_mcp_logout` and forgets the stored grant of a
server the user never named. Run on head + PR #378 (the state this ships into).
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/tmp/lop-mcp-merged")

_home = Path(tempfile.mkdtemp(prefix="reauth-home-"))
_cwd = Path(tempfile.mkdtemp(prefix="reauth-cwd-"))
(_home / ".local-operator").mkdir(parents=True)
(_home / ".local-operator" / "mcp.json").write_text(
    json.dumps(
        {
            "mcpServers": {
                "linear": {
                    "type": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "auth": {"type": "oauth"},
                }
            }
        }
    )
)
os.environ["HOME"] = str(_home)
os.chdir(_cwd)

from local_operator.session.mcp_status import McpStartupOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeMcpManager,
    McpSession,
    _factory,
    _set_editor_line,
)


async def main() -> None:
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _ = load_all_mcp_configs(os.getcwd())
    manager = FakeMcpManager(list(configs), ["linear"])
    manager._configs = dict(configs)
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))

    forgotten: list[str] = []
    FIRED = False

    def _fake_logout(name, cwd, store=None):
        # Stand in for the real auth.db deletion so the repro observes the
        # credential removal without touching a real store.
        forgotten.append(name)
        return None

    async with app.run_test(size=(100, 26)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)

        with (
            patch(
                "local_operator.mcp.auth.mcp_logged_out_servers",
                return_value={"https://mcp.linear.app/mcp"},
            ),
            patch("local_operator.mcp.auth.mcp_logout_server", _fake_logout),
        ):
            _set_editor_line(editor, "/mcp reauth lnr")
            for _ in range(8):
                await pilot.pause()
            print("typed buffer            : '/mcp reauth lnr'")
            print("picker rows             :", [n for n, _ in editor.picker.suggestions()])
            print("editor says destructive :", editor._argument_is_destructive())
            await pilot.press("enter")
            for _ in range(10):
                await pilot.pause()
            print("buffer AFTER enter      :", repr(editor.text))
            print("credentials forgotten   :", forgotten)
            from tests.unit.tui.test_app_pilot import _transcript_text

            print("transcript tail         :", _transcript_text(app)[-300:].replace("\n", " | "))

    fired = forgotten or FIRED
    if fired:
        print("\nRESULT: HAZARD — Enter FIRED and forgot the grant for a name never spelled.")
    else:
        print("\nRESULT: SAFE — the row FILLED; no credential was touched.")
        print("        A second, deliberate Enter is required to run it.")


asyncio.run(main())
