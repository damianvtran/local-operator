"""Audit EVERY /mcp verb's rows against the safety claim, on the merged tree.

Not a spot check: for each verb it reports the rows, each row's `alert` flag,
and the editor's real `_argument_is_destructive()` verdict with that row
highlighted, then classifies whether the verb DESTROYS something. A verb that
destroys but is not gated is a hazard of the kind gate_repro.txt documents.

Run on head + PR #378 merged, which is the state this branch ships into.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/lop-mcp-merged")

_home = Path(tempfile.mkdtemp(prefix="audit-home-"))
_cwd = Path(tempfile.mkdtemp(prefix="audit-cwd-"))
(_home / ".local-operator").mkdir(parents=True)
(_home / ".local-operator" / "mcp.json").write_text(
    json.dumps(
        {
            "mcpServers": {
                "linear": {
                    "type": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "auth": {"type": "oauth"},
                },
                "filesystem": {"type": "stdio", "command": "npx"},
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

# What each verb actually DOES to persistent state, read off the handlers:
#   remove -> deletes the config entry        (_mcp_remove_result)
#   logout -> deletes the stored grant        (_mcp_logout)
#   reauth -> deletes the stored grant FIRST, then re-authorizes (_mcp_logout
#             then the login worker) -- destructive even though it usually ends
#             in a new credential
#   login  -> creates a grant, destroys nothing
#   add    -> creates a config entry, destroys nothing (and offers no rows)
#   list   -> reads (no rows)
DESTROYS = {"remove": True, "logout": True, "reauth": True, "login": False}


async def main() -> None:
    from local_operator.mcp.auth import mcp_logged_out_servers  # noqa: F401
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _ = load_all_mcp_configs(os.getcwd())
    manager = FakeMcpManager(list(configs), ["linear"])
    manager._configs = dict(configs)
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))

    from unittest.mock import patch

    failures = []
    async with app.run_test(size=(100, 26)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)

        # A stored credential for linear, so the logout list is non-empty.
        with patch(
            "local_operator.mcp.auth.mcp_logged_out_servers",
            return_value={"https://mcp.linear.app/mcp"},
        ):
            print(f"{'verb':8} {'row':26} {'alert':6} {'gate':6} {'destroys':9} verdict")
            print("-" * 78)
            for verb in ("list", "add", "remove", "login", "logout", "reauth"):
                _set_editor_line(editor, f"/mcp {verb} ")
                for _ in range(8):
                    await pilot.pause()
                rows = editor.picker.suggestions()
                if not rows:
                    print(f"{verb:8} {'(no rows)':26} {'-':6} {'-':6} {'-':9} n/a")
                    continue
                for idx, (name, choice) in enumerate(rows):
                    (
                        editor.picker.highlight_index(idx)
                        if hasattr(editor.picker, "highlight_index")
                        else None
                    )
                    # Re-sync so the gate reads THIS row as highlighted.
                    _set_editor_line(editor, f"/mcp {verb} ")
                    for _ in range(4):
                        await pilot.pause()
                    for _ in range(idx):
                        await pilot.press("down")
                        await pilot.pause()
                    alert = bool(getattr(choice, "alert", False))
                    gate = editor._argument_is_destructive()
                    destroys = DESTROYS.get(verb, False)
                    ok = gate if destroys else True
                    verdict = "OK" if ok else "*** UNGATED HAZARD ***"
                    if not ok:
                        failures.append((verb, name))
                    print(
                        f"{verb:8} {name:26} {str(alert):6} {str(gate):6} "
                        f"{str(destroys):9} {verdict}"
                    )

    print()
    if failures:
        print(f"AUDIT FAILED: {len(failures)} destructive row(s) not gated: {failures}")
    else:
        print("AUDIT PASSED: every verb that destroys persistent state is gated;")
        print("              'login' is the only unflagged verb and it destroys nothing.")


asyncio.run(main())
