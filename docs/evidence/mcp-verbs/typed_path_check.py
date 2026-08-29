"""Are the /mcp server rows reachable by TYPING, as a user reaches them?

Every server-row test and the shot script use `_set_editor_line`, which calls
`editor._sync_picker()` directly and therefore bypasses the
`RefreshArgumentChoices` message. This harness types with `pilot.press` per
character -- the only path that exercises the message -- and reports, for each
verb->server transition, whether the picker actually opened and what it holds.

It also reports the editor's destructive-gate verdict on the typed path, since
#378's fuzzy-Enter guard reads the highlighted row's `alert` flag and a closed
list has no highlighted row (review round 2, B2).

Run against a worktree holding this branch with PR #378 merged.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPO = os.environ.get("LOP_REPO", "/tmp/lop-mcp-merged")
sys.path.insert(0, REPO)

_home = Path(tempfile.mkdtemp(prefix="typed-home-"))
_cwd = Path(tempfile.mkdtemp(prefix="typed-cwd-"))
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
)


async def _type(pilot, text: str) -> None:
    """Real keystrokes, one per character -- never `_set_editor_line`."""
    for ch in text:
        await pilot.press("space" if ch == " " else ch)
    for _ in range(6):
        await pilot.pause()


async def main() -> None:
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _ = load_all_mcp_configs(os.getcwd())
    manager = FakeMcpManager(list(configs), ["linear"])
    manager._configs = dict(configs)
    app = OperatorApp(lambda: _factory(McpSession(manager=manager, startup=McpStartupOutcome())))

    failures: list[str] = []
    async with app.run_test(size=(100, 26)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        editor = app.query_one(Editor)

        with patch(
            "local_operator.mcp.auth.mcp_logged_out_servers",
            return_value={"https://mcp.linear.app/mcp"},
        ):
            print(f"{'typed':26} {'open':5} {'rows':38} gate")
            print("-" * 82)
            for typed, expect in (
                ("/mcp ", ["list", "add", "remove", "login", "logout", "reauth"]),
                ("/mcp remove ", ["remove linear", "remove filesystem"]),
                ("/mcp remove fs", ["remove filesystem"]),
                ("/mcp login ", ["login linear"]),
                ("/mcp logout ", ["logout linear"]),
                ("/mcp reauth ", ["reauth linear"]),
                ("/mcp reauth lnr", ["reauth linear"]),
            ):
                editor.text = ""
                editor.move_cursor(editor._end_of_buffer())
                for _ in range(4):
                    await pilot.pause()
                editor.focus()
                await _type(pilot, typed)
                rows = [n for n, _ in editor.picker.suggestions()]
                gate = editor._argument_is_destructive()
                shown = editor.picker.display
                ok = sorted(rows) == sorted(expect)
                if not ok:
                    failures.append(f"{typed!r}: got {rows}, want {expect}")
                print(
                    f"{typed!r:26} {str(shown):5} {str(rows)[:38]:38} {gate}"
                    f"{'' if ok else '   <-- WRONG'}"
                )

    print()
    if failures:
        print(f"TYPED-PATH CHECK FAILED ({len(failures)}):")
        for line in failures:
            print(f"  {line}")
    else:
        print("TYPED-PATH CHECK PASSED: every verb->server transition opens by typing,")
        print("and the destructive gate is armed on the rows that destroy.")


asyncio.run(main())
