"""Rendered evidence for the operator status command (``display.composer.command``).

One running ``OperatorApp`` produces every frame, so frames differ only by the
configuration under test:

    01-default              no command configured (the shipped band)
    02-command              a command printing an account-quota style reading
    03-command-narrow       the same at 80 columns: the ladder sheds the title first
    04-command-removed      command cleared live: the segment is gone again

The command is a fixed ``echo`` so two runs produce the same frame; the
payload path is covered by the unit tests. Frame 02 is reached by a write
BELOW the notify hook plus a watcher poll — the shape another process's
``lop config edit`` lands as — so it also shows the edit applied without a
restart.

    LOP_REPO=<tree> env -u NO_COLOR TERM=xterm-256color \\
        .venv/bin/python scripts/status_command_shot.py OUTDIR

On a tree without the key (a base before-frame) only ``01-default`` is
captured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("LOP_REPO") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(REPO))

import scripts.probe_isolation  # noqa: E402,F401  -- MUST be the first import

for _key in tuple(os.environ):
    if (
        _key.startswith("LOP_MOBILE_CHILD_")
        or _key.startswith("LOP_RUNTIME_")
        or _key == "LOP_MODEL_SELECTION_OVERRIDE"
    ):
        os.environ.pop(_key)

import asyncio  # noqa: E402

from rich.cells import cell_len  # noqa: E402
from rich.text import Text  # noqa: E402
from textual.widgets import Static  # noqa: E402

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.config_watch import process_watcher  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CONFIG_DIR = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])  # from probe_isolation
KEY = "display.composer.command"
COMMAND = "echo 'acct 5h 42% · 7d 18%'"
NEEDLE = "acct 5h 42%"


def _seed(app: OperatorApp) -> None:
    app._append_block(UserBlock("Can the band show my account usage all the time?"))
    answer = AssistantBlock()
    answer.update_text(
        "Set display.composer.command to any script that prints one line; the band "
        "runs it on an interval and shows its output beside the session figures."
    )
    answer.finalize_text()
    app._append_block(answer)
    status = app._status
    assert status is not None
    status.update(
        model_label="anthropic/claude-opus-5-5",
        effort="medium",
        cwd="/Users/tester/code/project",
        context_tokens=184_000,
        context_window=1_000_000,
        last_rate="96 tok/s",
        cost="$3.18",
        conversation_name="Band status command",
        streaming=False,
        failed=False,
        approvals_auto=False,
        approvals_always=False,
    )
    status.seed_duration(active_seconds=1290.0, activity_started_at=None)


def _write_elsewhere(value: str) -> None:
    setting = settings_io.BY_KEY[KEY]
    settings_io._store(ConfigManager(CONFIG_DIR), setting.path, value)
    process_watcher(CONFIG_DIR).poll_now()


def _report(app: OperatorApp, label: str) -> None:
    content = app.query_one("#status-band", Static).content
    painted = content.plain if isinstance(content, Text) else str(content)
    print(f"[{label}] painted({cell_len(painted)}): {painted!r}")


async def _until(app: OperatorApp, pilot, present: bool) -> None:
    for _ in range(400):
        await pilot.pause()
        content = app.query_one("#status-band", Static).content
        painted = content.plain if isinstance(content, Text) else ""
        if (NEEDLE in painted) is present:
            return
    raise AssertionError(f"band never {'showed' if present else 'dropped'} the command output")


async def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 24)) as pilot:
        for _ in range(200):
            if app._session is not None and app._unsubscribe_config_watch is not None:
                break
            await pilot.pause()
        _seed(app)
        await pilot.pause()
        save_capture(app, out / "01-default.svg")
        _report(app, "01-default")
        if KEY not in settings_io.BY_KEY:
            print("note: base tree has no display.composer.command; 01 only.")
            return

        _write_elsewhere(COMMAND)
        await _until(app, pilot, True)
        save_capture(app, out / "02-command.svg")
        _report(app, "02-command")

        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        save_capture(app, out / "03-command-narrow.svg")
        _report(app, "03-command-narrow")

        await pilot.resize_terminal(120, 24)
        _write_elsewhere("")
        await _until(app, pilot, False)
        save_capture(app, out / "04-command-removed.svg")
        _report(app, "04-command-removed")


asyncio.run(main())
