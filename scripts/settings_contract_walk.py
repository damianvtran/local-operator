"""Walk the whole /settings editing contract against the REAL app (#440).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/settings_contract_walk.py

Prints a SHA of config.yml's bytes after every gesture, which is the standard
of proof this page uses (#387 round 1, U1): a rewrite that moves only
`last_modified` is still a write. The claim the walk exists to demonstrate is
that the file does not exist at all until the one gesture that is an accept.

Modelled on `scripts/cascade_repro.py`: it drives the real `OperatorApp`
through its own slash command and key bindings against a scratch
`LOCAL_OPERATOR_CONFIG_DIR`, so nothing here can touch a developer's config and
nothing is asserted about a code path a user cannot reach.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

import yaml  # noqa: E402

# The scratch config dir has to exist BEFORE the app imports anything that
# resolves `config_dir()`, which is why this runs above the local imports —
# the same ordering `scripts/settings_shot.py` records.
SC = tempfile.mkdtemp(prefix="lo-settings-walk-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = SC

from local_operator.tui import theme as tm  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.settings_view import SettingsView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CFG = Path(SC) / "config.yml"


def h():
    return hashlib.sha256(CFG.read_bytes()).hexdigest()[:12] if CFG.exists() else "NO FILE"


def sel(v, k):
    for i, r in enumerate(v._rows):
        if r.kind == "setting" and r.setting and r.setting.key == k:
            v._selected = i
            v._repaint()
            return


async def main():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        v = app.query_one(SettingsView)
        print("1. page opened                       config:", h())

        sel(v, "retry.enabled")
        await pilot.press("enter")
        await pilot.pause()
        print("2. enter on a BOOL                   config:", h(), "| expanded:", v.expanded_key)
        await pilot.press("down")
        await pilot.pause()
        print("3. browsed onto 'off'                config:", h())
        await pilot.press("escape")
        await pilot.pause()
        print("4. esc                               config:", h())

        sel(v, "retry.maxRetries")
        await pilot.press("enter")
        for _ in range(len(v._buffer)):
            await pilot.press("backspace")
        for c in "77":
            await pilot.press(c)
        await pilot.pause()
        print("5. typed 77 into the editor          config:", h(), "| buffer:", v._buffer)
        await pilot.press("down")
        await pilot.pause()
        print("6. ARROW AWAY (used to COMMIT)       config:", h(), "| notice:", v.notice_text)

        sel(v, "tui.theme")
        await pilot.press("enter")
        await pilot.pause()
        for _ in range(3):
            await pilot.press("down")
            await pilot.pause()
            if tm.current_theme() != "dark":
                break
        print(
            "7. browsed themes                    config:", h(), "| live theme:", tm.current_theme()
        )
        await pilot.press("escape")
        await pilot.pause()
        print(
            "8. esc                               config:", h(), "| live theme:", tm.current_theme()
        )

        sel(v, "retry.maxRetries")
        await pilot.press("r")
        await pilot.pause()
        print("9. r on a DEFAULT row                config:", h(), "(used to CREATE the file)")

        # Now an actual accept.
        await pilot.press("enter")
        for _ in range(len(v._buffer)):
            await pilot.press("backspace")
        for c in "7":
            await pilot.press(c)
        await pilot.press("enter")
        await pilot.pause()

        print(
            "10. enter ACCEPTS                    config:",
            h(),
            "| stored:",
            yaml.safe_load(CFG.read_text())["values"]["retry"]["maxRetries"],
        )
        print("11. detail now:", v.render_lines_for_test()[-1].strip()[:80])
        print("12. r offered:", v._reset_hint.display)
        await pilot.press("r")
        await pilot.pause()
        vals = yaml.safe_load(CFG.read_text())["values"].get("retry", {})
        print("13. r RESTORES                       maxRetries in file:", "maxRetries" in vals)


asyncio.run(main())
