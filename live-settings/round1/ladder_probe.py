"""What the shed ladder actually picks for each affected row, at both widths."""
import asyncio, sys
sys.path.insert(0, "/tmp/lop-live-settings")
from rich.cells import cell_len
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.settings_view import SettingsView

KEYS = ["hosting", "model_name", "tool_approval_mode", "web_search.enabled", "web_fetch.enabled"]

async def run(W: int) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(W, 34)) as pilot:
        await pilot.pause()
        app._cmd_settings(app._notice)
        for _ in range(40):
            await pilot.pause()
            if app.query(SettingsView): break
        view = app.query_one(SettingsView)
        print(f"--- {W} cols, detail budget = {view._detail_width()} ---")
        for target in KEYS:
            for _ in range(400):
                r = view._rows[view._selected]
                if r.setting is not None and r.setting.key == target: break
                await pilot.press("down")
            await pilot.pause()
            d = view._detail_text.plain
            help_len = cell_len(view._rows[view._selected].setting.help)
            print(f"{target:24} help={help_len:3} painted={cell_len(d):3} "
                  f"key={'YES' if target in d else 'no '} clipped={'YES' if d.endswith('…') else 'no'}")

for w in (80, 100, 120):
    asyncio.run(run(w))
