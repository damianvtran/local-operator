"""R1: a /settings PAGE write must move THIS pane's own approval gate.

Drives the real OperatorApp and the real SettingsView._write path, then asks
the real `request_tool_approval` gate whether a command tool would run.
"""
import asyncio, sys
sys.path.insert(0, "/tmp/lop-live-settings")
import os, tempfile, pathlib

root = pathlib.Path(tempfile.mkdtemp(prefix="ev-pagewrite-"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)

from local_operator.config import ConfigManager
from local_operator import settings_io
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory

async def main():
    ConfigManager(root).set_config_value("hosting", "")
    setting = settings_io.resolve_key("tool_approval_mode")
    settings_io._store(ConfigManager(root), setting.path, "auto")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            await pilot.pause()
            if app._session is not None and app._unsubscribe_config_watch is not None: break
        print(f"boot: disk=auto  app gate = {'auto' if app._approve_all else 'ask'}")

        gate = await app.request_tool_approval("bash", "rm -rf /tmp/x")
        print(f"  a command tool under auto -> approved without a prompt? {gate}")

        print("\nOPERATOR WRITES `ask` THROUGH THE /settings PAGE (settings_io.write_setting,")
        print("the exact call SettingsView._write makes; notifies the watcher as source=local)")
        settings_io.write_setting(ConfigManager(root), setting, "ask")
        await pilot.pause()

        print(f"  disk      = {ConfigManager(root).get_config_value('tool_approval_mode')!r}")
        print(f"  app gate  = {'auto' if app._approve_all else 'ask'}   <-- MUST be ask")
        print(f"  band      = {'auto' if app._status._approvals_auto else 'ask'}")
        notices = [b.text() or '' for b in app.query(__import__('local_operator.tui.widgets.transcript', fromlist=['NoticeBlock']).NoticeBlock)]
        echo = [n for n in notices if 'config.yml changed' in n]
        print(f"  re-announced? {bool(echo)}  (must be False: the page is its own receipt)")

        # THE REPRODUCTION: does a command tool now actually prompt?
        parked = asyncio.ensure_future(app.request_tool_approval("bash", "rm -rf /tmp/x"))
        for _ in range(60):
            await pilot.pause()
            if app._approval is not None: break
        print(f"  a command tool now -> {'PROMPTS (card mounted)' if app._approval else 'AUTO-APPROVED — R1 STILL BROKEN'}")
        if app._approval:
            app._approval.resolve(False, answer='n')
        print(f"  denied -> tool ran? {await parked}")

        print("\nAnd the page can loosen its OWN pane back (a choice made HERE):")
        settings_io.write_setting(ConfigManager(root), setting, "auto")
        await pilot.pause()
        print(f"  app gate = {'auto' if app._approve_all else 'ask'}")

asyncio.run(main())
