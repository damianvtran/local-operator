"""Collapsed TUI bridge card whose resolver answered offline (D8 before/after)."""
import asyncio, sys
sys.path.insert(0, ".")
from scripts.visual_capture import isolate_capture, save_capture
isolate_capture()
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import TeamBridgeBlock

OFFLINE = {"v": 1, "more": 0, "items": [{"message_id": "bcmsg_1", "conversation_id": "bcv_12345678",
           "state": "unavailable", "reason": "offline"}]}

async def loader(*, local_only: bool):
    return OFFLINE

async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._append_block(TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=loader))
        for _ in range(5):
            await pilot.pause()
        b = [w for w in app.query(TeamBridgeBlock)][0]
        print("collapsed:", b._build_content(100).plain.splitlines()[0])
        b.toggle_expanded()
        for _ in range(5): await pilot.pause()
        print("open:     ", b._build_content(100).plain.splitlines()[0])
        save_capture(app, sys.argv[1])
        b.toggle_expanded()
        for _ in range(3): await pilot.pause()
        print("reclosed: ", b._build_content(100).plain.splitlines()[0])

asyncio.run(main())
