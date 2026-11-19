"""Real OperatorApp: open a bridge card while its mount pass is still in flight (m1)."""
import asyncio, sys
sys.path.insert(0, ".")
from scripts.visual_capture import isolate_capture
isolate_capture()
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import TeamBridgeBlock

calls = []
async def loader(*, local_only: bool):
    calls.append(local_only)
    if local_only:
        await asyncio.sleep(0.4)  # mount pass still in flight when the card opens
        return {"v": 1, "more": 0, "items": [{"message_id": "bcmsg_1", "conversation_id": "bcv_12345678",
                "state": "unavailable", "reason": "not_loaded"}]}
    return {"v": 1, "more": 0, "items": [{"message_id": "bcmsg_1", "conversation_id": "bcv_12345678",
            "state": "admitted", "body": "Hello from Ola", "source": "hub"}]}

async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        block = TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=loader)
        app._append_block(block)
        await pilot.pause()
        block.toggle_expanded()
        await pilot.pause()
        print("during:", calls, "| loading row:", "Loading message…" in block._build_content(100).plain)
        await asyncio.sleep(1.0)
        await pilot.pause()
        text = block._build_content(100).plain
        print("after:", calls, "| verified:", block._body_verified, "| body shown:", "Hello from Ola" in text,
              "| still loading:", "Loading message…" in text)

asyncio.run(main())
