"""The TUI auto-registers with the mobile control plane when a session is
adopted — this pins that contract: record published, control socket answers,
slash commands land through the app's own dispatch, and unmount unpublishes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.mark.asyncio
async def test_tui_auto_registers_and_answers_control() -> None:
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(50):
            if app._mobile_registrant is not None:
                break
            await pilot.pause(0.1)
        assert app._mobile_registrant is not None, "mobile registrant never started"

        from local_operator.mobile import registry

        records = registry.scan()
        assert records, "no discovery record published"
        record, state = records[0]
        assert state == "live"
        assert record.kind == "tui"

        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5)
        frame = json.loads(line)
        assert frame["op"] == "projection"
        assert frame["data"]["model_label"] == record.model_label

        writer.write(
            json.dumps({"op": "slash", "req": 1, "command": "goal", "args": "test goal"}).encode()
            + b"\n"
        )
        await writer.drain()
        acked = None
        for _ in range(10):
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            frame = json.loads(line)
            if frame.get("op") == "ack":
                acked = frame
                break
        assert acked is not None and "goal" in acked["detail"]
        writer.close()

    assert not registry.scan(), "record was not unpublished on exit"
