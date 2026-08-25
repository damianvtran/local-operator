"""Production loopback canonical frontend sync and isolation checks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.mobile import registry
from local_operator.mobile.attach_client import AttachClient
from local_operator.mobile.registrant import Registrant
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.remote import RemoteSession
from tests.unit.mobile.test_registrant import FakeHandle


async def _record(root: Path):  # noqa: ANN202
    for _ in range(100):
        rows = registry.scan(root)
        if rows and rows[0][1] == "live":
            return rows[0][0]
        await asyncio.sleep(0.02)
    raise AssertionError("record did not publish")


async def _never():
    raise AssertionError("takeover was not expected")


@pytest.mark.asyncio
async def test_owner_two_followers_share_full_1m_state_and_live_updates(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    first = second = None
    try:
        record = await _record(tmp_path)
        assert FRONTEND_CAPABILITY in record.capabilities
        first, second = await asyncio.gather(
            RemoteSession.connect(record, "s1", config_dir=tmp_path, takeover_factory=_never),
            RemoteSession.connect(record, "s1", config_dir=tmp_path, takeover_factory=_never),
        )
        assert first.frontend_state == second.frontend_state
        assert first.effective_model.context_window == 1_000_000

        handle._frontend.mutate(
            context_tokens=402_000,
            cumulative_parent_cost=2.75,
            active_agent="coder",
            active_team="lopdev",
        )
        for _ in range(100):
            if first.frontend_state.sequence == 1 and second.frontend_state.sequence == 1:
                break
            await asyncio.sleep(0.02)
        assert first.frontend_state == second.frontend_state
        assert first.frontend_state.cumulative_cost == 2.75
        assert first.active_agent == "coder"
        assert first.active_team_name == "lopdev"
    finally:
        if first is not None:
            await first.dispose()
        if second is not None:
            await second.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_daemon_never_receives_frontend_frames(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _record(tmp_path)
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        welcome = json.loads((await asyncio.wait_for(reader.readline(), 3)).decode())
        assert welcome["op"] == "projection"
        handle._frontend.mutate(context_tokens=99)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(reader.readline(), 0.15)
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        registrant.close()


@pytest.mark.asyncio
async def test_wrong_key_wrong_session_and_old_protocol_are_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _record(tmp_path)
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(b'{"key":"wrong","client":"attach","frontend_state":true}\n')
        await writer.drain()
        assert await asyncio.wait_for(reader.readline(), 3) == b""
        writer.close()
        await writer.wait_closed()

        client = AttachClient(lambda state: None, lambda reason: None, frontend_state=True)
        with pytest.raises(ConnectionError, match="another conversation"):
            await client.connect(record, "wrong-session")

        old = record.__class__(**{**record.to_json(), "protocol": 4, "capabilities": []})
        with pytest.raises(ConnectionError, match="lacks tui_state_v1"):
            await RemoteSession.connect(old, "s1", config_dir=tmp_path, takeover_factory=_never)
    finally:
        registrant.close()
