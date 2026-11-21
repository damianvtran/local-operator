"""Durable ask answers must not depend on an owner's display snapshot fitting.

Regression for a live 0.108.3 owner: its authenticated welcome succeeded but
its 1,122,901-byte canonical sync exceeded the 1 MiB line limit. The desktop
refused every answer before it reached the still-working ask queue. Exercise
real HTTP, a real runtime TCP socket, and the real durable queue; only model
output is scripted. Never use an operator session as the fixture.
"""

import asyncio
import json
import secrets

import pytest

from local_operator.asks import store
from local_operator.server.utils.desktop_sessions import DesktopSessionBridge
from local_operator.session.runtime import registry
from tests.e2e.harness import ScriptedStream, text_turn
from tests.e2e.test_desktop_ask_images import _questions, _running_app, _write_config

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_queued_answer_survives_an_oversized_display_sync(
    headless_tui_env, workspace, monkeypatch
):
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    _write_config(root)
    stream = ScriptedStream([text_turn("Acknowledged.")])
    async with _running_app(root, workspace, stream, token) as (client, target, session, sid):
        queue = session.ask_queue()
        ask_id = queue.enqueue(_questions(), None)["details"]["ask_id"]
        # Lifetime child accounting is not display history and cannot be fixed
        # by negotiating a bounded transcript page. Its exact values are kept;
        # this fixture exceeds the existing transport ceiling, not a new cap.
        session._frontend_state_store.mutate(
            child_costs={f"child-{i:05d}": 0.01 for i in range(60_000)}
        )
        record = next(r for r, _ in registry.scan(root, reap=False) if r.session_id == sid)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=2 * 1024 * 1024
        )
        try:
            writer.write(
                json.dumps(
                    {
                        "key": record.control_key,
                        "client": "attach",
                        "locality": "local",
                        "surface": "desktop",
                        "frontend_state": True,
                        "display_window": True,
                    }
                ).encode()
                + b"\n"
            )
            await writer.drain()
            welcome = json.loads(await asyncio.wait_for(reader.readline(), 10))
            assert welcome["op"] in ("welcome", "projection")
            error = json.loads(await asyncio.wait_for(reader.readline(), 10))
            assert error["op"] == "error", error
            assert "over the 1,048,576-byte socket line limit" in error["message"]
            print("NEGATIVE_CONTROL", error["message"])
        finally:
            writer.close()
            await writer.wait_closed()

        # The public HTTP door must retain lookup/retirement guards but skip
        # display acquisition for durable IDs. Two callers still contend at the
        # real queue; no local success is synthesized on a transport failure.
        body = {"ask_id": ask_id, "answers": {"q1": ["A"]}}
        original_acquire = DesktopSessionBridge.acquire

        async def display_dependent_acquire(bridge, *, read=False, control_only=False):
            # Same pre-fix dependency: require canonical sync before /answers.
            return await original_acquire(bridge, read=read)

        with monkeypatch.context() as baseline:
            baseline.setattr(DesktopSessionBridge, "acquire", display_dependent_acquire)
            refused = await client.post(target + "/answers", json=body)
            assert refused.status_code == 503, refused.text
            assert not any(
                e.get("kind") == "answered" for e in store.read_events(root / "sessions" / sid)
            )
            print("HTTP_BASELINE", refused.status_code, refused.json())

        responses = await asyncio.gather(
            *[
                client.post(target + "/answers", json=body),
                client.post(target + "/answers", json=body),
            ]
        )
        assert sorted(r.status_code for r in responses) == [200, 409], [r.text for r in responses]
        print("HTTP_COMPETING", [(r.status_code, r.json()) for r in responses])
        events = store.read_events(root / "sessions" / sid)
        settled = [e for e in events if e.get("kind") == "answered" and e.get("ask_id") == ask_id]
        assert len(settled) == 1, events
        print("DURABLE_SETTLEMENT", settled[0]["kind"], settled[0].get("answers"))
        invalid = await client.post(
            target + "/answers", json={"ask_id": "a-missing", "answers": {"q1": ["A"]}}
        )
        assert invalid.status_code == 409, invalid.text
        print("HTTP_INVALID", invalid.status_code, invalid.json())
