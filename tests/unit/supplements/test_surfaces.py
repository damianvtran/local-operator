"""Capability advertisement, `/v1/capabilities`, and the 404 route stubs (lane C0)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.server.features import feature_flags
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.supplements.contract import (
    SUPPLEMENTS_CAPABILITY,
    SUPPLEMENTS_READ_OP,
)
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn

DIGEST = "04c2d29b140086cc637ad19ef34012c5"


# --- GET /v1/capabilities ---------------------------------------------------------------


def test_the_static_feature_flag_is_published() -> None:
    assert feature_flags()["supplements"] == 1


# --- the owner record's capability list ----------------------------------------------------


class _WithoutReadOp(ServingSessionHandle):
    """The C0 case a real handle can no longer be: one that PREDATES the read op.

    Lane C1b implements ``supplements_for`` on the production handle, so "an owner whose
    handle has no read op" has to be built. A property that raises ``AttributeError`` is the
    one spelling that makes ``hasattr`` answer False -- the exact question the owner record
    asks -- without changing the class under test or patching the server.
    """

    @property
    def supplements_for(self):  # type: ignore[override]
        raise AttributeError(SUPPLEMENTS_READ_OP)


async def _server(tmp_path: Path, *, with_read_op: bool) -> RuntimeServer:
    directory = tmp_path / "sessions" / "supplements-capability"
    await seed_transcript(directory, [Message.user("row")])
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    holder = ServingSessionHandle if with_read_op else _WithoutReadOp
    handle = holder(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    if with_read_op:
        # what the engine lane (C1) adds for real; here only its PRESENCE is under test
        setattr(handle, SUPPLEMENTS_READ_OP, lambda anchors: {})
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    return server


@pytest.mark.asyncio
async def test_an_owner_whose_handle_has_no_read_op_does_not_advertise(tmp_path) -> None:
    """Advertising what the handle cannot honour is worse than omitting it."""
    server = await _server(tmp_path, with_read_op=False)
    try:
        assert SUPPLEMENTS_CAPABILITY not in server._record.capabilities
        assert "display-history-window-v1" in server._record.capabilities
    finally:
        server.close()


@pytest.mark.asyncio
async def test_an_owner_whose_handle_implements_the_read_op_advertises(tmp_path) -> None:
    server = await _server(tmp_path, with_read_op=True)
    try:
        assert SUPPLEMENTS_CAPABILITY in server._record.capabilities
    finally:
        server.close()


async def _attach(server: RuntimeServer, **declare: Any) -> dict[str, Any]:
    """Attach speaking the wire protocol directly (so an old viewer can be reproduced)."""
    record = server._record
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    auth = {"key": record.control_key, "client": "attach", "locality": "local", **declare}
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    try:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            assert line, "the owner closed before the welcome"
            frame = json.loads(line)
            if frame.get("op") in ("projection", "frontend_sync"):
                break
    finally:
        writer.close()
    # the connection the server built for us is the only one
    return {"conn": next(iter(server._clients.values())) if hasattr(server, "_clients") else None}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_has_op", "viewer_declares", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        # agent review R9: only the JSON boolean declares; truthy strings fail CLOSED
        (True, "false", False),
        (True, "1", False),
        (True, 1, False),
    ],
)
async def test_the_attach_gate_is_both_halves(
    tmp_path, owner_has_op: bool, viewer_declares: object, expected: bool
) -> None:
    server = await _server(tmp_path, with_read_op=owner_has_op)
    try:
        record = server._record
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        auth: dict[str, Any] = {
            "key": record.control_key,
            "client": "attach",
            "locality": "local",
            "frontend_state": True,
            "display_window": True,
        }
        if viewer_declares is not False:
            auth["supplements"] = viewer_declares
        writer.write(json.dumps(auth).encode() + b"\n")
        await writer.drain()
        try:
            for _ in range(50):
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                if json.loads(line).get("op") == "frontend_sync":
                    break
            connections = [
                c for c in vars(server).values() if isinstance(c, dict) for c in c.values()
            ]
            flags = [getattr(c, "supplements") for c in connections if hasattr(c, "supplements")]
            assert flags == [expected], flags
        finally:
            writer.close()
    finally:
        server.close()


# --- the route stubs ---------------------------------------------------------------------


def _relay(password: str = "pw") -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password=password)), follow_redirects=False)
    assert client.post("/login", data={"password": password}).status_code in (200, 303)
    return client


@pytest.mark.parametrize(
    "path",
    [
        f"/api/sessions/s1/supplements/{DIGEST}/document",
        "/api/sessions/s1/supplements/3f9c1a7e5b20/file?i=0",
    ],
)
def test_the_relay_stubs_answer_404_after_auth(path: str) -> None:
    assert _relay().get(path).status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        f"/api/sessions/s1/supplements/{DIGEST}/document",
        "/api/sessions/s1/supplements/3f9c1a7e5b20/file?i=0",
    ],
)
def test_the_relay_stubs_are_behind_the_gate(path: str) -> None:
    """Unauthenticated callers get the 401, learning nothing about the route's existence."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw")), follow_redirects=False)
    response = client.get(path)
    assert response.status_code == 401
    assert response.json()["error"] == "authentication required"
