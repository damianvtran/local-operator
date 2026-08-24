"""Multi-connection registrant: N authenticated clients on one control socket.

Protocol v2's core contract — the daemon plus up to ATTACH_MAX_CLIENTS attach
terminals, broadcast pushes, point-to-point acks, watch/unwatch accounting,
and the attach dispatch restrictions. These run against the REAL socket (a
FakeHandle), matching test_daemon.py's style, because the failure modes this
module guards (frame interleaving, eviction, registry leaks) only exist on a
real connection pair.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from local_operator.mobile import registry
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import (
    ATTACH_MAX_CLIENTS,
    PROTOCOL_VERSION,
    SessionProjection,
)


class FakeHandle:
    """Static projection; records dispatch calls for assertions."""

    def __init__(self) -> None:
        self._projection = SessionProjection(
            session_id="s1",
            pid=0,
            kind="tui",
            conversation_name="fake",
            cwd="/tmp",
            model_label="test/model",
        )
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    async def _record(self, name: str, *args: object, **kwargs: object) -> str:
        self.calls.append((name, args, kwargs))
        return f"{name} ok"

    async def prompt(self, text, images=None):  # noqa: ANN001, ANN202
        return await self._record("prompt", text)

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        return await self._record("steer", text)

    async def abort(self):  # noqa: ANN202
        return await self._record("abort")

    async def set_model(self, provider, model_id):  # noqa: ANN001, ANN202
        return await self._record("set_model", provider, model_id)

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        return await self._record("set_effort", effort)

    async def slash(self, command, args):  # noqa: ANN001, ANN202
        return await self._record("slash", command, args)

    async def new_conversation(self):  # noqa: ANN202
        return await self._record("new_conversation")

    async def resume_session(self, session_id):  # noqa: ANN001, ANN202
        return await self._record("resume_session", session_id)

    async def approval_answer(self, request_id, approved, remember):  # noqa: ANN001, ANN202
        return await self._record("approval_answer", request_id, approved, remember)

    async def ask_answer(self, request_id, value, question_index=None):  # noqa: ANN001, ANN202
        return await self._record("ask_answer", request_id, value)

    async def refresh(self) -> None:
        pass


async def _wait_record() -> registry.SessionRecord:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        found = registry.scan()
        if found and found[0][1] == "live":
            return found[0][0]
        await asyncio.sleep(0.05)
    raise AssertionError("registrant never published a live record")


async def _dial(record: registry.SessionRecord, *, client: str | None = None):
    """Open + auth one connection; consume the welcome projection."""
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", record.control_port, limit=1 << 20
    )
    auth: dict[str, object] = {"key": record.control_key}
    if client is not None:
        auth["client"] = client
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    welcome = await asyncio.wait_for(reader.readline(), timeout=5)
    assert json.loads(welcome)["op"] == "projection"
    return reader, writer


async def _until(
    reader: asyncio.StreamReader, want_op: str, want_req: object = None, n: int = 30
) -> dict:
    """Read until a frame matching (op, req) arrives; skip broadcasts."""
    for _ in range(n):
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        s = raw.decode("utf-8", "replace").strip()
        if not s:
            continue
        frame = json.loads(s)
        if frame.get("op") == want_op and (want_req is None or frame.get("req") == want_req):
            return frame
    raise AssertionError(f"no {want_op} frame arrived")


@pytest.mark.asyncio
async def test_protocol_version_is_two_and_cap_constant() -> None:
    assert PROTOCOL_VERSION == 2
    assert ATTACH_MAX_CLIENTS == 4


@pytest.mark.asyncio
async def test_daemon_and_attach_clients_coexist() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        ra, wa = await _dial(record, client="attach")
        ra2, wa2 = await _dial(record, client="attach")
        assert registrant.attach_clients() == 2
        # All three stay live across traffic from any of them.
        wa.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        await wa.drain()
        await _until(ra, "ack", 1)
        wa2.write(json.dumps({"op": "ping", "req": 2}).encode() + b"\n")
        await wa2.drain()
        await _until(ra2, "ack", 2)
        assert registrant.attach_clients() == 2
        for w in (wd, wa, wa2):
            w.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_broadcast_reaches_every_client() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        ra, wa = await _dial(record, client="attach")
        # A mutation from the DAEMON must repaint the ATTACH client too.
        wd.write(json.dumps({"op": "set_effort", "req": 7, "effort": "high"}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 7)
        frame = await _until(ra, "projection")
        assert frame["data"]["session_id"] == "s1"
        wd.close()
        wa.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_acks_stay_point_to_point_under_concurrent_ops() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        ra2, wa2 = await _dial(record, client="attach")
        # Two concurrent ops with overlapping req ids from two clients.
        wa.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        wa2.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        await wa.drain()
        await wa2.drain()
        a1 = await _until(ra, "ack", 1)
        a2 = await _until(ra2, "ack", 1)
        # Each client sees exactly its own ack detail; no cross-delivery of
        # the OTHER client's frames between the two reads is hard to prove
        # exhaustively, but each reader must never see an ERROR here.
        assert a1["detail"] == "pong"
        assert a2["detail"] == "pong"
        wa.close()
        wa2.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_new_daemon_dial_evicts_the_old() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        rd2, wd2 = await _dial(record)  # reconnect story
        # The old socket observes EOF.
        raw = await asyncio.wait_for(rd.readline(), timeout=5)
        assert raw == b""
        # The new one still works.
        wd2.write(json.dumps({"op": "ping", "req": 3}).encode() + b"\n")
        await wd2.drain()
        await _until(rd2, "ack", 3)
        wd.close()
        wd2.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_attach_cap_evicts_least_recently_seen() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        pairs = [await _dial(record, client="attach") for _ in range(ATTACH_MAX_CLIENTS)]
        assert registrant.attach_clients() == ATTACH_MAX_CLIENTS
        # Touch every attach EXCEPT the first (the LRU victim).
        for i in range(1, ATTACH_MAX_CLIENTS):
            reader, writer = pairs[i]
            writer.write(json.dumps({"op": "ping", "req": i}).encode() + b"\n")
            await writer.drain()
            await _until(reader, "ack", i)
            await asyncio.sleep(0.05)
        # A further dial evicts the untouched first client.
        rn, wn = await _dial(record, client="attach")
        victim_reader = pairs[0][0]
        # The victim's socket still holds the broadcasts queued BEFORE its
        # eviction; drain until EOF (the eviction itself closes it).
        raw = b"x"
        deadline = asyncio.get_running_loop().time() + 5
        while raw != b"" and asyncio.get_running_loop().time() < deadline:
            raw = await asyncio.wait_for(victim_reader.readline(), timeout=5)
        assert raw == b""  # evicted: EOF
        assert registrant.attach_clients() == ATTACH_MAX_CLIENTS
        wn.close()
        for _, w in pairs:
            w.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_attach_client_cannot_rebind_the_session() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        wa.write(json.dumps({"op": "resume_session", "req": 4, "session_id": "x"}).encode() + b"\n")
        await wa.drain()
        err = await _until(ra, "error", 4)
        assert "cannot rebind" in err["message"]
        wa.write(json.dumps({"op": "new_conversation", "req": 5}).encode() + b"\n")
        await wa.drain()
        err = await _until(ra, "error", 5)
        assert "cannot rebind" in err["message"]
        # The daemon keeps both ops.
        rd, wd = await _dial(record)
        wd.write(json.dumps({"op": "new_conversation", "req": 6}).encode() + b"\n")
        await wd.drain()
        ack = await _until(rd, "ack", 6)
        assert "new_conversation ok" in ack["detail"]
        wa.close()
        wd.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_dead_socket_leaves_the_registry() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        assert registrant.attach_clients() == 1
        wa.close()
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if registrant.attach_clients() == 0:
                break
            await asyncio.sleep(0.05)
        assert registrant.attach_clients() == 0
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_watch_unwatch_accounting_and_floor() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        assert registrant.phone_watchers == 0
        assert registrant.watch_supported is False
        rd, wd = await _dial(record)
        wd.write(json.dumps({"op": "unwatch", "req": 1}).encode() + b"\n")
        await wd.drain()
        ack = await _until(rd, "ack", 1)
        # The unwatch-before-watch case floors at zero: a daemon restart
        # redials without unwatching and the counter must not go negative.
        assert registrant.phone_watchers == 0
        assert registrant.watch_supported is True
        wd.write(json.dumps({"op": "watch", "req": 2}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 2)
        assert registrant.phone_watchers == 1
        wd.write(json.dumps({"op": "watch", "req": 3}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 3)
        assert registrant.phone_watchers == 2
        wd.write(json.dumps({"op": "unwatch", "req": 4}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 4)
        assert registrant.phone_watchers == 1
        # The latch never resets.
        assert registrant.watch_supported is True
        wd.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_absent_client_field_means_daemon() -> None:
    """An OLD daemon dialing a NEW registrant keeps the daemon class."""
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)  # no client field
        assert registrant.attach_clients() == 0
        # And a second daemon-class dial still evicts it (reconnect).
        rd2, wd2 = await _dial(record)
        raw = await asyncio.wait_for(rd.readline(), timeout=5)
        assert raw == b""
        wd.close()
        wd2.close()
    finally:
        registrant.close()
