"""``POST /v1/desktop/projects/{key}/request-update`` against a REAL registrant.

The route's own behaviour is what lives here: the frozen check-in text and its
substitutions, the mailbox+wake wire semantics, the strictly sequential loop, the
three-way per-session outcome mapping, and the per-project cooldown. Delivery is
exercised end to end through the shared peer-send core, not mocked — the target
is a real ``RuntimeServer`` publishing a real discovery record, dialled over its
real loopback control socket, exactly as ``tests/unit/tools/test_send_tool.py``
does for the ``send`` tool.

A project row may only link a 12-character hex session id, and the registrant's
own record carries the handle's non-hex ``s1``, so each deliverable target is an
ALIAS record published under a live pid that is NOT this process's (the parent's)
pointing at the registrant's socket — the same alias trick the send-tool suite
uses, and it also keeps the route's resolve off a self-record.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.request_update import (
    NEVER_STARTED_DETAIL,
    REQUEST_UPDATE_TEMPLATE,
    reset_cooldowns,
)
from local_operator.server.routes import capabilities, desktop_projects
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle

pytestmark = pytest.mark.asyncio

TOKEN = "projects-desktop-token"
SESSION_A = "4e92693767fa"
SESSION_B = "7b31c0d4a9e2"
SESSION_C = "1f0a55e83b7c"

#: The exact text the route must send for the ``payments`` test project (no
#: title), with ``{today}`` left for the assertion to fill from the local date.
EXPECTED_TEXT = REQUEST_UPDATE_TEMPLATE.format(display="payments", name="payments", today="{today}")


class _RecordingHandle(FakeHandle):
    """Records each peer delivery and whether any two overlapped.

    ``max_active`` is the sequential-loop assertion that does not depend on a
    clock: the route dials one target at a time, so a real sequential loop never
    lets two ``receive_peer_message`` calls be in flight together, while a
    ``gather``-shaped bug would. The tiny in-flight sleep widens that window so
    the reading is structural rather than accidental.
    """

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def receive_peer_message(  # noqa: ANN001, ANN202
        self, text, *, mode="mailbox", wake=False, sender=None
    ) -> str:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(
                ("receive_peer_message", (text,), {"mode": mode, "wake": wake, "sender": sender})
            )
            await asyncio.sleep(0.01)
            return "delivered to the mailbox (will be read on the next turn)"
        finally:
            self.active -= 1


@pytest.fixture(autouse=True)
def _fresh_cooldowns() -> None:
    """The cooldown map is process-global; every test starts from a clean one."""
    reset_cooldowns()


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("CMUX_") or name.startswith("LOP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_projects.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


async def _project_with(client: AsyncClient, *session_ids: str, name: str = "payments") -> str:
    """Create a project and link ``session_ids``; return its id."""
    created = await client.post("/v1/desktop/projects", json={"name": name})
    assert created.status_code == 200
    project_id = created.json()["result"]["id"]
    for session_id in session_ids:
        linked = await client.post(
            f"/v1/desktop/projects/{project_id}/links", json={"session_id": session_id}
        )
        assert linked.status_code == 200
    return project_id


class _Targets:
    """One real registrant plus the socket its aliases point at.

    A registry record is keyed by PID (one file per pid), so distinct aliases
    need distinct LIVE pids. ``sleepers`` are child processes this test owns and
    reaps in :meth:`close`: stable, alive for the whole test, and never this
    process — unlike an inherited ancestor pid, whose lifetime we do not control
    and which flaked when one exited mid-test.
    """

    def __init__(
        self,
        registrant: RuntimeServer,
        handle: _RecordingHandle,
        port: int,
        key: str,
        sleepers: "list[subprocess.Popen[bytes]]",
    ):
        self.registrant = registrant
        self.handle = handle
        self.port = port
        self.key = key
        self.sleepers = sleepers

    def publish(self, session_id: str, *, index: int = 0, started: bool = True) -> None:
        registry.publish(
            registry.SessionRecord(
                pid=self.sleepers[index].pid,
                kind="tui",
                session_id=session_id,
                conversation_name=session_id,
                cwd="/tmp",
                model_label="test/model",
                control_port=self.port,
                control_key=self.key,
                started=started,
            )
        )

    def close(self) -> None:
        self.registrant.close()
        for proc in self.sleepers:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 — a stuck reaper must not mask the result
                proc.kill()


def _sleeper_procs(count: int) -> "list[subprocess.Popen[bytes]]":
    """``count`` child processes we own and reap; their pids are live anchors."""
    return [
        subprocess.Popen(  # noqa: S603 — a fixed interpreter invocation, no shell
            [sys.executable, "-c", "import time; time.sleep(120)"]
        )
        for _ in range(count)
    ]


async def _wait_live(*session_ids: str, deadline_s: float = 30.0) -> None:
    """Block until every session is visible as a LIVE record.

    ``registry.publish`` writes the record synchronously, but the resolver's
    scan classifies by pid liveness (a ``ps`` probe), which is slow enough under
    fleet load that a route call firing immediately after a publish can race the
    first classification. Waiting here makes the delivery tests deterministic
    instead of dependent on host load.
    """
    wanted = set(session_ids)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while loop.time() < deadline:
        seen = {rec.session_id for rec, state in registry.scan() if state == "live"}
        if wanted <= seen:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"records never went live: {sorted(wanted)}")


async def _own_record(deadline_s: float = 30.0) -> registry.SessionRecord:
    """The registrant's own live record, waiting out a loaded host."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while loop.time() < deadline:
        for rec, state in registry.scan():
            if state == "live":
                return rec
        await asyncio.sleep(0.05)
    raise AssertionError("runtime never published a live record")


async def _start_targets() -> _Targets:
    handle = _RecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    sleepers = _sleeper_procs(2)
    try:
        # A fresh RuntimeServer starts ``started=False`` and its OWN record is
        # what the receive-side gate reads, so a delivery to it would be refused
        # as unengaged. Flip it the way a real first turn would.
        registrant.set_record_started(True)
        own = await _own_record()
        return _Targets(registrant, handle, own.control_port, own.control_key, sleepers)
    except BaseException:
        registrant.close()
        for proc in sleepers:
            proc.kill()
        raise


async def _dropping_socket() -> "tuple[asyncio.Server, int]":
    """A raw loopback socket that reads a request then closes without acking.

    This is the UNCONFIRMED case produced honestly: the transport fails after
    the message has been handed to the kernel, so nothing can say whether it
    landed — the class the route must report as ``unconfirmed``, not ``failed``.
    """

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read(4096)
        except Exception:  # noqa: BLE001 — a torn-down reader is the point
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _last_delivery(handle: _RecordingHandle) -> dict[str, Any]:
    name, args, kwargs = handle.calls[-1]
    assert name == "receive_peer_message"
    return {"text": args[0], **kwargs}


async def test_delivery_sends_the_frozen_text_as_a_waking_mailbox_drop(api) -> None:
    from datetime import date

    client, _root = api
    project_id = await _project_with(client, SESSION_A)
    targets = await _start_targets()
    try:
        targets.publish(SESSION_A)
        await _wait_live(SESSION_A)
        response = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["state"] == "sent"
        assert result["counts"] == {"total": 1, "delivered": 1, "unconfirmed": 0, "failed": 0}
        (row,) = result["sessions"]
        assert row["outcome"] == "delivered"
        assert row["session_id"] == SESSION_A
        assert result["requested_at"] is not None
        assert response.json()["message"] == "Requested an update from 1 session on payments."

        delivery = _last_delivery(targets.handle)
        assert delivery["text"] == EXPECTED_TEXT.format(today=date.today().isoformat())
        # Mailbox drop with wake=True — the `send` tool's own default, NOT a steer.
        assert delivery["mode"] == "mailbox"
        assert delivery["wake"] is True
        assert delivery["sender"]["conversation_name"] == "Projects"
    finally:
        targets.close()


async def test_two_targets_are_dialled_strictly_sequentially(api) -> None:
    client, _root = api
    project_id = await _project_with(client, SESSION_A, SESSION_B)
    targets = await _start_targets()
    try:
        targets.publish(SESSION_A, index=0)
        targets.publish(SESSION_B, index=1)
        await _wait_live(SESSION_A, SESSION_B)
        response = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
        result = response.json()["result"]
        assert result["counts"] == {"total": 2, "delivered": 2, "unconfirmed": 0, "failed": 0}
        assert [row["session_id"] for row in result["sessions"]] == [SESSION_A, SESSION_B]
        # Two dials, one at a time: a concurrent loop would overlap here.
        assert len(targets.handle.calls) == 2
        assert targets.handle.max_active == 1
    finally:
        targets.close()


async def test_an_empty_project_dials_nothing_and_says_so(api) -> None:
    client, _root = api
    project_id = await _project_with(client)  # no links
    response = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["state"] == "empty"
    assert body["result"]["sessions"] == []
    assert body["result"]["counts"]["total"] == 0
    assert body["message"] == "No linked sessions to ask. Link a session to payments first."


async def test_a_second_press_inside_the_window_dials_nothing_and_says_so(api) -> None:
    client, _root = api
    project_id = await _project_with(client, SESSION_A)
    targets = await _start_targets()
    try:
        targets.publish(SESSION_A)
        await _wait_live(SESSION_A)
        first = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
        assert first.json()["result"]["state"] == "sent"
        dials_after_first = len(targets.handle.calls)

        second = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
        assert second.status_code == 200
        body = second.json()
        assert body["result"]["state"] == "cooldown"
        assert body["result"]["cooldown_remaining_s"] == 60
        assert body["result"]["sessions"] == []
        # The numbers are whole seconds rounded up (the frozen rule), so the
        # elapsed figure can be 0 or 1 on a fast machine; the shape is fixed.
        assert body["message"].startswith("Update already requested ")
        assert body["message"].endswith(" on payments. Try again in 60 s.")
        assert " s ago" in body["message"]
        # No second dial.
        assert len(targets.handle.calls) == dials_after_first
    finally:
        targets.close()


async def test_an_unconfirmed_delivery_is_its_own_outcome(api) -> None:
    client, _root = api
    project_id = await _project_with(client, SESSION_A)
    server, port = await _dropping_socket()
    (sleeper,) = _sleeper_procs(1)
    try:
        # A live record whose socket accepts then closes without acking: the
        # message may have landed, so this is "unconfirmed", never "could not
        # reach".
        registry.publish(
            registry.SessionRecord(
                pid=sleeper.pid,
                kind="tui",
                session_id=SESSION_A,
                conversation_name=SESSION_A,
                cwd="/tmp",
                model_label="test/model",
                control_port=port,
                control_key="0" * 64,
                started=True,
            )
        )
        await _wait_live(SESSION_A)
        response = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
        result = response.json()["result"]
        (row,) = result["sessions"]
        assert row["outcome"] == "unconfirmed"
        assert row["detail"] == "delivery could not be confirmed"
        assert result["counts"] == {"total": 1, "delivered": 0, "unconfirmed": 1, "failed": 0}
        # Unconfirmed STARTS the cooldown: it may have landed.
        assert result["requested_at"] is not None
        assert response.json()["message"] == (
            "Could not confirm delivery on payments — the requests may still reach its sessions."
        )
    finally:
        server.close()
        await server.wait_closed()
        sleeper.terminate()
        try:
            sleeper.wait(timeout=5)
        except Exception:  # noqa: BLE001
            sleeper.kill()


async def test_a_never_started_linked_session_is_named_not_silently_dropped(api) -> None:
    client, _root = api
    project_id = await _project_with(client, SESSION_A)
    targets = await _start_targets()
    try:
        # A live record that has not run a turn yet: the unengaged gate refuses it.
        targets.publish(SESSION_A, started=False)
        await _wait_live(SESSION_A)
        response = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
        result = response.json()["result"]
        (row,) = result["sessions"]
        assert row["outcome"] == "failed"
        assert row["detail"] == NEVER_STARTED_DETAIL
        assert result["counts"]["failed"] == 1
        assert response.json()["message"] == (
            "The 1 linked sessions have not started yet — they become recipients "
            "after their first message."
        )
        # A pure refusal does NOT start the cooldown: nothing was handed off.
        assert result["requested_at"] is None
    finally:
        targets.close()


async def test_a_session_that_no_longer_exists_is_reported(api) -> None:
    client, _root = api
    # Linked, but neither a live record nor a session directory exists.
    project_id = await _project_with(client, SESSION_C)
    response = await client.post(f"/v1/desktop/projects/{project_id}/request-update", json={})
    result = response.json()["result"]
    (row,) = result["sessions"]
    assert row["outcome"] == "failed"
    assert row["detail"] == "no longer exists"


async def test_an_unknown_project_is_a_404(api) -> None:
    client, _root = api
    response = await client.post("/v1/desktop/projects/does-not-exist/request-update", json={})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "project_not_found"


async def test_the_capability_key_is_advertised(api) -> None:
    client, _root = api
    caps = await client.get("/v1/capabilities")
    assert caps.status_code == 200
    features: dict[str, Any] = caps.json()["result"]["features"]
    assert features.get("projects_request_update", 0) >= 1
    # The sibling key is untouched — this is its OWN key, not a bump.
    assert features.get("projects") == 1
