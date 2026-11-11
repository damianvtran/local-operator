"""A message typed to a BUSY session is queued, never rejected.

THE INCIDENT THESE TESTS PIN (2026-11-04). The operator typed two messages to a
session whose runtime was alive but mid-turn on a host at load ~100; the owner
did not acknowledge inside the desktop control envelope, ``POST /messages``
answered ``503 runtime_busy``, the renderer's bounded resends were spent the same
way, and the composer showed "The agent is busy, so your message wasn't sent" —
the two messages never reached the transcript.

The route now QUEUES such a message into the owner's own inbox
(``routes.desktop_sessions._queue_or_refuse``) when the owner drains it live
(``live-inbox-v1``), answering ``status: "queued"``. These cells drive the REAL
route against a REAL loopback owner (``_FakeOwner``) that parks the admission,
so the ack genuinely times out — no route double.

The at-most-once half (the owner took the socket write AND the spooled copy
exists) is proved against a real ``Session`` in
``tests/unit/session/runtime/test_live_inbox_drain.py``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from local_operator.mobile import attach_client
from local_operator.session.runtime import registry
from local_operator.session.runtime.inbox import SOURCE_USER, inbox_path, peek_inbox
from local_operator.session.runtime.types import LIVE_INBOX_CAPABILITY
from tests.unit.server.test_desktop_admission_ack import (
    DEADLOCK_GUARD_S,
    TOKEN,
    _Harness,
    _post,
)
from tests.unit.session.test_ownerless_read import _FakeOwner, _record

#: The admission ack bound for these cells. The production value is 15 s
#: (``attach_client.ACK_TIMEOUT_S``); the property under test is what the route
#: does once it expires, not how long it takes, so it is shortened.
ACK_S = 0.5


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic token, isolated config dir and HOME: no live session is reachable."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _short_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    original = attach_client.AttachClient._request_frame

    async def bounded(self, op, *, deadline_s=attach_client.ACK_TIMEOUT_S, **fields):
        if op in ("prompt", "steer"):
            deadline_s = ACK_S
        return await original(self, op, deadline_s=deadline_s, **fields)

    monkeypatch.setattr(attach_client.AttachClient, "_request_frame", bounded)


def _publish(root: Path, owner: _FakeOwner, session_id: str, *, live_inbox: bool) -> None:
    record = _record(session_id, owner.port, cwd=root)
    if live_inbox:
        record.capabilities = [*record.capabilities, LIVE_INBOX_CAPABILITY]
    registry.publish(record, root)
    (root / "sessions" / session_id / ".session.pid").write_text(str(record.pid))


@pytest.mark.asyncio
async def test_a_busy_owner_that_drains_live_gets_the_message_queued(tmp_path: Path) -> None:
    """503 runtime_busy before the fix; a durable queued receipt after it."""
    park = asyncio.Event()  # never set: the owner takes the frame and never acks
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            answer_prompts=True,
            park_prompts=park,
        )
        await owner.start()
        _publish(tmp_path, owner, harness.session_id, live_inbox=True)
        try:
            request_id = str(uuid.uuid4())
            response = await asyncio.wait_for(
                _post(harness, "are you there?", request_id=request_id), DEADLOCK_GUARD_S
            )
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            assert result["status"] == "queued", result
            assert result["command_id"] == request_id
            assert "busy" in result["detail"]

            rows = peek_inbox(tmp_path / "sessions" / harness.session_id)
            assert [(r.text, r.source, r.command_id, r.mode) for r in rows] == [
                ("are you there?", SOURCE_USER, request_id, "mailbox")
            ]

            # A RETRY OF THE SAME REQUEST REPLAYS THE RECEIPT, it does not spool
            # a second row (the receipt journal keys on the request id).
            again = await asyncio.wait_for(
                _post(harness, "are you there?", request_id=request_id), DEADLOCK_GUARD_S
            )
            assert again.status_code == 200, again.text
            assert again.json()["result"]["status"] == "queued"
            assert again.json()["result"]["replayed"] is True
            assert len(peek_inbox(tmp_path / "sessions" / harness.session_id)) == 1
        finally:
            park.set()
            await owner.stop()


@pytest.mark.asyncio
async def test_an_owner_that_does_not_drain_live_keeps_the_busy_refusal(tmp_path: Path) -> None:
    """No ``live-inbox-v1`` -> nothing would read the row; the honest answer stays 503."""
    park = asyncio.Event()
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            answer_prompts=True,
            park_prompts=park,
        )
        await owner.start()
        _publish(tmp_path, owner, harness.session_id, live_inbox=False)
        try:
            response = await asyncio.wait_for(
                _post(harness, request_id=str(uuid.uuid4())), DEADLOCK_GUARD_S
            )
            assert response.status_code == 503, response.text
            assert response.json()["detail"]["code"] == "runtime_busy"
            assert not inbox_path(tmp_path / "sessions" / harness.session_id).exists() or not (
                peek_inbox(tmp_path / "sessions" / harness.session_id)
            )
        finally:
            park.set()
            await owner.stop()


@pytest.mark.asyncio
async def test_a_steer_to_a_busy_owner_is_queued_as_a_steer(tmp_path: Path) -> None:
    """The mode rides the row, so the live drain joins it to the turn in flight."""
    park = asyncio.Event()
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            answer_prompts=True,
            park_prompts=park,
        )
        await owner.start()
        _publish(tmp_path, owner, harness.session_id, live_inbox=True)
        try:
            assert harness.client is not None
            request_id = str(uuid.uuid4())
            response = await asyncio.wait_for(
                harness.client.post(
                    f"/v1/desktop/sessions/{harness.session_id}/messages",
                    json={"request_id": request_id, "text": "also check X", "mode": "steer"},
                ),
                DEADLOCK_GUARD_S,
            )
            assert response.status_code == 200, response.text
            assert response.json()["result"]["status"] == "queued"
            raw = inbox_path(tmp_path / "sessions" / harness.session_id).read_text()
            assert json.loads(raw.splitlines()[0])["mode"] == "steer"
        finally:
            park.set()
            await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["queue", "both"])
async def test_a_queued_mode_to_a_busy_owner_is_refused_not_downgraded(
    tmp_path: Path, delivery: str
) -> None:
    """A ``queue``/``both`` send to a busy owner keeps the busy refusal.

    The spool row (``InboxLine(mode='steer'|...)``) carries no delivery mode,
    so queueing here would deliver the message as an URGENT steer while the
    receipt says "queued" — the silent downgrade the delivery modes exist to
    prevent. The honest answer is the capability refusal (409, from the
    desktop admit path's own gate — the owner never advertised the token);
    the renderer's resend applies.
    """
    park = asyncio.Event()
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            answer_prompts=True,
            park_prompts=park,
        )
        await owner.start()
        _publish(tmp_path, owner, harness.session_id, live_inbox=True)
        try:
            assert harness.client is not None
            response = await asyncio.wait_for(
                harness.client.post(
                    f"/v1/desktop/sessions/{harness.session_id}/messages",
                    json={
                        "request_id": str(uuid.uuid4()),
                        "text": "hold this, do not interrupt",
                        "delivery": delivery,
                    },
                ),
                DEADLOCK_GUARD_S,
            )
            assert response.status_code == 409, response.text
            assert response.json()["detail"] == (
                "this session's runtime predates delivery modes (queue/both); "
                "send it as a steer, or update the runtime"
            )
            rows = peek_inbox(tmp_path / "sessions" / harness.session_id)
            assert rows == [], "a refused delivery must not be spooled"
        finally:
            park.set()
            await owner.stop()
