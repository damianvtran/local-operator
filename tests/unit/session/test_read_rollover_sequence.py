"""A read's rollover must never rewind the canonical store behind the stream.

The incident (Aida, 2026-11-09): the desktop backend logged ``attach client:
owner frame could not be applied: frontend update is not the next state
sequence; the connection cannot continue`` 18 times in one day, each in a busy
stretch, and the control calls landing in the recovery that followed answered
``503`` (the operator's "Session owner is unavailable" toast on a model pick and
on ask answers).

The cause: both READ installs re-installed the sync's snapshot to publish their
rollover AFTER the deltas buffered behind that sync had been replayed (and after
an await that let more apply). The store rewound to the snapshot's sequence, the
transport's counter did not, and the next delta failed the store's exact ``+1``
check. A busy owner always has deltas in flight, which is why only busy sessions
hit it. These rows drive the real facade against a loopback owner speaking the
real wire frames: a sync followed at once by the next delta, then one more.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import READ_ATTACH_BUDGET_S
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendSync,
    sync_wire_payload,
)
from tests.unit.session.test_ownerless_read import (
    _cold_viewer,
    _FakeOwner,
    _publish_live,
    _seed,
    _until,
)

SESSION = "abcdef123456"


def _sync(sequence: int) -> dict[str, Any]:
    """A canonical sync whose snapshot is AT ``sequence``, as a real owner's is.

    The shared fixture's sync leaves the snapshot at sequence 0 under a frame
    that says 1; rows about sequence arithmetic need the two to agree.
    """
    sync = FrontendSync(
        epoch="fake-owner",
        sequence=sequence,
        snapshot=FrontendSessionState(session_id=SESSION, epoch="fake-owner", sequence=sequence),
    )
    return {"op": "frontend_sync", "data": sync_wire_payload(sync)}


def _delta(sequence: int) -> dict[str, Any]:
    """The next ordered canonical delta, as a live owner streams it."""
    return {
        "op": "frontend_update",
        "data": {"epoch": "fake-owner", "sequence": sequence, "changes": {}},
    }


async def _send(owner: _FakeOwner, *frames: dict[str, Any]) -> None:
    """Write frames back to back in ONE write, as a busy owner's drain does."""
    payload = b"".join(json.dumps(frame).encode() + b"\n" for frame in frames)
    for writer in list(owner._writers):
        writer.write(payload)
        await writer.drain()


async def _assert_stream_survives(viewer: Any, owner: _FakeOwner) -> None:
    store = viewer._frontend_store
    assert store is not None
    client = viewer._client
    assert client is not None and client.connected
    await _send(owner, _delta(3))
    await _until(
        lambda: store.state.sequence == 3 or not client.connected,
        why="the third delta was neither applied nor refused",
    )
    assert client.connected, "a live connection was torn down over its own in-order delta"
    assert not viewer._recovering
    assert store.state.sequence == 3


@pytest.mark.asyncio
async def test_a_late_landing_sync_keeps_the_deltas_behind_it(tmp_path: Path) -> None:
    """The retained-dial arm: the read gave up, then the busy owner's sync landed."""
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION, tmp_path)  # welcomes, does not sync yet
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    try:
        assert await viewer.attach_existing(budget=0.2) is False
        assert viewer.attaching, "the read should have retained its dial"
        await _send(owner, _sync(1), _delta(2))
        await _until(lambda: not viewer.is_cold, why="the late sync never landed")
        await _assert_stream_survives(viewer, owner)
    finally:
        await viewer.dispose()
        await owner.stop()


@pytest.mark.asyncio
async def test_an_in_budget_read_sync_keeps_the_deltas_behind_it(tmp_path: Path) -> None:
    """The ``_bind_to(retain_unsynced=True)`` arm: the sync landed inside the budget."""
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    try:
        task = asyncio.create_task(viewer.attach_existing(budget=READ_ATTACH_BUDGET_S))
        await _until(lambda: owner.conns >= 1 and bool(owner._writers), why="no dial")
        # Let the welcome go out before the sync, as on the wire.
        await _until(lambda: viewer._client is not None, why="no client")
        await _send(owner, _sync(1), _delta(2))
        assert await task is True
        await _assert_stream_survives(viewer, owner)
    finally:
        await viewer.dispose()
        await owner.stop()
