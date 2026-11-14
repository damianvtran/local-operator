"""A live owner whose welcome is late is BUSY on every bind path, never unreachable.

The incident (Aida, 2026-11-09, desktop 0.52.13 / backend 0.101.37): choosing a
model in the desktop Model dialog answered ``503 runtime_unreachable`` -- the UI
appends "Session owner is unavailable. Reconnect and reconcile before retrying."
for exactly that code -- while the runtime was live and a hand retry switched it.

The cause these tests pin: a bind whose WELCOME read expired on the attach
client's own timer (``attach_client.ACK_TIMEOUT_S``) surfaced as a bare
``ConnectionError("owner did not send its state")``, which the control ladder
(``routes/desktop_sessions.py::errors``) answers as unreachable. Only the
``attach_existing(control_budget=...)`` branch converted a welcome expiry into
the busy disposition, and only when ITS OWN outer deadline fired first. The
foreground ``_ensure_bound`` / ``bind_runtime`` path (which the ``/commands``
route calls before forwarding the slash) dials with no deadline, so it waits on
the client's timer and reported a live owner as gone.

The timer is shortened per test rather than waited out: the property under test
is WHICH error the expiry raises, not how long the real envelope is. Below
``DESKTOP_CONTROL_ATTACH_S`` it also makes the route test deterministic, because
the client's timer then fires before the acquire's outer deadline -- the same
race any bind without that outer deadline always loses.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import attach_client
from local_operator.server.routes.desktop_sessions import _owner_busy
from local_operator.session.runtime import registry
from tests.unit.server.test_desktop_read_without_owner import TOKEN, _Harness
from tests.unit.session.test_ownerless_read import (
    OWNER_PID,
    SESSION_ID,
    _cold_viewer,
    _FakeOwner,
    _publish_live,
    _seed,
    _until,
)

#: Well under ``DESKTOP_CONTROL_ATTACH_S`` (3.0 s): see the module docstring.
WELCOME_TIMER_S = 0.5


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(attach_client, "ACK_TIMEOUT_S", WELCOME_TIMER_S)


async def _pick_model(harness: _Harness):
    """The desktop Model dialog's request: ``command.run("model", value)``."""
    assert harness.client is not None
    return await harness.client.post(
        f"/v1/desktop/sessions/{harness.session_id}/commands",
        json={
            "request_id": str(uuid.uuid4()),
            "command": "model",
            "args": "openai/gpt-5",
            "images": [],
        },
    )


@pytest.mark.asyncio
async def test_a_model_pick_on_a_live_owner_that_welcomes_late_is_busy(tmp_path: Path) -> None:
    """The reported request, end to end through the real route and facade."""
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path, mute=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            response = await _pick_model(harness)
        finally:
            await owner.stop()

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "runtime_busy", detail
    assert detail["retryable"] is True
    assert response.headers["retry-after"] == "2"


@pytest.mark.asyncio
async def test_a_model_pick_on_a_gone_owner_is_still_unreachable(tmp_path: Path) -> None:
    """Negative control: the record and pid stay, but nothing holds the port."""
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        await owner.stop()
        response = await _pick_model(harness)

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "runtime_unreachable", detail
    assert "retryable" not in detail


@pytest.mark.asyncio
async def test_the_foreground_bind_classifies_a_late_welcome_as_owner_alive(
    tmp_path: Path,
) -> None:
    """``bind_runtime`` -> ``_ensure_bound``: the path with no outer deadline.

    This is the bind the ``/commands`` route (and every lifecycle control
    route) calls before it forwards anything, and it is the one that waits on
    the client's welcome timer alone.
    """
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, mute=True)
    viewer = await _cold_viewer(tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    try:
        with pytest.raises(ConnectionError) as raised:
            await viewer.bind_runtime()
    finally:
        await viewer.dispose()
        await owner.stop()

    assert owner.conns >= 1, "the bind never dialled the owner"
    assert _owner_busy(raised.value), repr(raised.value)
    # The historical sentence is kept: the TUI's resume retry matches it.
    assert str(raised.value) == "owner did not send its state"


@pytest.mark.asyncio
async def test_the_foreground_bind_to_a_closed_port_is_not_owner_alive(tmp_path: Path) -> None:
    """Negative control for the facade path: a refused dial stays unreachable."""
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    viewer = await _cold_viewer(tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    await owner.stop()
    try:
        with pytest.raises(ConnectionError) as raised:
            await viewer.bind_runtime()
    finally:
        await viewer.dispose()

    assert not _owner_busy(raised.value), repr(raised.value)


@pytest.mark.asyncio
async def test_an_owner_that_exits_between_bind_attempts_is_not_owner_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newer proof of absence outranks an earlier attempt's busy disposition.

    Agent review round 1, R1-1: attempt 1 hits a live owner that welcomes late
    (``OwnerWelcomeTimeout``, ``owner_alive``), the owner then exits and its
    record is removed, and attempt 2's ``locate`` finds no owner at all.
    ``_bind_under_lock`` used to rethrow the cached attempt-1 error, so a
    KNOWN-GONE owner reached the ladder as ``runtime_busy`` — the exact
    mislabel the dead-owner control above exists to forbid, only reached via
    a retry. The sentence must survive (the TUI matches it); only the
    disposition is dropped.
    """
    monkeypatch.setattr(attach_client, "ACK_TIMEOUT_S", 0.1)
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, mute=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    original = attach_client.AttachClient.connect

    async def connect_then_owner_exits(client: Any, record: Any, session_id: str) -> Any:
        try:
            return await original(client, record, session_id)
        except attach_client.OwnerWelcomeTimeout:
            # The owner goes away AFTER proving itself alive: the socket closes
            # and its record is unpublished, so the next locate() finds nothing.
            await owner.stop()
            registry.record_path(OWNER_PID, tmp_path).unlink()
            raise

    monkeypatch.setattr(attach_client.AttachClient, "connect", connect_then_owner_exits)
    try:
        with pytest.raises(ConnectionError) as raised:
            await viewer.bind_runtime()
    finally:
        await viewer.dispose()
        await owner.stop()

    assert owner.conns >= 1, "the bind never dialled the owner"
    assert not _owner_busy(raised.value), repr(raised.value)
    assert str(raised.value) == "owner did not send its state"
    # The earlier attempt stays reachable for diagnosis.
    assert isinstance(raised.value.__cause__, attach_client.OwnerWelcomeTimeout)


@pytest.mark.asyncio
async def test_an_ask_answer_to_a_live_owner_that_welcomes_late_is_owner_alive(
    tmp_path: Path,
) -> None:
    """``POST /answers`` for a queued ask binds through the SAME foreground path.

    ``ask_respond`` -> ``_ask_client_method`` -> ``_ensure_bound``: an answer to
    a busy owner must be retryable too, and the welcome precedes the ask frame,
    so nothing was delivered when the bind gave up.
    """
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, mute=True)
    viewer = await _cold_viewer(tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    try:
        with pytest.raises(ConnectionError) as raised:
            await viewer.ask_respond("ask-synthetic", {"q1": ["yes"]})
    finally:
        await viewer.dispose()
        await owner.stop()

    assert owner.conns >= 1, "the answer never dialled the owner"
    assert _owner_busy(raised.value), repr(raised.value)


async def _recovering_viewer(tmp_path: Path) -> tuple[Any, _FakeOwner]:
    """A viewer whose stream to a LIVE owner dropped and is re-dialling it.

    The owner answers the first dial, then sheds this viewer (the busy-owner
    shapes: an event-queue overflow, or the sequence rewind fixed alongside
    this) and stops welcoming new dials in time, so the recovery is still in
    flight when the next control call lands. Its record and pid stay live.
    """
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, sync_on_connect=True, answer_watch=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    assert await viewer.attach_existing(control_budget=5.0) is True
    owner.mute = True
    for writer in list(owner._writers):
        writer.close()
    owner._writers.clear()
    await _until(lambda: viewer._recovering, why="the drop never started a recovery")
    return viewer, owner


def _bury(tmp_path: Path) -> None:
    """The owner is GONE: no record names it any more."""
    registry.record_path(OWNER_PID, tmp_path).unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize("gone", [False, True], ids=["live-owner", "gone-owner"])
async def test_a_slash_during_recovery_is_busy_only_while_the_owner_lives(
    tmp_path: Path, gone: bool
) -> None:
    """``route_shared_slash``'s reconnecting arm: the ``/commands`` model pick."""
    viewer, owner = await _recovering_viewer(tmp_path)
    try:
        if gone:
            _bury(tmp_path)
        with pytest.raises(ConnectionError) as raised:
            await viewer.route_shared_slash("model", "openai/gpt-5")
    finally:
        await viewer.dispose()
        await owner.stop()

    assert _owner_busy(raised.value) is (not gone), repr(raised.value)
    # The TUI's sentence is unchanged on both arms.
    assert "try /model again" in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("gone", [False, True], ids=["live-owner", "gone-owner"])
async def test_an_ask_answer_during_recovery_is_busy_only_while_the_owner_lives(
    tmp_path: Path, gone: bool
) -> None:
    """``_ask_client_method``'s no-connection arm: ``POST /answers`` mid-reconnect."""
    viewer, owner = await _recovering_viewer(tmp_path)
    try:
        if gone:
            _bury(tmp_path)
        with pytest.raises(ConnectionError) as raised:
            await viewer.ask_respond("ask-synthetic", {"q1": ["yes"]})
    finally:
        await viewer.dispose()
        await owner.stop()

    assert _owner_busy(raised.value) is (not gone), repr(raised.value)
