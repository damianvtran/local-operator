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

import pytest

from local_operator.mobile import attach_client
from local_operator.server.routes.desktop_sessions import _owner_busy
from tests.unit.server.test_desktop_read_without_owner import TOKEN, _Harness
from tests.unit.session.test_ownerless_read import (
    SESSION_ID,
    _cold_viewer,
    _FakeOwner,
    _publish_live,
    _seed,
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
