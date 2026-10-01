"""The 503 an ANSWER route gives a live owner carries its disposition.

THE DEFECT THIS REPRODUCES. ``POST /v1/desktop/sessions/{id}/answers`` answered a
spurious ``503 runtime_unreachable`` — "reconnect and reconcile" — for an owner
that was provably alive. The live arm on this tree is the lost ACKNOWLEDGEMENT:
the answer reached a connected, synced runtime, the runtime settled the gate, and
the ack back to the backend timed out. ``OwnerAckTimeout`` is a
``ConnectionError``, and the route ladder read every bare ``ConnectionError`` as
"nobody is there", so a request that had already been honoured came back with the
remedy for a dead socket.

THE DISCRIMINATOR IS THE CODE, not the status: both shapes are 503. Before, the
call came back ``runtime_unreachable`` with no disposition. After, it carries
``runtime_busy`` with ``retryable: true`` — the same shape the
``RuntimeUnresponsiveError`` arm already answers, because it is the same fact.

The NEGATIVE is asserted in the same file, because the fix must not turn real
unreachability into a promise: an owner whose port is closed still refuses
``runtime_unreachable`` with no ``retryable`` field.

Everything runs in an isolated config root with synthetic ids. The owner-side
silence in the first test is the one thing simulated (the real one costs an
eight-second envelope); the route, the acquire, the control attach, the ladder
and the 503 body are the real path.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile.attach_client import OwnerAckTimeout
from tests.unit.server.test_desktop_read_without_owner import (
    DEADLOCK_GUARD_S,
    TOKEN,
    _Harness,
)
from tests.unit.session.test_ownerless_read import _FakeOwner, _publish_live


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sibling module's autouse token fixture does not reach this module.

    ``_Harness`` builds its client with that module's ``TOKEN``, so without this
    every route answers the unmanaged-mode refusal rather than the shape under
    test.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)


async def _wait_for_owner_state(bridge: Any, *, timeout: float = DEADLOCK_GUARD_S) -> Any:
    """Wait for the bridge's facade to hold the OWNER's canonical state.

    ``frontend_state`` alone is not the signal: a cold facade carries a store
    seeded from the session's disk preview, so it answers without ever having
    heard from the owner. ``not is_cold`` is the term that means a client is
    connected AND its canonical state landed, which is the precondition the
    control routes need.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remote = bridge.remote
        if remote is not None and remote.owner_reachable and not remote.is_cold:
            return remote
        await asyncio.sleep(0.02)
    raise AssertionError("the owner's canonical state never synced")


@pytest.mark.asyncio
async def test_an_owner_that_never_acks_the_answer_refuses_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arm C over the real route: the owner is alive and synced, its ack is lost."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path, sync_on_connect=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        base = f"/v1/desktop/sessions/{harness.session_id}"

        # ONE held context: the pool keeps one bridge per session while a caller
        # holds a reference, so the POST below reaches the SAME facade this test
        # read the epoch from — otherwise the route would build a fresh
        # disk-seeded facade and the epoch check would refuse before the ack leg.
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            remote = await _wait_for_owner_state(bridge)

            async def _never_acks(*args: Any, **kwargs: Any) -> str:
                raise OwnerAckTimeout("owner did not answer 'approval_answer' within 8s")

            monkeypatch.setattr(remote, "answer_gate", _never_acks)
            response = await harness.client.post(
                f"{base}/answers",
                json={
                    "epoch": remote.frontend_state.epoch,
                    "request_id": "gate-1",
                    "approved": True,
                },
            )

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "runtime_busy", detail
        assert detail["retryable"] is True, detail
        assert detail["retry_after_ms"] == 2000
        assert response.headers["retry-after"] == "2"
        assert detail["message"].startswith("Session owner is unavailable.")
        await owner.stop()


@pytest.mark.asyncio
async def test_a_genuinely_dead_owner_still_refuses_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NEGATIVE: real unreachability is not papered over by a retryable code.

    The record stays published and the pid stays alive while the port is closed,
    which is the shape a crashed runtime leaves. ``retryable`` is a promise that
    resending the same request will help; an owner nobody can dial has earned no
    such promise, and this is the assertion that keeps the new disposition from
    being applied to it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        await owner.stop()
        base = f"/v1/desktop/sessions/{harness.session_id}"

        response = await harness.client.post(
            f"{base}/answers",
            json={"epoch": "fake-owner", "request_id": "gate-1", "approved": True},
        )

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "runtime_unreachable", detail
        assert "retryable" not in detail, detail
        assert detail["message"].startswith("Session owner is unavailable.")
