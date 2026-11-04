"""A changed borrow receipt republishes the serving host's access claim.

The composer band ("<owner> cannot provide access ... Last checked: <time>") is
the host's ``model_access`` claim, which embeds this process's effective-access
receipt. The host computed it only at open and on a model switch, so a band set
while the owner device slept stayed frozen, with its old ``observed_at``,
through every later verified borrow from that owner. These drive the real
``ServingSessionHandle`` and ``FrontendStateStore`` against real receipts and a
real placement document (synthetic owner and recipient; no network, no
credential, no OAuth refresh).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import effective
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.effective import (
    note_lifecycle_observation,
    note_resolution,
)
from local_operator.network.credentials.types import BrokerError
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.network.test_credentials_placement import OWNER, _point_config_at
from tests.unit.network.test_native_hot_access import _native, _source
from tests.unit.session.runtime.test_serving import _AccessSession

SELECTOR = "openai/gpt-6.1-sol"


@pytest.fixture(autouse=True)
def _fresh_receipts() -> Any:
    # Receipts and listeners are process-wide by design; a test must neither
    # inherit nor leave any.
    effective._receipts.clear()
    listeners = list(effective._listeners)
    yield
    effective._receipts.clear()
    assert effective._listeners == listeners, "a host left its listener installed"


@pytest.fixture
def mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    _point_config_at(monkeypatch, tmp_path)
    source = _source(tmp_path)
    monkeypatch.setattr(MeshCredentialClient, "for_this_device", lambda *_a, **_k: source)
    yield source
    source.close()


def _band(session: Any) -> Any:
    return session._frontend_state_store.state.model_access


async def _settle(handle: ServingSessionHandle) -> None:
    # One loop turn for ``call_soon_threadsafe``, then the off-loop read itself.
    for _ in range(3):
        await asyncio.sleep(0)
        publish = handle._model_access_publish
        if publish is not None:
            await publish


async def _host(tmp_path: Path, selector: str = SELECTOR) -> tuple[Any, ServingSessionHandle]:
    session = _AccessSession(selector)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), config_dir=tmp_path
    )
    return session, handle


@pytest.mark.asyncio
async def test_verified_borrow_clears_a_stale_owner_unavailable_band(
    tmp_path: Path, mesh: Any
) -> None:
    note_resolution(tmp_path, "openai", BrokerError(code="owner_offline", owner_device=OWNER))
    session, handle = await _host(tmp_path)
    before = _band(session)
    assert before is not None and before.effective_state == "owner_unavailable"
    assert before.state == "temporarily_unavailable"
    time.sleep(0.01)  # observed_at is wall time; make "newer" observable

    note_resolution(tmp_path, "openai", _native())
    await _settle(handle)

    after = _band(session)
    assert after is not None
    assert (after.effective_state, after.state) == ("effective_borrow", "ok")
    assert after.owner_device == OWNER
    # "Last checked" is the NEW check, not the frozen one.
    assert after.observed_at is not None and before.observed_at is not None
    assert after.observed_at > before.observed_at
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_newer_refusal_is_republished_too(tmp_path: Path, mesh: Any) -> None:
    # The republish is a recomputation, not a "clear": a refusal after a working
    # borrow must reach the band just as promptly, keeping its keyless history.
    note_resolution(tmp_path, "openai", _native())
    session, handle = await _host(tmp_path)
    assert _band(session).effective_state == "effective_borrow"

    note_resolution(tmp_path, "openai", BrokerError(code="grant_invalid", owner_device=OWNER))
    await _settle(handle)

    band = _band(session)
    assert (band.effective_state, band.state) == ("withdrawn", "temporarily_unavailable")
    assert band.last_grant_at is not None
    await handle.dispose()


@pytest.mark.asyncio
async def test_revoked_permission_stays_blocked_after_a_receipt(tmp_path: Path, mesh: Any) -> None:
    # A revoke is read from the placement document AHEAD of any receipt: a
    # verified grant recorded afterwards must not present access as usable.
    note_resolution(tmp_path, "openai", BrokerError(code="owner_offline", owner_device=OWNER))
    session, handle = await _host(tmp_path)
    document = mesh.placement
    assert document is not None
    document.revoke("openai", "recipient", by=OWNER)

    note_resolution(tmp_path, "openai", _native())
    await _settle(handle)

    band = _band(session)
    assert band.effective_state == "withdrawn"
    assert band.state != "ok"
    await handle.dispose()


@pytest.mark.asyncio
async def test_lifecycle_withdrawal_is_republished_and_never_usable(
    tmp_path: Path, mesh: Any
) -> None:
    note_resolution(tmp_path, "openai", _native())
    session, handle = await _host(tmp_path)

    assert note_lifecycle_observation(tmp_path, "openai", "withdrawn", owner=OWNER) is not None
    await _settle(handle)

    assert _band(session).effective_state == "withdrawn"
    assert _band(session).state != "ok"
    await handle.dispose()


@pytest.mark.asyncio
async def test_another_providers_receipt_never_restates_this_band(
    tmp_path: Path, mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A GitHub (or any other provider's) success must never clear this
    # session's refusal, and a sibling root's receipt is a different device.
    note_resolution(tmp_path, "openai", BrokerError(code="owner_offline", owner_device=OWNER))
    session, handle = await _host(tmp_path)
    reads: list[str] = []
    real = handle._schedule_model_access_publish

    def counted() -> Any:
        reads.append("read")
        return real()

    monkeypatch.setattr(handle, "_schedule_model_access_publish", counted)

    note_resolution(tmp_path, "github", BrokerError(code="owner_offline", owner_device=OWNER))
    other = tmp_path / "sibling"
    other.mkdir()
    note_resolution(other, "openai", _native())
    await _settle(handle)

    assert reads == []
    assert _band(session).effective_state == "owner_unavailable"
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_receipt_written_off_the_session_loop_still_republishes(
    tmp_path: Path, mesh: Any
) -> None:
    note_resolution(tmp_path, "openai", BrokerError(code="owner_offline", owner_device=OWNER))
    session, handle = await _host(tmp_path)

    writer = threading.Thread(target=note_resolution, args=(tmp_path, "openai", _native()))
    writer.start()
    writer.join(5)
    await _settle(handle)

    assert _band(session).effective_state == "effective_borrow"
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_disposed_host_publishes_nothing_and_unsubscribes(
    tmp_path: Path, mesh: Any
) -> None:
    note_resolution(tmp_path, "openai", BrokerError(code="owner_offline", owner_device=OWNER))
    session, handle = await _host(tmp_path)
    await handle.dispose()

    note_resolution(tmp_path, "openai", _native())
    await _settle(handle)

    assert handle._model_access_publish is None
    assert _band(session).effective_state == "owner_unavailable"


def test_a_failing_listener_never_fails_the_resolve(tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []

    def broken(root: str, provider: str) -> None:
        raise RuntimeError("listener bug")

    remove_broken = effective.add_listener(broken)
    remove_seen = effective.add_listener(lambda root, provider: seen.append((root, provider)))
    try:
        note_resolution(tmp_path, "openai", _native())
    finally:
        remove_broken()
        remove_seen()
    assert seen == [(str(tmp_path.resolve()), "openai")]
