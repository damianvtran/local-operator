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
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import effective
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.effective import (
    note_lifecycle_observation,
    note_resolution,
)
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.network import test_credentials_consent as owner_rig
from tests.unit.network.test_credentials_consent import (  # noqa: F401 — fixture by import
    rig,
)
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


class _ReceiptDuringRead:
    """Writes ONE receipt from another thread inside the host's claim read.

    That is the window review R1-1 / QA Q10 measured: a receipt that lands after
    the in-flight read has consumed the receipt store but before it applies its
    answer. Deterministic, because the write happens inside the read itself.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, root: Path, result: Any) -> None:
        from local_operator.providers import model_access

        self.reads = 0
        real = model_access.serving_model_access_here

        def read(selector: str, **kwargs: Any) -> Any:
            claim = real(selector, **kwargs)
            self.reads += 1
            if self.reads == 1:
                writer = threading.Thread(target=note_resolution, args=(root, "openai", result))
                writer.start()
                writer.join(5)
            return claim

        monkeypatch.setattr(model_access, "serving_model_access_here", read)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,during,expected",
    [
        # A newer DENIAL during a recovery's read: a wrong ``ok`` is the harmful one.
        (
            _native,
            lambda: BrokerError(code="owner_offline", owner_device=OWNER),
            "owner_unavailable",
        ),
        # A newer RECOVERY during a denial's read: the operator's own symptom.
        (
            lambda: BrokerError(code="owner_offline", owner_device=OWNER),
            _native,
            "effective_borrow",
        ),
    ],
)
async def test_the_latest_receipt_wins_over_a_read_already_in_flight(
    tmp_path: Path,
    mesh: Any,
    monkeypatch: pytest.MonkeyPatch,
    first: Any,
    during: Any,
    expected: str,
) -> None:
    session, handle = await _host(tmp_path)
    try:
        probe = _ReceiptDuringRead(monkeypatch, tmp_path, during())

        note_resolution(tmp_path, "openai", first())
        await _settle(handle)

        assert _band(session).effective_state == expected
        latest = effective._receipts[(effective._root(tmp_path), "openai")].state
        assert latest == expected
        # One re-read for the one receipt behind the read, not a loop.
        assert probe.reads == 2
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_receipts_after_the_reread_cap_still_reach_the_band(
    tmp_path: Path, mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A writer that lands a receipt inside EVERY read: one task stops at the cap,
    # applies its last answer, and hands the remainder to one fresh task, so the
    # newest receipt still wins and the work stays bounded.
    from local_operator.providers import model_access
    from local_operator.session.runtime import serving

    real = model_access.serving_model_access_here
    reads: list[int] = []
    flips = [_native, lambda: BrokerError(code="owner_offline", owner_device=OWNER)]

    def read(selector: str, **kwargs: Any) -> Any:
        claim = real(selector, **kwargs)
        reads.append(1)
        if len(reads) <= serving._MODEL_ACCESS_REREADS:
            result = flips[len(reads) % 2]()
            writer = threading.Thread(target=note_resolution, args=(tmp_path, "openai", result))
            writer.start()
            writer.join(5)
        return claim

    session, handle = await _host(tmp_path)
    try:
        monkeypatch.setattr(model_access, "serving_model_access_here", read)
        note_resolution(tmp_path, "openai", BrokerError(code="owner_offline", owner_device=OWNER))
        for _ in range(4):
            await _settle(handle)

        latest = effective._receipts[(effective._root(tmp_path), "openai")].state
        assert _band(session).effective_state == latest
        assert len(reads) == serving._MODEL_ACCESS_REREADS + 1
    finally:
        await handle.dispose()


def _owner_row(rig: Any, account: str) -> None:  # noqa: F811 — the imported fixture's name
    rig.auth.upsert_credential(
        owner_rig.P,
        {
            "type": "oauth",
            "email": f"{account}@example.invalid",
            "access": f"synthetic-access-{account}",
            "refresh": f"synthetic-refresh-{account}",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": account,
        },
    )


def _owner_reply(rig: Any, affinity: Any = None) -> dict[str, Any]:  # noqa: F811
    frame: dict[str, Any] = {
        "op": "net_broker",
        "kind": "grant",
        "req": 1,
        "key": owner_rig.P,
        "provider": owner_rig.P,
        "from_device": rig.peer.device_id,
        "for_session": "synthetic-session",
    }
    if affinity is not None:
        frame["affinity"] = affinity
    link = SimpleNamespace(device_id=rig.peer.device_id, network_id=owner_rig.N, epoch=1)
    return rig.broker.on_broker(link, frame)["detail"]


@pytest.mark.asyncio
async def test_a_denied_account_is_never_lent_and_a_pooled_one_is_provider_access(
    rig: Any,  # noqa: F811 — the owner rig, by import
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-4 / QA Q9b against the REAL owner broker's decision, not a hand-built grant.

    Receipts are keyed by provider because the OWNER picks the account per lend
    (``owner.grant``; a borrower's affinity only steers that choice) and re-checks
    the borrower's account-scoped denials against the lent identity at mint
    (``consent.serving_guard`` -> ``check_borrow``). A denied account is therefore
    never served: as the only row, or with the borrower anchored to it, the owner
    refuses. With a second, permitted row the owner lends THAT one, which is real
    provider access. The band says the provider is usable; it names no account.
    """
    rig.auth.delete_credentials_for_provider(owner_rig.P)
    _owner_row(rig, "account-one")
    owner_rig.consent.restrict_and_revoke(
        owner_rig.N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": owner_rig.P, "account_ref": "account-one"},
        "synthetic-account-withdrawal",
        rig.root,
    )
    with owner_rig.placement.mutate(owner_rig.N, rig.root, self_device=rig.own.device_id) as doc:
        doc.grant(owner_rig.P, rig.peer.device_id, by=rig.own.device_id)
    denied = rig.auth.list_credentials(owner_rig.P)[0]
    only_denied = _owner_reply(rig)
    _owner_row(rig, "account-two")
    anchored = _owner_reply(rig, {"account_id": denied.id, "account_created_at": denied.created_at})
    pooled = _owner_reply(rig)

    # The borrower: its own root and a placement naming the same owner, so the
    # owner's real replies are what the receipt and the band are computed from.
    borrower = tmp_path / "borrower"
    borrower.mkdir()
    document = owner_rig.placement.PlacementDocument(
        "n_borrower", root=borrower, written_by=rig.own.device_id
    )
    document.declare(
        owner_rig.P, owner_device=rig.own.device_id, provider=owner_rig.P, by=rig.own.device_id
    )
    document.grant(owner_rig.P, "recipient", scope="session", by=rig.own.device_id)
    source = MeshCredentialClient(
        root=borrower, self_device="recipient", network_id="n_borrower", placement=document
    )
    monkeypatch.setattr(MeshCredentialClient, "for_this_device", lambda *_a, **_k: source)
    session = _AccessSession(f"{owner_rig.P}/synthetic-model")
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(borrower), config_dir=borrower
    )
    try:
        for detail in (only_denied, anchored):
            assert detail.get("code") == "not_a_holder", detail
            assert "access_token" not in detail
            note_resolution(borrower, owner_rig.P, BrokerError.from_detail(detail))
            await _settle(handle)
            band = _band(session)
            assert (band.effective_state, band.state) == ("withdrawn", "temporarily_unavailable")

        assert pooled.get("kind") == "grant", pooled
        assert pooled["identity"]["account_id"] == "account-two"
        note_resolution(borrower, owner_rig.P, Grant.from_detail(pooled))
        await _settle(handle)
        band = _band(session)
        assert (band.effective_state, band.state) == ("effective_borrow", "ok")
        assert "account" not in band.model_dump_json()
    finally:
        await handle.dispose()
        source.close()


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
