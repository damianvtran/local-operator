"""Round-one regressions: marker authority, early-wait budget and short grants.

The fence/peer cases are assertion-bearing adaptations of the reviewer's probes
on PR #3330, issuecomment-6592160355. Real loopback requests and flock ownership
matter here: a mocked immediate failure cannot reproduce the validity race.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any

import httpx
import pytest

from local_operator.mcp import auth as auth_mod
from tests.unit.mcp.rotating_oauth_server import FakeTokenEndpoint
from tests.unit.mcp.test_oauth_inflight_refresh_lead import (
    _connected_provider,
    _resource_request,
)
from tests.unit.mcp.test_oauth_refresh_durability import (  # noqa: F401
    CLIENT_ID,
    SERVER_URL,
    _cfg,
    _endpoints_for,
    _isolated_lock_dir,
)


def _expire_at_fence(monkeypatch: pytest.MonkeyPatch, provider: Any) -> None:
    coordinate = provider._coordinate_inflight_refresh

    async def expire() -> None:
        await coordinate()
        provider.context.token_expiry_time = time.time() - 1

    monkeypatch.setattr(provider, "_coordinate_inflight_refresh", expire)


@pytest.mark.asyncio
async def test_r1_stale_fence_preserves_marker_and_family(monkeypatch: pytest.MonkeyPatch) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, store = await _connected_provider(endpoint, remaining=120)
        endpoints = _endpoints_for(endpoint.token_endpoint)
        assert (
            await auth_mod._refresh_oauth_token_locked(SERVER_URL, storage, endpoints)
            == "refreshed"
        )
        storage.mark_send_unconfirmed("refresh-1")
        # A peer sent refresh-1 and lost its response: the server is at refresh-2
        # but persistence still holds refresh-1. This is the marker's purpose.
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint.token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": "refresh-1",
                    "client_id": CLIENT_ID,
                },
            )
            assert response.status_code == 200
        marker = store.rows[0].data[auth_mod.GRANT_UNCONFIRMED_SEND_KEY].copy()
        _expire_at_fence(monkeypatch, provider)
        with pytest.raises(auth_mod.McpRefreshContendedError):
            await _resource_request(provider)
        auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert store.rows[0].data.get(auth_mod.GRANT_UNCONFIRMED_SEND_KEY) == marker
        store.rows[0].data["tokens_obtained_at"] = time.time() - 4000
        peer = auth_mod.build_oauth_provider(
            SERVER_URL, _cfg(), store=store, interactive=False, endpoints=endpoints
        )
        with pytest.raises(auth_mod.McpRefreshUnconfirmedError):
            await _resource_request(peer)
        auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert len(endpoint.requests) == 2
        assert endpoint.reuse_attempts == 0
        assert not endpoint.revoked


@pytest.mark.asyncio
async def test_r1_inflight_peer_marker_is_not_an_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with FakeTokenEndpoint(response_delay_s=0.4) as endpoint:
        provider, storage, _ = await _connected_provider(endpoint, remaining=120)

        async def peer() -> Any:
            async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
                return await auth_mod._refresh_oauth_token_locked(
                    SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint), lock=lock
                )

        task = asyncio.create_task(peer())
        try:
            await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
            _expire_at_fence(monkeypatch, provider)
            with pytest.raises(auth_mod.McpRefreshContendedError):
                await _resource_request(provider)
        finally:
            result = await task
            auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert result == "refreshed"
        assert (await storage.get_tokens()).refresh_token == "refresh-1"
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0
        assert not endpoint.revoked


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [45, 1])
async def test_r2_held_lock_does_not_spend_access_validity(remaining: float) -> None:
    import fcntl

    async with FakeTokenEndpoint() as endpoint:
        provider, storage, store = await _connected_provider(endpoint, remaining=remaining)
        # A normal one-hour grant must enter the full 60s lead at 45s left;
        # the helper's 60s grant would now be outside its capped 30s lead.
        tokens = await storage.get_tokens()
        assert tokens is not None
        tokens.expires_in = 3600
        await storage.set_tokens(tokens)
        store.rows[0].data["tokens_obtained_at"] = time.time() + remaining - 3600
        provider.context.current_tokens = tokens
        provider.context.token_expiry_time = storage.stored_token_expiry()
        path = auth_mod._oauth_refresh_lock_path(SERVER_URL)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)

        async def release() -> None:
            await asyncio.sleep(1.5)
            fcntl.flock(fd, fcntl.LOCK_UN)

        releasing = asyncio.create_task(release())
        start = time.monotonic()
        try:
            headers = await _resource_request(provider)
            elapsed = time.monotonic() - start
        finally:
            await releasing
            os.close(fd)
        assert headers == ["Bearer access-0"]
        assert elapsed < 1.0, "early refresh must not wait for the held flock"
        assert provider.context.current_tokens.refresh_token == "refresh-0"
        assert endpoint.requests == []


@pytest.mark.asyncio
async def test_r2_slow_exchange_detaches_before_access_expires() -> None:
    async with FakeTokenEndpoint(response_delay_s=1.5) as endpoint:
        provider, storage, _ = await _connected_provider(endpoint, remaining=1)
        start = time.monotonic()
        try:
            headers = await _resource_request(provider)
            elapsed = time.monotonic() - start
            assert headers == ["Bearer access-0"]
            assert elapsed < 0.8
            assert provider.context.current_tokens.refresh_token == "refresh-0"
        finally:
            # Do not abandon our detached exchange or socket at test teardown:
            # it must keep its lock and persist the eventual rotation.
            await auth_mod.drain_refresh_exchanges(timeout_s=5)
        assert len(endpoint.requests) == 1
        assert (await storage.get_tokens()).refresh_token == "refresh-1"
        assert not storage.send_unconfirmed()
        assert endpoint.reuse_attempts == 0


class _ShortLived(FakeTokenEndpoint):
    def _decide(self, form: dict[str, str]) -> tuple[int, dict[str, Any]]:
        status, payload = super()._decide(form)
        if status == 200:
            payload["expires_in"] = 30
        return status, payload


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_r3_short_lived_tokens_do_not_refresh_each_request(concurrent: bool) -> None:
    async with _ShortLived() as endpoint:
        provider, _, _ = await _connected_provider(endpoint, remaining=10)
        if concurrent:
            headers = await asyncio.gather(_resource_request(provider), _resource_request(provider))
        else:
            headers = [await _resource_request(provider) for _ in range(4)]
        assert headers == [["Bearer access-1"]] * (2 if concurrent else 4)
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_r3_concurrent_rotation_still_inside_effective_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _ShortLived() as endpoint:
        provider, _, _ = await _connected_provider(endpoint, remaining=10)
        # Model a peer's already-aged short grant. Both requests pass the gate
        # before the first await; even a rotation still inside the effective
        # lead must be recognised relative to that pre-wait snapshot.
        monkeypatch.setattr(
            provider._refresh_coord_storage, "stored_token_expiry", lambda: time.time() + 5
        )
        # This test is about adopting a peer's rotation, not the early-wait
        # bound: with the real 250ms budget the second request falls back to
        # the old token whenever it waits that long on the refresh lock (a
        # loaded CI runner), which is designed behaviour pinned separately by
        # test_r3_waiter_past_budget_sends_old_token_without_refreshing.
        monkeypatch.setattr(auth_mod, "_EARLY_REFRESH_WAIT_BUDGET_S", 30.0)
        headers = await asyncio.gather(_resource_request(provider), _resource_request(provider))
        assert headers == [["Bearer access-1"], ["Bearer access-1"]]
        assert len(endpoint.requests) == 1


@pytest.mark.asyncio
async def test_r3_waiter_past_budget_sends_old_token_without_refreshing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _ShortLived() as endpoint:
        provider, _, _ = await _connected_provider(endpoint, remaining=10)
        monkeypatch.setattr(
            provider._refresh_coord_storage, "stored_token_expiry", lambda: time.time() + 5
        )
        monkeypatch.setattr(auth_mod, "_EARLY_REFRESH_WAIT_BUDGET_S", 0.05)
        # A peer (same process, separate fd: flock conflicts) holds the refresh
        # lock for the whole request, so the wait can only end by the budget
        # expiring. No sleeps decide the outcome: the holder releases only
        # after the request has returned.
        async with auth_mod._oauth_refresh_lock(SERVER_URL):
            headers = await _resource_request(provider)
        assert headers == ["Bearer access-0"]
        # The waiter must not spend the grant behind the holder's back.
        assert endpoint.requests == []


@pytest.mark.asyncio
async def test_r4_fence_message_matches_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, _, _ = await _connected_provider(endpoint, remaining=120)
        _expire_at_fence(monkeypatch, provider)
        with pytest.raises(auth_mod.McpRefreshContendedError) as caught:
            await _resource_request(provider)
        ledger = auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert caught.value.reason_code == ledger == auth_mod.REFRESH_REFUSAL_UNATTRIBUTED
        assert str(caught.value) == str(
            auth_mod.McpRefreshContendedError(
                SERVER_URL, reason_code=auth_mod.REFRESH_REFUSAL_UNATTRIBUTED
            )
        )
        assert "another session holds" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["get_tokens", "stored_token_expiry"])
async def test_r5_outside_lead_guard_rejects_m9_mutation(
    monkeypatch: pytest.MonkeyPatch, read: str
) -> None:
    # Inject M9 at the provider boundary, then run the original negative
    # control. It must actually turn RED for either unsolicited storage read;
    # a guard installed on the seed instance would silently let both through.
    from tests.unit.mcp.test_oauth_inflight_refresh_lead import (
        test_outside_lead_does_not_lock_or_reread,
    )

    build = auth_mod.build_oauth_provider

    def mutated_provider(*args: Any, **kwargs: Any) -> Any:
        provider = build(*args, **kwargs)
        coordinate = provider._coordinate_inflight_refresh

        async def m9() -> None:
            if read == "get_tokens":
                await provider.context.storage.get_tokens()
            else:
                provider._refresh_coord_storage.stored_token_expiry()
            await coordinate()

        provider._coordinate_inflight_refresh = m9
        return provider

    monkeypatch.setattr(auth_mod, "build_oauth_provider", mutated_provider)
    with pytest.raises(pytest.fail.Exception, match="must not lock or re-read"):
        await test_outside_lead_does_not_lock_or_reread(monkeypatch)


def test_r6_refresh_documentation_is_reflowed() -> None:
    # These are intentionally source-format checks: R6 is comment-only and has
    # no runtime behavior to regress. Keep the affected prose readable without
    # coupling the test to executable statements or exact line numbers.
    source = inspect.getsource(auth_mod._make_refresh_coordinating_provider)
    assert "refreshes — so a\n    session can hold" not in source
    assert "# the moment any sibling process\n" not in source
