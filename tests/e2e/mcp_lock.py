"""Rig that puts a real MCP connect inside the real OAuth refresh lock.

This is the load-bearing half of the ``/resume`` liveness test. To reproduce
#401 the app has to reach the genuine article — ``_oauth_refresh_lock`` in
``local_operator/mcp/auth.py``, taking a genuine ``flock`` on a genuine file —
because the deadlock is a property of the syscalls, not of the control flow. A
mocked lock cannot deadlock, so a test built on one would pass against the
broken code.

Getting there requires three conditions, all set up here:

1. **The lock must be contended**, or the acquire succeeds instantly and the
   cancellation window never opens. A foreign holder takes it first.
2. **The stored grant must be expired**, or ``ensure_mcp_oauth_fresh`` returns
   before it ever reaches the lock (a still-valid token is not refreshed).
3. **Endpoint discovery must not touch the network**, or the refresh fails on
   a DNS lookup well before the lock.

The holder is a SEPARATE open file description on the same path, not a second
``flock`` on the same fd. ``flock`` locks are held per open-file-description,
so re-locking the same descriptor would silently succeed and the rig would
quietly test nothing.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.unit.mcp.test_auth import FakeAuthStore

#: A URL that resolves nowhere. Nothing here may make a network call, and a
#: reserved-by-RFC ``.test`` TLD makes an accidental one fail fast and loudly
#: rather than reaching a real host.
SERVER_URL = "https://mcp.invalid.test/mcp"


class ManagedFakeAuthStore(FakeAuthStore):
    """The unit suite's in-memory store, plus the ``close`` the manager needs.

    ``McpManager``'s injected-store parameter is typed ``ManagedAuthStore`` —
    ``StructuralAuthStore`` plus a ``close`` it calls in ``disconnect_all``
    when it owns the store. The unit fake predates that surface and has no
    database handle to release. Subclassing to add the one missing method keeps
    the store's behaviour identical to the one the auth tests exercise, while
    satisfying the protocol honestly rather than widening a production type or
    silencing the type checker at the call site.
    """

    def close(self) -> None:
        """Nothing to release: this store is a list in memory."""


def candidate_lock_paths(config_dir: Path) -> list[Path]:
    """Every lock filename the code under test might use, across versions.

    #401 changed the lock from one global file to one file per server keyed by
    a URL digest. This test has to hold BOTH: the whole point of the stage is
    that it is run against the pre-fix code to prove it goes red, and a rig
    that only knew the post-fix filename would leave the pre-fix lock
    uncontended and pass against the very defect it exists to catch.

    Keep this list append-only for the same reason.
    """
    digest = hashlib.sha256(SERVER_URL.encode("utf-8")).hexdigest()[:16]
    return [
        config_dir / "mcp_oauth_refresh.lock",  # before #401: one global lock
        config_dir / f"mcp_oauth_refresh_{digest}.lock",  # after #401: per server
    ]


@contextmanager
def foreign_lock_holder(config_dir: Path) -> Iterator[list[Path]]:
    """Hold every candidate lock file for the duration of the block.

    Simulates the peer this lock exists to coordinate with: another
    local-operator process mid-refresh, or one that died holding the lock.
    Both are ordinary on a machine running several sessions, and both are what
    parks a connect long enough for a ``/resume`` to cancel it.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = candidate_lock_paths(config_dir)
    fds: list[int] = []
    try:
        for path in paths:
            fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            fds.append(fd)
        yield paths
    finally:
        for fd in fds:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def stub_endpoint_discovery(monkeypatch: Any) -> None:
    """Answer OAuth metadata discovery locally instead of over HTTP.

    Uses the module's OWN ``_fallback_endpoints_for`` — the shape the SDK
    synthesizes when discovery finds nothing — so the refresh path proceeds
    with a structurally real endpoint set rather than a hand-built stand-in
    that could drift from what the code expects.
    """
    from local_operator.mcp import auth as auth_mod

    async def _local_discovery(server_url: str) -> Any:
        return auth_mod._fallback_endpoints_for(server_url)

    monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", _local_discovery)


async def seed_expired_grant(store: Any) -> None:
    """Store a grant old enough that the proactive refresh actually runs.

    ``ensure_mcp_oauth_fresh`` returns early for a token still inside its
    lifetime, so without an EXPIRED one the connect never reaches the lock and
    the test silently exercises nothing. ``tokens_obtained_at`` is backdated
    because expiry is computed from it, mirroring
    ``tests/unit/mcp/test_auth.py``'s day-old-grant setup.
    """
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    from local_operator.mcp.auth import McpTokenStorage

    storage = McpTokenStorage(SERVER_URL, store)
    await storage.set_client_info(
        OAuthClientInformationFull(client_id="e2e-client", client_secret="e2e-secret")
    )
    # A refresh_token must be present: with none, the refresh is skipped.
    await storage.set_tokens(
        OAuthToken(access_token="expired-access", refresh_token="refresh", expires_in=1)
    )
    store.rows[0].data["tokens_obtained_at"] = time.time() - 86_400


def oauth_server_config() -> Any:
    """The server config that sends ``_connect_server`` down the OAuth path."""
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    return MCPHttpServerConfig(url=SERVER_URL, auth=MCPAuthConfig(type="oauth"))


async def parked_mcp_manager(config_dir: Path, cwd: Path, monkeypatch: Any) -> Any:
    """A real ``McpManager`` with a real connect parked inside the real lock.

    Returns the manager with one server deferred past the startup gate and its
    connect task sitting in ``_oauth_refresh_lock``'s acquire — the precise
    state a TUI is in when a user types ``/resume`` while MCP servers are
    still coming up, which is the state #401 deadlocked from.
    """
    from local_operator.mcp.manager import McpManager

    stub_endpoint_discovery(monkeypatch)
    store = ManagedFakeAuthStore()
    await seed_expired_grant(store)

    manager = McpManager(str(cwd), auth_store=store)
    # The real discovery round: the 250 ms gate passes, the OAuth server misses
    # it (it is parked in the lock), and the manager files a live continuation
    # task for it. That deferred task is what a dispose then cancels.
    await manager._connect_round({"e2e-oauth": oauth_server_config()}, {"e2e-oauth": "test"})
    return manager
