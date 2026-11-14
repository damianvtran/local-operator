"""``lop serve`` owns the Team Bridge hub connection pool (bridge-instant-load).

The lifespan binds one pool to the server loop and closes it at shutdown; the
pool's own invariants are pinned in ``tests/unit/team_bridge/test_http_pool.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from local_operator.team_bridge import http_pool


def test_the_serve_lifespan_owns_and_closes_the_pool(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from local_operator.server.app import app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")

    async def running_pool() -> http_pool.HubHttpPool | None:
        return http_pool._POOLS.get(asyncio.get_running_loop())

    with TestClient(app) as client:
        portal = client.portal
        assert portal is not None
        pool = portal.call(running_pool)
        assert pool is not None and not pool.closed, "lop serve must bind a hub pool"
        held = pool.client("https://hub.example/v1", None)
    assert pool.closed and held.is_closed, "shutdown must close the pool's clients"
