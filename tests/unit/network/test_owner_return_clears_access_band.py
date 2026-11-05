"""An owner that goes away and comes BACK clears the borrower's access band.

Two real relays on loopback, mutually authenticated, real placement share and
pull, the owner's real ``AuthStore`` refreshing against a rotating loopback IdP
(see ``test_credentials_real_link``). The borrower's band is published by a real
``ServingSessionHandle``. The owner is taken down with ``RelayServer.stop`` and
brought back as a FRESH relay on the same root and port — the restarted-owner
shape — never the operator's device, services or network.

Before the fix the band went to ``owner_unavailable`` while the owner was gone
and stayed there, with the same "Last checked", after the owner returned and a
request was served again: only a model switch recomputed it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator.network import relay
from local_operator.network.credentials import effective
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.session.frontend_state import FrontendModelAccess
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _borrower_client,
    _lop_network,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)
from tests.unit.session.runtime.test_serving import _AccessSession

SELECTOR = f"{STUB_PROVIDER}/stub-model"


def _band(session: Any) -> FrontendModelAccess:
    claim = session._frontend_state_store.state.model_access
    assert claim is not None, "the host published no access claim"
    return claim


async def _settle(handle: ServingSessionHandle) -> None:
    for _ in range(3):
        await asyncio.sleep(0)
        publish = handle._model_access_publish
        if publish is not None:
            await publish


@pytest.mark.asyncio
async def test_owner_down_then_back_clears_the_band_and_keeps_one_refresh(
    mesh: Any,  # noqa: F811 — the imported fixture
) -> None:
    effective._receipts.clear()
    _share(mesh)
    _pull(mesh)
    client = _borrower_client(mesh)
    assert isinstance(client.request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant)
    posts_after_first = len(mesh.idp.posts)
    assert posts_after_first == 1

    session = _AccessSession(SELECTOR)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(mesh.b.root), config_dir=mesh.b.root
    )
    assert _band(session).effective_state == "effective_borrow"

    # OWNER GONE (lid closed): the next request is refused as owner_offline.
    fresh: relay.RelayServer | None = None
    try:
        port = mesh.a.settings.port  # what bind() actually took, and what B dials
        mesh.a.stop()
        assert net_fixtures.wait_for(lambda: not any(link.alive for link in mesh.b.links.values()))
        offline = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-2")
        assert isinstance(offline, BrokerError) and offline.code == "owner_offline", offline
        await _settle(handle)
        down = _band(session)
        assert (down.effective_state, down.state) == (
            "owner_unavailable",
            "temporarily_unavailable",
        )
        time.sleep(0.01)

        # OWNER BACK: a fresh relay on the same root and port; the borrower's
        # next request is served again over a NEW link.
        fresh = relay.RelayServer(
            root=mesh.a.root,
            settings=relay.NetworkSettings(port=port, listen_address="127.0.0.1"),
            identity=mesh.a.identity,
            audit=mesh.a.audit,
        )
        fresh.bind()
        fresh.bind_control()
        fresh.start()
        # A transport reconnect alone is NOT access: no resolve, no band change.
        assert _band(session).effective_state == ("owner_unavailable")
        back = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            back = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-3")
            if isinstance(back, Grant):
                break
            await asyncio.sleep(0.25)
        assert isinstance(back, Grant), back
        await _settle(handle)
        up = _band(session)
        assert (up.effective_state, up.state) == ("effective_borrow", "ok")
        assert up.observed_at is not None and down.observed_at is not None
        assert up.observed_at > down.observed_at, "Last checked must be the NEW check"
        # The owner's held grant was served, not refreshed again for the return.
        assert len(mesh.idp.posts) == posts_after_first

        # A REVOKE after the return stays blocked: the band is recomputed, not cleared.
        assert (
            _lop_network("credential", "revoke", STUB_PROVIDER, "--from", mesh.b.identity.name) == 0
        )
        refused = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-4")
        assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
        await _settle(handle)
        assert _band(session).state != "ok"
    finally:
        await handle.dispose()
        if fresh is not None:
            fresh.stop()
        effective._receipts.clear()
