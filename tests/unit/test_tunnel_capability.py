"""The control plane may omit CF credentials only for a willing relay client."""

from contextlib import closing

import httpx
import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import config
from local_operator.tunnels.api import RadientTunnels


@pytest.mark.asyncio
@pytest.mark.parametrize("forced_cloudflare", [False, True])
@pytest.mark.parametrize("api_path", [None, "/v1/push/register"])
async def test_relay_capability_respects_transport_and_request_family(
    tmp_path, monkeypatch, forced_cloudflare, api_path
):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        config,
        "transport_override",
        lambda: config.TRANSPORT_CLOUDFLARE if forced_cloudflare else None,
    )
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct-1", "access": "synthetic"}
        )
    seen = []

    def receive(request):
        seen.append(request)
        # This is also an old-server control: an unaware server need not echo or
        # negotiate the header to keep returning its existing response shape.
        return httpx.Response(200, json={"result": {"cloudflared_token": "legacy"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
        result = await RadientTunnels(row.id, client).request(
            "POST", "/tunnel-1/connect", api_path=api_path
        )
    assert result == {"cloudflared_token": "legacy"}
    assert len(seen) == 1
    expected = "radient-tunnel.v1" if api_path is None and not forced_cloudflare else None
    assert seen[0].headers.get("X-Radient-Tunnel-Protocol") == expected
