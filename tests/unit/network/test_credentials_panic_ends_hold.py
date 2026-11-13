"""A panicked network stops serving credentials: held grants AND lent GitHub tokens.

THE DEFECT (lease review M1, 2026-11-13): ``lop network panic`` — raised here or
received from a peer — only latches ``record.trust = "untrusted"``. That closes
every link, so the borrower's next request is ``owner_offline``, which is exactly
the outcome the owner-unreachable hold bridges. The hold's admission checked
``member.active`` and never trust, so a panicked network went on serving the held
bearer until its lease ended. On the owner, ``GithubLender.revoke_all_holders``
existed but had no caller, so a lent GitHub token outlived the panic by its hour.

Pinned here, on the real two-relay ``mesh`` fixture where it matters:

* untrusted borrower → the held grant is refused and the slot dropped (RED on the
  pre-fix tree: it was served ``held_offline`` with the same bearer);
* an ACTIVE-trust borrower still bridges an owner outage (no regression);
* both relay panic paths (raised and received) ask the owner to revoke, and the
  owner revoke DELETEs every lent GitHub token for that network only.

Synthetic ids only; isolated roots come from the fixtures.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import relay, store
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials.mcp_bearer import BrokeredBearerAuth
from local_operator.network.credentials.types import BrokerError, Grant
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_github import (  # noqa: F401 — fixtures by import
    BORROWER_DEVICE,
    OWNER_DEVICE,
    _ask_grant,
    _FakeGithub,
    _no_secret_broker_daemon,
    app_keypair,
    env,
    github_api,
    owner,
)
from tests.unit.network.test_credentials_held_offline import (  # noqa: F401 — fixtures
    STUB_PROVIDER,
    _pair,
    _state,
    _stop_owner,
    borrowed,
    devices,
    idp,
    mesh,
    serve_shaped_relay,
    stub_provider,
)

OTHER_HOLDER = "d_00000000000000000000000000000033"


def _set_trust(mesh: Any, trust: str) -> None:  # noqa: F811 — the fixture's name
    for record in store.list_networks(mesh.b.root):
        relay.set_trust(record, trust=trust, reason="synthetic panic", root=mesh.b.root)
    assert all(r.trust == trust for r in store.list_networks(mesh.b.root))


def test_an_active_network_still_bridges_an_owner_outage(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """The control row: trust stays ``active``, so the hold serves as before."""
    client, grant = borrowed
    _stop_owner(mesh)
    assert all(r.trust == "active" for r in store.list_networks(mesh.b.root))
    held = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(held, Grant), held
    assert held.held_offline is True
    assert held.access_token == grant.access_token


def test_a_panicked_network_serves_no_held_grant(mesh: Any, borrowed: Any) -> None:  # noqa: F811
    """The ported probe: owner gone, trust latched ``untrusted`` → refused, slot dropped."""
    client, grant = borrowed
    _stop_owner(mesh)
    _set_trust(mesh, "untrusted")
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError), (
        f"a panicked network served {type(refused).__name__} "
        f"held_offline={getattr(refused, 'held_offline', None)}"
    )
    assert _state(mesh, client).usable is False
    # The slot is gone, not merely skipped: the hot read cannot resurrect it either.
    assert client.grants.get(STUB_PROVIDER, "sess-1") is None
    again = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(again, BrokerError), again
    assert grant.access_token not in repr(again)


def _cache_github(client: Any, grant: Grant, monkeypatch: pytest.MonkeyPatch) -> Grant:
    """A live GitHub slot on the borrower; ``should_borrow`` forced (no GitHub share)."""
    gh = dataclasses.replace(grant, access_token="ghs_SYNTHETIC_panic")
    client.grants.put(github_mod.GITHUB_KEY, "sess-gh", gh)
    monkeypatch.setattr(type(client), "should_borrow", lambda self, key: True)
    return gh


def test_an_active_network_still_serves_the_cached_github_grant(
    mesh: Any, borrowed: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Control for the GitHub gate: active trust keeps today's no-dial reuse."""
    client, grant = borrowed
    gh = _cache_github(client, grant, monkeypatch)
    _stop_owner(mesh)
    served = client.request_grant_sync(github_mod.GITHUB_KEY, session_id="sess-gh")
    assert isinstance(served, Grant) and served.access_token == gh.access_token


def test_a_panicked_network_serves_no_cached_github_grant(
    mesh: Any, borrowed: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Review M1 / QA Q-R1-1: the github gate read the cache past the trust fence.

    RED on 85e3519c: the same ``ghs_`` grant came back after the panic (and with
    the owner away nobody had revoked it at GitHub).
    """
    client, grant = borrowed
    gh = _cache_github(client, grant, monkeypatch)
    _stop_owner(mesh)
    _set_trust(mesh, "untrusted")
    refused = client.request_grant_sync(github_mod.GITHUB_KEY, session_id="sess-gh")
    assert isinstance(refused, BrokerError), f"served {type(refused).__name__} after panic"
    assert gh.access_token not in repr(refused)
    assert client.grants.peek(github_mod.GITHUB_KEY, "sess-gh") is None


def test_a_panicked_network_serves_no_cached_mcp_bearer(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """Review M1: ``_live_bearer`` is MCP's first read per request and had no check.

    An MCP OAuth bearer has no issuer revoke, so before the fix it was served for
    up to ``grant_ttl_s - REASK_MARGIN`` after the panic (RED on 85e3519c:
    ``'mcp_SYNTHETIC_panic'``).
    """
    client, grant = borrowed
    key = "mcp:panic.example"
    client.grants.put(key, "", dataclasses.replace(grant, access_token="mcp_SYNTHETIC_panic"))
    auth = BrokeredBearerAuth(url="https://panic.example/mcp", key=key, client=client)
    assert auth._live_bearer() == "mcp_SYNTHETIC_panic"  # control: trusted, served
    _stop_owner(mesh)
    _set_trust(mesh, "untrusted")
    assert auth._live_bearer() == ""
    assert client.grants.peek(key, "") is None


def test_both_panic_paths_ask_the_owner_to_revoke(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Raised (``_ctl_panic`` on A) and received (``_op_panic`` on B) both call it."""
    seen: list[tuple[str, str]] = []

    def spy(server: Any, network_id: str) -> None:
        seen.append((str(server.identity.device_id), network_id))

    monkeypatch.setattr(relay.RelayServer, "_revoke_lent_on_panic", spy)
    # The received half needs a live link for the alarm to travel on.
    assert net_fixtures.wait_for(lambda: any(link.alive for link in mesh.a.links.values()))
    mesh.a._ctl_panic({"network": mesh.network_id, "reason": "synthetic"})
    assert net_fixtures.wait_for(lambda: len(seen) >= 2), seen
    assert sorted(seen) == sorted(
        [
            (str(mesh.a.identity.device_id), mesh.network_id),
            (str(mesh.b.identity.device_id), mesh.network_id),
        ]
    )


def test_owner_panic_revokes_every_lent_github_token(
    owner: Any, github_api: _FakeGithub  # noqa: F811
) -> None:
    """Every holder's outstanding token is DELETEd, through the relay's own hook."""
    owner.document.grant(github_mod.GITHUB_KEY, OTHER_HOLDER, scope="device", by=OWNER_DEVICE)
    owner.document.save()
    first = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    second = _ask_grant(owner, OTHER_HOLDER)["access_token"]
    server = SimpleNamespace(**{owner_mod._BROKER_ATTR: owner.broker})

    # A panic in ANOTHER network touches nothing here.
    other = relay.RelayServer._revoke_lent_on_panic(server, "n_other")  # type: ignore[arg-type]
    other.join(timeout=10)
    assert github_api.revoke_requests == []

    thread = relay.RelayServer._revoke_lent_on_panic(server, "n_gh")  # type: ignore[arg-type]
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert sorted(github_api.revoke_requests) == sorted([first, second])
    assert not github_api.live(first) and not github_api.live(second)


def test_a_relay_that_never_brokered_revokes_nothing() -> None:
    """No broker on the relay → zero, and no broker is built inside the incident path."""
    server = SimpleNamespace()
    assert owner_mod.revoke_lent_on_panic(server, "n_gh") == 0
    assert not hasattr(server, owner_mod._BROKER_ATTR)


def test_a_revoke_thread_that_cannot_start_never_raises_into_the_panic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review n1: at interpreter shutdown ``Thread.start`` raises; the hook must not."""

    def refuse(self: Any) -> None:
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(relay.threading.Thread, "start", refuse)
    server = SimpleNamespace()
    thread = relay.RelayServer._revoke_lent_on_panic(server, "n_gh")  # type: ignore[arg-type]
    assert not thread.is_alive()
