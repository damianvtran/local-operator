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

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import relay, store
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import owner as owner_mod
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
