"""GitHub and MCP logins ride the offline hold too, and still die at once on a panic.

THE DECISION (Damian, 2026-11-14, verbatim): GitHub and MCP logins — "Yes, GitHub and
MCP" — get the same 24 h offline hold as model logins (#3922). Before it they stopped
at the 15-minute grant window, so node sessions lost push, PRs and MCP servers while
the owner's Mac slept. Design: ``docs/design/mesh-credential-renewal.md`` §A.

WHAT MADE IT UNSAFE BEFORE, AND IS PINNED HERE: two scheduled DELETEs killed a lent
GitHub App installation token at the grant WINDOW end — the owner's lender and the
borrower's own belt. Holding the token past the window would have handed out a dead
token, so both now fire at the HOLD end, and the three causes that must end access
NOW (a panic, a share revoke, a lost membership) DELETE it on the borrower at once,
because the owner may be the device that is away.

Real path: two relays on loopback (the repo's ``mesh`` fixture), the owner relay
stopped the way a closed lid stops it, a loopback ``api.github.com`` whose revoke is
the only authority on liveness (``_FakeGithub``), and a real MCP OAuth row persisted
through ``McpTokenStorage``. The hold is scaled down (grant 2 s, hold 6 s) so the
window end and the hold end are both observable in bounded time. Synthetic ids only.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any

import pytest
import yaml

from local_operator.network import store
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import mcp_state
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.types import (
    BrokerError,
    CredentialRef,
    Grant,
    GrantScope,
)
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_github import (  # noqa: F401 — fixtures by import
    BORROWER_DEVICE,
    OWNER_DEVICE,
    SCRATCH,
    _ask_grant,
    _Env,
    _FakeGithub,
    _no_secret_broker_daemon,
    _seed_app_key,
    app_keypair,
    env,
    github_api,
    owner,
)
from tests.unit.network.test_credentials_held_offline import _stop_owner
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _lop_network,
    _pulled_or_held,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)

GRANT_TTL_S = 2.0
HOLD_S = 6.0
MCP_URL = "https://mcp.hold.test/mcp"
MCP_KEY = f"mcp:{MCP_URL}"


def _write_config(root: Any, *, repositories: list[str] | None = None) -> None:
    credentials: dict[str, Any] = {"grant_ttl_s": GRANT_TTL_S, "offline_hold_s": HOLD_S}
    if repositories is not None:
        credentials["github"] = {"repositories": list(repositories)}
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": credentials}}}), encoding="utf-8"
    )


@pytest.fixture
def held(
    mesh: Any,  # noqa: F811
    github_api: _FakeGithub,  # noqa: F811
    app_keypair: tuple[str, str],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Owner A lends GitHub (App arm) and one MCP login to B; B borrows each once."""
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    _write_config(mesh.a.root, repositories=[SCRATCH])
    _write_config(mesh.b.root)
    _seed_app_key(mesh.a.root, app_keypair[0])
    # The borrower's belt / immediate revoke reads the module's base at CALL time.
    monkeypatch.setattr(github_mod, "GITHUB_API_BASE", github_api.url)
    # ``raising=False`` so the cell runs (and fails on its assertion) on a base tree.
    monkeypatch.setattr(github_mod, "_REVOKED_NOW", {}, raising=False)

    auth = AuthStore(config_dir=mesh.a.root)
    try:
        asyncio.run(
            McpTokenStorage(MCP_URL, auth).set_tokens(
                OAuthToken.model_validate(
                    {
                        "access_token": "mcp-held-access",
                        "refresh_token": "mcp-held-refresh",
                        "expires_in": 3600,
                    }
                )
            )
        )
    finally:
        auth.close()
    for key in (github_mod.GITHUB_KEY, MCP_KEY):
        assert _lop_network("credential", "share", key, "--with", mesh.b.identity.name) == 0
        assert _pulled_or_held(mesh, key)
    # The owner relay's lender mints and revokes against the loopback GitHub. Its
    # broker exists once something is shared; the lender is built on the first mint.
    broker = owner_mod.broker_for_relay(mesh.a)
    assert broker is not None
    broker._github_minter = github_mod.GithubMinter(base_url=github_api.url)  # noqa: SLF001
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    gh_env, gh_token = github_mod.borrowed_git_env(client=client, root=mesh.b.root)
    assert gh_token.startswith("ghs_") and gh_env["GH_TOKEN"] == gh_token
    gh_grant = client.grants.peek(github_mod.GITHUB_KEY, "")
    assert isinstance(gh_grant, Grant) and gh_grant.hold_until_ms > gh_grant.grant_expires_at_ms
    mcp_grant = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(mcp_grant, Grant), mcp_grant
    assert mcp_grant.access_token == "mcp-held-access"
    try:
        yield type(
            "Held",
            (),
            {"client": client, "gh": gh_grant, "mcp": mcp_grant, "api": github_api, "mesh": mesh},
        )
    finally:
        # A belt scheduled at a hold end must not outlive this cell's loopback GitHub.
        cancel = getattr(github_mod, "cancel_pending_self_revokes", None)
        if cancel is not None:
            cancel()


def _past_window(grant: Grant) -> None:
    assert net_fixtures.wait_for(
        lambda: time.time() * 1000 > grant.grant_expires_at_ms + 200, timeout_s=10.0
    )


def test_github_and_mcp_are_served_through_the_hold_and_refused_after_it(held: Any) -> None:
    """The headline: owner away past the 15-min window → both still serve; past the hold → not.

    RED on c651774d: both came back ``owner_offline`` right after the window, and the
    borrower's belt had already DELETEd the App token at the window end.
    """
    client, net, api = held.client, held.mesh, held.api
    _stop_owner(net)
    _past_window(held.gh)

    gh = client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY)
    assert isinstance(gh, Grant), f"github lapsed at the grant window: {gh!r}"
    assert gh.held_offline is True and gh.access_token == held.gh.access_token
    # NEVER REVOKED AT THE WINDOW END WHILE HELD: neither the owner's lender nor the
    # borrower's belt fired, so the token the hold serves still authenticates.
    assert held.gh.access_token not in api.revoke_requests, "a window-end DELETE killed it"
    assert api.live(held.gh.access_token)
    # The real git path serves the same held token.
    _env, token = github_mod.borrowed_git_env(client=client, root=net.b.root)
    assert token == held.gh.access_token

    mcp = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(mcp, Grant), f"mcp lapsed at the grant window: {mcp!r}"
    assert mcp.held_offline is True and mcp.access_token == "mcp-held-access"
    # The durable evidence every MCP/GitHub status surface reads says "held" and until when.
    rows = mcp_state.load(net.network_id, net.b.root)
    for key, grant in ((MCP_KEY, mcp), (github_mod.GITHUB_KEY, gh)):
        assert rows[key]["state"] == "held", rows
        assert rows[key]["until"] == pytest.approx(grant.held_until_ms / 1000.0)
        assert mcp_state.held_until_clause(rows[key]["until"]).endswith(" UTC")
    # The token's own expiry still caps the hold: never past the App token's hour.
    assert gh.held_until_ms <= held.gh.token_expires_at_ms

    # Past the hold: refused, and the belt DELETEd the App token at the HOLD end.
    assert net_fixtures.wait_for(
        lambda: time.time() * 1000 > gh.held_until_ms + 200, timeout_s=HOLD_S + 5
    )
    for key, provider in ((github_mod.GITHUB_KEY, github_mod.GITHUB_KEY), (MCP_KEY, "mcp-oauth")):
        after = client.request_grant_sync(key, provider=provider)
        assert isinstance(after, BrokerError) and after.code == "owner_offline", after
    assert net_fixtures.wait_for(lambda: not api.live(held.gh.access_token), timeout_s=5.0)
    assert api.revoke_requests.count(held.gh.access_token) == 1, api.revoke_requests
    rows = mcp_state.load(net.network_id, net.b.root)
    assert rows[MCP_KEY]["state"] == "unavailable", rows


def test_a_panic_ends_the_github_hold_and_deletes_the_token_at_once(held: Any) -> None:
    """Panic while held: refused, and the BORROWER DELETEs the App token (owner is away)."""
    client, net, api = held.client, held.mesh, held.api
    _stop_owner(net)
    _past_window(held.gh)
    assert isinstance(
        client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY), Grant
    ), "precondition: the hold serves"
    held_mcp = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(held_mcp, Grant) and held_mcp.held_offline
    rows = mcp_state.load(net.network_id, net.b.root)
    assert {rows[k]["state"] for k in (github_mod.GITHUB_KEY, MCP_KEY)} == {"held"}, rows
    with store.mutate(net.network_id, net.b.root) as copy:
        copy.trust = "untrusted"
        store.save(copy, net.b.root)
    refused = client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY)
    assert isinstance(refused, BrokerError), "a panicked node served a held GitHub token"
    # #3947 m2: the panic ends EVERY held reading on disk at once, not at ``until`` —
    # the MCP row too, though nothing has read the MCP key since. (The GitHub row was
    # already replaced by the refused dial's own ``unavailable`` evidence.)
    rows = mcp_state.load(net.network_id, net.b.root)
    for key in (github_mod.GITHUB_KEY, MCP_KEY):
        assert rows[key]["state"] != "held" and "until" not in rows[key], rows
    assert rows[MCP_KEY]["state"] == "revoked", rows
    assert net_fixtures.wait_for(lambda: not api.live(held.gh.access_token), timeout_s=5.0)
    assert api.revoke_requests.count(held.gh.access_token) == 1, api.revoke_requests
    # Long before the hold end: this was the panic's DELETE, not the belt's.
    assert time.time() * 1000 < held.gh.hold_until_ms
    mcp = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(mcp, BrokerError), "a panicked node served a held MCP bearer"


def test_a_share_revoke_ends_the_github_hold_and_deletes_the_token_at_once(held: Any) -> None:
    """The pull's withdrawal while the owner is away: the placement change kills it now.

    The owner's revoke reaches a node as its relay's pull rewriting the placement file
    (#3922 Q2); with the owner relay stopped, that rewrite is applied here directly.
    """
    client, net, api = held.client, held.mesh, held.api
    _stop_owner(net)
    _past_window(held.gh)
    assert isinstance(
        client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY), Grant
    )
    document = placement_mod.PlacementDocument.load(
        net.network_id, net.b.root, self_device=net.borrower
    )
    document.revoke(github_mod.GITHUB_KEY, net.borrower, by=net.owner)
    document.save()
    refused = client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY)
    assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
    assert net_fixtures.wait_for(lambda: not api.live(held.gh.access_token), timeout_s=5.0)
    assert api.revoke_requests.count(held.gh.access_token) == 1, api.revoke_requests
    assert time.time() * 1000 < held.gh.hold_until_ms
    # #3947 m2: the git push clause stops saying "held until" the moment it ends.
    row = mcp_state.load(net.network_id, net.b.root)[github_mod.GITHUB_KEY]
    assert row["state"] == "revoked" and row["code"] == "not_a_holder", row
    from local_operator.network import readiness

    assert "held" not in readiness.github_fact(net.b.root)


def test_an_unrelated_placement_edit_does_not_kill_the_held_token(held: Any) -> None:
    """Negative control for the withdrawal hook: only a GitHub withdrawal DELETEs."""
    client, net, api = held.client, held.mesh, held.api
    document = placement_mod.PlacementDocument.load(
        net.network_id, net.b.root, self_device=net.borrower
    )
    document.revoke(MCP_KEY, net.borrower, by=net.owner)
    document.save()
    assert client.placement is not None  # the re-read that clears the cache
    time.sleep(0.5)
    assert held.gh.access_token not in api.revoke_requests and api.live(held.gh.access_token)


def test_a_lost_membership_ends_the_github_hold_and_deletes_the_token_at_once(
    held: Any,
) -> None:
    client, net, api = held.client, held.mesh, held.api
    _stop_owner(net)
    _past_window(held.gh)
    with store.mutate(net.network_id, net.b.root) as copy:
        me = copy.self_member()
        assert me is not None
        me.removed_at = time.time()
        store.save(copy, net.b.root)
    refused = client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY)
    assert isinstance(refused, BrokerError), refused
    assert net_fixtures.wait_for(lambda: not api.live(held.gh.access_token), timeout_s=5.0)
    assert api.revoke_requests.count(held.gh.access_token) == 1, api.revoke_requests


def test_a_forced_refresh_of_github_is_never_served_from_the_hold(held: Any) -> None:
    """Negative control: the hold bridges an outage; it never answers a forced refresh."""
    client, net = held.client, held.mesh
    _stop_owner(net)
    _past_window(held.gh)
    forced = client.request_grant_sync(
        github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY, force_refresh=True
    )
    assert isinstance(forced, BrokerError) and forced.code == "owner_offline", forced


def test_an_mcp_bearer_the_server_refused_is_not_held_again(held: Any) -> None:
    """A 401 on a held bearer ends it: driven through the real ``BrokeredBearerAuth``.

    Review #3947 m3: the server is an ``httpx2`` mock that refuses everything with a
    401, and the connect goes through the auth flow MCP really uses. The held bearer
    is sent once; the flow reports, drops the slot and re-asks, the owner is away and
    the hold has nothing left to serve, so neither the retry nor a later request
    carries a bearer at all.
    """
    import httpx2

    from local_operator.network.credentials.mcp_bearer import BrokeredBearerAuth

    client, net = held.client, held.mesh
    _stop_owner(net)
    _past_window(held.mcp)
    sent: list[str] = []

    def refuse(request: Any) -> Any:
        sent.append(request.headers.get("Authorization", "<none>"))
        return httpx2.Response(401)

    auth = BrokeredBearerAuth(url=MCP_URL, key=MCP_KEY, client=client)

    async def connect_twice() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(refuse), auth=auth) as http:
            await http.get(MCP_URL)
            await http.get(MCP_URL)

    asyncio.run(connect_twice())
    assert sent == ["Bearer mcp-held-access", "<none>"], sent
    assert client.grants.peek(MCP_KEY, "") is None
    again = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(again, BrokerError) and again.code == "owner_offline", again


def test_member_rm_on_the_owner_revokes_the_removed_nodes_app_token(held: Any) -> None:
    """#3947 M2: ``member rm`` DELETEs the removed node's App token at once, on the OWNER.

    The removed node is the one that cannot be trusted to revoke it itself, and since
    the lender's scheduled DELETE moved to the hold end (the token's hour) nothing
    else would end it inside that hour. Owner relay up; the node is not asked.
    """
    net, api = held.mesh, held.api
    token = held.gh.access_token
    assert api.live(token)
    net.a._ctl_member_rm({"network": net.network_id, "device_id": net.borrower})  # noqa: SLF001
    assert net_fixtures.wait_for(lambda: not api.live(token), timeout_s=10.0)
    assert token in api.revoke_requests
    assert time.time() * 1000 < held.gh.hold_until_ms, "that was the hold-end belt, not rm"
    broker = owner_mod.broker_for_relay(net.a)
    assert broker is not None
    assert broker._lender().outstanding(holder=net.borrower) == 0  # noqa: SLF001


def test_a_node_that_leaves_has_its_app_token_revoked_by_the_owner(held: Any) -> None:
    """#3947 M2's sibling: a ``disconnect`` (``net_leave``) ends the leaver's App token too."""
    net, api = held.mesh, held.api
    token = held.gh.access_token
    net.b._ctl_disconnect({"network": net.network_id})  # noqa: SLF001
    assert net_fixtures.wait_for(lambda: not api.live(token), timeout_s=10.0)
    assert token in api.revoke_requests
    # The scaled hold is 6 s, so a DELETE after it could be the belt: only one
    # before the hold end is the owner's revoke on the leave.
    assert time.time() * 1000 < held.gh.hold_until_ms, "that was the hold-end belt"


def test_removal_revoke_is_scoped_to_its_network_and_devices(
    owner: Any, github_api: _FakeGithub  # noqa: F811
) -> None:
    """The owner-side primitive: only the named devices, only the broker's network."""
    from types import SimpleNamespace

    _write_config(owner.root, repositories=[SCRATCH])
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    server = SimpleNamespace(**{owner_mod._BROKER_ATTR: owner.broker})  # noqa: SLF001
    assert owner_mod.revoke_lent_on_removal(server, "n_other", [BORROWER_DEVICE]) == 0
    assert owner_mod.revoke_lent_on_removal(server, "n_gh", ["d_" + "9" * 32]) == 0
    assert github_api.revoke_requests == [] and github_api.live(token)
    assert owner_mod.revoke_lent_on_removal(server, "n_gh", [BORROWER_DEVICE]) == 1
    assert github_api.revoke_requests == [token] and not github_api.live(token)


def test_an_owner_restriction_ends_a_held_github_token_at_once(held: Any) -> None:
    """#3947 m1 on the reachable path: a ``consent_withdrawn`` answer heard before the
    outage refuses the held serve AND DELETEs the token, like a revoke."""
    client, net, api = held.client, held.mesh, held.api
    client.state.observe(
        github_mod.GITHUB_KEY, "consent_withdrawn", owner_device=net.owner, retry_after_ms=1
    )
    client.state.save()
    _stop_owner(net)
    _past_window(held.gh)
    refused = client.request_grant_sync(github_mod.GITHUB_KEY, provider=github_mod.GITHUB_KEY)
    assert isinstance(refused, BrokerError) and refused.code == "consent_withdrawn", refused
    assert net_fixtures.wait_for(lambda: not api.live(held.gh.access_token), timeout_s=5.0)
    assert time.time() * 1000 < held.gh.hold_until_ms, "that was the hold-end belt"


@pytest.mark.parametrize("verdict", ["consent_withdrawn", "consent_unverifiable"])
def test_the_hot_read_revokes_on_every_admission_verdict(
    held: Any, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    """#3947 m1: ``_serve_cached`` follows ``_held_or``'s rule — any verdict DELETEs.

    White-box on purpose: GitHub serves through ``live_grant`` today and never reaches
    the hot read, so the rule is pinned at the method, with a reuse stamp that makes
    the read young enough to reach the shared admission fence.
    """
    client, api = held.client, held.api
    # A window long enough that the cache's re-ask margin does not hide the slot.
    gh = dataclasses.replace(held.gh, grant_expires_at_ms=int(time.time() * 1000) + 600_000)
    client.grants.put(github_mod.GITHUB_KEY, "sess-hot", gh, lifecycle_fence="f", selection={})
    monkeypatch.setattr(type(client), "_lifecycle_fence", lambda self, key: "f")
    monkeypatch.setattr(
        type(client), "_hold_admission", lambda self, policy, document, key: (verdict, "")
    )
    assert (
        client._serve_cached(  # noqa: SLF001
            github_mod.GITHUB_KEY, "sess-hot", model_id="", selection={}
        )
        is None
    )
    assert client.grants.peek(github_mod.GITHUB_KEY, "sess-hot") is None
    assert net_fixtures.wait_for(lambda: not api.live(gh.access_token), timeout_s=5.0)


def test_a_personal_token_hold_can_be_capped_alone(tmp_path: Any) -> None:
    """The PAT/``gh`` cap: defaults to the hold (Damian, "Full 24 h for personal tokens
    too"), can only shorten it, and a garbled value reads as the hold."""
    from local_operator.network.credentials import github_token_hold_s, offline_hold_s

    root = tmp_path
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": {}}}}), encoding="utf-8"
    )
    assert github_token_hold_s(root) == offline_hold_s(root) == 86400.0
    for value, expected in ((3600, 3600.0), (10**9, 86400.0), (1, 900.0), ("x", 86400.0)):
        (root / "config.yml").write_text(
            yaml.safe_dump(
                {"values": {"network": {"credentials": {"github_token_hold_s": value}}}}
            ),
            encoding="utf-8",
        )
        assert github_token_hold_s(root) == expected, value


# -- the owner's lender: the scheduled DELETE moved to the hold end -------------


def test_the_owner_lender_revokes_at_the_hold_end_not_the_window_end(
    owner: Any, github_api: _FakeGithub  # noqa: F811
) -> None:
    """RED on c651774d: the lender's DELETE fired at ``grant_expires_at_ms``."""
    _write_config(owner.root, repositories=[SCRATCH])
    detail = _ask_grant(owner, BORROWER_DEVICE)
    token = detail["access_token"]
    grant_end = detail["grant_expires_at_ms"] / 1000.0
    hold_end = detail["hold_until_ms"] / 1000.0
    assert hold_end - grant_end >= HOLD_S - GRANT_TTL_S - 1
    assert net_fixtures.wait_for(lambda: time.time() > grant_end + 0.5, timeout_s=10.0)
    assert github_api.revoke_requests == [], "the owner revoked at the grant window end"
    assert github_api.live(token)
    assert net_fixtures.wait_for(lambda: not github_api.live(token), timeout_s=HOLD_S + 5)
    assert github_api.revoke_requests == [token]
    assert time.time() >= hold_end - 0.1


def test_owner_share_revoke_still_deletes_the_held_token_at_once(
    owner: Any, github_api: _FakeGithub  # noqa: F811
) -> None:
    _write_config(owner.root, repositories=[SCRATCH])
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    assert owner.broker.revoke_outstanding(github_mod.GITHUB_KEY, BORROWER_DEVICE) == 1
    assert github_api.revoke_requests == [token] and not github_api.live(token)


# -- units: which tokens are revocable, and when the belt fires -----------------


def _gh_grant(*, token_exp_ms: int, hold_until_ms: int = 0, held_until_ms: int = 0) -> Grant:
    now_ms = int(time.time() * 1000)
    return Grant(
        access_token="ghs_unit",
        kind="bearer",
        token_expires_at_ms=token_exp_ms,
        grant_expires_at_ms=now_ms + 2_000,
        hold_until_ms=hold_until_ms,
        held_until_ms=held_until_ms,
        credential_ref=CredentialRef(
            owner_device=OWNER_DEVICE,
            owner_device_name="owner-laptop",
            provider=github_mod.GITHUB_KEY,
            kind=github_mod.GITHUB_KIND,
            credential_id=0,
        ),
        served_by=OWNER_DEVICE,
        scope=GrantScope(kind="device"),
    )


def test_only_an_installation_token_is_revocable() -> None:
    """PAT and ``gh`` arms send ``token_expires_at_ms = 0``: no DELETE is aimed at them."""
    assert github_mod.is_installation_grant(_gh_grant(token_exp_ms=int(time.time() * 1000) + 1))
    assert not github_mod.is_installation_grant(_gh_grant(token_exp_ms=0))
    assert github_mod.revoke_now([_gh_grant(token_exp_ms=0)]) == []


def test_the_belt_fires_at_the_hold_end(tmp_path: Any) -> None:
    now_ms = int(time.time() * 1000)
    _write_config(tmp_path)
    # A held serve carries the end it was served under.
    held = _gh_grant(token_exp_ms=now_ms + 3_600_000, held_until_ms=now_ms + 5_000)
    assert github_mod.belt_revoke_at_ms(held, tmp_path) == now_ms + 5_000
    # A fresh, stamped grant: the owner's hold, capped by the token and this device.
    fresh = _gh_grant(token_exp_ms=now_ms + 3_600_000, hold_until_ms=now_ms + 4_000)
    from local_operator.network.credentials.client import GrantCache

    GrantCache().put(github_mod.GITHUB_KEY, "", fresh)  # stamps the receipt
    assert github_mod.belt_revoke_at_ms(fresh, tmp_path) == now_ms + 4_000
    # An older owner that states no hold: today's window end, unchanged.
    old = _gh_grant(token_exp_ms=now_ms + 3_600_000)
    assert github_mod.belt_revoke_at_ms(old, tmp_path) == old.grant_expires_at_ms


# -- the MCP status surface (``/mcp``, ``lop network mcp credentials``) -------------


def _status_row(evidence: dict[str, Any]) -> dict[str, Any]:
    from types import SimpleNamespace

    from local_operator.mcp import credential_status

    return credential_status._oauth_row(  # noqa: SLF001
        "linear",
        url=MCP_URL,
        key=MCP_KEY,
        peer_name="node2",
        placement={"owner_device": OWNER_DEVICE, "owner_device_name": "owner-laptop"},
        holder=True,
        # A stand-in for ``_GrantIndex``: the two reads ``_oauth_row`` makes.
        grant=SimpleNamespace(  # type: ignore[arg-type]
            state=lambda _url: "absent", dead=lambda _url: False
        ),
        evidence=evidence,
        origin_peer=False,
        mesh_gone=False,
        refs=[],
    )


def test_the_mcp_status_row_says_held_until_and_lapses_to_unreachable() -> None:
    until = time.time() + 1800
    evidence = {
        "state": "held",
        "code": "owner_offline",
        "owner_device": OWNER_DEVICE,
        "owner_name": "owner-laptop",
        "at": time.time(),
        "until": until,
    }
    row = _status_row(evidence)
    hhmm = time.strftime("%H:%M UTC", time.gmtime(until))
    assert row["state"] == "synced", row
    assert row["reason"] == (
        f"held from owner-laptop until {hhmm} — lent before it went offline; nothing is "
        "renewed until it is back"
    )
    lapsed = _status_row({**evidence, "until": time.time() - 1})
    assert lapsed["state"] == "not_yet_synced", lapsed
    assert "held from" not in lapsed["reason"]
    # Same reading as a plain "owner unreachable" row: one remedy, not two.
    assert lapsed["reason"] == _status_row({**evidence, "state": "unavailable"})["reason"]


def test_held_is_in_the_closed_vocabulary_and_survives_a_load(tmp_path: Any) -> None:
    """``load`` drops a state outside ``STATES``; ``held`` must round-trip with its end."""
    assert "held" in mcp_state.STATES
    until = time.time() + 60
    mcp_state.record("n_x", MCP_KEY, state="held", code="owner_offline", until=until, root=tmp_path)
    row = mcp_state.load("n_x", tmp_path)[MCP_KEY]
    assert row["state"] == "held" and row["until"] == pytest.approx(until)
    mcp_state.record("n_x", MCP_KEY, state="unavailable", code="owner_offline", root=tmp_path)
    assert "until" not in mcp_state.load("n_x", tmp_path)[MCP_KEY]


def test_the_owner_states_the_personal_token_cap_on_the_grant(
    owner: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The cap reaches the wire: a PAT/``gh`` grant's ``hold_until_ms`` honours it."""
    (owner.root / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": {"github_token_hold_s": 1800}}}}),
        encoding="utf-8",
    )
    before = int(time.time() * 1000)
    grant = owner.broker._grant_from_owner_token(  # noqa: SLF001
        token="ghp_SYNTHETIC_cap", key=github_mod.GITHUB_KEY, provider="", common={}
    )
    after = int(time.time() * 1000)
    assert before + 1_800_000 <= grant.hold_until_ms <= after + 1_800_000
    assert not github_mod.is_installation_grant(grant)


def test_the_revoke_receipt_names_the_hold_bound_and_the_github_remedy(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3947 M1: a revoke stops a node that HEARS it; a lost one keeps a PAT for the hold."""
    from local_operator.network import cli as network_cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    lines = "\n".join(
        network_cli._github_revoke_lines("node2", 0, github_mod.SOURCE_TOKEN)  # noqa: SLF001
    )
    assert "as soon as it hears this revoke" in lines
    assert "(lost, stolen or offline) it may keep using it for up to 24 h" in lines
    assert "revoke the PAT at GitHub" in lines
    assert "window end" not in lines
    payload = network_cli._github_revocation_payload(  # noqa: SLF001
        0, 900, github_mod.SOURCE_TOKEN
    )
    assert payload["lent_grant_max_s"] == 86400 and payload["grant_window_s"] == 900
    app = network_cli._github_revocation_payload(1, 900, github_mod.SOURCE_APP)  # noqa: SLF001
    assert app["lent_grant_max_s"] == 3600
    assert "window end" not in "\n".join(
        network_cli._github_revoke_lines("node2", None, github_mod.SOURCE_APP)  # noqa: SLF001
    )
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": {"github_token_hold_s": 7200}}}}),
        encoding="utf-8",
    )
    capped = "\n".join(
        network_cli._github_revoke_lines("node2", 0, github_mod.SOURCE_GH)  # noqa: SLF001
    )
    assert "for up to 2 h" in capped
