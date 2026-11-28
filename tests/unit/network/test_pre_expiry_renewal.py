"""G10: the holder relay renews a lent provider/MCP access token before it expires.

THE STALL THIS FILE PINS (measured RED on main, n=2): a session borrows a key, the
owner stays linked for longer than one access-token life while that session makes no
request (a long tool call, a build), then the owner goes away. The session's held
grant ended at its token's expiry, nobody re-asked while the owner was up, so the
next turn found no runway and failed after the quiet wait. With G10 the holder's
relay re-asks, once per token life and just inside the owner's refresh skew, so the
owner's NEWEST access token is what the session holds when the owner leaves.

The rig is the real one: two relays on loopback (``mesh``), the real share/pull CLI,
a rotating loopback IdP that refuses a spent refresh token (so ``posts`` is a fact),
and the real brokered turn through ``MeshAwareAuthStore``. Knobs are shrunk keeping
the shipped ORDER (lead < owner skew < borrower re-ask margin): token life 8 s,
owner skew 3 s (lead 2.25 s), margin 4 s. The step is driven by
``renew_push.pre_expiry_tick`` (the exact callable the definitions syncer runs every
15 s) on a short loop here, because a 15 s tick cannot be waited on per cell.

Every binding security condition of the G10 approval has its own NEGATIVE cell
(``Q2``..``Q12`` in the names). Tokens are compared as booleans/digests only.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import effective
from local_operator.network.credentials import owner as owner_broker_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import renew_push
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.providers import auth_store as auth_store_mod
from local_operator.providers.failover import ProviderError
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credential_limit_skip import _NoLocalLogins
from tests.unit.network.test_credentials_held_offline import _stop_owner
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    BORROWER_DEVICE,
    BORROWER_TWO,
    STUB_PROVIDER,
    _ask_grant,
    _grant_frame,
    _Link,
    idp,
    owner,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _audit,
    _lop_network,
    _pull,
    _pulled_or_held,
    _share,
    mesh,
)
from tests.unit.network.test_owner_quiet_wait import SESSION, _bound, _Provider, _turn
from tests.unit.network.test_relay_e2e import devices  # noqa: F401

TOKEN_LIFE_S = 8
SKEW_MS = 3_000  # owner refresh skew (shipped 60 s); lead = 0.75 * skew = 2.25 s
MARGIN_MS = 4_000  # borrower re-ask margin (shipped 120 s); skew < margin as shipped
WAIT_S = 3.0  # PR-D quiet-wait bound (shipped grant_ttl_s/2)
IDLE_PAST_LIFE_S = 2.5
TICK_S = 0.2
#: The name each end of a share refuses a renewal ask with (``renew_push.renew_key``).
NAMED: dict[str, str] = {
    "unshare": "refused:not_a_holder",
    "panic": "refused:untrusted",
    "consent_withdrawn": "refused:consent_withdrawn",
    "member_rm": "refused:member_not_active",
}


def _dig(token: str | None) -> str | None:
    return hashlib.sha256(token.encode()).hexdigest()[:10] if token else None


@pytest.fixture()
def knobs(mesh: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    handler = mesh.idp._server.RequestHandlerClass  # noqa: SLF001
    real_reply = handler._reply  # noqa: SLF001

    def short_reply(self: Any, status: int, payload: dict[str, Any]) -> None:
        if "expires_in" in payload:
            payload = {**payload, "expires_in": TOKEN_LIFE_S}
        real_reply(self, status, payload)

    monkeypatch.setattr(handler, "_reply", short_reply)
    monkeypatch.setattr(auth_store_mod, "OAUTH_REFRESH_SKEW_MS", SKEW_MS)
    monkeypatch.setattr(client_mod, "REASK_MARGIN_MS", MARGIN_MS)
    from local_operator.mcp import auth as mcp_auth

    monkeypatch.setattr(mcp_auth, "REFRESH_SKEW_S", SKEW_MS / 1000)
    _bound(monkeypatch, WAIT_S)
    effective._receipts.clear()  # noqa: SLF001


class _OwnerFrames:
    """Every ``grant`` frame the OWNER's broker receives (``for_session``, flags)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.frames: list[dict[str, Any]] = []
        real = owner_broker_mod.MeshCredentialBroker.grant
        frames = self.frames

        def spy(broker: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
            frames.append(dict(frame))
            return real(broker, link, frame)

        monkeypatch.setattr(owner_broker_mod.MeshCredentialBroker, "grant", spy)

    def renew(self) -> list[dict[str, Any]]:
        return [f for f in self.frames if f.get("for_session") == renew_push.RENEW_SESSION]


@pytest.fixture()
def frames(monkeypatch: pytest.MonkeyPatch) -> _OwnerFrames:
    return _OwnerFrames(monkeypatch)


class _Ticker:
    """Runs the registered tick entry for the owner member on a short loop."""

    def __init__(self, mesh: Any) -> None:  # noqa: F811
        self._mesh = mesh
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="g10-ticker")

    def _run(self) -> None:
        tick = getattr(renew_push, "pre_expiry_tick", None)  # absent on main: a no-op
        while not self._stop.wait(TICK_S):
            if tick is not None:
                tick(self._mesh.b, self._mesh.owner)

    def __enter__(self) -> _Ticker:
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)


def _ticks(mesh: Any, n: int = 3) -> None:  # noqa: F811
    tick = getattr(renew_push, "pre_expiry_tick", None)
    for _ in range(n):
        if tick is not None:
            tick(mesh.b, mesh.owner)


def _setup(mesh: Any) -> tuple[MeshCredentialClient, MeshAwareAuthStore]:  # noqa: F811
    _share(mesh)
    _pull(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    return client, MeshAwareAuthStore(_NoLocalLogins(), mesh=client, config_dir=mesh.b.root)


async def _borrow_turn(store: MeshAwareAuthStore) -> list[str]:
    provider = _Provider()
    await asyncio.wait_for(_turn(store, provider), timeout=20.0)
    return provider.seen


def _owner_grant_rows(mesh: Any) -> list[dict[str, Any]]:  # noqa: F811
    return _audit(mesh.a, "credential.grant")


def _renew_rows(mesh: Any) -> list[dict[str, Any]]:  # noqa: F811
    return [
        r
        for r in _owner_grant_rows(mesh)
        if (r.get("session_id") or (r.get("detail") or {}).get("session_id"))
        == renew_push.RENEW_SESSION
    ]


def _holder_rows(mesh: Any, event: str) -> list[dict[str, Any]]:  # noqa: F811
    """``event`` rows in the HOLDER's audit file (``audit.write_event`` appends there)."""
    from local_operator.network.store import audit_path

    path = audit_path(mesh.b.root)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.get("event") == event]


def _files_text(root: Path) -> str:
    chunks: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            try:
                chunks.append(path.read_bytes().decode("utf-8", "replace"))
            except OSError:
                continue
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# R1 / R6 / Q4 / Q9: the provider stall, fixed; the lease cap holds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r1_provider_a_quiet_session_holds_the_newest_token_when_the_owner_leaves(
    mesh: Any, knobs: None, frames: _OwnerFrames, pre_expiry_renewal: Any  # noqa: F811
) -> None:
    """RED on main: ``token_changed`` False and the turn fails after the quiet wait."""
    client, store = _setup(mesh)
    first = await _borrow_turn(store)
    # A second live session borrows the same key inside the life (R2 single refresher).
    second = client.request_grant_sync(STUB_PROVIDER, session_id="sess-2")
    assert isinstance(second, Grant), second
    held_first = client.grants.peek(STUB_PROVIDER, SESSION)
    assert isinstance(held_first, Grant)
    posts_after_borrow = len(mesh.idp.posts)
    token_exp = held_first.token_expires_at_ms / 1000
    with _Ticker(mesh):
        assert net_fixtures.wait_for(lambda: time.time() > token_exp + IDLE_PAST_LIFE_S, 30.0)
    # One owner refresh in the idle life, however many sessions plus the relay.
    assert len(mesh.idp.posts) - posts_after_borrow == 1, mesh.idp.posts
    _stop_owner(mesh)
    provider = _Provider()
    error: ProviderError | None = None
    try:
        await asyncio.wait_for(_turn(store, provider), timeout=40.0)
    except ProviderError as exc:
        error = exc
    assert error is None, f"the turn failed with the owner gone: {error}"
    token_changed = bool(provider.seen) and provider.seen[0] != first[0]
    assert token_changed, "R1: the peer never held a newer token"
    served = client.grants.peek(STUB_PROVIDER, SESSION)
    assert isinstance(served, Grant)

    # R6 / Q10: the renewed hold is still capped by its own token and its receipt.
    from local_operator.network.credentials import offline_hold_s

    cap_s = offline_hold_s(mesh.b.root)
    now_boot = client_mod._sleep_inclusive_clock()  # noqa: SLF001
    hold_end = served.hold_end_ms(cap_s, now_boot)
    assert hold_end <= served.token_expires_at_ms
    assert hold_end <= max(served.hold_until_ms, served.grant_expires_at_ms)
    assert hold_end <= served.received_wall * 1000 + cap_s * 1000
    # ... and no bearer is served past its own token life.
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > served.token_expires_at_ms, 20.0)
    late = client.request_grant_sync(STUB_PROVIDER, session_id=SESSION)
    assert isinstance(late, BrokerError), "a bearer was served past its own token life"

    # Q7: never force_refresh; Q2/Q9: labelled, audited as an ordinary grant.
    renew_frames = frames.renew()
    assert renew_frames, "the step never asked"
    assert all(f.get("force_refresh") is False for f in renew_frames), renew_frames
    assert {f.get("renew_trigger") for f in renew_frames} == {"pre_expiry"}
    rows = _renew_rows(mesh)
    assert rows and all((r.get("detail") or {}).get("trigger") == "pre_expiry" for r in rows)

    # Q4: the renewed lend is in the owner's ledger, so member rm names it.
    from local_operator.network.credentials import lends

    ledger = lends._load(lends.lends_path(mesh.network_id, mesh.a.root))  # noqa: SLF001
    row = next(r for r in ledger if (r["holder"], r["key"]) == (mesh.borrower, STUB_PROVIDER))
    assert row["until_ms"] >= max(served.hold_until_ms, served.grant_expires_at_ms)

    # Q9 / Q6: no token material in either audit, nor anywhere on B's disk.
    tokens = {t for t in (*first, *provider.seen) if t}
    assert tokens
    from local_operator.network.store import audit_path

    mesh.a.audit.flush()
    a_text = audit_path(mesh.a.root).read_text(encoding="utf-8")
    assert "pre_expiry" in a_text, "the owner's audit file is not the one read here"
    b_text = _files_text(mesh.b.root)
    for token in tokens:
        assert token not in a_text and token not in b_text, _dig(token)


@pytest.mark.asyncio
async def test_r2_control_a_session_that_used_the_key_inside_the_life_is_held(
    mesh: Any, knobs: None, pre_expiry_renewal: Any  # noqa: F811
) -> None:
    """The pass control (GREEN on main too): the rig discriminates."""
    client, store = _setup(mesh)
    await _borrow_turn(store)
    await asyncio.sleep(1.0)
    seen = await _borrow_turn(store)
    _stop_owner(mesh)
    provider = _Provider()
    await asyncio.wait_for(_turn(store, provider), timeout=20.0)
    assert provider.seen == [seen[0]]
    held = client.request_grant_sync(STUB_PROVIDER, session_id=SESSION)
    assert isinstance(held, Grant) and held.held_offline


# ---------------------------------------------------------------------------
# R9 / Q6: MCP
# ---------------------------------------------------------------------------


def _mcp_owner(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> tuple[str, Any, list[str]]:
    """An MCP login on the owner whose refresh is a counting stub honouring the skew.

    ``ensure_mcp_oauth_fresh`` needs a real authorization server; the stub keeps its
    contract (refresh only when ``_stored_token_is_fresh`` says no, rotate both
    tokens) and counts the refreshes as the IdP POSTs.
    """
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp import auth as mcp_auth
    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    url = "https://mcp.g10.test/mcp"
    key = f"mcp:{url}"
    auth = AuthStore(config_dir=mesh.a.root)
    posts: list[str] = []

    def _token(n: int) -> Any:
        return OAuthToken.model_validate(
            {
                "access_token": f"mcp-access-{n}",
                "refresh_token": f"mcp-refresh-{n}",
                "expires_in": TOKEN_LIFE_S,
            }
        )

    asyncio.run(McpTokenStorage(url, auth).set_tokens(_token(0)))

    async def _refresh(server_url: str, cfg: Any, store: Any = None) -> None:
        storage = McpTokenStorage(server_url, store)
        if mcp_auth._stored_token_is_fresh(storage, await storage.get_tokens()):  # noqa: SLF001
            return None
        posts.append(server_url)
        await storage.set_tokens(_token(len(posts)))
        return None

    monkeypatch.setattr(mcp_auth, "ensure_mcp_oauth_fresh", _refresh)
    assert _lop_network("credential", "share", key, "--with", mesh.b.identity.name) == 0
    assert _pulled_or_held(mesh, key)
    return key, auth, posts


def test_r9_mcp_a_quiet_session_holds_the_newest_bearer_and_no_refresh_token(
    mesh: Any,  # noqa: F811
    knobs: None,
    frames: _OwnerFrames,
    pre_expiry_renewal: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on main (``owner_offline``: MCP was never renewed by the relay)."""
    key, auth, posts = _mcp_owner(mesh, monkeypatch)
    try:
        client = MeshCredentialClient.for_this_device(mesh.b.root)
        assert client is not None
        first = client.request_grant_sync(key, provider="mcp-oauth", session_id=SESSION)
        assert isinstance(first, Grant), first
        token_exp = first.token_expires_at_ms / 1000
        with _Ticker(mesh):
            assert net_fixtures.wait_for(lambda: time.time() > token_exp + IDLE_PAST_LIFE_S, 30.0)
        _stop_owner(mesh)
        after = client.request_grant_sync(key, provider="mcp-oauth", session_id=SESSION)
        assert isinstance(after, Grant) and after.held_offline, after
        assert after.access_token != first.access_token, "R9: MCP bearer never changed"
        assert len(posts) == 1, posts  # one owner refresh in the idle life
        assert all(f.get("force_refresh") is False for f in frames.renew())
        # Q6: the refresh token stays on the owner; the bearer is never persisted.
        held = renew_push.held_renewal(mesh.b, key, mesh.owner)
        assert held is not None and not {"refresh", "refresh_token"} & set(held)
        b_text = _files_text(mesh.b.root)
        for n in range(len(posts) + 1):
            assert f"mcp-refresh-{n}" not in b_text
            assert f"mcp-access-{n}" not in b_text
    finally:
        auth.close()


# ---------------------------------------------------------------------------
# Direct-seeded cells: a borrow, then one condition, then the step
# ---------------------------------------------------------------------------


@pytest.fixture()
def borrowed(mesh: Any, pre_expiry_renewal: Any) -> MeshCredentialClient:  # noqa: F811
    """B borrowed the stub provider through its relay from THIS (live) process."""
    _share(mesh)
    _pull(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    grant = client.request_grant_sync(STUB_PROVIDER, session_id=SESSION)
    assert isinstance(grant, Grant), grant
    return client


def _due(mesh: Any, key: str = STUB_PROVIDER, asker: Any = None) -> None:  # noqa: F811
    """Make the active entry's token fall inside the lead (as if time had passed)."""
    from local_operator.secrets.peer import process_info

    asker = asker or process_info(os.getpid())
    exp = int(time.time() * 1000) + renew_push.pre_expiry_lead_ms(key) // 2
    renew_push.note_session_borrow(mesh.b, key, mesh.owner, asker, exp)
    entry = getattr(mesh.b, renew_push._ACTIVE_ATTR)[key]  # noqa: SLF001
    entry["token_exp_ms"] = exp
    entry.pop("asked_for", None)
    # The relay's held renewal (if any) ages with the same clock.
    held = getattr(mesh.b, renew_push._HELD_ATTR, {})  # noqa: SLF001
    if key in held:
        grant, stamp = held[key]
        held[key] = (dataclasses.replace(grant, token_expires_at_ms=exp), stamp)


def test_q2_the_leg1_borrow_records_this_process_and_a_relay_ask_never_does(
    mesh: Any, borrowed: MeshCredentialClient  # noqa: F811
) -> None:
    """Q2/Q3: activity comes only from a session's served borrow, pinned by pid+start."""
    entry = getattr(mesh.b, renew_push._ACTIVE_ATTR)[STUB_PROVIDER]  # noqa: SLF001
    assert [a.pid for a in entry["askers"]] == [os.getpid()]
    renew_push.forget_active(mesh.b)
    relay_client = MeshCredentialClient.for_relay(mesh.b)
    assert renew_push.renew_key(mesh.b, relay_client, mesh.owner, STUB_PROVIDER) == "renewed"
    assert renew_push.active_keys(mesh.b) == [], "a relay ask kept itself alive"
    # A refused session ask (not a holder of this key) records nothing either.
    refused = borrowed.request_grant_sync("openai", session_id=SESSION)
    assert isinstance(refused, BrokerError)
    assert renew_push.active_keys(mesh.b) == []


def test_q2_a_forged_or_stale_process_claim_earns_no_renewal() -> None:
    """``asker_pid``/``asker_start`` count only for that exact live incarnation."""
    from local_operator.network.credentials import _control_asker
    from local_operator.secrets.peer import process_info

    me = process_info(os.getpid())
    assert me is not None
    good = {"asker_pid": me.pid, "asker_start": me.start_time}
    assert _control_asker(good) is not None
    assert _control_asker({**good, "asker_start": me.start_time + 1}) is None
    assert _control_asker({"asker_pid": str(me.pid), "asker_start": me.start_time}) is None
    assert _control_asker({}) is None


def test_q3_the_step_stops_when_the_borrowing_session_exits(
    mesh: Any, borrowed: MeshCredentialClient, frames: _OwnerFrames  # noqa: F811
) -> None:
    """Q3: a real child process borrows; while it lives the step asks, once it exits never."""
    from local_operator.secrets.peer import process_info

    renew_push.forget_active(mesh.b)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert net_fixtures.wait_for(lambda: process_info(child.pid) is not None, 5.0)
        identity = process_info(child.pid)
        _due(mesh, asker=identity)
        _ticks(mesh, 1)
        assert len(frames.renew()) == 1, "the live session's key was not renewed"
    finally:
        child.kill()
        child.wait(timeout=10)
    _due(mesh, asker=identity)  # a fresh token life, the same (now dead) session only
    entry = getattr(mesh.b, renew_push._ACTIVE_ATTR)[STUB_PROVIDER]  # noqa: SLF001
    entry["askers"] = [identity]
    _ticks(mesh, 3)
    assert len(frames.renew()) == 1, "the step asked for a session that has exited"
    assert renew_push.active_keys(mesh.b) == []


def test_q7_one_ask_per_token_life_and_at_most_four_an_hour(
    mesh: Any, borrowed: MeshCredentialClient, frames: _OwnerFrames  # noqa: F811
) -> None:
    _due(mesh)
    _ticks(mesh, 5)
    assert len(frames.renew()) == 1, "more than one ask in one token life"
    for _ in range(8):  # eight more "token lives" inside the same hour
        time.sleep(0.002)  # a distinct expiry per life
        _due(mesh)
        _ticks(mesh, 2)
    assert len(frames.renew()) == renew_push.PRE_EXPIRY_MAX_ASKS_PER_HOUR
    assert all(f.get("force_refresh") is False for f in frames.renew())


def test_q7_the_owners_own_rate_limit_still_applies_and_the_step_backs_off(
    mesh: Any, borrowed: MeshCredentialClient, frames: _OwnerFrames  # noqa: F811
) -> None:
    """The owner relay's slow-op slots full: ``rate_limited``; the step keeps the key."""
    import local_operator.network.relay as relay_mod

    _pool, slots = mesh.a._slow_executor()  # noqa: SLF001
    taken = 0
    while slots.acquire(blocking=False):
        taken += 1
    assert taken == relay_mod.SLOW_OP_MAX_PENDING
    try:
        _due(mesh)
        before = len(_renew_rows(mesh))
        _ticks(mesh, 3)
    finally:
        for _ in range(taken):
            slots.release()
    assert len(_renew_rows(mesh)) == before, "the owner served past its own limit"
    assert renew_push.active_keys(mesh.b) == [STUB_PROVIDER], "backed off by stopping"
    assert _holder_rows(mesh, "credential.renewal_refused") == []


def test_q7_a_refusal_by_name_stops_the_key_and_is_audited_on_the_holder(
    mesh: Any, borrowed: MeshCredentialClient, frames: _OwnerFrames  # noqa: F811
) -> None:
    """Revoked on the owner only (B not yet told): the ask is refused ``not_a_holder``."""
    relay_client = MeshCredentialClient.for_relay(mesh.b)
    assert renew_push.renew_key(mesh.b, relay_client, mesh.owner, STUB_PROVIDER) == "renewed"
    with placement_mod.mutate(mesh.network_id, mesh.a.root, self_device=mesh.owner) as document:
        document.revoke(STUB_PROVIDER, mesh.borrower, by=mesh.owner)
    _due(mesh)
    _ticks(mesh, 3)
    refused = _audit(mesh.a, "credential.grant_refused")
    assert any((r.get("detail") or {}).get("code") == "not_a_holder" for r in refused), refused
    assert len(frames.renew()) == 2  # the seeding ask, then exactly one refused ask
    assert renew_push.held_keys(mesh.b) == [] and renew_push.active_keys(mesh.b) == []
    rows = _holder_rows(mesh, "credential.renewal_refused")
    assert len(rows) == 1 and rows[0]["detail"]["code"] == "not_a_holder", rows
    assert rows[0]["detail"]["act"] == mesh.owner and rows[0]["detail"]["sub"] == mesh.borrower


@pytest.mark.parametrize("change", ["unshare", "panic", "consent_withdrawn", "member_rm"])
def test_q5_q6_each_end_of_the_share_drops_the_held_renewal_and_sends_nothing(
    mesh: Any, borrowed: MeshCredentialClient, frames: _OwnerFrames, change: str  # noqa: F811
) -> None:
    relay_client = MeshCredentialClient.for_relay(mesh.b)
    assert renew_push.renew_key(mesh.b, relay_client, mesh.owner, STUB_PROVIDER) == "renewed"
    assert renew_push.held_keys(mesh.b) == [STUB_PROVIDER]
    asks_before = len(frames.renew())
    if change == "unshare":
        assert (
            _lop_network("credential", "revoke", STUB_PROVIDER, "--from", mesh.b.identity.name) == 0
        )
        _pull(mesh)
    elif change == "panic":
        mesh.b._ctl_panic({"network": mesh.network_id})  # noqa: SLF001
    elif change == "consent_withdrawn":
        assert relay_client is not None
        policy = relay_client._policy()  # noqa: SLF001
        with policy.transaction() as document:
            policy.deny(document, mesh.owner, STUB_PROVIDER, "consent_withdrawn")
    else:
        mesh.a._ctl_member_rm(  # noqa: SLF001
            {"network": mesh.network_id, "device_id": mesh.borrower}
        )
    _due(mesh)
    _ticks(mesh, 3)
    assert renew_push.held_keys(mesh.b) == [], f"the renewal outlived {change}"
    assert len(frames.renew()) == asks_before, f"an ask was sent after {change}"
    # And a renewal ask made anyway (here even a DIALLING one, which the step never
    # makes) is refused by name, sends no frame and holds nothing.
    again = MeshCredentialClient.for_relay(mesh.b)
    assert again is not None
    if change == "member_rm":
        assert net_fixtures.wait_for(
            lambda: not again._self_membership_active(), 10.0
        )  # noqa: SLF001
    outcome = renew_push.renew_key(mesh.b, again, mesh.owner, STUB_PROVIDER, dial=True)
    assert outcome == NAMED[change], outcome
    assert len(frames.renew()) == asks_before
    assert renew_push.held_keys(mesh.b) == []


@pytest.mark.parametrize("change", ["none", "deny", "revoke", "member_rm"])
def test_q11_a_change_between_fetch_and_adopt_drops_the_renewal(
    mesh: Any, borrowed: MeshCredentialClient, change: str  # noqa: F811
) -> None:
    """The deterministic seam: fetch, then the change, then ``_adopt_renewal`` itself.

    The caller's stamp is the FETCH stamp (as if the session's admission had read
    the policy before the change landed), so only the adopt-time re-check can refuse.
    ``none`` is the positive control: nothing changed, the renewal is adopted.
    """
    relay_client = MeshCredentialClient.for_relay(mesh.b)
    assert renew_push.renew_key(mesh.b, relay_client, mesh.owner, STUB_PROVIDER) == "renewed"
    grant, fetch_stamp = getattr(mesh.b, renew_push._HELD_ATTR)[STUB_PROVIDER]  # noqa: SLF001
    renewal = (dataclasses.replace(grant, grant_id="g-renewal"), fetch_stamp)
    if change == "deny":
        policy = borrowed._policy()  # noqa: SLF001
        with policy.transaction() as document:
            policy.deny(document, mesh.owner, STUB_PROVIDER, "consent_withdrawn")
    elif change == "revoke":
        with placement_mod.mutate(
            mesh.network_id, mesh.b.root, self_device=mesh.borrower
        ) as document:
            document.revoke(STUB_PROVIDER, mesh.borrower, by=mesh.owner)
        borrowed = MeshCredentialClient.for_relay(mesh.b) or borrowed
        assert not borrowed.should_borrow(STUB_PROVIDER)
    elif change == "member_rm":
        mesh.a._ctl_member_rm(  # noqa: SLF001
            {"network": mesh.network_id, "device_id": mesh.borrower}
        )
        assert net_fixtures.wait_for(
            lambda: not borrowed._self_membership_active(), 10.0
        )  # noqa: SLF001
    policy = borrowed._policy()  # noqa: SLF001
    with policy.transaction() as document:
        borrowed._adopt_renewal(  # noqa: SLF001
            STUB_PROVIDER, "sess-adopt", renewal, fetch_stamp, policy=policy, document=document
        )
    adopted = borrowed.grants.peek(STUB_PROVIDER, "sess-adopt")
    if change == "none":
        assert adopted is not None and adopted.grant_id == "g-renewal"
    else:
        assert adopted is None, f"a renewal fetched before {change} was adopted after it"


def test_q12_owner_down_means_no_dial_and_no_ask(
    mesh: Any,  # noqa: F811
    borrowed: MeshCredentialClient,
    frames: _OwnerFrames,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stop_owner(mesh)
    dials: list[str] = []
    for name in ("_ensure_link_with_reason", "_ensure_link_with_probe", "_ensure_link"):
        real: Callable[..., Any] | None = getattr(mesh.b, name, None)
        if real is None:
            continue

        def counted(*args: Any, _real: Callable[..., Any] = real, **kwargs: Any) -> Any:
            dials.append("dial")
            return _real(*args, **kwargs)

        monkeypatch.setattr(mesh.b, name, counted)
    _due(mesh)
    _ticks(mesh, 5)
    assert dials == [], "the background step dialled an unreachable owner"
    assert frames.renew() == []
    assert renew_push.active_keys(mesh.b) == [STUB_PROVIDER], "an absent owner is not a stop"


def test_fence_b1_is_never_advertised_and_no_refresh_token_reaches_the_holder(
    mesh: Any, borrowed: MeshCredentialClient, frames: _OwnerFrames  # noqa: F811
) -> None:
    """G10 never touches B1 (refresh-token lending): only ``grant`` frames, no refresh."""
    from local_operator.network import wire
    from local_operator.network.credentials import renewal

    assert renewal.RENEWAL_CAP_STRING not in wire.LINK_CAPABILITIES
    for link in mesh.b.links.values():
        assert renewal.RENEWAL_CAP_STRING not in getattr(link, "local_capabilities", ())
    _due(mesh)
    _ticks(mesh, 2)
    assert frames.renew(), "the step never asked"
    assert all(f.get("kind") == "grant" for f in frames.frames)
    held = renew_push.held_renewal(mesh.b, STUB_PROVIDER, mesh.owner)
    assert held is not None and not {"refresh", "refresh_token"} & set(held)
    assert mesh.idp.current_refresh not in _files_text(mesh.b.root)


def test_the_step_rides_the_definitions_tick() -> None:
    from local_operator.network import definitions
    from local_operator.network.credentials import owner as owner_mod

    renew_push.install()
    assert renew_push.pre_expiry_tick in definitions._tick_steps()  # noqa: SLF001
    assert getattr(renew_push.pre_expiry_tick, "independent_cadence", False)
    assert owner_mod._RENEW_SESSION == renew_push.RENEW_SESSION  # noqa: SLF001
    # The lead stays inside the owner's skew and above the shipped 15 s tick.
    from local_operator.network.sync import SYNC_TICK_S

    assert SYNC_TICK_S * 1000 < renew_push.pre_expiry_lead_ms("anthropic") < 60_000


# ---------------------------------------------------------------------------
# Q8 cross-member: two holders renew one key in one token life -> one POST
# ---------------------------------------------------------------------------


def test_q8_two_members_renewing_one_key_cost_one_idp_post(
    owner: Any, idp: Any  # noqa: F811
) -> None:
    owner.broker._loop.loop()  # noqa: SLF001
    replies: list[dict[str, Any]] = []
    gate = threading.Barrier(2)

    def ask(device: str) -> None:
        gate.wait(timeout=10)
        frame = _grant_frame(device, session=renew_push.RENEW_SESSION)
        frame["renew_trigger"] = "pre_expiry"
        replies.append(owner.broker.on_broker(_Link(device), frame)["detail"])

    threads = [threading.Thread(target=ask, args=(d,)) for d in (BORROWER_DEVICE, BORROWER_TWO)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert [d.get("kind") for d in replies] == ["grant", "grant"], replies
    assert idp.posts == ["refresh-0"], idp.posts
    assert len({d["access_token"] for d in replies}) == 1
    owner.audit.flush()
    rows = [r for r in owner.audit.tail(50) if r.get("event") == "credential.grant"]
    assert {(r.get("detail") or {}).get("trigger") for r in rows} == {"pre_expiry"}
    # A free-text trigger never reaches the owner's audit (closed vocabulary).
    frame = _grant_frame(BORROWER_DEVICE, session=renew_push.RENEW_SESSION)
    frame["renew_trigger"] = "anything at all"
    owner.broker.on_broker(_Link(BORROWER_DEVICE), frame)
    owner.audit.flush()
    last = [r for r in owner.audit.tail(50) if r.get("event") == "credential.grant"][-1]
    assert "trigger" not in (last.get("detail") or {})
    assert _ask_grant  # imported for parity with the owner file's helpers
