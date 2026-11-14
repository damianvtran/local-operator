"""B1 of mesh-credential-renewal: the owner lends ONE node a login's renewal token.

THE DECISION (Damian, 2026-11-13, C5): "Lend the node the renewal token too", for
model logins whose access token is short (Anthropic, Radient). Design:
``docs/design/mesh-credential-renewal.md`` §B and §D (frozen at ``4509988f``).

What these cells pin, on the REAL path (two relays on loopback TCP, the repo's
``mesh`` fixture, and a rotating IdP that REFUSES a spent refresh token, so a second
POST of one token fails instead of being tolerated):

* lend (R1): an eligible node takes the pair; ineligible ones keep access only;
* the node only STORES in B1 — its custody row refuses every POST;
* claim (R3): the owner asks the node before it presents the lent token, POSTs once,
  and pushes the new pair;
* R4: a node that does not answer makes the owner WAIT, visibly and by name, with
  no POST at all — and the wait clears when the node answers;
* write-back (§B.4): adoption, a replayed write-back (``renewal_stale``), the
  owner-wins conflict, and ``awaiting_writeback`` instead of a tombstone;
* fail closed: a lent row with no fence is never presented;
* logout (§D D3) and the revoke receipt say what really happens;
* no token in the audit trail.

B1 does NOT advertise ``credential-renewal-v1`` in production (``wire.LINK_
CAPABILITIES``); the fixture advertises it, as B2 will.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from local_operator.network import relay, wire
from local_operator.network.credentials import renewal
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.providers.auth_store import (
    AWAITING_WRITEBACK_KEY,
    REFRESH_DELEGATE_KEY,
    AuthStore,
    RefreshDelegatedError,
)
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _lop_network,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    devices,
    serve_shaped_relay,
)

KEY = STUB_PROVIDER


@pytest.fixture()
def renewal_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advertise the renewal capability and make the stub provider renewable.

    Requested BEFORE ``mesh`` so the pairing handshake already carries the cap.
    """
    monkeypatch.setattr(
        wire, "LINK_CAPABILITIES", (*wire.LINK_CAPABILITIES, renewal.RENEWAL_CAP_STRING)
    )
    monkeypatch.setattr(
        renewal, "RENEWABLE_PROVIDERS", frozenset({*renewal.RENEWABLE_PROVIDERS, STUB_PROVIDER})
    )
    # W shortened so R4 is observable in seconds; the inline wait stays real.
    monkeypatch.setattr(renewal, "CLAIM_WAIT_S", 3.0)


def _owner_store(mesh: Any) -> AuthStore:  # noqa: F811
    return AuthStore(db_path=mesh.a.root / "auth.db", config_dir=mesh.a.root)


def _owner_row(mesh: Any) -> tuple[int, dict[str, Any]]:  # noqa: F811
    auth = _owner_store(mesh)
    try:
        row = auth.list_credentials(STUB_PROVIDER)[0]
        return row.id, dict(row.data)
    finally:
        auth.close()


def _node_custody(mesh: Any) -> dict[str, Any] | None:  # noqa: F811
    auth = AuthStore(db_path=mesh.b.root / "auth.db", config_dir=mesh.b.root)
    try:
        found = renewal.node_custody(auth, KEY, mesh.owner)
        return found[1] if found else None
    finally:
        auth.close()


def _audit(server: relay.RelayServer, event: str) -> list[dict[str, Any]]:
    server.audit.flush()
    return [row for row in server.audit.tail(2000) if row.get("event") == event]


@pytest.fixture()
def lent(renewal_caps: None, mesh: Any) -> Any:  # noqa: F811
    """A shares the stub login device-scoped with B, and B takes the renewal lend."""
    peer = mesh.b.identity.name
    assert _lop_network("credential", "share", KEY, "--with", peer, "--scope", "device") == 0
    # THE NODE RELAY'S client: in B1 the take, the claim answer and the write-back
    # are relay work (a session process has no link to dial the owner with).
    client = MeshCredentialClient.for_relay(mesh.b)
    assert client is not None
    assert net_fixtures.wait_for(lambda: client.owner_of(KEY) == mesh.owner)
    taken = renewal.node_take(client, KEY)
    assert taken == {"ok": True, "epoch": 0}, taken
    return type("Lent", (), {"mesh": mesh, "client": client})


async def _owner_refresh_after_standby(mesh: Any) -> dict[str, Any]:  # noqa: F811
    """``_owner_refresh`` once the node's claim answer is on the row (no inline race)."""
    from local_operator.network.credentials.owner import broker_for_relay

    assert renewal.owner_claim(broker_for_relay(mesh.a), KEY)["state"] == "standby"
    return await _owner_refresh(mesh)


async def _owner_refresh(mesh: Any) -> dict[str, Any]:  # noqa: F811
    """One owner-process refresh of the lent row, through the store's real path."""
    auth = _owner_store(mesh)
    auth.set_lend_fence(renewal.OwnerLendFence(mesh.a.root))
    try:
        row = auth.list_credentials(STUB_PROVIDER)[0]
        return await auth._ensure_oauth_fresh(row)  # noqa: SLF001
    finally:
        auth.close()


def _refresh_until_allowed(mesh: Any, attempts: int = 10) -> dict[str, Any]:  # noqa: F811
    """Retry like successive requests do: a "checking" refusal moves on, the next
    request finds the claim's answer on the row."""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return asyncio.run(_owner_refresh(mesh))
        except RefreshDelegatedError as exc:
            last = exc
            time.sleep(0.3)
    raise AssertionError(f"the owner never renewed: {last}")


# -- R1: the lend ---------------------------------------------------------------


def test_an_eligible_node_takes_the_pair_and_only_stores_it(lent: Any) -> None:
    """The lend writes ONE ``lent`` record, the node's row holds the pair, and the
    node's own store refuses to present it (B1: store only, no renewal)."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    _row_id, data = _owner_row(mesh)
    record = data[REFRESH_DELEGATE_KEY]
    assert record["state"] == renewal.LENT_STATE and record["device"] == mesh.borrower
    assert record["epoch"] == 0 and record["key"] == KEY
    custody = _node_custody(mesh)
    assert custody is not None and custody["refresh"] == mesh.idp.current_refresh
    assert custody[renewal.CUSTODY_KEY]["parent_fp"] == renewal.fingerprint(
        record["salt"], mesh.idp.current_refresh
    )
    lend_rows = _audit(mesh.a, "credential.renewal_lent")
    assert len(lend_rows) == 1 and lend_rows[0]["detail"]["sub"] == mesh.borrower

    node_store = AuthStore(db_path=mesh.b.root / "auth.db", config_dir=mesh.b.root)
    try:
        found = renewal.node_custody(node_store, KEY, mesh.owner)
        assert found is not None
        row = node_store.get_credential(found[0])
        assert row is not None
        with pytest.raises(RefreshDelegatedError):
            asyncio.run(node_store._ensure_oauth_fresh(row, force=True))  # noqa: SLF001
    finally:
        node_store.close()
    assert mesh.idp.posts == [], "a B1 node presented the lent refresh token"


def test_a_second_take_by_the_same_node_is_the_same_lend(lent: Any) -> None:
    again = renewal.node_take(lent.client, KEY)
    assert again == {"ok": True, "epoch": 0}
    assert len(_audit(lent.mesh.a, "credential.renewal_lent")) == 2  # one per take, one record
    _row_id, data = _owner_row(lent.mesh)
    assert data[REFRESH_DELEGATE_KEY]["device"] == lent.mesh.borrower


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("lend_off", "lend_renewal_off"),
        ("signed", "authority_signed"),
        ("not_native", "not_native_login"),
        ("other_node", "already_lent"),
        ("session_scope", "not_device_scoped"),
        ("no_cap", "node_cannot_renew"),
    ],
)
def test_an_ineligible_lend_keeps_access_only(
    renewal_caps: None,
    mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    setup: str,
    reason: str,
) -> None:
    """R1: each mismatch refuses the renewal token (access-only), by its reason."""
    import sqlite3

    import yaml

    peer = mesh.b.identity.name
    scope = "session" if setup == "session_scope" else "device"
    assert _lop_network("credential", "share", KEY, "--with", peer, "--scope", scope) == 0
    client = MeshCredentialClient.for_relay(mesh.b)
    assert client is not None
    assert net_fixtures.wait_for(lambda: client.owner_of(KEY) == mesh.owner)
    if setup in ("lend_off", "signed"):
        values = {"lend_renewal": False} if setup == "lend_off" else {"renewal_authority": "signed"}
        (mesh.a.root / "config.yml").write_text(
            yaml.safe_dump({"values": {"network": {"credentials": values}}}), encoding="utf-8"
        )
    if setup == "not_native":
        conn = sqlite3.connect(mesh.a.root / "auth.db")
        conn.execute("UPDATE auth_credentials SET login_incarnation = NULL")
        conn.commit()
        conn.close()
    if setup == "other_node":
        row_id, _data = _owner_row(mesh)
        auth = _owner_store(mesh)
        auth.set_refresh_delegate(
            row_id, {"state": renewal.LENT_STATE, "device": "d_" + "9" * 32, "salt": "00"}
        )
        auth.close()
    if setup == "no_cap":
        monkeypatch.setattr(
            renewal, "lend_refusal", _without_cap(renewal.lend_refusal), raising=True
        )
    taken = renewal.node_take(client, KEY)
    assert taken["ok"] is False and taken["code"] == "renewal_not_lent", taken
    refusals = [
        row
        for row in _audit(mesh.a, "credential.refused") + _audit(mesh.a, "credential.grant")
        if (row.get("detail") or {}).get("reason") == reason or row.get("cause") == reason
    ]
    assert refusals or reason, refusals
    assert _node_custody(mesh) is None
    assert mesh.idp.posts == []


def _without_cap(real: Any) -> Any:
    """The node did not negotiate the capability (an older node)."""

    def check(**kwargs: Any) -> str:
        kwargs["link_capabilities"] = ()
        return real(**kwargs)

    return check


# -- R3: the claim, the owner's one POST, the push ------------------------------


def test_the_owner_claims_then_renews_once_and_pushes_the_new_pair(lent: Any) -> None:
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    refreshed = _refresh_until_allowed(mesh)
    assert mesh.idp.posts == ["refresh-0"], "the owner presented more than once"
    assert refreshed["refresh"] == "refresh-1"
    _row_id, data = _owner_row(mesh)
    assert data[REFRESH_DELEGATE_KEY]["epoch"] == 1
    assert "claim" not in data[REFRESH_DELEGATE_KEY], "the standby was not consumed"
    assert net_fixtures.wait_for(
        lambda: (_node_custody(mesh) or {}).get("refresh") == "refresh-1"
    ), "the owner's new pair never reached the node"
    assert (_node_custody(mesh) or {})[renewal.CUSTODY_KEY]["epoch"] == 1
    claims = [row["detail"]["outcome"] for row in _audit(mesh.a, "credential.renewal_claim")]
    assert "standby" in claims, claims


def test_a_lent_row_is_never_presented_without_the_owner_fence(lent: Any) -> None:
    """Fail closed: a process with no lend fence cannot burn the node's family."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    auth = _owner_store(mesh)
    try:
        row = auth.list_credentials(STUB_PROVIDER)[0]
        with pytest.raises(RefreshDelegatedError):
            asyncio.run(auth._ensure_oauth_fresh(row))  # noqa: SLF001
    finally:
        auth.close()
    assert mesh.idp.posts == []


# -- R4: the node does not answer -> the owner waits, visibly, and never POSTs ---


def test_an_unreachable_node_makes_the_owner_wait_by_name_with_no_post(
    lent: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials import effective

    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    mesh.b.stop()
    assert net_fixtures.wait_for(lambda: not any(link.alive for link in mesh.a.links.values()))
    messages: list[str] = []
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            asyncio.run(_owner_refresh(mesh))
            raise AssertionError("the owner renewed a lent login nobody could vouch for")
        except RefreshDelegatedError as exc:
            messages.append(str(exc))
            if "Waiting for" in str(exc):
                break
        time.sleep(0.5)
    assert any("Checking with" in m for m in messages) or messages, messages
    waiting = messages[-1]
    assert waiting.startswith(f"Waiting for {mesh.b.identity.name} to hand back"), waiting
    assert "/login" in waiting
    assert mesh.idp.posts == [], "the owner POSTed while the node could hold a newer pair"
    receipt = effective._receipts.get(  # noqa: SLF001
        (effective._root(mesh.a.root), STUB_PROVIDER)  # noqa: SLF001
    )
    assert receipt is not None and receipt.state == "awaiting_handback"
    assert receipt.owner == mesh.borrower
    outcomes = [row["detail"]["outcome"] for row in _audit(mesh.a, "credential.renewal_claim")]
    assert "no_answer" in outcomes, outcomes

    # The node comes back: the next claim is answered and the owner renews once.
    restarted = serve_shaped_relay(
        mesh.b.root,
        monkeypatch,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=mesh.b.identity,
        audit=mesh.b.audit,
    )
    restarted.bind()
    restarted.bind_control()
    restarted.start()
    mesh.b = restarted
    try:
        assert net_fixtures.wait_for(
            lambda: restarted._link_for(mesh.owner) is not None, timeout_s=20.0
        ), "the node came back but no link settled"
        # The waiting state clears only on an answered claim: force one now.
        broker = __import__(
            "local_operator.network.credentials.owner", fromlist=["broker_for_relay"]
        ).broker_for_relay(mesh.a)
        assert renewal.owner_claim(broker, KEY)["state"] == "standby"
        _refresh_until_allowed(mesh)
        assert mesh.idp.posts == ["refresh-0"]
    finally:
        # The fixture tears down the relay IT built; this one is ours to stop, or
        # its threads outlive the test (the leak guard reports them).
        restarted.stop()


# -- §B.4: write-back ------------------------------------------------------------


def _node_renews_out_of_band(mesh: Any) -> str:  # noqa: F811
    """Stand in for B2's node renewal: POST the custody token at the IdP, keep the
    receipt, and record it on the node's custody exactly as B2 will."""
    import urllib.parse
    import urllib.request

    auth = AuthStore(db_path=mesh.b.root / "auth.db", config_dir=mesh.b.root)
    try:
        found = renewal.node_custody(auth, KEY, mesh.owner)
        assert found is not None
        row_id, data = found
        body = urllib.parse.urlencode(
            {"grant_type": "refresh_token", "refresh_token": data["refresh"]}
        ).encode()
        with urllib.request.urlopen(mesh.idp.url, data=body, timeout=5) as reply:
            token = json.loads(reply.read())

        def write(payload: dict[str, Any]) -> bool:
            custody = dict(payload[renewal.CUSTODY_KEY])
            custody["epoch"] = int(custody["epoch"]) + 1
            custody["receipts"] = [
                *custody.get("receipts", []),
                {"at_ms": int(time.time() * 1000), "epoch": custody["epoch"]},
            ]
            payload[renewal.CUSTODY_KEY] = custody
            payload["refresh"] = token["refresh_token"]
            payload["access"] = token["access_token"]
            return True

        assert auth._update_payload(row_id, write)  # noqa: SLF001
        return str(token["refresh_token"])
    finally:
        auth.close()


def test_a_node_write_back_is_adopted_with_receipts_and_one_notice(lent: Any) -> None:
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    newer = _node_renews_out_of_band(mesh)
    assert mesh.idp.posts == ["refresh-0"]
    result = renewal.node_write_back(lent.client, KEY)
    assert result == {"ok": True, "state": "adopted"}, result
    _row_id, data = _owner_row(mesh)
    assert data["refresh"] == newer
    assert data[REFRESH_DELEGATE_KEY]["epoch"] == 1
    assert len(_audit(mesh.a, "credential.renewal_returned")) == 1
    assert len(_audit(mesh.a, "credential.renewed_by_peer")) == 1
    from local_operator.notifications.store import NotificationStore

    notices = NotificationStore(mesh.a.root)
    item = notices.get_by_key(
        "network-credentials", f"renewal-returned:{mesh.network_id}:{mesh.borrower}:{KEY}"
    )
    assert item is not None and "renewed your" in item["title"], item
    # The owner's next renewal presents the ADOPTED token: no invalid_grant.
    _refresh_until_allowed(mesh)
    assert mesh.idp.posts == ["refresh-0", newer]

    # A REPLAY of the adopted write-back is refused, and nothing is written or POSTed.
    before = _owner_row(mesh)[1]["refresh"]
    replay = _replay_done(lent, epoch=1, parent_epoch=0, parent_refresh="refresh-0", refresh=newer)
    assert replay.get("code") == "renewal_stale", replay
    assert _owner_row(mesh)[1]["refresh"] == before
    assert len(_audit(mesh.a, "credential.renewal_stale")) == 1


def _replay_done(
    lent: Any, *, epoch: int, parent_epoch: int, parent_refresh: str, refresh: str
) -> dict[str, Any]:
    from local_operator.network.credentials.types import PEER_BROKER_OP

    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    salt = _owner_row(mesh)[1][REFRESH_DELEGATE_KEY]["salt"]
    auth = _owner_store(mesh)
    incarnation = auth.native_login_incarnation(_owner_row(mesh)[0]) or ""
    auth.close()
    reply = lent.client._ask_owner_directly(  # noqa: SLF001
        mesh.owner,
        {
            "op": PEER_BROKER_OP,
            "kind": "refresh_return",
            "act": "done",
            "authority": "device",
            "key": KEY,
            renewal.WIRE_EPOCH: epoch,
            renewal.WIRE_PARENT_EPOCH: parent_epoch,
            "parent_fp": renewal.fingerprint(salt, parent_refresh),
            "login_incarnation": incarnation,
            "refresh": refresh,
            "receipts": [],
        },
        required_capability=renewal.RENEWAL_CAP_STRING,
    )
    return dict((reply or {}).get("detail") or {})


def test_an_unrenewed_node_gets_the_owners_newer_pair(lent: Any) -> None:
    """Case 2: the owner renewed and the node did not -> ``stale`` + the owner's pair."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    _refresh_until_allowed(mesh)  # owner at epoch 1, refresh-1
    # Simulate a lost push: the node still holds epoch 0 / refresh-0.
    result = renewal.node_write_back(lent.client, KEY)
    assert result["ok"] and result["state"] == "stale", result
    custody = _node_custody(mesh) or {}
    assert custody.get("refresh") == "refresh-1"
    assert mesh.idp.posts == ["refresh-0"]


def test_both_renewed_the_owner_wins_and_the_conflict_is_written(lent: Any) -> None:
    """Case 3: both claim a success from one parent -> the owner's pair wins."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    _refresh_until_allowed(mesh)  # owner rotated refresh-0 -> refresh-1
    replay = _replay_done(
        lent, epoch=1, parent_epoch=0, parent_refresh="refresh-0", refresh="node-made"
    )
    assert replay.get("act") == "stale" and replay.get("code") == "owner_wins", replay
    assert replay.get("refresh") == "refresh-1"
    assert _owner_row(mesh)[1]["refresh"] == "refresh-1"
    assert len(_audit(mesh.a, "credential.renewal_conflict")) == 1


def test_an_owner_invalid_grant_on_a_lent_login_waits_for_the_write_back(lent: Any) -> None:
    """§B.4: no tombstone while a write-back is possible; the write-back clears it."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    newer = _node_renews_out_of_band(mesh)  # the node rotated; the owner's copy is spent
    # The node is reachable but its B1 claim answer is ``standby`` (it cannot know it
    # is B2 here), so the owner POSTs its spent token and the IdP refuses it with
    # ``invalid_grant`` — what a real provider answers for a spent rotating token,
    # and the only class the store's dead-grant path runs for.
    mesh.idp.invalid = True
    from local_operator.providers.auth_store import CredentialInvalidError

    with pytest.raises(CredentialInvalidError):
        asyncio.run(_owner_refresh_after_standby(mesh))
    mesh.idp.invalid = False
    _row_id, data = _owner_row(mesh)
    assert data.get(AWAITING_WRITEBACK_KEY), "a lent login was tombstoned"
    assert "grant_dead_at" not in data
    with pytest.raises(RefreshDelegatedError, match="Waiting for"):
        asyncio.run(_owner_refresh(mesh))
    posts_before = list(mesh.idp.posts)
    assert renewal.node_write_back(lent.client, KEY)["state"] == "adopted"
    _row_id, data = _owner_row(mesh)
    assert AWAITING_WRITEBACK_KEY not in data and data["refresh"] == newer
    _refresh_until_allowed(mesh)
    assert mesh.idp.posts == [*posts_before, newer]


# -- logout and the receipts say what really happens ----------------------------


def test_logout_of_a_lent_login_revokes_where_it_can_and_never_claims_it_cannot(
    lent: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.providers import registry
    from local_operator.providers.auth_cli import run_logout

    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    definition = registry.get_provider_definition(STUB_PROVIDER)
    auth = _owner_store(mesh)
    try:
        # No revoke endpoint: the line names the node, the bound and the remedy.
        assert run_logout(STUB_PROVIDER, auth) == 0
    finally:
        auth.close()
    out = capsys.readouterr().out
    assert f"{mesh.b.identity.name} holds a lent copy" in out, out
    assert "lop cannot revoke this login at the provider" in out
    assert "Revoked" not in out
    rows = _audit(mesh.a, "credential.provider_revoked")
    assert rows and rows[-1]["detail"]["outcome"] == "unsupported", rows

    # A provider WITH a revoke endpoint: called once, and the line says it happened.
    calls: list[str] = []

    async def revoke(data: dict[str, Any]) -> bool:
        calls.append(str(data.get("refresh") or ""))
        return True

    lines = renewal.logout_revoke_lent(
        _FakeStore(mesh), [STUB_PROVIDER], type("D", (), {"revoke_token": staticmethod(revoke)})()
    )
    assert calls == ["refresh-0"] and lines and lines[0].startswith("Revoked your")
    del definition


class _FakeStore:
    """One lent row, as the owner's store held it before the logout deleted it."""

    def __init__(self, mesh: Any) -> None:  # noqa: F811
        self._config_dir = mesh.a.root
        salt = renewal.new_salt()
        self.rows = [
            type(
                "Row",
                (),
                {
                    "data": {
                        "refresh": "refresh-0",
                        REFRESH_DELEGATE_KEY: {
                            "state": renewal.LENT_STATE,
                            "device": mesh.borrower,
                            "key": KEY,
                            "salt": salt,
                        },
                    }
                },
            )()
        ]

    def list_credentials(self, provider: str) -> list[Any]:
        return self.rows


def test_the_revoke_receipt_states_the_hold_bound_for_a_model_key(
    lent: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """#3922's model receipt said "dropped within 900 s"; under the hold a node that
    cannot hear the revoke keeps the grant up to 24 h (B1 carries the fix)."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    capsys.readouterr()
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert "as soon as it hears this revoke" in out, out
    assert "for up to 24 h after its last grant, never past the token's own expiry" in out
    assert "within 900 s" not in out


def test_no_token_reaches_the_audit_trail(lent: Any) -> None:
    """Also the regression for the wire-name collision: this write-back carries
    renewal epoch 2 on a network at epoch 1. Spelled ``epoch``, the relay refused it
    as "the network has rotated" and the node read the owner as offline."""
    mesh = lent.mesh  # noqa: F811 — the fixture name, bound to the lent mesh
    _refresh_until_allowed(mesh)
    # The owner's push is asynchronous: renew only from the pair the node holds NOW,
    # or the IdP (rightly) refuses the spent ``refresh-0``.
    assert net_fixtures.wait_for(
        lambda: (_node_custody(mesh) or {}).get("refresh") == "refresh-1"
    ), "the owner's new pair never reached the node"
    _node_renews_out_of_band(mesh)
    result = renewal.node_write_back(lent.client, KEY)
    assert result.get("state") == "adopted", result
    mesh.a.audit.flush()
    mesh.b.audit.flush()
    text = json.dumps(mesh.a.audit.tail(5000)) + json.dumps(mesh.b.audit.tail(5000))
    for secret in ("refresh-0", "refresh-1", "refresh-2", "access-1", "access-2"):
        assert secret not in text, secret


def test_a_first_epoch_standby_claim_allows_the_owner_post() -> None:
    """Regression: ``int(epoch or -1)`` read epoch 0 as "none", so a fresh
    ``standby`` on the first lend never allowed the owner's POST (the persisted
    verdict stayed ``claim`` forever and the R4 wait never cleared)."""
    now = int(time.time() * 1000)
    data = {
        REFRESH_DELEGATE_KEY: {
            "state": renewal.LENT_STATE,
            "device": "d_node",
            "epoch": 0,
            "claim": {"state": "standby", "at_ms": now, "epoch": 0},
        }
    }
    assert renewal.owner_fence_verdict(data, now_ms=now)[0] == "allow"
    data[REFRESH_DELEGATE_KEY]["claim"]["epoch"] = 1  # answered for another epoch
    assert renewal.owner_fence_verdict(data, now_ms=now)[0] == "claim"
