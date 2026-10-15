"""Signed future-login consent → a RUNNING recipient session resolves a real Grant.

The seam no unit cell can cover: two real relays on loopback TCP, the owner's REAL
``AuthStore`` upsert as the login, a rotating loopback IdP that refuses a spent
refresh token, an approval that is really signed and really executed through
``consent.provision_from_run`` (software operator key — the existing approvals-store
fixture, no native signer), and the product's own lifecycle cadence
(``DefinitionsSyncer.tick``) as the ONLY driver. Nothing here calls ``credential
share`` or ``credential_placement`` (the manual pull), and the recipient's
``MeshAwareAuthStore`` is built BEFORE the approval and the login — the session this
slice is for.

Capabilities are set by the test the way an operator would (``set_member_capabilities``);
the reconciler never writes them, and ``test_..._notready`` pins that.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import definitions, relay, store
from local_operator.network.credentials import consent, placement, reconcile
from local_operator.network.credentials.store import build_auth_store
from local_operator.network.relay import set_member_capabilities
from local_operator.providers.auth_store import AuthStore
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_credentials_consent import request as signed_request
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


def _login(root: Any, idp_: RotatingIdP, *, account: str, expired: bool = True) -> None:
    """The owner's interactive login, through the product's own write path."""
    auth = AuthStore(config_dir=root)
    try:
        auth.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "stale-access",
                "expires": int(time.time() * 1000) + (-60_000 if expired else 3_600_000),
                "refresh": idp_.current_refresh,
                "email": f"{account}@example.test",
                "account_id": account,
            },
        )
    finally:
        auth.close()


@pytest.fixture()
def rmesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A and recipient B, paired, relays up, NO login, NO approval, session built."""
    from local_operator.operator import trust

    request.getfixturevalue("stub_provider")
    stub_idp: RotatingIdP = request.getfixturevalue("idp")
    pair_devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch, role="drive")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    monkeypatch.setattr(
        trust,
        "load_anchor",
        lambda uid=None: trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="software fixture",
            exists=False,
        ),
    )
    with store.mutate(record.network_id, server_b.root) as copy:
        row = copy.member(server_a.identity.device_id)
        assert row is not None
        row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    server_b.start()
    definitions.stop_syncer(server_a)
    definitions.stop_syncer(server_b)
    anchor = _make_key(server_a.root)
    # Only A listens in this rig, so the recipient's relay dials once (as its own cadence
    # does in production); A then reaches B over that same authenticated link.
    dial = lambda: server_b._ensure_link_with_reason(server_a.identity.device_id)  # noqa: E731
    link_b, reason = dial()
    assert link_b is not None, reason
    # THE RUNNING SESSION: built before approval and before any login exists.
    session = build_auth_store(server_b.root, db_path=server_b.root / "auth.db")
    clock = {"now": 10_000.0}

    def tick() -> list[tuple[str, str]]:
        # Past every per-step floor. BOTH relays tick, as in production: B's own cadence
        # is what dials A (only A listens here), and A's reply path reuses that link.
        clock["now"] += 100.0
        definitions.DefinitionsSyncer(server_b).tick(now=clock["now"])
        return definitions.DefinitionsSyncer(server_a).tick(now=clock["now"])

    def grant_caps() -> None:
        """What an operator does with `lop network member grant`; never the reconciler."""
        with store.mutate(record.network_id, server_a.root) as rec:
            set_member_capabilities(
                rec, device_id=server_b.identity.device_id, grant=["broker_credential"]
            )
            store.save(rec, server_a.root)

    def approve(account: str | None = "acct-1") -> str:
        peer = server_b.identity.to_json()["public_key"]
        fp = (
            "sha256:"
            + hashlib.sha256(base64.urlsafe_b64decode(peer + "=" * (-len(peer) % 4))).hexdigest()
        )
        rig = SimpleNamespace(
            root=server_a.root,
            own=server_a.identity,
            peer=server_b.identity,
            anchor=anchor,
            subject={
                "owner_device": server_a.identity.device_id,
                "network_id": record.network_id,
                "recipient_device": server_b.identity.device_id,
                "public_key_fingerprint": fp,
                "native_context": {"account_id": account} if account else {},
            },
        )

        def narrow(data: dict[str, Any]) -> None:
            data["inherit"]["providers"] = {"ids": [STUB_PROVIDER], "future_logins": True}
            data["scope"] = {"principal_ref": anchor.key_id}
            if account:
                data["scope"]["account_id"] = account

        aid, run = signed_request(rig, change=narrow)
        consent.provision_from_run(aid, run, rig.subject, server_a.root)
        return aid

    def borrow(session_id: str = "sess-1") -> Any:
        return asyncio.run(session.get_oauth_access(STUB_PROVIDER, session_id))

    try:
        yield SimpleNamespace(
            a=server_a,
            b=server_b,
            network_id=record.network_id,
            owner=server_a.identity.device_id,
            recipient=server_b.identity.device_id,
            idp=stub_idp,
            session=session,
            tick=tick,
            grant_caps=grant_caps,
            approve=approve,
            borrow=borrow,
            dial=dial,
            record=record,
        )
    finally:
        session.close()
        server_b.stop()


def _holder(m: Any, root: Any, device: str) -> Any:
    doc = placement.PlacementDocument.load(m.network_id, root, self_device=device)
    entry = doc.entry(STUB_PROVIDER)
    return entry.holder(m.recipient) if entry else None


def test_login_after_session_start_heals_on_ticks_alone(rmesh: Any) -> None:
    """The headline: approve, THEN log in, then only the cadence runs."""
    m = rmesh
    m.grant_caps()
    aid = m.approve("acct-1")
    assert m.borrow() is None, "no holder yet: the session must be on its plain path"
    _login(m.a.root, m.idp, account="acct-1")
    for _ in range(3):
        m.tick()
    holder = _holder(m, m.b.root, m.recipient)
    assert holder is not None and holder.granted_by == "consent:" + aid, holder
    access = m.borrow()
    assert access is not None, "the RUNNING session did not heal"
    assert access.access_token.startswith("access-")
    assert len(m.idp.posts) == 1, m.idp.posts
    # The recipient never holds the refresh token, only the short-lived bearer.
    assert not (m.b.root / "auth.db").exists() or "refresh-0" not in (
        m.b.root / "auth.db"
    ).read_bytes().decode("latin1")
    # No storm: further ticks and borrows are quiet while the access is still valid.
    for _ in range(3):
        m.tick()
    assert m.borrow("sess-2") is not None
    assert len(m.idp.posts) == 1, m.idp.posts


def _frames(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Every ``net_broker`` frame any link SENDS, as ``(kind, sender)`` — the wire census."""
    sent: list[tuple[str, str]] = []
    original = relay.PeerLink.request

    def spy(self: Any, frame: dict[str, Any], **kw: Any) -> Any:
        if isinstance(frame, dict) and frame.get("op") == "net_broker":
            sent.append((str(frame.get("kind")), str(frame.get("from_device") or "")))
        return original(self, frame, **kw)

    monkeypatch.setattr(relay.PeerLink, "request", spy)
    return sent


def _ready(m: Any, account: str = "acct-1") -> str:
    m.grant_caps()
    aid = m.approve(account)
    _login(m.a.root, m.idp, account=account)
    return aid


def test_missing_recipient_capability_is_notready_and_nothing_is_written(rmesh: Any) -> None:
    m = rmesh
    m.approve("acct-1")  # NO grant_caps: the recipient's row lacks broker_credential
    _login(m.a.root, m.idp, account="acct-1")
    members = store.record_path(m.network_id, m.a.root).read_bytes()
    for _ in range(3):
        m.tick()
    assert "notready:recipient_lacks_broker_credential" in reconcile.status_for(m.a, m.recipient)
    assert store.record_path(m.network_id, m.a.root).read_bytes() == members
    assert not placement.placement_path(m.network_id, m.a.root).exists()
    assert m.borrow() is None


def test_an_expiring_login_refreshes_once_per_expiry_and_never_storms(rmesh: Any) -> None:
    m = rmesh
    _ready(m)
    for _ in range(2):
        m.tick()
    assert m.borrow() is not None and len(m.idp.posts) == 1
    for expiry in (2, 3):
        _login(m.a.root, m.idp, account="acct-1")  # the owner's token expires again
        for _ in range(2):
            m.tick()
        assert m.borrow(f"s{expiry}") is not None
        assert m.borrow(f"s{expiry}b") is not None
        assert len(m.idp.posts) == expiry, m.idp.posts
    assert len(set(m.idp.posts)) == len(m.idp.posts), "a spent refresh token was presented twice"


def test_revoke_offline_then_relogin_never_regrants(rmesh: Any) -> None:
    m = rmesh
    _ready(m)
    for _ in range(2):
        m.tick()
    assert m.borrow() is not None
    consent.restrict_and_revoke(
        m.network_id,
        m.owner,
        m.recipient,
        {"credential_key": STUB_PROVIDER},
        "owner-revoke:t1",
        m.a.root,
    )
    _login(m.a.root, m.idp, account="acct-1")  # logout/relogin cannot clear a deny
    for _ in range(4):
        m.tick()
    assert _holder(m, m.a.root, m.owner) is None
    assert _holder(m, m.b.root, m.recipient) is None, "the revoke was not pushed to the recipient"
    posts = len(m.idp.posts)
    assert m.borrow("after-revoke") is None
    assert len(m.idp.posts) == posts, "a revoked recipient cost the IdP a refresh"


def test_same_account_relogin_resumes_but_another_account_is_a_named_owner_remedy(
    rmesh: Any,
) -> None:
    m = rmesh
    _ready(m)
    for _ in range(2):
        m.tick()
    assert m.borrow() is not None
    _login(m.a.root, m.idp, account="acct-1")
    m.tick()
    assert m.borrow("same") is not None
    posts = len(m.idp.posts)
    # The owner signs OUT of the approved account and in as another one. (While the
    # approved account's row still exists it keeps being served and the other row is
    # never picked or refreshed — pinned in the unit cells.)
    owner_auth = AuthStore(config_dir=m.a.root)
    owner_auth.delete_credentials_for_provider(STUB_PROVIDER)
    owner_auth.close()
    _login(m.a.root, m.idp, account="acct-2")  # NOT the approved account
    for _ in range(2):
        m.tick()
    assert m.borrow("other") is None
    refusal = m.session._refusals[STUB_PROVIDER]  # noqa: SLF001 — the borrower's last answer
    assert refusal.code == "scope_mismatch", refusal
    from local_operator.network.credentials.messages import render_broker_error

    sentence = render_broker_error(
        refusal, key=STUB_PROVIDER, owner_name="device-a", provider=STUB_PROVIDER
    )
    assert "device-a" in sentence and f"lop login {STUB_PROVIDER}" in sentence, sentence
    assert len(m.idp.posts) == posts, "the unapproved account was refreshed (forbidden IdP POST)"


def test_invalid_grant_names_the_owner_and_an_owner_relogin_recovers_without_restart(
    rmesh: Any,
) -> None:
    m = rmesh
    _ready(m)
    for _ in range(2):
        m.tick()
    assert m.borrow() is not None
    _login(m.a.root, m.idp, account="acct-1")
    m.idp.invalid = True
    assert m.borrow("dead") is None
    refusal = m.session._refusals[STUB_PROVIDER]  # noqa: SLF001
    # The stub IdP's refusal reaches the owner as a generic refresh failure; the point
    # is the SENTENCE: it names the OWNER and the owner-side remedy, and the session is
    # not left parked silently.
    from local_operator.network.credentials.messages import render_broker_error

    sentence = render_broker_error(
        refusal, key=STUB_PROVIDER, owner_name="device-a", provider=STUB_PROVIDER
    )
    assert refusal.code in {"grant_invalid", "refresh_failed", "no_local_credential"}, refusal
    assert "device-a" in sentence and f"lop login {STUB_PROVIDER}" in sentence, sentence
    m.idp.invalid = False
    m.idp.current_refresh = "refresh-after-relogin"
    _login(m.a.root, m.idp, account="acct-1")
    for _ in range(2):
        m.tick()
    assert m.borrow("recovered") is not None, "relogin on the owner did not heal the session"


def test_reconnect_and_recipient_restart_converge_without_owner_restart(
    rmesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = rmesh
    _ready(m)
    for _ in range(2):
        m.tick()
    assert _holder(m, m.b.root, m.recipient) is not None
    # Recipient "restarts": its placement is gone and its links are down.
    placement.placement_path(m.network_id, m.b.root).unlink()
    for link in list(m.b.links.values()):
        link.close()
    net_fixtures.wait_for(lambda: not any(x.alive for x in m.b.links.values()))
    m.tick()  # B redials A as its own cadence does; A re-pushes on the new incarnation
    link, reason = m.dial()
    assert link is not None, reason
    for _ in range(3):
        m.tick()
    assert _holder(m, m.b.root, m.recipient) is not None, "the push did not follow the reconnect"
    assert m.borrow("after-restart") is not None


def test_an_old_peer_receives_zero_new_frames(rmesh: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    m = rmesh
    sent = _frames(monkeypatch)
    # An old peer never negotiated the lifecycle feature: the gate is the SAME one the
    # lifecycle sender uses, so the push must be silent too.
    monkeypatch.setattr(
        "local_operator.network.credentials.lifecycle.negotiated", lambda link: False
    )
    _ready(m)
    for _ in range(3):
        m.tick()
    assert ("placement", m.owner) not in sent, sent
    assert _holder(m, m.b.root, m.recipient) is None
