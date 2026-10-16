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


def test_a_lifecycle_only_peer_receives_zero_push_frames(
    rmesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The v0.71.3 shape: it negotiates ``credential-lifecycle-v1`` and has NO push handler.

    Capabilities are fixed in the HANDSHAKE, so the link is closed and re-dialled with the
    push string removed from what this build advertises: both ends then hold exactly the
    lifecycle string and nothing else, which is what a previous-release peer is. The
    lifecycle announcement (its existing kind) must still flow; the new frame must not.
    """
    from local_operator.network import wire

    m = rmesh
    for link in list(m.b.links.values()):
        link.close()
    net_fixtures.wait_for(lambda: not any(x.alive for x in m.b.links.values()))
    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(c for c in wire.LINK_CAPABILITIES if c != wire.CREDENTIAL_PLACEMENT_PUSH_V1),
    )
    link, reason = m.dial()
    assert link is not None, reason
    assert wire.CREDENTIAL_LIFECYCLE_V1 in link.capabilities
    assert wire.CREDENTIAL_PLACEMENT_PUSH_V1 not in link.capabilities
    sent = _frames(monkeypatch)
    _ready(m)
    for _ in range(3):
        m.tick()
    assert ("placement", m.owner) not in sent, sent
    assert _holder(m, m.b.root, m.recipient) is None
    reading = reconcile.status_for(m.a, m.recipient)
    assert "skipped:peer_lacks_placement_push" in reading and "upgrade" in reading, reading
    # The holder exists on the OWNER (projection is local); only delivery is withheld.
    assert _holder(m, m.a.root, m.owner) is not None


# =====================================================================================
# The PREVIOUS RELEASE as a real receiver (review R1 / QA Q1)
# =====================================================================================
#
# ``credential-lifecycle-v1`` shipped in v0.71.3 with no handler for the push. A mock that
# flips ``negotiated`` cannot model that peer: only the real source can. The cell below
# runs the receiver as a SUBPROCESS importing an exported copy of the pre-slice source
# (``LOP_PREVIOUS_RELEASE_TREE``, e.g. ``git archive 3f8a0cb07 local_operator | tar -x``),
# so what it advertises and what it can handle are that release's own. It is skipped —
# loudly, with the reason — where no such tree is supplied (CI checks out the PR only);
# the PR body records the run that exercised it.

_PREVIOUS_RECEIVER = """
import sys, time, json
from pathlib import Path
import importlib.util
import local_operator
assert local_operator.__file__.startswith(sys.argv[1] + "/"), local_operator.__file__
assert importlib.util.find_spec("local_operator.network.credentials.reconcile") is None
from local_operator.network import audit as audit_mod, identity as idm, relay, store
from local_operator.network.credentials import placement

root = Path(sys.argv[2])
with_broker = sys.argv[3] == "1"
me = idm.load(root)
network = store.list_networks(root)[0]
if with_broker:
    # An old receiver that ALSO owns a credential: it has a broker, so the frame would go
    # down the legacy ``placement_frame`` merge instead of a not-implemented refusal.
    with placement.mutate(network.network_id, root, self_device=me.device_id) as doc:
        doc.declare("ownedbyold", owner_device=me.device_id, provider="ownedbyold",
                    kind="oauth-rotating", by=me.device_id)
server = relay.RelayServer(
    settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
    identity=me, audit=audit_mod.AuditLog(root),
)
server.start()
peer = [m.device_id for m in network.active_members() if m.device_id != me.device_id][0]
print("READY", flush=True)
while True:
    time.sleep(0.5)
    link, _why = server._ensure_link_with_reason(peer)  # the recipient dials, as its cadence does
    if link is not None and not getattr(server, "_caps_reported", False):
        server._caps_reported = True
        print("CAPS " + ",".join(sorted(link.capabilities)), flush=True)
    trigger = root / "finite-control-request"
    result = root / "finite-control-result.json"
    if link is not None and trigger.exists() and not result.exists():
        key = trigger.read_text()
        reply = link.request({
            "op": "net_broker", "kind": "grant", "key": key, "provider": key,
            "from_device": me.device_id, "for_session": "finite-control", "req": "old-finite-1",
        })
        detail = (reply or {}).get("detail") or {}
        result.write_text(json.dumps({"kind": detail.get("kind"), "code": detail.get("code"),
                                     "has_access": bool(detail.get("access_token"))}))
"""


@pytest.mark.parametrize("with_broker", [False, True], ids=["brokerless", "broker-present"])
def test_the_previous_release_receives_zero_push_frames_and_nothing_changes(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    with_broker: bool,
) -> None:
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    from local_operator.network import wire
    from local_operator.operator import trust

    previous = os.environ.get("LOP_PREVIOUS_RELEASE_TREE", "")
    if not previous or not (Path(previous) / "local_operator").is_dir():
        pytest.skip(
            "LOP_PREVIOUS_RELEASE_TREE is not set: export the pre-slice source "
            "(git archive 3f8a0cb07 local_operator | tar -x -C <dir>) to run this cell"
        )
    previous = str(Path(previous).resolve())
    request.getfixturevalue("stub_provider")
    rotating: RotatingIdP = request.getfixturevalue("idp")
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
    definitions.stop_syncer(server_a)
    definitions.stop_syncer(server_b)
    anchor = _make_key(server_a.root)
    with store.mutate(record.network_id, server_a.root) as rec:
        set_member_capabilities(
            rec, device_id=server_b.identity.device_id, grant=["broker_credential"]
        )
        store.save(rec, server_a.root)
    script = request.getfixturevalue("tmp_path") / "previous_receiver.py"
    script.write_text(_PREVIOUS_RECEIVER)
    env = {
        "HOME": str(server_b.root.parent),
        "LOCAL_OPERATOR_CONFIG_DIR": str(server_b.root),
        "PATH": os.environ["PATH"],
        "PYTHONPATH": previous,
        "TERM": "xterm-256color",
    }
    log_path = script.with_suffix(".log")
    output = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, str(script), previous, str(server_b.root), "1" if with_broker else "0"],
        cwd=previous,
        env=env,
        start_new_session=True,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sent = _frames(monkeypatch)
    try:
        ready = net_fixtures.wait_for(
            lambda: "READY" in log_path.read_text() or proc.poll() is not None, timeout_s=30
        )
        assert ready and proc.poll() is None, log_path.read_text()[-4000:]
        rig = SimpleNamespace(
            root=server_a.root,
            own=server_a.identity,
            peer=server_b.identity,
            anchor=anchor,
            subject={
                "owner_device": server_a.identity.device_id,
                "network_id": record.network_id,
                "recipient_device": server_b.identity.device_id,
                "public_key_fingerprint": "sha256:"
                + hashlib.sha256(
                    base64.urlsafe_b64decode(
                        (pk := server_b.identity.to_json()["public_key"]) + "=" * (-len(pk) % 4)
                    )
                ).hexdigest(),
                "native_context": {"account_id": "acct-1"},
            },
        )

        def narrow(data: dict[str, Any]) -> None:
            data["inherit"]["providers"] = {"ids": [STUB_PROVIDER], "future_logins": True}
            data["scope"] = {"principal_ref": anchor.key_id, "account_id": "acct-1"}

        aid, run = signed_request(rig, change=narrow)
        consent.provision_from_run(aid, run, rig.subject, server_a.root)
        _login(server_a.root, rotating, account="acct-1")
        # Settle the authenticated link's membership receipt before taking the snapshot:
        # the handshake updates last_seen (not a capability write by the reconciler).
        assert net_fixtures.wait_for(lambda: any(x.alive for x in server_a.links.values()))
        from tests.unit.network.test_readiness_link import _await_membership_pull

        warm = next(x for x in server_a.links.values() if x.alive)
        _await_membership_pull(warm)
        members = store.record_path(record.network_id, server_a.root).read_bytes()
        clock = {"now": 50_000.0}
        syncer = definitions.DefinitionsSyncer(server_a)
        link_seen = False
        for _ in range(8):
            clock["now"] += 100.0
            syncer.tick(now=clock["now"])
            link_seen = link_seen or any(x.alive for x in server_a.links.values())
            time.sleep(0.4)
        assert link_seen, "the previous-release receiver never connected"
        peer_link = next(x for x in server_a.links.values() if x.alive)
        # THE DISCRIMINATING FACT: this peer negotiated the lifecycle string and has no push bit.
        assert wire.CREDENTIAL_LIFECYCLE_V1 in peer_link.capabilities
        assert wire.CREDENTIAL_PLACEMENT_PUSH_V1 not in peer_link.capabilities
        assert wire.CREDENTIAL_LIFECYCLE_V1 in peer_link.local_capabilities
        assert [k for k, _f in sent if k == "placement"] == [], sent
        reading = reconcile.status_for(server_a, server_b.identity.device_id)
        assert "delivery=skipped:peer_lacks_placement_push" in reading, reading
        assert "upgrade" in reading, reading
        # Projection is local to the owner; nothing reached the old receiver's disk, and no
        # capability was written by the reconciler.
        owner_doc = placement.PlacementDocument.load(
            record.network_id, server_a.root, self_device=server_a.identity.device_id
        )
        holder = owner_doc.entry(STUB_PROVIDER)
        assert holder is not None and holder.holder(server_b.identity.device_id) is not None
        old_doc = placement.PlacementDocument.load(
            record.network_id, server_b.root, self_device=server_b.identity.device_id
        )
        assert old_doc.entry(STUB_PROVIDER) is None, "the old receiver was handed the row"
        assert store.record_path(record.network_id, server_a.root).read_bytes() == members
        # Legacy finite-grant CONTROL over the SAME old-source authenticated link: an
        # explicit share remains explicit, and the old peer can still ask for its bearer.
        with placement.mutate(
            record.network_id, server_a.root, self_device=server_a.identity.device_id
        ) as doc:
            doc.grant(STUB_PROVIDER, server_b.identity.device_id, by=server_a.identity.device_id)
        placement_before = placement.placement_path(record.network_id, server_a.root).read_bytes()
        (server_b.root / "finite-control-request").write_text(STUB_PROVIDER)
        result_path = server_b.root / "finite-control-result.json"
        assert net_fixtures.wait_for(
            result_path.exists
        ), "old peer did not resolve its finite grant"
        import json

        finite_result = json.loads(result_path.read_text())
        assert finite_result == {"kind": "grant", "code": None, "has_access": True}, finite_result
        assert len(rotating.posts) == 1
        assert (
            placement.placement_path(record.network_id, server_a.root).read_bytes()
            == placement_before
        )
        assert [kind for kind, _sender in sent if kind == "placement"] == []
    finally:
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(10)
        finally:
            output.close()
            definitions.stop_syncer(server_a)


@pytest.mark.parametrize("change", ["account", "withdraw"])
@pytest.mark.parametrize("remove_guard", [False, True], ids=["guarded", "negative-control"])
def test_a_real_refresh_cannot_mint_after_its_identity_or_approval_changes(
    rmesh: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    remove_guard: bool,
) -> None:
    """R5: flip identity/consent INSIDE the real refresh, after its loopback IdP POST.

    Pre-refresh row selection has already approved acct-1 when the change happens. A
    refresh may return a different account, or consent may be withdrawn while the IdP
    is answering. Neither bearer may escape at mint. The negative controls neutralise
    ONLY derived_mint_check: both then produce a real Grant, discriminating this guard
    from the earlier serve_gate and the general holder check.
    """
    from dataclasses import replace

    from local_operator.providers import registry

    m = rmesh
    _ready(m)
    for _ in range(2):
        m.tick()
    provider = registry.get_provider_definition(STUB_PROVIDER)
    assert provider is not None and provider.refresh_token is not None
    real_refresh = provider.refresh_token
    changed: list[str] = []

    async def refresh_then_change(creds: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        refreshed = await real_refresh(creds, **kwargs)
        assert len(m.idp.posts) == 1, "the real IdP POST must precede this boundary"
        changed.append(change)
        if change == "account":
            refreshed = dict(refreshed, account_id="acct-2", email="acct-2@example.test")
        else:
            authority = consent.ConsentDocument(m.network_id, m.owner, m.a.root)
            policy = authority.load()
            policy["recipients"][m.recipient]["enabled"] = False
            authority.save(policy, policy["revision"])
        return refreshed

    monkeypatch.setitem(
        registry._BY_ID,  # noqa: SLF001 — same registry seam as stub_provider
        STUB_PROVIDER,
        replace(provider, refresh_token=refresh_then_change),
    )
    if remove_guard:
        monkeypatch.setattr(reconcile, "derived_mint_check", lambda *a, **k: None)
    access = m.borrow("mid-refresh")
    assert changed == [change] and len(m.idp.posts) == 1
    if remove_guard:
        assert access is not None, "negative control must serve if the mint guard is removed"
        if change == "account":
            assert access.account_id == "acct-2"
    else:
        assert access is None, "the changed authority/identity escaped as a bearer"
        refusal = m.session._refusals[STUB_PROVIDER]  # noqa: SLF001 — real caller's refusal
        assert refusal.code == ("scope_mismatch" if change == "account" else "consent_withdrawn")
        assert refusal.owner_device == m.owner


def test_the_same_facade_heals_on_the_real_shipped_cadence_threads(rmesh: Any) -> None:
    """Acceptance path: actual 15 s run loops, never a manual tick/share/pull/restart.

    rmesh constructs the caller facade before approval AND login. Both real syncer
    threads then run the complete registered cadence (definitions/state/lifecycle), so
    this covers the scheduler seam in addition to the deterministic tick cells above.
    """
    from local_operator.network.sync import SYNC_TICK_S

    m = rmesh
    facade, delegate = m.session, m.session._mesh  # noqa: SLF001 — assert SAME running caller
    assert m.borrow() is None
    m.grant_caps()
    m.approve("acct-1")
    members = store.record_path(m.network_id, m.a.root).read_bytes()
    syncers = [definitions.DefinitionsSyncer(server) for server in (m.a, m.b)]
    assert all(syncer._tick_s == SYNC_TICK_S == 15.0 for syncer in syncers)  # noqa: SLF001
    for syncer in syncers:
        syncer.start()
    try:
        started = time.monotonic()
        _login(m.a.root, m.idp, account="acct-1")
        assert net_fixtures.wait_for(
            lambda: _holder(m, m.b.root, m.recipient) is not None, timeout_s=45
        ), "the running caller never received the approved holder"
        elapsed = time.monotonic() - started
        access = m.borrow("same-running-session")
        assert m.session is facade and m.session._mesh is delegate  # noqa: SLF001
        assert access is not None and access.access_token == "access-1"
        assert m.idp.posts == ["refresh-0"]
        assert store.record_path(m.network_id, m.a.root).read_bytes() == members
        recipient_store = AuthStore(db_path=m.b.root / "auth.db", config_dir=m.b.root)
        try:
            assert recipient_store.list_credentials(STUB_PROVIDER, include_disabled=True) == []
        finally:
            recipient_store.close()
        print(
            f"real_cadence: same_facade=true tick_s=15 holder_after_s={elapsed:.2f} "
            "first_resolve=idp_posts:1 recipient_native_rows:0 members_byte_identical=true"
        )
    finally:
        for syncer in syncers:
            syncer.stop()
        for syncer in syncers:
            syncer.join(timeout=15)
            assert not syncer.is_alive(), "the acceptance rig leaked a syncer thread"


@pytest.mark.parametrize("revoke", [False, True], ids=["grant", "revoke"])
def test_a_failed_real_link_push_is_retried_on_the_next_tick(
    rmesh: Any, monkeypatch: pytest.MonkeyPatch, revoke: bool
) -> None:
    """R2: one failed send must not park either a new holder or a revoke for 300 seconds."""
    m = rmesh
    _ready(m)
    if revoke:
        m.tick()
        assert _holder(m, m.b.root, m.recipient) is not None
        assert m.borrow() is not None
        consent.restrict_and_revoke(
            m.network_id,
            m.owner,
            m.recipient,
            {"credential_key": STUB_PROVIDER},
            "real-link-transient-revoke",
            m.a.root,
        )
    requests = 0
    original = relay.PeerLink.request

    def fail_once(self: Any, frame: dict[str, Any], **kwargs: Any) -> Any:
        nonlocal requests
        if frame.get("origin") == reconcile.PUSH_ORIGIN and frame.get("from_device") == m.owner:
            requests += 1
            if requests == 1:
                return None  # PeerLink.request's actual result on timeout/dead send
        return original(self, frame, **kwargs)

    monkeypatch.setattr(relay.PeerLink, "request", fail_once)
    m.tick()
    assert requests == 1
    assert (_holder(m, m.b.root, m.recipient) is not None) is revoke
    assert "retrying next tick" in reconcile.status_for(m.a, m.recipient)
    posts = len(m.idp.posts)
    m.tick()
    assert requests == 2, "the next cadence pass did not retry the failed push"
    assert (_holder(m, m.b.root, m.recipient) is not None) is not revoke
    assert "delivery=" not in reconcile.status_for(m.a, m.recipient)
    if revoke:
        assert m.borrow("after-revoke") is None and len(m.idp.posts) == posts
    else:
        assert m.borrow() is not None and len(m.idp.posts) == 1
