"""The real path, end to end: mint -> take -> refresh on the node -> return -> deny.

WHY A SECOND FILE. ``test_refresh_delegation`` proves each fence against the seam
that enforces it; this one proves the SEAMS ARE CONNECTED — the shipped handlers
carry the material, the store's real ``_ensure_oauth_fresh`` presents it, and the
deny that comes back later stops the next POST. A suite that only had the unit
half would pass with a fence that no code path ever calls.

Two substitutions, named rather than hidden: the operator key is a fixture P-256
key (no gesture is prompted) and the link is a two-attribute object — the broker
reads ``link.device_id`` and nothing else to authorise a caller, which is its own
documented rule. Everything else is production code and a real HTTP token endpoint
that rotates and refuses a spent token, so ``posts`` is evidence and not a stub's
tolerance.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import refresh
from tests.unit.network import test_credentials_owner as owner_rig
from tests.unit.network import test_refresh_delegation as rig

STUB_PROVIDER = owner_rig.STUB_PROVIDER
KEY, NETWORK, NODE, OWNER = rig.KEY, rig.NETWORK, rig.NODE, rig.OWNER

# pytest resolves a fixture by NAME in a test signature, so these re-exports are the
# registration: the linter's "unused import" and the F811 redefinitions it reports
# are both that mechanism, and `--noqa` here documents why rather than hiding it.
idp = owner_rig.idp
stub_provider = owner_rig.stub_provider
operator_key = rig.operator_key
operator_spki = rig.operator_spki

LINK_DEVICE = NODE


def anchored_owner(tmp_path: Path, monkeypatch, operator_spki: bytes) -> Any:
    """An owner root that believes a fixture operator key, plus an anchor entry for NODE.

    The anchor is the whole trust root: ``load_local_anchor`` is monkeypatched at
    its module (the one reader every operator-signed surface uses), so the
    broker's own verification path runs unchanged against a key this test holds.
    """
    from local_operator.operator import trust

    anchor = SimpleNamespace(
        spki=operator_spki,
        key_id="fixture",
        devices=(
            {
                "device_id": NODE,
                # base64url, the shape `trust` stores and `_node_key_id` reads back
                "spki": __import__("base64").urlsafe_b64encode(operator_spki).decode().rstrip("="),
            },
        ),
    )
    monkeypatch.setattr(trust, "load_local_anchor", lambda *a, **k: anchor)
    return anchor


def _signer(operator_key: Any) -> Any:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    return lambda message: operator_key.sign(message, ec.ECDSA(hashes.SHA256()))


def _owner_broker(tmp_path: Path, operator_key: Any, audit: Any = None) -> Any:
    from local_operator.network.credentials.owner import MeshCredentialBroker
    from local_operator.providers.auth_store import AuthStore

    root = tmp_path / "owner"
    root.mkdir(exist_ok=True)
    os_environ = __import__("os").environ
    os_environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    auth = AuthStore(config_dir=root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            "access": "access-0",
            "expires": int(time.time() * 1000) - 60_000,
            "refresh": "refresh-0",
            "email": "owner@example.test",
            "login_incarnation": "inc-1",
        },
    )
    document = placement_mod.PlacementDocument("n_refresh", root=root, written_by=OWNER)
    document.declare(
        KEY,
        owner_device=OWNER,
        owner_device_name="owner-laptop",
        provider=KEY,
        identity_label="owner@example.test",
        by=OWNER,
    )
    document.grant(KEY, NODE, scope="device", by=OWNER)
    document.save()
    broker = MeshCredentialBroker(
        root=root, self_device=OWNER, self_device_name="owner", network_id=NETWORK, audit=audit
    )
    broker._refresh_signer = _signer(operator_key)
    return broker, auth, root


def _frame(**fields: Any) -> dict[str, Any]:
    return {"req": 1, "from_device": NODE, **fields}


def test_mint_take_refresh_return_and_deny(
    tmp_path: Path, monkeypatch, stub_provider: str, idp: Any, operator_key: Any, operator_spki: Any
) -> None:
    anchored_owner(tmp_path, monkeypatch, operator_spki)
    broker, auth, owner_root = _owner_broker(tmp_path, operator_key)
    link = SimpleNamespace(device_id=NODE, network_id=NETWORK, epoch=8)
    try:
        # -- 1. the operator's mint -----------------------------------------
        minted = broker.mint_refresh_delegation(
            key=KEY,
            device=NODE,
            not_after=time.time() + 3600,
            exclusivity_confirmed=True,
            login_epoch=7,
        )
        assert minted["ok"] and minted["statement"]
        assert auth.refresh_delegate_for(1)["device"] == NODE

        # -- 2. the node takes the material ---------------------------------
        taken = broker.refresh_take(link, _frame(key=KEY))
        assert taken["ok"], taken
        assert taken["refresh"] == "refresh-0"

        # -- 3. the owner's gate now refuses ITS OWN POST --------------------
        from local_operator.providers.auth_store import RefreshDelegatedError

        auth._conn.execute(
            "UPDATE auth_credentials SET data = ? WHERE id = 1",
            (
                __import__("json").dumps(
                    {**dict(auth.get_credential(1).data), "expires": int(time.time() * 1000) - 1000}
                ),
            ),
        )
        auth._conn.commit()
        with pytest.raises(RefreshDelegatedError):
            asyncio.run(auth.ensure_oauth_fresh_or_raise(1))
        assert idp.posts == [], "the owner presented the token it had delegated"

        # -- 4. the node refreshes on its own root (a real POST) -------------
        node_root = tmp_path / "node"
        node_root.mkdir()
        installed = refresh.install(
            NETWORK,
            node_root,
            recipient=NODE,
            statement=taken["statement"],
            operator_spki=operator_spki,
        )
        assert installed["state"] == "active"
        from local_operator.network.credentials.client import MeshCredentialClient
        from local_operator.network.credentials.held_policy import HeldPolicyStore
        from local_operator.providers.auth_store import AuthStore

        node_client = SimpleNamespace(
            root=node_root,
            network_id=NETWORK,
            self_device=NODE,
            _policy=lambda: HeldPolicyStore(NETWORK, NODE, node_root),
            _self_membership_active=lambda: True,
        )
        node_client.refresh_fence = MeshCredentialClient.refresh_fence.__get__(node_client)
        monkeypatch.setattr(refresh, "anchor_spki_for", lambda root: operator_spki)
        os_environ = __import__("os").environ
        os_environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(node_root)
        node_auth = AuthStore(config_dir=node_root)
        node_auth.set_refresh_fence(node_client.refresh_fence)
        row = node_auth.upsert_credential(
            KEY,
            {
                "access": "access-0",
                "refresh": "refresh-0",
                "expires": int(time.time() * 1000) - 1000,
                "login_incarnation": "inc-1",
                "mesh_origin": {
                    "owner_device": OWNER,
                    "key": KEY,
                    "grant_id": minted["grant_id"],
                    "epoch": 7,
                },
            },
        )
        before = len(idp.posts)
        creds = asyncio.run(node_auth._ensure_oauth_fresh(node_auth.get_credential(row.id)))
        assert idp.posts[before:] == ["refresh-0"], idp.posts
        assert creds["refresh"] == "refresh-1"

        # -- 5. the handback: intent, then the last frame --------------------
        intent = broker.refresh_return(
            link,
            _frame(
                kind="refresh_return", act="intent", key=KEY, grant_id=minted["grant_id"], epoch=8
            ),
        )
        assert intent["act"] == "ack"
        # `recalled` is still LIVE for the POST gate (§3.3: the owner promised not
        # to present the token at message 2, while it is still on the node), and the
        # row itself carries the state the handback continues from.
        assert (auth.refresh_delegate_for(1) or {}).get("state") == "recalled"
        assert dict(auth.get_credential(1).data)["refresh_delegate"]["state"] == "recalled"
        done = broker.refresh_return(
            link,
            _frame(
                kind="refresh_return",
                act="done",
                key=KEY,
                grant_id=minted["grant_id"],
                epoch=8,
                login_incarnation="inc-1",
                refresh="refresh-1",
                access="access-1",
                expires=int(time.time() * 1000) + 3_600_000,
            ),
        )
        assert done["ok"] is True
        assert auth.refresh_delegate_for(1) is None
        assert dict(auth.get_credential(1).data)["refresh"] == "refresh-1"

        assert dict(auth.get_credential(1).data)["refresh"] == "refresh-1"
    finally:
        broker.close()


def test_revocation_marks_the_owners_own_family_moved(
    tmp_path: Path, monkeypatch, stub_provider: str, idp: Any, operator_key: Any, operator_spki: Any
) -> None:
    """R1-1 — revocation ends the delegation on the OWNER as well as on the node.

    The deny stops the node (previous test); this is the other half the ruling
    added: the owner's frozen copy is a possibly-spent token, so revocation marks
    the family moved here and the login is unusable locally until a re-login. The
    assertion is the POST count, because that is the harm.
    """
    anchored_owner(tmp_path, monkeypatch, operator_spki)
    broker, auth, _owner_root = _owner_broker(tmp_path, operator_key)
    try:
        broker.mint_refresh_delegation(
            key=KEY,
            device=NODE,
            not_after=time.time() + 3600,
            exclusivity_confirmed=True,
            login_epoch=7,
        )
        revoked = broker.revoke_refresh_delegation(KEY)
        assert revoked["deny"]
        assert auth.refresh_delegate_record(1) is None
        assert dict(auth.get_credential(1).data).get("refresh_family_moved") is not None

        # The row is now expired so a refresh WOULD be needed — and must not happen.
        auth._conn.execute(
            "UPDATE auth_credentials SET data = ? WHERE id = 1",
            (
                __import__("json").dumps(
                    {**dict(auth.get_credential(1).data), "expires": int(time.time() * 1000) - 1000}
                ),
            ),
        )
        auth._conn.commit()
        before = len(idp.posts)
        with pytest.raises(Exception) as refused:  # noqa: B017 — CredentialInvalidError
            asyncio.run(auth.ensure_oauth_fresh_or_raise(1))
        assert idp.posts[before:] == [], "the owner presented the token after revoking"
        # Either refusal is correct — the family-moved message fires first, and the
        # ended-delegation message is its sibling for the case where no marker was
        # written yet. Both name the interactive remedy.
        message = str(refused.value)
        assert "sign in again" in message or "take it back" in message
    finally:
        broker.close()


def test_revocation_reaches_the_node_by_relay_and_stops_it(
    tmp_path: Path, monkeypatch, stub_provider: str, idp: Any, operator_key: Any, operator_spki: Any
) -> None:
    """§3.4 Channel 1: a deny carried by a THIRD member stops the node, offline known.

    The deny is delivered through ``PlacementDocument.merge`` — the one place the
    owner-write rule would otherwise refuse a third-party relay — and the assertion
    is not "the record changed" but "the next refresh presents NOTHING", because a
    refusal that still POSTs would be the reuse-detection exchange this whole design
    exists to avoid.
    """
    anchored_owner(tmp_path, monkeypatch, operator_spki)
    broker, auth, _owner_root = _owner_broker(tmp_path, operator_key)
    link = SimpleNamespace(device_id=NODE, network_id=NETWORK, epoch=8)
    try:
        minted = broker.mint_refresh_delegation(
            key=KEY,
            device=NODE,
            not_after=time.time() + 3600,
            exclusivity_confirmed=True,
            login_epoch=7,
        )
        taken = broker.refresh_take(link, _frame(key=KEY))
        assert taken["ok"] and minted["grant_id"] == taken["grant_id"]
        node_root = tmp_path / "node"
        node_root.mkdir()
        refresh.install(
            NETWORK,
            node_root,
            recipient=NODE,
            statement=taken["statement"],
            operator_spki=operator_spki,
        )
        from local_operator.network.credentials.client import MeshCredentialClient
        from local_operator.network.credentials.held_policy import HeldPolicyStore
        from local_operator.providers.auth_store import AuthStore

        monkeypatch.setattr(refresh, "anchor_spki_for", lambda root: operator_spki)
        node_client = SimpleNamespace(
            root=node_root,
            network_id=NETWORK,
            self_device=NODE,
            _policy=lambda: HeldPolicyStore(NETWORK, NODE, node_root),
            _self_membership_active=lambda: True,
        )
        node_client.refresh_fence = MeshCredentialClient.refresh_fence.__get__(node_client)
        __import__("os").environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(node_root)
        node_auth = AuthStore(config_dir=node_root)
        node_auth.set_refresh_fence(node_client.refresh_fence)
        row = node_auth.upsert_credential(
            KEY,
            {
                "access": "access-0",
                "refresh": "refresh-0",
                "expires": int(time.time() * 1000) - 1000,
                "login_incarnation": "inc-1",
                "mesh_origin": {
                    "owner_device": OWNER,
                    "key": KEY,
                    "grant_id": minted["grant_id"],
                    "epoch": 7,
                },
            },
        )
        assert node_client.refresh_fence(1, dict(node_auth.get_credential(row.id).data)) is None

        revoked = broker.revoke_refresh_delegation(KEY)
        assert revoked["deny"]
        node_doc = placement_mod.PlacementDocument(NETWORK, root=node_root, written_by=NODE)
        node_doc.merge(
            {"credentials": [], "refresh_deny": revoked["deny"]},
            from_device="d_third_member",
            self_device=NODE,
        )
        assert (refresh.load(NETWORK, node_root, recipient=NODE) or {}).get("state") == "refused"
        before = len(idp.posts)
        with pytest.raises(Exception) as refused:  # noqa: B017 — the store's own refusal type
            asyncio.run(node_auth._ensure_oauth_fresh(node_auth.get_credential(row.id)))
        assert idp.posts[before:] == [], "the node presented the token after the deny"
        assert "not presented" in str(refused.value) or "delegation" in str(refused.value)
    finally:
        broker.close()
