"""The refresh delegation's four fences, each with a control that fails without it.

WHAT THESE TESTS ARE FOR. `docs/design/mesh-refresh-delegation.md` §3.2-§3.5 name
four properties that are the feature rather than extras around it, and §5 asks for
tests that fail when a fence is removed:

* **exclusivity by construction** — the owner never presents the delegated family;
* **the epoch handoff and the last-frame rule** — a lower epoch never overwrites, and
  a handback is adopted only from the same login lineage;
* **deny wins, including offline** — a revocation delivered by a third member stops
  the node, and it stays stopped across a reconnection;
* **"it cannot tell" resolves to stop** — unprovable evidence refuses the POST.

The negative controls are the point: a node with NO delegation must not refresh
(T1), a forged or expired statement must not verify (T2), and a delegation minted
without the operator's exclusivity confirmation must be refused (§3.5). Each test
that asserts a refusal also asserts that the stub identity provider recorded ZERO
POSTs, because "refused" without "nothing was presented" would pass against a
store that had stopped returning a token for an unrelated reason.

NOT HERE, AND SAID SO RATHER THAN IMPLIED: T7/T8's full three-message handback over
a live two-device relay and T12's two-process race need the paired-relay rig
(`test_credentials_real_link.py`) and are routed to QA; what is tested here is the
adoption arithmetic and the compare-and-set those paths rest on. T13 (confinement)
is a one-off measurement on the PR, not a unit test.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import refresh, types
from local_operator.operator import verify as operator_verify
from tests.unit.network import test_credentials_owner as owner_rig

STUB_PROVIDER = owner_rig.STUB_PROVIDER
# pytest resolves a fixture by NAME in a test signature; binding the names here is
# how this module registers them, so a module-level assignment (the same spelling
# ``test_credentials_owner.py`` itself uses) is deliberate rather than an import.
idp = owner_rig.idp
stub_provider = owner_rig.stub_provider

KEY = STUB_PROVIDER
OWNER = "d_owner0000000001"
NODE = "d_node00000000002"
NETWORK = "n_refresh"


# ---------------------------------------------------------------------------
# A fixture operator key: signing here never prompts, and never touches the host
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def operator_key() -> Any:
    from cryptography.hazmat.primitives.asymmetric import ec

    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture(scope="module")
def operator_spki(operator_key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return operator_key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )


def _sign(operator_key: Any, grant: operator_verify.RefreshGrant) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    return grant.encode(signature=operator_key.sign(grant.payload(), ec.ECDSA(hashes.SHA256())))


def _grant(
    operator_key: Any,
    *,
    node: str = NODE,
    key: str = KEY,
    owner: str = OWNER,
    window_s: float = 3600.0,
    epoch: int = 7,
    incarnation: str = "inc-1",
    relay_deny: bool = False,
    not_after: int | None = None,
) -> tuple[operator_verify.RefreshGrant, str]:
    now = int(time.time())
    grant = operator_verify.RefreshGrant(
        network_id=NETWORK,
        owner_device=owner,
        owner_spki_fp=operator_verify.key_id_for(b""),
        node_device=node,
        node_spki_fp="node-fp",
        key=key,
        provider=key,
        credential_kind="oauth",
        login_incarnation=incarnation,
        family_epoch_at_issue=epoch,
        grant_id=f"rd_{epoch:04d}",
        issued_at=now - 60,
        not_after=int(now + window_s) if not_after is None else not_after,
        max_refreshes=20,
        exclusive_holder=True,
        relay_deny=relay_deny,
    )
    return grant, _sign(operator_key, grant)


# ---------------------------------------------------------------------------
# The statement itself (§3.7g)
# ---------------------------------------------------------------------------


def test_signed_statement_verifies_and_a_tampered_body_does_not(
    operator_key, operator_spki
) -> None:
    grant, statement = _grant(operator_key)
    assert (
        operator_verify.verify_refresh_grant(
            statement, operator_spki=operator_spki, now=time.time(), node_device=NODE, key=KEY
        )
        == grant
    )

    # TAMPER: flip one field in the envelope. The signature covers bytes RE-DERIVED
    # from the parsed body, so an edited body cannot be made to verify by editing it.
    body_b64, _, sig_b64 = statement.partition(".")
    raw = operator_verify._unb64(body_b64)
    assert raw is not None
    tampered = raw.replace(b'"max_refreshes":20', b'"max_refreshes":21')
    assert tampered != raw
    forged = f"{operator_verify._b64(tampered)}.{sig_b64}"
    assert (
        operator_verify.verify_refresh_grant(
            forged, operator_spki=operator_spki, now=time.time(), node_device=NODE, key=KEY
        )
        is None
    )


def test_expired_grant_refused_but_deny_past_its_window_verifies(
    operator_key, operator_spki
) -> None:
    now = int(time.time())
    _, expired = _grant(operator_key, not_after=now - 1)
    assert (
        operator_verify.verify_refresh_grant(
            expired, operator_spki=operator_spki, now=time.time(), node_device=NODE
        )
        is None
    )

    # A DENY is minted with `not_after` AT the revoke moment, so its own expiry is in
    # the past by construction; refusing it on that basis would make revocation
    # undeliverable (§3.4 Channel 1).
    _, deny = _grant(operator_key, relay_deny=True, not_after=now - 1)
    held = operator_verify.verify_refresh_grant(
        deny, operator_spki=operator_spki, now=time.time(), node_device=NODE
    )
    assert held is not None and held.relay_deny


def test_statement_for_another_device_or_key_does_not_verify(operator_key, operator_spki) -> None:
    _, statement = _grant(operator_key)
    for kwargs in (
        {"node_device": "d_someone_else"},
        {"key": "anthropic"},
    ):
        assert (
            operator_verify.verify_refresh_grant(
                statement, operator_spki=operator_spki, now=time.time(), **kwargs
            )
            is None
        )
    # No anchor at all is a refusal, never a pass (§6 item 3, encoded as a stop).
    assert (
        operator_verify.verify_refresh_grant(statement, operator_spki=None, now=time.time()) is None
    )


# ---------------------------------------------------------------------------
# The node's document: install, active, stop (§3.4)
# ---------------------------------------------------------------------------


def test_install_refuses_an_unverified_statement(
    tmp_path: Path, operator_key, operator_spki
) -> None:
    _, statement = _grant(operator_key)
    other = operator_verify.RefreshGrant(
        **{**_grant(operator_key)[0].fields(), "node_device": NODE}
    )
    # Sign with a DIFFERENT key than the one we verify against: the negative control
    # for "the signature is the authority".
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    stranger = ec.generate_private_key(ec.SECP256R1())
    forged = other.encode(signature=stranger.sign(other.payload(), ec.ECDSA(hashes.SHA256())))
    with pytest.raises(refresh.RefreshRefused):
        refresh.install(
            NETWORK, tmp_path, recipient=NODE, statement=forged, operator_spki=operator_spki
        )
    assert refresh.load(NETWORK, tmp_path, recipient=NODE) is None

    installed = refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    assert installed["state"] == "active"
    assert (
        refresh.active(NETWORK, tmp_path, recipient=NODE, key=KEY, operator_spki=operator_spki)
        is not None
    )


def test_no_delegation_means_no_active_grant(tmp_path: Path, operator_spki) -> None:
    """T1's negative control: the file is absent, so there is nothing to refresh with."""
    assert refresh.load(NETWORK, tmp_path, recipient=NODE) is None
    assert (
        refresh.active(NETWORK, tmp_path, recipient=NODE, key=KEY, operator_spki=operator_spki)
        is None
    )


def test_deny_stops_the_node_and_survives_a_reload(
    tmp_path: Path, operator_key, operator_spki
) -> None:
    """T5/T6: a deny delivered by anyone stops this device, and stays stopped."""
    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )

    # The deny names THIS device and the same grant: that is the whole test of
    # whether a relayed revocation is about us.
    _, deny = _grant(operator_key, relay_deny=True, not_after=int(time.time()))
    assert refresh.note_deny(
        deny, network_id=NETWORK, root=tmp_path, recipient=NODE, operator_spki=operator_spki
    )
    document = refresh.load(NETWORK, tmp_path, recipient=NODE)
    assert document is not None and document["state"] == "refused"
    # Survives a reload from disk, which is what "offline" means: no owner contact.
    assert (
        refresh.active(NETWORK, tmp_path, recipient=NODE, key=KEY, operator_spki=operator_spki)
        is None
    )


def test_deny_for_another_device_is_ignored(tmp_path: Path, operator_key, operator_spki) -> None:
    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    _, elsewhere = _grant(
        operator_key, node="d_other_node", relay_deny=True, not_after=int(time.time())
    )
    assert not refresh.note_deny(
        elsewhere, network_id=NETWORK, root=tmp_path, recipient=NODE, operator_spki=operator_spki
    )
    assert (
        refresh.active(NETWORK, tmp_path, recipient=NODE, key=KEY, operator_spki=operator_spki)
        is not None
    )


def test_unprovable_document_raises_rather_than_reading_as_no(
    tmp_path: Path, operator_key, operator_spki
) -> None:
    """§3.4's fail-closed: a file we cannot prove is NOT "no delegation"."""
    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    path = refresh.refresh_path(NETWORK, tmp_path)
    path.chmod(0o644)  # group/world readable: cannot be proven ours
    with pytest.raises(refresh.RefreshUnknown):
        refresh.active(NETWORK, tmp_path, recipient=NODE, key=KEY, operator_spki=operator_spki)


def test_epoch_handback_rules() -> None:
    """§3.3 Rule B: a lower epoch never overwrites, and a new lineage never wins."""
    assert refresh.handback_holds(9, epoch_at_issue=7, incarnation="inc", login_incarnation="inc")
    assert not refresh.handback_holds(
        7, epoch_at_issue=7, incarnation="inc", login_incarnation="inc"
    )
    assert not refresh.handback_holds(
        9, epoch_at_issue=7, incarnation="inc2", login_incarnation="inc"
    )


# ---------------------------------------------------------------------------
# Fence 1: the owner never presents a delegated family (§3.2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def owner_auth(tmp_path: Path, monkeypatch, stub_provider: str) -> Any:
    """An owner store with ONE already-expired login, so an ask MUST refresh to answer.

    Built here rather than borrowed from ``test_credentials_owner``'s ``owner``
    fixture because that fixture yields the module's ``Owner`` harness, not the
    store: reaching into its internals would couple this file to another suite's
    scaffolding, and every assertion below is about the store's own seam.
    """
    root = tmp_path / "owner"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            "access": "access-0",
            "expires": int(time.time() * 1000) - 60_000,
            "refresh": "refresh-0",
            "email": "owner@example.test",
        },
    )
    return auth


def _expire_row(auth: Any, row_id: int) -> None:
    """Make the row's access token stale so the next ask MUST refresh to answer."""
    row = auth.get_credential(row_id)
    data = dict(row.data)
    data["expires"] = int(time.time() * 1000) - 60_000
    auth._conn.execute(
        "UPDATE auth_credentials SET data = ? WHERE id = ?",
        (__import__("json").dumps(data), row_id),
    )
    auth._conn.commit()


def _refresh_once(auth: Any, row_id: int) -> dict[str, Any]:
    return asyncio.run(auth.ensure_oauth_fresh_or_raise(row_id))


def test_owner_gate_refuses_the_post_and_serves_a_live_bearer(owner_auth: Any, idp: Any) -> None:
    """Fence 1, both halves: no POST while delegated, and a working bearer is still served."""
    auth = owner_auth
    row = auth.list_credentials(STUB_PROVIDER)[0]
    _expire_row(auth, row.id)
    auth.set_refresh_delegate(
        row.id,
        {
            "device": NODE,
            "grant_id": "rd_0007",
            "epoch": 7,
            "issued_at": time.time(),
            "not_after": time.time() + 3600,
            "state": "active",
        },
    )
    from local_operator.providers.auth_store import RefreshDelegatedError

    with pytest.raises(RefreshDelegatedError):
        _refresh_once(auth, row.id)
    assert idp.posts == [], "the owner presented the delegated refresh token"

    # The serving half: a token still inside its lifetime is returned byte-for-byte.
    current = auth.get_credential(row.id)
    live = dict(current.data)
    live["access"] = "access-live"
    live["expires"] = int(time.time() * 1000) + 600_000
    auth._conn.execute(
        "UPDATE auth_credentials SET data = ? WHERE id = ?",
        (__import__("json").dumps(live), row.id),
    )
    auth._conn.commit()
    served = _refresh_once(auth, row.id)
    assert served["access"] == "access-live"
    assert idp.posts == []


def test_expired_delegation_record_no_longer_blocks(owner_auth: Any, idp: Any) -> None:
    """The pure fold: `not_after` passing ends the block, with no writer involved."""
    auth = owner_auth
    row = auth.list_credentials(STUB_PROVIDER)[0]
    _expire_row(auth, row.id)
    auth.set_refresh_delegate(
        row.id,
        {
            "device": NODE,
            "grant_id": "rd_0007",
            "epoch": 7,
            "issued_at": time.time() - 7200,
            "not_after": time.time() - 1,
            "state": "active",
        },
    )
    served = _refresh_once(auth, row.id)
    # The fold re-opened the POST path: exactly one exchange, and the row moved on.
    assert len(idp.posts) == 1
    assert served["refresh"] != "refresh-0"


def test_family_moved_marker_stops_refresh_for_good(owner_auth: Any, idp: Any) -> None:
    """§3.5: the node reported the family moved, so nothing re-presents it."""
    auth = owner_auth
    row = auth.list_credentials(STUB_PROVIDER)[0]
    _expire_row(auth, row.id)
    assert auth.mark_family_moved(row.id)
    from local_operator.providers.auth_store import CredentialInvalidError

    with pytest.raises(CredentialInvalidError):
        _refresh_once(auth, row.id)
    assert idp.posts == []
    # And the delegation record it carried is gone: the family it authorised is not here.
    assert auth.refresh_delegate_for(row.id) is None


def test_handback_adoption_is_compare_and_set(owner_auth: Any) -> None:
    """A handback for a superseded lineage is refused rather than overwriting a login."""
    auth = owner_auth
    row = auth.list_credentials(STUB_PROVIDER)[0]
    current = dict(auth.get_credential(row.id).data)
    current["login_incarnation"] = "inc-now"
    auth._conn.execute(
        "UPDATE auth_credentials SET data = ? WHERE id = ?",
        (__import__("json").dumps(current), row.id),
    )
    auth._conn.commit()
    assert not auth.adopt_refresh_handback(
        row.id, refresh="refresh-9", access="access-9", expires=1, incarnation="inc-old"
    )
    assert dict(auth.get_credential(row.id).data)["refresh"] == "refresh-0"
    assert auth.adopt_refresh_handback(
        row.id, refresh="refresh-9", access="access-9", expires=1, incarnation="inc-now"
    )
    adopted = dict(auth.get_credential(row.id).data)
    assert adopted["refresh"] == "refresh-9" and adopted["access"] == "access-9"


# ---------------------------------------------------------------------------
# Fence 3/4: the node's pre-POST fence, and the deny watermark
# ---------------------------------------------------------------------------


def _node_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A MeshCredentialClient rooted at ``tmp_path`` with a real held-policy store."""
    from types import SimpleNamespace

    from local_operator.network.credentials import client as client_mod

    mesh = SimpleNamespace(
        root=tmp_path,
        network_id=NETWORK,
        self_device=NODE,
        _policy=lambda: HeldPolicy(NETWORK, NODE, tmp_path),
        _self_membership_active=lambda: True,
    )
    mesh.refresh_fence = client_mod.MeshCredentialClient.refresh_fence.__get__(mesh)
    monkeypatch.setattr(refresh, "anchor_spki_for", lambda root: _SPKI.get("value"))
    return mesh


def HeldPolicy(network_id: str, recipient: str, root: Path) -> Any:
    from local_operator.network.credentials.held_policy import HeldPolicyStore

    return HeldPolicyStore(network_id, recipient, root)


_SPKI: dict[str, Any] = {}


def test_fence_allows_a_live_delegation_and_refuses_without_one(
    tmp_path: Path, monkeypatch, operator_key, operator_spki
) -> None:
    _SPKI["value"] = operator_spki
    mesh = _node_client(tmp_path, monkeypatch)
    delegated_row = {
        "mesh_origin": {"owner_device": OWNER, "key": KEY, "grant_id": "rd_0007", "epoch": 7}
    }
    # T1's control: the row looks delegated but no statement exists.
    assert mesh.refresh_fence(1, delegated_row) is not None
    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    assert mesh.refresh_fence(1, delegated_row) is None
    # An ordinary local row (no provenance) is never fenced.
    assert mesh.refresh_fence(1, {"access": "a", "refresh": "r"}) is None


def test_fence_refuses_after_a_revocation(
    tmp_path: Path, monkeypatch, operator_key, operator_spki
) -> None:
    """T5/T6: deny wins offline, and it stays won with no owner in sight."""
    _SPKI["value"] = operator_spki
    mesh = _node_client(tmp_path, monkeypatch)
    delegated_row = {
        "mesh_origin": {"owner_device": OWNER, "key": KEY, "grant_id": "rd_0007", "epoch": 7}
    }
    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    assert mesh.refresh_fence(1, delegated_row) is None

    _, deny = _grant(operator_key, relay_deny=True, not_after=int(time.time()))
    assert refresh.note_deny(
        deny, network_id=NETWORK, root=tmp_path, recipient=NODE, operator_spki=operator_spki
    )
    refusal = mesh.refresh_fence(1, delegated_row)
    assert refusal is not None and "not presented" in refusal


def test_fence_refuses_on_the_per_key_deny_watermark(
    tmp_path: Path, monkeypatch, operator_key, operator_spki
) -> None:
    """§3.4 step 3: a watermark that is not in NONRESTRICTIVE refuses the POST."""
    from local_operator.network.credentials.held_policy import HeldPolicyStore

    _SPKI["value"] = operator_spki
    mesh = _node_client(tmp_path, monkeypatch)
    delegated_row = {
        "mesh_origin": {"owner_device": OWNER, "key": KEY, "grant_id": "rd_0007", "epoch": 7}
    }
    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    assert mesh.refresh_fence(1, delegated_row) is None

    policy = HeldPolicyStore(NETWORK, NODE, tmp_path)
    with policy.transaction(initialize=True) as document:
        policy.deny(document, OWNER, KEY, "revoked")
    refusal = mesh.refresh_fence(1, delegated_row)
    assert refusal is not None and "revoked" in refusal


def test_owner_policy_binding_uses_a_live_delegation(
    tmp_path: Path, monkeypatch, operator_key, operator_spki
) -> None:
    """§3.9 store.py: a live delegation IS the local answer for that owner binding."""
    from types import SimpleNamespace

    from local_operator.network.credentials.store import MeshAwareAuthStore

    _SPKI["value"] = operator_spki
    store = MeshAwareAuthStore(None, mesh=None, config_dir=tmp_path)
    mesh = SimpleNamespace(network_id=NETWORK, self_device=NODE, refresh_fence=lambda *a: None)
    monkeypatch.setattr(type(store), "_mesh", property(lambda self: mesh))
    binding = SimpleNamespace(policy="owner", owner_device=OWNER)
    assert store._owner_policy_skips_local(binding) is True  # today: broker to the owner

    _, statement = _grant(operator_key)
    refresh.install(
        NETWORK, tmp_path, recipient=NODE, statement=statement, operator_spki=operator_spki
    )
    monkeypatch.setattr(refresh, "anchor_spki_for", lambda root: _SPKI["value"])
    assert store._owner_policy_skips_local(binding) is False  # delegated: use the local row


# ---------------------------------------------------------------------------
# The mint's two refusals (§3.5's mint gate, §3.2's single record)
# ---------------------------------------------------------------------------


def test_mint_requires_exclusivity_confirmation(tmp_path: Path) -> None:
    from local_operator.network.credentials.owner import MeshCredentialBroker
    from local_operator.network.types import MeshRefusal

    broker = MeshCredentialBroker(
        root=tmp_path, self_device=OWNER, self_device_name="owner", network_id=NETWORK
    )
    try:
        with pytest.raises(MeshRefusal) as refused:
            broker.mint_refresh_delegation(
                key=KEY, device=NODE, not_after=time.time() + 3600, exclusivity_confirmed=False
            )
        assert refused.value.code == "interactive_required"
    finally:
        broker.close()


# ---------------------------------------------------------------------------
# Nothing material in the log (§3.7f, T11)
# ---------------------------------------------------------------------------


def test_audit_allowlists_drop_material() -> None:
    from local_operator.network.audit import _sanitize_detail

    detail = _sanitize_detail(
        "credential.refresh_taken",
        {
            "credential_key": KEY,
            "act": NODE,
            "sub": OWNER,
            "grant_id": "rd_0007",
            "epoch": 7,
            "refresh_token": "refresh-0",
            "access_token": "access-0",
            "token": "refresh-0",
        },
    )
    assert set(detail) == {"credential_key", "act", "sub", "grant_id", "epoch"}
    assert "refresh-0" not in str(detail) and "access-0" not in str(detail)


def test_new_broker_kinds_are_declared_once() -> None:
    assert "refresh_take" in types.BROKER_KINDS and "refresh_return" in types.BROKER_KINDS
    assert types.REFRESH_CAP_STRING == "credential-refresh-v1"
    # §3.7a: no new peer op — the two kinds ride the existing one.
    assert types.PEER_BROKER_OP == "net_broker"
